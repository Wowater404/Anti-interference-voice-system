# -*- coding: utf-8 -*-
"""
DeepFilterNet3 微调训练脚本（比赛场景降噪适配）
=============================================
数据: LibriSpeech clean (48k) + DEMAND 环境噪声 (48k), 在线混合合成带噪
训练: 复用 df 官方 Loss (config.ini 自带 maskloss/spectralloss/sdrloss) + AdamW + cosine
输出: 微调权重 -> pretrained/deepfilternet3_finetuned/checkpoints/model_finetuned.pt

用法:
  python tools/train_df3_finetune.py \
    --data_dir "F:/.../df3_finetune/data" \
    --out_dir "F:/.../voice_pipeline/pretrained/deepfilternet3_finetuned" \
    --epochs 3 --steps_per_epoch 400 --batch 1 --lr 1e-4 --device cuda
"""
import os
import sys
import json
import time
import argparse
import glob

# [重要] 先 import torch 再写 os.environ (Windows 段错误规避)
import torch  # noqa: F401

# cuDNN PATH 注入
_lib_bin = os.path.abspath(os.path.join(os.path.dirname(sys.executable), 'Library', 'bin'))
if os.path.isdir(_lib_bin):
    os.environ['PATH'] = _lib_bin + os.pathsep + os.environ.get('PATH', '')

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ---- df 库导入 (确保 init_df 使用项目内权重) ----
from df.enhance import init_df, df_features
from df.loss import Loss, Istft
from df.model import ModelParams
from df.utils import as_complex, get_device

# 微调默认输出目录 (项目内)
DEFAULT_OUT = os.path.join(PROJECT_ROOT, "pretrained", "deepfilternet3_finetuned")
# 基础权重目录 (预训练)
BASE_MODEL_DIR = os.path.join(PROJECT_ROOT, "pretrained", "deepfilternet3")

SNRS = [-5, 0, 5, 10, 15, 20]  # 训练 SNR 档位 (比赛覆盖 -5~20dB)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="prepare_df3_data.py 输出目录 (含 clean/ noise/ data_meta.json)")
    ap.add_argument("--out_dir", default=DEFAULT_OUT, help="微调权重输出目录")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--steps_per_epoch", type=int, default=400, help="每 epoch 训练步数")
    ap.add_argument("--val_steps", type=int, default=50, help="验证步数 (test 片段)")
    ap.add_argument("--lr", type=float, default=1e-4, help="微调学习率 (官方 5e-4, 微调用小一点)")
    ap.add_argument("--batch", type=int, default=1, help="batch size (DF3 特征管线按 1 设计)")
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_segments(data_dir, split="clean"):
    """加载片段路径列表"""
    seg_dir = os.path.join(data_dir, split)
    files = sorted(glob.glob(os.path.join(seg_dir, "*.npy")))
    return files


def mix_audio(clean, noise, snr_db, rng):
    """按目标 SNR 混合: noisy = clean + noise*scale
    clean/noise: np.float32 [N] 48k
    """
    n = len(clean)
    if len(noise) > n:
        start = rng.integers(0, len(noise) - n + 1)
        noise = noise[start:start + n]
    else:
        # 噪声短: 循环填充
        noise = np.resize(noise, n)
    p_clean = float(np.sqrt(np.mean(clean ** 2)) + 1e-12)
    p_noise = float(np.sqrt(np.mean(noise ** 2)) + 1e-12)
    scale = p_clean / (p_noise * (10 ** (snr_db / 20)))
    noisy = clean + noise * scale
    # 防削波
    peak = np.max(np.abs(noisy))
    if peak > 0.99:
        noisy = noisy / peak * 0.99
    return noisy.astype(np.float32)


def eval_sisdr(clean, est, eps=1e-10):
    """负 SI-SDR 计算 (越小越好)"""
    n = min(len(clean), len(est))  # ISTFT 帧对齐会导致长度差一帧
    clean = clean[:n] - np.mean(clean[:n])
    est = est[:n] - np.mean(est[:n])
    alpha = np.dot(est, clean) / (np.dot(clean, clean) + eps)
    target = alpha * clean
    noise = est - target
    sisdr = 10 * np.log10(np.dot(target, target) / (np.dot(noise, noise) + eps))
    return float(sisdr)


def main():
    args = parse_args()
    set_seed(args.seed)

    # ---- 0. 加载 df config (get_device/Loss 依赖) ----
    from df.config import config as df_config
    df_config.load(os.path.join(BASE_MODEL_DIR, "config.ini"))
    dev = get_device()
    print(f"device: {dev}")

    # ---- 1. 加载 DF3 预训练模型 ----
    print(f"加载 DeepFilterNet3 预训练: {BASE_MODEL_DIR}")
    model, df_state, _ = init_df(model_base_dir=BASE_MODEL_DIR, post_filter=False, log_level="WARNING")
    model = model.to(dev)
    nb_df = getattr(model, "nb_df", getattr(model, "df_bins", ModelParams().nb_df))
    p = ModelParams()
    print(f"nb_df={nb_df}, sr={p.sr}, fft={p.fft_size}, hop={p.hop_size}")

    # ---- 2. 损失 (官方复合损失, 读模型 config.ini) ----
    istft = Istft(p.fft_size, p.hop_size, torch.as_tensor(df_state.fft_window().copy())).to(dev)
    loss_fn = Loss(df_state, istft).to(dev)
    loss_fn.train()

    # ---- 3. 数据 ----
    clean_files = load_segments(args.data_dir, "clean")
    noise_files = load_segments(args.data_dir, "noise")
    with open(os.path.join(args.data_dir, "data_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    print(f"clean 片段: {len(clean_files)}, noise 片段: {len(noise_files)}")
    if len(clean_files) == 0 or len(noise_files) == 0:
        raise RuntimeError("数据为空, 请先运行 prepare_df3_data.py")

    # 验证集 (独立 test-clean 片段, 取最后 200 个)
    n_val = min(200, len(clean_files))
    val_files = clean_files[-n_val:]
    train_files = clean_files[:-n_val] if len(clean_files) > n_val else clean_files

    # ---- 4. 优化器 ----
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)
    total_steps = args.epochs * args.steps_per_epoch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    rng = np.random.default_rng(args.seed)

    def forward_step(clean_path, is_train):
        """单样本: 特征 -> forward -> loss (返回 loss, 增强波形, 干净波形, 带噪波形)"""
        clean_wave = np.load(clean_path)  # float32 [N]
        noise_path = noise_files[rng.integers(0, len(noise_files))]
        noise_wave = np.load(noise_path)
        snr_db = float(SNRS[rng.integers(0, len(SNRS))])
        noisy_wave = mix_audio(clean_wave, noise_wave, snr_db, rng)

        clean_t = torch.from_numpy(clean_wave).float().to(dev).unsqueeze(0)  # [1, N]
        noisy_t = torch.from_numpy(noisy_wave).float().to(dev).unsqueeze(0)  # [1, N]

        with torch.no_grad():
            # 特征提取 (numpy 域, 无梯度; 需 CPU tensor 输入, df_features 内部转 device)
            spec_n, erb_n, specf_n = df_features(noisy_t.cpu(), df_state, nb_df, device=dev)
            spec_c, _, _ = df_features(clean_t.cpu(), df_state, nb_df, device=dev)

        if hasattr(model, "reset_h0"):
            model.reset_h0(batch_size=1, device=dev)

        enh, m, lsnr, other = model(spec_n, erb_n, specf_n)
        snrs = torch.tensor([snr_db], device=dev)
        err = loss_fn(clean=spec_c, noisy=spec_n, enhanced=enh, mask=m, lsnr=lsnr, snrs=snrs)

        if not is_train:
            with torch.no_grad():
                enh_wave = istft(enh).detach().cpu().numpy().squeeze()
            return err, enh_wave, clean_wave, noisy_wave
        return err, None, None, None

    # ---- 5. 训练 ----
    best_val = -999.0
    log = {"args": vars(args), "epochs": []}
    for epoch in range(args.epochs):
        model.train()
        loss_fn.train()
        t0 = time.time()
        epoch_loss = []
        for step in range(args.steps_per_epoch):
            cf = train_files[rng.integers(0, len(train_files))]
            opt.zero_grad()
            err, _, _, _ = forward_step(cf, True)
            if torch.isnan(err) or torch.isinf(err):
                print(f"  [warn] step {step} loss={err.item():.4f}, skip")
                continue
            err.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            epoch_loss.append(err.item())
            if (step + 1) % 100 == 0:
                avg = np.mean(epoch_loss[-100:])
                print(f"  epoch{epoch} step {step+1}/{args.steps_per_epoch} loss={avg:.4f} lr={opt.param_groups[0]['lr']:.2e}", flush=True)

        # 验证 (test-clean 片段, 算 SI-SDR 提升)
        model.eval()
        loss_fn.eval()
        sisdr_noisy, sisdr_enh = [], []
        with torch.no_grad():
            for vf in val_files[:args.val_steps]:
                _, enh_wave, clean_w, noisy_w = forward_step(vf, False)
                sisdr_noisy.append(eval_sisdr(clean_w, noisy_w))
                sisdr_enh.append(eval_sisdr(clean_w, enh_wave))
        imp = float(np.mean(sisdr_enh) - np.mean(sisdr_noisy))
        print(f"epoch{epoch} 完成: train_loss={np.mean(epoch_loss):.4f} | "
              f"val SI-SDR: noisy={np.mean(sisdr_noisy):.2f}dB enh={np.mean(sisdr_enh):.2f}dB 提升={imp:+.2f}dB "
              f"({time.time()-t0:.0f}s)", flush=True)
        log["epochs"].append({
            "epoch": epoch, "train_loss": float(np.mean(epoch_loss)),
            "val_sisdr_noisy": float(np.mean(sisdr_noisy)),
            "val_sisdr_enh": float(np.mean(sisdr_enh)),
            "sisdr_improve": imp, "time_s": round(time.time() - t0, 1),
        })
        if imp > best_val:
            best_val = imp
            os.makedirs(os.path.join(args.out_dir, "checkpoints"), exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.out_dir, "checkpoints", "model_finetuned.pt"))
            print(f"  ★ 最佳模型已保存 (SI-SDR 提升 {imp:+.2f}dB)")

    # ---- 6. 输出 ----
    import shutil
    os.makedirs(args.out_dir, exist_ok=True)
    # 复制 config.ini (推理 init_df 需要)
    shutil.copy(os.path.join(BASE_MODEL_DIR, "config.ini"), os.path.join(args.out_dir, "config.ini"))
    with open(os.path.join(args.out_dir, "train_log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f"\n训练完成! 微调权重 -> {args.out_dir}")
    print(f"  checkpoints/model_finetuned.pt + config.ini")
    print(f"  val SI-SDR 提升: {best_val:+.2f}dB")


if __name__ == "__main__":
    main()
