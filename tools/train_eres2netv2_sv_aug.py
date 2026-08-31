# -*- coding: utf-8 -*-
"""
ERes2NetV2 声纹训练脚本（在线增强版：干扰人声 + 背景噪音，损失监控）

基于 V17.1/tools/train_eres2netv2_finetune.py 改造，保留：
  - 双向 margin 对比损失（防 embedding 塌缩）
  - 冻结主干前半 + 全部 BN 设 eval
  - 1:1 平衡采样、验证集 EER / pos_sim / neg_sim / emb_std 监控
新增：
  - 在线增强：每个 batch 实时把"背景噪音 + 干扰人声"按随机 SNR 混入波形
    （噪声源用 D:/mobvoi_dataset/mobvoi_subset/wav：真实中文带噪语音，
     最贴近 datasetA 的落地带噪分布 → "与 datasetA 越相近越好"）
  - 损失监控：逐 batch 打印 + 逐 epoch 汇总 + 训练结束后绘制 loss 曲线 PNG
  - 支持直接吃现有 folds（cnceleb_folds / himia_folds / folds）的 jsonl

增强设计（与 datasetA 风格对齐）：
  - 背景噪音 (background)：随机 mobvoi 片段，SNR ∈ [bg_snr_min, bg_snr_max]
  - 干扰人声 (interference speech)：再混入一段 mobvoi 语音（重叠说话人），
    SNR ∈ [sp_snr_min, sp_snr_max]（通常比背景噪音高，模拟"有人同时说话"）
  - 两者独立以概率触发；kws/cmd 两端各自独立加噪 → 学噪声不变的说话人表征

用法：
  # 先用 HI-MIA 唤醒词域 folds（最贴近 datasetA 的 kws+cmd 场景）做带噪训练
  python tools/train_eres2netv2_sv_aug.py \
      --data_root "D:/声纹训练包_5.5h_v2/声纹训练包_5.5h_v2/声纹训练包_5.5h" \
      --train_jsonl himia_folds/folds/fold_full/train.jsonl \
      --val_jsonl   himia_folds/folds/fold_full/val.jsonl \
      --noise_dir "D:/mobvoi_dataset/mobvoi_subset/wav" \
      --epochs 12 --lr 1e-4 --batch 64 --workers 4 --out_dir runs/sv_aug_himia

  # 用 CN-Celeb 预训练（通用说话人表征）
  python tools/train_eres2netv2_sv_aug.py \
      --data_root "..." --train_jsonl cnceleb_folds/folds/fold_full/train.jsonl \
      --val_jsonl cnceleb_folds/folds/fold_full/val.jsonl \
      --noise_dir "D:/mobvoi_dataset/mobvoi_subset/wav" \
      --epochs 20 --lr 1e-3 --batch 64 --out_dir runs/sv_aug_cnceleb

  # 关闭增强（对照实验）
  --no_aug
"""
import os
import sys
import json
import time
import argparse
import random
import glob
import numpy as np

# === cuDNN DLL 修复：注入 Library/bin 到 PATH ===
_lib_bin = os.path.abspath(os.path.join(os.path.dirname(sys.executable), os.pardir, 'Library', 'bin'))
if os.path.isdir(_lib_bin):
    os.environ['PATH'] = _lib_bin + os.pathsep + os.environ.get('PATH', '')

# 消除显存碎片型 OOM：允许分配器按需扩展段, 而非死守预留块(之前 PowerShell $env 未传给 pythonw 子进程故失效)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch

# cuDNN 加载失败兜底：禁用后用 PyTorch 原生卷积（慢但可用）
torch.backends.cudnn.enabled = False

import torch.nn as nn
import torch.nn.functional as F
import torchaudio.compliance.kaldi as Kaldi
import soundfile as sf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.environ.setdefault('MODELSCOPE_CACHE',
                      os.path.join(PROJECT_ROOT, 'pretrained', 'modelscope_cache'))

FREEZE_MODULES = ["conv1", "bn1", "layer1", "layer2"]
NUM_FRAMES = 149  # 约 1.5s，fbank 10ms 帧移
SR = 16000


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)))


# ==================== 双向 margin 损失（防塌缩） ====================
def margin_loss(sim_pos, sim_neg, pos_margin=0.7, neg_margin=0.3):
    loss_pos = F.relu(pos_margin - sim_pos).mean()
    loss_neg = F.relu(sim_neg - neg_margin).mean()
    return loss_pos + loss_neg


# ==================== 模型加载 ====================
def load_model(device):
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    sv_pipeline = pipeline(
        task=Tasks.speaker_verification,
        model="iic/speech_eres2netv2_sv_zh-cn_16k-common",
    )
    wrapper = sv_pipeline.model
    emb_model = wrapper.embedding_model
    emb_model.to(device)
    return wrapper, emb_model


def setup_trainable(emb_model):
    for top in FREEZE_MODULES:
        obj = emb_model
        ok = True
        for part in top.split('.'):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                ok = False
                break
        if ok:
            for p in obj.parameters():
                p.requires_grad = False
    frozen = sum(p.numel() for p in emb_model.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in emb_model.parameters() if p.requires_grad)
    emb_model.train()
    bn_count = 0
    for m in emb_model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()
            bn_count += 1
    print(f"  冻结参数: {frozen/1e6:.2f}M, 可训练参数: {trainable/1e6:.2f}M, BN设eval: {bn_count}层")
    return [p for p in emb_model.parameters() if p.requires_grad]


# ==================== fbank / 裁剪 ====================
def wav_to_fbank(wav: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
    feat = Kaldi.fbank(t, num_mel_bins=80)  # [T, 80]
    return feat.numpy().astype(np.float32)


def crop_or_pad(feat, num_frames, rng):
    T = feat.shape[0]
    if T >= num_frames:
        start = int(rng.integers(0, T - num_frames + 1))
        return feat[start:start + num_frames]
    reps = int(np.ceil(num_frames / T))
    return np.tile(feat, (reps, 1))[:num_frames]


# ==================== 数据 ====================
class PairData:
    def __init__(self, jsonl_path):
        self.records = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                label = 1 if r["识别文本"] is not None else 0
                self.records.append((r["唤醒音频"], r["识别音频"], label))

    def __len__(self):
        return len(self.records)


class RawStore:
    """全部干净波形的内存缓存（一次性加载，训练时零 IO）。
    带内存上限保护：超出则回退到按需从磁盘读取（仍可用，仅稍慢）。"""

    def __init__(self, rel_paths, data_root, max_samples_memory=20000, workers=4):
        self.data_root = data_root
        self.cache = {}
        self.use_ram = True
        unique = sorted({p for p in rel_paths})
        print(f"  RawStore: {len(unique)} 个唯一音频, 上限 {max_samples_memory}")
        # 先探测总长度估内存
        est = 0
        for p in unique[:200]:
            try:
                info = sf.info(os.path.join(data_root, p))
                est += int(min(info.frames, SR * 30))
            except Exception:
                pass
        est_bytes = (est / max(len(unique[:200]), 1)) * len(unique) * 4
        if est_bytes > max_samples_memory * SR * 4:
            self.use_ram = False
            print(f"  预估内存 {est_bytes/1e9:.1f}GB > 上限，回退按需磁盘读取模式")
            return
        t0 = time.time()
        for p in unique:
            try:
                y, _ = sf.read(os.path.join(data_root, p), dtype='float32', always_2d=False)
                if y.ndim > 1:
                    y = y[:, 0]
                if len(y) > SR * 30:
                    y = y[:SR * 30]
                self.cache[p] = y.astype(np.float32)
            except Exception as e:
                self.cache[p] = np.zeros(SR, dtype=np.float32)
        print(f"  RawStore 加载完成: {len(self.cache)} 条, 耗时 {time.time()-t0:.0f}s")

    def get(self, rel):
        if self.use_ram:
            return self.cache[rel]
        y, _ = sf.read(os.path.join(self.data_root, rel), dtype='float32', always_2d=False)
        if y.ndim > 1:
            y = y[:, 0]
        if len(y) > SR * 30:
            y = y[:SR * 30]
        return y.astype(np.float32)


def build_noise_pool(noise_dir, max_n=1200, seed=0):
    """加载噪声/干扰源池（mobvoi 真实中文带噪语音）。返回 list[np.ndarray](16k)。"""
    rng = random.Random(seed)
    files = []
    for ext in ("*.wav", "*.flac"):
        files += glob.glob(os.path.join(noise_dir, "**", ext), recursive=True)
    if max_n and len(files) > max_n:
        files = rng.sample(files, max_n)
    pool = []
    for f in files:
        try:
            y, sr = sf.read(f, dtype='float32', always_2d=False)
            if y.ndim > 1:
                y = y[:, 0]
            if sr != SR:
                # 简单重采样（用 torch 最近邻，足够做噪声）
                import torchaudio.functional as AF
                y = AF.resample(torch.from_numpy(y), sr, SR).numpy()
            if len(y) > SR * 20:
                y = y[:SR * 20]
            if len(y) < SR * 0.3:
                continue
            pool.append(y.astype(np.float32))
        except Exception:
            pass
    print(f"  噪声池: {len(pool)} 条 (来自 {noise_dir})")
    return pool


def add_noise(clean: np.ndarray, noise: np.ndarray, snr_db: float, rng) -> np.ndarray:
    """按 snr_db 把 noise 混入 clean（波形域加性混合）。"""
    cl = len(clean)
    if len(noise) >= cl:
        s = int(rng.integers(0, len(noise) - cl + 1))
        ns = noise[s:s + cl]
    else:
        reps = cl // len(noise) + 1
        ns = np.tile(noise, reps)[:cl]
    c_rms = rms(clean) + 1e-8
    n_rms = rms(ns) + 1e-8
    gain = (c_rms / n_rms) * (10 ** (-snr_db / 20.0))
    out = clean + ns * gain
    pk = float(np.max(np.abs(out))) if out.size else 0.0
    if pk > 0.99:
        out = out * (0.99 / pk)
    return out.astype(np.float32)


def apply_aug(clean, noise_pool, rng, bg_prob, bg_snr, sp_prob, sp_snr):
    """在线增强：背景噪音 + 干扰人声（二者独立触发）。"""
    out = clean
    if noise_pool and rng.random() < bg_prob:
        ns = noise_pool[int(rng.integers(0, len(noise_pool)))]
        snr = rng.uniform(*bg_snr)
        out = add_noise(out, ns, snr, rng)
    if noise_pool and rng.random() < sp_prob:
        ns = noise_pool[int(rng.integers(0, len(noise_pool)))]
        snr = rng.uniform(*sp_snr)
        out = add_noise(out, ns, snr, rng)
    return out


# ==================== 评估 ====================
def compute_eer(sims, labels):
    best_eer, best_thr = 1.0, 0.0
    pos = [s for s, l in zip(sims, labels) if l == 1]
    neg = [s for s, l in zip(sims, labels) if l == 0]
    for t in np.arange(0.0, 1.0, 0.005):
        frr = sum(1 for s in pos if s < t) / max(len(pos), 1)
        far = sum(1 for s in neg if s >= t) / max(len(neg), 1)
        eer = (frr + far) / 2
        if eer < best_eer:
            best_eer, best_thr = eer, t
    return best_eer, best_thr


@torch.no_grad()
def validate(emb_model, raw_store, val_data, device, rng):
    torch.cuda.empty_cache()  # 释放训练期累积的显存缓存, 避免验证峰值叠加触发碎片 OOM
    emb_model.eval()
    sims, labels = [], []
    for kws_path, cmd_path, label in val_data.records:
        embs = []
        for path in [kws_path, cmd_path]:
            raw = raw_store.get(path)
            feat = torch.from_numpy(wav_to_fbank(raw)).to(device)
            feat = feat - feat.mean(dim=0, keepdim=True)
            e = emb_model(feat.unsqueeze(0)).squeeze(0)
            e = e / (e.norm() + 1e-8)
            embs.append(e)
        sims.append(float(torch.dot(embs[0], embs[1]).item()))
        labels.append(label)
    emb_model.train()
    for m in emb_model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()
    eer, thr = compute_eer(sims, labels)
    pos_mean = float(np.mean([s for s, l in zip(sims, labels) if l == 1]))
    neg_mean = float(np.mean([s for s, l in zip(sims, labels) if l == 0]))
    return eer, thr, pos_mean, neg_mean


# ==================== 训练 ====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="音频根目录(jsonl 内相对路径的基准)")
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--val_jsonl", required=True)
    ap.add_argument("--noise_dir", default=None, help="噪声/干扰源目录(mobvoi 等真实带噪语音)")
    ap.add_argument("--no_aug", action="store_true", help="关闭在线增强(对照)")
    ap.add_argument("--bg_prob", type=float, default=0.7, help="背景噪音触发概率")
    ap.add_argument("--bg_snr_min", type=float, default=-5.0)
    ap.add_argument("--bg_snr_max", type=float, default=10.0)
    ap.add_argument("--sp_prob", type=float, default=0.5, help="干扰人声触发概率")
    ap.add_argument("--sp_snr_min", type=float, default=0.0)
    ap.add_argument("--sp_snr_max", type=float, default=12.0)
    ap.add_argument("--noise_max", type=int, default=1200, help="噪声池最大条数")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--pos_margin", type=float, default=0.7)
    ap.add_argument("--neg_margin", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--log_every", type=int, default=20, help="每 N 个 batch 打印一次 loss")
    ap.add_argument("--max_samples_memory", type=int, default=20000)
    ap.add_argument("--init_from", default=None, help="从已有权重续训(如阶段1预训练权重)")
    ap.add_argument("--resume", action="store_true", help="从 out_dir/sv_aug_last.pt 断点续训(含优化器状态)")
    ap.add_argument("--save_every_steps", type=int, default=200, help="每 N 个 batch 保存一次 sv_aug_last.pt(0=仅 epoch 末)")
    ap.add_argument("--gpu_mem_frac", type=float, default=0, help="限制 PyTorch 显存上限比例(0-1), 0=不限制(直接 run2 原样); 如0.85=≤85%%")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and getattr(args, "gpu_mem_frac", 0) > 0:
        torch.cuda.set_per_process_memory_fraction(min(args.gpu_mem_frac, 1.0))
    out_dir = args.out_dir or os.path.join(PROJECT_ROOT, "runs", "sv_aug")
    os.makedirs(out_dir, exist_ok=True)
    # 日志重定向: pythonw(无控制台)下 stdout 为 None, 重定向到文件避免崩溃且保证日志落盘
    try:
        if sys.stdout is None:
            sys.stdout = open(os.path.join(out_dir, "train_stdout.log"), "a", encoding="utf-8", buffering=1)
        if sys.stderr is None:
            sys.stderr = sys.stdout
    except Exception:
        pass
    aug_on = (not args.no_aug) and bool(args.noise_dir)
    print(f"device={device}, aug={'ON' if aug_on else 'OFF'}, batch={args.batch}, out={out_dir}", flush=True)

    # 数据
    train_data = PairData(os.path.join(args.data_root, args.train_jsonl))
    val_data = PairData(os.path.join(args.data_root, args.val_jsonl))
    n_pos = sum(1 for r in train_data.records if r[2] == 1)
    print(f"train: {len(train_data)} 对 (pos={n_pos}, neg={len(train_data)-n_pos}), val: {len(val_data)} 对", flush=True)

    # 波形缓存
    all_paths = sorted({p for rec in train_data.records + val_data.records for p in rec[:2]})
    raw_store = RawStore(all_paths, args.data_root, max_samples_memory=args.max_samples_memory, workers=args.workers)

    # 噪声池
    noise_pool = []
    if aug_on:
        noise_pool = build_noise_pool(args.noise_dir, max_n=args.noise_max)

    # 模型
    wrapper, emb_model = load_model(device)
    start_epoch = 0
    optimizer_state = None
    best_eer_ckpt = float("inf")
    resume_path = None
    if args.resume:
        cand = os.path.join(out_dir, "sv_aug_last.pt")
        if os.path.isfile(cand):
            resume_path = cand
    if resume_path is None and args.init_from:
        resume_path = args.init_from
    if resume_path:
        print(f"  加载续训权重: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            emb_model.load_state_dict(ckpt["model"])
            optimizer_state = ckpt.get("optimizer")
            start_epoch = int(ckpt.get("epoch", 0))
            best_eer_ckpt = float(ckpt.get("best_eer", float("inf")))
        else:
            emb_model.load_state_dict(ckpt)
    trainable_params = setup_trainable(emb_model)

    # 基线 EER
    eer0, thr0, pm0, nm0 = validate(emb_model, raw_store, val_data, device, np.random.default_rng(0))
    print(f"[基线] val EER={eer0:.4f} @thr={thr0:.3f}, pos_sim={pm0:.3f}, neg_sim={nm0:.3f}", flush=True)

    pos_records = [r for r in train_data.records if r[2] == 1]
    neg_records = [r for r in train_data.records if r[2] == 0]
    half = args.batch // 2
    print(f"平衡采样: pos={len(pos_records)}, neg={len(neg_records)}, 每步各{half}", flush=True)

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs - start_epoch, 1))
    if optimizer_state is not None:
        try:
            optimizer.load_state_dict(optimizer_state)
            print(f"  已恢复优化器状态 (从 epoch {start_epoch} 续训)", flush=True)
        except Exception as e:
            print(f"  [WARN] 优化器状态恢复失败, 重新初始化: {e}", flush=True)

    best_eer = best_eer_ckpt if optimizer_state is not None else float("inf")  # resume 时继承历史最佳
    log = {"args": vars(args), "baseline_eer": eer0, "epochs": []}
    batch_log = []  # (epoch, step, loss)
    rng = np.random.default_rng(20260826 + start_epoch)

    def _save_last(ep):
        torch.save({"model": emb_model.state_dict(), "optimizer": optimizer.state_dict(),
                    "epoch": ep + 1, "best_eer": best_eer},
                   os.path.join(out_dir, "sv_aug_last.pt"))

    def _save_best(ep):
        torch.save({"model": emb_model.state_dict(), "optimizer": optimizer.state_dict(),
                    "epoch": ep + 1, "best_eer": best_eer},
                   os.path.join(out_dir, "sv_aug_best.pt"))

    def get_feat(rel):
        raw = raw_store.get(rel)
        if aug_on:
            raw = apply_aug(raw, noise_pool, rng, args.bg_prob,
                            (args.bg_snr_min, args.bg_snr_max),
                            args.sp_prob, (args.sp_snr_min, args.sp_snr_max))
        return crop_or_pad(wav_to_fbank(raw), NUM_FRAMES, rng)

    def make_pair_batch(pos_idx, neg_idx):
        feats = []
        for i in pos_idx:
            feats.append(get_feat(pos_records[i][0]))
        for i in pos_idx:
            feats.append(get_feat(pos_records[i][1]))
        for i in neg_idx:
            feats.append(get_feat(neg_records[i][0]))
        for i in neg_idx:
            feats.append(get_feat(neg_records[i][1]))
        X = torch.from_numpy(np.stack(feats)).to(device)
        X = X - X.mean(dim=1, keepdim=True)  # CMN
        return X

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        epoch_loss, n_batch = 0.0, 0
        pos_order = rng.permutation(len(pos_records))

        for step, start in enumerate(range(0, len(pos_order), half)):
            pos_idx = pos_order[start:start + half]
            if len(pos_idx) < half:
                break
            neg_idx = rng.integers(0, len(neg_records), half)

            try:
                X = make_pair_batch(pos_idx, neg_idx)
            except Exception as e:
                print(f"  [E{epoch+1} step {step+1}] make_pair_batch 异常, 跳过: {e}", flush=True)
                continue
            if bool((torch.isnan(X) | torch.isinf(X)).any()):
                print(f"  [E{epoch+1} step {step+1}] 跳过坏 batch (含 NaN/Inf)", flush=True)
                continue
            emb = F.normalize(emb_model(X), dim=1)
            B = len(pos_idx)
            ek_pos, ec_pos = emb[:B], emb[B:2 * B]
            ek_neg, ec_neg = emb[2 * B:3 * B], emb[3 * B:]
            sim_pos = (ek_pos * ec_pos).sum(dim=1)
            sim_neg = (ek_neg * ec_neg).sum(dim=1)
            loss = margin_loss(sim_pos, sim_neg, args.pos_margin, args.neg_margin)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 5.0)
            optimizer.step()

            lv = loss.item()
            epoch_loss += lv
            n_batch += 1
            batch_log.append((epoch + 1, step + 1, lv))
            if args.log_every and (step + 1) % args.log_every == 0:
                print(f"  [E{epoch+1} step {step+1}] loss={lv:.4f}", flush=True)
            if args.save_every_steps and (step + 1) % args.save_every_steps == 0:
                _save_last(epoch)

        scheduler.step()

        # 塌缩监控
        with torch.no_grad():
            idx = rng.choice(len(train_data), min(64, len(train_data)), replace=False)
            feats = [get_feat(train_data.records[i][0]) for i in idx]
            X = torch.from_numpy(np.stack(feats)).to(device)
            X = X - X.mean(dim=1, keepdim=True)
            e = F.normalize(emb_model(X), dim=1)
            emb_std = float(e.std(dim=0).mean().item())

        eer, thr, pm, nm = validate(emb_model, raw_store, val_data, device, rng)
        dt = time.time() - t0
        improved = eer < best_eer
        if improved:
            best_eer = eer
            _save_best(epoch)
        # 兜底: 每个 epoch 结束都保存最新权重(含优化器状态), 确保训练结束必有可用 checkpoint
        _save_last(epoch)

        ep_log = {"epoch": epoch + 1, "loss": epoch_loss / max(n_batch, 1),
                  "val_eer": eer, "val_thr": float(thr), "pos_sim": pm,
                  "neg_sim": nm, "emb_std": emb_std, "time_s": round(dt, 1), "best": improved}
        log["epochs"].append(ep_log)
        # 每个 epoch 增量保存，避免中途异常丢失成果
        with open(os.path.join(out_dir, "train_log.json"), "w", encoding="utf-8") as _f:
            json.dump(log, _f, ensure_ascii=False, indent=2)
        print(f"[E{epoch+1}/{args.epochs}] loss={ep_log['loss']:.4f} val_EER={eer:.4f}@thr={thr:.3f} "
              f"pos={pm:.3f} neg={nm:.3f} emb_std={emb_std:.4f} {'★BEST' if improved else ''} ({dt:.0f}s)",
              flush=True)
        if device == "cuda":
            print(f"[mem] allocated={torch.cuda.memory_allocated()/1e6:.0f}MB reserved={torch.cuda.memory_reserved()/1e6:.0f}MB", flush=True)

        if emb_std < 0.01:
            print(f"⚠️ emb_std={emb_std:.4f} < 0.01, 疑似 embedding 塌缩! 停止训练")
            break

    with open(os.path.join(out_dir, "train_log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "batch_loss.json"), "w", encoding="utf-8") as f:
        json.dump(batch_log, f)

    # 损失曲线
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        epochs = [e["epoch"] for e in log["epochs"]]
        losses = [e["loss"] for e in log["epochs"]]
        eers = [e["val_eer"] for e in log["epochs"]]
        fig, ax1 = plt.subplots(figsize=(9, 5))
        ax1.plot(epochs, losses, "b-o", label="train loss")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("loss", color="b")
        ax1.tick_params(axis='y', labelcolor="b")
        ax2 = ax1.twinx()
        ax2.plot(epochs, eers, "r-s", label="val EER")
        ax2.set_ylabel("val EER", color="r")
        ax2.tick_params(axis='y', labelcolor="r")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=120)
        print(f"损失曲线已保存: {out_dir}/loss_curve.png")
    except Exception as e:
        print(f"[WARN] 损失曲线绘制失败(matplotlib 缺失?): {e}")

    print(f"训练完成. 基线EER={eer0:.4f} → 最佳EER={best_eer:.4f}, 权重: {out_dir}/sv_aug_best.pt", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as _e:
        import traceback as _tb
        _msg = "=== TRAIN CRASH ===\n" + "".join(_tb.format_exception(type(_e), _e, _e.__traceback__))
        sys.stderr.write(_msg)
        try:
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_crash.log"), "w", encoding="utf-8") as _cf:
                _cf.write(_msg)
        except Exception:
            pass
        raise
