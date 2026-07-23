# -*- coding: utf-8 -*-
"""
CAM++ 声纹模型微调训练脚本 V2 (batch加速版)

训练目标: 说话人验证对比学习
  - pos 样本对 (kws, cmd): 同一说话人 → label=1
  - neg 样本对 (kws, cmd): 不同说话人 → label=0
  - 损失: CosineEmbeddingLoss(margin=0.2)

V2 性能优化 (V1逐条forward每epoch约45分钟 → V2每epoch约30秒):
  1. fbank 特征预提取缓存 (多进程, 只做一次)
  2. 固定 NUM_FRAMES=149 帧随机裁剪 (kws全长/cmd随机裁, 每epoch位置随机=天然增强)
  3. kws+cmd 合并真 batch (batch_size=64)

防塌缩措施 (吸取 Resemblyzer 微调塌缩教训):
  1. 冻结主干前半 (head/tdnn/block1/transit1), 只训练后半 (block2起)
  2. 全部 BatchNorm 设 eval (用预训练 running stats, 不被小数据带偏)
  3. 小学习率 1e-4 + 验证集 EER 监控, 保存最佳 checkpoint
  4. 每 epoch 打印 embedding 跨样本 std, 检测塌缩

用法:
  python tools/train_camplus_finetune.py \
      --aug_root "F:/龙虾/2026-07-18-13-57-00/datasetA_aug" \
      --fold 0 --epochs 10 --lr 1e-4 --batch 64 --workers 8
"""
import os
import sys
import json
import time
import argparse
import numpy as np

# === PyTorch 2.5 兼容性修复 (同 voiceprint.py) ===
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.fsdp as _fsdp
if not hasattr(_fsdp, 'CPUOffloadPolicy'):
    class _CPUOffloadPolicy:
        def __init__(self, *a, **k): pass
    _fsdp.CPUOffloadPolicy = _CPUOffloadPolicy
if not hasattr(_fsdp, 'MixedPrecisionPolicy'):
    class _MixedPrecisionPolicy:
        def __init__(self, *a, **k): pass
    _fsdp.MixedPrecisionPolicy = _MixedPrecisionPolicy

import torchaudio.compliance.kaldi as Kaldi
import soundfile as sf
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    import noisereduce as _nr_module
except ImportError:
    _nr_module = None

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.environ.setdefault('MODELSCOPE_CACHE',
                      os.path.join(PROJECT_ROOT, 'pretrained', 'modelscope_cache'))

FREEZE_MODULES = ["head", "xvector.tdnn", "xvector.block1", "xvector.transit1"]
NUM_FRAMES = 149  # 约1.5s (kws全长), fbank 10ms帧移
SR = 16000


def margin_loss(sim_pos, sim_neg, pos_margin=0.7, neg_margin=0.3):
    """
    双向margin对比损失 (防塌缩: 不把pos推向cos=1, 避免embedding空间收窄)
      pos: 只要求 sim > pos_margin(0.7), 不推向1
      neg: 要求 sim < neg_margin(0.3)
      pos/neg 之间保持 0.4 间隔, embedding空间无需塌缩即可满足
    """
    loss_pos = F.relu(pos_margin - sim_pos).mean()
    loss_neg = F.relu(sim_neg - neg_margin).mean()
    return loss_pos + loss_neg


# ==================== 模型加载 ====================

def load_camplus_for_train(device):
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    sv_pipeline = pipeline(
        task=Tasks.speaker_verification,
        model="iic/speech_campplus_sv_zh-cn_16k-common",
    )
    wrapper = sv_pipeline.model
    emb_model = wrapper.embedding_model
    emb_model.to(device)
    return wrapper, emb_model


def setup_trainable(emb_model):
    """冻结主干前半 + 全部BN设eval"""
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


# ==================== fbank 特征缓存 ====================

def _extract_fbank_one(args):
    """子进程: 提取单个音频的fbank特征
    与pipeline预处理一致: cmd先降噪(noisereduce stationary=True prop_decrease=0.8), kws不降噪
    解决训练(原始音频)/推理(降噪音频)分布不匹配导致的neg sim升高问题"""
    rel_path, data_root = args
    y, _ = sf.read(os.path.join(data_root, rel_path), dtype='float32')
    if os.path.basename(rel_path).startswith('cmd_') and _nr_module is not None:
        y = _nr_module.reduce_noise(y=y, sr=SR, stationary=True, prop_decrease=0.8)
    wav = torch.from_numpy(y.astype(np.float32)).unsqueeze(0)
    feat = Kaldi.fbank(wav, num_mel_bins=80)  # [T, 80]
    return rel_path, feat.numpy().astype(np.float32)


class FbankCache:
    """全部音频的fbank特征内存缓存"""

    def __init__(self, rel_paths, data_root, workers=8):
        self.cache = {}
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_extract_fbank_one, (p, data_root)) for p in rel_paths]
            for i, fut in enumerate(as_completed(futures)):
                path, feat = fut.result()
                self.cache[path] = feat
                if (i + 1) % 2000 == 0:
                    print(f"  fbank缓存: {i+1}/{len(rel_paths)} ({time.time()-t0:.0f}s)", flush=True)
        total_mb = sum(f.nbytes for f in self.cache.values()) / 1e6
        print(f"  fbank缓存完成: {len(self.cache)}条, {total_mb:.0f}MB, 耗时{time.time()-t0:.0f}s")

    def get(self, rel_path):
        return self.cache[rel_path]


def crop_or_pad(feat, num_frames, rng):
    """长的随机裁剪, 短的repeat补齐"""
    T = feat.shape[0]
    if T >= num_frames:
        start = int(rng.integers(0, T - num_frames + 1))
        return feat[start:start + num_frames]
    reps = int(np.ceil(num_frames / T))
    return np.tile(feat, (reps, 1))[:num_frames]


# ==================== 数据 ====================

class PairData:
    def __init__(self, jsonl_path):
        self.records = []  # (kws_path, cmd_path, label)
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                label = 1 if r["识别文本"] is not None else 0
                self.records.append((r["唤醒音频"], r["识别音频"], label))

    def __len__(self):
        return len(self.records)


def make_batch(cache, records, indices, rng, device):
    """构造一个batch: kws和cmd合并 [2B, NUM_FRAMES, 80]"""
    feats = []
    labels = []
    for i in indices:
        kws_path, cmd_path, label = records[i]
        feats.append(crop_or_pad(cache.get(kws_path), NUM_FRAMES, rng))
        labels.append(label)
    for i in indices:
        kws_path, cmd_path, label = records[i]
        feats.append(crop_or_pad(cache.get(cmd_path), NUM_FRAMES, rng))
    X = torch.from_numpy(np.stack(feats)).to(device)  # [2B, T, 80]
    X = X - X.mean(dim=1, keepdim=True)  # CMN (时间维)
    y = torch.tensor([1.0 if l == 1 else -1.0 for l in labels], device=device)
    return X, y


def make_pair_batch(cache, pos_records, neg_records, pos_idx, neg_idx, rng, device):
    """构造1:1平衡batch: [kws_pos(B), cmd_pos(B), kws_neg(B), cmd_neg(B)] → [4B,T,80]"""
    feats = []
    for i in pos_idx:
        feats.append(crop_or_pad(cache.get(pos_records[i][0]), NUM_FRAMES, rng))
    for i in pos_idx:
        feats.append(crop_or_pad(cache.get(pos_records[i][1]), NUM_FRAMES, rng))
    for i in neg_idx:
        feats.append(crop_or_pad(cache.get(neg_records[i][0]), NUM_FRAMES, rng))
    for i in neg_idx:
        feats.append(crop_or_pad(cache.get(neg_records[i][1]), NUM_FRAMES, rng))
    X = torch.from_numpy(np.stack(feats)).to(device)
    X = X - X.mean(dim=1, keepdim=True)  # CMN
    return X


# ==================== 评估 ====================

def compute_eer(sims, labels):
    best_eer, best_thr = 1.0, 0.0
    pos_sims = [s for s, l in zip(sims, labels) if l == 1]
    neg_sims = [s for s, l in zip(sims, labels) if l == 0]
    for t in np.arange(0.0, 1.0, 0.005):
        frr = sum(1 for s in pos_sims if s < t) / max(len(pos_sims), 1)
        far = sum(1 for s in neg_sims if s >= t) / max(len(neg_sims), 1)
        eer = (frr + far) / 2
        if eer < best_eer:
            best_eer, best_thr = eer, t
    return best_eer, best_thr


@torch.no_grad()
def validate(emb_model, cache, val_data, device):
    """验证: 全长特征逐条 (与推理一致)"""
    emb_model.eval()
    sims, labels = [], []
    for kws_path, cmd_path, label in val_data.records:
        embs = []
        for path in [kws_path, cmd_path]:
            feat = torch.from_numpy(cache.get(path)).to(device)
            feat = feat - feat.mean(dim=0, keepdim=True)  # CMN
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
    ap.add_argument("--aug_root", required=True)
    ap.add_argument("--fold", type=str, default="0", help="折编号(0-4)或 full(全量训练)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--pos_margin", type=float, default=0.7, help="pos sim 目标下限")
    ap.add_argument("--neg_margin", type=float, default=0.3, help="neg sim 目标上限")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = args.out_dir or os.path.join(PROJECT_ROOT, "runs", f"fold_{args.fold}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"device={device}, fold={args.fold}, batch={args.batch}, out={out_dir}", flush=True)

    # 数据
    fold_dir = os.path.join(args.aug_root, "folds", f"fold_{args.fold}")
    train_data = PairData(os.path.join(fold_dir, "train.jsonl"))
    val_data = PairData(os.path.join(fold_dir, "val.jsonl"))
    n_pos = sum(1 for r in train_data.records if r[2] == 1)
    print(f"train: {len(train_data)} 对 (pos={n_pos}, neg={len(train_data)-n_pos}), val: {len(val_data)} 对", flush=True)

    # fbank 缓存 (train + val 全部音频)
    all_paths = sorted({p for rec in train_data.records + val_data.records for p in rec[:2]})
    cache = FbankCache(all_paths, args.aug_root, workers=args.workers)

    # 模型
    wrapper, emb_model = load_camplus_for_train(device)
    trainable_params = setup_trainable(emb_model)

    # 基线 EER
    eer0, thr0, pm0, nm0 = validate(emb_model, cache, val_data, device)
    print(f"[基线] val EER={eer0:.4f} @thr={thr0:.3f}, pos_sim={pm0:.3f}, neg_sim={nm0:.3f}", flush=True)

    # 平衡采样: pos/neg 分开, 每步各取 half (消除pos拉力优势, 防塌缩)
    pos_records = [r for r in train_data.records if r[2] == 1]
    neg_records = [r for r in train_data.records if r[2] == 0]
    half = args.batch // 2
    print(f"平衡采样: pos={len(pos_records)}, neg={len(neg_records)}, 每步各{half}", flush=True)

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_eer = eer0
    log = {"args": vars(args), "baseline_eer": eer0, "epochs": []}
    rng = np.random.default_rng(int(args.fold) if args.fold.isdigit() else 999)

    for epoch in range(args.epochs):
        t0 = time.time()
        epoch_loss, n_batch = 0.0, 0
        pos_order = rng.permutation(len(pos_records))

        for start in range(0, len(pos_order), half):
            pos_idx = pos_order[start:start + half]
            if len(pos_idx) < half:
                break
            neg_idx = rng.integers(0, len(neg_records), half)  # 有放回采样
            X = make_pair_batch(cache, pos_records, neg_records, pos_idx, neg_idx, rng, device)
            emb = F.normalize(emb_model(X), dim=1)  # [4B, D]
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
            epoch_loss += loss.item()
            n_batch += 1

        scheduler.step()

        # 塌缩监控
        with torch.no_grad():
            idx = rng.choice(len(train_data), min(64, len(train_data)), replace=False)
            X, _ = make_batch(cache, train_data.records, idx, rng, device)
            e = F.normalize(emb_model(X), dim=1)
            emb_std = float(e.std(dim=0).mean().item())

        eer, thr, pm, nm = validate(emb_model, cache, val_data, device)
        dt = time.time() - t0
        improved = eer < best_eer
        if improved:
            best_eer = eer
            torch.save(emb_model.state_dict(), os.path.join(out_dir, "camplus_finetuned_best.pt"))

        ep_log = {"epoch": epoch + 1, "loss": epoch_loss / max(n_batch, 1),
                  "val_eer": eer, "val_thr": float(thr), "pos_sim": pm,
                  "neg_sim": nm, "emb_std": emb_std, "time_s": round(dt, 1),
                  "best": improved}
        log["epochs"].append(ep_log)
        print(f"[E{epoch+1}/{args.epochs}] loss={ep_log['loss']:.4f} val_EER={eer:.4f}@thr={thr:.3f} "
              f"pos={pm:.3f} neg={nm:.3f} emb_std={emb_std:.4f} {'★BEST' if improved else ''} ({dt:.0f}s)",
              flush=True)

        if emb_std < 0.01:
            print(f"⚠️ 警告: emb_std={emb_std:.4f} < 0.01, 疑似embedding塌缩! 停止训练")
            break

    with open(os.path.join(out_dir, "train_log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f"训练完成. 基线EER={eer0:.4f} → 最佳EER={best_eer:.4f}, 权重: {out_dir}/camplus_finetuned_best.pt", flush=True)


if __name__ == "__main__":
    main()
