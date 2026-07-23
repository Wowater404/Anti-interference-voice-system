# -*- coding: utf-8 -*-
"""
fold_0 验证折(368条未见数据)无偏对比: 预训练 vs 微调 阈值表
这是唯一可信的评估 (模型没见过这些数据)
"""
import os
import sys
import json
import numpy as np

import torch
import torch.distributed.fsdp as _fsdp
if not hasattr(_fsdp, 'CPUOffloadPolicy'):
    class _CPUOffloadPolicy:
        def __init__(self, *a, **k): pass
    _fsdp.CPUOffloadPolicy = _CPUOffloadPolicy
if not hasattr(_fsdp, 'MixedPrecisionPolicy'):
    class _MixedPrecisionPolicy:
        def __init__(self, *a, **k): pass
    _fsdp.MixedPrecisionPolicy = _MixedPrecisionPolicy

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.environ.setdefault('MODELSCOPE_CACHE',
                      os.path.join(PROJECT_ROOT, 'pretrained', 'modelscope_cache'))

from modules.voiceprint import CAMPlusExtractor
import soundfile as sf

DATA_ROOT = "C:/Users/善水/Desktop/datasetA/datasetA"
AUG_ROOT = "F:/龙虾/2026-07-18-13-57-00/datasetA_aug"
FT = os.path.join(PROJECT_ROOT, "runs/fold_0/camplus_finetuned_best.pt")
FT_SIMS = os.path.join(PROJECT_ROOT, "runs", "ft_sims_full.npz")


def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # fold_0 验证折 orig_id
    val_recs = load_jsonl(os.path.join(AUG_ROOT, "folds", "fold_0", "val.jsonl"))
    val_pos_ids = [r["orig_id"] for r in val_recs if r["识别文本"] is not None]
    val_neg_ids = [r["orig_id"] for r in val_recs if r["识别文本"] is None]
    print(f"fold_0 验证折: pos={len(val_pos_ids)}, neg={len(val_neg_ids)}")

    # id 不连续 (pos 0-2999, neg 1000-5399), npz按行号存, 需 id→行号映射
    pos_all = load_jsonl(os.path.join(DATA_ROOT, "pos.jsonl"))
    neg_all = load_jsonl(os.path.join(DATA_ROOT, "neg.jsonl"))
    pos_id2row = {r["id"]: i for i, r in enumerate(pos_all)}
    neg_id2row = {r["id"]: i for i, r in enumerate(neg_all)}

    # 微调模型 sim: 从全量 npz 按 id→行号 取
    ft = np.load(FT_SIMS)
    ft_pos = ft["pos"][[pos_id2row[i] for i in val_pos_ids]]
    ft_neg = ft["neg"][[neg_id2row[i] for i in val_neg_ids]]

    # 预训练模型 sim: 对这368条重新提取
    print("提取预训练模型 sim ...")
    ext = CAMPlusExtractor(device=device, finetuned_path=None)
    ext.load()

    def sim_of(rec):
        y_k, _ = sf.read(os.path.join(DATA_ROOT, rec["唤醒音频"]), dtype='float32')
        y_c, _ = sf.read(os.path.join(DATA_ROOT, rec["识别音频"]), dtype='float32')
        ek = ext.extract(y_k, 16000)
        ec = ext.extract(y_c, 16000)
        return float(np.dot(ek, ec) / (np.linalg.norm(ek) * np.linalg.norm(ec) + 1e-8))

    pt_pos = np.array([sim_of(pos_all[pos_id2row[i]]) for i in val_pos_ids])
    pt_neg = np.array([sim_of(neg_all[neg_id2row[i]]) for i in val_neg_ids])

    print(f"\n{'模型':<10} | pos_mean | neg_mean | pos/neg分离")
    print("-" * 55)
    print(f"{'预训练':<10} | {pt_pos.mean():.4f}   | {pt_neg.mean():.4f}   | {pt_pos.mean()-pt_neg.mean():.4f}")
    print(f"{'微调fold0':<10} | {ft_pos.mean():.4f}   | {ft_neg.mean():.4f}   | {ft_pos.mean()-ft_neg.mean():.4f}")

    for name, ps, ns, thr_list in [("预训练", pt_pos, pt_neg, [0.28, 0.35]),
                                    ("微调fold0", ft_pos, ft_neg, [0.28, 0.50, 0.59, 0.65, 0.70, 0.75, 0.80])]:
        print(f"\n[{name}] 阈值表 (验证折, 无偏):")
        print("阈值  | pos接受率     | neg拒识率(RR) | neg假接受数")
        for t in thr_list:
            acc = (ps >= t).mean()
            rej = (ns < t).mean()
            fa = (ns >= t).sum()
            print(f" {t:.2f}  | {acc*100:6.2f}% ({(ps>=t).sum():3d}/{len(ps)}) | {rej*100:6.2f}%        | {fa:3d}/{len(ns)}")

    np.savez(os.path.join(PROJECT_ROOT, "runs", "fold0_val_sims.npz"),
             pt_pos=pt_pos, pt_neg=pt_neg, ft_pos=ft_pos, ft_neg=ft_neg)


if __name__ == "__main__":
    main()
