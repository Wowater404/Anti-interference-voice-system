# -*- coding: utf-8 -*-
"""直接用指定权重对fold_0验证折提取sim并输出阈值表"""
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
FT = os.path.join(PROJECT_ROOT, "runs/fold_0_v2/camplus_finetuned_best.pt")


def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val = load_jsonl(os.path.join(AUG_ROOT, "folds", "fold_0", "val.jsonl"))

    ext = CAMPlusExtractor(device=device, finetuned_path=FT)
    ext.load()

    pos_sims, neg_sims = [], []
    for i, r in enumerate(val):
        y_k, _ = sf.read(os.path.join(AUG_ROOT, r["唤醒音频"]), dtype='float32')
        y_c, _ = sf.read(os.path.join(AUG_ROOT, r["识别音频"]), dtype='float32')
        ek = ext.extract(y_k, 16000)
        ec = ext.extract(y_c, 16000)
        s = float(np.dot(ek, ec) / (np.linalg.norm(ek) * np.linalg.norm(ec) + 1e-8))
        if r["识别文本"] is not None:
            pos_sims.append(s)
        else:
            neg_sims.append(s)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(val)}", flush=True)

    ps, ns = np.array(pos_sims), np.array(neg_sims)
    print(f"\nV3 fold_0验证折: pos_mean={ps.mean():.4f} neg_mean={ns.mean():.4f}")
    print(f"neg分布: p50={np.percentile(ns,50):.3f} p75={np.percentile(ns,75):.3f} p90={np.percentile(ns,90):.3f} max={ns.max():.3f}")
    print("\n阈值  | pos接受率      | neg拒识率(RR) | 假接受")
    for t in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.67]:
        acc = (ps >= t).mean(); rej = (ns < t).mean(); fa = (ns >= t).sum(); pa = (ps >= t).sum()
        print(f" {t:.2f}  | {acc*100:5.1f}% ({pa:3d}/{len(ps)}) | {rej*100:5.1f}%        | {fa:2d}/{len(ns)}")
    np.savez(os.path.join(PROJECT_ROOT, "runs", "fold0_val_v3_true.npz"), pos=ps, neg=ns)


if __name__ == "__main__":
    main()
