# -*- coding: utf-8 -*-
"""验证假设: V3微调模型在降噪后音频上的sim分布是否偏移(训练用原始音频, 推理用降噪音频)"""
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
from modules.denoiser import create_denoiser
import soundfile as sf

DATA_ROOT = "C:/Users/善水/Desktop/datasetA/datasetA"
AUG_ROOT = "F:/龙虾/2026-07-18-13-57-00/datasetA_aug"
V3 = os.path.join(PROJECT_ROOT, "runs/fold_0_v2/camplus_finetuned_best.pt")


def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val = load_jsonl(os.path.join(AUG_ROOT, "folds", "fold_0", "val.jsonl"))

    ext = CAMPlusExtractor(device=device, finetuned_path=V3)
    ext.load()
    denoiser = create_denoiser({"model": "noisereduce"}, device)
    denoiser.load()

    pos_raw, neg_raw, pos_dn, neg_dn = [], [], [], []
    for i, r in enumerate(val):
        y_k, _ = sf.read(os.path.join(AUG_ROOT, r["唤醒音频"]), dtype='float32')
        y_c, _ = sf.read(os.path.join(AUG_ROOT, r["识别音频"]), dtype='float32')
        y_dn = denoiser.denoise(y_c, 16000)
        ek = ext.extract(y_k, 16000)
        # 原始cmd的sim
        ec_raw = ext.extract(y_c, 16000)
        s_raw = float(np.dot(ek, ec_raw) / (np.linalg.norm(ek) * np.linalg.norm(ec_raw) + 1e-8))
        # 降噪后cmd的sim
        ec_dn = ext.extract(y_dn, 16000)
        s_dn = float(np.dot(ek, ec_dn) / (np.linalg.norm(ek) * np.linalg.norm(ec_dn) + 1e-8))
        if r["识别文本"] is not None:
            pos_raw.append(s_raw); pos_dn.append(s_dn)
        else:
            neg_raw.append(s_raw); neg_dn.append(s_dn)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(val)}", flush=True)

    pr, nr, pd, nd = map(np.array, [pos_raw, neg_raw, pos_dn, neg_dn])
    print(f"\npos: 原始sim={pr.mean():.4f}  降噪sim={pd.mean():.4f}  (差{pd.mean()-pr.mean():+.4f})")
    print(f"neg: 原始sim={nr.mean():.4f}  降噪sim={nd.mean():.4f}  (差{nd.mean()-nr.mean():+.4f})  ← 关键")
    print(f"\nneg原始: p75={np.percentile(nr,75):.3f} p90={np.percentile(nr,90):.3f}")
    print(f"neg降噪: p75={np.percentile(nd,75):.3f} p90={np.percentile(nd,90):.3f} max={nd.max():.3f}")
    print("\n降噪音频上阈值表(V3模型):")
    print("阈值  | pos接受率 | neg拒识率 | 假接受")
    for t in [0.50, 0.60, 0.67, 0.70, 0.75, 0.80, 0.85]:
        acc = (pd >= t).mean(); rej = (nd < t).mean(); fa = (nd >= t).sum()
        print(f" {t:.2f}  |  {acc*100:5.1f}%  |  {rej*100:5.1f}%   | {fa:2d}/{len(nd)}")


if __name__ == "__main__":
    main()
