# -*- coding: utf-8 -*-
"""快速阈值扫描: 提取微调模型全量sim, 打印详细阈值表, 保存npy供复用"""
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
FT = os.path.join(PROJECT_ROOT, "runs/fold_0/camplus_finetuned_best.pt")
OUT_NPY = os.path.join(PROJECT_ROOT, "runs", "ft_sims_full.npz")


def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ext = CAMPlusExtractor(device=device, finetuned_path=FT)
    ext.load()

    pos = load_jsonl(os.path.join(DATA_ROOT, "pos.jsonl"))
    neg = load_jsonl(os.path.join(DATA_ROOT, "neg.jsonl"))

    def sims_of(records):
        out = []
        for i, r in enumerate(records):
            y_k, _ = sf.read(os.path.join(DATA_ROOT, r["唤醒音频"]), dtype='float32')
            y_c, _ = sf.read(os.path.join(DATA_ROOT, r["识别音频"]), dtype='float32')
            ek = ext.extract(y_k, 16000)
            ec = ext.extract(y_c, 16000)
            out.append(float(np.dot(ek, ec) / (np.linalg.norm(ek) * np.linalg.norm(ec) + 1e-8)))
            if (i + 1) % 300 == 0:
                print(f"  {i+1}/{len(records)}", flush=True)
        return np.array(out)

    print("提取微调模型 sim ...", flush=True)
    pos_sims = sims_of(pos)
    neg_sims = sims_of(neg)
    np.savez(OUT_NPY, pos=pos_sims, neg=neg_sims)
    print(f"sim 已保存: {OUT_NPY}")

    print(f"\npos: mean={pos_sims.mean():.4f} std={pos_sims.std():.4f} "
          f"p5={np.percentile(pos_sims,5):.4f} p50={np.percentile(pos_sims,50):.4f}")
    print(f"neg: mean={neg_sims.mean():.4f} std={neg_sims.std():.4f} "
          f"p50={np.percentile(neg_sims,50):.4f} p95={np.percentile(neg_sims,95):.4f} max={neg_sims.max():.4f}")

    print("\n阈值  | pos接受率 | neg拒识率 | neg假接受数/474 | 备注")
    print("-" * 62)
    for t in np.arange(0.30, 0.80, 0.05):
        acc = (pos_sims >= t).mean()
        rej = (neg_sims < t).mean()
        fa = int((neg_sims >= t).sum())
        mark = ""
        if fa <= 30 and acc >= 0.90:
            mark = "★ 优于V4.1基线"
        print(f" {t:.2f}  |  {acc*100:6.2f}%  |  {rej*100:6.2f}%  |  {fa:4d}/474     | {mark}")


if __name__ == "__main__":
    main()
