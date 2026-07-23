# -*- coding: utf-8 -*-
"""
微调声纹模型快速评估: 对比 预训练 vs 微调 在全量 datasetA 上的 sim 分布

只跑声纹阶段 (不跑降噪/分离/ASR), 秒级完成:
  1. 对全部 1838 条原始样本, 分别用两个模型提取 kws/cmd embedding 算 sim
  2. 对比 pos/neg sim 分布 (均值/分离度)
  3. 扫描阈值, 对比各阈值下的 接受率/拒识率/EER
  4. 输出推荐阈值 (供流水线 vp_threshold 使用)

用法:
  python tools/eval_finetuned_voiceprint.py \
      --data_root "C:/Users/善水/Desktop/datasetA/datasetA" \
      --finetuned runs/fold_0/camplus_finetuned_best.pt
"""
import os
import sys
import json
import argparse
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


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def batch_sims(extractor, records, data_root, sr=16000):
    """对全部样本算 kws-cmd 的 cosine sim"""
    import soundfile as sf
    sims = []
    for i, r in enumerate(records):
        y_k, _ = sf.read(os.path.join(data_root, r["唤醒音频"]), dtype='float32')
        y_c, _ = sf.read(os.path.join(data_root, r["识别音频"]), dtype='float32')
        ek = extractor.extract(y_k, sr)
        ec = extractor.extract(y_c, sr)
        sim = float(np.dot(ek, ec) / (np.linalg.norm(ek) * np.linalg.norm(ec) + 1e-8))
        sims.append(sim)
        if (i + 1) % 300 == 0:
            print(f"  进度 {i+1}/{len(records)}", flush=True)
    return np.array(sims)


def dist_stats(name, pos_sims, neg_sims):
    print(f"\n[{name}]")
    print(f"  pos sim: mean={pos_sims.mean():.4f} std={pos_sims.std():.4f} "
          f"min={pos_sims.min():.4f} max={pos_sims.max():.4f}")
    print(f"  neg sim: mean={neg_sims.mean():.4f} std={neg_sims.std():.4f} "
          f"min={neg_sims.min():.4f} max={neg_sims.max():.4f}")
    # Fisher 判别比: 类间距离/类内方差, 越大区分度越好
    fisher = (pos_sims.mean() - neg_sims.mean()) ** 2 / (pos_sims.var() + neg_sims.var() + 1e-12)
    print(f"  Fisher判别比: {fisher:.4f} (越大区分度越好)")
    return fisher


def threshold_sweep(name, pos_sims, neg_sims, key_thresholds=(0.25, 0.28, 0.30, 0.32, 0.35)):
    print(f"\n[{name}] 阈值扫描 (pos接受率 / neg拒识率):")
    best = (1.0, 0.0)
    for t in np.arange(0.10, 0.60, 0.01):
        frr = float((pos_sims < t).mean())
        far = float((neg_sims >= t).mean())
        eer = (frr + far) / 2
        if eer < best[0]:
            best = (eer, t)
    for t in key_thresholds:
        acc = float((pos_sims >= t).mean())
        rej = float((neg_sims < t).mean())
        print(f"  thr={t:.2f}: pos接受={acc*100:.1f}%  neg拒识={rej*100:.1f}%")
    print(f"  ★ 最佳EER={best[0]:.4f} @ thr={best[1]:.2f}")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--finetuned", required=True, help="微调权重路径")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    pos = load_jsonl(os.path.join(args.data_root, "pos.jsonl"))
    neg = load_jsonl(os.path.join(args.data_root, "neg.jsonl"))
    print(f"pos={len(pos)}, neg={len(neg)}, device={args.device}")

    results = {}
    for name, ft_path in [("预训练CAM++", None), ("微调CAM++", args.finetuned)]:
        print(f"\n===== {name} =====")
        ext = CAMPlusExtractor(device=args.device, finetuned_path=ft_path)
        ext.load()
        pos_sims = batch_sims(ext, pos, args.data_root)
        neg_sims = batch_sims(ext, neg, args.data_root)
        results[name] = {"pos": pos_sims, "neg": neg_sims}
        del ext
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 对比
    print("\n" + "=" * 60)
    f_pre = dist_stats("预训练CAM++", results["预训练CAM++"]["pos"], results["预训练CAM++"]["neg"])
    f_ft = dist_stats("微调CAM++", results["微调CAM++"]["pos"], results["微调CAM++"]["neg"])
    e_pre = threshold_sweep("预训练CAM++", results["预训练CAM++"]["pos"], results["预训练CAM++"]["neg"])
    e_ft = threshold_sweep("微调CAM++", results["微调CAM++"]["pos"], results["微调CAM++"]["neg"])

    print("\n===== 总结 =====")
    print(f"Fisher判别比: {f_pre:.4f} → {f_ft:.4f} ({'↑改善' if f_ft > f_pre else '↓变差'})")
    print(f"最佳EER: {e_pre[0]:.4f}@{e_pre[1]:.2f} → {e_ft[0]:.4f}@{e_ft[1]:.2f}")
    # 固定阈值0.28下对比
    for name in results:
        p, n = results[name]["pos"], results[name]["neg"]
        print(f"  {name} @thr=0.28: pos接受={(p>=0.28).mean()*100:.1f}% neg拒识={(n<0.28).mean()*100:.1f}%")

    out = {
        "finetuned": args.finetuned,
        "pretrain": {"pos_mean": float(results["预训练CAM++"]["pos"].mean()),
                     "neg_mean": float(results["预训练CAM++"]["neg"].mean()),
                     "best_eer": e_pre[0], "best_thr": e_pre[1]},
        "finetune": {"pos_mean": float(results["微调CAM++"]["pos"].mean()),
                     "neg_mean": float(results["微调CAM++"]["neg"].mean()),
                     "best_eer": e_ft[0], "best_thr": e_ft[1]},
        "fisher": {"pretrain": f_pre, "finetune": f_ft},
    }
    out_path = os.path.join(PROJECT_ROOT, "runs", "voiceprint_compare.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n对比结果已保存: {out_path}")


if __name__ == "__main__":
    main()
