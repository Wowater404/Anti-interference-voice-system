# -*- coding: utf-8 -*-
"""
fold_0 验证折(368条未见数据)完整流水线对比: 预训练CAM++ vs V3微调CAM++
跑真实 pipeline (降噪→自适应分离→声纹→ASR), 输出 CER/RR 对比
这是接入前最终的无偏验证
"""
import os
import sys
import json
import time
import unicodedata
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.environ.setdefault('MODELSCOPE_CACHE',
                      os.path.join(PROJECT_ROOT, 'pretrained', 'modelscope_cache'))

from config import PipelineConfig
from pipeline import VoicePipeline

AUG_ROOT = "F:/龙虾/2026-07-18-13-57-00/datasetA_aug"
V3_WEIGHTS = os.path.join(PROJECT_ROOT, "runs/fold_0_v2/camplus_finetuned_best.pt")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "default.yaml")


def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def normalize_text(t):
    t = unicodedata.normalize('NFKC', t).lower()
    return ''.join(c for c in t if not unicodedata.category(c).startswith('P'))


def editdist(s1, s2):
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]; dp[0] = i
        for j in range(1, n + 1):
            tmp = dp[j]
            dp[j] = prev if s1[i-1] == s2[j-1] else 1 + min(dp[j], dp[j-1], prev)
            prev = tmp
    return dp[n]


def run_pipeline_on_val(name, finetuned_path, vp_thr, vp_thr_sep, val_recs):
    """用指定声纹权重/阈值跑完整流水线, 返回每条结果"""
    print(f"\n===== {name} (thr={vp_thr}/{vp_thr_sep}) =====")
    config = PipelineConfig(CONFIG_PATH)
    # 注入声纹配置
    config._cfg["voiceprint"]["cam_plus"]["finetuned_path"] = finetuned_path
    config._cfg["voiceprint"]["threshold"] = vp_thr
    config._cfg["separation"]["vp_threshold_separated"] = vp_thr_sep
    config._cfg["separation"]["enable"] = True

    pipe = VoicePipeline(config)
    pipe.load_models()

    results = []
    t0 = time.time()
    for i, r in enumerate(val_recs):
        kws_path = os.path.join(AUG_ROOT, r["唤醒音频"])
        cmd_path = os.path.join(AUG_ROOT, r["识别音频"])
        label = r["识别文本"] if r["识别文本"] is not None else ""
        out = pipe.process_sample(kws_path, cmd_path, sample_id=str(r["orig_id"]), label=label)
        results.append({
            "orig_id": r["orig_id"],
            "label": label,
            "content": out["content"],
            "similarity": float(out["similarity"]),
            "is_target": out["is_target"],
            "is_pos": label != "",
        })
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(val_recs)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"  完成 {len(val_recs)} 条, 耗时 {time.time()-t0:.0f}s")
    del pipe
    return results


def calc_metrics(results, name):
    pos = [r for r in results if r["is_pos"]]
    neg = [r for r in results if not r["is_pos"]]
    total_err = sum(editdist(normalize_text(r["label"]), normalize_text(r["content"])) for r in pos)
    total_len = sum(len(normalize_text(r["label"])) for r in pos)
    cer = total_err / total_len if total_len else 0
    neg_rej = sum(1 for r in neg if not r["is_target"])
    rr = neg_rej / len(neg) if neg else 0
    pos_acc = sum(1 for r in pos if r["is_target"])
    score = (1 - cer) * 0.4 + rr * 0.4
    print(f"\n[{name}] CER={cer:.4f} RR={rr:.4f} pos接受={pos_acc}/{len(pos)} "
          f"neg假接受={len(neg)-neg_rej}/{len(neg)} Score(CER40+RR40)={score:.4f}")
    return {"cer": cer, "rr": rr, "pos_acc": pos_acc, "neg_fa": len(neg) - neg_rej, "score": score}


def main():
    val = load_jsonl(os.path.join(AUG_ROOT, "folds", "fold_0", "val.jsonl"))
    print(f"fold_0 验证折: {len(val)} 条 (pos={sum(1 for r in val if r['识别文本'] is not None)})")

    # V4.1 基线: 预训练声纹, thr=0.28/0.35
    r_base = run_pipeline_on_val("V4.1基线(预训练)", None, 0.28, 0.35, val)
    m_base = calc_metrics(r_base, "V4.1基线(预训练)")

    # V3 微调声纹, thr=0.67/0.72
    r_v3 = run_pipeline_on_val("V3微调声纹", V3_WEIGHTS, 0.67, 0.72, val)
    m_v3 = calc_metrics(r_v3, "V3微调声纹")

    print("\n===== 总结 (fold_0验证折, 无偏) =====")
    print(f"{'指标':<12} | {'V4.1基线':>10} | {'V3微调':>10} | {'变化':>10}")
    print(f"{'CER':<12} | {m_base['cer']:>10.4f} | {m_v3['cer']:>10.4f} | {m_v3['cer']-m_base['cer']:>+10.4f}")
    print(f"{'RR':<12} | {m_base['rr']:>10.4f} | {m_v3['rr']:>10.4f} | {m_v3['rr']-m_base['rr']:>+10.4f}")
    print(f"{'Score':<12} | {m_base['score']:>10.4f} | {m_v3['score']:>10.4f} | {m_v3['score']-m_base['score']:>+10.4f}")

    out = {"baseline": m_base, "v3_finetuned": m_v3,
           "baseline_results": r_base, "v3_results": r_v3}
    with open(os.path.join(PROJECT_ROOT, "runs", "fold0_pipeline_compare.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n已保存: runs/fold0_pipeline_compare.json")


if __name__ == "__main__":
    main()
