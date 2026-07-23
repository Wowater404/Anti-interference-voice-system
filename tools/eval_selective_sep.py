# -*- coding: utf-8 -*-
"""
fold_0验证折: 选择性分离策略 真实pipeline验证
策略A: separation.enable=true, sep_trigger_min=0.50, vp_threshold=0.67, vp_threshold_separated=0.80
对比: 禁分离 baseline (Score=0.5936)
"""
import os, sys, json, time, unicodedata
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.environ.setdefault('MODELSCOPE_CACHE', os.path.join(PROJECT_ROOT, 'pretrained', 'modelscope_cache'))
from config import PipelineConfig
from pipeline import VoicePipeline

AUG_ROOT = "F:/龙虾/2026-07-18-13-57-00/datasetA_aug"
V3W = os.path.join(PROJECT_ROOT, "finetuned_models/camplus_v3_fold0.pt")
VAL_JSONL = os.path.join(AUG_ROOT, "folds", "fold_0", "val.jsonl")

def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]

def norm(t):
    t = unicodedata.normalize('NFKC', t).lower()
    return ''.join(c for c in t if not unicodedata.category(c).startswith('P'))

def editdist(a, b):
    m, n = len(a), len(b); dp = list(range(n+1))
    for i in range(1, m+1):
        pv = dp[0]; dp[0] = i
        for j in range(1, n+1):
            tp = dp[j]; dp[j] = pv if a[i-1]==b[j-1] else 1+min(dp[j],dp[j-1],pv); pv = tp
    return dp[n]

def run_strategy(thr, sep_enable, sep_min, sep_thr, tag):
    config = PipelineConfig(os.path.join(PROJECT_ROOT, "configs", "default.yaml"))
    config._cfg["voiceprint"]["cam_plus"]["finetuned_path"] = V3W
    config._cfg["voiceprint"]["threshold"] = thr
    config._cfg["separation"]["enable"] = sep_enable
    config._cfg["separation"]["sep_trigger_min"] = sep_min
    config._cfg["separation"]["vp_threshold_separated"] = sep_thr
    pipe = VoicePipeline(config)
    pipe.load_models()

    val = load_jsonl(VAL_JSONL)
    results = []
    t0 = time.time()
    for i, r in enumerate(val):
        out = pipe.process_sample(
            os.path.join(AUG_ROOT, r["唤醒音频"]),
            os.path.join(AUG_ROOT, r["识别音频"]),
            sample_id=str(r["orig_id"]),
            label=r["识别文本"] or ""
        )
        results.append({
            "orig_id": r["orig_id"],
            "label": r["识别文本"] or "",
            "content": out["content"],
            "similarity": out["similarity"],
            "is_target": out["is_target"],
            "is_pos": r["识别文本"] is not None,
        })
        if (i+1) % 50 == 0:
            print(f"  [{tag}] {i+1}/{len(val)}...", flush=True)
    elapsed = time.time() - t0
    del pipe

    pos = [r for r in results if r["is_pos"]]
    neg = [r for r in results if not r["is_pos"]]
    err = sum(editdist(norm(r["label"]), norm(r["content"])) for r in pos)
    ln = sum(len(norm(r["label"])) for r in pos)
    cer = err / ln if ln else 0
    rr = sum(1 for r in neg if not r["is_target"]) / len(neg)
    acc = sum(1 for r in pos if r["is_target"])
    fa = sum(1 for r in neg if r["is_target"])
    score = (1 - cer) * 0.4 + rr * 0.4
    print(f"[{tag}] CER={cer:.4f} RR={rr:.4f} pos={acc}/{len(pos)} fa={fa}/{len(neg)} Score={score:.4f} ({elapsed:.0f}s)", flush=True)
    return {"cer": cer, "rr": rr, "score": score, "pos_acc": acc, "neg_fa": fa, "results": results}

if __name__ == "__main__":
    all_results = {}

    # 策略A: 选择性分离 floor=0.50 sep_thr=0.80
    print("===== 策略A: 选择性分离 floor=0.50 sep_thr=0.80 =====", flush=True)
    all_results["selective_050_080"] = run_strategy(0.67, True, 0.50, 0.80, "sel_050_080")

    # 策略A2: 选择性分离 floor=0.50 sep_thr=0.85
    print("===== 策略A2: 选择性分离 floor=0.50 sep_thr=0.85 =====", flush=True)
    all_results["selective_050_085"] = run_strategy(0.67, True, 0.50, 0.85, "sel_050_085")

    # 保存
    out_path = os.path.join(PROJECT_ROOT, "runs/selective_sep_eval.json")
    # 去掉results中的详细数据, 只保留指标
    summary = {k: {kk: v for kk, v in val.items() if kk != "results"} for k, val in all_results.items()}
    summary["baseline_nosep"] = {"cer": 0.4527, "rr": 0.9368, "score": 0.5936, "pos_acc": 242, "neg_fa": 6}
    summary["full_sep"] = {"cer": 0.4396, "rr": 0.6842, "score": 0.4979, "pos_acc": 258, "neg_fa": 30}
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n===== 总结 =====", flush=True)
    print(f"{'策略':<30} | CER    | RR     | Score  | pos接受   | neg假接受", flush=True)
    print("-" * 85, flush=True)
    for tag, label in [("baseline_nosep", "禁分离 (baseline)"),
                       ("full_sep", "全分离 thr=0.67"),
                       ("selective_050_080", "选择性 floor=0.50 sep=0.80"),
                       ("selective_050_085", "选择性 floor=0.50 sep=0.85")]:
        r = summary[tag]
        print(f"{label:<30} | {r['cer']:.4f} | {r['rr']:.4f} | {r['score']:.4f} | {r['pos_acc']:3d}/273  | {r['neg_fa']:2d}/95", flush=True)
