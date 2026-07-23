# -*- coding: utf-8 -*-
"""
fold_0验证折: 选择性分离 + 跳变上限 组合策略
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

def run_strategy(thr, sep_enable, sep_min, sep_thr, jump_cap, tag):
    config = PipelineConfig(os.path.join(PROJECT_ROOT, "configs", "default.yaml"))
    config._cfg["voiceprint"]["cam_plus"]["finetuned_path"] = V3W
    config._cfg["voiceprint"]["threshold"] = thr
    config._cfg["separation"]["enable"] = sep_enable
    config._cfg["separation"]["sep_trigger_min"] = sep_min
    config._cfg["separation"]["vp_threshold_separated"] = sep_thr
    config._cfg["separation"]["sim_jump_cap"] = jump_cap
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
        if (i+1) % 100 == 0:
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

    # 策略D: 选择性分离 + 跳变上限0.30
    print("===== 策略D: floor=0.50 sep=0.80 jump_cap=0.30 =====", flush=True)
    all_results["D_050_080_030"] = run_strategy(0.67, True, 0.50, 0.80, 0.30, "D_jump030")

    # 策略D2: 选择性分离 + 跳变上限0.25
    print("===== 策略D2: floor=0.50 sep=0.80 jump_cap=0.25 =====", flush=True)
    all_results["D2_050_080_025"] = run_strategy(0.67, True, 0.50, 0.80, 0.25, "D2_jump025")

    # 保存
    out_path = os.path.join(PROJECT_ROOT, "runs/jumpcap_sep_eval.json")
    summary = {k: {kk: v for kk, v in val.items() if kk != "results"} for k, val in all_results.items()}
    summary["baseline_nosep"] = {"cer": 0.4527, "rr": 0.9368, "score": 0.5936, "pos_acc": 242, "neg_fa": 6}
    summary["selective_no_jumcap"] = {"cer": 0.4484, "rr": 0.9263, "score": 0.5912, "pos_acc": 250, "neg_fa": 7}
    summary["full_sep"] = {"cer": 0.4396, "rr": 0.6842, "score": 0.4979, "pos_acc": 258, "neg_fa": 30}
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n===== 总结 =====", flush=True)
    print(f"{'策略':<40} | CER    | RR     | Score  | pos接受   | neg假接受", flush=True)
    print("-" * 95, flush=True)
    for tag, label in [("baseline_nosep", "禁分离 (baseline)"),
                       ("full_sep", "全分离 thr=0.67"),
                       ("selective_no_jumcap", "选择性分离 (无跳变上限)"),
                       ("D_050_080_030", "选择性+跳变0.30"),
                       ("D2_050_080_025", "选择性+跳变0.25")]:
        r = summary[tag]
        print(f"{label:<40} | {r['cer']:.4f} | {r['rr']:.4f} | {r['score']:.4f} | {r['pos_acc']:3d}/273  | {r['neg_fa']:2d}/95", flush=True)
