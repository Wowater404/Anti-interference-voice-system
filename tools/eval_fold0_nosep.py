# -*- coding: utf-8 -*-
"""fold_0验证折: V3微调(降噪版) + 禁分离, 测真实CER/RR"""
import os, sys, json, time, unicodedata
import numpy as np
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.environ.setdefault('MODELSCOPE_CACHE', os.path.join(PROJECT_ROOT, 'pretrained', 'modelscope_cache'))
from config import PipelineConfig
from pipeline import VoicePipeline
AUG_ROOT = "F:/龙虾/2026-07-18-13-57-00/datasetA_aug"
V3W = os.path.join(PROJECT_ROOT, "runs/fold_0_v3_dn/camplus_finetuned_best.pt")

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

def run(thr):
    config = PipelineConfig(os.path.join(PROJECT_ROOT, "configs", "default.yaml"))
    config._cfg["voiceprint"]["cam_plus"]["finetuned_path"] = V3W
    config._cfg["voiceprint"]["threshold"] = thr
    config._cfg["separation"]["enable"] = False   # 禁分离
    pipe = VoicePipeline(config); pipe.load_models()
    val = load_jsonl(os.path.join(AUG_ROOT, "folds", "fold_0", "val.jsonl"))
    res = []
    for i, r in enumerate(val):
        out = pipe.process_sample(os.path.join(AUG_ROOT, r["唤醒音频"]), os.path.join(AUG_ROOT, r["识别音频"]),
                                  sample_id=str(r["orig_id"]), label=r["识别文本"] or "")
        res.append({"label": r["识别文本"] or "", "content": out["content"], "is_target": out["is_target"], "is_pos": r["识别文本"] is not None})
    del pipe
    return res

def metrics(res, tag):
    pos = [r for r in res if r["is_pos"]]; neg = [r for r in res if not r["is_pos"]]
    err = sum(editdist(norm(r["label"]), norm(r["content"])) for r in pos)
    ln = sum(len(norm(r["label"])) for r in pos)
    cer = err/ln if ln else 0
    rr = sum(1 for r in neg if not r["is_target"])/len(neg)
    acc = sum(1 for r in pos if r["is_target"])
    score = (1-cer)*0.4 + rr*0.4
    print(f"[{tag}] CER={cer:.4f} RR={rr:.4f} pos接受={acc}/{len(pos)} neg假接受={sum(1 for r in neg if r['is_target'])}/{len(neg)} Score={score:.4f}", flush=True)
    return score

if __name__ == "__main__":
    for thr in [0.67, 0.70, 0.75]:
        metrics(run(thr), f"V3微调+禁分离 thr={thr}")
