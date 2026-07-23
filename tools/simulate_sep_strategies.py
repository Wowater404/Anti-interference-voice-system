# -*- coding: utf-8 -*-
"""
模拟不同分离策略的CER/RR, 不需要跑ASR
利用已有数据: 降噪sim(npz) + pipeline分离后sim(JSON) + ASR结果(JSON)
"""
import json, os, sys, unicodedata
import numpy as np

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ = os.path.join(PROJECT, "runs/fold0_val_v3_true.npz")      # 降噪sim
PIPE_JSON = os.path.join(PROJECT, "runs/fold0_pipeline_compare_v3dn.json")  # 分离后sim+ASR
NOSEP_JSON = os.path.join(PROJECT, "runs/eval_fold0_nosep.log")  # 禁分离结果

AUG_ROOT = "F:/龙虾/2026-07-18-13-57-00/datasetA_aug"
VAL_JSONL = os.path.join(AUG_ROOT, "folds/fold_0/val.jsonl")

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

# 加载数据
npz = np.load(NPZ)
dn_pos, dn_neg = npz['pos'], npz['neg']  # 降噪sim

pj = json.load(open(PIPE_JSON, encoding='utf-8'))
v3_res = pj['v3_results']  # 分离后的结果(含separated sim + ASR content)
base_res = pj['baseline_results']  # V4.1基线结果

val = [json.loads(l) for l in open(VAL_JSONL, encoding='utf-8') if l.strip()]
pos_recs = [r for r in val if r['识别文本'] is not None]
neg_recs = [r for r in val if r['识别文本'] is None]

# 建id→降噪sim映射
pos_dn = {r['orig_id']: float(dn_pos[i]) for i, r in enumerate(pos_recs)}
neg_dn = {r['orig_id']: float(dn_neg[i]) for i, r in enumerate(neg_recs)}

# 建id→分离后sim+content映射
v3_by_id = {r['orig_id']: r for r in v3_res}

def simulate(vp_threshold, sep_trigger_min, vp_threshold_separated):
    """
    模拟策略:
    1. sim_denoised >= vp_threshold → 直接接受, ASR用降噪音频(=禁分离的content)
    2. sim_denoised < sep_trigger_min → 直接拒识
    3. sep_trigger_min <= sim_denoised < vp_threshold → 分离, 接受如果sep_sim >= vp_threshold_separated
       ASR用分离后音频(=pipeline分离的content)
    """
    pos_accepted = 0
    pos_total = len(pos_recs)
    neg_fa = 0
    neg_total = len(neg_recs)
    cer_errors = 0
    cer_total_len = 0
    
    for r in pos_recs:
        oid = r['orig_id']
        label = r['识别文本']
        label_n = norm(label)
        cer_total_len += len(label_n)
        
        dn_sim = pos_dn[oid]
        v3 = v3_by_id[oid]
        
        if dn_sim >= vp_threshold:
            # 直接接受, ASR用降噪音频
            # 禁分离时ASR的content需要从nosep结果拿, 但我们没有per-sample的nosep JSON
            # 近似: 用v3_res的content (如果v3的sim >= vp_threshold, 说明没分离, content就是降噪ASR)
            # 实际上v3_res里sim >= vp_threshold的样本没有经过分离, content就是降噪ASR
            pos_accepted += 1
            cer_errors += editdist(label_n, norm(v3['content']))
        elif dn_sim < sep_trigger_min:
            # 直接拒识, 输出空
            cer_errors += len(label_n)  # 删除错误
        else:
            # 分离
            sep_sim = v3['similarity']  # 分离后的sim
            if sep_sim >= vp_threshold_separated:
                pos_accepted += 1
                cer_errors += editdist(label_n, norm(v3['content']))
            else:
                cer_errors += len(label_n)  # 拒识=删除错误
    
    for r in neg_recs:
        oid = r['orig_id']
        dn_sim = neg_dn[oid]
        v3 = v3_by_id[oid]
        
        if dn_sim >= vp_threshold:
            neg_fa += 1  # 直接接受(假接受)
        elif dn_sim < sep_trigger_min:
            pass  # 直接拒识
        else:
            sep_sim = v3['similarity']
            if sep_sim >= vp_threshold_separated:
                neg_fa += 1  # 分离后假接受
    
    cer = cer_errors / cer_total_len if cer_total_len else 0
    rr = 1 - neg_fa / neg_total
    score = (1 - cer) * 0.4 + rr * 0.4
    return {
        'cer': cer, 'rr': rr, 'score': score,
        'pos_acc': pos_accepted, 'pos_total': pos_total,
        'neg_fa': neg_fa, 'neg_total': neg_total
    }

# 测试多种策略
print("=" * 80)
print("分离策略模拟 (fold_0验证折 368条, 基于已有分离后sim+ASR数据)")
print("=" * 80)

strategies = [
    # (vp_threshold, sep_trigger_min, vp_threshold_separated, 描述)
    (0.67, 1.01, 0.80, "禁分离 (baseline)"),
    (0.67, 0.00, 0.67, "全分离 thr=0.67 (当前V3 pipeline)"),
    (0.67, 0.50, 0.80, "选择性分离: floor=0.50 sep_thr=0.80"),
    (0.67, 0.50, 0.75, "选择性分离: floor=0.50 sep_thr=0.75"),
    (0.67, 0.50, 0.85, "选择性分离: floor=0.50 sep_thr=0.85"),
    (0.67, 0.55, 0.80, "选择性分离: floor=0.55 sep_thr=0.80"),
    (0.67, 0.45, 0.80, "选择性分离: floor=0.45 sep_thr=0.80"),
    (0.67, 0.50, 0.90, "选择性分离: floor=0.50 sep_thr=0.90"),
    (0.70, 0.50, 0.80, "vp=0.70 floor=0.50 sep_thr=0.80"),
    (0.65, 0.50, 0.80, "vp=0.65 floor=0.50 sep_thr=0.80"),
]

print(f"\n{'策略':<42} | CER    | RR     | Score  | pos接受   | neg假接受")
print("-" * 95)
for vp, floor, sep_thr, desc in strategies:
    r = simulate(vp, floor, sep_thr)
    print(f"{desc:<42} | {r['cer']:.4f} | {r['rr']:.4f} | {r['score']:.4f} | {r['pos_acc']:3d}/{r['pos_total']} | {r['neg_fa']:2d}/{r['neg_total']}")

print("\n注意: CER基于已有ASR结果近似(直接接受用降噪ASR, 分离接受用分离ASR), 实际可能有微小差异")
