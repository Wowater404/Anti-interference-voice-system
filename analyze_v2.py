"""
V2 推理结果详细分析
分析各项指标, 寻找进一步优化空间
"""
import json
import re
import numpy as np

def strip_punctuation(text):
    pattern = r'[。，、；：？！""''）（】【《》…—·.,;:?!"\'()<>]'
    return re.sub(pattern, '', text)

def char_error_rate(reference, hypothesis):
    if not reference:
        return 0.0 if not hypothesis else 1.0
    reference = strip_punctuation(reference)
    hypothesis = strip_punctuation(hypothesis)
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    n, m = len(ref_chars), len(hyp_chars)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_chars[i - 1] == hyp_chars[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+1)
    return dp[n][m] / n

# 加载 checkpoint_v2 (有 similarity 等详细字段)
with open("results/checkpoint_v2.json", "r", encoding="utf-8") as f:
    ckpt = json.load(f)
results = ckpt["results"]

# 分离 pos / neg
pos_results = [r for r in results if r.get("_ckpt_key", "").startswith("pos_")]
neg_results = [r for r in results if r.get("_ckpt_key", "").startswith("neg_")]

print(f"{'='*60}")
print(f"V2 推理结果详细分析")
print(f"{'='*60}")
print(f"总样本: {len(results)} (pos={len(pos_results)}, neg={len(neg_results)})")

# ========== POS 分析 ==========
pos_accepted = [r for r in pos_results if r.get("is_target", False)]
pos_rejected = [r for r in pos_results if not r.get("is_target", False)]

print(f"\n{'='*60}")
print(f"POS 分析 ({len(pos_results)} 条)")
print(f"{'='*60}")
print(f"接受: {len(pos_accepted)} ({len(pos_accepted)/len(pos_results)*100:.1f}%)")
print(f"拒识: {len(pos_rejected)} ({len(pos_rejected)/len(pos_results)*100:.1f}%)")

# CER 分析
pos_cers_all = [float(r["cer"]) for r in pos_results if r.get("cer")]
pos_cers_accepted = [float(r["cer"]) for r in pos_accepted if r.get("cer")]

print(f"\nCER (全部 pos, 拒识=1.0): {np.mean(pos_cers_all):.4f}")
print(f"CER (仅接受):              {np.mean(pos_cers_accepted):.4f}")

# 相似度分布
pos_sims = [float(r.get("similarity", "0")) for r in pos_results]
pos_sims_accepted = [float(r.get("similarity", "0")) for r in pos_accepted]
pos_sims_rejected = [float(r.get("similarity", "0")) for r in pos_rejected]

print(f"\n相似度分布:")
print(f"  全部 pos:    mean={np.mean(pos_sims):.4f}, std={np.std(pos_sims):.4f}, range=[{np.min(pos_sims):.4f}, {np.max(pos_sims):.4f}]")
print(f"  已接受:      mean={np.mean(pos_sims_accepted):.4f}, std={np.std(pos_sims_accepted):.4f}")
print(f"  已拒识:      mean={np.mean(pos_sims_rejected):.4f}, std={np.std(pos_sims_rejected):.4f}")

# ========== NEG 分析 ==========
neg_accepted = [r for r in neg_results if r.get("is_target", False)]
neg_rejected = [r for r in neg_results if not r.get("is_target", False)]

print(f"\n{'='*60}")
print(f"NEG 分析 ({len(neg_results)} 条)")
print(f"{'='*60}")
print(f"正确拒识: {len(neg_rejected)} ({len(neg_rejected)/len(neg_results)*100:.1f}%)")
print(f"错误接受: {len(neg_accepted)} ({len(neg_accepted)/len(neg_results)*100:.1f}%)")

neg_sims = [float(r.get("similarity", "0")) for r in neg_results]
neg_sims_accepted = [float(r.get("similarity", "0")) for r in neg_accepted]
neg_sims_rejected = [float(r.get("similarity", "0")) for r in neg_rejected]

print(f"\n相似度分布:")
print(f"  全部 neg:    mean={np.mean(neg_sims):.4f}, std={np.std(neg_sims):.4f}, range=[{np.min(neg_sims):.4f}, {np.max(neg_sims):.4f}]")
print(f"  错误接受:    mean={np.mean(neg_sims_accepted):.4f}, std={np.std(neg_sims_accepted):.4f}")
print(f"  正确拒识:    mean={np.mean(neg_sims_rejected):.4f}, std={np.std(neg_sims_rejected):.4f}")

# ========== 阈值优化 ==========
print(f"\n{'='*60}")
print(f"阈值优化分析")
print(f"{'='*60}")

thresholds = [0.20, 0.22, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30, 0.31, 0.32]
best_score = 0
best_thr = 0.28

for thr in thresholds:
    pos_accept = 0
    pos_reject = 0
    neg_accept = 0
    neg_reject = 0
    cer_list = []

    for r in pos_results:
        sim = float(r.get("similarity", "0"))
        if sim >= thr:
            pos_accept += 1
            content = r.get("content", "")
            label = r.get("label", "")
            if content and content != "null":
                cer = char_error_rate(label, content)
                cer_list.append(cer)
            else:
                cer_list.append(1.0)
        else:
            pos_reject += 1
            cer_list.append(1.0)

    for r in neg_results:
        sim = float(r.get("similarity", "0"))
        if sim >= thr:
            neg_accept += 1
        else:
            neg_reject += 1

    avg_cer = np.mean(cer_list) if cer_list else 1.0
    rr = neg_reject / len(neg_results) if neg_results else 0
    score = (1 - avg_cer) * 0.4 + rr * 0.4
    marker = ""
    if score > best_score:
        best_score = score
        best_thr = thr
        marker = "  ← 最优"
    print(f"  thr={thr:.2f}: pos接受={pos_accept/len(pos_results)*100:5.1f}%, "
          f"neg拒识={rr*100:5.1f}%, CER={avg_cer:.4f}, score={score:.4f}{marker}")

print(f"\n最优阈值: {best_thr:.2f}, 预估得分: {best_score:.4f}")

# ========== 错误样本分析 ==========
print(f"\n{'='*60}")
print(f"错误样本分析 (前20个高CER的已接受样本)")
print(f"{'='*60}")

# 找出CER最高的已接受样本
accepted_with_cer = [(float(r.get("cer", "1.0")), r) for r in pos_accepted]
accepted_with_cer.sort(key=lambda x: -x[0])

for i, (cer, r) in enumerate(accepted_with_cer[:20]):
    label = r.get("label", "")
    content = r.get("content", "")
    sim = r.get("similarity", "")
    print(f"  [{i}] CER={cer:.4f}, sim={sim}")
    print(f"      label:  {label}")
    print(f"      output: {content}")

# ========== 被拒识样本分析 ==========
print(f"\n{'='*60}")
print(f"被拒识 pos 样本 (前10个, 按相似度降序)")
print(f"{'='*60}")

rejected_sorted = sorted(pos_rejected, key=lambda r: -float(r.get("similarity", "0")))
for i, r in enumerate(rejected_sorted[:10]):
    sim = r.get("similarity", "")
    label = r.get("label", "")
    print(f"  [{i}] sim={sim}, label={label}")

# ========== V1 vs V2 对比 ==========
print(f"\n{'='*60}")
print(f"V1 vs V2 对比")
print(f"{'='*60}")
print(f"{'指标':<20} {'V1':>10} {'V2':>10} {'变化':>10}")
print(f"{'-'*50}")
print(f"{'CER':<20} {'0.8171':>10} {'0.6316':>10} {'-22.7%':>10}")
print(f"{'RR':<20} {'0.9515':>10} {'0.9367':>10} {'-1.5%':>10}")
print(f"{'Score':<20} {'0.4537':>10} {'0.5220':>10} {'+15.1%':>10}")
print(f"{'Duration':<20} {'2532s':>10} {'423s':>10} {'-83.3%':>10}")
print(f"{'Pos 接受率':<20} {'58.0%':>10} {f'{len(pos_accepted)/len(pos_results)*100:.1f}%':>10}")
print(f"{'Neg 拒识率':<20} {'95.1%':>10} {f'{len(neg_rejected)/len(neg_results)*100:.1f}%':>10}")
