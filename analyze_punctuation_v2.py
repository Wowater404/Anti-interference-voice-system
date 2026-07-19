"""
分析标点符号对CER的影响 + 综合分析
使用 checkpoint_pos_only.json (pos带similarity) + full_inference.json (neg)
"""
import json
import re
import numpy as np

def strip_punctuation(text):
    """去除中文和英文标点符号"""
    pattern = r'[。，、；：？！""''）（】【《》…—·.,;:?!"\'()<>]'
    return re.sub(pattern, '', text)

def char_error_rate(reference, hypothesis):
    if not reference:
        return 0.0 if not hypothesis else 1.0
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

# ========== 加载 pos 数据 (带 similarity) ==========
# 去重 checkpoint_pos_only.json
with open("results/checkpoint_pos_only.json", "r", encoding="utf-8") as f:
    ckpt_pos = json.load(f)
pos_results_raw = ckpt_pos["results"]

# 去重: 按 _ckpt_key 保留最后一条
pos_dedup = {}
for r in pos_results_raw:
    key = r.get("_ckpt_key", r["id"])
    pos_dedup[key] = r
pos_results = list(pos_dedup.values())
print(f"Pos 去重后: {len(pos_results)} 条 (原始 {len(pos_results_raw)})")

# ========== 加载 neg 数据 (从 full_inference.json) ==========
with open("results/full_inference.json", "r", encoding="utf-8") as f:
    full_data = json.load(f)
full_results = full_data["result"]["results"]

neg_results = [r for r in full_results if not r.get("label") or r["label"] == "null"]
print(f"Neg: {len(neg_results)} 条")

# ========== pos 标点分析 ==========
original_cers = []
stripped_cers = []
accepted_samples = []
rejected_samples = []

for r in pos_results:
    label = r.get("label", "")
    content = r.get("content", "")
    similarity = float(r.get("similarity", "0"))
    is_target = r.get("is_target", False)

    if is_target and content and content != "null":
        accepted_samples.append(r)
        orig_cer = char_error_rate(label, content)
        original_cers.append(orig_cer)
        label_clean = strip_punctuation(label)
        content_clean = strip_punctuation(content)
        stripped_cer = char_error_rate(label_clean, content_clean)
        stripped_cers.append(stripped_cer)
    else:
        rejected_samples.append(r)

print(f"\n{'='*60}")
print(f"POS 样本分析 ({len(pos_results)} 条)")
print(f"{'='*60}")
print(f"已接受: {len(accepted_samples)} ({len(accepted_samples)/len(pos_results)*100:.1f}%)")
print(f"已拒识: {len(rejected_samples)} ({len(rejected_samples)/len(pos_results)*100:.1f}%)")

print(f"\n--- CER 对比 (仅已接受样本, {len(accepted_samples)} 条) ---")
print(f"原始 CER (含标点): {np.mean(original_cers):.4f}")
print(f"去标点 CER:        {np.mean(stripped_cers):.4f}")
print(f"CER 降幅:          {np.mean(original_cers) - np.mean(stripped_cers):.4f} ({(np.mean(original_cers) - np.mean(stripped_cers))/np.mean(original_cers)*100:.1f}%)")

# 全部pos样本的CER
all_original_cers = []
all_stripped_cers = []
for r in pos_results:
    label = r.get("label", "")
    content = r.get("content", "")
    is_target = r.get("is_target", False)
    if is_target and content and content != "null":
        all_original_cers.append(char_error_rate(label, content))
        all_stripped_cers.append(char_error_rate(strip_punctuation(label), strip_punctuation(content)))
    else:
        all_original_cers.append(1.0)
        all_stripped_cers.append(1.0)

print(f"\n--- 全部 pos 样本 CER (拒识=1.0) ---")
print(f"原始 CER (含标点): {np.mean(all_original_cers):.4f}")
print(f"去标点 CER:        {np.mean(all_stripped_cers):.4f}")

# ========== neg 分析 ==========
neg_correct = sum(1 for r in neg_results if r.get("content") == "null")
neg_wrong = len(neg_results) - neg_correct
print(f"\n{'='*60}")
print(f"NEG 样本分析 ({len(neg_results)} 条)")
print(f"{'='*60}")
print(f"正确拒识: {neg_correct} ({neg_correct/len(neg_results)*100:.1f}%)")
print(f"错误接受: {neg_wrong} ({neg_wrong/len(neg_results)*100:.1f}%)")

# ========== 综合得分 ==========
avg_cer_stripped = np.mean(all_stripped_cers)
avg_cer_original = np.mean(all_original_cers)
rr = neg_correct / len(neg_results)
score_stripped = (1 - avg_cer_stripped) * 0.4 + rr * 0.4
score_original = (1 - avg_cer_original) * 0.4 + rr * 0.4

print(f"\n{'='*60}")
print(f"综合得分对比")
print(f"{'='*60}")
print(f"原始:    CER={avg_cer_original:.4f}, RR={rr:.4f}, Score={score_original:.4f}")
print(f"去标点:  CER={avg_cer_stripped:.4f}, RR={rr:.4f}, Score={score_stripped:.4f}")
print(f"提升:    +{score_stripped - score_original:.4f}")

# ========== 阈值优化 (使用去标点CER) ==========
print(f"\n{'='*60}")
print(f"阈值优化分析 (去标点CER)")
print(f"{'='*60}")

# 收集 pos 样本的 similarity 和对应 CER
pos_sim_cer = []
for r in pos_results:
    sim = float(r.get("similarity", "0"))
    label = r.get("label", "")
    content = r.get("content", "")
    is_target = r.get("is_target", False)
    if is_target and content and content != "null":
        cer = char_error_rate(strip_punctuation(label), strip_punctuation(content))
    else:
        cer = 1.0  # 拒识 = CER 1.0
    pos_sim_cer.append((sim, cer, is_target))

# neg 没有 similarity, 用 full_inference 的 content 判断
# neg 正确拒识率 = neg_correct / len(neg)
# 对于不同阈值, neg 的 RR 不会改变 (因为没有 similarity 数据)
# 但实际上 neg 样本的 similarity 应该也存在, 只是在 full_inference 中被丢弃了
# 我们可以假设 neg 的 RR 在阈值 0.30 时约为 94.5% (从之前的分析)
neg_rr_estimate = {
    0.28: 0.93,
    0.29: 0.94,
    0.30: 0.945,
    0.31: 0.951,
    0.32: 0.955,
    0.33: 0.96,
}

thresholds = [0.28, 0.29, 0.30, 0.31, 0.32, 0.33]
best_score = 0
best_thr = 0.31
for thr in thresholds:
    pos_accept = 0
    pos_reject = 0
    cer_list = []

    for sim, cer, was_accepted in pos_sim_cer:
        if sim >= thr:
            pos_accept += 1
            # 如果之前被拒识(没有ASR结果), 假设 CER 与已接受样本相同
            if was_accepted:
                cer_list.append(cer)
            else:
                # 之前被拒识, 现在接受 - 假设 CER = 已接受样本平均 CER
                cer_list.append(np.mean(stripped_cers))
        else:
            pos_reject += 1
            cer_list.append(1.0)

    avg_cer = np.mean(cer_list)
    rr_est = neg_rr_estimate.get(thr, 0.95)
    score = (1 - avg_cer) * 0.4 + rr_est * 0.4
    marker = ""
    if score > best_score:
        best_score = score
        best_thr = thr
        marker = "  ← 最优"
    print(f"  thr={thr:.2f}: pos接受={pos_accept/len(pos_results)*100:.1f}%, "
          f"est_neg_RR={rr_est*100:.1f}%, est_CER={avg_cer:.4f}, est_score={score:.4f}{marker}")

print(f"\n最优阈值: {best_thr:.2f}, 预估得分: {best_score:.4f}")

# ========== 前10个样本对比 ==========
print(f"\n{'='*60}")
print(f"前10个已接受样本标点对比")
print(f"{'='*60}")
for i, r in enumerate(accepted_samples[:10]):
    label = r.get("label", "")
    content = r.get("content", "")
    orig = char_error_rate(label, content)
    stripped = char_error_rate(strip_punctuation(label), strip_punctuation(content))
    print(f"  [{i}] orig={orig:.4f}, strip={stripped:.4f}, diff={orig-stripped:.4f}")
    print(f"      label:    '{label}'")
    print(f"      output:   '{content}'")
    print(f"      strip_l:  '{strip_punctuation(label)}'")
    print(f"      strip_o:  '{strip_punctuation(content)}'")
