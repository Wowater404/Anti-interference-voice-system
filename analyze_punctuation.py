"""
分析标点符号对CER的影响
从已有推理结果中重新计算去除标点后的CER
"""
import json
import re
import numpy as np

def strip_punctuation(text):
    """去除中文和英文标点符号"""
    # 中文标点: 。 ， 、 ； ： ？ ！ " " ' ' （ ） 【 】 《 》 … — ·
    # 英文标点: . , ; : ? ! " ' ( ) [ ] < > ...
    # 保留数字、字母、汉字
    pattern = r'[。，、；：？！""''）（】【《》…—·.,;:?!"\'()<>]'
    return re.sub(pattern, '', text)

def char_error_rate(reference, hypothesis):
    """计算CER"""
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
                dp[i][j] = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + 1
                )
    return dp[n][m] / n

# 加载推理结果
checkpoint_path = "F:/龙虾/2026-07-18-13-57-00/voice_pipeline/results/checkpoint.json"
with open(checkpoint_path, "r", encoding="utf-8") as f:
    ckpt = json.load(f)

results = ckpt["results"]

# 分离 pos / neg
pos_results = [r for r in results if r.get("label") and r["label"] != "null"]
neg_results = [r for r in results if not r.get("label") or r["label"] == "null"]

print(f"总样本数: {len(results)}")
print(f"pos: {len(pos_results)}, neg: {len(neg_results)}")

# 分析 pos 样本
original_cers = []
stripped_cers = []
accepted_samples = []
rejected_samples = []

for r in pos_results:
    label = r.get("label", "")
    content = r.get("content", "")
    similarity = float(r.get("similarity", "0"))
    is_target = r.get("is_target", False)

    if is_target and content != "null" and content:
        # 已接受的样本
        accepted_samples.append(r)
        # 原始CER
        orig_cer = char_error_rate(label, content)
        original_cers.append(orig_cer)
        # 去标点CER
        label_clean = strip_punctuation(label)
        content_clean = strip_punctuation(content)
        stripped_cer = char_error_rate(label_clean, content_clean)
        stripped_cers.append(stripped_cer)
    else:
        rejected_samples.append(r)

print(f"\n=== pos 样本分析 ===")
print(f"已接受: {len(accepted_samples)} ({len(accepted_samples)/len(pos_results)*100:.1f}%)")
print(f"已拒识: {len(rejected_samples)} ({len(rejected_samples)/len(pos_results)*100:.1f}%)")

print(f"\n=== CER 对比 (仅已接受样本) ===")
print(f"原始 CER (含标点): {np.mean(original_cers):.4f}")
print(f"去标点 CER:        {np.mean(stripped_cers):.4f}")
print(f"CER 降幅:          {np.mean(original_cers) - np.mean(stripped_cers):.4f} ({(np.mean(original_cers) - np.mean(stripped_cers))/np.mean(original_cers)*100:.1f}%)")

# 全部pos样本的CER (拒识=1.0)
all_original_cers = []
all_stripped_cers = []
for r in pos_results:
    label = r.get("label", "")
    content = r.get("content", "")
    is_target = r.get("is_target", False)
    if is_target and content != "null" and content:
        all_original_cers.append(char_error_rate(label, content))
        all_stripped_cers.append(char_error_rate(strip_punctuation(label), strip_punctuation(content)))
    else:
        all_original_cers.append(1.0)
        all_stripped_cers.append(1.0)

print(f"\n=== 全部 pos 样本 CER (拒识=1.0) ===")
print(f"原始 CER (含标点): {np.mean(all_original_cers):.4f}")
print(f"去标点 CER:        {np.mean(all_stripped_cers):.4f}")

# neg 分析
neg_correct = sum(1 for r in neg_results if r.get("content") == "null")
neg_wrong = len(neg_results) - neg_correct
print(f"\n=== neg 样本分析 ===")
print(f"正确拒识: {neg_correct} ({neg_correct/len(neg_results)*100:.1f}%)")
print(f"错误接受: {neg_wrong} ({neg_wrong/len(neg_results)*100:.1f}%)")

# 综合得分估算
avg_cer_stripped = np.mean(all_stripped_cers)
rr = neg_correct / len(neg_results)
score_stripped = (1 - avg_cer_stripped) * 0.4 + rr * 0.4

avg_cer_original = np.mean(all_original_cers)
score_original = (1 - avg_cer_original) * 0.4 + rr * 0.4

print(f"\n=== 综合得分对比 ===")
print(f"原始:    CER={avg_cer_original:.4f}, RR={rr:.4f}, Score={score_original:.4f}")
print(f"去标点:  CER={avg_cer_stripped:.4f}, RR={rr:.4f}, Score={score_stripped:.4f}")

# 展示前10个样本对比
print(f"\n=== 前10个已接受样本对比 ===")
for i, r in enumerate(accepted_samples[:10]):
    label = r.get("label", "")
    content = r.get("content", "")
    orig = char_error_rate(label, content)
    stripped = char_error_rate(strip_punctuation(label), strip_punctuation(content))
    print(f"  [{i}] orig_cer={orig:.4f}, strip_cer={stripped:.4f}")
    print(f"      label:  {label}")
    print(f"      output: {content}")
    print(f"      label_clean:  {strip_punctuation(label)}")
    print(f"      output_clean: {strip_punctuation(content)}")

# 阈值优化分析 (使用去标点CER)
print(f"\n=== 阈值优化分析 (去标点CER) ===")
thresholds = [0.28, 0.29, 0.30, 0.31, 0.32, 0.33]
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
                cer_list.append(char_error_rate(strip_punctuation(label), strip_punctuation(content)))
            else:
                cer_list.append(1.0)  # 接受但没有文本
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
    rr_val = neg_reject / len(neg_results) if neg_results else 0
    score = (1 - avg_cer) * 0.4 + rr_val * 0.4
    print(f"  thr={thr:.2f}: pos接受={pos_accept/len(pos_results)*100:.1f}%, "
          f"neg拒识={rr_val*100:.1f}%, CER={avg_cer:.4f}, 得分={score:.4f}"
          f"{'  ← 最优' if thr == 0.30 else ''}")
