# -*- coding: utf-8 -*-
"""用官方 CER 代码验证提交文件"""
import unicodedata
import string
import editdistance
import json


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = text.strip()
    normalized_chars = []
    for ch in text:
        if ch in string.whitespace or unicodedata.category(ch).startswith("P"):
            continue
        normalized_chars.append(ch)
    return "".join(normalized_chars)


class CERMetric:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_chars = 0
        self.total_errors = 0
        self.per_sample_results = []

    def update(self, preds, targets):
        if isinstance(preds, str):
            preds = [preds]
        if isinstance(targets, str):
            targets = [targets]
        assert len(preds) == len(targets)
        for pred, target in zip(preds, targets):
            orig_pred = pred
            orig_target = target
            norm_pred = normalize_text(pred)
            norm_target = normalize_text(target)
            errors = editdistance.eval(norm_pred, norm_target)
            char_cnt = len(norm_target)
            if char_cnt == 0:
                cer_value = 0.0 if errors == 0 else 1.0
            else:
                cer_value = errors / char_cnt
            self.total_errors += errors
            self.total_chars += char_cnt
            self.per_sample_results.append({
                "orig_pred": orig_pred, "orig_target": orig_target,
                "norm_pred": norm_pred, "norm_target": norm_target,
                "errors": errors, "target_chars": char_cnt, "cer": cer_value,
            })

    def compute(self):
        if self.total_chars == 0:
            overall_cer = 0.0 if self.total_errors == 0 else 1.0
        else:
            overall_cer = self.total_errors / self.total_chars
        return {"cer": overall_cer, "total_errors": self.total_errors,
                "total_chars": self.total_chars, "per_sample": self.per_sample_results}


# 加载提交文件
d = json.load(open('results/submission_datasetA_dual_full.json', encoding='utf-8'))
res = d['result']['results']
pos = res[:1364]
neg = res[1364:]

print("=== 用官方 CER 代码验证提交文件 ===")
print()

# 1. 验证 pos 部分 avg_cer
metric = CERMetric()
for r in pos:
    metric.update(r['content'], r['label'])
official_cer = metric.compute()
print(f"[pos部分] 官方代码 avg_cer: {official_cer['cer']:.6f}")
print(f"          提交文件 avg_cer: {d['result']['avg_cer']}")
print(f"          总错误: {official_cer['total_errors']}, 总字符: {official_cer['total_chars']}")

# 2. 验证每条 cer 与提交文件一致
mismatch = 0
for i, (r, sr) in enumerate(zip(pos, metric.per_sample_results)):
    sub_cer = r['cer']
    off_cer = sr['cer']
    if abs(sub_cer - off_cer) > 1e-6:
        mismatch += 1
        if mismatch <= 3:
            print(f"  ⚠️ pos[{i}]: 提交={sub_cer} 官方={off_cer:.4f} pred={r['content'][:15]!r} label={r['label'][:15]!r}")
print(f"pos 每条 cer 不一致数: {mismatch}/1364")

# 3. 验证 RR
n_fa = sum(1 for r in neg if r['content'])
rr = 1 - n_fa / len(neg)
print(f"\n[neg部分] 官方计算 RR: {rr:.6f} (提交: {d['result']['avg_rr']})")
print(f"          假接受: {n_fa}/474")

# 4. 大小写问题检查: 找 label 或 content 含大写字母/全角字符的样本
print("\n=== 大小写/全角检查 (NFKC会处理的case) ===")
found = 0
for i, r in enumerate(pos):
    import re
    if re.search(r'[A-Z]', r['label']) or re.search(r'[Ａ-Ｚａ-ｚ０-９]', r['label']):
        found += 1
        if found <= 5:
            print(f"  pos[{i}]: label={r['label']!r} content={r['content']!r}")
print(f"含大写/全角字符的 pos label: {found} 条")

print("\n=== 结论 ===")
print(f"官方 avg_cer 匹配: {'✅' if abs(official_cer['cer'] - d['result']['avg_cer']) < 1e-6 else '❌'}")
print(f"官方 RR 匹配: {'✅' if abs(rr - d['result']['avg_rr']) < 1e-6 else '❌'}")
