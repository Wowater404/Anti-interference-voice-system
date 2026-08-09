# -*- coding: utf-8 -*-
"""生成 datasetA 测试结果提交文件 (美的群通知新格式, 官方CER口径)"""
import json, sys
sys.path.insert(0, '.')

SRC = 'results/final_inference_dual_full.json'
OUT = 'results/submission_datasetA_dual.json'

# ===== 官方 CER 代码 (逐字复制, 保证口径一致) =====
import unicodedata
import string
import editdistance

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


def official_per_sample_cer(pred, target):
    """官方口径: 单条 CER (target为空时特殊处理)"""
    norm_pred = normalize_text(pred)
    norm_target = normalize_text(target)
    errors = editdistance.eval(norm_pred, norm_target)
    char_cnt = len(norm_target)
    if char_cnt == 0:
        return 0.0 if errors == 0 else 1.0
    return errors / char_cnt


# ===== 加载数据 =====
d = json.load(open(SRC, encoding='utf-8'))
res = d['result']['results']
assert len(res) == 1838, f"条数错误: {len(res)}"

pos = res[:1364]
neg = res[1364:]

# ===== 每条 cer 用官方口径重算 (从 content/label) =====
results_out = []
for r in res:
    content = r.get("content") or ""
    label = r.get("label") or ""
    cer = official_per_sample_cer(content, label)
    results_out.append({
        "id": str(r["id"]),
        "content": content,
        "label": label,
        "cer": cer,  # 官方完整精度
    })

# ===== avg_cer: pos 部分 micro-average (官方口径) =====
total_err = 0
total_chars = 0
for r in pos:
    np_ = normalize_text(r.get("content") or "")
    nt_ = normalize_text(r.get("label") or "")
    total_err += editdistance.eval(np_, nt_)
    total_chars += len(nt_)
avg_cer = total_err / total_chars if total_chars else 0

# ===== avg_rr: neg 部分拒识率 =====
n_fa = sum(1 for r in neg if r.get("content"))
avg_rr = 1 - n_fa / len(neg)

# ===== duration: 毫秒 =====
dur_s = float(d['result']['duration'])
dur_ms = round(dur_s * 1000)

submission = {
    "result": {
        "results": results_out,
        "avg_cer": avg_cer,
        "avg_rr": avg_rr,
        "duration": dur_ms,
    }
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(submission, f, ensure_ascii=False, indent=2)

print("=== 提交文件已生成 (官方CER口径) ===")
print(f"文件: {OUT}")
print(f"总条数: {len(results_out)} (pos {len(pos)} + neg {len(neg)})")
print(f"avg_cer (pos): {avg_cer:.6f} (总错误 {total_err}, 总字符 {total_chars})")
print(f"avg_rr (neg):  {avg_rr:.6f}")
print(f"duration: {dur_ms} ms ({dur_s:.1f}s)")
print()
print("=== 结果样例 ===")
print("pos:", json.dumps(results_out[0], ensure_ascii=False))
print("pos含大小写:", json.dumps(results_out[32], ensure_ascii=False))
print("neg:", json.dumps(results_out[1364], ensure_ascii=False))
