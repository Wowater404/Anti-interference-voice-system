"""
评估指标计算
CER (字符错误率) 和 RR (拒识率)

V3 更新 (2026-07-19): 对齐官方CER计算方式
  - normalize_text: NFKC + lowercase + 全Unicode P*过滤 (与官方脚本一致)
  - CER聚合: micro-average = total_errors / total_chars (非macro-average)
  - editdistance库: 与官方使用相同的Levenshtein距离实现
"""
import unicodedata
import string
import editdistance
import numpy as np
from typing import List, Dict, Optional


# ===========================================================================
# 文本归一化 (对齐官方CER脚本的 normalize_text)
# ===========================================================================
def normalize_text(text: str) -> str:
    """
    ASR 文本归一化 (与官方CER评估脚本完全一致):

    1. Unicode 规范化 (NFKC) —— 全半角转换
    2. 转小写
    3. 去掉前后空白
    4. 移除所有标点 (Unicode 类别 P*) 和空白字符 (包括内部空格)
    """
    if text is None:
        return ""
    text = str(text)

    # 1. NFKC 规范化 (全角字母数字转半角等)
    text = unicodedata.normalize("NFKC", text)

    # 2. 转小写
    text = text.lower()

    # 3. 去掉前后空白
    text = text.strip()

    # 4. 过滤: 去掉所有标点和空白 (只保留字母、数字、汉字等非标点非空白字符)
    normalized_chars = []
    for ch in text:
        if ch in string.whitespace or unicodedata.category(ch).startswith("P"):
            continue
        normalized_chars.append(ch)

    return "".join(normalized_chars)


# ===========================================================================
# 兼容接口: 保留旧函数名, 内部调用 normalize_text
# ===========================================================================
def strip_punctuation(text: str) -> str:
    """
    去除标点符号 (兼容旧接口)
    已升级为官方 normalize_text, 额外做NFKC+小写处理
    """
    return normalize_text(text)


# ===========================================================================
# CER 计算 (对齐官方: editdistance + micro-average)
# ===========================================================================
def char_error_rate(reference: str, hypothesis: str) -> float:
    """
    计算单条样本的字符错误率 (CER)
    CER = (S + D + I) / N = editdistance / len(reference)

    使用官方 normalize_text 归一化后, 用 editdistance 库计算 Levenshtein 距离

    Args:
        reference: 参考文本 (ground truth)
        hypothesis: 识别文本 (预测结果)

    Returns:
        CER 值, 范围 [0, +inf), 越低越好
    """
    norm_ref = normalize_text(reference)
    norm_hyp = normalize_text(hypothesis)

    char_cnt = len(norm_ref)

    if char_cnt == 0:
        return 0.0 if len(norm_hyp) == 0 else 1.0

    errors = editdistance.eval(norm_hyp, norm_ref)
    return errors / char_cnt


def compute_micro_cer(results: List[Dict]) -> float:
    """
    微平均 CER (与官方评估方式一致):
      total_errors / total_chars

    将所有 pos 样本的编辑距离和参考文本长度累加, 最后一次性计算
    这与逐条计算CER再取平均 (macro-average) 不同

    FAQ#8: 正样本被错误拒识时, 按删除错误计算
      → content="" (空字符串), editdistance("", target) = len(target) (全删除)

    Args:
        results: pos 样本结果列表, 每项含 content 和 label

    Returns:
        micro-average CER
    """
    total_errors = 0
    total_chars = 0

    for r in results:
        pred = r.get("content", "")
        target = r.get("label", "")

        norm_pred = normalize_text(pred)
        norm_target = normalize_text(target)
        char_cnt = len(norm_target)

        if char_cnt == 0:
            # target为空 (neg样本), 不计入CER
            continue

        errors = editdistance.eval(norm_pred, norm_target)
        total_errors += errors
        total_chars += char_cnt

    if total_chars == 0:
        return 0.0 if total_errors == 0 else 1.0

    return total_errors / total_chars


# ===========================================================================
# RR 计算
# ===========================================================================
def rejection_rate(
    neg_results: List[Dict],
    reject_label: str = ""
) -> float:
    """
    计算拒识率 (RR)
    RR = 正确拒识的非目标语音数 / 总非目标语音数

    FAQ#1: RR字段写不写都可以, 评测时根据提交的推理结果统一计算
    FAQ#9: 负样本如果被识别出内容, 只计算RR, 不统计CER

    拒识判定: content 为空字符串 或 "null" (兼容两种格式)

    Args:
        neg_results: 负样本推理结果列表, 每项含 content 字段
        reject_label: 拒识时输出的标签 (默认空字符串)

    Returns:
        RR 值, 范围 [0, 1], 越高越好
    """
    if not neg_results:
        return 0.0
    correct_reject = sum(
        1 for r in neg_results
        if not r.get("content") or r.get("content") in ("", "null", None)
    )
    return correct_reject / len(neg_results)


# ===========================================================================
# 综合评估
# ===========================================================================
def evaluate(
    pos_results: List[Dict],
    neg_results: List[Dict],
    reject_label: str = ""
) -> Dict:
    """
    综合评估: 计算 CER 和 RR (对齐官方评估方式)

    FAQ#7: 拒识类音频只统计拒识率RR, 不统计CER
    FAQ#8: 正样本被错误拒识, 按删除错误计算CER
    FAQ#9: 负样本只计算RR, 不统计CER
    FAQ#4: CER 40% + RR 40% + 推理效率 20% (推理时间10% + 内存10%)

    CER计算方式: micro-average (total_errors / total_chars)
    与官方 CERMetric.compute() 一致

    Args:
        pos_results: 正样本结果, 每项含 content, label
        neg_results: 负样本结果, 每项含 content

    Returns:
        dict: {final_cer, rejection_rate, final_score, pos_count, neg_count}
    """
    # CER (仅正样本, micro-average)
    avg_cer = compute_micro_cer(pos_results)

    # 更新每条样本的cer字段 (用于提交JSON)
    for r in pos_results:
        pred = r.get("content", "")
        target = r.get("label", "")
        norm_pred = normalize_text(pred)
        norm_target = normalize_text(target)
        char_cnt = len(norm_target)
        if char_cnt == 0:
            r["cer"] = "0.0000"
        else:
            errors = editdistance.eval(norm_pred, norm_target)
            r["cer"] = f"{errors / char_cnt:.4f}"

    # RR (仅负样本)
    rr = rejection_rate(neg_results, reject_label)

    # 综合得分: CER 权重 40%, RR 权重 40% (效率 20% 需额外测量)
    final_score = (1 - avg_cer) * 0.4 + rr * 0.4

    return {
        "final_cer": f"{avg_cer:.4f}",
        "rejection_rate": f"{rr:.4f}",
        "final_score": f"{final_score:.4f}",
        "pos_count": len(pos_results),
        "neg_count": len(neg_results),
    }
