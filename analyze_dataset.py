"""
数据集分析脚本
分析 datasetA 的统计信息, 验证数据格式和音频属性
"""
import os
import sys
import json
import wave
import numpy as np
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from utils.audio import load_wav, get_duration


def analyze_jsonl(jsonl_path: str, data_root: str, split_name: str):
    """分析 JSONL 文件"""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    print(f"\n{'=' * 60}")
    print(f"数据集: {split_name} ({jsonl_path})")
    print(f"{'=' * 60}")
    print(f"样本总数: {len(samples)}")
    print(f"字段: {list(samples[0].keys())}")

    # 唤醒文本分析
    kws_texts = [s["唤醒文本"] for s in samples]
    unique_kws = sorted(set(kws_texts))
    print(f"\n唤醒文本种类: {len(unique_kws)}")
    for kws in unique_kws:
        count = kws_texts.count(kws)
        print(f"  '{kws}': {count} 次")

    # 识别文本分析
    labels = [s["识别文本"] for s in samples]
    null_count = sum(1 for l in labels if l is None or l == "null")
    valid_labels = [l for l in labels if l and l != "null"]
    print(f"\n识别文本:")
    print(f"  有效标签: {len(valid_labels)} 条")
    print(f"  空标签(null): {null_count} 条")

    if valid_labels:
        label_lengths = [len(l) for l in valid_labels]
        print(f"  标签长度: min={min(label_lengths)}, max={max(label_lengths)}, avg={np.mean(label_lengths):.1f}")
        print(f"  样例:")
        for i in range(min(5, len(valid_labels))):
            print(f"    {valid_labels[i]}")

    # ID 分析
    ids = [s["id"] for s in samples]
    unique_ids = set(ids)
    print(f"\nID 分析:")
    print(f"  总样本: {len(ids)}, 唯一ID: {len(unique_ids)}")
    if unique_ids:
        print(f"  ID范围: {min(unique_ids)} - {max(unique_ids)}")

    # 音频文件分析 (采样)
    print(f"\n音频文件分析 (采样前20条):")
    kws_durations = []
    cmd_durations = []
    for s in samples[:20]:
        kws_path = os.path.join(data_root, s["唤醒音频"])
        cmd_path = os.path.join(data_root, s["识别音频"])

        if os.path.exists(kws_path):
            kws_durations.append(get_duration(kws_path))
        if os.path.exists(cmd_path):
            cmd_durations.append(get_duration(cmd_path))

    if kws_durations:
        print(f"  唤醒音频时长: {np.mean(kws_durations):.2f}s (全部 {kws_durations[0]:.2f}s)")
    if cmd_durations:
        print(f"  识别音频时长: min={min(cmd_durations):.2f}s, max={max(cmd_durations):.2f}s, avg={np.mean(cmd_durations):.2f}s")

    # 检查音频格式
    if samples:
        kws_path = os.path.join(data_root, samples[0]["唤醒音频"])
        if os.path.exists(kws_path):
            with wave.open(kws_path, "rb") as wf:
                print(f"\n  音频格式:")
                print(f"    采样率: {wf.getframerate()}Hz")
                print(f"    声道数: {wf.getnchannels()}")
                print(f"    位深度: {wf.getsampwidth() * 8}bit")

    return samples


def main():
    data_root = "F:/挑杯资料/datasetA"

    if len(sys.argv) > 1:
        data_root = sys.argv[1]

    print(f"数据集根目录: {data_root}")

    # 分析 pos
    pos_path = os.path.join(data_root, "pos.jsonl")
    if os.path.exists(pos_path):
        pos_samples = analyze_jsonl(pos_path, data_root, "pos (正样本)")

    # 分析 neg
    neg_path = os.path.join(data_root, "neg.jsonl")
    if os.path.exists(neg_path):
        neg_samples = analyze_jsonl(neg_path, data_root, "neg (负样本)")

    # 汇总
    print(f"\n{'=' * 60}")
    print("数据集汇总")
    print(f"{'=' * 60}")
    if pos_samples:
        print(f"pos (正样本): {len(pos_samples)} 条 — 用于测试 CER (字错率)")
    if neg_samples:
        print(f"neg (负样本): {len(neg_samples)} 条 — 用于测试 RR (拒识率)")
    if pos_samples and neg_samples:
        total = len(pos_samples) + len(neg_samples)
        print(f"总计: {total} 条样本")

    # 比赛评分说明
    print(f"\n{'=' * 60}")
    print("比赛评分标准")
    print(f"{'=' * 60}")
    print("  CER (字错率):       权重 40% — 在 pos 上评估")
    print("  RR  (拒识率):       权重 40% — 在 neg 上评估")
    print("  推理效率:           权重 20% — 推理时间 10% + 内存 10%")
    print("  (信噪比鲁棒性:      测试集分布 -5db ~ 5db)")
    print("  (重叠语音识别:      说话人重叠率 0 ~ 100%, 最多2人)")


if __name__ == "__main__":
    main()
