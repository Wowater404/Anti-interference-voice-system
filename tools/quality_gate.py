# -*- coding: utf-8 -*-
"""
质量门控模块 (Quality Gate) — 训练前脏样本检测与过滤

背景:
  训练脚本对所有样本统一执行 降噪+自适应分离。但"处理过"不等于"数据干净":
    - 语义脏: neg 实际是同一个人 (标注错误) / pos 分离选错轨 → 处理救不了
    - 失真脏: 分离伪影导致 sim 跳变异常 → 处理反而制造脏
  本模块在训练前扫描预处理结果, 用可量化的指标检测这些脏样本并过滤。

检测项 (每项可独立开关, 阈值可配置):
  G1 neg_high_sim   : neg 样本 kws-cmd 相似度过高 → 疑似"同人误标"或"锚点错误"
                      规则: label=0 且 sim_abs > threshold (默认0.55)
  G2 pos_low_sim    : pos 样本分离后仍低 sim → 目标声纹提取失败 → 训练数据是脏的
                      规则: label=1 且 sep_best_sim < threshold (默认0.30)
  G3 jump_artifact  : 分离后 sim 跳变超过上限 → 疑似分离伪影 (与推理侧 sim_jump_cap 对齐)
                      规则: sep_triggered 且 (sep_best_sim - sim_abs) > jump_cap (默认0.30)
  G4 sep_no_improve : 触发了分离但分离后无提升 → 分离白做, 样本质量存疑 (仅报告不过滤)

输入:
  datasetA_aug_processed/pos_aug_processed.jsonl, neg_aug_processed.jsonl
  (需包含预处理时记录的字段: sim_abs, sep_best_sim, sep_triggered)

输出:
  <out>/quality_report.json   — 过滤统计 + 每类明细
  <out>/filtered_*.jsonl      — 过滤后的训练 jsonl (可直接喂 make_folds)
  <out>/quality_sim_dist.csv  — 全量 sim 分布 (供人工抽查定阈值)

用法:
  python tools/quality_gate.py \
      --processed_root "F:/龙虾/2026-07-18-13-57-00/datasetA_aug_processed" \
      --out_root "F:/龙虾/2026-07-18-13-57-00/datasetA_aug_processed/gated" \
      --neg_high_sim 0.55 --pos_low_sim 0.30 --jump_cap 0.30
  # 只出报告不过滤: --no_filter
"""
import os
import json
import csv
import argparse
from collections import defaultdict


def load_jsonl(path):
    """加载jsonl文件. Returns: list[dict]"""
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, records):
    """写出jsonl文件"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def get_sim_abs(r):
    """从样本记录提取分离前绝对相似度 sim_abs (缺失时返回 None)"""
    v = r.get("sim_abs", r.get("sim_abs_weighted", None))
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def get_sep_best_sim(r):
    """提取分离后最佳相似度 sep_best_sim (缺失时返回 None)"""
    v = r.get("sep_best_sim", None)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def get_sep_triggered(r):
    """是否触发了分离"""
    v = r.get("sep_triggered", r.get("cmd_separated", False))
    if isinstance(v, str):
        return v.lower() in ("true", "1", "yes")
    return bool(v)


def is_pos(r):
    """识别文本非空 = pos (同人)"""
    return r.get("识别文本") is not None


def check_sample(r, cfg):
    """对单条样本跑全部检测项. Returns: list[str] 命中的规则名"""
    hits = []
    label = "pos" if is_pos(r) else "neg"
    sim_abs = get_sim_abs(r)
    sep_best = get_sep_best_sim(r)
    sep_trig = get_sep_triggered(r)

    # G1: neg 高 sim (同人误标 / 锚点错误)
    if cfg["neg_high_sim"] is not None and label == "neg" and sim_abs is not None:
        if sim_abs > cfg["neg_high_sim"]:
            hits.append("G1_neg_high_sim")

    # G2: pos 分离后仍低 sim (目标提取失败)
    if cfg["pos_low_sim"] is not None and label == "pos" and sep_best is not None:
        if sep_best < cfg["pos_low_sim"]:
            hits.append("G2_pos_low_sim")

    # G3: 分离伪影 (跳变超限)
    if cfg["jump_cap"] is not None and sep_trig and sim_abs is not None and sep_best is not None:
        jump = sep_best - sim_abs
        if jump > cfg["jump_cap"]:
            hits.append("G3_jump_artifact")

    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_root", required=True,
                    help="预处理输出目录 (含 pos_aug_processed.jsonl / neg_aug_processed.jsonl)")
    ap.add_argument("--out_root", required=True, help="质量门控输出目录")
    ap.add_argument("--neg_high_sim", type=float, default=0.55,
                    help="G1阈值: neg 样本 sim_abs 高于此值视为可疑 (None=关闭)")
    ap.add_argument("--pos_low_sim", type=float, default=0.30,
                    help="G2阈值: pos 样本分离后 sim 低于此值视为提取失败 (None=关闭)")
    ap.add_argument("--jump_cap", type=float, default=0.30,
                    help="G3阈值: 分离后 sim 跳变超过此值视为伪影 (None=关闭)")
    ap.add_argument("--no_filter", action="store_true",
                    help="只出报告, 不过滤 (用于先看分布定阈值)")
    args = ap.parse_args()

    cfg = {
        "neg_high_sim": args.neg_high_sim,
        "pos_low_sim": args.pos_low_sim,
        "jump_cap": args.jump_cap,
    }
    # 路径归一化: 兼容 Windows 反斜杠 / Unix 正斜杠混用
    processed_root = os.path.normpath(args.processed_root)
    out_root = os.path.normpath(args.out_root)
    os.makedirs(out_root, exist_ok=True)

    # 加载 pos / neg 预处理结果
    all_records = []
    for split in ["pos", "neg"]:
        p = os.path.join(processed_root, f"{split}_aug_processed.jsonl")
        if not os.path.exists(p):
            print(f"⚠️ 未找到 {p}, 跳过 {split}")
            continue
        recs = load_jsonl(p)
        for r in recs:
            r["_split"] = split
        all_records.extend(recs)
        print(f"[{split}] 加载 {len(recs)} 条")

    if not all_records:
        print("错误: 没有加载到任何样本, 请先运行 prepare_processed_train_data.py")
        return

    # 全量检测
    stats = defaultdict(int)          # 规则名 → 命中数
    rule_detail = defaultdict(list)   # 规则名 → [样本id]
    has_gate_fields = 0
    missing_gate_fields = 0
    for r in all_records:
        sim_abs = get_sim_abs(r)
        sep_best = get_sep_best_sim(r)
        if sim_abs is not None or sep_best is not None:
            has_gate_fields += 1
        else:
            missing_gate_fields += 1
        hits = check_sample(r, cfg)
        for h in hits:
            stats[h] += 1
            rule_detail[h].append({
                "id": r.get("id"), "split": r.get("_split"),
                "sim_abs": sim_abs, "sep_best_sim": sep_best,
            })

    # 输出 sim 分布 CSV (供人工定阈值)
    csv_path = os.path.join(out_root, "quality_sim_dist.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "split", "label", "sim_abs", "sep_best_sim", "sep_triggered", "hits"])
        for r in all_records:
            sim_abs = get_sim_abs(r)
            sep_best = get_sep_best_sim(r)
            hits = "+".join(check_sample(r, cfg)) or "-"
            w.writerow([r.get("id"), r.get("_split"),
                        "pos" if is_pos(r) else "neg",
                        "" if sim_abs is None else f"{sim_abs:.4f}",
                        "" if sep_best is None else f"{sep_best:.4f}",
                        get_sep_triggered(r), hits])
    print(f"\nsim 分布 CSV → {csv_path}")

    # 过滤
    if args.no_filter:
        print("\n[--no_filter] 只出报告, 未过滤")
        filtered = all_records
    else:
        filtered = []
        for r in all_records:
            hits = check_sample(r, cfg)
            if not hits:
                filtered.append(r)
        print(f"\n过滤: {len(all_records)} → {len(filtered)} (剔除 {len(all_records) - len(filtered)})")

        # 分 pos/neg 写出过滤后的 jsonl
        for split in ["pos", "neg"]:
            out = [r for r in filtered if r["_split"] == split]
            out_path = os.path.join(out_root, f"{split}_aug_gated.jsonl")
            write_jsonl(out_path, out)
            print(f"  → {out_path} ({len(out)} 条)")

    # 报告
    report = {
        "config": cfg,
        "total": len(all_records),
        "has_gate_fields": has_gate_fields,
        "missing_gate_fields": missing_gate_fields,
        "rule_hits": dict(stats),
        "detail": {k: v[:50] for k, v in rule_detail.items()},  # 每类最多50条明细
    }
    if not args.no_filter:
        report["filtered_total"] = len(filtered)
        report["removed"] = len(all_records) - len(filtered)
    report_path = os.path.join(out_root, "quality_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告 → {report_path}")

    print("\n===== 质量门控统计 =====")
    print(f"总样本: {report['total']} (含门控字段: {has_gate_fields}, 缺字段: {missing_gate_fields})")
    if missing_gate_fields > 0:
        print("⚠️ 有样本缺少 sim_abs/sep_best_sim 字段, 请确认预处理脚本已记录这些字段")
    if not stats:
        print("✅ 无命中规则 (数据看起来干净, 或字段缺失导致检测失效)")
    else:
        for rule, n in sorted(stats.items(), key=lambda x: -x[1]):
            print(f"  {rule}: {n} 条")


if __name__ == "__main__":
    main()
