# -*- coding: utf-8 -*-
"""
五折交叉验证划分 (会议决议: 4/5训练 + 1/5测试)

关键原则:
  1. 按 orig_id 划分 —— 同一样本的所有增强版本必须在同一折, 防止数据泄漏
  2. pos/neg 分层抽样 —— 每折保持 pos:neg 比例
  3. train.jsonl 含全部增强版本; val.jsonl 仅含原始版本 (aug_type=orig), 保证验证干净

输出:
  datasetA_aug/folds/fold_{0..4}/train.jsonl, val.jsonl
  datasetA_aug/folds/folds_summary.json

用法:
  python tools/make_folds.py --aug_root "F:/龙虾/2026-07-18-13-57-00/datasetA_aug" --n_folds 5
"""
import os
import json
import argparse
import random
from collections import defaultdict


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug_root", required=True, help="增强数据集根目录 (含 pos_aug.jsonl/neg_aug.jsonl)")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pos = load_jsonl(os.path.join(args.aug_root, "pos_aug.jsonl"))
    neg = load_jsonl(os.path.join(args.aug_root, "neg_aug.jsonl"))

    # 按 orig_id 分组
    def group_by_orig(records):
        groups = defaultdict(list)
        for r in records:
            groups[r["orig_id"]].append(r)
        return groups

    pos_groups = group_by_orig(pos)
    neg_groups = group_by_orig(neg)
    pos_ids = sorted(pos_groups.keys())
    neg_ids = sorted(neg_groups.keys())
    rng.shuffle(pos_ids)
    rng.shuffle(neg_ids)

    def split_folds(ids, n):
        """把 id 列表均匀分成 n 份"""
        folds = [[] for _ in range(n)]
        for i, x in enumerate(ids):
            folds[i % n].append(x)
        return folds

    pos_fold_ids = split_folds(pos_ids, args.n_folds)
    neg_fold_ids = split_folds(neg_ids, args.n_folds)

    summary = {"n_folds": args.n_folds, "seed": args.seed, "folds": []}
    for k in range(args.n_folds):
        val_pos_ids = set(pos_fold_ids[k])
        val_neg_ids = set(neg_fold_ids[k])

        train_recs, val_recs = [], []
        for oid, recs in pos_groups.items():
            if oid in val_pos_ids:
                # 验证集只放原始版本
                val_recs.extend([r for r in recs if r["aug_type"] == "orig"])
            else:
                train_recs.extend(recs)
        for oid, recs in neg_groups.items():
            if oid in val_neg_ids:
                val_recs.extend([r for r in recs if r["aug_type"] == "orig"])
            else:
                train_recs.extend(recs)

        rng.shuffle(train_recs)
        rng.shuffle(val_recs)

        fold_dir = os.path.join(args.aug_root, "folds", f"fold_{k}")
        write_jsonl(os.path.join(fold_dir, "train.jsonl"), train_recs)
        write_jsonl(os.path.join(fold_dir, "val.jsonl"), val_recs)

        n_train_orig = len({r["orig_id"] for r in train_recs})
        n_val_orig = len({r["orig_id"] for r in val_recs})
        info = {
            "fold": k,
            "train_aug_samples": len(train_recs),
            "train_orig_samples": n_train_orig,
            "val_samples": len(val_recs),
            "val_pos": sum(1 for r in val_recs if r["识别文本"] is not None),
            "val_neg": sum(1 for r in val_recs if r["识别文本"] is None),
        }
        summary["folds"].append(info)
        print(f"fold_{k}: train {info['train_orig_samples']}原始→{len(train_recs)}增强条 | "
              f"val {len(val_recs)}条 (pos={info['val_pos']}, neg={info['val_neg']})")

    write_jsonl(os.path.join(args.aug_root, "folds", "folds_summary.json"), [summary])
    print("五折划分完成!")


if __name__ == "__main__":
    main()
