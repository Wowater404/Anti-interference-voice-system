# -*- coding: utf-8 -*-
"""
增量增强补跑: 只生成新增的增强类型, 追加到已完成的增强数据集

用途: datasetA_aug 已用 augment_dataset.py 生成 16 种增强,
      之后 AUG_TYPES 新增了 pitch_m2f / pitch_f2m 两种,
      本脚本只补这两种, 不重跑全部 (省时间)。

用法:
  python tools/augment_dataset_incremental.py \
      --src "C:/Users/善水/Desktop/datasetA用于训练" \
      --dst "F:/龙虾/2026-07-18-13-57-00/datasetA_aug" \
      --workers 8
"""
import os
import sys
import json
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from augment_dataset import (
    SR, AUG_TYPES, process_one_sample,
)

# 只处理已存在文件里没有的新增强类型
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="原始 datasetA 根目录")
    ap.add_argument("--dst", required=True, help="增强输出根目录 (已含16种)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    # 读取已有 jsonl, 确定已有 aug_type
    new_types = []
    for split in ["pos", "neg"]:
        jp = os.path.join(args.dst, f"{split}_aug.jsonl")
        if not os.path.exists(jp):
            print(f"⚠️ 未找到 {jp}, 跳过 {split}")
            continue
        with open(jp, encoding="utf-8") as f:
            records = [json.loads(l) for l in f if l.strip()]
        existing = {r["aug_type"] for r in records}
        for name, _, _ in AUG_TYPES:
            if name not in existing and name not in new_types:
                new_types.append(name)

    if not new_types:
        print("✅ 无需补跑: 所有增强类型已存在")
        return
    print(f"需补跑的增强类型: {new_types}")

    # 加载原始 jsonl
    split_records = {}
    for split in ["pos", "neg"]:
        with open(os.path.join(args.src, f"{split}.jsonl"), encoding="utf-8") as f:
            split_records[split] = [json.loads(l) for l in f if l.strip()]

    # 构建干扰池 (新类型里若有 overlap 需要; 这里主要是 pitch, 用不到但保持接口一致)
    tasks = []
    for split, recs in split_records.items():
        pool = [(r["识别音频"], r["id"]) for r in recs]
        for rec in recs:
            tasks.append((args.src, args.dst, split, rec, pool, new_types))

    # 逐样本生成新类型, 追加记录
    all_new = {"pos": [], "neg": []}
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(process_one_sample, t) for t in tasks]
        for fut in as_completed(futures):
            split, records = fut.result()
            all_new[split].extend(records)  # process_one_sample 已按 only_types 过滤
            done += 1
            if done % 200 == 0 or done == len(tasks):
                print(f"  进度: {done}/{len(tasks)}", flush=True)

    for split in ["pos", "neg"]:
        jp = os.path.join(args.dst, f"{split}_aug.jsonl")
        with open(jp, encoding="utf-8") as f:
            existing = [json.loads(l) for l in f if l.strip()]
        existing.extend(all_new[split])
        existing.sort(key=lambda r: r["id"])
        with open(jp, "w", encoding="utf-8") as f:
            for r in existing:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{split}: {len(all_new[split])} 新条追加 → 共 {len(existing)} 条")

    print("增量补跑完成!")


if __name__ == "__main__":
    main()
