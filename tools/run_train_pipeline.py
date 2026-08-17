# -*- coding: utf-8 -*-
"""
训练流水线主控脚本 — 一键: 原始音频 → 质量门控 → folds → 声纹模型

串联:
  ① augment_dataset.py       原始 datasetA → 增强数据 (每样本8版本)     [可跳过]
  ② prepare_processed_train_data.py  增强数据 → 降噪+自适应分离处理     [必做]
  ③ quality_gate.py          处理后数据 → 脏样本检测+过滤               [核心新增]
  ④ make_folds.py            过滤后数据 → 5折划分 (按orig_id防泄漏)
  ⑤ train_camplus_finetune.py / train_eres2netv2_finetune.py → 训练

设计要点 (与组内讨论对齐):
  - 训练/推理流程一致: ② 与 pipeline.py 的 _process_kws / cmd 处理完全一致
  - 质量门控: 检测"语义脏"(neg高sim误标/pos分离后低sim) 和 "失真脏"(分离伪影)
  - 文件级独立: kws/cmd 各自处理, 质量标签各自记录, 不做 id 级对齐
  - 断点续跑: 每阶段完成后生成标记文件, 重跑自动跳过已完成阶段

用法:
  # 全流程 (含增强; 需指定原始 datasetA 目录)
  python tools/run_train_pipeline.py \
      --src "C:/Users/善水/Desktop/datasetA/datasetA" \
      --work_root "F:/龙虾/2026-07-18-13-57-00/datasetA_aug" \
      --processed_root "F:/龙虾/2026-07-18-13-57-00/datasetA_aug_processed" \
      --train_model camplus --fold 0

  # 只做 ②③④ (已有增强数据, 跳过增强)
  python tools/run_train_pipeline.py \
      --work_root "F:/龙虾/2026-07-18-13-57-00/datasetA_aug" \
      --processed_root "F:/龙虾/2026-07-18-13-57-00/datasetA_aug_processed" \
      --skip_augment --train_model camplus --fold full

  # 先只看质量门控报告 (不过滤, 定阈值用)
  python tools/run_train_pipeline.py \
      --work_root ... --processed_root ... --skip_augment --skip_train --no_gate_filter

参数:
  --src            原始 datasetA 目录 (含 pos.jsonl/neg.jsonl; 只有跑增强时需要)
  --work_root      增强数据根目录 (augment 的输出 / make_folds 的输入)
  --processed_root 处理后数据目录 (prepare 的输出 / quality_gate 的输入)
  --gate_root      质量门控输出目录 (默认 processed_root/gated)
  --skip_augment   跳过增强阶段 (已有增强数据时)
  --skip_prepare   跳过预处理阶段 (已处理过时)
  --skip_gate      跳过质量门控 (不想要过滤时)
  --skip_train     不训练 (只跑数据处理, 看门控报告)
  --no_gate_filter 门控只出报告不过滤 (先定阈值用)
  --train_model    camplus | eres2netv2 | both (默认 both)
  --fold           0-4 或 full (默认 0)
  --epochs --lr --batch --workers  训练参数 (透传给训练脚本)
  --config         pipeline config 路径 (默认 default.yaml)
"""
import os
import sys
import json
import time
import shutil
import argparse
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS_DIR = os.path.join(PROJECT_ROOT, "tools")


def run_cmd(cmd, desc, cwd=None):
    """执行子进程命令, 打印输出"""
    print(f"\n{'=' * 64}\n▶ {desc}\n{'=' * 64}")
    print("$", " ".join(cmd))
    t0 = time.time()
    env = dict(os.environ)
    result = subprocess.run(cmd, cwd=cwd or PROJECT_ROOT, env=env)
    if result.returncode != 0:
        print(f"❌ 阶段失败: {desc} (exit={result.returncode})")
        sys.exit(result.returncode)
    print(f"✅ {desc} 完成, 用时 {time.time()-t0:.0f}s")
    return True


def write_done(marker):
    """写入阶段完成标记"""
    os.makedirs(os.path.dirname(marker), exist_ok=True)
    with open(marker, "w", encoding="utf-8") as f:
        f.write(json.dumps({"done": True, "time": time.time()}))


def is_done(marker, skip=False):
    """检查阶段标记; skip=True 视为强制跳过"""
    if skip:
        print(f"⏭️  [跳过] 阶段标记: {os.path.basename(marker)} (用户指定跳过)")
        return True
    if os.path.exists(marker):
        print(f"⏭️  [跳过] 已完成: {marker}")
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None, help="原始 datasetA 目录 (增强阶段用)")
    ap.add_argument("--work_root", required=True, help="增强数据根目录")
    ap.add_argument("--processed_root", required=True, help="处理后数据目录")
    ap.add_argument("--gate_root", default=None, help="质量门控输出目录 (默认 processed_root/gated)")
    ap.add_argument("--skip_augment", action="store_true")
    ap.add_argument("--skip_prepare", action="store_true")
    ap.add_argument("--skip_gate", action="store_true")
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--no_gate_filter", action="store_true", help="门控只出报告不过滤")
    ap.add_argument("--train_model", default="both", choices=["camplus", "eres2netv2", "both"])
    ap.add_argument("--fold", default="0")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    gate_root = args.gate_root or os.path.join(args.processed_root, "gated")
    py = sys.executable
    cfg_opt = ["--config", args.config] if args.config else []
    t_all = time.time()

    # ============================================================
    # ① 增强 (可选)
    # ============================================================
    marker_aug = os.path.join(args.work_root, ".stage_augment.done")
    if not is_done(marker_aug, skip=args.skip_augment):
        if not args.src:
            print("❌ 需要 --src (原始 datasetA 目录) 才能跑增强, 或用 --skip_augment 跳过")
            sys.exit(1)
        run_cmd(
            [py, os.path.join(TOOLS_DIR, "augment_dataset.py"),
             "--src", args.src, "--dst", args.work_root, "--workers", str(args.workers)],
            "① 数据增强 (每样本8版本)",
        )
        write_done(marker_aug)

    # ============================================================
    # ② 预处理 (kws/cmd 降噪+自适应分离, 与推理对齐)
    # ============================================================
    marker_prep = os.path.join(args.processed_root, ".stage_prepare.done")
    if not is_done(marker_prep, skip=args.skip_prepare):
        run_cmd(
            [py, os.path.join(TOOLS_DIR, "prepare_processed_train_data.py"),
             "--data_root", args.work_root, "--out_root", args.processed_root,
             "--workers", str(args.workers)] + cfg_opt,
            "② 数据预处理 (kws/cmd 降噪+自适应分离)",
        )
        write_done(marker_prep)

    # ============================================================
    # ③ 质量门控 (脏样本检测+过滤)
    # ============================================================
    if args.skip_gate:
        print("⏭️  [跳过] 质量门控 (用户指定)")
    else:
        gate_cmd = [
            py, os.path.join(TOOLS_DIR, "quality_gate.py"),
            "--processed_root", args.processed_root, "--out_root", gate_root,
        ]
        if args.no_gate_filter:
            gate_cmd.append("--no_filter")
        run_cmd(gate_cmd, "③ 质量门控 (脏样本检测" + ("报告" if args.no_gate_filter else "与过滤") + ")")

    # 决定 folds 的数据源: 过滤后 or 未过滤
    if args.skip_gate or args.no_gate_filter:
        folds_root = args.processed_root
        print(f"\n[folds 数据源] 使用未过滤数据: {folds_root}")
    else:
        # 把过滤后的 jsonl 放回 processed_root 供 make_folds 读 (以 *_aug_gated.jsonl 命名)
        for split in ["pos", "neg"]:
            gated = os.path.join(gate_root, f"{split}_aug_gated.jsonl")
            if os.path.exists(gated):
                dst = os.path.join(args.processed_root, f"{split}_aug_gated.jsonl")
                shutil.copy2(gated, dst)
        # 生成 make_folds 能读的 pos_aug_processed.jsonl (gated 版)
        for split in ["pos", "neg"]:
            gated = os.path.join(args.processed_root, f"{split}_aug_gated.jsonl")
            target = os.path.join(args.processed_root, f"{split}_aug_processed.jsonl")
            if os.path.exists(gated):
                shutil.copy2(gated, target)
                print(f"[folds] 用 gated 数据覆盖 {os.path.basename(target)} ({os.path.getsize(target)}B)")
        folds_root = args.processed_root

    # ============================================================
    # ④ 5折划分
    # ============================================================
    marker_folds = os.path.join(folds_root, ".stage_folds.done")
    if not is_done(marker_folds, skip=False):
        run_cmd(
            [py, os.path.join(TOOLS_DIR, "make_folds.py"),
             "--aug_root", folds_root, "--n_folds", "5"],
            "④ 5折交叉验证划分 (按 orig_id 防泄漏)",
        )
        write_done(marker_folds)

    if args.skip_train:
        print("\n🏁 [--skip_train] 数据处理完成, 未训练")
        print(f"  门控报告: {os.path.join(gate_root, 'quality_report.json')}")
        print(f"  sim分布:  {os.path.join(gate_root, 'quality_sim_dist.csv')}")
        return

    # ============================================================
    # ⑤ 训练
    # ============================================================
    train_scripts = []
    if args.train_model in ("camplus", "both"):
        train_scripts.append(("CAM++", "train_camplus_finetune.py"))
    if args.train_model in ("eres2netv2", "both"):
        train_scripts.append(("ERes2NetV2", "train_eres2netv2_finetune.py"))

    for name, script in train_scripts:
        run_cmd(
            [py, os.path.join(TOOLS_DIR, script),
             "--aug_root", folds_root, "--fold", args.fold,
             "--epochs", str(args.epochs), "--lr", str(args.lr),
             "--batch", str(args.batch), "--workers", str(args.workers)],
            f"⑤ 训练 {name} (fold={args.fold})",
        )

    print(f"\n🎉 全流程完成! 总用时 {time.time()-t_all:.0f}s")
    print(f"  质量门控报告: {os.path.join(gate_root, 'quality_report.json')}")
    print(f"  训练权重: {folds_root}/runs/fold_{args.fold}/")


if __name__ == "__main__":
    main()
