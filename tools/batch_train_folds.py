# -*- coding: utf-8 -*-
"""
五折交叉验证批量训练 (串行自动接续 fold_2 → fold_3 → fold_4)
每折完成后自动启动下一折, 全部跑完输出汇总
"""
import os, sys, json, time, subprocess

AUG_ROOT = "../datasetA_aug_processed"
FOLDS = [2, 3, 4]  # fold_0/1 已完成
EPOCHS = 10
LR = 1e-4
BATCH = 64

PYTHON = r"F:/work/Anaconda/envs/zhinnegjiaju/python.exe"
SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_camplus_finetune.py")

results = {}
for fold in FOLDS:
    out_dir = f"runs/camplus_v7_fold{fold}"
    cmd = [
        PYTHON, SCRIPT,
        "--aug_root", AUG_ROOT,
        "--fold", str(fold),
        "--epochs", str(EPOCHS),
        "--lr", str(LR),
        "--batch", str(BATCH),
        "--workers", "8",
        "--out_dir", out_dir,
    ]
    print(f"\n{'='*60}\n[fold_{fold}] 开始训练 {time.strftime('%H:%M:%S')}\n{'='*60}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    dt = (time.time() - t0) / 60
    print(f"[fold_{fold}] 完成, 用时 {dt:.1f} 分钟, returncode={proc.returncode}", flush=True)
    if proc.returncode != 0:
        print(f"[fold_{fold}] 错误输出 (尾部):")
        print("\n".join(proc.stderr.strip().split("\n")[-10:]))
        results[fold] = {"returncode": proc.returncode, "error": True}
        continue
    # 读日志
    log_path = os.path.join(out_dir, "train_log.json")
    if os.path.exists(log_path):
        with open(log_path, encoding="utf-8") as f:
            log = json.load(f)
        best = min([e for e in log["epochs"] if e.get("best")], key=lambda x: x["val_eer"], default=None)
        results[fold] = {
            "baseline_eer": log.get("baseline_eer"),
            "best_eer": best["val_eer"] if best else None,
            "best_epoch": best["epoch"] if best else None,
        }
        print(f"[fold_{fold}] baseline_eer={log.get('baseline_eer'):.4f} → best_eer={best['val_eer']:.4f}" if best else f"[fold_{fold}] 无best", flush=True)

print("\n" + "=" * 60)
print("五折批量训练完成! 汇总:")
print("=" * 60)
for fold, r in results.items():
    if r.get("error"):
        print(f"fold_{fold}: 失败 (returncode={r['returncode']})")
    else:
        print(f"fold_{fold}: baseline {r['baseline_eer']:.4f} → best {r['best_eer']:.4f} (epoch {r['best_epoch']})")
