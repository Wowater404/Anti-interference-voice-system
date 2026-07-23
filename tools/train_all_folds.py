# -*- coding: utf-8 -*-
"""
批量训练 fold 1-4 (fold_0 已单独验证完成后运行)
串行执行, 每折日志输出到 runs/fold_k/train_stdout.log
"""
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUG_ROOT = "F:/龙虾/2026-07-18-13-57-00/datasetA_aug"
PY = "F:/work/Anaconda/envs/zhinnegjiaju/python.exe"

def main():
    folds = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [1, 2, 3, 4]
    for fold in folds:
        out_dir = os.path.join(PROJECT_ROOT, "runs", f"fold_{fold}")
        log_path = os.path.join(out_dir, "train_stdout.log")
        os.makedirs(out_dir, exist_ok=True)
        print(f"===== fold_{fold} 训练开始 {time.strftime('%H:%M:%S')} =====", flush=True)
        with open(log_path, "w", encoding="utf-8") as lf:
            proc = subprocess.run(
                [PY, "-u", os.path.join(PROJECT_ROOT, "tools", "train_camplus_finetune.py"),
                 "--aug_root", AUG_ROOT, "--fold", str(fold),
                 "--epochs", "10", "--lr", "1e-4", "--batch", "64", "--workers", "8"],
                cwd=PROJECT_ROOT, stdout=lf, stderr=subprocess.STDOUT)
        print(f"===== fold_{fold} 结束 code={proc.returncode} {time.strftime('%H:%M:%S')} =====", flush=True)

if __name__ == "__main__":
    main()
