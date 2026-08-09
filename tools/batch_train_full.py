# -*- coding: utf-8 -*-
"""
fold_full 全量训练 (CAM++ → ERes2NetV2 串行)
产出最终比赛模型: camplus_v7_full.pt + eres2netv2_v7_full.pt
"""
import os, sys, json, time, subprocess

AUG_ROOT = "../datasetA_aug_processed"
PYTHON = r"F:/work/Anaconda/envs/zhinnegjiaju/python.exe"
TOOLS = os.path.dirname(os.path.abspath(__file__))

JOBS = [
    {"name": "camplus_v7_full", "script": "train_camplus_finetune.py",
     "out": "runs/camplus_v7_full"},
    {"name": "eres2netv2_v7_full", "script": "train_eres2netv2_finetune.py",
     "out": "runs/eres2netv2_v7_full"},
]

for job in JOBS:
    script = os.path.join(TOOLS, job["script"])
    cmd = [PYTHON, script, "--aug_root", AUG_ROOT, "--fold", "full",
           "--epochs", "10", "--lr", "1e-4", "--batch", "64",
           "--workers", "8", "--out_dir", job["out"]]
    print(f"\n{'='*60}\n[{job['name']}] 开始 fold_full 训练 {time.strftime('%H:%M:%S')}\n{'='*60}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    dt = (time.time() - t0) / 60
    print(f"[{job['name']}] 完成, 用时 {dt:.1f} 分钟, returncode={proc.returncode}", flush=True)
    if proc.returncode != 0:
        print(f"[{job['name']}] 错误 (尾部):")
        print("\n".join(proc.stderr.strip().split("\n")[-15:]))
        continue
    # 读日志
    log_path = os.path.join(job["out"], "train_log.json")
    if os.path.exists(log_path):
        with open(log_path, encoding="utf-8") as f:
            log = json.load(f)
        epochs = log.get("epochs", [])
        last = epochs[-1] if epochs else {}
        print(f"[{job['name']}] epochs={len(epochs)}, 最后loss={last.get('loss'):.4f}, "
              f"val_eer={last.get('val_eer'):.4f}(监控口径,含训练数据)", flush=True)

print("\n" + "=" * 60)
print("fold_full 全部完成!")
print("=" * 60)
for job in JOBS:
    pt = os.path.join(job["out"], "camplus_finetuned_best.pt")
    if os.path.exists(pt):
        print(f"  ✅ {job['name']}: {pt} ({os.path.getsize(pt)/1e6:.1f}MB)")
    else:
        print(f"  ❌ {job['name']}: 未找到权重")
