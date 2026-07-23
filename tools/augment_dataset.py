# -*- coding: utf-8 -*-
"""
datasetA 数据增强脚本 (会议决议: 加噪声/调音量/片段截取/变声)

每条样本 (kws + cmd 成对) 生成 8 个版本:
  1. orig          原始 (直接重采样复制, 保证格式统一)
  2. vol_up        音量 +4dB
  3. vol_down      音量 -4dB
  4. noise_w15     叠加白噪声 SNR=15dB
  5. noise_p20     叠加粉噪声 SNR=20dB (近似环境噪声)
  6. crop          随机片段截取 (kws保留90%, cmd保留80%)
  7. pitch_p2      变声 +2 半音 (近似男变女/音色改变)
  8. pitch_m3      变声 -3 半音

输出:
  datasetA_aug/pos/  datasetA_aug/neg/        增强音频
  datasetA_aug/pos_aug.jsonl  neg_aug.jsonl   增强标注 (保持原字段 + aug_type/orig_id)
  datasetA_aug/aug_report.json                 增强统计

用法:
  python tools/augment_dataset.py --src "C:/Users/善水/Desktop/datasetA/datasetA" \
      --dst "F:/龙虾/2026-07-18-13-57-00/datasetA_aug" --workers 8
"""
import os
import sys
import json
import argparse
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

SR = 16000

# 增强类型定义: (aug_type, 函数名, 参数)
AUG_TYPES = [
    ("orig", None, None),
    ("vol_up", "gain", +4.0),
    ("vol_down", "gain", -4.0),
    ("noise_w15", "noise_white", 15.0),
    ("noise_p20", "noise_pink", 20.0),
    ("crop", "crop", None),          # kws/cmd 比例不同, 函数内处理
    ("pitch_p2", "pitch", +2.0),
    ("pitch_m3", "pitch", -3.0),
]


def gain_db(y, db):
    """音量调整: 增益 db"""
    return y * (10.0 ** (db / 20.0))


def _sig_power(y):
    return float(np.mean(y ** 2)) + 1e-12


def add_white_noise(y, snr_db, rng):
    """按目标 SNR 叠加白噪声"""
    noise = rng.standard_normal(len(y)).astype(np.float32)
    scale = np.sqrt(_sig_power(y) / (10.0 ** (snr_db / 10.0)) / _sig_power(noise))
    return y + noise * scale


def _pink_noise(n, rng):
    """生成粉噪声 (1/f 频谱)"""
    white = rng.standard_normal(n)
    X = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0  # 避免除零
    X = X / np.sqrt(freqs)
    pink = np.fft.irfft(X, n=n)
    return (pink / (np.max(np.abs(pink)) + 1e-9)).astype(np.float32)


def add_pink_noise(y, snr_db, rng):
    """按目标 SNR 叠加粉噪声 (低频能量多, 更接近环境/家电噪声)"""
    noise = _pink_noise(len(y), rng)
    scale = np.sqrt(_sig_power(y) / (10.0 ** (snr_db / 10.0)) / _sig_power(noise))
    return y + noise * scale


def random_crop(y, keep_ratio, rng):
    """随机片段截取: 保留 keep_ratio 比例, 随机起点"""
    keep = int(len(y) * keep_ratio)
    if keep >= len(y):
        return y
    start = int(rng.integers(0, len(y) - keep + 1))
    return y[start:start + keep]


def pitch_shift(y, n_steps):
    """变声: 音高平移 n_steps 个半音 (不改变时长)"""
    return librosa.effects.pitch_shift(y.astype(np.float32), sr=SR, n_steps=n_steps)


def apply_aug(y, aug_type, kind, rng):
    """对单个音频应用增强. kind: 'kws'/'cmd' (截取比例不同)"""
    for name, fn, param in AUG_TYPES:
        if name != aug_type:
            continue
        if fn is None:
            return y.copy()
        if fn == "gain":
            return gain_db(y, param)
        if fn == "noise_white":
            return add_white_noise(y, param, rng)
        if fn == "noise_pink":
            return add_pink_noise(y, param, rng)
        if fn == "crop":
            ratio = 0.90 if kind == "kws" else 0.80  # kws短, 保留更多
            return random_crop(y, ratio, rng)
        if fn == "pitch":
            return pitch_shift(y, param)
    raise ValueError(f"unknown aug_type: {aug_type}")


def process_one_sample(args):
    """处理一条样本: 对 (kws, cmd) 生成全部 8 种增强, 返回 jsonl 记录列表"""
    src_root, dst_root, split, rec = args
    rng = np.random.default_rng(seed=rec["id"] * 1000 + (0 if split == "pos" else 7))

    kws_src = os.path.join(src_root, rec["唤醒音频"])
    cmd_src = os.path.join(src_root, rec["识别音频"])
    y_kws, _ = librosa.load(kws_src, sr=SR, mono=True)
    y_cmd, _ = librosa.load(cmd_src, sr=SR, mono=True)

    out_dir = os.path.join(dst_root, split)
    os.makedirs(out_dir, exist_ok=True)

    records = []
    for ai, (aug_type, _, _) in enumerate(AUG_TYPES):
        # kws 与 cmd 使用同一种增强, 保持配对关系
        y_k = apply_aug(y_kws, aug_type, "kws", rng)
        y_c = apply_aug(y_cmd, aug_type, "cmd", rng)
        # 防削波
        y_k = np.clip(y_k, -1.0, 1.0).astype(np.float32)
        y_c = np.clip(y_c, -1.0, 1.0).astype(np.float32)

        base = os.path.basename(rec["唤醒音频"]).replace(".wav", "")
        cbase = os.path.basename(rec["识别音频"]).replace(".wav", "")
        kws_rel = f"{split}/{base}_{aug_type}.wav"
        cmd_rel = f"{split}/{cbase}_{aug_type}.wav"
        sf.write(os.path.join(dst_root, kws_rel), y_k, SR, subtype="PCM_16")
        sf.write(os.path.join(dst_root, cmd_rel), y_c, SR, subtype="PCM_16")

        records.append({
            "id": rec["id"] * 100 + ai,
            "orig_id": rec["id"],
            "唤醒音频": kws_rel,
            "唤醒文本": rec["唤醒文本"],
            "识别音频": cmd_rel,
            "识别文本": rec["识别文本"],
            "aug_type": aug_type,
        })
    return split, records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="原始 datasetA 根目录 (含 pos.jsonl/neg.jsonl)")
    ap.add_argument("--dst", required=True, help="增强输出根目录")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    all_records = {"pos": [], "neg": []}

    tasks = []
    for split in ["pos", "neg"]:
        with open(os.path.join(args.src, f"{split}.jsonl"), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    tasks.append((args.src, args.dst, split, json.loads(line)))

    print(f"总样本: {len(tasks)}, 增强类型: {len(AUG_TYPES)} 种, "
          f"预计输出音频: {len(tasks) * len(AUG_TYPES) * 2} 个")

    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(process_one_sample, t) for t in tasks]
        for fut in as_completed(futures):
            split, records = fut.result()
            all_records[split].extend(records)
            done += 1
            if done % 100 == 0 or done == len(tasks):
                print(f"  进度: {done}/{len(tasks)}", flush=True)

    report = {}
    for split in ["pos", "neg"]:
        all_records[split].sort(key=lambda r: r["id"])
        out_jsonl = os.path.join(args.dst, f"{split}_aug.jsonl")
        with open(out_jsonl, "w", encoding="utf-8") as f:
            for r in all_records[split]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        report[split] = {
            "orig_samples": sum(1 for t in tasks if t[2] == split),
            "aug_samples": len(all_records[split]),
            "jsonl": out_jsonl,
        }
        print(f"{split}: {report[split]['orig_samples']} 原始 → {len(all_records[split])} 增强条")

    with open(os.path.join(args.dst, "aug_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("增强完成!")


if __name__ == "__main__":
    main()
