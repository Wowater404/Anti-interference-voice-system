# -*- coding: utf-8 -*-
"""
DeepFilterNet3 微调 - 数据准备脚本
把 LibriSpeech clean (16k flac) + DEMAND 噪声 (48k wav) 预处理为 48k 短片段缓存

输出:
  df3_finetune/data/
    clean/xxx.npy        # 48kHz float32 干净语音片段 (每个 3s, 固定长度)
    clean_meta.json      # 片段列表
    noise/xxx.npy        # 48kHz float32 环境噪声片段 (每类随机切片)
    noise_meta.json

用法:
  python tools/prepare_df3_data.py \
    --librispeech_dir "F:/.../raw/librispeech" \
    --noise_dir "F:/.../raw/noise" \
    --out_dir "F:/.../df3_finetune/data" \
    --seg_len_s 3.0 --workers 4
"""
import os
import sys
import json
import time
import argparse
import zipfile

# [重要] 先 import torch 再写 os.environ (Windows 段错误规避)
import torch  # noqa: F401

# cuDNN PATH 注入
_lib_bin = os.path.abspath(os.path.join(os.path.dirname(sys.executable), 'Library', 'bin'))
if os.path.isdir(_lib_bin):
    os.environ['PATH'] = _lib_bin + os.pathsep + os.environ.get('PATH', '')

import numpy as np
import librosa
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

SR_TARGET = 48000  # DF3 原生采样率
SEG_LEN = 3.0      # 片段长度(秒)


def extract_tar_gz(tar_path, out_dir):
    """解压 tar.gz (LibriSpeech)"""
    import tarfile
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(out_dir)
    print(f"解压完成: {tar_path} -> {out_dir}")


def extract_zip(zip_path, out_dir):
    """解压 zip (DEMAND)"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(out_dir)
    print(f"解压完成: {zip_path} -> {out_dir}")


def load_flac_to_48k(flac_path):
    """加载 flac → 48kHz float32 波形 (DF3 采样率)"""
    y, sr = librosa.load(flac_path, sr=None, mono=True)
    if sr != SR_TARGET:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR_TARGET)
    return y.astype(np.float32)


def process_clean_file(args):
    """单个 clean 文件 → 3s 片段 npy"""
    flac_path, out_clean_dir, idx = args
    y = load_flac_to_48k(flac_path)
    n_seg = max(1, int(len(y) / (SR_TARGET * SEG_LEN)))
    saved = []
    for i in range(n_seg):
        seg = y[i * int(SR_TARGET * SEG_LEN): (i + 1) * int(SR_TARGET * SEG_LEN)]
        if len(seg) < SR_TARGET * 2:  # 少于 2s 丢弃
            continue
        name = f"c_{idx:06d}_{i:02d}.npy"
        np.save(os.path.join(out_clean_dir, name), seg)
        saved.append(name)
    return len(saved)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--librispeech_dir", required=True, help="含 dev-clean.tar.gz / test-clean.tar.gz 的目录")
    ap.add_argument("--noise_dir", required=True, help="含 DEMAND *48k.zip 的目录")
    ap.add_argument("--out_dir", required=True, help="输出数据目录")
    ap.add_argument("--seg_len_s", type=float, default=SEG_LEN)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    clean_dir = out_dir / "clean"
    noise_dir = out_dir / "noise"
    clean_dir.mkdir(parents=True, exist_ok=True)
    noise_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. LibriSpeech 解压 ----
    ls_dir = Path(args.librispeech_dir)
    for tar in ["dev-clean.tar.gz", "test-clean.tar.gz"]:
        p = ls_dir / tar
        if p.exists():
            extract_tar_gz(str(p), str(out_dir / "librispeech"))
    # 找所有 flac
    flacs = sorted(list((out_dir / "librispeech").rglob("*.flac")))
    print(f"LibriSpeech flac 数量: {len(flacs)}")

    # ---- 2. clean 片段 ----
    print("生成 clean 片段...")
    t0 = time.time()
    tasks = [(str(f), str(clean_dir), i) for i, f in enumerate(flacs)]
    total = 0
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(process_clean_file, t) for t in tasks]
            for fu in as_completed(futs):
                total += fu.result()
    else:
        for t in tasks:
            total += process_clean_file(t)
    print(f"clean 片段: {total} 个, 用时 {time.time()-t0:.0f}s")

    # ---- 3. DEMAND 噪声 ----
    print("处理 DEMAND 噪声...")
    noise_zips = sorted(list(Path(args.noise_dir).glob("*48k.zip")))
    noise_meta = {}
    for z in noise_zips:
        cat = z.stem.replace("_48k", "")
        extract_zip(str(z), str(out_dir / "noise_raw"))
        wavs = sorted(list((out_dir / "noise_raw").rglob("*.wav")))
        n_saved = 0
        for w in wavs:
            y, sr = librosa.load(str(w), sr=None, mono=True)
            if sr != SR_TARGET:
                y = librosa.resample(y, orig_sr=sr, target_sr=SR_TARGET)
            y = y.astype(np.float32)
            # 切成 3s 片段
            n_seg = max(1, int(len(y) / (SR_TARGET * SEG_LEN)))
            for i in range(n_seg):
                seg = y[i * int(SR_TARGET * SEG_LEN): (i + 1) * int(SR_TARGET * SEG_LEN)]
                if len(seg) < SR_TARGET * 2:
                    continue
                name = f"n_{cat}_{n_saved:05d}.npy"
                np.save(noise_dir / name, seg)
                n_saved += 1
        noise_meta[cat] = n_saved
        print(f"  {cat}: {n_saved} 个噪声片段")
    # 记录总数
    noise_total = sum(noise_meta.values())
    print(f"噪声片段总数: {noise_total}")

    # ---- 4. 元数据 ----
    meta = {
        "sr": SR_TARGET,
        "seg_len_s": SEG_LEN,
        "n_clean": total,
        "n_noise": noise_total,
        "noise_cats": noise_meta,
    }
    with open(out_dir / "data_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"元数据 -> {out_dir / 'data_meta.json'}")
    print("数据准备完成!")


if __name__ == "__main__":
    main()
