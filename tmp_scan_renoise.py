# -*- coding: utf-8 -*-
"""
Renoise 参数扫描 v2 (预加载 + 缓存 kws 声纹, 加速)
流程: 预加载样本+缓存kws embedding → 每组参数: 降噪→提取cmd声纹→sims→zscore→判定
"""
import os, sys, json, time
import numpy as np
# 动态 PATH 注入: 将当前 Python 所在 conda 环境的 Library/bin 前置 (修复 cuDNN DLL 加载顺序)
_lib_bin = os.path.abspath(os.path.join(os.path.dirname(sys.executable), os.pardir, 'Library', 'bin'))
if os.path.isdir(_lib_bin):
    os.environ['PATH'] = _lib_bin + os.pathsep + os.environ.get('PATH', '')
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('MODELSCOPE_CACHE', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pretrained', 'modelscope_cache'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PipelineConfig
from pipeline import VoicePipeline
from modules.voiceprint import EnsembleVoiceprintExtractor
from modules.denoiser import create_denoiser
from utils.audio import load_wav

DATA_ROOT = "F:/挑杯资料/datasetA"
N_POS = 150   # 抽样 pos 数
N_NEG = 75    # 抽样 neg 数

GRID = [
    # (stationary, prop_decrease, n_std_thresh, n_fft)
    (True,  0.5, 1.5, 1024),
    (True,  0.6, 1.5, 1024),
    (True,  0.8, 1.5, 1024),   # 组员默认
    (True,  0.9, 1.5, 1024),
    (True,  0.8, 1.0, 1024),
    (True,  0.8, 2.0, 1024),
    (True,  0.8, 1.5, 512),
    (True,  0.8, 1.5, 2048),
    (False, 0.8, 1.5, 1024),
]

def load_samples():
    pos, neg = [], []
    for name, out, n in [("pos", pos, N_POS), ("neg", neg, N_NEG)]:
        with open(os.path.join(DATA_ROOT, f"{name}.jsonl"), encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= n: break
                out.append(json.loads(line))
    return pos, neg

def main():
    cfg = PipelineConfig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "default.yaml"))
    pipeline = VoicePipeline(cfg)
    # 只加载声纹+降噪+分离, 跳过 ASR (扫描仅需声纹判定, 避免 cuDNN 冲突)
    pipeline.voiceprint_extractor.load()
    pipeline.denoiser.load()
    pipeline.separator.load()
    sr = pipeline.config.sample_rate
    ens = pipeline.voiceprint_extractor

    pos, neg = load_samples()
    samples = [(s, True) for s in pos] + [(s, False) for s in neg]
    n_pos, n_neg = len(pos), len(neg)
    print(f"抽样: pos={n_pos}, neg={n_neg}, 总={len(samples)}")

    # 预加载音频 + 缓存 kws 声纹
    print("预加载音频 + 提取 kws 声纹...")
    kws_embs = []
    cmd_audios = []
    labels = []
    t0 = time.time()
    for sample, is_pos in samples:
        kws_audio, _ = load_wav(os.path.join(DATA_ROOT, sample["唤醒音频"]), sr)
        cmd_audio, _ = load_wav(os.path.join(DATA_ROOT, sample["识别音频"]), sr)
        kws_embs.append(ens.extract_all(kws_audio, sr))
        cmd_audios.append(cmd_audio)
        labels.append(sample.get("识别文本", None) if is_pos else None)
    print(f"  预加载完成, 用时 {time.time()-t0:.1f}s\n")

    print(f"{'stat':<6}{'prop':<7}{'nstd':<6}{'nfft':<6} | {'CER↓':<9}{'RR':<7}{'Score80':<9}{'posAcc':<8}{'negRej':<8}")
    print("-" * 78)

    best = None
    for p in GRID:
        denoiser = create_denoiser({
            "model": "renoise", "enable": True,
            "renoise": {"stationary": p[0], "prop_decrease": p[1],
                        "n_std_thresh": p[2], "n_fft": p[3]},
        }, pipeline.device)
        denoiser.load()

        raw_sims = []
        t1 = time.time()
        for i, (sample, is_pos) in enumerate(samples):
            denoised = denoiser.denoise(cmd_audios[i], sr)
            den_emb_all = ens.extract_all(denoised, sr)
            sims = tuple(ens.cosine_sim(kws_embs[i][k], den_emb_all[k]) for k in range(3))
            raw_sims.append(sims)

        fused = EnsembleVoiceprintExtractor.zscore_fuse(raw_sims, ens.weights)
        thr = ens.threshold

        total_ed = 0.0
        total_len = 0.0
        for i in range(n_pos):
            ln = len(labels[i])
            total_len += ln
            if fused[i] >= thr:
                total_ed += 0.0
            else:
                total_ed += ln
        cer = total_ed / total_len if total_len > 0 else 0.0

        n_neg_rej = int((fused[n_pos:] < thr).sum())
        rr = n_neg_rej / n_neg if n_neg > 0 else 0.0
        pos_acc = int((fused[:n_pos] >= thr).sum()) / n_pos
        score = (1 - cer) * 40 + rr * 40
        dt = time.time() - t1

        print(f"{str(p[0]):<6}{p[1]:<7.1f}{p[2]:<6.1f}{p[3]:<6} | "
              f"{cer:<9.4f}{rr:<7.4f}{score:<9.2f}{pos_acc:<8.3f}{rr:<8.3f}  {dt:.0f}s")
        if best is None or score > best["score"]:
            best = {"params": p, "score": score, "cer": cer, "rr": rr}

    print(f"\n★ 抽样最优: {best['params']} (Score80≈{best['score']:.2f}, CER={best['cer']:.4f}, RR={best['rr']:.4f})")
    print("注: 抽样zscore分布≠全量, 仅选方向, 需全量确认")

if __name__ == "__main__":
    main()
