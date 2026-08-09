# -*- coding: utf-8 -*-
"""
DeepFilterNet 参数扫描 (抽样子集, 快速方向)
扫 atten_lim_db 和 post_filter 对声纹判定+RR的影响
"""
import os, sys, json, time
import numpy as np
# bash层已注入PATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PipelineConfig
from pipeline import VoicePipeline
from modules.voiceprint import EnsembleVoiceprintExtractor
from modules.denoiser import create_denoiser
from utils.audio import load_wav

DATA_ROOT = "F:/挑杯资料/datasetA"
N_POS = 80    # 抽样 pos 数 (DeepFilterNet CPU慢, 用更小样本)
N_NEG = 40    # 抽样 neg 数

# 参数网格: (atten_lim_db, post_filter)
GRID = [
    (4,  False),
    (6,  False),   # 当前默认
    (8,  False),
    (10, False),
    (12, False),
    (6,  True),
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
    pipeline.voiceprint_extractor.load()
    pipeline.separator.load()   # 分离保持关闭由 config 决定, 这里仅加载声纹
    sr = pipeline.config.sample_rate
    ens = pipeline.voiceprint_extractor

    pos, neg = load_samples()
    samples = [(s, True) for s in pos] + [(s, False) for s in neg]
    n_pos, n_neg = len(pos), len(neg)
    print(f"抽样: pos={n_pos}, neg={n_neg}")

    # 预加载音频 + kws 声纹
    kws_embs, cmd_audios, labels = [], [], []
    for sample, is_pos in samples:
        kws_audio, _ = load_wav(os.path.join(DATA_ROOT, sample["唤醒音频"]), sr)
        cmd_audio, _ = load_wav(os.path.join(DATA_ROOT, sample["识别音频"]), sr)
        kws_embs.append(ens.extract_all(kws_audio, sr))
        cmd_audios.append(cmd_audio)
        labels.append(sample.get("识别文本", None) if is_pos else None)
    print("预加载完成\n")

    print(f"{'atten':<6}{'postF':<7} | {'CER↓':<9}{'RR':<7}{'Score80':<9}{'posAcc':<8}{'negRej':<8}")
    print("-" * 70)

    best = None
    for atten, postf in GRID:
        # 重建降噪器 (post_filter 需要重新 load)
        denoiser = create_denoiser({
            "model": "deepfilternet3", "enable": True,
            "deepfilternet3": {"model_dir": None, "atten_lim_db": atten},
        }, 'cpu')
        denoiser.load()
        if postf:  # post_filter=True 需要重载模型 (df内部参数)
            try:
                import df.enhance as de
                from df.utils import get_device
                denoiser.model, denoiser.df_state, _ = de.init_df(
                    model_base_dir=None, post_filter=True, log_level="WARNING")
            except Exception as e:
                print(f"  post_filter=True 重载失败: {e}")

        raw_sims = []
        t1 = time.time()
        for i, (sample, is_pos) in enumerate(samples):
            denoised = denoiser.denoise(cmd_audios[i], sr)
            den_emb_all = ens.extract_all(denoised, sr)
            sims = tuple(ens.cosine_sim(kws_embs[i][k], den_emb_all[k]) for k in range(3))
            raw_sims.append(sims)

        fused = EnsembleVoiceprintExtractor.zscore_fuse(raw_sims, ens.weights)
        thr = ens.threshold

        total_ed = total_len = 0.0
        for i in range(n_pos):
            ln = len(labels[i]); total_len += ln
            if fused[i] < thr: total_ed += ln
        cer = total_ed / total_len if total_len else 0
        rr = int((fused[n_pos:] < thr).sum()) / n_neg
        pos_acc = int((fused[:n_pos] >= thr).sum()) / n_pos
        score = (1 - cer) * 40 + rr * 40
        dt = time.time() - t1

        print(f"{atten:<6}{str(postf):<7} | {cer:<9.4f}{rr:<7.4f}{score:<9.2f}{pos_acc:<8.3f}{rr:<8.3f}  {dt:.0f}s")
        if best is None or score > best["score"]:
            best = {"atten": atten, "postf": postf, "score": score, "cer": cer, "rr": rr}

    print(f"\n★ 抽样最优: atten_lim_db={best['atten']}, post_filter={best['postf']} (Score80≈{best['score']:.2f})")

if __name__ == "__main__":
    main()
