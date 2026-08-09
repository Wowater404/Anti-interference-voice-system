# -*- coding: utf-8 -*-
"""
融合权重 + zscore阈值 联合扫描 (验证折368条, 双微调模型)
Step1: 提取368条原始三模型相似度 (不需要ASR, 快)
Step2: 扫描 权重网格 × 阈值网格 → 找最优组合
"""
import os, sys, json, time
import numpy as np

_lib_bin = os.path.abspath(os.path.join(os.path.dirname(sys.executable), os.pardir, 'Library', 'bin'))
if os.path.isdir(_lib_bin):
    os.environ['PATH'] = _lib_bin + os.pathsep + os.environ.get('PATH', '')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import PipelineConfig
from pipeline import VoicePipeline
from utils.audio import load_wav
from utils.metrics import compute_micro_cer
from modules.voiceprint import EnsembleVoiceprintExtractor

DATA_ROOT = "F:/挑杯资料/datasetA"
VAL_JSONL = os.path.join(PROJECT_ROOT, "..", "datasetA_aug_processed", "folds", "fold_0", "val.jsonl")
CONFIG = os.path.join(PROJECT_ROOT, "configs", "verify_dual_finetune.yaml")
SIMS_CACHE = os.path.join(PROJECT_ROOT, "results", "val_fold0_dual_raw_sims.npz")


def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def extract_raw_sims():
    """提取368条验证折的原始三模型相似度"""
    print("加载流水线...")
    cfg = PipelineConfig(CONFIG)
    pipeline = VoicePipeline(cfg)
    pipeline.voiceprint_extractor.load()
    pipeline.denoiser.load()
    if pipeline.separation_enabled:
        pipeline.separator.load()
    sr = pipeline.config.sample_rate
    ens = pipeline.voiceprint_extractor

    val_recs = load_jsonl(VAL_JSONL)
    val_pos_ids = set(r["orig_id"] for r in val_recs if r.get("识别文本") is not None)
    val_neg_names = set()
    for r in val_recs:
        if r.get("识别文本") is None:
            kws = os.path.basename(r["唤醒音频"]).replace("_orig.wav", ".wav")
            cmd = os.path.basename(r["识别音频"]).replace("_orig_processed.wav", ".wav")
            val_neg_names.add((kws, cmd))

    # 取样本: pos 用 id 匹配, neg 用文件名匹配
    samples = []
    for split in ["pos", "neg"]:
        with open(os.path.join(DATA_ROOT, f"{split}.jsonl"), encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if split == "pos" and r["id"] in val_pos_ids:
                        samples.append(r)
                    elif split == "neg":
                        kws = os.path.basename(r["唤醒音频"])
                        cmd = os.path.basename(r["识别音频"])
                        if (kws, cmd) in val_neg_names:
                            samples.append(r)

    # 逐条: 降噪→(自适应分离)→三模型sims
    raw_sims, labels, truths = [], [], []
    t0 = time.time()
    for i, sample in enumerate(samples):
        kws_audio, _ = load_wav(os.path.join(DATA_ROOT, sample["唤醒音频"]), sr)
        cmd_audio, _ = load_wav(os.path.join(DATA_ROOT, sample["识别音频"]), sr)
        label = sample.get("识别文本", None)

        denoised = pipeline.denoiser.denoise(cmd_audio, sr)
        kws_emb_all = ens.extract_all(kws_audio, sr)
        den_emb_all = ens.extract_all(denoised, sr)
        sims = tuple(ens.cosine_sim(kws_emb_all[k], den_emb_all[k]) for k in range(3))

        # 自适应分离 (与pipeline一致: sim_abs < vp_threshold 触发)
        w = ens.weights
        sim_abs = w[0]*sims[0] + w[1]*sims[1] + w[2]*sims[2]
        if pipeline.separation_enabled and pipeline.sep_trigger_min <= sim_abs < pipeline.vp_threshold:
            pipeline.separator.set_reference_audio(kws_audio, sr)
            separated, sources = pipeline.separator.separate(denoised, sr)
            if separated is not None:
                sep_emb = ens.extract_all(separated, sr)
                sep_sims = tuple(ens.cosine_sim(kws_emb_all[k], sep_emb[k]) for k in range(3))
                sims = sep_sims  # 用分离后sims

        raw_sims.append(sims)
        labels.append(label is not None)
        truths.append(label)
        if (i+1) % 50 == 0:
            print(f"  {i+1}/{len(samples)} 用时{time.time()-t0:.0f}s", flush=True)

    np.savez(SIMS_CACHE, sims=np.array(raw_sims), labels=np.array(labels),
             truths=np.array([t if t else "" for t in truths]))
    print(f"原始sims已缓存: {SIMS_CACHE}")
    return np.array(raw_sims), np.array(labels), truths


def scan(raw_sims, labels, truths):
    """扫描权重×阈值"""
    n = len(labels)
    pos_mask = labels
    neg_mask = ~labels

    best = {"score": -1}
    w_candidates = [
        (0.4, 0.4, 0.2),  # 当前
        (0.45, 0.45, 0.1),
        (0.5, 0.4, 0.1),
        (0.4, 0.5, 0.1),
        (0.5, 0.5, 0.0),
        (0.35, 0.45, 0.2),
        (0.45, 0.35, 0.2),
        (0.6, 0.3, 0.1),
        (0.3, 0.6, 0.1),
        (0.5, 0.35, 0.15),
        (0.35, 0.5, 0.15),
    ]
    thr_range = np.arange(-0.35, 0.05, 0.01)

    for w in w_candidates:
        fused = EnsembleVoiceprintExtractor.zscore_fuse(raw_sims, w)
        for thr in thr_range:
            accept = fused >= thr
            pos_acc = accept[pos_mask].sum()
            neg_fa = accept[neg_mask].sum()
            # 计算CER: 拒识=空字符串=删除错误
            total_ed = total_len = 0
            for i in range(n):
                if labels[i]:  # pos
                    gt = truths[i]
                    total_len += len(gt)
                    if not accept[i]:
                        total_ed += len(gt)
            cer = total_ed / total_len if total_len else 0
            rr = 1 - neg_fa / neg_mask.sum()
            score = (1 - cer) * 40 + rr * 40
            if score > best["score"]:
                best = {"score": score, "w": w, "thr": round(float(thr), 2),
                        "cer": cer, "rr": rr, "pos_acc": pos_acc, "neg_fa": int(neg_fa)}

    print("\n=== 扫描结果 ===")
    print(f"最优: 权重={best['w']}, 阈值={best['thr']}")
    print(f"Score={best['score']:.2f} | CER={best['cer']:.4f} | RR={best['rr']:.4f}")
    print(f"pos接受={best['pos_acc']}/{pos_mask.sum()} ({best['pos_acc']/pos_mask.sum()*100:.1f}%)")
    print(f"neg假接受={best['neg_fa']}/{neg_mask.sum()}")
    print()
    print("=== 当前配置对比 (w=0.4/0.4/0.2, thr=-0.17) ===")
    fused = EnsembleVoiceprintExtractor.zscore_fuse(raw_sims, (0.4, 0.4, 0.2))
    accept = fused >= -0.17
    pos_acc = accept[pos_mask].sum()
    neg_fa = accept[neg_mask].sum()
    total_ed = total_len = 0
    for i in range(n):
        if labels[i]:
            gt = truths[i]
            total_len += len(gt)
            if not accept[i]:
                total_ed += len(gt)
    cer = total_ed / total_len if total_len else 0
    rr = 1 - neg_fa / neg_mask.sum()
    score = (1 - cer) * 40 + rr * 40
    print(f"Score={score:.2f} | CER={cer:.4f} | RR={rr:.4f} | pos接受={pos_acc}/{pos_mask.sum()} | neg假接受={neg_fa}")


if __name__ == "__main__":
    if os.path.exists(SIMS_CACHE):
        print(f"使用缓存: {SIMS_CACHE}")
        d = np.load(SIMS_CACHE, allow_pickle=True)
        raw_sims, labels = d["sims"], d["labels"]
        truths = [str(t) for t in d["truths"]]
    else:
        raw_sims, labels, truths = extract_raw_sims()
    scan(raw_sims, labels, truths)
