# -*- coding: utf-8 -*-
"""
训练数据处理脚本: 用 Renoise降噪 + SpEx+自适应分离 处理 datasetA_aug 的 cmd 音频

目的: 让训练数据分布与推理一致 (V5血泪教训: 训练/推理分布必须匹配)
  - 推理流程: kws用原始音频提声纹 → cmd先Renoise降噪 → 若相似度不足则SpEx+分离 → 提声纹
  - 训练数据: kws保持原始(不处理), cmd处理为"Renoise降噪+自适应分离后"音频

输出:
  datasetA_aug_processed/
    pos/cmd_*_processed.wav      (处理后的cmd)
    neg/cmd_*_processed.wav
    pos_aug_processed.jsonl      (路径指向处理后的cmd, kws路径不变)
    neg_aug_processed.jsonl

用法:
  python tools/prepare_processed_train_data.py \
    --data_root datasetA_aug --out_root datasetA_aug_processed \
    --workers 4 --device auto
"""
import os, sys, json, time, argparse
import numpy as np

# 动态 PATH 注入 (cuDNN DLL 修复)
_lib_bin = os.path.abspath(os.path.join(os.path.dirname(sys.executable), os.pardir, 'Library', 'bin'))
if os.path.isdir(_lib_bin):
    os.environ['PATH'] = _lib_bin + os.pathsep + os.environ.get('PATH', '')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import PipelineConfig
from pipeline import VoicePipeline
from modules.denoiser import create_denoiser
from utils.audio import load_wav, save_wav
from modules.voiceprint import EnsembleVoiceprintExtractor


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="datasetA_aug 目录")
    ap.add_argument("--out_root", required=True, help="输出目录 (datasetA_aug_processed)")
    ap.add_argument("--config", default=None, help="pipeline config (默认用 default.yaml)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="auto")
    return ap.parse_args()


def process_sample(pipeline, sample, data_root, out_root, sr):
    """
    处理单条样本: kws保持原始, cmd做Renoise降噪+自适应分离
    Returns: (new_cmd_relpath, None) 或 (new_cmd_relpath, error)
    """
    kws_rel = sample["唤醒音频"]
    cmd_rel = sample["识别音频"]
    sample_id = sample["id"]

    # 1. 读音频
    kws_audio, _ = load_wav(os.path.join(data_root, kws_rel), sr)
    cmd_audio, _ = load_wav(os.path.join(data_root, cmd_rel), sr)

    # 2. cmd 降噪 (Renoise, 纯CPU)
    denoised = pipeline.denoiser.denoise(cmd_audio, sr)

    # 3. 自适应分离判断: 用降噪后cmd提声纹, 与kws算相似度
    #    分离只在绝对相似度 < vp_threshold (且 >= sep_trigger_min) 时触发
    sep_enabled = pipeline.separation_enabled
    separated = None
    if sep_enabled:
        # 提取声纹 (kws原始, cmd降噪后)
        kws_emb_all = pipeline.voiceprint_extractor.extract_all(kws_audio, sr)
        cmd_emb_all = pipeline.voiceprint_extractor.extract_all(denoised, sr)
        sims = tuple(pipeline.voiceprint_extractor.cosine_sim(kws_emb_all[i], cmd_emb_all[i])
                     for i in range(3))
        # 绝对融合相似度 (加权平均)
        w = pipeline.voiceprint_extractor.weights
        sim_abs = w[0] * sims[0] + w[1] * sims[1] + w[2] * sims[2]

        if pipeline.sep_trigger_min <= sim_abs < pipeline.vp_threshold:
            # 触发分离
            pipeline.separator.set_reference_audio(kws_audio, sr)
            separated, sources = pipeline.separator.separate(denoised, sr)
            # 声纹辅助选轨: 选与kws最相似的音轨
            if separated is not None and sources:
                best_sim, best_audio = -1.0, separated
                for src in sources:
                    src_emb = pipeline.voiceprint_extractor.extract_all(src, sr)
                    src_sim = pipeline.voiceprint_extractor.cosine_sim(kws_emb_all[0], src_emb[0])
                    if src_sim > best_sim and (src_sim - sim_abs) <= pipeline.sim_jump_cap:
                        best_sim, best_audio = src_sim, src
                separated = best_audio

    final_cmd = separated if separated is not None else denoised

    # 4. 保存处理后的cmd
    #    输出相对路径: 与输入同结构, 但cmd文件名加 _processed 后缀
    subdir = os.path.dirname(cmd_rel)  # "pos" 或 "neg"
    base = os.path.basename(cmd_rel).replace(".wav", "_processed.wav")
    new_cmd_rel = os.path.join(subdir, base)
    out_cmd_path = os.path.join(out_root, new_cmd_rel)
    os.makedirs(os.path.dirname(out_cmd_path), exist_ok=True)
    save_wav(out_cmd_path, final_cmd, sr)

    # 5. 返回更新后的样本 (kws路径不变, cmd指向处理后的)
    new_sample = dict(sample)
    new_sample["识别音频"] = new_cmd_rel
    new_sample["cmd_processed"] = True
    new_sample["sep_triggered"] = separated is not None
    return new_sample


def main():
    args = parse_args()
    sr = 16000
    cfg_path = args.config or os.path.join(PROJECT_ROOT, "configs", "default.yaml")

    print("加载流水线 (声纹+降噪+分离)...")
    cfg = PipelineConfig(cfg_path)
    pipeline = VoicePipeline(cfg)
    pipeline.voiceprint_extractor.load()
    pipeline.denoiser.load()
    if pipeline.separation_enabled:
        pipeline.separator.load()
    print(f"分离启用: {pipeline.separation_enabled}")

    os.makedirs(args.out_root, exist_ok=True)

    for split in ["pos", "neg"]:
        jsonl_in = os.path.join(args.data_root, f"{split}_aug.jsonl")
        jsonl_out = os.path.join(args.out_root, f"{split}_aug_processed.jsonl")
        with open(jsonl_in, encoding="utf-8") as f:
            samples = [json.loads(l) for l in f if l.strip()]
        print(f"\n[{split}] 处理 {len(samples)} 条...")

        t0 = time.time()
        # 断点续跑: 若输出wav已存在则跳过 (17:00保存进度后重启可继续)
        skipped = 0
        with open(jsonl_out, "w", encoding="utf-8") as fout:
            for i, s in enumerate(samples):
                # 计算输出路径 (与 process_sample 一致)
                cmd_rel = s["识别音频"]
                base = os.path.basename(cmd_rel).replace(".wav", "_processed.wav")
                out_cmd_rel = os.path.join(os.path.dirname(cmd_rel), base)
                if os.path.isfile(os.path.join(args.out_root, out_cmd_rel)):
                    skipped += 1
                    ns = dict(s)
                    ns["识别音频"] = out_cmd_rel
                    ns["cmd_processed"] = True
                    fout.write(json.dumps(ns, ensure_ascii=False) + "\n")
                    continue
                try:
                    ns = process_sample(pipeline, s, args.data_root, args.out_root, sr)
                    fout.write(json.dumps(ns, ensure_ascii=False) + "\n")
                except Exception as e:
                    print(f"  ⚠️ 样本 {s.get('id')} 失败: {e}, 跳过")
                if (i + 1) % 2000 == 0:
                    print(f"  {i+1}/{len(samples)} 用时{time.time()-t0:.0f}s (跳过{skipped})", flush=True)
        print(f"[{split}] 完成, 用时 {time.time()-t0:.0f}s, 跳过{skipped} → {jsonl_out}")

    print("\n全部完成!")
    print(f"输出: {args.out_root}")
    print("下一步: 用 make_folds.py 对 datasetA_aug_processed 重新划分, 然后训练")


if __name__ == "__main__":
    main()
