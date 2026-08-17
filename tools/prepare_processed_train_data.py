# -*- coding: utf-8 -*-
"""
训练数据处理脚本: 处理 datasetA_aug 的 kws 和 cmd 音频, 与推理侧完全对齐

目的: 训练/推理分布必须匹配 (V5血泪教训 + V9 kws 处理对齐)
  - 推理侧 (V15): kws 走 _process_kws(降噪→唤醒词匹配→不够才盲分离) → 提声纹
                 cmd 走 DeepFilterNet3 降噪 → 若相似度不足则 SpEx+ 分离 → 提声纹
  - 训练侧 (本脚本): kws 同样走 _process_kws, cmd 走 DeepFilterNet3 降噪 + 自适应分离
  - 关键: kws 和 cmd 的处理逻辑与 pipeline.py 完全一致, 保证训练/推理分布对齐

输出:
  datasetA_aug_processed/
    pos/kws_*_processed.wav     (处理后的 kws: 降噪+自适应盲分离)
    pos/cmd_*_processed.wav     (处理后的 cmd: 降噪+自适应分离)
    neg/kws_*_processed.wav
    neg/cmd_*_processed.wav
    pos_aug_processed.jsonl     (kws 和 cmd 都指向处理后的文件)
    neg_aug_processed.jsonl

用法:
  python tools/prepare_processed_train_data.py \
    --data_root datasetA_aug --out_root datasetA_aug_processed \
    --workers 4 --device auto
"""
import os, sys, json, time, argparse
import numpy as np
# [重要] 必须先 import torch，再写 os.environ/os.chdir：
# torch 2.x 在 Windows 上，import torch 之前任何 os.environ 写入会导致段错误。
import torch  # noqa: F401

# [2026-08-15] cuDNN 加载修复 (根因: zhinnegjiaju\Library\bin 不在 PATH,
# 导致 cudnn64_9.dll 的子 DLL 找不到 → "Invalid handle: Cannot load symbol cudnnGetVersion")。
# 修复: 在 cudnn 首次加载前把正确的 Library/bin 注入 PATH。
# 注意: cudnn 是延迟加载的 (首次 cuda 卷积时才解析 DLL), 所以 import torch 后注入仍有效。
_lib_bin = os.path.abspath(os.path.join(os.path.dirname(sys.executable), 'Library', 'bin'))
if os.path.isdir(_lib_bin):
    os.environ['PATH'] = _lib_bin + os.pathsep + os.environ.get('PATH', '')
    print(f"[PPS] PATH 注入: {_lib_bin}", flush=True)

# 兜底开关: 若环境仍有 cuDNN 问题, 可设 PPS_DISABLE_CUDNN=1 强制禁用 (慢但正确)。
if os.environ.get("PPS_DISABLE_CUDNN", "0") == "1":
    torch.backends.cudnn.enabled = False
    print("[PPS] cuDNN 已禁用 (PPS_DISABLE_CUDNN=1, 兜底模式)", flush=True)

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
    ap.add_argument("--config", default=None, help="pipeline config (默认用 verify_dual_full.yaml = DeepFilterNet3降噪+V15逻辑)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="auto")
    return ap.parse_args()


def process_sample(pipeline, sample, data_root, out_root, sr):
    """
    处理单条样本: kws 和 cmd 都做降噪(+kws 自适应盲分离), cmd 再做自适应 SpEx+ 分离
    (与推理侧 pipeline.py 完全对齐: kws 走 _process_kws, cmd 走降噪+分离)
    """
    kws_rel = sample["唤醒音频"]
    cmd_rel = sample["识别音频"]
    kws_text = sample.get("唤醒文本", None)

    # 1. 读音频
    kws_audio, _ = load_wav(os.path.join(data_root, kws_rel), sr)
    cmd_audio, _ = load_wav(os.path.join(data_root, cmd_rel), sr)

    # 2. cmd 降噪 (V10: 提前到 kws 之前, 作 kws 选轨参照)
    denoised = pipeline.denoiser.denoise(cmd_audio, sr)

    # 3. kws 处理 (V10: 降噪→盲分离→用 cmd 声纹选轨, 与推理侧对齐)
    #    选轨声纹复用, 不再重复提取
    kws_processed, kws_meta, _, kws_emb_all = pipeline._process_kws(
        kws_audio, kws_text, sr,
        ref_embedding=pipeline.voiceprint_extractor.extract(denoised, sr),
    )

    # 4. 自适应分离判断: 用降噪后 cmd 提声纹, 与处理后的 kws 算相似度
    #    分离只在绝对相似度 < vp_threshold (且 >= sep_trigger_min) 时触发
    sep_enabled = pipeline.separation_enabled
    separated = None
    sim_abs = None          # 分离前绝对融合相似度 (质量门控用)
    sep_best_sim = None     # 分离后最佳相似度 (质量门控用)
    if sep_enabled:
        # 提取声纹 (处理后的 kws, 降噪后的 cmd)
        if kws_emb_all is None or len(kws_emb_all) != 3:
            kws_emb_all = pipeline.voiceprint_extractor.extract_all(kws_processed, sr)
        cmd_emb_all = pipeline.voiceprint_extractor.extract_all(denoised, sr)
        sims = tuple(pipeline.voiceprint_extractor.cosine_sim(kws_emb_all[i], cmd_emb_all[i])
                     for i in range(3))
        # 绝对融合相似度 (加权平均)
        w = pipeline.voiceprint_extractor.weights
        sim_abs = float(w[0] * sims[0] + w[1] * sims[1] + w[2] * sims[2])
        sep_best_sim = sim_abs  # 未分离时最佳=分离前

        if pipeline.sep_trigger_min <= sim_abs < pipeline.vp_threshold:
            # 触发分离 (用处理后的干净 kws 做参考)
            pipeline.separator.set_reference_audio(kws_processed, sr)
            separated, sources = pipeline.separator.separate(denoised, sr)
            # 声纹辅助选轨: 选与 kws 最相似的音轨
            if separated is not None and sources:
                best_sim, best_audio = -1.0, separated
                for src in sources:
                    src_emb = pipeline.voiceprint_extractor.extract_all(src, sr)
                    src_sim = pipeline.voiceprint_extractor.cosine_sim(kws_emb_all[0], src_emb[0])
                    if src_sim > best_sim and (src_sim - sim_abs) <= pipeline.sim_jump_cap:
                        best_sim, best_audio = src_sim, src
                separated = best_audio
                sep_best_sim = float(best_sim)

    final_cmd = separated if separated is not None else denoised

    # 5. 保存处理后的 kws 和 cmd
    subdir = os.path.dirname(cmd_rel)  # "pos" 或 "neg"

    # kws_*_processed.wav
    kws_base = os.path.basename(kws_rel).replace(".wav", "_processed.wav")
    new_kws_rel = os.path.join(subdir, kws_base)
    out_kws_path = os.path.join(out_root, new_kws_rel)
    os.makedirs(os.path.dirname(out_kws_path), exist_ok=True)
    save_wav(out_kws_path, kws_processed, sr)

    # cmd_*_processed.wav
    cmd_base = os.path.basename(cmd_rel).replace(".wav", "_processed.wav")
    new_cmd_rel = os.path.join(subdir, cmd_base)
    out_cmd_path = os.path.join(out_root, new_cmd_rel)
    os.makedirs(os.path.dirname(out_cmd_path), exist_ok=True)
    save_wav(out_cmd_path, final_cmd, sr)

    # 6. 返回更新后的样本 (kws 和 cmd 都指向处理后的)
    new_sample = dict(sample)
    new_sample["唤醒音频"] = new_kws_rel
    new_sample["识别音频"] = new_cmd_rel
    new_sample["kws_processed"] = True
    new_sample["cmd_processed"] = True
    new_sample["sep_triggered"] = separated is not None
    new_sample["kws_separated"] = kws_meta.get("kws_separated", False)
    # 质量门控字段: 分离前绝对相似度 + 分离后最佳相似度 (供 quality_gate.py 检测脏样本)
    if sim_abs is not None:
        new_sample["sim_abs"] = round(sim_abs, 6)
    if sep_best_sim is not None:
        new_sample["sep_best_sim"] = round(sep_best_sim, 6)
    return new_sample


def main():
    args = parse_args()
    sr = 16000
    # [2026-08-16] 默认 config 改为 verify_dual_full.yaml:
    # V15 已确认 DeepFilterNet3 (atten_lim_db=10, post_filter=False) 为最优参数组合
    # (CER 0.3484, RR 0.9951, Score 65.87), 默认.yaml 仍是 GTCRN 旧配置, 误用会让预处理结果偏离推理侧.
    cfg_path = args.config or os.path.join(PROJECT_ROOT, "configs", "verify_dual_full.yaml")

    print("加载流水线 (声纹+降噪+分离+ASR+kws盲分离)...")
    cfg = PipelineConfig(cfg_path)
    pipeline = VoicePipeline(cfg)
    pipeline.voiceprint_extractor.load()
    pipeline.denoiser.load()
    if pipeline.separation_enabled:
        pipeline.separator.load()
    # kws 自适应处理需要 ASR(唤醒词匹配) 和 kws 盲分离模型
    pipeline.asr.load()
    if pipeline.kws_enable:
        pipeline.kws_separator.load()
    print(f"分离启用: {pipeline.separation_enabled}")
    print(f"kws 自适应处理启用: {pipeline.kws_enable}")

    os.makedirs(args.out_root, exist_ok=True)

    for split in ["pos", "neg"]:
        jsonl_in = os.path.join(args.data_root, f"{split}_aug.jsonl")
        jsonl_out = os.path.join(args.out_root, f"{split}_aug_processed.jsonl")
        with open(jsonl_in, encoding="utf-8") as f:
            samples = [json.loads(l) for l in f if l.strip()]
        print(f"\n[{split}] 处理 {len(samples)} 条...")

        t0 = time.time()
        # 断点续跑: kws 和 cmd 处理文件都已存在则跳过
        skipped = 0
        with open(jsonl_out, "w", encoding="utf-8") as fout:
            for i, s in enumerate(samples):
                # 计算输出路径 (与 process_sample 一致)
                kws_rel = s["唤醒音频"]
                cmd_rel = s["识别音频"]
                kws_base = os.path.basename(kws_rel).replace(".wav", "_processed.wav")
                out_kws_rel = os.path.join(os.path.dirname(kws_rel), kws_base)
                cmd_base = os.path.basename(cmd_rel).replace(".wav", "_processed.wav")
                out_cmd_rel = os.path.join(os.path.dirname(cmd_rel), cmd_base)

                if (os.path.isfile(os.path.join(args.out_root, out_cmd_rel))
                        and os.path.isfile(os.path.join(args.out_root, out_kws_rel))):
                    skipped += 1
                    ns = dict(s)
                    ns["唤醒音频"] = out_kws_rel
                    ns["识别音频"] = out_cmd_rel
                    ns["kws_processed"] = True
                    ns["cmd_processed"] = True
                    # 断点续跑时保留旧的质量门控字段 (若旧 jsonl 已有则沿用)
                    for fld in ("sim_abs", "sep_best_sim"):
                        if fld in s:
                            ns[fld] = s[fld]
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
