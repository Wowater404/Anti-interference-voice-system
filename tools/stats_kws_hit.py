# -*- coding: utf-8 -*-
"""kws 唤醒词命中率统计（复用 _match_kws 三层逻辑，与推理行为一致）"""
import os
import sys
import json

import torch  # noqa
_lib_bin = os.path.abspath(os.path.join(os.path.dirname(sys.executable), 'Library', 'bin'))
if os.path.isdir(_lib_bin):
    os.environ['PATH'] = _lib_bin + os.pathsep + os.environ.get('PATH', '')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# [2026-08-17] 提前 import funasr: 轻量模式下跳过声纹加载后,
# funasr 首次 import 会与 modules 导入顺序冲突 (8-14 已记录此坑), 必须先于 pipeline import。
# 全量模式声纹加载会隐式初始化 funasr 依赖, 轻量模式无此兜底。
try:
    from funasr import AutoModel  # noqa
    _FUNASR_OK = True
except Exception as _e:  # noqa
    _FUNASR_OK = False
    print(f"[stats_kws_hit] 提前 import funasr 失败: {_e}", flush=True)

from config import PipelineConfig
from pipeline import VoicePipeline
from utils.audio import load_wav


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/verify_dual_full.yaml",
                    help="流水线配置文件（用于对比 V15/V16 降噪）")
    ap.add_argument("--tag", default="default", help="输出结果标识")
    ap.add_argument("--no-recheck", action="store_true",
                    help="跳过 Nano 中文复核（复现 V16 旧版行为）")
    ap.add_argument("--no-denoise", action="store_true",
                    help="不做降噪直接输入原始音频（复现 V12）")
    ap.add_argument("--limit", type=int, default=0,
                    help="只统计前 N 条（快速验证，0=全量）")
    ap.add_argument("--resume", type=int, default=0,
                    help="从第 N 条开始继续（断点续跑，跳过已处理样本）")
    ap.add_argument("--timeout", type=float, default=0,
                    help="单样本超时秒数（0=不限制；防 ASR/分离卡死）")
    ap.add_argument("--early-stop", action="store_true",
                    help="语言早停: 优先语言命中即停, 未命中补跑另一语言 (命中率不变)")
    ap.add_argument("--energy-gate", action="store_true",
                    help="能量法保守版: 判单人→先降噪整体匹配, 未命中再各轨")
    args = ap.parse_args()

    data_root = r"F:/挑杯资料/datasetA"
    cfg = PipelineConfig(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                      args.config))
    print(f"配置: {args.config} | 降噪 model_dir: {cfg.denoise.get('deepfilternet3', {}).get('model_dir')}", flush=True)
    pipe = VoicePipeline(cfg)
    # 省时优化开关 (默认 False, 通过命令行覆盖)
    if args.early_stop:
        pipe.kws_early_stop = True
    if args.energy_gate:
        pipe.kws_energy_gate = True
    print(f"优化开关: early_stop={pipe.kws_early_stop} energy_gate={pipe.kws_energy_gate}", flush=True)
    pipe.load_models(load_kws_only=True)  # 轻量加载: 跳过声纹/Paraformer (6GB 显存保护)
    sr = cfg.audio.get("sample_rate", 16000)

    samples = []
    for split in ["pos", "neg"]:
        with open(os.path.join(data_root, f"{split}.jsonl"), encoding="utf-8") as f:
            samples += [dict(s, split=split) for s in
                        (json.loads(l) for l in f if l.strip())]

    stats = {"total": 0, "en": 0, "zh": 0,
             "para_hit": 0,       # 双模式直接命中
             "sep_hit": 0,        # 盲分离选轨命中
             "sep_single_skip": 0,  # 能量法判单人跳过
             "sep_fail": 0,       # 盲分离后仍未命中（降噪整体兜底）
             "no_match": 0}       # 未命中兜底
    sep_hit_ids = []
    if args.limit > 0:
        samples = samples[:args.limit]
    resume = args.resume
    # 断点续跑: 保留已统计部分, 从 resume 处继续
    if resume > 0:
        samples = samples[resume:]
    for i0, s in enumerate(samples):
        i = i0 + resume
        kws_path = os.path.join(data_root, s["唤醒音频"])
        kws_text = s.get("唤醒文本")
        audio, _ = load_wav(kws_path, sr)
        # V17: 匹配用原始音频 (91.35% 方案), 盲分离用降噪后音频; 直接复用 _match_kws 保持一致
        denoised = audio if args.no_denoise else pipe.denoiser.denoise(audio, sr)
        is_en = any(c.isalpha() and ord(c) < 128 for c in (kws_text or ""))
        if is_en:
            stats["en"] += 1
        else:
            stats["zh"] += 1
        # 单样本超时保护: 用线程跑 _match_kws, 超时则记兜底 (防 ASR/分离卡死)
        if args.timeout > 0:
            import threading
            res_box = {}
            def _run():
                try:
                    res_box["val"] = pipe._match_kws(audio, kws_text, sr, sep_audio=denoised)
                except Exception as ex:
                    res_box["err"] = ex
            th = threading.Thread(target=_run, daemon=True)
            th.start()
            th.join(timeout=args.timeout)
            if th.is_alive():
                print(f"  [超时跳过] idx={i} kw={kws_text!r} ({args.timeout}s 未完成)", flush=True)
                matched, kws_meta = False, {}
            elif "err" in res_box:
                print(f"  [异常] idx={i} kw={kws_text!r}: {res_box['err']}", flush=True)
                matched, kws_meta = False, {}
            else:
                _, kws_meta, matched = res_box["val"]
        else:
            try:
                _, kws_meta, matched = pipe._match_kws(audio, kws_text, sr, sep_audio=denoised)
            except Exception:
                matched = False
                kws_meta = {}
        if matched:
            stats["para_hit" if not kws_meta.get("kws_separated") else "sep_hit"] += 1
            if kws_meta.get("kws_separated"):
                sep_hit_ids.append(s["id"])
        else:
            if kws_meta.get("kws_separated"):
                stats["sep_fail"] += 1
            else:
                stats["no_match"] += 1
        stats["total"] += 1
        if (i + 1) % 200 == 0:
            print(f"  进度 {i+1}/{len(samples)}: 命中 {stats['para_hit']+stats['sep_hit']} "
                  f"({(stats['para_hit']+stats['sep_hit'])/(i+1)*100:.1f}%)", flush=True)

    n = stats["total"]
    hit = stats["para_hit"] + stats["sep_hit"]
    print("\n========== kws 唤醒词命中率统计 ==========")
    print(f"总样本: {n}（中文 {stats['zh']} / 英文 {stats['en']}）")
    print(f"① 直接命中（Paraformer/英文Nano）: {stats['para_hit']} ({stats['para_hit']/n*100:.2f}%)")
    print(f"② 盲分离选轨补中: {stats['sep_hit']} ({stats['sep_hit']/n*100:.2f}%)")
    print(f"③ 能量法判单人跳过: {stats['sep_single_skip']} ({stats['sep_single_skip']/n*100:.2f}%)")
    print(f"④ 兜底（降噪整体/未匹配）: {stats['sep_fail']+stats['no_match']} ({(stats['sep_fail']+stats['no_match'])/n*100:.2f}%)")
    print(f"★ 最终总命中率: {hit}/{n} = {hit/n*100:.2f}%")
    print(f"   未命中数: {n-hit}")
    print("===========================================")
    json.dump({"stats": stats, "n": n, "final_hit_rate": hit/n,
               "config": args.config, "model_dir": cfg.denoise.get('deepfilternet3', {}).get('model_dir'),
               "no_recheck": args.no_recheck, "no_denoise": args.no_denoise}, open(
        f"F:/龙虾/2026-07-18-13-57-00/kws_hit_stats_{args.tag}.json", "w", encoding="utf-8"),
        ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
