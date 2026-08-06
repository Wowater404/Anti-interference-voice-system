"""
抗干扰语音指令识别流水线 - 推理入口脚本

用法:
  # 处理整个 datasetA
  python run_inference.py --data_root "F:/挑杯资料/datasetA" --split all

  # 仅处理正样本 (测试 CER)
  python run_inference.py --data_root "F:/挑杯资料/datasetA" --split pos

  # 仅处理负样本 (测试 RR)
  python run_inference.py --data_root "F:/挑杯资料/datasetA" --split neg

  # 处理单条样本
  python run_inference.py --kws "F:/挑杯资料/datasetA/pos/kws_0.wav" \
                          --cmd "F:/挑杯资料/datasetA/pos/cmd_0.wav" \
                          --label "空调开到制热调到二十五度风量调到百分之三十"

  # 指定配置文件
  python run_inference.py --config configs/default.yaml --data_root "F:/挑杯资料/datasetA"
"""
import os
import sys
import json
import argparse
from pathlib import Path

# 添加项目根目录到 path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config import PipelineConfig
from pipeline import VoicePipeline


def main():
    parser = argparse.ArgumentParser(
        description="抗干扰语音指令识别流水线推理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 数据输入
    parser.add_argument("--data_root", type=str, default=None,
                        help="数据集根目录 (含 pos.jsonl, neg.jsonl)")
    parser.add_argument("--split", type=str, default="all",
                        choices=["pos", "neg", "all"],
                        help="处理哪个数据集")

    # 单样本模式
    parser.add_argument("--kws", type=str, default=None,
                        help="唤醒音频路径 (单样本模式)")
    parser.add_argument("--cmd", type=str, default=None,
                        help="识别音频路径 (单样本模式)")
    parser.add_argument("--label", type=str, default=None,
                        help="真实标签 (评估用)")

    # 配置
    parser.add_argument("--config", type=str, default=None,
                        help="配置文件路径 (默认 configs/default.yaml)")
    parser.add_argument("--output", type=str, default=None,
                        help="输出结果文件路径")
    parser.add_argument("--text-output", type=str, default=None,
                        help="另存为可用记事本打开的 UTF-8 JSON 文本路径")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="断点续传文件路径 (自动恢复已处理样本)")

    # 设备
    parser.add_argument("--device", type=str, default=None,
                        choices=["cuda", "cpu"],
                        help="推理设备 (覆盖配置文件)")
    parser.add_argument("--separator-model", "--separator_model", type=str, default=None,
                        help=("覆盖人声分离模型，例如 sepformer16k、sepformer、"
                              "spex_plus、passthrough"))
    parser.add_argument("--sep-trigger-min", type=float, default=None,
                        help="覆盖分离触发相似度下限")
    parser.add_argument("--vp-threshold", type=float, default=None,
                        help="覆盖未分离样本的声纹接受阈值")
    parser.add_argument("--vp-threshold-separated", type=float, default=None,
                        help="覆盖分离样本的声纹接受阈值")
    parser.add_argument("--sim-jump-cap", type=float, default=None,
                        help="覆盖分离后声纹相似度允许的最大跳变量")

    args = parser.parse_args()

    # 加载配置
    config_path = args.config or os.path.join(PROJECT_ROOT, "configs", "default.yaml")
    config = PipelineConfig(config_path)

    # 覆盖设备
    if args.device:
        config._cfg["device"] = args.device
    if args.separator_model:
        separation_cfg = config._cfg.setdefault("separation", {})
        separation_cfg["model"] = args.separator_model
        separation_cfg["enable"] = args.separator_model not in ("passthrough", "none")
    if args.sep_trigger_min is not None:
        config._cfg.setdefault("separation", {})["sep_trigger_min"] = args.sep_trigger_min
    if args.vp_threshold is not None:
        config._cfg.setdefault("voiceprint", {})["threshold"] = args.vp_threshold
    if args.vp_threshold_separated is not None:
        config._cfg.setdefault("separation", {})[
            "vp_threshold_separated"
        ] = args.vp_threshold_separated
    if args.sim_jump_cap is not None:
        config._cfg.setdefault("separation", {})["sim_jump_cap"] = args.sim_jump_cap

    print(f"当前分离模型: {config.separation.get('model', 'passthrough')}")

    # 创建流水线
    pipeline = VoicePipeline(config)
    pipeline.load_models()

    # 输出路径
    output_path = args.output or os.path.join(
        config.output.get("result_dir", os.path.join(PROJECT_ROOT, "results")),
        "submission.json"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ================================================================
    # 模式1: 单样本推理
    # ================================================================
    if args.kws and args.cmd:
        print("\n" + "=" * 60)
        print("单样本推理模式")
        print("=" * 60)
        print(f"  唤醒音频: {args.kws}")
        print(f"  识别音频: {args.cmd}")
        print(f"  标签: {args.label}")

        result = pipeline.process_sample(
            kws_path=args.kws,
            cmd_path=args.cmd,
            label=args.label,
            sample_id="single"
        )

        print("\n" + "=" * 60)
        print("推理结果:")
        print(f"  识别文本: {result['content']}")
        print(f"  声纹相似度: {result['similarity']}")
        print(f"  是否目标说话人: {result['is_target']}")
        if result.get("cer"):
            print(f"  CER: {result['cer']}")
        print("\n各阶段耗时:")
        for stage, t in result.get("stages_time", {}).items():
            print(f"  {stage}: {t:.3f}s")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存至: {output_path}")

    # ================================================================
    # 模式2: 数据集批量推理
    # ================================================================
    elif args.data_root:
        print("\n" + "=" * 60)
        print(f"数据集批量推理模式 (split={args.split})")
        print(f"  数据根目录: {args.data_root}")
        print("=" * 60)

        # 断点续传路径
        ckpt_path = args.checkpoint or os.path.join(
            os.path.dirname(output_path), "checkpoint.json"
        )

        if pipeline.is_ensemble:
            submission = pipeline.process_dataset_ensemble(
                args.data_root, args.split, checkpoint_path=ckpt_path
            )
        else:
            submission = pipeline.process_dataset(
                args.data_root, args.split, checkpoint_path=ckpt_path
            )

        # 打印汇总结果
        print("\n" + "=" * 60)
        print("推理完成! 汇总结果:")
        print("=" * 60)
        print(f"  正样本数: {submission['metrics']['pos_count']}")
        print(f"  负样本数: {submission['metrics']['neg_count']}")
        print(f"  最终 CER: {submission['result']['final_cer']}")
        print(f"  拒识率 RR: {submission['metrics']['rejection_rate']}")
        print(f"  综合得分: {submission['metrics']['final_score']}")
        print(f"  总推理时间: {submission['result']['duration']}s")

        # 官方提交结构的顶层只能包含 result；metrics 仅供终端汇总使用。
        official_submission = {"result": submission["result"]}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(official_submission, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存至: {output_path}")

        text_output_path = args.text_output or str(Path(output_path).with_suffix(".txt"))
        with open(text_output_path, "w", encoding="utf-8") as f:
            json.dump(official_submission, f, ensure_ascii=False, indent=2)
        print(f"记事本文件已保存至: {text_output_path}")

    else:
        parser.print_help()
        print("\n错误: 请指定 --data_root 或 --kws + --cmd")


if __name__ == "__main__":
    main()
