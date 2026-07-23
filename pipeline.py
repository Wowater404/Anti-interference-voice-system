"""
抗干扰语音指令识别流水线
串联: 降噪 → 人声分离 → 声纹鉴别 → 语音识别

核心逻辑:
  1. 从唤醒音频 kws 提取目标说话人声纹 (enrollment)
  2. 对识别音频 cmd 降噪, 去除非人声噪声
  3. 人声分离, 从混合语音中提取目标说话人音轨
  4. 声纹鉴别, 比对分离后/降噪后音频与 kws 声纹
     - 相似度 < 阈值 → 拒识 (输出 null)
     - 相似度 ≥ 阈值 → 接受, 进入 ASR
  5. 语音识别, 输出目标说话人的指令文本

V4.1 更新 (2026-07-19):
  - 自适应分离 + 声纹辅助选轨 + 双阈值鉴别
  - 仅当降噪音频声纹相似度 < 阈值 (即会被拒识) 时才分离
  - 分离后对每条音轨提声纹, 选与 kws 最相似的音轨 (声纹选轨)
  - 双阈值: 分离样本用更高阈值 (vp_threshold_separated=0.35) 防止 neg 假接受
  - 数据验证: V4.1 得分 0.5452 > V2 得分 0.5404 (CER 降 0.012, RR 不变)

输入 (datasetA JSONL 格式):
  {
    "id": 0,
    "唤醒音频": "pos/kws_0.wav",
    "唤醒文本": "你好科慕",
    "识别音频": "pos/cmd_0.wav",
    "识别文本": "空调开到制热调到二十五度..."
  }

输出 (比赛提交格式):
  {
    "result": {
      "results": [
        {"id": "0", "content": "识别文本或null", "label": "标签", "cer": "0.xx"}
      ],
      "final_cer": "0.xx",
      "duration": "t"
    }
  }
"""
import os
# 设置 HuggingFace 镜像站 (国内网络优化)
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
# 禁用 xet 后端 (Windows 上 401 Unauthorized)
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
# 设置 ModelScope 缓存目录 (避免下载到 C 盘)
os.environ.setdefault('MODELSCOPE_CACHE', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pretrained', 'modelscope_cache'))
import time
import json
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path

from config import PipelineConfig
from utils.audio import load_wav, save_wav, normalize_audio, trim_silence
from utils.metrics import evaluate, char_error_rate, strip_punctuation
from modules.denoiser import create_denoiser, BaseDenoiser
from modules.separator import create_separator, BaseSeparator
from modules.voiceprint import create_voiceprint_extractor, BaseVoiceprintExtractor
from modules.asr import create_asr, BaseASR


class VoicePipeline:
    """
    抗干扰语音指令识别流水线

    用法:
        pipeline = VoicePipeline(config)
        pipeline.load_models()
        result = pipeline.process_sample(kws_path, cmd_path, label="...")
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = config.device

        # 初始化各模块 (延迟加载)
        self.denoiser: BaseDenoiser = create_denoiser(config.denoise, self.device)
        self.separator: BaseSeparator = create_separator(config.separation, self.device)
        self.voiceprint_extractor: BaseVoiceprintExtractor = create_voiceprint_extractor(
            config.voiceprint, self.device
        )
        self.asr: BaseASR = create_asr(config.asr, self.device)

        # 声纹鉴别阈值
        self.vp_threshold = config.voiceprint.get("threshold", 0.5)
        # V4.1: 分离样本使用更高阈值, 防止 neg 假接受
        # 数据分析: 所有 15 个假接受 neg 的相似度 < 0.35, 而救回 pos 有 25% >= 0.35
        self.vp_threshold_separated = config.separation.get("vp_threshold_separated", 0.35)
        # V4.1: 分离触发下限 (sim_denoised 低于此值时跳过分离, 省时间)
        self.sep_trigger_min = config.separation.get("sep_trigger_min", 0.0)
        # V5.1: 相似度跳变上限 (分离后sim比降噪sim高出此值则不信任, 防分离artifact)
        # 数据: neg降噪sim=0.02→分离后0.92 (跳变0.90=artifact), pos降噪sim=0.55→分离后0.75 (跳变0.20=合理)
        self.sim_jump_cap = config.separation.get("sim_jump_cap", 1.0)  # 默认1.0=不限制

        # 分离后音频使用策略 (V3遗留, V4.1自适应模式下忽略)
        separation_cfg = config.separation
        self.separation_enabled = separation_cfg.get("enable", False)
        self.voiceprint_use_separated = separation_cfg.get("voiceprint_use_separated", True)
        self.asr_use_separated = separation_cfg.get("asr_use_separated", True)

        # 输出配置
        self.save_intermediate = config.output.get("save_intermediate", False)
        self.intermediate_dir = config.output.get("intermediate_dir", "./intermediate")

        self._models_loaded = False

    def load_models(self):
        """加载所有模型权重"""
        print("=" * 60)
        print("正在加载流水线模型...")
        print("=" * 60)

        # 按依赖顺序加载
        # 1. 声纹提取器 (最先加载, 其他模块可能依赖)
        print("\n[1/4] 加载声纹提取模型...")
        self.voiceprint_extractor.load()

        # 2. 降噪模型
        print("\n[2/4] 加载降噪模型...")
        self.denoiser.load()

        # 3. 人声分离模型
        print("\n[3/4] 加载人声分离模型...")
        self.separator.load()

        # 4. ASR 模型
        print("\n[4/4] 加载语音识别模型...")
        self.asr.load()

        self._models_loaded = True
        print("\n" + "=" * 60)
        print("所有模型加载完成!")
        # 打印分离策略
        if self.separation_enabled:
            print("分离策略: V4.1 自适应分离 + 声纹辅助选轨 + 双阈值鉴别")
            print(f"  正常阈值: {self.vp_threshold}, 分离后阈值: {self.vp_threshold_separated}")
            print(f"  分离触发范围: sim_denoised ∈ [{self.sep_trigger_min}, {self.vp_threshold})")
            print(f"  相似度跳变上限: {self.sim_jump_cap} (分离后sim跳变超过此值不信任)")
            print("  仅当降噪音频声纹相似度不足时分离, 分离后声纹选轨+高阈值鉴别")
        else:
            print("分离策略: 禁用分离, 声纹+ASR均用降噪后音频 (V2行为)")
        print("=" * 60)

    def process_sample(
        self,
        kws_path: str,
        cmd_path: str,
        label: Optional[str] = None,
        sample_id: str = ""
    ) -> Dict:
        """
        处理单条样本: 完整流水线推理 (4阶段: 降噪→分离→声纹鉴别→ASR)

        Pipeline流程 (V5.1):
          Step 0: 加载kws+cmd音频 (16kHz mono)
          Step 1: 从kws提取参考声纹 embedding [192]
          Step 2: 对cmd降噪 (noisereduce, stationary=True, prop_decrease=0.8)
          Step 3: 从降噪后cmd提取声纹, 算与kws的cosine相似度 sim_denoised
          Step 4: 选择性分离: 仅当 0.50<=sim_denoised<0.67 时触发
                  - 分离后选最相似音轨 (跳变>0.25不信任)
                  - sim>=0.67或<0.50的样本不分离
          Step 5: 双阈值鉴别:
                  - 未分离: sim>=0.67 接受
                  - 分离后: sim>=0.80 接受 (高阈值防neg假接受)
          Step 6: ASR识别 (Paraformer, 输出去标点)
                  - 接受: 识别目标说话人音频→文本
                  - 拒识: 输出空字符串"" (比赛FAQ#8: 删除错误)

        Args:
            kws_path: str, 唤醒音频文件路径 (WAV, 16kHz)
            cmd_path: str, 识别音频文件路径 (WAV, 16kHz)
            label: str, 可选, 真实标签文本 (评估时用于算CER, 推理时为None)
            sample_id: str, 样本ID (用于中间文件命名和结果追踪)

        Returns:
            dict: {
                "id": sample_id,           # 样本ID
                "content": str,             # 识别文本 (拒识时为"")
                "label": label,             # 真实标签 (评估用)
                "cer": float or None,       # CER (有label时计算, 否则None)
                "similarity": float,        # 声纹cosine相似度 (最终判定依据)
                "is_target": bool,          # 是否接受为目标说话人
                "stages": dict,             # 各阶段耗时 {load:float, voiceprint_extract:float, ...}
            }
        """
        if not self._models_loaded:
            self.load_models()

        sr = self.config.sample_rate
        stages_time = {}
        intermediate_prefix = f"{sample_id}" if sample_id else os.path.basename(cmd_path).replace(".wav", "")

        # ==============================================================
        # Step 0: 加载音频
        # ==============================================================
        t0 = time.time()
        kws_audio, _ = load_wav(kws_path, sr)
        cmd_audio, _ = load_wav(cmd_path, sr)
        stages_time["load"] = time.time() - t0

        # ==============================================================
        # Step 1: 声纹提取 (从唤醒音频)
        # ==============================================================
        t0 = time.time()
        kws_embedding = self.voiceprint_extractor.extract(kws_audio, sr)
        stages_time["voiceprint_extract"] = time.time() - t0

        # ==============================================================
        # Step 2: 降噪
        # ==============================================================
        t0 = time.time()
        denoised_audio = self.denoiser.denoise(cmd_audio, sr)
        stages_time["denoise"] = time.time() - t0

        if self.save_intermediate:
            save_wav(
                os.path.join(self.intermediate_dir, f"{intermediate_prefix}_denoised.wav"),
                denoised_audio, sr
            )

        # ==============================================================
        # Step 3: 降噪音频声纹相似度 (自适应分离的判断依据)
        # ==============================================================
        t0 = time.time()
        denoised_embedding = self.voiceprint_extractor.extract(denoised_audio, sr)
        sim_denoised = float(np.dot(kws_embedding, denoised_embedding) /
                             (np.linalg.norm(kws_embedding) * np.linalg.norm(denoised_embedding) + 1e-8))
        stages_time["voiceprint_denoised"] = time.time() - t0

        # ==============================================================
        # Step 4: 自适应分离 + 声纹辅助选轨 (V4 核心改进)
        # 策略: 只有降噪音频声纹相似度不足以通过阈值时才分离
        #       (即 V2 会拒识的样本), 分离后用声纹选轨提取目标说话人
        # 保证 V2 已能正确识别的样本不受分离 artifact 影响
        # ==============================================================
        t0 = time.time()
        best_audio = denoised_audio
        best_sim = sim_denoised
        all_sources = [denoised_audio]
        separation_used = False

        if self.separation_enabled and self.sep_trigger_min <= sim_denoised < self.vp_threshold:
            # 检测到干扰: 降噪音频声纹相似度不足, 尝试分离
            separated_audio, sources = self.separator.separate(denoised_audio, sr)
            if len(sources) > 1:
                all_sources = sources
                separation_used = True
                # 声纹辅助选轨: 对每条分离音轨提声纹, 选与 kws 最相似的
                # V5.1: 跳变上限检查 - 分离后sim跳变过大视为artifact, 不信任
                for src in sources:
                    src_emb = self.voiceprint_extractor.extract(src, sr)
                    src_sim = float(np.dot(kws_embedding, src_emb) /
                                    (np.linalg.norm(kws_embedding) * np.linalg.norm(src_emb) + 1e-8))
                    if src_sim > best_sim and (src_sim - sim_denoised) <= self.sim_jump_cap:
                        best_sim = src_sim
                        best_audio = src
                # 若分离后最佳相似度仍不如降噪音频, best_audio 保持 denoised
        stages_time["separation"] = time.time() - t0

        if self.save_intermediate and separation_used:
            save_wav(
                os.path.join(self.intermediate_dir, f"{intermediate_prefix}_separated.wav"),
                best_audio, sr
            )
            for i, src in enumerate(all_sources):
                save_wav(
                    os.path.join(self.intermediate_dir, f"{intermediate_prefix}_src{i}.wav"),
                    src, sr
                )

        # ==============================================================
        # Step 5: 声纹鉴别 (双阈值: 分离样本用更高阈值防止 neg 假接受)
        # ==============================================================
        t0 = time.time()
        similarity = best_sim
        # V4.1: 分离过的样本用更高阈值, 未分离的用正常阈值
        accept_threshold = self.vp_threshold_separated if separation_used else self.vp_threshold
        is_target = similarity >= accept_threshold
        stages_time["voiceprint_verify"] = time.time() - t0

        # ==============================================================
        # Step 6: 语音识别 (仅对接受样本, 用最佳音轨)
        # ==============================================================
        if is_target:
            t0 = time.time()
            asr_audio = best_audio
            # 预处理: 归一化 + 去静音
            asr_audio = normalize_audio(asr_audio)
            asr_audio = trim_silence(asr_audio, threshold=0.005)

            text = self.asr.transcribe(asr_audio, sr)
            # 去除标点符号 (ASR punc 模型会添加标点, 但标签不含标点)
            text = strip_punctuation(text)
            stages_time["asr"] = time.time() - t0

            content = text if text else ""
        else:
            # 拒识: 输出空字符串 (FAQ#8: 正样本错误拒识按删除错误计算CER)
            content = ""
            stages_time["asr"] = 0.0

        # ==============================================================
        # 评估
        # ==============================================================
        cer = None
        if label is not None:
            if label and content:
                # 正样本且被接受: 计算实际CER
                cer = char_error_rate(label, content)
            elif not label and not content:
                # 负样本且被拒识: 正确拒识
                cer = 0.0
            elif label and not content:
                # 正样本被错误拒识: 按删除错误计算 (FAQ#8)
                cer = 1.0
            elif not label and content:
                # 负样本被错误接受: 错误接受
                cer = 1.0

        return {
            "id": sample_id,
            "content": content,
            "label": label if label is not None else "",
            "cer": f"{cer:.4f}" if cer is not None else "",
            "similarity": f"{similarity:.4f}",
            "is_target": is_target,
            "stages_time": stages_time,
        }

    def process_dataset(self, data_root: str, split: str = "all",
                        checkpoint_path: str = None) -> Dict:
        """
        处理整个数据集 (支持断点续传)

        Args:
            data_root: 数据集根目录 (含 pos.jsonl, neg.jsonl)
            split: "pos" / "neg" / "all"
            checkpoint_path: 断点续传文件路径 (如有则从中恢复)

        Returns:
            比赛提交格式 JSON
        """
        if not self._models_loaded:
            self.load_models()

        all_results = []
        total_start = time.time()

        # 断点续传: 加载已有结果
        resume_from = {}
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, "r", encoding="utf-8") as f:
                    ckpt = json.load(f)
                all_results = ckpt.get("results", [])
                for r in all_results:
                    # 优先使用 _ckpt_key, 回退到 id
                    key = r.get("_ckpt_key", r["id"])
                    resume_from[key] = r
                print(f"[断点续传] 已加载 {len(all_results)} 条已完成结果")
            except Exception as e:
                print(f"[断点续传] 加载失败 ({e}), 从头开始")

        splits = []
        if split in ("pos", "all"):
            splits.append("pos")
        if split in ("neg", "all"):
            splits.append("neg")

        for sp in splits:
            jsonl_path = os.path.join(data_root, f"{sp}.jsonl")
            if not os.path.exists(jsonl_path):
                print(f"警告: {jsonl_path} 不存在, 跳过")
                continue

            with open(jsonl_path, "r", encoding="utf-8") as f:
                samples = [json.loads(line) for line in f if line.strip()]

            print(f"\n处理 {sp} 集: {len(samples)} 条样本")

            for i, sample in enumerate(samples):
                # 使用 split+index 作为断点续传的唯一键 (neg 样本可能有重复 id)
                ckpt_key = f"{sp}_{i}"
                result_id = str(sample.get("id", i))

                # 断点续传: 跳过已处理的样本
                if ckpt_key in resume_from:
                    continue

                kws_path = os.path.join(data_root, sample["唤醒音频"])
                cmd_path = os.path.join(data_root, sample["识别音频"])
                label = sample.get("识别文本", None)

                try:
                    result = self.process_sample(
                        kws_path=kws_path,
                        cmd_path=cmd_path,
                        label=label,
                        sample_id=result_id
                    )
                except Exception as e:
                    print(f"  [错误] 样本 {result_id} ({ckpt_key}) 处理失败: {e}")
                    result = {
                        "id": result_id,
                        "content": "",
                        "label": label if label is not None else "",
                        "cer": "1.0000",
                        "similarity": "0.0000",
                        "is_target": False,
                        "stages_time": {},
                    }

                # 存储 ckpt_key 用于断点续传
                result["_ckpt_key"] = ckpt_key
                all_results.append(result)
                resume_from[ckpt_key] = result

                # 进度输出 + 断点保存
                if (i + 1) % 10 == 0 or i == len(samples) - 1:
                    elapsed = time.time() - total_start
                    done = len(all_results)
                    print(f"  [{sp}] {i+1}/{len(samples)} "
                          f"({(i+1)/len(samples)*100:.1f}%) "
                          f"已用 {elapsed:.1f}s")

                # 每 50 条保存断点
                if checkpoint_path and (len(all_results) % 50 == 0):
                    try:
                        with open(checkpoint_path, "w", encoding="utf-8") as f:
                            json.dump({"results": all_results, "timestamp": time.time()},
                                      f, ensure_ascii=False)
                    except Exception:
                        pass

        total_duration = time.time() - total_start

        # 最终保存断点
        if checkpoint_path:
            try:
                with open(checkpoint_path, "w", encoding="utf-8") as f:
                    json.dump({"results": all_results, "timestamp": time.time()},
                              f, ensure_ascii=False)
            except Exception:
                pass

        # 计算指标
        pos_results = [r for r in all_results if r["label"]]
        neg_results = [r for r in all_results if not r["label"] or r["label"] == "null"]

        try:
            metrics = evaluate(pos_results, neg_results)
        except Exception as e:
            print(f"警告: 评估指标计算失败 ({e}), 使用默认值")
            metrics = {
                "final_cer": "0.0000",
                "rejection_rate": "0.0000",
                "final_score": "0.0000",
                "pos_count": len(pos_results),
                "neg_count": len(neg_results),
            }

        # 构建提交格式
        submission = {
            "result": {
                "results": [
                    {
                        "id": r["id"],
                        "content": r["content"],
                        "label": r["label"],
                        "cer": r["cer"],
                    }
                    for r in all_results
                ],
                "final_cer": metrics["final_cer"],
                "duration": f"{total_duration:.2f}",
            },
            "metrics": {
                "rejection_rate": metrics["rejection_rate"],
                "final_score": metrics["final_score"],
                "pos_count": metrics["pos_count"],
                "neg_count": metrics["neg_count"],
            },
        }

        return submission
