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
# [重要] 必须先 import torch，再写 os.environ：
# torch 2.x 在 Windows 上，import torch 之前任何 os.environ 写入/os.chdir 会
# 导致 torch._C 加载时 access violation 段错误。已验证：先 import torch 则稳定。
import torch  # noqa: F401
# [2026-08-15] cuDNN 加载修复: 把 conda 环境 Library/bin 注入 PATH,
# 否则 cudnn64_9.dll 的子 DLL 找不到 → "Invalid handle: Cannot load symbol cudnnGetVersion"。
# cudnn 延迟加载, import torch 后注入仍有效。
import sys as _sys
_lib_bin = os.path.abspath(os.path.join(os.path.dirname(_sys.executable), 'Library', 'bin'))
if os.path.isdir(_lib_bin):
    os.environ['PATH'] = _lib_bin + os.pathsep + os.environ.get('PATH', '')
# Hugging Face 下载端点由环境变量 HF_ENDPOINT 控制，不在代码中强制覆盖。
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
from utils.digit_normalize import normalize_digits
from modules.denoiser import create_denoiser, BaseDenoiser
from modules.separator import (
    create_separator, BaseSeparator, PassThroughSeparator, SepFormer16kSeparator,
)
from modules.voiceprint import (
    create_voiceprint_extractor, BaseVoiceprintExtractor,
    EnsembleVoiceprintExtractor,
)
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
        # enable=false 时统一降级为 passthrough (日志/检查口径一致)
        if config.separation.get("enable", True):
            self.separator_model_name = config.separation.get("model", "passthrough")
        else:
            self.separator_model_name = "passthrough"

        # 初始化各模块 (延迟加载)
        self.denoiser: BaseDenoiser = create_denoiser(config.denoise, self.device)
        self.separator: BaseSeparator = create_separator(config.separation, self.device)
        # V9: kws 自适应盲分离 (降噪 → 唤醒词匹配 → 不够才盲分离)
        # kws 自己就是 enrollment, 没有参考可用, 只能用盲分离 SepFormer16k;
        # 而 cmd 侧的 SpEx+ 是目标提取, 需要参考, 两者不对称。
        self.kws_enable = config.separation.get("kws_enable", True)
        self.kws_trigger_sim = config.separation.get("kws_trigger_sim", 0.5)
        # [2026-08-15] 能量法预检: 盲分离两轨中第二轨能量占比 < 此阈值判为单人
        # (盲分离对单人 kws 产生的是伪影轨, 能量显著低于主轨, 无需各轨匹配)
        self.kws_sep_energy_th = config.separation.get("kws_sep_energy_th", 0.15)
        # [2026-08-15] 指令识别"先轻后重": 指令词表 (从全量 label 标定, 含≥1词判Paraformer输出可靠)
        self._instruct_words = [
            "什么", "调到", "吃什", "模式", "打开", "空调", "播放", "二十", "风速", "到二",
            "关闭", "开启", "度调", "温度", "开到", "速调", "把温", "食物", "适合", "灯光",
            "关掉", "哪些", "百分", "分之", "到最", "些食", "合吃", "到百", "窗帘", "一点",
            "我要", "我想", "想听", "客厅", "防直", "直吹", "到十", "吃哪", "音乐", "CO",
        ]
        _sep16k_cfg = config.separation.get("sepformer16k", {})
        self.kws_separator = SepFormer16kSeparator(
            device=self.device,
            max_speakers=2,
            huggingface_repo=_sep16k_cfg.get(
                "huggingface_repo", "speechbrain/sepformer-whamr16k"
            ),
        )
        self.voiceprint_extractor = create_voiceprint_extractor(
            config.voiceprint, self.device
        )
        self.asr: BaseASR = create_asr(config.asr, self.device)
        # [2026-08-15] kws 唤醒词匹配专用 ASR (Fun-ASR-Nano):
        # 91.35% 命中率是 Nano 专属 (Paraformer 中文64.6%/英文25% 不达标)。
        # 混合策略: 中文先 Paraformer 快匹配 → 未命中 Nano 复核; 英文直接 Nano。
        # 由 load_models 按 config.asr.kws_asr 加载。
        self.kws_asr: Optional[BaseASR] = None

        # 是否为三模型 Z-score 集成模式
        self.is_ensemble = isinstance(self.voiceprint_extractor, EnsembleVoiceprintExtractor)

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
        if self.config.voiceprint.get("enable", True):
            if self.is_ensemble:
                # 集成模式: 检查三个子模型
                ens = self.voiceprint_extractor
                if ens.cam.model is None or ens.eres.model is None or ens.rnet.model is None:
                    raise RuntimeError(
                        "三模型集成声纹加载失败 (CAM++/ERes2NetV2/ResNetSE),"
                        "已停止流水线"
                    )
            elif self.voiceprint_extractor.model is None:
                raise RuntimeError(
                    f"声纹模型 {self.config.voiceprint.get('model')} 加载失败，"
                    "已停止流水线，防止使用零向量产生无效结果"
                )

        # 2. 降噪模型
        print("\n[2/4] 加载降噪模型...")
        self.denoiser.load()

        # 3. 人声分离模型
        print("\n[3/4] 加载人声分离模型...")
        print(f"  当前分离模型: {self.separator_model_name}")
        self.separator.load()
        if (
            self.separation_enabled
            and self.separator_model_name not in ("passthrough", "none")
            and not isinstance(self.separator, PassThroughSeparator)
            and self.separator.model is None
        ):
            raise RuntimeError(
                f"分离模型 {self.separator_model_name} 加载失败，"
                "已停止流水线，防止静默退化为直通"
            )

        # 3.5. kws 盲分离模型 (V9: 用于 kws 自适应处理, 无参考盲分离)
        if self.kws_enable:
            print("\n[kws] 加载 kws 盲分离模型 (SepFormer16k)...")
            self.kws_separator.load()
        else:
            print("\n[kws] kws 盲分离已禁用 (kws_enable=false)")

        # 4. ASR 模型
        print("\n[4/4] 加载语音识别模型...")
        self.asr.load()
        if self.config.asr.get("enable", True) and self.asr.model is None:
            raise RuntimeError(
                f"ASR 模型 {self.config.asr.get('model')} 加载失败，"
                "已停止流水线，防止输出空识别结果"
            )

        # 4b. kws 中文快匹配专用 ASR (Paraformer) [2026-08-15 21:20 V13]
        # 角色互换: 主 asr=Nano (指令识别+英文匹配), kws_asr=Paraformer (中文快匹配 0.26s)
        _kws_asr_cfg = self.config.asr.get("kws_asr") or {}
        if _kws_asr_cfg.get("enable", True):
            from modules.asr import create_asr as _create_asr
            _kws_model = _kws_asr_cfg.get("model", "paraformer")
            if _kws_model == "paraformer":
                _kws_cfg = {
                    "model": "paraformer",
                    "paraformer": {
                        "model_name": _kws_asr_cfg.get("model_name", "paraformer-zh"),
                        "vad_model": _kws_asr_cfg.get("vad_model", "fsmn-vad"),
                        "punc_model": _kws_asr_cfg.get("punc_model"),
                        "hotwords": _kws_asr_cfg.get("hotwords", ""),
                    },
                }
            else:  # fun_asr_nano
                _kws_cfg = {
                    "model": "fun_asr_nano",
                    "fun_asr_nano": {
                        "model_dir": _kws_asr_cfg.get("model_dir", "FunAudioLLM/Fun-ASR-Nano-2512"),
                        "language": _kws_asr_cfg.get("language", "中文"),
                        "device": _kws_asr_cfg.get("device", "cuda:0"),
                        "hotwords": None,
                    },
                }
            self.kws_asr = _create_asr(_kws_cfg, self.device)
            self.kws_asr.load()
            if self.kws_asr.model is None:
                print("[kws_asr] 警告: 快匹配 ASR 加载失败, 唤醒词匹配降级为仅主 ASR")
                self.kws_asr = None

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

    # ==============================================================
    # V9: kws 自适应处理辅助方法
    # 思路: 唤醒音频先降噪 → 用 ASR 匹配唤醒文本判断是否够干净
    #       → 不够干净(疑似多人声重叠)才盲分离 → 分离后各自匹配唤醒文本选路
    # ==============================================================
    def _asr_text(self, audio: np.ndarray, sr: int, asr: Optional[BaseASR] = None) -> str:
        """ASR 识别 + 去标点, 返回纯文本 (用于唤醒词匹配)"""
        try:
            m = asr if asr is not None else self.asr
            return strip_punctuation(m.transcribe(audio, sr))
        except Exception:
            return ""

    # [2026-08-15] V12: 唤醒词匹配工具 (91.35% 方案的 to_phonetic + 多策略)
    def _to_phonetic(self, text: str) -> str:
        """中文→无调拼音, 英文→保留字母; 统一小写去空格"""
        if not text:
            return ""
        if any("\u4e00" <= c <= "\u9fff" for c in text):
            try:
                from pypinyin import lazy_pinyin
                return "".join(lazy_pinyin(text)).lower()
            except ImportError:
                return text.lower()
        return text.lower().replace(" ", "")

    def _edit_dist(self, a: str, b: str) -> int:
        """字符级编辑距离 (DP)"""
        r, h = list(a), list(b)
        dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
        for i in range(len(r) + 1):
            dp[i][0] = i
        for j in range(len(h) + 1):
            dp[0][j] = j
        for i in range(1, len(r) + 1):
            for j in range(1, len(h) + 1):
                c = 0 if r[i - 1] == h[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + c)
        return dp[len(r)][len(h)]

    def _wakeword_sim(self, kw: str, hyp: str) -> float:
        """唤醒词匹配相似度 = max(序列相似度, 编辑距离分, 子串覆盖), 范围 [0,1]"""
        kp = self._to_phonetic(kw)
        hp = self._to_phonetic(hyp)
        if not kp or not hp:
            return 0.0
        lev = self._edit_dist(kp, hp)
        s1 = 1 - lev / len(kp)                      # 序列相似度 (分母=kw长度, 1-CER式)
        s2 = 1 - lev / max(len(kp), len(hp))        # 编辑距离分 (分母=较长者)
        s3 = 0.0
        if kp in hp or hp in kp:                     # 子串覆盖
            s3 = min(len(kp), len(hp)) / max(len(kp), len(hp))
        return max(0.0, s1, s2, s3)

    def _match_kws(self, audio: np.ndarray, kws_text: str, sr: int):
        """
        V12 混合唤醒词匹配 (先轻后重):
          英文 → Nano 英文模式直接匹配 (Paraformer 英文必败)
          中文 → Paraformer 快匹配 → 未命中 Nano 复核
          全部未命中 → 盲分离 → 各轨匹配选轨
        返回 (best_audio, meta, matched)
          best_audio: 匹配命中的音频 (选轨结果或降噪整体)
          meta: 处理记录 (kws_separated 等)
          matched: 是否命中唤醒词 (sim >= kws_trigger_sim)
        """
        meta = {"kws_denoised": True, "kws_separated": False}
        is_english = any(c.isalpha() and ord(c) < 128 for c in kws_text)

        def _match(asr, seg, lang=None):
            if lang is not None and asr is not None:
                old = getattr(asr, "language", None)
                try:
                    if old:
                        asr.language = lang
                    hyp = self._asr_text(seg, sr, asr)
                finally:
                    if old:
                        asr.language = old
            else:
                hyp = self._asr_text(seg, sr, asr)
            return self._wakeword_sim(kws_text, hyp)

        # 1. 英文: 主 asr (Nano) 英文模式直接匹配 (Paraformer 英文必败)
        if is_english:
            if _match(self.asr, audio, "英文") >= self.kws_trigger_sim:
                return audio, meta, True

        # 2. 中文: kws_asr (Paraformer) 快匹配 (0.26s, 命中约65%走快路径)
        elif not is_english and self.kws_asr is not None:
            if _match(self.kws_asr, audio) >= self.kws_trigger_sim:
                return audio, meta, True

        # 3. 全部未命中 → 盲分离 → 能量法预检 → 各轨匹配选轨 (选"说唤醒词的轨")
        try:
            _, sources = self.kws_separator.separate(audio, sr)
        except Exception:
            return audio, meta, False
        if not sources or len(sources) <= 1:
            return audio, meta, False
        # [2026-08-15] 能量法预检: 第二轨能量占比低 = 单人 (盲分离伪影),
        # 直接用降噪整体提声纹, 跳过各轨 ASR 匹配 (省 ~0.5s)。
        if self._sep_is_single(sources, self.kws_sep_energy_th):
            return audio, meta, False
        meta["kws_separated"] = True
        best_audio, best_sim = audio, -1.0
        for src in sources:
            # V13: 英文各轨用主 asr (Nano 英文), 中文各轨用 kws_asr (Paraformer)
            if is_english:
                sim = _match(self.asr, src, "英文")
            else:
                sim = _match(self.kws_asr, src) if self.kws_asr is not None else _match(self.asr, src)
            if sim > best_sim:
                best_sim, best_audio = sim, src
        return best_audio, meta, best_sim >= self.kws_trigger_sim

    def _sep_is_single(self, sources, threshold: float = 0.15) -> bool:
        """能量法: 分离两轨中能量较小的轨占比 < threshold 判为单人 (盲分离伪影)。

        依据: 对单人 kws, SepFormer 输出的第二轨是低能量伪影;
              真·双人重叠时两轨能量可比。阈值由 config.separation.kws_sep_energy_th 控制。
        """
        if not sources or len(sources) < 2:
            return True
        e = [float(np.sqrt(np.mean(s.astype(np.float64) ** 2)) + 1e-9) for s in sources]
        ratio = min(e) / max(e)
        return ratio < threshold

    def _instruct_reliable(self, text: str) -> bool:
        """指令识别可靠性判据: content 含 ≥1 个指令词判为可靠。
        标定 (V12 全量 1301 条): 判可靠 64% 实际 CER 0.259; 判不可靠 36% 实际 CER 0.954。
        可靠输出直接用 Paraformer 快路径, 不可靠才 Nano 复核 (兜底 CER)。
        """
        if not text:
            return False
        return sum(1 for w in self._instruct_words if w in text) >= 1

    def _recognize_instruct(self, audio: np.ndarray, sr: int) -> str:
        """指令识别"先轻后重" (2026-08-15):
          kws_asr (Paraformer, 0.26s) 快识别 → 词汇判据可靠则直接用;
          不可靠 → 主 asr (Nano, 0.84s) 复核, 保 CER。
        预期: cmd 平均 0.64*0.26 + 0.36*0.84 ≈ 0.47s, 整体 CER 由 Nano 兜底。
        """
        if self.kws_asr is not None:  # kws_asr 为 Paraformer (快)
            try:
                text = strip_punctuation(self.kws_asr.transcribe(audio, sr))
            except Exception:
                text = ""
            if self._instruct_reliable(text):
                return text
        return strip_punctuation(self.asr.transcribe(audio, sr))

    def _text_sim(self, a: str, b: str) -> float:
        """字符相似度 = 1 - CER, 范围 [0,1]; 空串返回 0"""
        if not a or not b:
            return 0.0
        return max(0.0, 1.0 - char_error_rate(b, a))

    def _process_kws(self, audio: np.ndarray, kws_text: Optional[str], sr: int,
                     ref_embedding: Optional[np.ndarray] = None,
                     ref_emb_all: Optional[tuple] = None):
        """
        kws 处理 (V10 优化版, 2026-08-15):
          降噪 → 盲分离 → 用参考声纹(cmd)选轨

        相比 V9 的改动:
          - 砍掉 ASR 唤醒词匹配 (实测对 <1s 超短唤醒词 0% 通过, 纯浪费)
          - 盲分离无条件执行 (不再用 ASR 判定, 因为判定必然失败)
          - 分离选轨改用「参考声纹(cmd)相似度」, 比 ASR 可靠 (实测 20/20 可分)
          - 选轨时提取的声纹直接返回复用, 避免重复提取

        Args:
            audio: kws 原始音频 [N] float32
            kws_text: 唤醒词文本 (保留接口, V10 不再用于匹配)
            sr: 采样率
            ref_embedding: 参考声纹 (cmd 降噪后的 embedding), 用于选轨
            ref_emb_all: 参考声纹三模型 tuple (可选, 与 ref_embedding 二选一)

        Returns:
            (best_audio, meta, best_embedding, best_emb_all):
              best_audio: 选轨后的 kws 音频 (单人目标声纹)
              meta: 处理记录
              best_embedding: 选轨声纹 (复用, 避免重提)
              best_emb_all: 三模型声纹 tuple (复用)
        """
        meta = {"kws_denoised": True, "kws_separated": False}

        # 辅助: 提取三模型声纹并合成融合 embedding (只前向一次)
        def _extract_fused(audio):
            all_emb = self.voiceprint_extractor.extract_all(audio, sr)
            if self.is_ensemble:
                w = self.voiceprint_extractor.weights
                parts = []
                for i, e in enumerate(all_emb):
                    n = float(np.linalg.norm(e))
                    parts.append((e / n if n > 1e-8 else e) * w[i])
                fused = np.concatenate(parts).astype(np.float32)
            else:
                fused = all_emb[0]
            return fused, all_emb

        # Step 1: 降噪
        denoised = self.denoiser.denoise(audio, sr)

        # V12 决策 (2026-08-15): 恢复"唤醒词文本匹配"核心架构 (91.35% 方案)。
        # 流程: 降噪 → 混合匹配 (中文Paraformer快匹配/英文Nano) → 未命中盲分离选轨。
        # 匹配命中 → 用命中音频提声纹锚点 (更干净); 未命中 → 降噪整体 (V11 行为兜底)。
        if kws_text and self.kws_enable:
            matched_audio, kws_meta, matched = self._match_kws(denoised, kws_text, sr)
            meta.update(kws_meta)
            if matched:
                emb, all_emb = _extract_fused(matched_audio)
                return matched_audio, meta, emb, all_emb

        # 未命中/无需匹配: 降噪整体提声纹 (V11 兜底, 区分度实测最优)
        emb, all_emb = _extract_fused(denoised)
        return denoised, meta, emb, all_emb

    def process_sample(
        self,
        kws_path: str,
        cmd_path: str,
        label: Optional[str] = None,
        sample_id: str = "",
        kws_text: Optional[str] = None
    ) -> Dict:
        """
        处理单条样本: 完整流水线推理 (4阶段: 降噪→分离→声纹鉴别→ASR)

        Pipeline流程 (V9):
          Step 0: 加载kws+cmd音频 (16kHz mono)
          Step 1: kws 自适应处理 (降噪→唤醒词匹配→不够才盲分离) 后提参考声纹
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
        # Step 1: cmd 降噪 + 提声纹 (V10: 提前到 kws 之前, 作 kws 选轨参照)
        # ==============================================================
        t0 = time.time()
        denoised_audio = self.denoiser.denoise(cmd_audio, sr)
        stages_time["denoise"] = time.time() - t0

        if self.save_intermediate:
            save_wav(
                os.path.join(self.intermediate_dir, f"{intermediate_prefix}_denoised.wav"),
                denoised_audio, sr
            )

        t0 = time.time()
        denoised_embedding = self.voiceprint_extractor.extract(denoised_audio, sr)
        stages_time["voiceprint_denoised"] = time.time() - t0

        # ==============================================================
        # Step 2: kws 处理 (V10: 降噪→盲分离→用 cmd 声纹选轨) + 声纹提取
        # 选轨提取的声纹直接复用, 不再重复提取
        # ==============================================================
        t0 = time.time()
        kws_processed, kws_meta, kws_embedding, kws_emb_all = self._process_kws(
            kws_audio, kws_text, sr,
            ref_embedding=denoised_embedding,
        )
        stages_time["voiceprint_extract"] = time.time() - t0
        stages_time["kws_process"] = kws_meta

        # ==============================================================
        # Step 3: 降噪音频声纹相似度 (自适应分离的判断依据)
        # ==============================================================
        sim_denoised = float(np.dot(kws_embedding, denoised_embedding) /
                             (np.linalg.norm(kws_embedding) * np.linalg.norm(denoised_embedding) + 1e-8))

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
            # 目标提取模型需要唤醒音频作为参考（V9: 用处理后的干净 kws）；盲分离模型此调用为空操作。
            self.separator.set_reference_audio(kws_processed, sr)
            separated_audio, sources = self.separator.separate(denoised_audio, sr)
            if sources and (
                len(sources) > 1
                or getattr(self.separator, "is_target_extractor", False)
            ):
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

            text = self._recognize_instruct(asr_audio, sr)
            # 去除标点符号 (ASR punc 模型会添加标点, 但标签不含标点)
            text = strip_punctuation(text)
            # 数字归一化: 阿拉伯数字 → 中文数字 (对齐官方 label 写法, 26度→二十六度)
            text = normalize_digits(text)
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
            "separation_model": self.separator_model_name,
            "separation_used": separation_used,
            "separated_source_count": len(all_sources),
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
                kws_text = sample.get("唤醒文本", None)

                try:
                    result = self.process_sample(
                        kws_path=kws_path,
                        cmd_path=cmd_path,
                        label=label,
                        sample_id=result_id,
                        kws_text=kws_text
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

    def process_dataset_ensemble(self, data_root: str, split: str = "all",
                                  checkpoint_path: str = None) -> Dict:
        """
        三模型 Z-score 集成批量处理 (CAM++ + ERes2NetV2 + ResNetSE)

        流程:
          Phase 1: 降噪 + 三模型声纹提取 + 原始相似度计算 (全部样本)
          Phase 2: Z-score 归一化 + 加权融合 + 阈值判定
          Phase 3: Fun-ASR-Nano ASR 识别 (仅接受样本)
          Phase 4: 构建 JSONL 提交格式

        Args:
            data_root: 数据集根目录 (含 pos.jsonl, neg.jsonl)
            split: "pos" / "neg" / "all"
            checkpoint_path: 断点续传文件路径

        Returns:
            比赛提交格式 JSON
        """
        if not self._models_loaded:
            self.load_models()

        assert self.is_ensemble, "process_dataset_ensemble 需要 ensemble 模式"

        total_start = time.time()
        ens = self.voiceprint_extractor
        sr = self.config.sample_rate

        # === 加载样本 ===
        splits = []
        if split in ("pos", "all"):
            splits.append("pos")
        if split in ("neg", "all"):
            splits.append("neg")

        all_samples = []
        for sp in splits:
            jsonl_path = os.path.join(data_root, f"{sp}.jsonl")
            if not os.path.exists(jsonl_path):
                continue
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if line.strip():
                        rec = json.loads(line)
                        rec["_split"] = sp
                        rec["_idx"] = i
                        all_samples.append(rec)

        print(f"\n[Ensemble] 共 {len(all_samples)} 条样本")

        # === Phase 1: 降噪 + 声纹提取 + 原始相似度 ===
        print("\n[Phase 1] 降噪 + 三模型声纹提取...")
        raw_sims = []  # [(cam_sim, eres_sim, rnet_sim), ...]
        denoised_audios = []  # 保存降噪后音频供 ASR 使用
        sample_ids = []
        labels = []

        for i, sample in enumerate(all_samples):
            kws_path = os.path.join(data_root, sample["唤醒音频"])
            cmd_path = os.path.join(data_root, sample["识别音频"])
            label = sample.get("识别文本", None)
            kws_text = sample.get("唤醒文本", None)

            kws_audio, _ = load_wav(kws_path, sr)
            cmd_audio, _ = load_wav(cmd_path, sr)

            # cmd 降噪 (V10: 提前到 kws 之前, 作 kws 选轨参照)
            denoised = self.denoiser.denoise(cmd_audio, sr)

            # 三模型声纹提取 (降噪后 cmd)
            den_cam, den_eres, den_rnet = ens.extract_all(denoised, sr)
            den_fused = self.voiceprint_extractor.extract(denoised, sr)

            # kws 处理 (V10: 降噪 → 盲分离 → 用 cmd 融合声纹选轨), 三模型声纹复用
            kws_processed, _, _, kws_all = self._process_kws(
                kws_audio, kws_text, sr,
                ref_embedding=den_fused,
            )
            kws_cam, kws_eres, kws_rnet = kws_all[0], kws_all[1], kws_all[2]

            # 加权相似度 (绝对空间, 用于自适应分离判断)
            _w = ens.weights
            _sim_abs = (_w[0] * ens.cosine_sim(kws_cam, den_cam)
                        + _w[1] * ens.cosine_sim(kws_eres, den_eres)
                        + _w[2] * ens.cosine_sim(kws_rnet, den_rnet))

            # 自适应分离: 仅低置信样本 (sim ∈ [sep_trigger_min, vp_threshold))
            # 对齐训练数据 (datasetA_aug_processed 只对低置信样本分离)
            if (self.separation_enabled and self.separator is not None
                    and self.sep_trigger_min <= _sim_abs < self.vp_threshold):
                _sep_result = {"audio": None, "sources": None, "ok": False}
                def _do_sep():
                    try:
                        self.separator.set_reference_audio(kws_processed, sr)
                        _sep_result["audio"], _sep_result["sources"] = self.separator.separate(denoised, sr)
                        _sep_result["ok"] = True
                    except Exception as _e:
                        print(f"  [分离异常] id={sample.get('id')}: {_e}", flush=True)
                        _sep_result["audio"] = None
                import threading
                _t = threading.Thread(target=_do_sep, daemon=True)
                _t.start()
                _t.join(timeout=60)  # 60s 超时, 防单样本卡死拖死全量
                if _t.is_alive():
                    print(f"  [分离超时] id={sample.get('id')}, 跳过分离(用降噪音频)", flush=True)
                elif _sep_result["ok"] and _sep_result["audio"] is not None and len(_sep_result["audio"]) > 0:
                    # 声纹选轨 + 退化回退 (对齐 process_sample):
                    # 对分离出的每条音轨提声纹, 选与 kws 最相似的一路;
                    # 若分离后 sim 未提升或跳变超过 sim_jump_cap, 视为 artifact, 回退降噪音频
                    sources = _sep_result.get("sources") or [_sep_result["audio"]]
                    best_audio = denoised
                    best_sim = _sim_abs
                    sep_improved = False
                    for src in sources:
                        if src is None or len(src) == 0:
                            continue
                        s_cam, s_eres, s_rnet = ens.extract_all(src, sr)
                        s_sim = (_w[0] * ens.cosine_sim(kws_cam, s_cam)
                                 + _w[1] * ens.cosine_sim(kws_eres, s_eres)
                                 + _w[2] * ens.cosine_sim(kws_rnet, s_rnet))
                        if s_sim > best_sim and (s_sim - _sim_abs) <= self.sim_jump_cap:
                            best_sim = s_sim
                            best_audio = src
                            sep_improved = True
                    if sep_improved:
                        denoised = best_audio
                        den_cam, den_eres, den_rnet = ens.extract_all(denoised, sr)
                    else:
                        print(f"  [分离回退] id={sample.get('id')}: 分离未提升sim(降噪{_sim_abs:.3f}), 用降噪音频", flush=True)

            # 三模型 cosine 相似度
            cam_sim = ens.cosine_sim(kws_cam, den_cam)
            eres_sim = ens.cosine_sim(kws_eres, den_eres)
            rnet_sim = ens.cosine_sim(kws_rnet, den_rnet)

            raw_sims.append((cam_sim, eres_sim, rnet_sim))
            denoised_audios.append(denoised)
            sample_ids.append(str(sample.get("id", i)))
            labels.append(label)

            if (i + 1) % 50 == 0 or i == len(all_samples) - 1:
                elapsed = time.time() - total_start
                print(f"  [{i+1}/{len(all_samples)}] ({(i+1)/len(all_samples)*100:.1f}%) "
                      f"已用 {elapsed:.1f}s", flush=True)

        # === Phase 2: Z-score 归一化 + 融合 + 阈值判定 ===
        print("\n[Phase 2] Z-score 归一化 + 融合...")
        fused_sims = EnsembleVoiceprintExtractor.zscore_fuse(raw_sims, ens.weights)
        accept_flags = fused_sims >= ens.threshold

        n_accept = int(accept_flags.sum())
        print(f"  阈值: {ens.threshold} (Z-score)")
        print(f"  接受: {n_accept}, 拒识: {len(accept_flags) - n_accept}")

        # === Phase 3: ASR 识别 (仅接受样本) ===
        print("\n[Phase 3] Fun-ASR-Nano ASR 识别...")
        results = []
        asr_count = 0

        for i, sample in enumerate(all_samples):
            is_accepted = bool(accept_flags[i])
            label = labels[i]
            sample_id = sample_ids[i]

            if is_accepted:
                asr_audio = normalize_audio(denoised_audios[i])
                asr_audio = trim_silence(asr_audio, threshold=0.005)
                text = self._recognize_instruct(asr_audio, sr)
                text = strip_punctuation(text)
                # 数字归一化: 阿拉伯数字 → 中文数字 (对齐官方 label 写法)
                text = normalize_digits(text)
                asr_count += 1
            else:
                text = ""

            # CER 计算
            cer = None
            if label is not None:
                if label and text:
                    cer = char_error_rate(label, text)
                elif not label and not text:
                    cer = 0.0
                elif (label and not text) or (not label and text):
                    cer = 1.0

            results.append({
                "id": sample_id,
                "content": text,
                "label": label if label is not None else "",
                "cer": f"{cer:.4f}" if cer is not None else "",
                "fused_sim": f"{fused_sims[i]:.4f}",
                "is_target": is_accepted,
            })

            if (asr_count % 10 == 0 and asr_count > 0) or i == len(all_samples) - 1:
                elapsed = time.time() - total_start
                print(f"  [{i+1}/{len(all_samples)}] ASR已跑 {asr_count} 条, "
                      f"已用 {elapsed:.1f}s", flush=True)

        # === Phase 4: 构建提交 JSON ===
        total_duration = time.time() - total_start

        pos_results = [r for r in results if r["label"]]
        neg_results = [r for r in results if not r["label"] or r["label"] == "null"]

        try:
            metrics = evaluate(pos_results, neg_results)
        except Exception as e:
            print(f"警告: 评估指标计算失败 ({e})")
            metrics = {
                "final_cer": "0.0000",
                "rejection_rate": "0.0000",
                "final_score": "0.0000",
                "pos_count": len(pos_results),
                "neg_count": len(neg_results),
            }

        submission = {
            "result": {
                "results": [
                    {"id": r["id"], "content": r["content"], "label": r["label"], "cer": r["cer"]}
                    for r in results
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
