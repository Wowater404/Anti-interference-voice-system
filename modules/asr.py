"""
Stage 4: 语音识别模块 (ASR)
对通过声纹鉴别的目标说话人音频进行文字识别

支持模型:
  - Fun-ASR-Nano-2512 (FunAudioLLM): 自回归 LLM 架构, SenseVoice+Qwen3-0.6B, 中文识别 SOTA
  - SenseVoice (ModelScope/iic): 非自回归, 15x实时速度
  - Paraformer (FunASR): 非自回归, 速度快
  - Whisper (OpenAI): 多语言, 鲁棒性好, 但较慢
  - SherpaONNX: ONNX 推理, 极轻量

输入: 目标说话人音频 (np.ndarray, float32, [-1,1])
输出: 识别文本 (str)
"""
import os
import numpy as np
from typing import Optional
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class BaseASR:
    """语音识别模型基类"""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = None
        self._loaded = False

    def load(self):
        raise NotImplementedError

    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> str:
        """
        语音识别推理

        Args:
            audio: 输入音频, float32, [-1, 1]
            sr: 采样率

        Returns:
            识别文本
        """
        raise NotImplementedError


class SenseVoiceASR(BaseASR):
    """
    SenseVoice 语音识别器 (ModelScope/iic)
    - 与 CAM++ 同属 iic 生态, 通过 modelscope 下载模型
    - 非自回归端到端架构, 10秒音频仅需70ms推理 (15x实时速度)
    - 支持语音识别(ASR) + 语种识别 + 情感识别 + 声学事件检测
    - 模型: iic/SenseVoiceSmall
    - 安装: pip install funasr modelscope

    I/O 契约:
      输入: audio [N] float32, 值域[-1,1], sr=16000
      输出: str, 识别文本 (经 rich_transcription_postprocess 去除特殊标签)
    """

    def __init__(self, device: str = "cpu",
                 model_id: str = "iic/SenseVoiceSmall",
                 vad_model: str = "fsmn-vad",
                 language: str = "zh",
                 use_itn: bool = True):
        super().__init__(device)
        self.model_id = model_id
        self.vad_model = vad_model
        self.language = language
        self.use_itn = use_itn

    def load(self):
        """加载 SenseVoice 模型 (通过 modelscope 下载, funasr 推理)"""
        try:
            from modelscope import snapshot_download
            from funasr import AutoModel

            model_dir = snapshot_download(self.model_id)
            os.environ.setdefault('MODELSCOPE_CACHE',
                                  os.path.join(PROJECT_ROOT, 'pretrained', 'modelscope_cache'))

            model_kwargs = {
                "model": model_dir,
                "trust_remote_code": True,
                "device": self.device,
            }

            if self.vad_model:
                model_kwargs["vad_model"] = self.vad_model
                model_kwargs["vad_kwargs"] = {"max_single_segment_time": 30000}

            self.model = AutoModel(**model_kwargs)
            self._loaded = True
            print(f"[SenseVoice] 模型加载成功 ({self.model_id})")
        except ImportError:
            print("[SenseVoice] 警告: funasr/modelscope 未安装, 使用直通模式")
            print("  安装命令: pip install funasr modelscope")
            self._loaded = True
        except Exception as e:
            print(f"[SenseVoice] 加载失败: {e}")
            print("[SenseVoice] 使用直通模式")
            self._loaded = True

    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> str:
        """
        SenseVoice 语音识别 (非自回归, 极速)

        内部流程: 音频→临时WAV文件→FunASR.generate→rich_transcription_postprocess→返回

        Args:
            audio: np.ndarray [N], float32, 目标说话人音频波形, 值域[-1,1], 16kHz单声道
            sr: int, 采样率 (默认16000)

        Returns:
            text: str, 识别出的文本 (经 rich_transcription_postprocess 去除特殊标签)
                  (若模型未加载或识别失败则返回空字符串"")
        """
        if not self._loaded:
            self.load()

        if self.model is None:
            return ""

        import tempfile
        from utils.audio import save_wav

        # FunASR 需要文件路径输入
        tmp_path = os.path.join(tempfile.gettempdir(), "sensevoice_tmp.wav")
        save_wav(tmp_path, audio, sr)

        try:
            result = self.model.generate(
                input=tmp_path,
                cache={},
                language=self.language,
                use_itn=self.use_itn,
                batch_size_s=300,
                merge_vad=True,
                merge_length_s=15,
            )
            try:
                from funasr.utils.postprocess_utils import rich_transcription_postprocess
                text = rich_transcription_postprocess(result[0]["text"]) if result else ""
            except ImportError:
                text = result[0]["text"] if result else ""
        except Exception as e:
            print(f"[SenseVoice] 识别错误: {e}")
            text = ""
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        return text


class ParaformerASR(BaseASR):
    """
    Paraformer 语音识别器 (FunASR)
    - 非自回归模型, 推理速度快
    - 中文识别 SOTA
    - 支持 VAD 断句 + 标点恢复
    - 安装: pip install funasr modelscope
    """

    def __init__(self, device: str = "cpu",
                 model_name: str = "paraformer-zh",
                 vad_model: str = "fsmn-vad",
                 punc_model: str = "ct-punc",
                 hotwords: str = ""):
        super().__init__(device)
        self.model_name = model_name
        self.vad_model = vad_model
        self.punc_model = punc_model
        self.hotwords = hotwords

    def load(self):
        """加载 Paraformer 模型"""
        try:
            from funasr import AutoModel

            model_kwargs = {
                "model": self.model_name,
                "device": self.device,
            }

            # VAD (语音活动检测, 自动断句)
            if self.vad_model:
                model_kwargs["vad_model"] = self.vad_model
                model_kwargs["vad_kwargs"] = {"max_single_segment_time": 30000}

            # 标点恢复
            if self.punc_model:
                model_kwargs["punc_model"] = self.punc_model

            # 热词 (领域词汇增强)
            if self.hotwords:
                model_kwargs["hotword"] = self.hotwords

            self.model = AutoModel(**model_kwargs)
            self._loaded = True
            print("[Paraformer] 模型加载成功")
        except ImportError:
            print("[Paraformer] 警告: funasr 未安装, 使用直通模式")
            print("  安装命令: pip install funasr modelscope")
            self._loaded = True

    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> str:
        """
        Paraformer 语音识别 (非自回归, 速度快)

        内部流程: 音频→临时WAV文件→FunASR.generate→提取text→去标点→返回

        Args:
            audio: np.ndarray [N], float32, 目标说话人音频波形, 值域[-1,1], 16kHz单声道
            sr: int, 采样率 (默认16000)

        Returns:
            text: str, 识别出的中文文本 (已去标点)
                  (若模型未加载或识别失败则返回空字符串"")
        """
        if not self._loaded:
            self.load()

        if self.model is None:
            return ""

        import tempfile
        import os as _os
        from utils.audio import save_wav

        # FunASR 需要文件路径输入
        tmp_path = _os.path.join(tempfile.gettempdir(), "paraformer_tmp.wav")
        save_wav(tmp_path, audio, sr)

        try:
            gen_kwargs = {
                "input": tmp_path,
                "batch_size_s": 300,
            }
            # [2026-08-15] 启用热词偏置 (config 已配: 空调,洗碗机,灯光...)
            # 注意: 空/None 不传 (funasr 对空 hotwords 可能崩溃)
            if self.hotwords:
                gen_kwargs["hotwords"] = self.hotwords
            result = self.model.generate(**gen_kwargs)
            text = result[0]["text"] if result else ""
            # [2026-08-15] 文本清洗: 官方 normalize_text (NFKC+小写+去空白/标点)
            # Paraformer 输出可能带字符间空格 (无 punc 模型时), 必须清洗后才与 label 可比
            if text:
                from utils.metrics import strip_punctuation
                text = strip_punctuation(text)
        except Exception as e:
            print(f"[Paraformer] 识别错误: {e}")
            text = ""
        finally:
            if _os.path.exists(tmp_path):
                _os.remove(tmp_path)

        return text


class WhisperASR(BaseASR):
    """
    Whisper 语音识别器 (OpenAI)
    - 多语言, 鲁棒性好
    - 适合噪声环境
    - 安装: pip install openai-whisper
    """

    def __init__(self, device: str = "cpu",
                 model_size: str = "base",
                 language: str = "zh"):
        super().__init__(device)
        self.model_size = model_size
        self.language = language

    def load(self):
        """加载 Whisper 模型"""
        try:
            import whisper

            self.model = whisper.load_model(
                self.model_size,
                device=self.device
            )
            self._loaded = True
            print(f"[Whisper-{self.model_size}] 模型加载成功")
        except ImportError:
            print("[Whisper] 警告: openai-whisper 未安装, 使用直通模式")
            print("  安装命令: pip install openai-whisper")
            self._loaded = True

    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> str:
        """Whisper 识别"""
        if not self._loaded:
            self.load()

        if self.model is None:
            return ""

        # Whisper 直接接受 numpy 数组
        result = self.model.transcribe(
            audio,
            language=self.language,
            task="transcribe",
            fp16=False if self.device == "cpu" else True,
        )
        return result.get("text", "").strip()


class SherpaONNXASR(BaseASR):
    """
    SherpaONNX 语音识别器
    - ONNX 格式推理, 极轻量
    - 适合比赛效率评分
    - 安装: pip install sherpa-onnx
    """

    def __init__(self, device: str = "cpu",
                 encoder: Optional[str] = None,
                 decoder: Optional[str] = None,
                 tokens: Optional[str] = None):
        super().__init__(device)
        self.encoder = encoder
        self.decoder = decoder
        self.tokens = tokens

    def load(self):
        """加载 SherpaONNX 模型"""
        try:
            import sherpa_onnx

            if self.encoder and self.decoder and self.tokens:
                self.model = sherpa_onnx.OfflineRecognizer(
                    encoder=self.encoder,
                    decoder=self.decoder,
                    tokens=self.tokens,
                    num_threads=1,
                    provider="cpu",
                )
                self._loaded = True
                print("[SherpaONNX] 模型加载成功")
            else:
                print("[SherpaONNX] 警告: 未指定模型路径, 使用直通模式")
                self._loaded = True
        except ImportError:
            print("[SherpaONNX] 警告: sherpa-onnx 未安装, 使用直通模式")
            print("  安装命令: pip install sherpa-onnx")
            self._loaded = True

    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> str:
        """SherpaONNX 识别"""
        if not self._loaded:
            self.load()

        if self.model is None:
            return ""

        import sherpa_onnx

        # 创建音频流
        stream = self.model.create_stream()
        stream.accept_waveform(sr, audio.tolist())
        stream.input_finished()

        text = self.model.decode_stream(stream)
        return text


class FunASRNanoASR(BaseASR):
    """
    Fun-ASR-Nano-2512 语音识别器 (FunAudioLLM)
    - 自回归 LLM 架构: SenseVoice 编码器 + Qwen3-0.6B LLM 解码器, 800M 参数
    - 中文识别 SOTA, CER 显著优于 Paraformer
    - 不支持 batch decoding, 必须用 batch_size=1
    - hotwords=None 会崩溃, 必须条件传参
    - 输出字段: text_tn (ITN处理后无标点) 优先于 text (含标点)
    - 安装: pip install funasr modelscope
    """

    def __init__(self, device: str = "cuda:0",
                 model_dir: str = "FunAudioLLM/Fun-ASR-Nano-2512",
                 language: str = "中文",
                 hotwords=None):
        super().__init__(device)
        self.model_dir = model_dir
        self.language = language
        self.hotwords = hotwords

    def load(self):
        """加载 Fun-ASR-Nano-2512 模型"""
        try:
            from funasr import AutoModel

            # [2026-08-15] fp16 加速: 环境变量 PPS_ASR_FP16=1 启用半精度推理
            # (LLM 自回归解码在 fp16 下约快 1.5-2x, CER 通常不变或微降, 需 A/B 验证)
            _kwargs = dict(
                model=self.model_dir,
                device=self.device,
                disable_update=True,
            )
            if os.environ.get("PPS_ASR_FP16", "0") == "1":
                _kwargs["dtype"] = "fp16"
                print("[Fun-ASR-Nano] fp16 半精度推理已启用 (PPS_ASR_FP16=1)")
            self.model = AutoModel(**_kwargs)
            self._loaded = True
            print(f"[Fun-ASR-Nano] 模型加载成功 ({self.model_dir})")
        except ImportError:
            print("[Fun-ASR-Nano] 警告: funasr 未安装, 使用直通模式")
            print("  安装命令: pip install funasr modelscope")
            self._loaded = True
        except Exception as e:
            print(f"[Fun-ASR-Nano] 加载失败: {e}")
            print("[Fun-ASR-Nano] 使用直通模式")
            self._loaded = True

    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> str:
        """
        Fun-ASR-Nano 语音识别 (自回归 LLM)

        内部流程: 音频→临时WAV文件→FunASR.generate→提取text_tn→返回

        Args:
            audio: np.ndarray [N], float32, 目标说话人音频波形, 值域[-1,1], 16kHz单声道
            sr: int, 采样率 (默认16000)

        Returns:
            text: str, 识别出的文本 (优先使用 text_tn 字段, 无标点)
                  (若模型未加载或识别失败则返回空字符串"")
        """
        if not self._loaded:
            self.load()

        if self.model is None:
            return ""

        import tempfile
        from utils.audio import save_wav

        tmp_path = os.path.join(tempfile.gettempdir(), "funasr_nano_tmp.wav")
        save_wav(tmp_path, audio, sr)

        try:
            gen_kwargs = {
                "input": tmp_path,
                "cache": {},
                "batch_size": 1,
                "language": self.language,
            }
            if self.hotwords is not None:
                gen_kwargs["hotwords"] = self.hotwords

            result = self.model.generate(**gen_kwargs)

            text = ""
            if result:
                text = result[0].get("text_tn", "") or result[0].get("text", "")
        except Exception as e:
            print(f"[Fun-ASR-Nano] 识别错误: {e}")
            text = ""
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        return text


def create_asr(config: dict, device: str = "cpu") -> BaseASR:
    """
    工厂函数: 根据配置创建 ASR 模型

    Args:
        config: dict, asr配置字典 (含model/fun_asr_nano/paraformer等字段)
        device: str, "cuda" 或 "cpu"

    Returns:
        BaseASR 子类实例 (FunASRNanoASR / SenseVoiceASR / ParaformerASR / WhisperASR / SherpaONNXASR)
    """
    model_name = config.get("model", "fun_asr_nano")

    if model_name == "fun_asr_nano":
        cfg = config.get("fun_asr_nano", {})
        return FunASRNanoASR(
            device=cfg.get("device", "cuda:0" if device != "cpu" else "cpu"),
            model_dir=cfg.get("model_dir", "FunAudioLLM/Fun-ASR-Nano-2512"),
            language=cfg.get("language", "中文"),
            hotwords=cfg.get("hotwords"),
        )
    elif model_name == "sensevoice":
        cfg = config.get("sensevoice", {})
        return SenseVoiceASR(
            device=device,
            model_id=cfg.get("model_id", "iic/SenseVoiceSmall"),
            vad_model=cfg.get("vad_model", "fsmn-vad"),
            language=cfg.get("language", "zh"),
            use_itn=cfg.get("use_itn", True),
        )
    elif model_name == "paraformer":
        cfg = config.get("paraformer", {})
        return ParaformerASR(
            device=device,
            model_name=cfg.get("model_name", "paraformer-zh"),
            vad_model=cfg.get("vad_model", "fsmn-vad"),
            punc_model=cfg.get("punc_model", "ct-punc"),
            hotwords=cfg.get("hotwords", ""),
        )
    elif model_name == "whisper":
        cfg = config.get("whisper", {})
        return WhisperASR(
            device=device,
            model_size=cfg.get("model_size", "base"),
            language=cfg.get("language", "zh"),
        )
    elif model_name == "sherpa_onnx":
        cfg = config.get("sherpa_onnx", {})
        return SherpaONNXASR(
            device=device,
            encoder=cfg.get("encoder"),
            decoder=cfg.get("decoder"),
            tokens=cfg.get("tokens"),
        )
    else:
        print(f"[ASR] 未知模型 {model_name}, 使用 SenseVoice")
        return SenseVoiceASR(device=device)
