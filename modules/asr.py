"""
Stage 4: 语音识别模块 (ASR)
对通过声纹鉴别的目标说话人音频进行文字识别

支持模型:
  - Paraformer (FunASR): 阿里开源, 中文非自回归 ASR, 速度快
  - Whisper (OpenAI):    多语言, 鲁棒性好, 但较慢
  - SherpaONNX:          ONNX 推理, 极轻量

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
        """Paraformer 识别"""
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
            result = self.model.generate(
                input=tmp_path,
                batch_size_s=300,
            )
            text = result[0]["text"] if result else ""
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


def create_asr(config: dict, device: str = "cpu") -> BaseASR:
    """
    工厂函数: 根据配置创建 ASR 模型
    """
    model_name = config.get("model", "paraformer")

    if model_name == "paraformer":
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
        print(f"[ASR] 未知模型 {model_name}, 使用 Paraformer")
        return ParaformerASR(device=device)
