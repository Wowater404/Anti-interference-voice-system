"""
Stage 1: 降噪模块
去除非人声噪声 (空调噪声、环境噪声等), 提升后续阶段输入质量

支持模型:
  - noisereduce:    纯Python降噪, 无需额外编译, 适合快速验证
  - DeepFilterNet3: 轻量实时降噪, 推理速度极快, 适合比赛效率要求 (需Rust编译)
  - FullSubNet+:    基于子带+全带的降噪, 质量更高但计算量更大
  - DEMUCS:         通用音频源分离, 可去除各类非人声干扰

输入: 原始 cmd 音频 (np.ndarray, float32, [-1,1])
输出: 降噪后音频 (np.ndarray, float32, [-1,1])
"""
import os
import numpy as np
from typing import Optional
import sys

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class BaseDenoiser:
    """降噪模型基类"""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = None
        self._loaded = False

    def load(self):
        """加载模型权重 (延迟加载)"""
        raise NotImplementedError

    def denoise(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        降噪推理

        Args:
            audio: 输入音频, float32, [-1, 1]
            sr: 采样率

        Returns:
            降噪后音频, float32, [-1, 1]
        """
        raise NotImplementedError

    def is_loaded(self) -> bool:
        return self._loaded


class DeepFilterNet3Denoiser(BaseDenoiser):
    """
    DeepFilterNet3 降噪器
    - 基于深度滤波器的实时降噪
    - 极低延迟, 适合比赛效率评分
    - 安装: pip install deepfilternet
    """

    def __init__(self, device: str = "cpu", model_dir: Optional[str] = None,
                 atten_lim_db: int = 6):
        super().__init__(device)
        self.model_dir = model_dir
        self.atten_lim_db = atten_lim_db

    def load(self):
        """加载 DeepFilterNet3 模型"""
        try:
            import torch
            from df.enhance import init_df, enhance

            self.model, self.df_state, _ = init_df(
                model_base_dir=self.model_dir,
                post_filter=False,
                log_level="WARNING"
            )
            self._enhance_fn = enhance
            self._loaded = True
            print("[DeepFilterNet3] 模型加载成功")
        except ImportError:
            print("[DeepFilterNet3] 警告: deepfilternet 未安装, 使用直通模式")
            print("  安装命令: pip install deepfilternet")
            self._loaded = True  # 直通模式

    def denoise(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """DeepFilterNet3 降噪"""
        if not self._loaded:
            self.load()

        if self.model is None:
            # 直通模式 (未安装依赖时)
            return audio

        import torch
        # DeepFilterNet 原生 48kHz, 需要重采样
        from df.enhance import init_df

        df_sr = self.df_state.sr()
        if sr != df_sr:
            from utils.audio import resample
            audio_df = resample(audio, sr, df_sr)
        else:
            audio_df = audio

        # 推理
        enhanced = self._enhance_fn(
            self.model,
            self.df_state,
            torch.from_numpy(audio_df).to(self.device)
        )
        enhanced = enhanced.cpu().numpy()

        # 重采样回原始采样率
        if df_sr != sr:
            from utils.audio import resample
            enhanced = resample(enhanced, df_sr, sr)

        return enhanced


class FullSubNetPlusDenoiser(BaseDenoiser):
    """
    FullSubNet+ 降噪器
    - 联合全带和子带特征
    - 降噪质量高, 适合低 SNR 场景
    - 安装: pip install fullsubnet
    """

    def __init__(self, device: str = "cpu", checkpoint: Optional[str] = None):
        super().__init__(device)
        self.checkpoint = checkpoint

    def load(self):
        """加载 FullSubNet+ 模型"""
        try:
            import torch
            from fullsubnet.model import FullSubNetPlus

            self.model = FullSubNetPlus(
                num_channels=1,
                num_features=256,
                hidden_size=512,
            )
            if self.checkpoint and os.path.exists(self.checkpoint):
                ckpt = torch.load(self.checkpoint, map_location=self.device)
                self.model.load_state_dict(ckpt["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()
            self._loaded = True
            print("[FullSubNet+] 模型加载成功")
        except ImportError:
            print("[FullSubNet+] 警告: fullsubnet 未安装, 使用直通模式")
            self._loaded = True

    def denoise(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """FullSubNet+ 降噪"""
        if not self._loaded:
            self.load()

        if self.model is None:
            return audio

        import torch
        # 分帧处理 (FullSubNet 需要固定帧长)
        frame_length = 512
        hop_length = 256

        # STFT 分析
        from scipy.signal import stft, istft
        f, t, Zxx = stft(audio, fs=sr, nperseg=frame_length, noverlap=frame_length - hop_length)
        mag = np.abs(Zxx)[np.newaxis, np.newaxis, ...]  # [1, 1, F, T]

        with torch.no_grad():
            mag_t = torch.from_numpy(mag).to(self.device)
            enhanced_mag = self.model(mag_t)
            enhanced_mag = enhanced_mag.cpu().numpy()[0, 0]

        # 重建波形
        enhanced_Zxx = Zxx * (enhanced_mag / (mag[0, 0] + 1e-8))
        _, enhanced = istft(enhanced_Zxx, fs=sr, nperseg=frame_length, noverlap=frame_length - hop_length)

        return enhanced[:len(audio)]


class NoiseReduceDenoiser(BaseDenoiser):
    """
    noisereduce 降噪器
    - 基于频谱门限 (Spectral Gating) 的经典降噪
    - 纯Python实现, 无需编译, 安装简单
    - 适合快速验证流水线
    - 安装: pip install noisereduce
    """

    def __init__(self, device: str = "cpu", stationary: bool = True,
                 prop_decrease: float = 0.8):
        super().__init__(device)
        self.stationary = stationary
        self.prop_decrease = prop_decrease

    def load(self):
        """加载 noisereduce"""
        try:
            import noisereduce
            self._nr = noisereduce
            self._loaded = True
            print("[NoiseReduce] 模型加载成功 (频谱门限降噪)")
        except ImportError:
            print("[NoiseReduce] 警告: noisereduce 未安装, 使用直通模式")
            print("  安装命令: pip install noisereduce")
            self._loaded = True

    def denoise(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        noisereduce 降噪 (频谱门限法)

        Args:
            audio: np.ndarray [N] 或 [N, 1], float32, 原始音频波形, 值域[-1,1]
            sr: int, 采样率 (默认16000)

        Returns:
            np.ndarray [N], float32, 降噪后音频波形, 值域[-1,1]
            (若noisereduce未安装则直通返回原始音频)
        """
        if not self._loaded:
            self.load()

        if self.model is None and not hasattr(self, '_nr'):
            return audio

        # noisereduce 支持 1D 和 2D 输入
        if audio.ndim == 1:
            enhanced = self._nr.reduce_noise(
                y=audio,
                sr=sr,
                stationary=self.stationary,
                prop_decrease=self.prop_decrease,
            )
        else:
            enhanced = self._nr.reduce_noise(
                y=audio,
                sr=sr,
                stationary=self.stationary,
                prop_decrease=self.prop_decrease,
            )

        return enhanced.astype(np.float32)


class GTCRNDenoiser(BaseDenoiser):
    """
    GTCRN 降噪器 (深度学习, 默认推荐)
    - 仅 48.2K 参数, 33.0 MMACs/s, 实时推理
    - ShuffleNetV2 + SFE + TRA + Dual-Path GRNN
    - Complex Ratio Mask: 同时修复幅度和相位
    - 预训练权重: DNS3 (通用) / VCTK (人声)

    输入/输出: np.ndarray, float32, 单声道, [-1, 1], 16kHz
    """

    def __init__(self, device: str = "cpu", checkpoint: str = "dns3",
                 n_fft: int = 512, hop_length: int = 256):
        super().__init__(device)
        self.checkpoint_name = checkpoint
        self.n_fft = n_fft
        self.hop_length = hop_length

        model_map = {
            "dns3": "model_trained_on_dns3.tar",
            "vctk": "model_trained_on_vctk.tar",
        }
        ckpt_file = model_map.get(checkpoint, "model_trained_on_dns3.tar")
        self.checkpoint_path = os.path.join(
            PROJECT_ROOT, "pretrained", "gtcrn", ckpt_file
        )
        self._window = None
        self._torch_device = None

    def load(self):
        """加载 GTCRN 模型权重"""
        try:
            import torch
            from modules.gtcrn import GTCRN

            actual_device = self.device
            if actual_device == "auto":
                actual_device = "cuda" if torch.cuda.is_available() else "cpu"

            self._torch_device = torch.device(actual_device)
            self.model = GTCRN().to(self._torch_device).eval()

            if os.path.exists(self.checkpoint_path):
                ckpt = torch.load(self.checkpoint_path,
                                  map_location=self._torch_device,
                                  weights_only=True)
                self.model.load_state_dict(ckpt["model"])
            else:
                print(f"[GTCRN] 警告: checkpoint 不存在 ({self.checkpoint_path}), "
                      f"使用未训练权重")

            self._window = torch.hann_window(self.n_fft).pow(0.5)
            self._loaded = True
            print(f"[GTCRN] 模型加载成功 (checkpoint={self.checkpoint_name}, "
                  f"device={actual_device})")
        except ImportError as e:
            print(f"[GTCRN] 警告: 依赖缺失 ({e}), 使用直通模式")
            self._loaded = True

    def denoise(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        GTCRN 降噪

        Args:
            audio: np.ndarray, float32, [-1, 1], 单声道
            sr: 原始采样率 (非 16kHz 自动重采样)

        Returns:
            np.ndarray, float32, [-1, 1], 单声道, 16kHz
        """
        if not self._loaded:
            self.load()

        if self.model is None:
            return audio

        import torch

        # 重采样到 16kHz
        if sr != 16000:
            from utils.audio import resample
            audio = resample(audio, sr, 16000)

        mix = torch.from_numpy(audio.astype(np.float32))

        # STFT
        spec = torch.stft(
            mix, self.n_fft, self.hop_length, self.n_fft,
            self._window, return_complex=True,
        )
        spec = torch.view_as_real(spec).to(self._torch_device)  # (F, T, 2)

        # 推理
        with torch.no_grad():
            output = self.model(spec[None])[0]  # (F, T, 2)

        output = torch.view_as_complex(output.contiguous()).cpu()

        # iSTFT
        enh = torch.istft(
            output, self.n_fft, self.hop_length, self.n_fft, self._window
        )
        enhanced = enh.detach().cpu().numpy().astype(np.float32)

        # 匹配原始长度
        if len(enhanced) < len(audio):
            enhanced = np.pad(enhanced, (0, len(audio) - len(enhanced)))
        else:
            enhanced = enhanced[:len(audio)]

        # 严格裁剪到 [-1, 1]
        return np.clip(enhanced, -1.0, 1.0)


class PassThroughDenoiser(BaseDenoiser):
    """直通降噪器 (不处理, 用于调试)"""

    def load(self):
        self._loaded = True
        print("[PassThrough] 直通模式 (不做降噪)")

    def denoise(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        return audio


def create_denoiser(config: dict, device: str = "cpu") -> BaseDenoiser:
    """
    工厂函数: 根据配置创建降噪器

    Args:
        config: denoise 配置字典
        device: 推理设备

    Returns:
        降噪器实例
    """
    if not config.get("enable", True):
        return PassThroughDenoiser(device)

    model_name = config.get("model", "noisereduce")

    if model_name == "gtcrn":
        cfg = config.get("gtcrn", {})
        return GTCRNDenoiser(
            device=device,
            checkpoint=cfg.get("checkpoint", "dns3"),
            n_fft=cfg.get("n_fft", 512),
            hop_length=cfg.get("hop_length", 256),
        )
    elif model_name == "noisereduce":
        cfg = config.get("noisereduce", {})
        return NoiseReduceDenoiser(
            device=device,
            stationary=cfg.get("stationary", True),
            prop_decrease=cfg.get("prop_decrease", 0.8),
        )
    elif model_name == "deepfilternet3":
        cfg = config.get("deepfilternet3", {})
        return DeepFilterNet3Denoiser(
            device=device,
            model_dir=cfg.get("model_dir"),
            atten_lim_db=cfg.get("atten_lim_db", 6),
        )
    elif model_name == "fullsubnet_plus":
        cfg = config.get("fullsubnet_plus", {})
        return FullSubNetPlusDenoiser(
            device=device,
            checkpoint=cfg.get("checkpoint"),
        )
    else:
        print(f"[Denoiser] 未知模型 {model_name}, 使用直通模式")
        return PassThroughDenoiser(device)
