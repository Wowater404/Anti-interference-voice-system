"""
Stage 1: 降噪模块
基于频谱门限 (Spectral Gating) 去除环境噪声, 提升后续阶段输入质量

自包含实现, 无外部降噪库依赖

--- 用法 ---
    单文件可直接使用, 无需其他项目文件:

        from renoise import create_denoiser

        config = {
            "model": "renoise",
            "renoise": {"stationary": True, "prop_decrease": 0.8},
            "enable": True,
        }
        denoiser = create_denoiser(config, device="cpu")
        denoiser.load()
        clean = denoiser.denoise(noisy_audio, sr=16000)

--- requirements.txt ---
    numpy>=1.24.0
    scipy>=1.10.0
"""
import numpy as np
from scipy.signal import stft, istft
from scipy.ndimage import uniform_filter


def _spectral_gating(
    audio: np.ndarray,
    sr: int,
    stationary: bool = True,
    prop_decrease: float = 0.8,
    n_std_thresh: float = 1.5,
    n_fft: int = 1024,
) -> np.ndarray:
    """
    频谱门限降噪核心算法

    步骤:
    1. STFT 变换: 将音频从时域变换到时频域
    2. 噪声估计: 估计每个频段的噪声底噪
       - stationary=True:  全局均值 (假设噪声恒定, 如空调、风扇)
       - stationary=False: 低分位数估计 (适应时变噪声)
    3. 阈值计算: 噪声底噪 + n_std * 噪声标准差
    4. 软掩膜: 低于阈值的时频点按 prop_decrease 比例渐进衰减
    5. 掩膜平滑: 频域+时域平滑减少音乐噪声伪影
    6. ISTFT 重建: 从修改后的时频谱重建时域信号

    参数:
        audio:          输入音频 (1D numpy array)
        sr:             采样率 (Hz)
        stationary:     静态噪声假设, True 适合持续背景噪声
        prop_decrease:  噪声抑制强度 (0-1, 越大抑制越多, 推荐 0.6~0.9)
        n_std_thresh:   阈值标准差倍数 (越高越保守, 推荐 1.0~2.0)
        n_fft:          FFT 点数 (影响频率分辨率)

    返回:
        enhanced:       降噪后音频 (float32, 与原音频等长)
    """
    if len(audio) == 0:
        return audio

    # 转为 float64 保证 STFT 精度
    audio = audio.astype(np.float64)

    hop_length = n_fft // 4  # 75% overlap, 平衡时间分辨率与平滑度

    # =====================================================================
    # 1. STFT 变换 (短时傅里叶变换)
    # =====================================================================
    f, t, Zxx = stft(
        audio, fs=sr, nperseg=n_fft,
        noverlap=n_fft - hop_length,
        nfft=n_fft, window='hann'
    )
    # Zxx: (n_freq, n_frames) 复数矩阵
    mag = np.abs(Zxx)       # 幅度谱
    phase = np.angle(Zxx)   # 相位谱

    eps = 1e-10  # 防止 log10(0)

    # =====================================================================
    # 2. 噪声底噪估计
    # =====================================================================
    if stationary:
        # 静态噪声: 假设噪声统计特性不随时间变化
        # 对所有时间帧取平均得到每频段的噪声幅值
        noise_mag = np.mean(mag, axis=-1, keepdims=True)
    else:
        # 非静态噪声: 语音是稀疏的 (只在部分频段/时间出现),
        # 而噪声遍布所有时间帧 → 低分位数 = 噪声底噪
        noise_mag = np.percentile(mag, 15, axis=-1, keepdims=True)

    # =====================================================================
    # 3. 阈值计算
    # =====================================================================
    noise_db = 20 * np.log10(noise_mag + eps)

    if stationary:
        # 统计噪声在每频段的波动幅度
        noise_std_db = np.std(20 * np.log10(mag + eps), axis=-1, keepdims=True)
    else:
        # 非静态模式使用固定 6 dB 作为默认波动估计
        noise_std_db = np.ones_like(noise_db) * 6.0

    # 阈值 = 噪声底噪 + n_std_thresh 倍标准差
    # 低于阈值的时频点被认为是"噪声主导"
    thresh_db = noise_db + n_std_thresh * noise_std_db

    # =====================================================================
    # 4. 软掩膜计算
    # =====================================================================
    sig_db = 20 * np.log10(mag + eps)
    db_above = sig_db - thresh_db   # >0 = 信号主导, <0 = 噪声主导

    # 渐进衰减函数: 越低于阈值, 衰减越多
    #   db_above >= 0   → gain = 1.0  (信号主导, 不衰减)
    #   db_above = -10  → gain ≈ 0.40
    #   db_above = -20  → gain ≈ 0.16
    #   prop_decrease 越大, 衰减越激进
    gain = np.where(
        db_above >= 0,
        1.0,
        10.0 ** (prop_decrease * db_above / 20.0)
    )

    # =====================================================================
    # 5. 掩膜平滑 (减少"音乐噪声"伪影)
    # =====================================================================
    # 频域平滑: 相邻频率使用相近的增益, 避免孤立频段
    gain = uniform_filter(gain, size=(3, 1), mode='reflect')
    # 时域平滑: 相邻帧使用相近的增益, 避免突变
    gain = uniform_filter(gain, size=(1, 3), mode='reflect')

    # =====================================================================
    # 6. 应用掩膜 & ISTFT 重建
    # =====================================================================
    mag_enhanced = mag * gain
    Zxx_enhanced = mag_enhanced * np.exp(1j * phase)

    _, enhanced = istft(
        Zxx_enhanced, fs=sr, nperseg=n_fft,
        noverlap=n_fft - hop_length,
        nfft=n_fft, window='hann'
    )

    # 对齐原始长度 (ISTFT 可能因边界处理产生微小长度差异)
    if len(enhanced) > len(audio):
        enhanced = enhanced[:len(audio)]
    elif len(enhanced) < len(audio):
        enhanced = np.pad(enhanced, (0, len(audio) - len(enhanced)))

    return enhanced.astype(np.float32)


class BaseDenoiser:
    """降噪模型基类"""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = None
        self._loaded = False

    def load(self):
        raise NotImplementedError

    def denoise(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        raise NotImplementedError

    def is_loaded(self) -> bool:
        return self._loaded


class NoiseReduceDenoiser(BaseDenoiser):
    """
    频谱门限降噪器 (自包含实现, 无需安装 noisereduce)

    参数:
        stationary:    True=静态噪声假设 (适合空调等持续噪声)
        prop_decrease: 噪声抑制比例 (0-1, 越大抑制越多, 推荐 0.8)
    """

    def __init__(self, device: str = "cpu", stationary: bool = True,
                 prop_decrease: float = 0.8):
        super().__init__(device)
        self.stationary = stationary
        self.prop_decrease = prop_decrease

    def load(self):
        self._loaded = True
        print(f"[NoiseReduce] 频谱门限降噪就绪 "
              f"(stationary={self.stationary}, prop_decrease={self.prop_decrease})")

    def denoise(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        if not self._loaded:
            self.load()

        enhanced = _spectral_gating(
            audio=audio,
            sr=sr,
            stationary=self.stationary,
            prop_decrease=self.prop_decrease,
        )
        return enhanced.astype(np.float32)


class PassThroughDenoiser(BaseDenoiser):
    """直通降噪器 (不做降噪, 调试用)"""

    def load(self):
        self._loaded = True
        print("[PassThrough] 直通模式 (不做降噪)")

    def denoise(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        return audio


def create_denoiser(config: dict, device: str = "cpu") -> BaseDenoiser:
    """
    工厂函数: 根据配置创建降噪器

    config 示例:
        {"model": "renoise", "renoise": {"stationary": true, "prop_decrease": 0.8}, "enable": true}

    用法:
        denoiser = create_denoiser(config, device="cpu")
        denoiser.load()
        clean = denoiser.denoise(noisy_audio, sr=16000)
    """
    if not config.get("enable", True):
        return PassThroughDenoiser(device)

    model_name = config.get("model", "renoise")

    if model_name == "renoise":
        cfg = config.get("renoise", {})
        return NoiseReduceDenoiser(
            device=device,
            stationary=cfg.get("stationary", True),
            prop_decrease=cfg.get("prop_decrease", 0.8),
        )

    print(f"[Denoiser] 未知模型 {model_name}, 使用直通模式")
    return PassThroughDenoiser(device)
