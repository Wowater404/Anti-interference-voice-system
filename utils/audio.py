"""
音频处理工具
统一的音频读写、重采样、格式转换
"""
import os
import wave
import numpy as np
from typing import Tuple, Optional


def load_wav(file_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    加载 WAV 文件, 返回 float32 numpy 数组

    Args:
        file_path: WAV 文件路径
        target_sr: 目标采样率, 不匹配时自动重采样

    Returns:
        (audio_data, sample_rate) — audio_data 为 float32, 范围 [-1, 1]
    """
    with wave.open(file_path, "rb") as wf:
        n_frames = wf.getnframes()
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        raw_data = wf.readframes(n_frames)

    # 转换为 numpy 数组
    dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
    dtype = dtype_map.get(sampwidth, np.int16)
    audio = np.frombuffer(raw_data, dtype=dtype).astype(np.float32)

    # 多声道转单声道 (取平均)
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    # 归一化到 [-1, 1]
    max_val = float(2 ** (8 * sampwidth - 1))
    audio = audio / max_val

    # 重采样 (简单线性插值, 生产环境建议用 librosa/torchaudio)
    if sample_rate != target_sr:
        n_samples = int(len(audio) * target_sr / sample_rate)
        indices = np.linspace(0, len(audio) - 1, n_samples)
        audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
        sample_rate = target_sr

    return audio, sample_rate


def save_wav(file_path: str, audio: np.ndarray, sample_rate: int = 16000) -> None:
    """
    保存 numpy 数组为 WAV 文件 (16-bit PCM)

    Args:
        file_path: 输出路径
        audio: float32 数组, 范围 [-1, 1]
        sample_rate: 采样率
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # clip 并转换为 int16
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)

    with wave.open(file_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """重采样音频"""
    if orig_sr == target_sr:
        return audio
    n_samples = int(len(audio) * target_sr / orig_sr)
    indices = np.linspace(0, len(audio) - 1, n_samples)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def get_duration(file_path: str) -> float:
    """获取音频时长 (秒)"""
    with wave.open(file_path, "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """峰值归一化到 [-1, 1]"""
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    return audio


def trim_silence(audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """去除首尾静音"""
    indices = np.where(np.abs(audio) > threshold)[0]
    if len(indices) == 0:
        return audio
    return audio[indices[0]:indices[-1] + 1]
