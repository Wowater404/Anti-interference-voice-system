"""
配置加载模块
从 YAML 配置文件加载流水线参数
"""
import os
from pathlib import Path
from typing import Any
import yaml


class PipelineConfig:
    """流水线配置管理"""

    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), "configs", "default.yaml"
            )
        with open(config_path, "r", encoding="utf-8") as f:
            self._cfg = yaml.safe_load(f)

    @property
    def audio(self) -> dict:
        return self._cfg.get("audio", {})

    @property
    def sample_rate(self) -> int:
        return self.audio.get("sample_rate", 16000)

    @property
    def denoise(self) -> dict:
        return self._cfg.get("denoise", {})

    @property
    def separation(self) -> dict:
        return self._cfg.get("separation", {})

    @property
    def voiceprint(self) -> dict:
        return self._cfg.get("voiceprint", {})

    @property
    def asr(self) -> dict:
        return self._cfg.get("asr", {})

    @property
    def dataset(self) -> dict:
        return self._cfg.get("dataset", {})

    @property
    def output(self) -> dict:
        return self._cfg.get("output", {})

    @property
    def device(self) -> str:
        """获取推理设备, 支持 auto 自动检测"""
        device = self._cfg.get("device", "cpu")
        if device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
            except ImportError:
                pass
            return "cpu"
        return device

    def get(self, key: str, default: Any = None) -> Any:
        """通用键值获取"""
        return self._cfg.get(key, default)
