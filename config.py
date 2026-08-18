"""
配置加载模块
从 YAML 配置文件加载流水线参数

配置结构 (configs/default.yaml):
  audio:        采样率等音频参数
  denoise:      Stage1 降噪配置 (model/enable/参数)
  separation:   Stage2 分离配置 (model/enable/阈值/触发条件)
  voiceprint:   Stage3 声纹配置 (model/threshold/微调权重路径)
  asr:          Stage4 语音识别配置 (model/热词/VAD/标点)
  device:       推理设备 ("auto"=自动检测CUDA, "cuda", "cpu")
  output:       输出格式配置
"""
import os
from pathlib import Path
from typing import Any
import yaml


class PipelineConfig:
    """
    流水线配置管理
    加载YAML配置文件, 通过属性访问各阶段配置字典

    用法:
        config = PipelineConfig("configs/default.yaml")
        config._cfg["voiceprint"]["threshold"] = 0.67  # 运行时修改
        sr = config.sample_rate  # 16000
    """

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
                    # [2026-08-18] 返回带设备号的标准格式 "cuda:N"。
                    # 之前返回 "cuda" (无 :N) 会让部分库 (modelscope/sherpa) 解析失败,
                    # 警告 "Could not parse CUDA device string 'cuda'" 后 fallback。
                    return f"cuda:{torch.cuda.current_device()}"
            except ImportError:
                pass
            return "cpu"
        return device

    def get(self, key: str, default: Any = None) -> Any:
        """通用键值获取"""
        return self._cfg.get(key, default)
