"""
SpEx+ 16 kHz目标说话人提取器。

模型结构改编自 ex7remum/DLA_Speaker_Separation 的 MIT 许可实现，
其实现又基于 xuchenglin28/speaker_extraction_SpEx。推理输入为：
  - 混合/识别音频 [B, T]
  - 目标说话人的参考/唤醒音频 [B, T_ref]
  - 参考音频有效长度 [B]

权重来源与复现信息由 pretrained/spex_plus/SOURCE.txt 记录。
"""

import hashlib
import os
import sys
import types
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class GlobalLayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.beta = nn.Parameter(torch.zeros(dim, 1))
        self.gamma = nn.Parameter(torch.ones(dim, 1))

    def forward(self, x):
        mean = torch.mean(x, (1, 2), keepdim=True)
        var = torch.mean((x - mean) ** 2, (1, 2), keepdim=True)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta


class TCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int = 256,
        conv_channels: int = 512,
        kernel_size: int = 3,
        dilation: int = 1,
    ):
        super().__init__()
        self.conv1x1 = nn.Conv1d(in_channels, conv_channels, 1)
        self.prelu1 = nn.PReLU()
        self.norm1 = GlobalLayerNorm(conv_channels)
        padding = (dilation * (kernel_size - 1)) // 2
        self.dconv = nn.Conv1d(
            conv_channels,
            conv_channels,
            kernel_size,
            groups=conv_channels,
            padding=padding,
            dilation=dilation,
            bias=True,
        )
        self.prelu2 = nn.PReLU()
        self.norm2 = GlobalLayerNorm(conv_channels)
        self.sconv = nn.Conv1d(conv_channels, in_channels, 1, bias=True)

    def forward(self, x):
        y = self.conv1x1(x)
        y = self.norm1(self.prelu1(y))
        y = self.dconv(y)
        y = self.norm2(self.prelu2(y))
        return self.sconv(y) + x


class TCNBlockSpk(nn.Module):
    def __init__(
        self,
        in_channels: int = 256,
        spk_embed_dim: int = 256,
        conv_channels: int = 512,
        kernel_size: int = 3,
        dilation: int = 1,
    ):
        super().__init__()
        self.conv1x1 = nn.Conv1d(
            in_channels + spk_embed_dim, conv_channels, 1
        )
        self.prelu1 = nn.PReLU()
        self.norm1 = GlobalLayerNorm(conv_channels)
        padding = (dilation * (kernel_size - 1)) // 2
        self.dconv = nn.Conv1d(
            conv_channels,
            conv_channels,
            kernel_size,
            groups=conv_channels,
            padding=padding,
            dilation=dilation,
            bias=True,
        )
        self.prelu2 = nn.PReLU()
        self.norm2 = GlobalLayerNorm(conv_channels)
        self.sconv = nn.Conv1d(conv_channels, in_channels, 1, bias=True)

    def forward(self, x, ref):
        repeated_ref = ref.unsqueeze(-1).repeat(1, 1, x.shape[-1])
        y = self.conv1x1(torch.cat([x, repeated_ref], dim=1))
        y = self.norm1(self.prelu1(y))
        y = self.dconv(y)
        y = self.norm2(self.prelu2(y))
        return self.sconv(y) + x


class ResBlock(nn.Module):
    def __init__(self, in_dims: int, out_dims: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dims, out_dims, 1, bias=False)
        self.conv2 = nn.Conv1d(out_dims, out_dims, 1, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(out_dims)
        self.batch_norm2 = nn.BatchNorm1d(out_dims)
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.maxpool = nn.MaxPool1d(3)
        self.downsample = in_dims != out_dims
        if self.downsample:
            self.conv_downsample = nn.Conv1d(
                in_dims, out_dims, 1, bias=False
            )

    def forward(self, x):
        y = self.prelu1(self.batch_norm1(self.conv1(x)))
        y = self.batch_norm2(self.conv2(y))
        y = y + (self.conv_downsample(x) if self.downsample else x)
        return self.maxpool(self.prelu2(y))


class SpexPlus(nn.Module):
    """与公开检查点匹配的 SpEx+ 网络结构。"""

    def __init__(
        self,
        short_kernel_size: int = 20,
        middle_kernel_size: int = 80,
        long_kernel_size: int = 160,
        num_feats: int = 256,
        num_blocks: int = 8,
        n_proj_channels: int = 256,
        hidden_dim: int = 512,
        tcn_kernel_size: int = 3,
        num_spks: int = 90,
        spk_embed_dim: int = 256,
    ):
        super().__init__()
        self.short_kernel_size = short_kernel_size
        self.middle_kernel_size = middle_kernel_size
        self.long_kernel_size = long_kernel_size

        stride = short_kernel_size // 2
        self.encoder_short = nn.Conv1d(
            1, num_feats, short_kernel_size, stride=stride
        )
        self.encoder_middle = nn.Conv1d(
            1, num_feats, middle_kernel_size, stride=stride
        )
        self.encoder_long = nn.Conv1d(
            1, num_feats, long_kernel_size, stride=stride
        )
        self.layer_norm = nn.LayerNorm(3 * num_feats)
        self.proj = nn.Conv1d(3 * num_feats, n_proj_channels, 1)

        self.conv_block_1 = TCNBlockSpk(
            in_channels=n_proj_channels,
            spk_embed_dim=spk_embed_dim,
            conv_channels=hidden_dim,
            kernel_size=tcn_kernel_size,
        )
        self.conv_block_1_other = self._build_stacks(
            num_blocks, n_proj_channels, hidden_dim, tcn_kernel_size
        )
        self.conv_block_2 = TCNBlockSpk(
            in_channels=n_proj_channels,
            spk_embed_dim=spk_embed_dim,
            conv_channels=hidden_dim,
            kernel_size=tcn_kernel_size,
        )
        self.conv_block_2_other = self._build_stacks(
            num_blocks, n_proj_channels, hidden_dim, tcn_kernel_size
        )
        self.conv_block_3 = TCNBlockSpk(
            in_channels=n_proj_channels,
            spk_embed_dim=spk_embed_dim,
            conv_channels=hidden_dim,
            kernel_size=tcn_kernel_size,
        )
        self.conv_block_3_other = self._build_stacks(
            num_blocks, n_proj_channels, hidden_dim, tcn_kernel_size
        )
        self.conv_block_4 = TCNBlockSpk(
            in_channels=n_proj_channels,
            spk_embed_dim=spk_embed_dim,
            conv_channels=hidden_dim,
            kernel_size=tcn_kernel_size,
        )
        self.conv_block_4_other = self._build_stacks(
            num_blocks, n_proj_channels, hidden_dim, tcn_kernel_size
        )

        self.mask1 = nn.Conv1d(n_proj_channels, num_feats, 1)
        self.mask2 = nn.Conv1d(n_proj_channels, num_feats, 1)
        self.mask3 = nn.Conv1d(n_proj_channels, num_feats, 1)
        self.decoder_short = nn.ConvTranspose1d(
            num_feats, 1, short_kernel_size, stride=stride
        )
        self.decoder_middle = nn.ConvTranspose1d(
            num_feats, 1, middle_kernel_size, stride=stride
        )
        self.decoder_long = nn.ConvTranspose1d(
            num_feats, 1, long_kernel_size, stride=stride
        )

        self.layer_norm_spk = nn.LayerNorm(3 * num_feats)
        self.spk_encoder = nn.Sequential(
            nn.Conv1d(3 * num_feats, n_proj_channels, 1),
            ResBlock(n_proj_channels, n_proj_channels),
            ResBlock(n_proj_channels, hidden_dim),
            ResBlock(hidden_dim, hidden_dim),
            nn.Conv1d(hidden_dim, spk_embed_dim, 1),
        )
        self.class_head = nn.Linear(spk_embed_dim, num_spks)

    @staticmethod
    def _build_stacks(num_blocks, in_channels, conv_channels, kernel_size):
        return nn.Sequential(
            *[
                TCNBlock(
                    in_channels=in_channels,
                    conv_channels=conv_channels,
                    kernel_size=kernel_size,
                    dilation=2 ** block,
                )
                for block in range(1, num_blocks)
            ]
        )

    def _multi_scale_encode(self, audio):
        w1 = F.relu(self.encoder_short(audio))
        frames = w1.shape[-1]
        input_length = audio.shape[-1]
        stride = self.short_kernel_size // 2
        middle_length = (
            (frames - 1) * stride + self.middle_kernel_size
        )
        long_length = (frames - 1) * stride + self.long_kernel_size
        w2 = F.relu(
            self.encoder_middle(
                F.pad(audio, (0, middle_length - input_length))
            )
        )
        w3 = F.relu(
            self.encoder_long(F.pad(audio, (0, long_length - input_length)))
        )
        return w1, w2, w3

    def forward(self, audio_mix, audio_ref, len_ref):
        mix = audio_mix
        w1, w2, w3 = self._multi_scale_encode(mix)
        encoded = torch.cat([w1, w2, w3], dim=1)
        encoded = self.layer_norm(encoded.transpose(1, 2)).transpose(1, 2)
        encoded = self.proj(encoded)

        ref_w1, ref_w2, ref_w3 = self._multi_scale_encode(audio_ref)
        ref = torch.cat([ref_w1, ref_w2, ref_w3], dim=1)
        ref = self.layer_norm_spk(ref.transpose(1, 2)).transpose(1, 2)
        ref = self.spk_encoder(ref)
        ref_frames = (
            (len_ref - self.short_kernel_size)
            // (self.short_kernel_size // 2)
            + 1
        )
        ref_frames = ((ref_frames // 3) // 3) // 3
        ref = torch.sum(ref, dim=-1) / ref_frames.view(-1, 1).float()

        encoded = self.conv_block_1(encoded, ref)
        encoded = self.conv_block_1_other(encoded)
        encoded = self.conv_block_2(encoded, ref)
        encoded = self.conv_block_2_other(encoded)
        encoded = self.conv_block_3(encoded, ref)
        encoded = self.conv_block_3_other(encoded)
        encoded = self.conv_block_4(encoded, ref)
        encoded = self.conv_block_4_other(encoded)

        input_length = mix.shape[-1]
        s1 = self.decoder_short(w1 * F.relu(self.mask1(encoded))).squeeze(1)
        s1 = F.pad(s1, (0, input_length - s1.shape[-1]))
        s2 = self.decoder_middle(w2 * F.relu(self.mask2(encoded))).squeeze(1)
        s2 = F.pad(s2[:, :input_length], (0, input_length - s2[:, :input_length].shape[-1]))
        s3 = self.decoder_long(w3 * F.relu(self.mask3(encoded))).squeeze(1)
        s3 = F.pad(s3[:, :input_length], (0, input_length - s3[:, :input_length].shape[-1]))
        return {
            "s1": s1,
            "s2": s2,
            "s3": s3,
            "logits": self.class_head(ref),
        }


def _resolve_project_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checkpoint_weights(path: str) -> dict:
    """
    以 weights_only=True 安全读取旧版训练器保存的检查点。

    文件中包含一个 ConfigParser 元数据对象。这里仅注册空壳类型以读取
    张量和基础类型，不执行检查点内的任意 Python 代码。
    """
    module_names = ["hw_ss", "hw_ss.utils", "hw_ss.utils.parse_config"]
    previous = {name: sys.modules.get(name) for name in module_names}
    try:
        hw_ss_module = types.ModuleType("hw_ss")
        utils_module = types.ModuleType("hw_ss.utils")
        parse_module = types.ModuleType("hw_ss.utils.parse_config")
        config_parser = type("ConfigParser", (), {})
        config_parser.__module__ = "hw_ss.utils.parse_config"
        parse_module.ConfigParser = config_parser
        sys.modules.update(
            {
                "hw_ss": hw_ss_module,
                "hw_ss.utils": utils_module,
                "hw_ss.utils.parse_config": parse_module,
            }
        )
        with torch.serialization.safe_globals([config_parser]):
            checkpoint = torch.load(
                path, map_location="cpu", weights_only=True
            )
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module

    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise RuntimeError("检查点中未找到 state_dict")
    return checkpoint


class SpExPlusSeparator:
    """流水线适配器：用每条样本的唤醒音频提取目标说话人。"""

    is_target_extractor = True

    def __init__(
        self,
        device: str = "cpu",
        max_speakers: int = 2,
        checkpoint: str = "pretrained/spex_plus/checkpoint.pth",
        sample_rate: int = 16000,
        num_spks: int = 90,
        expected_sha256: Optional[str] = None,
        allow_fallback: bool = False,
    ):
        self.device = device
        self.max_speakers = max_speakers
        self.checkpoint = _resolve_project_path(checkpoint)
        self.sample_rate = int(sample_rate)
        self.num_spks = int(num_spks)
        self.expected_sha256 = (
            expected_sha256.lower() if expected_sha256 else None
        )
        self.allow_fallback = allow_fallback
        self.model = None
        self.reference_audio = None
        self.reference_sr = None
        self._loaded = False

    def set_reference_audio(self, audio: np.ndarray, sr: int = 16000):
        self.reference_audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        self.reference_sr = int(sr)

    def load(self):
        try:
            if not os.path.isfile(self.checkpoint):
                raise FileNotFoundError(
                    f"未找到 SpEx+ 权重: {self.checkpoint}"
                )
            actual_sha256 = _sha256(self.checkpoint)
            if (
                self.expected_sha256
                and actual_sha256 != self.expected_sha256
            ):
                raise RuntimeError(
                    "SpEx+ 权重 SHA256 不匹配: "
                    f"期望 {self.expected_sha256}, 实际 {actual_sha256}"
                )

            checkpoint = _load_checkpoint_weights(self.checkpoint)
            model = SpexPlus(num_spks=self.num_spks)
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            self.model = model.to(self.device).eval()
            self._loaded = True
            print(
                "[SpEx+] 模型加载成功 "
                f"(16kHz, epoch={checkpoint.get('epoch', 'unknown')}, "
                f"SHA256={actual_sha256[:12]}...)"
            )
        except Exception as exc:
            if not self.allow_fallback:
                raise RuntimeError(
                    "[SpEx+] 模型不可用，已停止流水线，防止误用直通结果"
                ) from exc
            print(f"[SpEx+] 警告: {exc}; 使用直通模式")
            self.model = None
            self._loaded = True

    @staticmethod
    def _resample(audio: np.ndarray, orig_sr: int, new_sr: int):
        if orig_sr == new_sr:
            return audio
        from scipy.signal import resample_poly
        from math import gcd

        common = gcd(orig_sr, new_sr)
        return resample_poly(
            audio, new_sr // common, orig_sr // common
        ).astype(np.float32)

    def separate(
        self,
        audio: np.ndarray,
        sr: int = 16000,
        target_embedding: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        if not self._loaded:
            self.load()
        if self.model is None:
            return audio, [audio]
        if self.reference_audio is None:
            raise RuntimeError(
                "[SpEx+] 缺少目标说话人参考音频；"
                "请先调用 set_reference_audio"
            )

        original_length = len(audio)
        mix = self._resample(
            np.asarray(audio, dtype=np.float32).reshape(-1),
            int(sr),
            self.sample_rate,
        )
        ref = self._resample(
            self.reference_audio,
            int(self.reference_sr),
            self.sample_rate,
        )
        if len(ref) < 800:
            ref = np.pad(ref, (0, 800 - len(ref)))

        mix_tensor = (
            torch.from_numpy(mix).unsqueeze(0).unsqueeze(0).to(self.device)
        )
        ref_tensor = (
            torch.from_numpy(ref).unsqueeze(0).unsqueeze(0).to(self.device)
        )
        ref_length = torch.tensor(
            [len(ref)], dtype=torch.long, device=self.device
        )
        with torch.inference_mode():
            output = self.model(mix_tensor, ref_tensor, ref_length)["s1"]
        extracted = output[0].detach().cpu().numpy().astype(np.float32)

        if self.sample_rate != int(sr):
            extracted = self._resample(
                extracted, self.sample_rate, int(sr)
            )
        if len(extracted) < original_length:
            extracted = np.pad(
                extracted, (0, original_length - len(extracted))
            )
        extracted = extracted[:original_length]
        peak = float(np.max(np.abs(extracted))) if extracted.size else 0.0
        if peak > 1.0:
            extracted = extracted / peak * 0.99
        return extracted, [extracted]
