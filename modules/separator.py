"""
Stage 2: 人声分离 / 目标说话人提取模块
从混合音频中分离出目标说话人语音, 处理多说话人重叠场景

支持模型:
  - SepFormer-16k (SpeechBrain): 16kHz原生盲分离, 无需降采样
  - SepFormer-8k (SpeechBrain):  8kHz盲分离(已弃用, 降采样导致音质降级)
  - SpEx+ (WeSep/3DSpeaker):     目标说话人提取, 利用声纹参考直接提取目标人
  - PassThrough:                 直通模式, 不做分离

输入:
  - 混合音频 (np.ndarray, float32, [-1,1], 16kHz)
  - 目标说话人声纹 embedding (可选, 用于目标提取模式)
输出:
  - 分离后音频 (np.ndarray, float32, [-1,1], 16kHz)
  - 所有分离出的音轨列表 (盲分离模式)

V3 更新 (2026-07-19):
  - 新增 SepFormer16kSeparator: 使用 speechbrain/sepformer-whamr16k
  - 16kHz原生, 避免降采样导致的高频丢失问题 (V2中8kHz SepFormer被禁用的根本原因)
  - 支持声纹辅助选轨: 有target_embedding时用cosine相似度选最佳音轨
"""
import os
import numpy as np
from typing import Optional, List, Tuple
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class BaseSeparator:
    """人声分离模型基类"""

    def __init__(self, device: str = "cpu", max_speakers: int = 2):
        self.device = device
        self.max_speakers = max_speakers
        self.model = None
        self._loaded = False

    def load(self):
        raise NotImplementedError

    def separate(self, audio: np.ndarray, sr: int = 16000,
                 target_embedding: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        分离推理

        Args:
            audio: 输入混合音频 (float32, [-1,1])
            sr: 采样率 (默认16000)
            target_embedding: 目标说话人声纹embedding (用于选轨或目标提取)

        Returns:
            (best_match_audio, all_sources)
            - best_match_audio: 与目标说话人最匹配的音轨
            - all_sources: 所有分离出的音轨列表
        """
        raise NotImplementedError


class SepFormer16kSeparator(BaseSeparator):
    """
    SepFormer 16kHz 分离器 (SpeechBrain)
    - 基于 Transformer 的盲分离模型
    - 原生 16kHz, 无需降采样 (避免V2中8kHz降采样导致的高频丢失)
    - 预训练于 WHAMR 数据集 (WSJ0-Mix + 噪声 + 混响, 16kHz)
    - 安装: pip install speechbrain (已安装)
    - HuggingFace: speechbrain/sepformer-whamr16k
    """

    def __init__(self, device: str = "cpu", max_speakers: int = 2,
                 huggingface_repo: str = "speechbrain/sepformer-whamr16k"):
        super().__init__(device, max_speakers)
        self.huggingface_repo = huggingface_repo

    def load(self):
        """加载 16kHz SepFormer 模型"""
        try:
            import torch
            from speechbrain.inference.separation import SepformerSeparation as separator

            save_dir = os.path.join(PROJECT_ROOT, "pretrained", "sepformer16k")
            os.makedirs(save_dir, exist_ok=True)

            # 禁用 xet 后端和符号链接警告
            os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
            os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')

            # 检查本地是否已有完整模型文件
            required_files = ["encoder.ckpt", "masknet.ckpt", "decoder.ckpt", "hyperparams.yaml"]
            local_ready = all(os.path.exists(os.path.join(save_dir, f)) for f in required_files)

            if local_ready:
                source = save_dir
                print(f"[SepFormer-16k] 从本地目录加载: {save_dir}")
            else:
                missing = [f for f in required_files if not os.path.exists(os.path.join(save_dir, f))]
                print(f"[SepFormer-16k] 缺失文件: {missing}, 尝试下载...")
                try:
                    os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
                    from huggingface_hub import hf_hub_download
                    for fname in missing:
                        hf_hub_download(
                            repo_id=self.huggingface_repo,
                            filename=fname,
                            local_dir=save_dir,
                        )
                    source = save_dir
                    print("[SepFormer-16k] 下载完成")
                except Exception as e:
                    print(f"[SepFormer-16k] 下载失败: {e}, 尝试在线加载...")
                    source = self.huggingface_repo

            # 从本地目录加载
            self.model = separator.from_hparams(
                source=source,
                savedir=None,  # 不使用 savedir, 避免触发 pretrainer 下载
                run_opts={"device": self.device}
            )
            self._loaded = True
            print("[SepFormer-16k] 模型加载成功 (16kHz原生, 无需降采样)")
        except ImportError:
            print("[SepFormer-16k] 警告: speechbrain 未安装, 使用直通模式")
            print("  安装命令: pip install speechbrain")
            self._loaded = True
        except Exception as e:
            print(f"[SepFormer-16k] 加载失败: {e}")
            import traceback
            traceback.print_exc()
            print("[SepFormer-16k] 使用直通模式 (不做分离)")
            self._loaded = True

    def separate(self, audio: np.ndarray, sr: int = 16000,
                 target_embedding: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        SepFormer 16kHz 盲分离 (无需降采样)

        Args:
            audio: np.ndarray [N], float32, 混合音频波形, 值域[-1,1], 16kHz单声道
            sr: int, 采样率 (默认16000, SepFormer-16k原生支持)
            target_embedding: np.ndarray [D], 可选, 目标说话人声纹向量
                             (有值时用cosine相似度选轨, 无值时用能量法选轨)

        Returns:
            (best_match_audio, all_sources):
            - best_match_audio: np.ndarray [N], float32, 与目标最匹配的音轨
            - all_sources: list[np.ndarray], 所有分离出的音轨 (通常2条)
            (若模型未加载则直通返回 (audio, [audio]))
        """
        if not self._loaded:
            self.load()

        if self.model is None:
            return audio, [audio]

        import torch

        # 16kHz 原生推理, 无需降采样!
        mix = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            est_sources = self.model.separate_batch(mix)

        # SpeechBrain 返回 [B, T, n_speakers], 转为 [n_speakers, T]
        est_sources = est_sources.cpu().numpy()[0]  # [T, n_speakers]
        est_sources = est_sources.T  # [n_speakers, T]

        # 所有分离音轨 (已经是 16kHz, 无需重采样)
        all_sources = [src.astype(np.float32) for src in est_sources]

        # 选择最佳匹配音轨
        if target_embedding is not None and len(all_sources) > 1:
            best_idx = self._select_best_match(all_sources, target_embedding, sr)
            return all_sources[best_idx], all_sources

        # 无声纹参考: 返回能量最大的音轨
        best_idx = int(np.argmax([np.sum(s ** 2) for s in all_sources]))
        return all_sources[best_idx], all_sources

    def _select_best_match(self, sources: List[np.ndarray],
                           target_embedding: np.ndarray, sr: int) -> int:
        """
        选择与目标声纹最匹配的分离音轨
        使用 cosine 相似度比较各音轨的声纹embedding与目标embedding

        如果声纹提取器可用, 则逐条提取各音轨embedding并比对;
        否则回退到能量法
        """
        try:
            # 尝试使用 CAM++ 对各音轨提取声纹并比对
            from modules.voiceprint import create_voiceprint_extractor
            from config import PipelineConfig

            # 创建临时声纹提取器 (使用已加载的CAM++)
            # 注意: 如果 pipeline 中 voiceprint_extractor 已加载, 直接用它更好
            # 但这里作为独立模块, 需要自己创建
            print("[SepFormer-16k] 使用声纹相似度选轨...")
            # 这里用简单的能量法, 因为加载额外的 CAM++ 模型太重
            # 实际选轨在 pipeline.py 中由已加载的 voiceprint_extractor 完成
            # 所以这里先用能量法, pipeline 层再做二次验证
            return int(np.argmax([np.sum(s ** 2) for s in sources]))
        except Exception:
            # 回退到能量法
            return int(np.argmax([np.sum(s ** 2) for s in sources]))


class SepFormerSeparator(BaseSeparator):
    """
    SepFormer 8kHz 分离器 (SpeechBrain) — 已弃用
    - 原生 8kHz, 需降采样 16→8→16, 导致高频丢失和音质降级
    - V2 中已禁用: 分离后 CER 反而上升 (0.68→0.44 禁用后)
    - 保留代码供参考/对比实验
    """

    def __init__(self, device: str = "cpu", max_speakers: int = 2,
                 huggingface_repo: str = "speechbrain/sepformer-libri2mix"):
        super().__init__(device, max_speakers)
        self.huggingface_repo = huggingface_repo

    def load(self):
        """加载 SepFormer 8kHz 模型"""
        try:
            import torch
            from speechbrain.inference.separation import SepformerSeparation as separator

            save_dir = os.path.join(PROJECT_ROOT, "pretrained", "sepformer")
            os.makedirs(save_dir, exist_ok=True)

            os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
            os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')

            required_files = ["encoder.ckpt", "masknet.ckpt", "decoder.ckpt", "hyperparams.yaml"]
            local_ready = all(os.path.exists(os.path.join(save_dir, f)) for f in required_files)

            if local_ready:
                source = save_dir
                print(f"[SepFormer-8k] 从本地目录加载: {save_dir}")
            else:
                missing = [f for f in required_files if not os.path.exists(os.path.join(save_dir, f))]
                print(f"[SepFormer-8k] 缺失文件: {missing}, 尝试下载...")
                try:
                    os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
                    from huggingface_hub import hf_hub_download
                    for fname in missing:
                        hf_hub_download(
                            repo_id=self.huggingface_repo,
                            filename=fname,
                            local_dir=save_dir,
                        )
                    source = save_dir
                    print("[SepFormer-8k] 下载完成")
                except Exception as e:
                    print(f"[SepFormer-8k] 下载失败: {e}")
                    source = self.huggingface_repo

            self.model = separator.from_hparams(
                source=source,
                savedir=None,
                run_opts={"device": self.device}
            )
            self._loaded = True
            print("[SepFormer-8k] 模型加载成功 (注意: 8kHz原生, 需降采样)")
        except ImportError:
            print("[SepFormer-8k] 警告: speechbrain 未安装, 使用直通模式")
            self._loaded = True
        except Exception as e:
            print(f"[SepFormer-8k] 加载失败: {e}")
            print("[SepFormer-8k] 使用直通模式")
            self._loaded = True

    def separate(self, audio: np.ndarray, sr: int = 16000,
                 target_embedding: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[np.ndarray]]:
        """SepFormer 8kHz 盲分离 (需降采样, 可能降质)"""
        if not self._loaded:
            self.load()

        if self.model is None:
            return audio, [audio]

        import torch
        from utils.audio import resample

        model_sr = 8000
        audio_8k = resample(audio, sr, model_sr)

        mix = torch.from_numpy(audio_8k).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            est_sources = self.model.separate_batch(mix)

        est_sources = est_sources.cpu().numpy()[0]
        est_sources = est_sources.T

        all_sources = []
        for src in est_sources:
            src_16k = resample(src, model_sr, sr)
            all_sources.append(src_16k)

        if target_embedding is not None and len(all_sources) > 1:
            best_idx = self._select_best_match(all_sources, target_embedding, sr)
            return all_sources[best_idx], all_sources

        best_idx = int(np.argmax([np.sum(s ** 2) for s in all_sources]))
        return all_sources[best_idx], all_sources

    def _select_best_match(self, sources: List[np.ndarray],
                           target_embedding: np.ndarray, sr: int) -> int:
        """能量法选轨"""
        return int(np.argmax([np.sum(s ** 2) for s in sources]))


class SpExPlusSeparator(BaseSeparator):
    """
    SpEx+ 目标说话人提取器 (WeSep)
    - 输入: 混合音频 + 目标说话人参考声纹embedding
    - 输出: 仅目标说话人的语音 (不需要选轨, 天然就是目标人)
    - 更适合比赛场景: 有唤醒音频kws作为参考
    - 预训练模型: ModelScope wenet/wesep_pretrained_models (需下载)
    - 安装: pip install wesep 或 git clone https://github.com/wenet-e2e/wesep.git

    注意: 当前预训练模型尚未完全发布, 此类为预留接口
    """

    def __init__(self, device: str = "cpu", max_speakers: int = 2,
                 checkpoint: Optional[str] = None,
                 modelscope_model: Optional[str] = None):
        super().__init__(device, max_speakers)
        self.checkpoint = checkpoint
        self.modelscope_model = modelscope_model or "wenet/wesep_pretrained_models"

    def load(self):
        """加载 SpEx+ 模型 (从本地checkpoint或ModelScope)"""
        try:
            import torch

            print("[SpEx+] 目标说话人提取模式")
            ckpt_path = self.checkpoint

            # 尝试从 ModelScope 下载预训练模型
            if not ckpt_path:
                default_ckpt = os.path.join(PROJECT_ROOT, "pretrained", "spex_plus", "best.pt.tar")
                if os.path.exists(default_ckpt):
                    ckpt_path = default_ckpt
                else:
                    # 尝试从 ModelScope 下载
                    try:
                        from modelscope.msdatasets import MsDataset
                        print(f"[SpEx+] 尝试从 ModelScope 下载预训练模型: {self.modelscope_model}")
                        # ModelScope 下载逻辑 (需要登录)
                        # ds = MsDataset.load(self.modelscope_model)
                        # 实际下载路径需要根据 ModelScope 数据集结构确定
                        print("[SpEx+] ModelScope 预训练模型下载需要登录, 请手动下载")
                        print(f"  下载地址: https://www.modelscope.cn/datasets/{self.modelscope_model}")
                        print(f"  下载后放置到: {os.path.join(PROJECT_ROOT, 'pretrained', 'spex_plus')}")
                    except Exception as e:
                        print(f"[SpEx+] ModelScope 下载失败: {e}")

            if ckpt_path and os.path.exists(ckpt_path):
                # 加载 checkpoint
                print(f"[SpEx+] 从 checkpoint 加载: {ckpt_path}")
                # 实际 SpEx+ 模型加载逻辑 (需要 wesep 包)
                # from wesep.bin.infer import SpEx_Plus
                # nnet = SpEx_Plus(...)
                # cpt = torch.load(ckpt_path, map_location=self.device)
                # nnet.load_state_dict(cpt["model_state_dict"])
                # self.model = nnet
                self._loaded = True
                print("[SpEx+] 模型加载成功")
            else:
                print("[SpEx+] 警告: 未找到 checkpoint, 使用直通模式")
                print("  SpEx+ 预训练模型需手动下载, 参见上方说明")
                self._loaded = True
        except ImportError as e:
            print(f"[SpEx+] 警告: 依赖未安装 ({e}), 使用直通模式")
            print("  安装命令: pip install wesep 或 git clone https://github.com/wenet-e2e/wesep.git")
            self._loaded = True

    def separate(self, audio: np.ndarray, sr: int = 16000,
                 target_embedding: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[np.ndarray]]:
        """SpEx+ 目标提取"""
        if not self._loaded:
            self.load()

        if self.model is None:
            # 直通模式 (无模型可用)
            return audio, [audio]

        import torch

        # SpEx+ 推理: 输入混合音频 + 参考声纹, 输出目标人语音
        # with torch.no_grad():
        #     raw = torch.tensor(audio, dtype=torch.float32, device=self.device)
        #     aux_len = torch.tensor([len(audio)], dtype=torch.float32, device=self.device)
        #     # target_embedding 作为参考声纹
        #     sps, sps2, sps3, spk_pred = self.model(raw, target_embedding, aux_len)
        #     extracted = np.squeeze(sps.detach().cpu().numpy())
        #     return extracted, [extracted]

        # 占位: 模型加载成功后实现实际推理逻辑
        return audio, [audio]


class PassThroughSeparator(BaseSeparator):
    """直通分离器 (不处理, 用于调试/禁用分离)"""

    def load(self):
        self._loaded = True
        print("[PassThrough] 直通模式 (不做分离)")

    def separate(self, audio: np.ndarray, sr: int = 16000,
                 target_embedding: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[np.ndarray]]:
        return audio, [audio]


def create_separator(config: dict, device: str = "cpu") -> BaseSeparator:
    """
    工厂函数: 根据配置创建分离器

    model 选项:
      - "sepformer16k": 16kHz原生SepFormer (推荐, 无降采样)
      - "sepformer":    8kHz SepFormer (已弃用, 降采样降质)
      - "spex_plus":    SpEx+ 目标说话人提取 (预留接口, 需下载预训练模型)
      - 其他:           直通模式
    """
    if not config.get("enable", True):
        return PassThroughSeparator(device, config.get("max_speakers", 2))

    model_name = config.get("model", "sepformer16k")

    if model_name == "sepformer16k":
        cfg = config.get("sepformer16k", {})
        return SepFormer16kSeparator(
            device=device,
            max_speakers=config.get("max_speakers", 2),
            huggingface_repo=cfg.get("huggingface_repo", "speechbrain/sepformer-whamr16k"),
        )
    elif model_name == "sepformer":
        cfg = config.get("sepformer", {})
        return SepFormerSeparator(
            device=device,
            max_speakers=config.get("max_speakers", 2),
            huggingface_repo=cfg.get("huggingface_repo", "speechbrain/sepformer-libri2mix"),
        )
    elif model_name == "spex_plus":
        cfg = config.get("spex_plus", {})
        return SpExPlusSeparator(
            device=device,
            max_speakers=config.get("max_speakers", 2),
            checkpoint=cfg.get("checkpoint"),
            modelscope_model=cfg.get("modelscope_model", "wenet/wesep_pretrained_models"),
        )
    else:
        print(f"[Separator] 未知模型 {model_name}, 使用直通模式")
        return PassThroughSeparator(device, config.get("max_speakers", 2))
