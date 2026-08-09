"""
Stage 3: 声纹鉴别模块
从唤醒音频提取目标说话人声纹, 与分离后音频比对, 决定接受/拒识

支持模型:
  - ECAPA-TDNN (SpeechBrain): 性能/速度均衡, VoxCeleb 预训练
  - CAM++ (3DSpeaker/ModelScope): 阿里开源, 中文场景优化
  - ERes2NetV2 (3DSpeaker/ModelScope): 增强残差网络, 中文场景优化
  - ResNetSE (WeSpeaker/sherpa-onnx): 轻量 ONNX 推理, CN-Celeb 训练
  - Ensemble (3-model Z-score): CAM++ + ERes2NetV2 + ResNetSE 加权融合

输入:
  - 唤醒音频 kws (np.ndarray)  → 提取参考声纹
  - 分离后音频 (np.ndarray)     → 提取待验证声纹
输出:
  - similarity: cosine 相似度 (float)
  - is_target: 是否接受 (bool), 基于阈值判定
"""
import os
import numpy as np
from typing import Optional, Tuple
import sys

# === 旧版 PyTorch 2.5 兼容性修复 ===
# modelscope/transformers 的旧版本需要 FSDP2 API
# (CPUOffloadPolicy, MixedPrecisionPolicy)，但 PyTorch 2.5.1 的 fsdp
# 顶层模块未导出这些类。PyTorch 2.6+ 已不需要这个补丁，而且主动导入
# FSDP 会触发 SpeechBrain 可选依赖的延迟导入，因此仅在 2.5 及更旧版本启用。
try:
    import torch

    torch_version = tuple(
        int(part) for part in torch.__version__.split("+", 1)[0].split(".")[:2]
    )
    if torch_version <= (2, 5):
        import torch.distributed.fsdp as _fsdp

        if not hasattr(_fsdp, "CPUOffloadPolicy"):
            class _CPUOffloadPolicy:
                def __init__(self, *args, **kwargs):
                    pass

            _fsdp.CPUOffloadPolicy = _CPUOffloadPolicy

        if not hasattr(_fsdp, "MixedPrecisionPolicy"):
            class _MixedPrecisionPolicy:
                def __init__(self, *args, **kwargs):
                    pass

            _fsdp.MixedPrecisionPolicy = _MixedPrecisionPolicy
except (ImportError, ValueError):
    # torch 尚未安装或版本字符串无法解析时，不阻断模块导入；
    # 后续模型 load() 会给出实际依赖错误。
    pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class BaseVoiceprintExtractor:
    """声纹提取模型基类"""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = None
        self._loaded = False

    def load(self):
        raise NotImplementedError

    def extract(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        提取声纹 embedding

        Args:
            audio: 输入音频, float32
            sr: 采样率

        Returns:
            embedding: 归一化后的声纹向量, shape [embedding_dim]
        """
        raise NotImplementedError

    def verify(self, audio_enroll: np.ndarray, audio_test: np.ndarray,
               sr: int = 16000, threshold: float = 0.5) -> Tuple[float, bool]:
        """
        声纹验证: 比较两段音频的声纹相似度

        Args:
            audio_enroll: 注册音频 (唤醒词)
            audio_test: 待验证音频 (分离后)
            sr: 采样率
            threshold: 接受阈值

        Returns:
            (similarity, is_target)
        """
        emb_enroll = self.extract(audio_enroll, sr)
        emb_test = self.extract(audio_test, sr)

        # cosine similarity
        similarity = float(np.dot(emb_enroll, emb_test) /
                          (np.linalg.norm(emb_enroll) * np.linalg.norm(emb_test) + 1e-8))

        is_target = similarity >= threshold
        return similarity, is_target


class ECAPA_TDNN_Extractor(BaseVoiceprintExtractor):
    """
    ECAPA-TDNN 声纹提取器 (SpeechBrain)
    - VoxCeleb 预训练, 192维 embedding
    - 性能/速度均衡, 比赛主流选择
    - 安装: pip install speechbrain
    """

    def __init__(self, device: str = "cpu",
                 huggingface_repo: str = "speechbrain/spkrec-ecapa-voxceleb",
                 embedding_dim: int = 192):
        super().__init__(device)
        self.huggingface_repo = huggingface_repo
        self.embedding_dim = embedding_dim

    def load(self):
        """加载 ECAPA-TDNN 模型"""
        try:
            import torch
            from speechbrain.inference.speaker import EncoderClassifier

            # 先用 huggingface_hub 下载模型到本地, 避免 SYMLINK 问题
            save_dir = os.path.join(PROJECT_ROOT, "pretrained", "ecapa_tdnn")
            os.makedirs(save_dir, exist_ok=True)

            try:
                from huggingface_hub import snapshot_download
                local_path = snapshot_download(
                    repo_id=self.huggingface_repo,
                    local_dir=save_dir,
                    local_dir_use_symlinks=False,  # Windows 不用 symlink
                )
                source = local_path
            except Exception as e:
                print(f"[ECAPA-TDNN] HuggingFace 下载失败: {e}")
                print("[ECAPA-TDNN] 尝试直接从 SpeechBrain 加载...")
                source = self.huggingface_repo

            self.model = EncoderClassifier.from_hparams(
                source=source,
                savedir=save_dir,
                run_opts={"device": self.device}
            )
            self._loaded = True
            print("[ECAPA-TDNN] 模型加载成功")
        except ImportError:
            print("[ECAPA-TDNN] 警告: speechbrain 未安装, 使用直通模式")
            print("  安装命令: pip install speechbrain")
            self._loaded = True
        except Exception as e:
            print(f"[ECAPA-TDNN] 加载失败: {e}")
            print("[ECAPA-TDNN] 使用直通模式 (声纹比对将返回固定结果)")
            self._loaded = True

    def extract(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """提取 ECAPA-TDNN 声纹"""
        if not self._loaded:
            self.load()

        if self.model is None:
            # 直通模式: 返回随机向量 (仅用于调试)
            rng = np.random.RandomState(42)
            return rng.randn(self.embedding_dim).astype(np.float32)

        import torch
        # SpeechBrain ECAPA-TDNN 原生 16kHz
        wav = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model.encode_batch(wav)
        embedding = embedding.cpu().numpy()[0]
        # L2 归一化
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        return embedding


class CAMPlusExtractor(BaseVoiceprintExtractor):
    """
    CAM++ 声纹提取器 (3DSpeaker / ModelScope)
    - 阿里开源, 中文场景优化
    - 192维 embedding
    - 安装: pip install modelscope
    """

    def __init__(self, device: str = "cpu",
                 model_id: str = "iic/speech_campplus_sv_zh-cn_16k-common",
                 embedding_dim: int = 192,
                 finetuned_path: Optional[str] = None):
        super().__init__(device)
        self.model_id = model_id
        self.embedding_dim = embedding_dim
        # 微调权重路径 (datasetA增强数据对比学习微调, tools/train_camplus_finetune.py 产出)
        # 为 None 时使用官方预训练权重 (zero-shot)
        self.finetuned_path = finetuned_path

    def load(self):
        """加载 CAM++ 模型 (通过 ModelScope pipeline, 可选加载微调权重)"""
        try:
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks

            sv_pipeline = pipeline(
                task=Tasks.speaker_verification,
                model=self.model_id,
            )
            # 保存底层模型对象, 直接调用可获取 embedding
            self.model = sv_pipeline.model
            self.pipeline = sv_pipeline
            self._loaded = True
            print(f"[CAM++] 模型加载成功 (threshold={sv_pipeline.thr})")

            # 加载微调权重 (覆盖 embedding_model)
            if self.finetuned_path:
                # 相对路径解析为项目根目录相对 (避免依赖运行目录)
                ft_path = self.finetuned_path
                if not os.path.isabs(ft_path):
                    ft_path = os.path.join(PROJECT_ROOT, ft_path)
                if os.path.exists(ft_path):
                    import torch
                    state = torch.load(ft_path, map_location="cpu")
                    self.model.embedding_model.load_state_dict(state, strict=True)
                    self.model.embedding_model.eval()
                    print(f"[CAM++] 已加载微调权重: {ft_path}")
                else:
                    print(f"[CAM++] 警告: 微调权重不存在 {ft_path}, 使用预训练权重")
        except ImportError as e:
            print(f"[CAM++] 警告: modelscope 导入失败 ({e}), 使用直通模式")
            print("  可能缺少依赖, 尝试: pip install addict datasets simplejson")
            self._loaded = True
        except Exception as e:
            print(f"[CAM++] 加载失败: {e}, 使用直通模式")
            self._loaded = True

    def extract(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        提取 CAM++ 声纹 embedding (直接调用底层模型, 无需临时文件)

        Args:
            audio: np.ndarray [N], float32, 音频波形, 值域[-1,1], 单声道
            sr: int, 采样率 (默认16000, CAM++要求16kHz)

        Returns:
            embedding: np.ndarray [192], float32, L2归一化后的声纹向量
                      (若模型未加载则返回随机向量, 不中断流水线)
        """
        if not self._loaded:
            self.load()

        if self.model is None:
            rng = np.random.RandomState(42)
            return rng.randn(self.embedding_dim).astype(np.float32)

        import torch

        # 确保 float32
        audio_f32 = audio.astype(np.float32)
        wav = torch.from_numpy(audio_f32)

        with torch.no_grad():
            embedding = self.model(wav)  # [1, 192]

        embedding = embedding.cpu().numpy().flatten()
        # L2 归一化
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        return embedding


class ERes2NetV2Extractor(BaseVoiceprintExtractor):
    """
    ERes2NetV2 声纹提取器 (3DSpeaker / ModelScope)
    - 增强残差网络, 中文场景优化
    - 192维 embedding
    - 安装: pip install modelscope
    """

    def __init__(self, device: str = "cpu",
                 model_id: str = "iic/speech_eres2netv2_sv_zh-cn_16k-common",
                 embedding_dim: int = 192,
                 finetuned_path: Optional[str] = None):
        super().__init__(device)
        self.model_id = model_id
        self.embedding_dim = embedding_dim
        # 微调权重路径 (datasetA增强数据对比学习微调, tools/train_eres2netv2_finetune.py 产出)
        # 为 None 时使用官方预训练权重 (zero-shot)
        self.finetuned_path = finetuned_path

    def load(self):
        """加载 ERes2NetV2 模型 (通过 ModelScope pipeline, 可选加载微调权重)"""
        try:
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks

            sv_pipeline = pipeline(
                task=Tasks.speaker_verification,
                model=self.model_id,
            )
            self.model = sv_pipeline.model
            self.pipeline = sv_pipeline
            self._loaded = True
            print(f"[ERes2NetV2] 模型加载成功 (threshold={sv_pipeline.thr})")

            # 加载微调权重 (覆盖 embedding_model)
            if self.finetuned_path:
                ft_path = self.finetuned_path
                if not os.path.isabs(ft_path):
                    ft_path = os.path.join(PROJECT_ROOT, ft_path)
                if os.path.exists(ft_path):
                    import torch
                    state = torch.load(ft_path, map_location="cpu")
                    self.model.embedding_model.load_state_dict(state, strict=True)
                    self.model.embedding_model.eval()
                    print(f"[ERes2NetV2] 已加载微调权重: {ft_path}")
                else:
                    print(f"[ERes2NetV2] 警告: 微调权重不存在 {ft_path}, 使用预训练权重")
        except ImportError as e:
            print(f"[ERes2NetV2] 警告: modelscope 导入失败 ({e}), 使用直通模式")
            self._loaded = True
        except Exception as e:
            print(f"[ERes2NetV2] 加载失败: {e}, 使用直通模式")
            self._loaded = True

    def extract(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        提取 ERes2NetV2 声纹 embedding (直接调用底层模型, 无需临时文件)

        Args:
            audio: np.ndarray [N], float32, 音频波形, 值域[-1,1], 单声道
            sr: int, 采样率 (默认16000)

        Returns:
            embedding: np.ndarray [192], float32, L2归一化后的声纹向量
        """
        if not self._loaded:
            self.load()

        if self.model is None:
            rng = np.random.RandomState(42)
            return rng.randn(self.embedding_dim).astype(np.float32)

        import torch

        audio_f32 = audio.astype(np.float32)
        wav = torch.from_numpy(audio_f32)

        with torch.no_grad():
            embedding = self.model(wav)  # [1, 192]

        embedding = embedding.cpu().numpy().flatten()
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        return embedding


class ResNetSEExtractor(BaseVoiceprintExtractor):
    """
    ResNetSE 声纹提取器 (WeSpeaker / sherpa-onnx)
    - WeSpeaker ResNet34, CN-Celeb 训练
    - 256维 embedding, ONNX 推理, 极轻量
    - 安装: pip install sherpa-onnx
    """

    def __init__(self, device: str = "cpu",
                 model_path: Optional[str] = None,
                 num_threads: int = 4,
                 embedding_dim: int = 256):
        super().__init__(device)
        self.model_path = model_path or os.path.join(
            PROJECT_ROOT, "pretrained", "wespeaker_resnet34",
            "wespeaker_zh_cnceleb_resnet34.onnx"
        )
        self.num_threads = num_threads
        self.embedding_dim = embedding_dim

    def load(self):
        """加载 ResNetSE 模型 (sherpa-onnx SpeakerEmbeddingExtractor)"""
        try:
            import sherpa_onnx

            cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig()
            cfg.model = self.model_path
            cfg.num_threads = self.num_threads
            self.model = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
            self.embedding_dim = self.model.dim
            self._loaded = True
            print(f"[ResNetSE] 模型加载成功 (dim={self.model.dim})")
        except ImportError:
            print("[ResNetSE] 警告: sherpa-onnx 未安装, 使用直通模式")
            print("  安装命令: pip install sherpa-onnx")
            self._loaded = True
        except Exception as e:
            print(f"[ResNetSE] 加载失败: {e}, 使用直通模式")
            self._loaded = True

    def extract(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        提取 ResNetSE 声纹 embedding (sherpa-onnx 推理)

        Args:
            audio: np.ndarray [N], float32, 音频波形, 值域[-1,1], 单声道
            sr: int, 采样率 (默认16000)

        Returns:
            embedding: np.ndarray [256], float32, L2归一化后的声纹向量
        """
        if not self._loaded:
            self.load()

        if self.model is None:
            rng = np.random.RandomState(42)
            return rng.randn(self.embedding_dim).astype(np.float32)

        import sherpa_onnx

        stream = self.model.create_stream()
        stream.accept_waveform(sr, audio.tolist())
        stream.input_finished()
        embedding = np.array(self.model.compute(stream), dtype=np.float32)

        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        return embedding


class EnsembleVoiceprintExtractor:
    """
    三模型 Z-score 集成声纹提取器
    - CAM++ (0.4) + ERes2NetV2 (0.4) + ResNetSE (0.2)
    - Z-score 归一化后加权融合, 阈值 -0.17
    - 80分=59.33, 100分=74.58 (datasetA 最优方案)
    """

    def __init__(self, device: str = "cpu",
                 weights: Tuple[float, float, float] = (0.4, 0.4, 0.2),
                 threshold: float = -0.17,
                 cam_config: Optional[dict] = None,
                 eres_config: Optional[dict] = None,
                 rnet_config: Optional[dict] = None):
        self.device = device
        self.weights = weights
        self.threshold = threshold
        self.cam = CAMPlusExtractor(device=device, **(cam_config or {}))
        self.eres = ERes2NetV2Extractor(device=device, **(eres_config or {}))
        self.rnet = ResNetSEExtractor(device=device, **(rnet_config or {}))
        self._loaded = False

    def load(self):
        """加载三个声纹模型"""
        print("[Ensemble] 加载三模型 (CAM++ + ERes2NetV2 + ResNetSE)...")
        self.cam.load()
        self.eres.load()
        self.rnet.load()
        self._loaded = True
        print(f"[Ensemble] 三模型加载完成, 权重={self.weights}, 阈值={self.threshold}")

    def extract_all(self, audio: np.ndarray, sr: int = 16000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        用三个模型分别提取声纹 embedding

        Returns:
            (cam_emb, eres_emb, rnet_emb): 三个模型的 embedding
        """
        if not self._loaded:
            self.load()
        return (
            self.cam.extract(audio, sr),
            self.eres.extract(audio, sr),
            self.rnet.extract(audio, sr),
        )

    @staticmethod
    def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        """计算 cosine 相似度"""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    @staticmethod
    def zscore_fuse(sims_list: list, weights: Tuple[float, float, float]) -> np.ndarray:
        """
        Z-score 归一化 + 加权融合 (批量处理)

        Args:
            sims_list: [(cam_sims, eres_sims, rnet_sims)] 每个元素是 np.array
            weights: (w_cam, w_eres, w_rnet)

        Returns:
            fused_sims: np.array, 融合后的 Z-score 相似度
        """
        cam_sims = np.array([s[0] for s in sims_list])
        eres_sims = np.array([s[1] for s in sims_list])
        rnet_sims = np.array([s[2] for s in sims_list])

        cam_z = (cam_sims - cam_sims.mean()) / (cam_sims.std() + 1e-8)
        eres_z = (eres_sims - eres_sims.mean()) / (eres_sims.std() + 1e-8)
        rnet_z = (rnet_sims - rnet_sims.mean()) / (rnet_sims.std() + 1e-8)

        fused = weights[0] * cam_z + weights[1] * eres_z + weights[2] * rnet_z
        return fused


def create_voiceprint_extractor(config: dict, device: str = "cpu") -> BaseVoiceprintExtractor:
    """
    工厂函数: 根据配置创建声纹提取器

    Args:
        config: dict, voiceprint配置字典 (含model/cam_plus/eres2netv2等字段)
        device: str, "cuda" 或 "cpu"

    Returns:
        BaseVoiceprintExtractor 子类实例
        (CAMPlusExtractor / ERes2NetV2Extractor / ResNetSEExtractor /
         EnsembleVoiceprintExtractor / ECAPA_TDNN_Extractor)
    """
    model_name = config.get("model", "ensemble")

    if model_name == "ensemble":
        ens_cfg = config.get("ensemble", {})
        return EnsembleVoiceprintExtractor(
            device=device,
            weights=tuple(ens_cfg.get("weights", [0.4, 0.4, 0.2])),
            threshold=ens_cfg.get("threshold", -0.17),
            cam_config=_build_cam_config(config.get("cam_plus", {})),
            eres_config=_build_eres_config(config.get("eres2netv2", {})),
            rnet_config=_build_rnet_config(config.get("resnetse", {})),
        )
    elif model_name == "ecapa_tdnn":
        cfg = config.get("ecapa_tdnn", {})
        return ECAPA_TDNN_Extractor(
            device=device,
            huggingface_repo=cfg.get("huggingface_repo", "speechbrain/spkrec-ecapa-voxceleb"),
            embedding_dim=cfg.get("embedding_dim", 192),
        )
    elif model_name == "cam_plus":
        return CAMPlusExtractor(device=device, **_build_cam_config(config.get("cam_plus", {})))
    elif model_name == "eres2netv2":
        return ERes2NetV2Extractor(device=device, **_build_eres_config(config.get("eres2netv2", {})))
    elif model_name == "resnetse":
        return ResNetSEExtractor(device=device, **_build_rnet_config(config.get("resnetse", {})))
    else:
        print(f"[Voiceprint] 未知模型 {model_name}, 使用 Ensemble")
        return EnsembleVoiceprintExtractor(device=device)


def _build_cam_config(cfg: dict) -> dict:
    """从配置字典构建 CAMPlusExtractor 参数"""
    return {
        "model_id": cfg.get("model_id", "iic/speech_campplus_sv_zh-cn_16k-common"),
        "embedding_dim": cfg.get("embedding_dim", 192),
        "finetuned_path": cfg.get("finetuned_path"),
    }


def _build_eres_config(cfg: dict) -> dict:
    """从配置字典构建 ERes2NetV2Extractor 参数"""
    return {
        "model_id": cfg.get("model_id", "iic/speech_eres2netv2_sv_zh-cn_16k-common"),
        "embedding_dim": cfg.get("embedding_dim", 192),
        "finetuned_path": cfg.get("finetuned_path"),
    }


def _build_rnet_config(cfg: dict) -> dict:
    """从配置字典构建 ResNetSEExtractor 参数"""
    return {
        "model_path": cfg.get("model_path"),
        "num_threads": cfg.get("num_threads", 4),
        "embedding_dim": cfg.get("embedding_dim", 256),
    }


def extract_embedding(audio: np.ndarray, sr: int = 16000,
                      extractor: Optional[BaseVoiceprintExtractor] = None) -> np.ndarray:
    """
    便捷函数: 提取声纹 embedding
    可被其他模块 (如 separator) 调用
    """
    if extractor is None:
        extractor = ECAPA_TDNN_Extractor()
        extractor.load()
    return extractor.extract(audio, sr)
