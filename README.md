# 抗干扰语音指令识别系统 - 全预训练版本

流水线（全部使用官方预训练权重，无微调产物入库）：

`GTCRN → (SpEx+ 可选) → CAM++(预训练) → Paraformer`

> **当前状态**: 全预训练。声纹微调产物不随仓库上传（见 .gitignore），最后阶段微调后另行提交。
> **实测 (datasetA, 含SpEx+)**: thr=0.25, CER=0.6533, RR=0.9198, Score(80分制)=0.5066

## 纯推理仓库说明

**当前仓库不含任何训练/微调脚本与产物**，定位为"纯推理流水线"：

- 所有模块（GTCRN / SpEx+ / CAM++ / Paraformer）均为**官方预训练模型**，运行时自动加载或下载。
- **不需要训练**：训练（微调）推迟到最终阶段，按团队约定在模型全部选定、架构锁定后统一进行；训练脚本与增强数据仅在本地保留，不入库。
- 仓库内仅保留推理必需代码：`pipeline.py`、`run_inference.py`、`modules/`、`configs/`、`tools/download_spexplus.py`（权重下载）。

## 性能对比 (datasetA, 全部 80分制 = CER×40 + RR×40)

| 配置 | 阈值 | CER | RR | Score | 推理时间 |
|------|------|-----|-----|-------|---------|
| **当前仓库 (GTCRN + SpEx+ + 预训练声纹 + Paraformer)** | 0.25 | 0.6533 | 0.9198 | **50.66** | 632s |
| V2 基线 (noisereduce + Paraformer + 预训练声纹) | 0.28 | 0.5833 | 0.9367 | 54.14 | 423s(CPU) |

> 预训练声纹在 GTCRN 分布上区分力较弱（最优阈值 0.25，pos 接受率 55.5%）。
> 微调声纹可提升约 10 分（实测 59.84~60.76），将在最后阶段实施。

## 模型选型

| 模块 | 模型 | 版本说明 |
|------|------|---------|
| 降噪 | **GTCRN** (48K参数, 复数域mask, dns3 checkpoint) | 官方预训练, 随仓库上传 |
| 分离 | **SpEx+** 16kHz 目标说话人提取 | 可选启用, 权重运行时加载 |
| 声纹 | **CAM++** (3DSpeaker/ModelScope) | 官方预训练, 运行时自动下载 |
| ASR | **Paraformer** (FunASR, 热词增强) | 官方预训练, 运行时自动下载 |

## 关键说明

1. **当前为全预训练状态**: `voiceprint.cam_plus.finetuned_path = null`, 声纹使用官方预训练权重。
2. **微调产物不入库**: `finetuned_models/` 和 `runs/` 均被 .gitignore 排除; 训练脚本 `tools/train_camplus_finetune.py` 保留可复现。
3. **阈值**: 预训练模型最优 0.25 (datasetA 实测扫描); 若后续微调声纹, 需重新扫描阈值。
4. **分离**: SpEx+ 保留启用 (datasetB 多人场景需要); datasetA 上禁分离实测可再 +1 分 (RR 0.9873)。
5. **提交注意**: 官方 L20 环境需联网拉取预训练权重 (ModelScope/HF), 无法联网时需手动打包。

## 文件结构

```
voice_pipeline/
├── configs/default.yaml       # 主配置 (全预训练)
├── pipeline.py                # 4阶段流水线
├── modules/
│   ├── denoiser.py            # 降噪 (GTCRN/DeepFilterNet/noisereduce)
│   ├── gtcrn.py               # GTCRN 降噪模型实现
│   ├── separator.py           # 分离 (SpEx+/SepFormer/PassThrough)
│   ├── spexplus_separator.py  # SpEx+ 目标说话人提取
│   ├── voiceprint.py          # 声纹 (CAM++/ECAPA/WeSpeaker)
│   └── asr.py                 # ASR (Paraformer/SenseVoice/Whisper)
├── pretrained/gtcrn/          # GTCRN 预训练权重
├── tools/                     # 训练/评估/增强脚本
└── results/                   # 推理结果 (不入库)
```
