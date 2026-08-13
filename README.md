# 抗干扰语音指令识别系统

复杂交互场景下的抗干扰语音指令识别流水线，串联降噪、人声分离、声纹鉴别与语音识别四个阶段。

## ⚡ V8 最终方案 (2026-08-12, release/v8-dual-full 分支)

```
Renoise (降噪) → SpEx+ (自适应分离) → 双微调融合声纹 (CAM++v7+ERes2NetV2v7+ResNetSE) → Fun-ASR-Nano-2512 (ASR)
```

**datasetA 全量成绩: CER 0.3705 | RR 0.9937 | 80分 64.93 | 推理 1566s (GPU)**

- 微调模型权重经 **HuggingFace / GitHub Releases** 托管，clone 后执行 `python tools/download_finetuned.py`（无需 git-lfs）
- 详细使用指南见 **`使用说明.md`**（环境搭建/推理命令/常见问题/训练脚本）
- datasetA 提交文件见 **`results/submission_datasetA_dual.json`**（官方 CER 口径）
- **数字归一化**: ASR 输出后处理（26→二十六、30%→百分之三十），已接入 `pipeline.py`（`utils/digit_normalize.py`）
- **热词注意**: FunASR-Nano 热词为强干预，列表不全时副作用>收益（实测 170 好 vs 310 差），当前关闭（`hotwords: null`），如需启用必须先小样本 A/B 验证

### 快速开始

```bash
git clone https://github.com/Wowater404/Anti-interference-voice-system.git
cd voice_pipeline && python tools/download_finetuned.py
# 单条推理
python run_inference.py --kws 唤醒.wav --cmd 识别.wav
# 批量 (比赛格式)
python run_inference.py --config configs/verify_dual_full.yaml --data_root <dataset目录> --split all --output results/out.json
```

> ⚠️ **环境注意 (Windows)**: 运行推理前需确保 PATH 含 conda 的 `Library/bin`（cuDNN DLL 位置），否则 cuDNN 无法加载导致推理慢 3-5 倍。若遇 torch 段错误（`torch._C` access violation），先重启系统（多为环境类故障），不要直接禁用 cuDNN。

---

## 历史流水线（早期版本，已弃用）

```
GTCRN (降噪) → SpEx+ (人声分离) → 3-model Z-score Ensemble (声纹鉴别) → Fun-ASR-Nano-2512 (语音识别)
```

> 以下为早期版本描述，当前 V8 方案降噪已改用 **Renoise**（见上文），声纹已升级为**双微调融合**（CAM++v7 + ERes2NetV2v7 + ResNetSE）。

| 阶段 | 模型 | 说明 |
|------|------|------|
| Stage 1 降噪 | ~~GTCRN~~ → Renoise | V8 已切换到 Renoise（noisereduce 库，stationary=False 自适应）|
| Stage 2 分离 | SpEx+ | 目标说话人提取，以唤醒音频为参考（自适应触发）|
| Stage 3 声纹 | CAM++ + ERes2NetV2 + ResNetSE | Z-score 归一化加权融合 (0.4/0.4/0.2)，前两者 V7 微调 |
| Stage 4 识别 | Fun-ASR-Nano-2512 | SenseVoice 编码器 + Qwen3-0.6B，800M 参数 |

声纹鉴别阶段采用三模型 Z-score 集成方案：对 CAM++、ERes2NetV2、ResNetSE 三个模型
分别计算 cosine 相似度，经 Z-score 归一化后按 0.4/0.4/0.2 权重加权融合，以 -0.17
为阈值判定接受/拒识。Z-score 归一化消除了不同模型分数分布差异，使 ResNetSE 虽然
单独表现较弱但能在融合中提供互补信息。

分离模块采用**自适应触发**：仅对低置信样本（绝对相似度 < 0.28）触发 SpEx+ 分离，
对齐训练数据分布；分离后重提声纹并采用双阈值鉴别（`vp_threshold_separated`）。
历史教训：分离触发阈值不可与 Z-score 阈值混用（曾误用 -0.17 导致分离永不触发）。

## 纯推理仓库说明

**当前仓库不含训练/微调脚本与产物**，定位为"纯推理流水线"：

- 所有模块均为**官方预训练模型或已微调权重**，运行时自动加载或下载。
- **微调模型**：CAM++v7 / ERes2NetV2v7 微调权重经 HuggingFace/GitHub Releases 托管（`finetuned_models/` 目录），clone 后执行 `python tools/download_finetuned.py`。
- 仓库内仅保留推理必需代码：`pipeline.py`、`run_inference.py`、`modules/`、`configs/`、`tools/download_*.py`（权重下载）。

## 性能对比 (datasetA, 80分制 = CER×40 + RR×40)

| 配置 | CER | RR | Score | 推理时间 |
|------|-----|-----|-------|---------|
| **V8 双微调 + 数字归一化 (当前)** | **0.3705** | **0.9937** | **64.93** | 1566s (GPU) |
| V8 双微调 (基线, 无归一化) | 0.3793 | 0.9958 | 64.66 | 1427s (GPU) |
| V5 声纹微调 | 0.4527 | 0.9368 | 59.36 | - |
| V2 基线 (noisereduce + Paraformer + 预训练声纹) | 0.5833 | 0.9367 | 54.14 | 423s(CPU) |
| main 全预训练 (GTCRN + SpEx+ + 预训练CAM++ + Paraformer) | 0.6533 | 0.9198 | 50.66 | 632s |

> 微调声纹较预训练提升约 10 分（V5 已实施并验证 59.36）。
> 数字归一化后处理进一步降低 CER（0.3793→0.3705）。

## 最终实验结果（当前分支）

测试集：datasetA，共 1838 条（pos 1364 / neg 474）。

| 指标 | 结果 |
|------|------:|
| CER | 0.3705 |
| 拒识率 RR | 0.9937 |
| 识别项得分（80 分制） | 64.93 |
| 估算总分（100 分制） | ~78.9 |

评分公式：CER 得分 = (1 - CER) × 40，RR 得分 = RR × 40，识别得分 = CER 得分 + RR 得分。
100 分制另含推理时间（10 分）和内存占用（10 分），此处为估算值（不含推理效率的 10 分）。
> ⚠️ 上述为分支实测，合入 main 前建议在统一环境复核。

## 安装

建议使用 Python 3.12、PyTorch 和 torchaudio 的匹配 CUDA 版本。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

下载模型权重：

```powershell
# 微调声纹权重 (CAM++v7 / ERes2NetV2v7)
.\.venv\Scripts\python.exe tools\download_finetuned.py

# SpEx+ 人声分离模型
.\.venv\Scripts\python.exe tools\download_spexplus.py

# WeSpeaker ResNet34 声纹模型 (ResNetSE)
.\.venv\Scripts\python.exe tools\download_wespeaker.py
```

权重保存位置：

| 模型 | 路径 |
|------|------|
| ~~GTCRN~~ → Renoise | noisereduce 库内置（V8 不再使用 GTCRN） |
| SpEx+ | `pretrained/spex_plus/checkpoint.pth` |
| WeSpeaker ResNet34 | `pretrained/wespeaker_resnet34/wespeaker_zh_cnceleb_resnet34.onnx` |
| CAM++ / ERes2NetV2 (微调) | `finetuned_models/`（`python tools/download_finetuned.py` 下载） |
| Fun-ASR-Nano-2512 | ModelScope 运行时自动下载 |

SpEx+ 检查点固定 SHA256：

`2d6a2f2b404fd18a809eb82052fd64ef0bd986f410b1043bc666b54121e44b5c`

## 运行

单样本：

```powershell
.\.venv\Scripts\python.exe run_inference.py `
  --kws "path\to\kws.wav" `
  --cmd "path\to\cmd.wav" `
  --label "目标文本" `
  --output "results\single.json"
```

完整 datasetA（三模型集成模式自动启用）：

```powershell
.\.venv\Scripts\python.exe run_inference.py `
  --config configs\default.yaml `
  --data_root "path\to\datasetA" `
  --split all `
  --output "results\submission.json" `
  --checkpoint "results\checkpoint.json"
```

当配置文件中 `voiceprint.model` 设为 `ensemble` 时，`run_inference.py` 自动调用
`process_dataset_ensemble` 批量处理方法（Z-score 需要全量样本统计 mean/std）。

输出 JSON 顶层严格为：

```json
{
  "result": {
    "results": [],
    "final_cer": "0.0000",
    "duration": "0.00"
  }
}
```

## 关键文件

```text
configs/default.yaml              流水线配置（降噪/分离/声纹/ASR 四阶段参数）
config.py                         配置加载模块
pipeline.py                       完整推理流水线（含 ensemble 批量处理）
run_inference.py                  推理入口脚本
modules/denoiser.py               降噪模型工厂
modules/renoise.py                Renoise 降噪实现（当前 V8 使用）
modules/gtcrn.py                  GTCRN 降噪网络（历史遗留，已弃用）
modules/separator.py              分离/提取模型工厂
modules/spexplus_separator.py     SpEx+ 网络与流水线适配器
modules/voiceprint.py             声纹鉴别模块（CAM++/ERes2NetV2/ResNetSE/Ensemble）
modules/asr.py                    语音识别模块（Fun-ASR-Nano/SenseVoice/Paraformer/Whisper）
utils/audio.py                    音频 I/O 工具
utils/metrics.py                  CER / RR 评估指标
tools/download_finetuned.py       微调声纹权重下载 (HF/GitHub Releases, SHA256 校验)
tools/download_spexplus.py        SpEx+ 权重下载与 SHA256 校验
tools/download_wespeaker.py       WeSpeaker ResNet34 ONNX 权重下载
TEAM_INTEGRATION_GUIDE.md         团队合并规范 V2
```

`pretrained/`（GTCRN 除外）、`results/`、虚拟环境和缓存均不进入 Git 提交。
