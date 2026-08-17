# 抗干扰语音指令识别系统

复杂交互场景下的抗干扰语音指令识别流水线，串联降噪、人声分离、声纹鉴别与语音识别四个阶段。

## ⚡ V16 当前方案 (2026-08-16, release/v16 分支)

```
DeepFilterNet3微调 (降噪) → SpEx+ 自适应分离 (cmd) / SepFormer16k 盲分离 (kws) → 三模型声纹 (CAM++v8+ERes2NetV2v7+ResNetSE) → Nano + Paraformer 双ASR
```

**V8→V16 演进**: 降噪 Renoise→DeepFilterNet3微调 (SI-SDR +8.0→+14.9dB)；声纹 CAM++v7→v8 (EER 0.027→0.0003)；新增 kws 自适应盲分离 (V9)、唤醒词拼音匹配 91.35% (V12)、双ASR 快慢协同 (V13)、训练质量门控 + kws 预处理对齐 (V14)、DF3 微调 (V16)。

- 微调模型权重经 **HuggingFace / GitHub Releases** 托管，clone 后执行 `python tools/download_finetuned.py`（无需 git-lfs）
- **DF3 微调权重**: 由 `tools/train_df3_finetune.py` 训练产生（LibriSpeech clean + DEMAND 噪声），产出至 `pretrained/deepfilternet3_finetuned/`；无此权重时自动回退官方预训练
- 详细使用指南见 **`使用说明.md`**（环境搭建/推理命令/常见问题/训练脚本）
- datasetA 提交文件见 **`results/submission_datasetA_dual.json`**（官方 CER 口径）
- **数字归一化**: ASR 输出后处理（26→二十六、30%→百分之三十），已接入 `pipeline.py`（`utils/digit_normalize.py`）
- **热词注意**: FunASR-Nano 热词为强干预，列表不全时副作用>收益（实测 170 好 vs 310 差），当前关闭（`hotwords: null`），如需启用必须先小样本 A/B 验证

### 快速开始

```bash
git clone -b release/v16 https://github.com/Wowater404/Anti-interference-voice-system.git
cd voice_pipeline && python tools/download_finetuned.py
# 单条推理 (--kws-text 唤醒词文本, 用于唤醒词定位, 可选)
python run_inference.py --kws 唤醒.wav --cmd 识别.wav [--kws-text 你好科慕]
# 批量 (比赛格式)
python run_inference.py --config configs/verify_dual_full.yaml --data_root <dataset目录> --split all --output results/out.json
```

> ⚠️ **环境注意 (Windows)**: 运行推理前需确保 PATH 含 conda 的 `Library/bin`（cuDNN DLL 位置），否则 cuDNN 无法加载导致推理慢 3-5 倍。若遇 torch 段错误（`torch._C` access violation），先重启系统（多为环境类故障），不要直接禁用 cuDNN。

---

## 历史流水线（早期版本，已弃用）

```
V8:  Renoise (降噪) → SpEx+ (人声分离) → 3-model Z-score Ensemble (声纹鉴别) → Fun-ASR-Nano-2512 (语音识别)
V5:  noisereduce (降噪) → 声纹鉴别 (CAM++微调) → Paraformer (ASR)
```

> 以下为 V8 及更早版本描述，V16 方案降噪已改用 **DeepFilterNet3 微调**（见上文），声纹已升级为**三模型融合**（CAM++v8 + ERes2NetV2v7 + ResNetSE），识别升级为**双ASR 快慢协同**。

| 阶段 | 模型 | 说明 |
|------|------|------|
| Stage 1 降噪 | ~~GTCRN~~ → ~~Renoise~~ → **DeepFilterNet3微调** | V16 主降噪 DF3 微调（atten_lim_db=4, post_filter=False），Renoise 保留备用 |
| Stage 2 分离 | SpEx+ (cmd) / SepFormer16k (kws) | cmd 目标说话人提取（以唤醒音频为参考，自适应触发）；kws 无参考盲分离（能量法预检单/多人）|
| Stage 3 声纹 | CAM++ + ERes2NetV2 + ResNetSE | Z-score 归一化加权融合 (0.4/0.4/0.2)，CAM++v8/ERes2NetV2v7 微调 |
| Stage 4 识别 | Fun-ASR-Nano-2512 + Paraformer | Nano 指令识别兜底 + Paraformer 中文快匹配（kws 唤醒词定位）|

声纹鉴别阶段采用三模型 Z-score 集成方案：对 CAM++、ERes2NetV2、ResNetSE 三个模型
分别计算 cosine 相似度，经 Z-score 归一化后按 0.4/0.4/0.2 权重加权融合，以 -0.17
为阈值判定接受/拒识。Z-score 归一化消除了不同模型分数分布差异，使 ResNetSE 虽然
单独表现较弱但能在融合中提供互补信息。

分离模块采用**自适应触发**：仅对低置信样本（绝对相似度 < 0.28）触发 SpEx+ 分离，
对齐训练数据分布；分离后重提声纹并采用双阈值鉴别（`vp_threshold_separated`）。
历史教训：分离触发阈值不可与 Z-score 阈值混用（曾误用 -0.17 导致分离永不触发）。

## 纯推理仓库说明

**当前仓库定位为推理流水线**，V16 起附带训练脚本（`tools/train_*.py`）：

- 所有模块均为**官方预训练模型或已微调权重**，运行时自动加载或下载。
- **微调模型**：CAM++v8 / ERes2NetV2v7 微调权重经 HuggingFace/GitHub Releases 托管（`finetuned_models/` 目录），clone 后执行 `python tools/download_finetuned.py`。
- **DF3 降噪微调**：`tools/train_df3_finetune.py`（LibriSpeech + DEMAND），产物不入库，无权重时回退官方预训练。
- 仓库保留推理必需代码 + 训练流水线：`pipeline.py`、`run_inference.py`、`modules/`、`configs/`、`tools/download_*.py`、`tools/train_*.py`。

## 性能对比 (datasetA, 80分制 = CER×40 + RR×40)

| 配置 | CER | RR | Score | 推理时间 |
|------|-----|-----|-------|---------|
| **V16 DF3微调 + 三模型v8 + 双ASR (当前)** | **~0.3489** | **0.9951** | **~65.85** | 估算较 V8 快 30%+ |
| V8 双微调 + 数字归一化 | 0.3705 | 0.9937 | 64.93 | 1566s (GPU) |
| V8 双微调 (基线, 无归一化) | 0.3793 | 0.9958 | 64.66 | 1427s (GPU) |
| V5 声纹微调 | 0.4527 | 0.9368 | 59.36 | - |
| V2 基线 (noisereduce + Paraformer + 预训练声纹) | 0.5833 | 0.9367 | 54.14 | 423s(CPU) |
| main 全预训练 (GTCRN + SpEx+ + 预训练CAM++ + Paraformer) | 0.6533 | 0.9198 | 50.66 | 632s |

> V16 数据为 DF3 调参后的本地实测；正式定版前需跑 V15 vs V16 对照实验量化降噪微调收益（详见 `流水线架构比对_V8_vs_V16.md`）。

## 最终实验结果（当前分支）

测试集：datasetA，共 1838 条（pos 1364 / neg 474）。

| 指标 | 结果 |
|------|------:|
| CER | ~0.3489 (DF3 调参后) |
| 拒识率 RR | ~0.9951 |
| 识别项得分（80 分制） | ~65.85 |
| 估算总分（100 分制） | ~79+ |

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
| DeepFilterNet3 (微调) | `pretrained/deepfilternet3_finetuned/`（`tools/train_df3_finetune.py` 训练；缺失回退官方预训练） |
| SpEx+ | `pretrained/spex_plus/checkpoint.pth` |
| SepFormer16k | `pretrained/sepformer16k/`（SpeechBrain 自动下载） |
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
configs/default.yaml              流水线配置（降噪/分离/声纹/ASR 四阶段参数, V16 默认 DF3 微调）
configs/verify_dual_full.yaml     比赛验证配置（双ASR + 三模型集成, kws 自适应盲分离）
config.py                         配置加载模块
pipeline.py                       完整推理流水线（V9-V16: kws 盲分离/唤醒词定位/双ASR 快慢协同）
run_inference.py                  推理入口脚本（支持 --kws-text 唤醒词定位）
modules/denoiser.py               降噪模型工厂（DF3 微调权重加载）
modules/renoise.py                Renoise 降噪实现（备用）
modules/gtcrn.py                  GTCRN 降噪网络（历史遗留，已弃用）
modules/separator.py              分离/提取模型工厂（含 SepFormer16k 盲分离）
modules/spexplus_separator.py     SpEx+ 网络与流水线适配器
modules/voiceprint.py             声纹鉴别模块（CAM++/ERes2NetV2/ResNetSE/Ensemble）
modules/asr.py                    语音识别模块（Fun-ASR-Nano/SenseVoice/Paraformer/Whisper）
utils/audio.py                    音频 I/O 工具
utils/metrics.py                  CER / RR 评估指标
utils/digit_normalize.py          数字归一化后处理
tools/download_finetuned.py       微调声纹权重下载 (HF/GitHub Releases, SHA256 校验)
tools/download_spexplus.py        SpEx+ 权重下载与 SHA256 校验
tools/download_wespeaker.py       WeSpeaker ResNet34 ONNX 权重下载
tools/train_df3_finetune.py       [V16] DF3 降噪微调 (LibriSpeech + DEMAND)
tools/train_eres2netv2_finetune.py [V14] ERes2NetV2 说话人微调
tools/run_train_pipeline.py       [V14] 训练流水线入口（数据准备→质量门控→声纹微调）
tools/prepare_processed_train_data.py [V14] 训练数据预处理（kws+cmd 全链路降噪/分离）
tools/quality_gate.py             [V14] 训练数据质量门控过滤
tools/augment_dataset_incremental.py [V14] 增量数据增强
tools/measure_v16_perf.py         [V16] 推理性能剖析工具
唤醒词定位KWS方案.md              [V12] 唤醒词拼音匹配方案文档
训练流水线使用说明（质量门控版）.md [V14] 训练流水线使用文档
训练说明.md                       训练脚本说明
TEAM_INTEGRATION_GUIDE.md         团队合并规范 V2
```

`pretrained/`（GTCRN 除外）、`results/`、`runs/`、虚拟环境和缓存均不进入 Git 提交。
