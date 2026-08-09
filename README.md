# 抗干扰语音指令识别系统

复杂交互场景下的抗干扰语音指令识别流水线，串联降噪、人声分离、声纹鉴别与语音识别四个阶段。

## ⚡ V8 最终方案 (2026-08-09, release/v8-dual-full 分支)

```
Renoise (降噪) → SpEx+ (自适应分离) → 双微调融合声纹 (CAM++v7+ERes2NetV2v7+ResNetSE) → Paraformer (ASR)
```

**datasetA 全量成绩: CER 0.3793 | RR 0.9958 | 80分 64.66 | 推理 1427s | 峰值内存 7.14GB**

- 微调模型权重经 **Git LFS** 管理 (`finetuned_models/`)，clone 后执行 `git lfs pull`
- 详细使用指南见 **`使用说明.md`**（环境搭建/推理命令/常见问题/训练脚本）
- datasetA 提交文件见 **`results/submission_datasetA_dual.json`**（官方 CER 口径）

### 快速开始

```bash
git clone https://github.com/Wowater404/Anti-interference-voice-system.git
cd voice_pipeline && git lfs pull
# 单条推理
python run_inference.py --kws 唤醒.wav --cmd 识别.wav
# 批量 (比赛格式)
python run_inference.py --config configs/verify_dual_full.yaml --data_root <dataset目录> --split all --output results/out.json
```

---

## 历史流水线（早期版本）

```
GTCRN (降噪) → SpEx+ (人声分离) → 3-model Z-score Ensemble (声纹鉴别) → Fun-ASR-Nano-2512 (语音识别)
```

| 阶段 | 模型 | 说明 |
|------|------|------|
| Stage 1 降噪 | GTCRN | 48K 参数复数域 mask，实时推理 |
| Stage 2 分离 | SpEx+ | 目标说话人提取，以唤醒音频为参考 |
| Stage 3 声纹 | CAM++ + ERes2NetV2 + ResNetSE | Z-score 归一化加权融合 (0.4/0.4/0.2) |
| Stage 4 识别 | Fun-ASR-Nano-2512 | SenseVoice 编码器 + Qwen3-0.6B，800M 参数 |

声纹鉴别阶段采用三模型 Z-score 集成方案：对 CAM++、ERes2NetV2、ResNetSE 三个模型
分别计算 cosine 相似度，经 Z-score 归一化后按 0.4/0.4/0.2 权重加权融合，以 -0.17
为阈值判定接受/拒识。Z-score 归一化消除了不同模型分数分布差异，使 ResNetSE 虽然
单独表现较弱但能在融合中提供互补信息。

## 纯推理仓库说明

**当前仓库不含任何训练/微调脚本与产物**，定位为"纯推理流水线"：

- 所有模块均为**官方预训练模型**，运行时自动加载或下载。
- **不需要训练**：训练（微调）推迟到最终阶段，按团队约定（见 TEAM_INTEGRATION_GUIDE.md V2）在模型全部选定、架构锁定后由整合者统一进行；训练脚本与增强数据仅在本地保留，不入库。
- 仓库内仅保留推理必需代码：`pipeline.py`、`run_inference.py`、`modules/`、`configs/`、`tools/download_*.py`（权重下载）。

## 性能对比 (datasetA, 80分制 = CER×40 + RR×40)

| 配置 | 阈值 | CER | RR | Score | 推理时间 |
|------|------|-----|-----|-------|---------|
| main 全预训练 (GTCRN + SpEx+ + 预训练CAM++ + Paraformer) | 0.25 | 0.6533 | 0.9198 | **50.66** | 632s |
| V2 基线 (noisereduce + Paraformer + 预训练声纹) | 0.28 | 0.5833 | 0.9367 | 54.14 | 423s(CPU) |

> 预训练声纹在 GTCRN 分布上区分力较弱（最优阈值 0.25，pos 接受率 55.5%）。
> 微调声纹可提升约 10 分（实测 59.84~60.76），将在最后阶段实施。

## 最终实验结果（当前分支，待复核）

测试集：datasetA，共 1838 条（pos 1364 / neg 474）。

| 指标 | 结果 |
|------|------:|
| CER | 0.4365 |
| 拒识率 RR | 0.9198 |
| 识别项得分（80 分制） | 59.33 |
| 估算总分（100 分制） | 74.58 |

评分公式：CER 得分 = (1 - CER) × 40，RR 得分 = RR × 40，识别得分 = CER 得分 + RR 得分。
100 分制另含推理时间（10 分）和内存占用（10 分），此处为估算值。
> ⚠️ 上述为分支自称实测，合入 main 前需在统一环境复核。

## 安装

建议使用 Python 3.12、PyTorch 和 torchaudio 的匹配 CUDA 版本。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

下载模型权重：

```powershell
# SpEx+ 人声分离模型
.\.venv\Scripts\python.exe tools\download_spexplus.py

# WeSpeaker ResNet34 声纹模型 (ResNetSE)
.\.venv\Scripts\python.exe tools\download_wespeaker.py
```

权重保存位置：

| 模型 | 路径 |
|------|------|
| GTCRN | `pretrained/gtcrn/`（随仓库提交） |
| SpEx+ | `pretrained/spex_plus/checkpoint.pth` |
| WeSpeaker ResNet34 | `pretrained/wespeaker_resnet34/wespeaker_zh_cnceleb_resnet34.onnx` |
| CAM++ / ERes2NetV2 | ModelScope 运行时自动下载 |
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
modules/gtcrn.py                  GTCRN 降噪网络
modules/separator.py              分离/提取模型工厂
modules/spexplus_separator.py     SpEx+ 网络与流水线适配器
modules/voiceprint.py             声纹鉴别模块（CAM++/ERes2NetV2/ResNetSE/Ensemble）
modules/asr.py                    语音识别模块（Fun-ASR-Nano/SenseVoice/Paraformer/Whisper）
utils/audio.py                    音频 I/O 工具
utils/metrics.py                  CER / RR 评估指标
tools/download_spexplus.py        SpEx+ 权重下载与 SHA256 校验
tools/download_wespeaker.py       WeSpeaker ResNet34 ONNX 权重下载
TEAM_INTEGRATION_GUIDE.md         团队合并规范 V2
```

`pretrained/`（GTCRN 除外）、`results/`、虚拟环境和缓存均不进入 Git 提交。
