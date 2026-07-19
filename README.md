# 抗干扰语音指令识别流水线

> **项目**: XH-202615 复杂交互场景的抗干扰语音指令识别技术
> **发榜单位**: 美的集团
> **当前版本**: V4.1（自适应分离 + 声纹选轨 + 双阈值鉴别）

## 架构概览

```
唤醒音频 kws (1.5s)              识别音频 cmd (1.28~6.2s)
       │                                │
       ▼                                ▼
  [CAM++ ①]                     [Stage 1: 降噪]
  提取参考声纹                   noisereduce (谱减法)
       │                                │
       │ kws_embedding                  ▼ denoised_audio
       │                       [CAM++ ②] 算 sim_denoised
       │                                │
       │                     <sim_denoised < 0.28 ?>
       │                     ╱              ╲
       │              否(直接用)           是(自适应分离)
       │                     │              │
       │                     │     [Stage 2: SepFormer-16k]
       │                     │     盲分离 → 多条音轨
       │                     │              │
       │                     │     [CAM++ ③] 声纹选轨
       │                     │     选与 kws 最相似的音轨
       │                     ╲              ╱
       │                      ▼ best_audio
       │                [Stage 3: 声纹鉴别]
       │                双阈值: 未分离≥0.28 / 分离过≥0.35
       │                    ╱              ╲
       │              reject              accept
       │                 ╱                    ╲
       │         [输出: ""]          [Stage 4: ASR]
       │                          Paraformer (FunASR)
       │                                  │
       │                                  ▼
       └───────────────────────  [输出: 识别文本(去标点)]
```

**核心设计**: 声纹模型 (CAM++) 不是末端单一关卡, 而是贯穿流程的决策中枢 —— ①提取参考声纹 ②决策是否分离 ③选轨+最终鉴别。

## 比赛评分标准

| 指标 | 权重 | 说明 |
|------|------|------|
| CER (字错率) | 40% | 在 pos 正样本上评估, 越低越好 |
| RR (拒识率) | 40% | 在 neg 负样本上评估, 越高越好 |
| 推理效率 | 20% | 推理时间 10% + 内存占用 10% |

**官方 CER 计算**: NFKC 归一化 + lowercase + 全 Unicode P* 标点过滤 + editdistance + micro-average; 拒识样本输出空字符串 `""` (按删除错误计)。

## 当前性能 (V4.1, datasetA)

| 指标 | V2 (无分离) | V4 (分离 t=0.28) | **V4.1 (双阈值)** |
|------|------------|-----------------|-------------------|
| CER (micro) | 0.5857 | 0.5615 | **0.5738** |
| RR | 0.9367 | 0.9051 | **0.9367** |
| Score (CER40+RR40) | 0.5404 | 0.5374 | **0.5452** |
| Pos 接受数 | 900 | 948 | 913 |

V4.1 相比 V2: CER 降 0.012, RR 不变, Score 提升 +0.0048。

## 模型清单与下载

模型权重不随仓库上传 (2.4GB), 首次运行时自动下载到 `pretrained/` 目录:

| 阶段 | 模型 | 来源 | 模型 ID |
|------|------|------|---------|
| 降噪 | noisereduce | pip 包 | `pip install noisereduce` |
| 分离 | SepFormer-16k | HuggingFace | `speechbrain/sepformer-whamr16k` |
| 声纹 | CAM++ | ModelScope | `iic/speech_campplus_sv_zh-cn_16k-common` |
| ASR | Paraformer | ModelScope | `iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch` |
| VAD | FSMN-VAD | ModelScope | `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch` |
| 标点 | ct-punc | ModelScope | `iic/punc_ct-transformer_cn-en-common-vocab471067-large` |

> 比赛环境网络受限时, 提前在本地运行一次推理完成下载, 再将 `pretrained/` 目录拷贝到比赛服务器。

## 快速开始

### 1. 环境安装

```bash
# Python 3.10+ 推荐
conda create -n voice_pipeline python=3.10
conda activate voice_pipeline

# 安装 PyTorch (根据 CUDA 版本, CPU 版如下)
pip install torch torchaudio

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置

编辑 `configs/default.yaml`:

```yaml
# 关键配置项
voiceprint:
  threshold: 0.28              # 声纹鉴别阈值 (未分离样本)
  threshold_separated: 0.35    # 分离后样本阈值 (V4.1 双阈值)

separation:
  enable: true                 # 启用自适应分离
  max_speakers: 2
  sep_trigger_min: 0.0         # 分离触发下限 (可调高以跳过低相似度样本)

device: auto                   # auto/cpu/cuda
```

### 3. 批量推理

```bash
# 处理所有样本 (pos + neg)
python run_inference.py \
    --data_root "/path/to/datasetA" \
    --split all \
    --output results/full_inference.json \
    --checkpoint results/checkpoint.json

# 仅正样本 (测 CER)
python run_inference.py --data_root "/path/to/datasetA" --split pos

# 仅负样本 (测 RR)
python run_inference.py --data_root "/path/to/datasetA" --split neg
```

支持断点续传: 中断后重新运行相同命令, 自动从 checkpoint 恢复。

### 4. 输出格式

```json
{
    "result": {
        "results": [
            {"id": "0", "content": "空调开到制热调到二十五度", "label": "空调开到制热调到二十五度风量调到百分之三十", "cer": "0.125"}
        ],
        "final_cer": "0.5738",
        "duration": "1064.29"
    },
    "metrics": {
        "rejection_rate": "0.9367",
        "final_score": "0.5452",
        "pos_count": 1364,
        "neg_count": 474
    }
}
```

## 项目结构

```
voice_pipeline/
├── configs/
│   └── default.yaml           # 流水线配置 (模型/阈值/分离策略)
├── modules/
│   ├── denoiser.py            # Stage 1: 降噪 (noisereduce)
│   ├── separator.py           # Stage 2: 人声分离 (SepFormer-16k)
│   ├── voiceprint.py          # Stage 3: 声纹提取 (CAM++)
│   └── asr.py                 # Stage 4: 语音识别 (Paraformer)
├── utils/
│   ├── audio.py               # 音频I/O + 预处理 (归一化/去静音/去标点)
│   └── metrics.py             # 评估指标 (CER, RR)
├── config.py                  # 配置加载器
├── pipeline.py                # 流水线编排器 (V4.1 核心逻辑)
├── run_inference.py           # 推理入口 (支持断点续传)
├── analyze_dataset.py         # 数据集分析
├── analyze_v2.py              # V2 结果分析
├── analyze_v4.py              # V4 vs V2 对比分析
├── requirements.txt           # 依赖列表
└── README.md                  # 本文件
```

## V4.1 核心机制

### 自适应分离
仅当降噪音频声纹相似度 `sim_denoised < 0.28` (即 V2 会拒识的样本) 才触发分离, 避免分离 artifact 污染已能正确识别的样本。

### 声纹辅助选轨
SepFormer 是盲分离, 不知道哪条音轨是目标说话人。对分离出的每条音轨提声纹, 选与 kws 参考声纹 cosine 相似度最高的音轨。修复了早期版本用"能量法"选轨 (选声音最大的音轨, 往往是噪声) 的缺陷。

### 双阈值鉴别
- 未分离样本: `sim ≥ 0.28` 接受
- 分离过的样本: `sim ≥ 0.35` 接受 (分离可能提升干扰人的相似度, 需更严格阈值)

## 模型替换指南

各模块通过工厂模式创建, 替换模型只需:
1. 在 `modules/` 下新增子类, 实现 `load()` + 核心方法
2. 在 `configs/default.yaml` 修改对应 `model` 字段

**接口契约** (所有模块共享的音频格式: `np.ndarray, float32, [-1,1], 16kHz, 单声道`):
- 降噪: `denoise(audio, sr) → audio`
- 分离: `separate(audio, sr) → (best_audio, sources[])`
- 声纹: `extract(audio, sr) → embedding (L2归一化)`
- ASR: `transcribe(audio, sr) → str`

**注意事项**:
- 换声纹模型后, 阈值必须重新校准 (不同模型相似度分布不同)
- 换分离模型时, 若是目标说话人提取 (如 SpEx+), 不需要选轨, pipeline 逻辑需调整
- 换 ASR 模型时注意输入形式 (Paraformer 需临时 WAV 文件, Whisper 可直接接受 numpy)
- 比赛环境为 L20-46G GPU, 确保模型支持 CUDA 推理

## 已知问题与优化方向

1. **推理效率**: V4.1 耗时 1064s (V2 为 423s), 因 907/1838 样本触发分离但仅 13 个受益。可调高 `sep_trigger_min` 跳过低相似度 neg 样本的无效分离
2. **SepFormer 内部选轨缺陷**: `separator.py` 的 `_select_best_match` 用能量法, 已在 pipeline 层用声纹选轨绕过
3. **目标说话人提取**: 未来可用 SpEx+ 替代"盲分离+声纹选轨", 让声纹引导分离本身
