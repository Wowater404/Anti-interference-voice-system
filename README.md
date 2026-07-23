# 抗干扰语音指令识别流水线

> **项目**: XH-202615 复杂交互场景的抗干扰语音指令识别技术
> **发榜单位**: 美的集团
> **当前版本**: V5（微调声纹 + 禁分离）

## 架构概览

```
唤醒音频 kws (1.5s)              识别音频 cmd (1.28~6.2s)
       │                                │
       ▼                                ▼
  [CAM++ ①]                     [Stage 1: 降噪]
  提取参考声纹                   noisereduce (谱减法)
       │                                │
       │ kws_embedding                  ▼ denoised_audio
       │                       [CAM++ ②] 提声纹
       │                                │
       │                     sim = cos(kws_emb, cmd_emb)
       │                                │
       │                     <sim >= 0.67 (微调阈值) ?>
       │                     ╱              ╲
       │              是(目标说话人)      否(拒识)
       │                     │              │
       │                     ▼              ▼
       │              [Stage 2: ASR]    content = ""
       │              Paraformer
       │                     │
       └──────────────►  content (识别文本)
```

**V5 与 V4.1 的核心差异**：CAM++ 声纹模型用 datasetA 增强数据微调，且**禁用分离**。
微调模型本身够强（pos sim 0.8+），不需要分离救回；分离选轨反而放大 neg 假接受
（fold_0 验证折实测：降噪音频假接受 6/95，加分离后假接受 30/95）。

## 性能对比（fold_0 验证折，无偏）

| 配置 | CER | RR | pos接受 | neg假接受 | Score |
|------|-----|-----|---------|----------|-------|
| V4.1 基线（预训练+自适应分离） | 0.6071 | 0.9579 | 175/273 | 4/95 | 0.5403 |
| **V5 微调 + 禁分离** | **0.4527** | 0.9368 | **242/273** | 6/95 | **0.5936** |

Score 提升 **+0.053**（相对 +9.8%）：CER 大降 0.154（多救回 67 条 pos），RR 仅微降 0.021。

## 微调声纹模型（V5 核心）

CAM++ 用 datasetA 增强数据对比学习微调，训练方法（V3，防塌缩）：

- **数据增强**（`tools/augment_dataset.py`）：每条样本 8 倍增强
  （原始 / 音量±4dB / 白噪声SNR15 / 粉噪声SNR20 / 片段截取 / 变声±半音），
  pos 1364→10912 条，neg 474→3792 条
- **五折交叉验证**（`tools/make_folds.py`）：按 orig_id 划分防泄漏，val 仅用原始音频
- **双向 margin 损失**：pos>0.7, neg<0.3，留 0.4 间隔——不把 pos 推向 cos=1，避免 embedding 塌缩
- **1:1 平衡采样**：消除 pos 拉力优势
- **预处理一致**：kws 原始 + cmd 降噪，与 pipeline 推理完全一致（解决分布不匹配）
- **防塌缩**：冻结主干前半 + 全部 BN 设 eval + lr=1e-4 + EER 监控

模型：`finetuned_models/camplus_v3_fold0.pt`（见该目录 README）

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 单样本推理
python run_inference.py --kws "path/to/kws.wav" --cmd "path/to/cmd.wav" --label "指令文本"

# 3. 批量推理 (datasetA)
python run_inference.py --data_root "path/to/datasetA" --split all \
    --output results/output.json --checkpoint results/ckpt.json
```

首次运行自动从 ModelScope/HuggingFace 下载预训练模型到 `pretrained/`（约 2GB），
微调权重从 `finetuned_models/` 加载（已随仓库提供）。

## 重新训练微调模型（可选）

```bash
# 1. 数据增强
python tools/augment_dataset.py --src "path/to/datasetA" --dst "path/to/datasetA_aug" --workers 8

# 2. 五折划分
python tools/make_folds.py --aug_root "path/to/datasetA_aug" --n_folds 5

# 3. 训练 (fold_0)
python tools/train_camplus_finetune.py --aug_root "path/to/datasetA_aug" \
    --fold 0 --epochs 10 --batch 64 --workers 8

# 4. 验证折无偏评估
python tools/eval_fold0_nosep.py
```

## 目录结构

```
voice_pipeline/
├── run_inference.py      # 推理入口
├── pipeline.py           # 流水线核心 (降噪→声纹→ASR)
├── config.py             # 配置加载
├── configs/default.yaml  # 配置 (微调权重+阈值0.67+禁分离)
├── modules/              # 模型模块
│   ├── denoiser.py       #   降噪 (noisereduce)
│   ├── separator.py      #   分离 (SepFormer, V5已禁用)
│   ├── voiceprint.py     #   声纹 (CAM++, 支持微调权重)
│   └── asr.py            #   ASR (Paraformer)
├── utils/                # 音频/指标工具
├── tools/                # 数据增强/训练/评估脚本
├── finetuned_models/     # 微调声纹模型 (随仓库提供)
├── pretrained/           # 预训练模型 (运行时下载, 不上传)
└── results/              # 推理结果 (不上传)
```

## 比赛评分

- **CER 40%**: pos 样本字错误率（micro-average，拒识按删除错误）
- **RR 40%**: neg 样本拒识率
- **效率 20%**: 推理时间 10% + 内存 10%（禁分离比 V4.1 省时约 27%）

## 历史版本

- **V5**（当前）: 微调声纹 + 禁分离，Score 0.5936（fold_0 无偏）
- **V4.1**: 预训练声纹 + 自适应分离 + 双阈值，Score 0.5452
- **V3**: 全量分离（能量法选轨缺陷），Score 0.5285
- **V2**: 无分离基线，Score 0.5404
