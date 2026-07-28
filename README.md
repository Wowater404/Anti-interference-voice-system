# 抗干扰语音指令识别系统 - V6 (GTCRN + SpEx+ + CAM++微调 + Paraformer)

三组模型更换后的集成版本。流水线：

`GTCRN → (SpEx+ 可选) → CAM++(微调) → Paraformer`

> **当前版本**: V6（GTCRN降噪 + SpEx+分离 + GTCRN微调声纹 + Paraformer ASR）
> **最优配置**: 禁分离, thr=0.72, Score=0.6076
> **含分离**: thr=0.72, Score=0.5984

## 性能对比 (datasetA)

### 全预训练 (无微调)

| 配置 | CER | RR | Score |
|------|-----|-----|-------|
| V2 基线 (noisereduce + Paraformer + 预训练声纹) | 0.5857 | 0.9367 | 0.5404 |
| V6 预训练 (GTCRN + SpEx+ + Paraformer + 预训练声纹) | 0.6990 | 0.9409 | 0.4968 |

### 声纹微调后

| 配置 | CER | RR | Score | 推理时间 |
|------|-----|-----|-------|---------|
| V5 (noisereduce + Paraformer + noisereduce微调声纹) | 0.4527 | 0.9368 | 0.5936 | 423s |
| V6 含SpEx+ (GTCRN + SpEx+ + Paraformer + GTCRN微调声纹, thr=0.72) | 0.4661 | 0.9620 | 0.5984 | 811s |
| **V6 禁SpEx+ (GTCRN + Paraformer + GTCRN微调声纹, thr=0.72)** | **0.4684** | **0.9873** | **0.6076** | 718s |

### 消融对比

| 消融维度 | 含分离 | 禁分离 | 差异 |
|---------|--------|--------|------|
| Score | 0.5984 | 0.6076 | 禁分离 +0.009 |
| RR | 0.9620 | 0.9873 | 禁分离后 neg 假接受减半 |
| 推理时间 | 811s | 718s | 禁分离快 11% |

## 模型选型

| 模块 | 模型 | 版本说明 |
|------|------|---------|
| 降噪 | **GTCRN** (48K参数, 复数域mask, dns3 checkpoint) | 替代 noisereduce |
| 分离 | **SpEx+** 16kHz 目标说话人提取 | 替代 SepFormer, 可选启用 |
| 声纹 | **CAM++** 微调 (fold_0, val_EER=0.0643) | datasetA 8倍增强 + GTCRN降噪下微调 |
| ASR | **Paraformer** (FunASR, 热词增强) | 保留, SenseVoice 备选 |

## 声纹微调

- 训练脚本: `tools/train_camplus_finetune.py`
- 增强数据: `datasetA_aug` (14704条 cmd 已 GTCRN 降噪预处理)
- 训练配置: fold_0, 10 epochs, 双向margin损失, 1:1平衡采样
- 最佳模型: `runs/fold_0/camplus_finetuned_best.pt` (Epoch 7, val_EER=0.0643)
- 训练日志: `runs/fold_0/train_log.json`

### 训练过程

| Epoch | val_EER | pos_sim | 最佳 |
|-------|---------|---------|:--:|
| 基线 | 0.2138 | 0.263 | - |
| 1 | 0.0748 | 0.755 | ★ |
| 4 | 0.0661 | 0.793 | ★ |
| 7 | 0.0643 | 0.832 | ★ |

## 阈值扫描

| thr | CER | RR | Score |
|-----|-----|-----|-------|
| 0.63 | 0.4628 | 0.9135 | 0.5803 |
| 0.67 | 0.4631 | 0.9283 | 0.5861 |
| 0.70 | 0.4652 | 0.9430 | 0.5912 |
| **0.72** | **0.4661** | **0.9620** | **0.5984** |
| 0.75 | 0.4811 | 0.9768 | 0.5983 |

## 关键结论

1. **声纹微调是最大杠杆**: 全预训练 0.4968 → 微调后 0.5984 (+0.10)
2. **SpEx+分离在datasetA上是负资产**: 禁分离后 Score +0.009, 且推理快 11%
3. **GTCRN+微调声纹组合优于旧版**: V6禁分离 0.6076 > V5 0.5936 (+0.014)
4. **Paraformer与SenseVoice几乎打平**: 分别为 0.5984 vs 0.5988

## 文件结构

```
voice_pipeline/
├── configs/default.yaml       # 主配置 (GTCRN + SpEx+ + CAM++微调 + Paraformer)
├── pipeline.py                # 4阶段流水线
├── modules/
│   ├── denoiser.py            # 降噪 (GTCRN/DeepFilterNet/noisereduce)
│   ├── gtcrn.py               # GTCRN 降噪模型实现
│   ├── separator.py           # 分离 (SpEx+/SepFormer/PassThrough)
│   ├── spexplus_separator.py  # SpEx+ 目标说话人提取
│   ├── voiceprint.py          # 声纹 (CAM++ 微调/ECAPA/WeSpeaker)
│   └── asr.py                 # ASR (Paraformer/SenseVoice/Whisper)
├── finetuned_models/          # 微调模型权重
├── runs/fold_0/               # 训练输出 (best.pt + log)
├── pretrained/gtcrn/          # GTCRN 预训练权重
├── pretrained/spex_plus/      # SpEx+ 预训练权重
└── tools/                     # 训练/评估/增强脚本
```
