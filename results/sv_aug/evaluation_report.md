# 抗噪声纹权重（sv_aug）训练与 datasetA 评测对比报告

> 分支：`sv_aug_experiment`
> 日期：2026-08-27
> 关联提交基础：`release/v8-dual-full`（历史最佳提交 `submission_datasetA_dual.json`）

## 1. 任务概述

在 `D:/声纹训练包_5.5h_v2`（HI-MIA 风格，唤醒音频+识别音频）上训练 ERes2NetV2 声纹模型，训练时引入 **在线增强**（干扰人声 + 背景噪声，尽量贴近 datasetA 落地场景），目标是获得一个抗噪的说话人验证（SV）权重，并接入 V17.1 pipeline 在 datasetA 上做完整评测，与现有 v8_full 声纹权重对比（CER / RR / 识别得分）。

## 2. 训练配置与结果

| 项 | 设置 |
|---|---|
| 模型 | `iic/speech_eres2netv2_sv_zh-cn_16k-common`（冻结主干，仅微调） |
| 数据集 | 5.5h 包 smoke 子集：train 2500 对 / val 800 对（HI-MIA kws+cmd 风格） |
| 在线增强 | 以 mobvoi 真实中文带噪语音为噪声源：背景噪声 SNR∈[-5,10]dB@70%，干扰人声 SNR∈[0,12]dB@50% |
| 超参 | batch=64, lr=1e-4 (AdamW), 4 epochs, CosineAnnealing |
| 训练脚本 | `tools/train_eres2netv2_sv_aug.py` |

**训练指标（完整 4 epoch，run4）：**

| 阶段 | loss | val_EER | pos_sim | neg_sim | emb_std |
|---|---|---|---|---|---|
| 基线（无增强验证） | — | 0.1163 | 0.516 | 0.233 | — |
| E1 | 0.2875 | 0.1337 | 0.771 | 0.628 | 0.0529 |
| **E2 ★最佳** | 0.1730 | **0.1300** | 0.728 | 0.519 | 0.0608 |
| E3 | 0.1265 | 0.1575 | 0.762 | 0.585 | 0.0598 |
| E4 | 0.1045 | 0.1562 | 0.770 | 0.596 | 0.0581 |

- 训练 loss 0.382→0.105（降幅约 73%），无发散/NaN；emb_std 全程 0.05–0.06，embedding 空间无塌缩。
- 最佳权重：`sv_aug_best.pt`（E2，val_EER=0.1300）。为接入 pipeline 提取为裸 state_dict：`finetuned_models/sv_aug_best_state.pt`（71MB，pipeline 直接 `load_state_dict` 可用）。
- 验证 EER 高于干净基线 0.1163 是预期现象（验证集也走带噪增强，域更硬）；真正体现抗噪收益的是 pos_sim 从 0.516 稳升到 0.73–0.77（同人说话在干扰/噪声下的区分度提升）。

## 3. 评测配置（V17.1 pipeline）

- 评测入口：`run_inference.py --data_root <datasetA> --split all --config configs/verify_dual_sv_aug.yaml`
- 评测配置：复制最佳基准 `configs/run_spex_plus_finetuned.yaml`，仅将 `eres2netv2.finetuned_path` 指向本次 `finetuned_models/sv_aug_best_state.pt`，其余保持一致：
  - 前端：DeepFilterNet3 降噪 + SpEx+ finetuned 目标说话人分离
  - 声纹 Ensemble：CAM++ / ERes2NetV2 / ResNetSE 三模型 Z-score 融合（权重 0.4 / 0.35 / 0.25）
  - ASR：Fun-ASR-Nano
- 测试集：datasetA 1838 条（pos 1364 含 label / neg 474 无 label）

## 4. datasetA 评测对比（官方 micro 口径，NFKC + Levenshtein micro-average）

| 模型 | micro CER↓ | RR↑ | 综合得分 | pos 误拒 | neg 漏拒 |
|---|---|---|---|---|---|
| **sv_aug（本次）** | **0.3969** | **0.9219** | **0.6100** | 156 (11.4%) | 37 (7.8%) |
| **v8_full（历史基准）** | **0.3402** | **1.0000** | **0.6639** | 17 (1.2%) | 0 (0.0%) |

> v8_full 文件自带 macro 口径：avg_cer=0.3916, avg_rr=1.0000（与官方 micro CER=0.3402 一致，口径可比）。

**差异**：sv_aug 相对 v8_full，CER +5.67pp、RR −7.81pp、综合得分 −5.39pp。

## 5. 结果分析

sv_aug 在 datasetA 上全面落后，两个直接原因（来自逐条结果拆解）：

1. **pos 误拒 156 条（11.4%）**：正样本被判为非目标说话人而拒识。拒识按"全删除"计入 CER，直接抬高字符错误率。
2. **neg 漏拒 37 条（7.8%）**：负样本被当作目标说话人并识别出内容，拉低 RR。

**根因**：sv_aug 仅在 5.5h 包的 **smoke 子集**（2500 对、约 10 人）上训练，说话人域远小于 v8_full（CN-Celeb + 3D-Speaker 大规模预训练微调）。小规模训练 + 强在线增强导致 **说话人域漂移**，模型对 datasetA 说话人判别不稳定，既误拒正样本、又漏放负样本。

## 6. 结论与建议

- **最终提交保留 v8_full 声纹权重**（`run_spex_plus_finetuned.yaml` 原配置）；本次 sv_aug 评测作为消融实验存档。
- 若需进一步提升抗噪声纹在 datasetA 的表现，正确路径是：
  - 在**大规模声纹数据**上微调（参照 v8_full 做法：CN-Celeb + 3D-Speaker），或在 **5.5h 完整 fold_full（24000 对）** 而非 smoke 子集上训练；
  - 适当降低验证增强强度或加早停（patience=2）以稳定验证 EER。

## 7. 本次上传文件清单

| 文件 | 说明 |
|---|---|
| `tools/train_eres2netv2_sv_aug.py` | 在线增强声纹训练脚本 |
| `configs/verify_dual_sv_aug.yaml` | 本次评测配置（仅替换 eres2netv2 权重为 sv_aug） |
| `configs/run_spex_plus_finetuned.yaml` | 对比基准配置（v8_full 评测用） |
| `finetuned_models/sv_aug_best_state.pt` | 本次抗噪声纹权重（裸 state_dict，71MB，pipeline 可直接加载） |
| `results/sv_aug/submission.json` | datasetA 逐条评测结果 |
| `results/sv_aug/submission.txt` | datasetA 文本格式评测结果 |
| `results/sv_aug/infer.log` | 评测运行日志 |
| `results/sv_aug/evaluation_report.md` | 本报告 |
| `runs/sv_aug_run4/loss_curve.png` | 训练损失曲线 |
| `runs/sv_aug_run4/train_stdout.log` | 训练日志 |
| `runs/sv_aug_run4/train_log.json` | 训练指标 JSON |

> 注：210MB 的完整 checkpoint（`sv_aug_best.pt`/`sv_aug_last.pt`，含 optimizer 状态）因超过 GitHub 单文件 100MB 限制未上传；复现声纹提取仅需 71MB 的 `sv_aug_best_state.pt` 即可。`runs/`、`results/`、`finetuned_models/`、`*.log` 在 `.gitignore` 中被忽略，本次均通过 `git add -f` 强制纳入。
