# CAM++ 微调声纹模型

基于 datasetA 增强数据对比学习微调 CAM++（`iic/speech_campplus_sv_zh-cn_16k-common`）。

## 模型文件

| 文件 | 训练数据 | 说明 |
|------|---------|------|
| `camplus_v3_fold0.pt` | 80%（fold_0 train） | **推荐**。fold_0 验证折无偏验证 Score=0.5936 |
| `camplus_v3_full.pt` | 100%（全量） | 备选。见数据更多但无法在 A 上无偏评估 |

## 训练方法（V3，防塌缩）

- **损失**：双向 margin 对比损失（pos>0.7, neg<0.3，留 0.4 间隔）——不把 pos 推向 cos=1，避免 embedding 空间塌缩
- **采样**：1:1 平衡采样（消除 pos 拉力优势）
- **预处理**：kws 原始 + cmd 降噪（noisereduce），与 pipeline 推理完全一致（解决训练/推理分布不匹配）
- **防塌缩**：冻结主干前半（head/tdnn/block1/transit1）+ 全部 BatchNorm 设 eval + lr=1e-4 + EER 监控

训练脚本：`tools/train_camplus_finetune.py`，数据增强：`tools/augment_dataset.py`

## fold_0 验证折（无偏）完整流水线对比

| 配置 | CER | RR | Score | pos接受 | neg假接受 |
|------|-----|-----|-------|---------|----------|
| V4.1 基线（预训练声纹+自适应分离） | 0.6071 | 0.9579 | 0.5403 | 175/273 | 4/95 |
| **V5 微调 + 禁分离（本模型）** | **0.4527** | 0.9368 | **0.5936** | 242/273 | 6/95 |
| V5.1 微调 + 选择性分离 | 0.4478 | 0.9263 | 0.5914 | 249/273 | 7/95 |

Score 提升 **+0.053**（CER 大降 0.154，RR 仅微降 0.021）。V5.1 在保留分离能力下 Score 仅差 0.0022。

## 使用

`configs/default.yaml`：
```yaml
voiceprint:
  cam_plus:
    finetuned_path: finetuned_models/camplus_v3_fold0.pt
  threshold: 0.67   # 微调模型阈值（无分离时判定）
separation:
  enable: true              # V5.1: 选择性分离（保留分离能力）
  sep_trigger_min: 0.50     # 仅 0.50<=sim<0.67 的边界样本触发分离
  vp_threshold_separated: 0.80  # 分离后用更严格阈值判定
  sim_jump_cap: 0.25        # 分离后 sim 跳变>0.25 则不信任（防 artifact 选错轨）
```
