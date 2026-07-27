# 抗干扰语音指令识别系统 - SpEx+ 最终版

本分支在原 V5.1 流水线基础上，将多人语音处理阶段最终确定为
**SpEx+ 16 kHz 目标说话人提取模型**。

流水线：

`noisereduce -> SpEx+ -> CAM++ -> Paraformer`

SpEx+ 同时读取：

- 待识别的混合/指令音频；
- 同一条样本的唤醒音频，作为目标说话人参考。

模型直接输出一条目标说话人音轨。权重、配置或参考音频缺失时程序会
立即停止，不会静默退化为直通音频。

## 最终实验结果

测试集：datasetA，共 1838 条（pos 1364 / neg 474）。

| 指标 | 结果 |
|---|---:|
| CER | 0.5765 |
| 拒识率 RR | 0.9346 |
| 识别项得分 | 0.5432 |
| 总推理时间 | 353.57 秒 |
| SpEx+ 实际触发 | 761 条 |
| 平均触发提取时间 | 0.0636 秒 |
| 硬错误 | 0 |

识别项得分不包含比赛的推理时间和内存效率项。完整、可复核记录见
`experiment_logs/spex_plus_test.log`。

## 安装

建议使用 Python 3.12、PyTorch 和 torchaudio 的匹配 CUDA 版本。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

下载并严格校验 SpEx+ 官方检查点：

```powershell
.\.venv\Scripts\python.exe tools\download_spexplus.py
```

检查点保存位置：

`pretrained/spex_plus/checkpoint.pth`

固定 SHA256：

`2d6a2f2b404fd18a809eb82052fd64ef0bd986f410b1043bc666b54121e44b5c`

## 运行

单样本：

```powershell
.\.venv\Scripts\python.exe run_inference.py `
  --kws "path\to\kws.wav" `
  --cmd "path\to\cmd.wav" `
  --label "目标文本" `
  --output "results\spex_plus\single.json"
```

完整 datasetA：

```powershell
.\.venv\Scripts\python.exe run_inference.py `
  --config configs\default.yaml `
  --data_root "path\to\datasetA" `
  --split all `
  --output "results\spex_plus\datasetA_spexplus_submission.json" `
  --checkpoint "results\spex_plus\datasetA_spexplus_checkpoint.json"
```

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
configs/default.yaml                 SpEx+ 最终运行配置
modules/spexplus_separator.py        SpEx+ 网络与流水线适配器
modules/separator.py                 分离/提取模型工厂
pipeline.py                          完整推理流水线
tools/download_spexplus.py           权重下载与SHA256校验
experiment_logs/spex_plus_test.log   最终实验记录
```

`pretrained/`、`results/`、虚拟环境和缓存均不进入 Git 提交。项目保留
原流水线的通用组件及队友原有文件，但不包含其他候选分离模型的新增配置、
权重、下载脚本或实验结果。
