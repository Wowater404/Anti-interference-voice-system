# 唤醒词定位 KWS 方案（完整记录）

> 日期：2026-08-13
> 结论：唤醒词定位必须用专门的 KWS（关键词识别）模型，不能用通用 ASR。
> 本文档记录：问题诊断 → 为什么之前的方案不行 → 正解方案 → 实施步骤。

---

## 一、要解决什么问题

声纹模型微调（CAM++ / ERes2NetV2）和推理时，需要从**唤醒音频 kws** 里提取「目标说话人」的干净声纹 `emb_target`，再用它去识别音频 cmd 里指认目标说话人。

难点：kws 里除了目标说话人的唤醒词，还混着**噪声 + 干扰说话人**。要拿到干净的目标声纹，必须先做两件事：

1. **降噪**（去掉空调等非人声噪声）
2. **盲分离**（SepFormer，把多个说话人拆成单人音轨）

盲分离会把 kws 拆成 2 条单人音轨，但**不知道哪条是目标说话人**。这就是「唤醒词定位」要解决的问题：在拆出来的几条音轨里，认出「说唤醒词的那条」。

---

## 二、诊断结论（实测数据，2026-08-13）

### 事实 1：kws 里多人声重叠普遍，占 60%

能量法统计 40 条 pos 样本（盲分离后两路 RMS 能量判断）：

| 情况 | 占比 |
|------|------|
| 真·两路都有人声（重叠） | **60%**（24/40）|
| 单人声 / 无重叠 | 40%（16/40）|

**结论**：盲分离是必要的，不是过度设计。

### 事实 2：通用 ASR 识别超短唤醒词失效

用 Fun-ASR-Nano 识别「降噪后的 kws」再和唤醒文本比对，实测 30 条：

| 真实唤醒词 | ASR 识别结果 |
|---|---|
| 你好科慕 | 空调开的 / 姑娘我看 / 会有人 |
| hi colmo | 开高吗 / 开关吗 / 卖空了 / 25 |

- 字符相似度：**中位数 0**、均值 0.13、最高 0.29
- `相似度 < 0.5` 的样本：**100%**

**根因**：Fun-ASR-Nano 是通用 ASR，对 <1 秒的超短语音会"脑补"成常见短语，识别结果和真实唤醒词完全对不上。

### 事实 3：盲分离 + ASR 也救不回来

在盲分离后的干净单人音轨上再跑 ASR 匹配唤醒词：

| 指标 | 降噪后直接 ASR | 盲分离后最佳 ASR |
|------|:---:|:---:|
| 相似度中位数 | 0.000 | 0.286 |
| ≥0.5 占比 | 0% | 20% |

**结论**：盲分离只能拆开多人声，治不了"ASR 识别不了短语音"这个本。ASR 路线（含盲分离）整体不可靠。

---

## 三、为什么之前提的方案都不行

### 方案 A：ASR 匹配唤醒文本（已实现，被推翻）

`_process_kws` 现在就是「降噪 → ASR 匹配唤醒文本 → 不够才盲分离」。实测证明：ASR 识别短唤醒词失效，`sim < 0.5` 恒成立，导致 **kws 盲分离 100% 触发**，每条多花 1~2 秒，全量 14704 条要 11 小时，且选路也不准。

### 方案 B：用 cmd 声纹锚定 kws（被用户正确质疑）

思路：cmd 和 kws 同人，先用 cmd 的干净声纹当锚，去 kws 里选目标路。

**漏洞**：cmd 本身也可能含多个说话人（重叠），cmd 的声纹也是"目标 + 干扰"的混合，不干净。用不干净的锚去选路，选不准。用户质疑："cmd 多人声时怎么确定 cmd 里哪个是目标？"——这个漏洞成立。

### 根本症结

「谁是目标」的唯一可靠判据是**「谁说了唤醒词」**（唤醒词只有目标会说，且唤醒文本是白送的输入字段）。所以唤醒词定位绕不开。而唤醒词定位的两个候选方法里：

- ASR → 短语音失效（事实 2、3）
- 声纹 → 需要先有干净目标声纹，但目标声纹正是我们要提取的东西（先有鸡还是先有蛋）

**因此，唯一可靠的路径是：用专门为短关键词检测设计的 KWS 模型，而不是通用 ASR。**

---

## 四、正解方案：KWS 模型 + 完整闭环

### 技术选型

**sherpa-onnx 的 `KeywordSpotter`**（环境已装 sherpa-onnx 1.13.4）：

- 基于 zipformer 流式模型，**专门为关键词/唤醒词检测设计**，对 <1 秒短语音可靠（不同于通用 ASR）。
- 有现成中文模型：`sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01`（中文，wenetspeech 训练），或更新的中英混合 `sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20`。
- **支持运行时动态关键词**：`create_stream(keywords="...")` 可传任意唤醒词，不用写死词表 → 天然覆盖测试集 B 的新唤醒词。

### 完整闭环流程

```
输入：唤醒音频 kws + 唤醒文本 kws_text + 识别音频 cmd

① kws 降噪          → Renoise（去非人声噪声）
② kws 盲分离        → SepFormer16k 拆成 2 条单人音轨（无先验，不依赖音量）
③ 唤醒词定位（KWS） → 2 条音轨各跑 KeywordSpotter 检测 kws_text，命中者 = 目标 kws
④ 提目标声纹        → 从目标 kws 音轨提取 emb_target（CAM++/ERes2NetV2/ResNetSE 三模型）
⑤ cmd 降噪          → Renoise
⑥ cmd 盲分离/目标提取 → 拆成单人音轨
⑦ 声纹指认          → 用 emb_target 去 cmd 各音轨匹配，相似度最高 = 目标 cmd
⑧ ASR               → 目标 cmd 转文字（指令够长，通用 ASR 可靠，这里不需要 KWS）
```

**为什么这样能闭合**：
- 「cmd 多人声」→ ②⑥ 的盲分离解决（盲分离不需要任何先验，直接按统计独立性拆）
- 「谁是目标」→ ③ 的 KWS 解决（唤醒词只有目标会说，KWS 对短唤醒词可靠）
- 两个死结各自归位，不互相依赖。

---

## 五、实施步骤

### 第 1 步：下载中文 KWS 模型

```bash
cd voice_pipeline/pretrained
# 中文 KWS 模型（约几十 MB）
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2
tar xvf sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2
```

解压得到：`encoder-*.onnx`、`decoder-*.onnx`、`joiner-*.onnx`、`tokens.txt`、`keywords.txt`（示例）。

> 下载慢可用国内镜像 hf-mirror.com，或让已联网的电脑下好打包。

### 第 2 步：搞清关键词格式

- 打开模型自带的 `keywords.txt`（和 `keywords_raw.txt`），看懂中文关键词的书写格式（一般是 `汉字/拼音带声调` 或 token 序列，参照示例）。
- 把 22 种唤醒词（你好科慕、小美小美、hi colmo、小惟小惟……）写成同样格式。
- 参考 sherpa-onnx 官方 python 示例：`python-api-examples/keyword-spotter.py`。

### 第 3 步：封装 KWS 模块

新增 `modules/kws.py`，封装一个 `KwsSpotter` 类：

```python
import sherpa_onnx

class KwsSpotter:
    def __init__(self, model_dir):
        self.spotter = sherpa_onnx.KeywordSpotter(
            tokens=model_dir + "/tokens.txt",
            encoder=model_dir + "/encoder-*.onnx",
            decoder=model_dir + "/decoder-*.onnx",
            joiner=model_dir + "/joiner-*.onnx",
            keywords_file=model_dir + "/keywords.txt",   # 预置 22 种唤醒词
            num_threads=4, sample_rate=16000, feature_dim=80,
            provider="cpu",   # 或 cuda
        )

    def detect(self, audio: np.ndarray, sr: int, kws_text: str) -> bool:
        """在 audio 里检测 kws_text 是否出现，返回是否命中"""
        stream = self.spotter.create_stream(keywords=kws_text)  # 运行时动态关键词
        stream.accept_waveform(sr, audio.astype(np.float32))
        # 尾部补 0.4s 静音（KWS 需要 tail padding）
        tail = np.zeros(int(0.4 * sr), dtype=np.float32)
        stream.accept_waveform(sr, tail)
        while self.spotter.is_ready(stream):
            self.spotter.decode_stream(stream)
        result = self.spotter.get_result(stream)
        return bool(result.keyword)  # 命中唤醒词
```

### 第 4 步：改 `_process_kws`（pipeline.py）

把现在的「ASR 匹配唤醒文本」换成「KWS 检测唤醒词」：

```python
def _process_kws(self, audio, kws_text, sr):
    denoised = self.denoiser.denoise(audio, sr)
    if not kws_text or not self.kws_enable:
        return denoised, meta

    # 盲分离（无先验，直接拆；重叠占 60%，多数样本需要）
    _, sources = self.kws_separator.separate(denoised, sr)
    if not sources or len(sources) <= 1:
        return denoised, meta

    # KWS 定位：哪条音轨含唤醒词，哪条就是目标
    for src in sources:
        if self.kws_spotter.detect(src, sr, kws_text):
            return src, meta   # 命中唤醒词 → 目标 kws
    return denoised, meta      # 都没命中 → 回退降噪结果
```

### 第 5 步：同步训练侧 + 推理侧

- `tools/prepare_processed_train_data.py` 的 `process_sample` 调用同一个 `_process_kws`（已经是这样），自动对齐。
- `pipeline.py` 的 `process_sample` / `process_dataset_ensemble` 同步生效。
- 加载阶段新增 `pipeline.kws_spotter = KwsSpotter(...)` 和 `load()`。

### 第 6 步：验证

1. 单条验证：抽 30 条 kws，看 KWS 定位命中率（预期远高于 ASR 的 20%）。
2. 对比 `_process_kws` 改前/改后，`kws_separated` 触发率（预期从 100% 降到 ~60%，即只有真重叠才分离）。
3. 跑通清洗 + 训练，看 val EER 和误拒样本数是否改善。

---

## 六、注意事项

1. **运行时动态关键词优先**：`create_stream(keywords=...)` 传当前样本的唤醒文本，比写死 22 种词表更灵活，且天然支持测试集 B 的新唤醒词。
2. **KWS 模型采样率**：模型支持非 16kHz 输入，但统一用 16kHz 最稳；输入必须是单声道 16-bit。
3. **尾部补静音**：KWS 流式解码需要 0.4s tail padding，否则可能漏检。
4. **性能**：KWS 用 CPU 即可（模型小，RTF 很低），不占 GPU；真正耗 GPU 的是盲分离 SepFormer。
5. **回退保护**：两条音轨都没命中唤醒词时，回退用降噪结果，不要让流水线崩溃。
6. **cmd 侧不需要 KWS**：指令够长，通用 ASR 可靠，KWS 只用于 kws 侧的超短唤醒词定位。
7. **训练/推理对齐**：`_process_kws` 是唯一入口，训练侧（prepare_processed）和推理侧（pipeline）复用同一个方法，保证分布一致。

---

## 七、本次会话遗留的待办

- [ ] 下载中文 KWS 模型并封装 `modules/kws.py`
- [ ] 改 `_process_kws` 用 KWS 替代 ASR 匹配
- [ ] 验证 KWS 命中率 + 盲分离触发率下降
- [ ] 重新跑清洗（prepare_processed）→ make_folds → 训练 CAM++ / ERes2NetV2
- [ ] 两个声纹模型训完后，恢复 CAM++ 训练脚本的 `--init_from`（第 416 行）
- [ ] 数据清洗前记得：运行用 `export PATH=/f/work/Anaconda/envs/zhinnegjiaju/Library/bin:$PATH`（否则 cuDNN 报 cudnnCreate 错误）
