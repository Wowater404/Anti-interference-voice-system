# -*- coding: utf-8 -*-
"""降噪组更新记录汇总 PDF"""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

font_name = "Helvetica"
for fp in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"]:
    if os.path.exists(fp):
        try:
            pdfmetrics.registerFont(TTFont("CJK", fp))
            font_name = "CJK"
            break
        except Exception:
            pass

styles = getSampleStyleSheet()
title_style = ParagraphStyle("T", parent=styles["Title"], fontName=font_name, fontSize=16,
                             leading=22, textColor=colors.HexColor("#0C447C"), alignment=1, spaceAfter=4)
sub_style = ParagraphStyle("S", parent=styles["Normal"], fontName=font_name, fontSize=10,
                           leading=14, textColor=colors.HexColor("#5F5E5A"), alignment=1, spaceAfter=12)
h2_style = ParagraphStyle("H2", parent=styles["Heading2"], fontName=font_name, fontSize=13,
                          leading=18, textColor=colors.HexColor("#185FA5"), spaceBefore=12, spaceAfter=6)
body_style = ParagraphStyle("B", parent=styles["Normal"], fontName=font_name, fontSize=10,
                            leading=15, textColor=colors.HexColor("#2C2C2A"), spaceAfter=4)
note_style = ParagraphStyle("N", parent=body_style, fontSize=9, leading=13,
                            textColor=colors.HexColor("#5F5E5A"))

def P(text, style=body_style):
    return Paragraph(text, style)

doc = SimpleDocTemplate("results/降噪组更新记录汇总.pdf", pagesize=A4,
                        leftMargin=18*mm, rightMargin=18*mm, topMargin=18*mm, bottomMargin=18*mm)
story = []
story.append(Paragraph("降噪组更新记录汇总", title_style))
story.append(Paragraph("挑战杯 XH-202615 抗干扰语音指令识别 | 降噪模块演进全记录 (2026-07-18 ~ 08-09)", sub_style))

# ============ 一、时间线总览 ============
story.append(Paragraph("一、降噪方案演进时间线", h2_style))
timeline = [
    ["时间", "方案", "关键变化"],
    ["07-18", "DeepFilterNet3 (初选)", "立项时主选降噪方案 (48kHz需重采样, Rust编译) 备选 FullSubNet+"],
    ["07-19", "NoiseReduce (V2)", "接入频谱门限降噪; ASR改用16kHz降噪音频 (避免8kHz降采样失真), CER显著降低"],
    ["07-19", "SepFormer-16k 分离", "降噪+分离组合; 但分离后CER高于降噪 (0.4267 vs 0.3752), 全量分离弊大于利"],
    ["07-23", "训练/推理分布对齐", "发现训练用原始cmd但推理用降噪cmd → neg sim升高0.124 → RR崩; cmd降噪重训解决"],
    ["07-27", "GTCRN (dev分支)", "降噪替换为GTCRN的备选分支; 未成为主线"],
    ["08-06", "降噪=负优化 (3model)", "3model zscore+Fun-ASR-Nano下, 降噪+分离净负 (-5分); 逐样本归因: 误杀106pos > 救回62pos"],
    ["08-07", "Renoise 接入", "降噪组重做频谱门限降噪 (renoise.py); 修复工厂接线 (denoiser.py), 暴露4参数; SNR改善14.7dB"],
    ["08-07", "Renoise 参数调优", "9组参数抽样扫描 → 最优: stationary=False, prop=0.8, nstd=1.5, nfft=1024"],
    ["08-08", "Renoise 锁定", "Renoise+分离+双微调融合 = 最终方案核心 (验证折60.91 → 全量64.66)"],
]
t = Table(timeline, colWidths=[16*mm, 30*mm, 100*mm])
t.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0C447C")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("ALIGN", (0,0), (-1,-1), "CENTER"),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#B5D4F4")),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#E6F1FB")]),
    ("FONTNAME", (0,0), (-1,-1), font_name),
    ("FONTSIZE", (0,0), (-1,-1), 8.5),
    ("TOPPADDING", (0,0), (-1,-1), 4),
    ("BOTTOMPADDING", (0,0), (-1,-1), 4),
]))
story.append(t)
story.append(Spacer(1, 6))

# ============ 二、Renoise 最终方案 ============
story.append(Paragraph("二、最终降噪方案 (Renoise 频谱门限降噪)", h2_style))
story.append(P("<b>算法原理:</b> 频谱门限 (Spectral Gating) — 估计噪声基底, 低于阈值的时频点按比例衰减, 保留语音能量"))
story.append(P("<b>已调优参数 (抽样扫描9组选出):</b>"))
params = [
    ["参数", "含义", "最终值"],
    ["stationary", "噪声假设: True=静态(空调/风扇), False=低分位数(时变噪声)", "False (最优)"],
    ["prop_decrease", "噪声抑制强度 (0-1, 越大抑制越多)", "0.8"],
    ["n_std_thresh", "门限阈值倍数 (噪声基底×n倍为门限)", "1.5"],
    ["n_fft", "FFT窗口大小", "1024"],
]
pt = Table(params, colWidths=[28*mm, 90*mm, 30*mm])
pt.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0F6E56")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("ALIGN", (0,0), (-1,-1), "CENTER"),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#9FE1CB")),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#E1F5EE")]),
    ("FONTNAME", (0,0), (-1,-1), font_name),
    ("FONTSIZE", (0,0), (-1,-1), 8.5),
    ("TOPPADDING", (0,0), (-1,-1), 4),
    ("BOTTOMPADDING", (0,0), (-1,-1), 4),
]))
story.append(pt)
story.append(Spacer(1, 4))
story.append(P("验证: 接入工厂后降噪 SNR 改善 14.7dB"))

# ============ 三、关键决策与教训 ============
story.append(Paragraph("三、关键决策与教训", h2_style))
lessons = [
    "1. <b>16kHz 优先:</b> 8kHz 降采样丢失高频, 音质降级 → 全部降噪/分离在 16kHz 原生域完成",
    "2. <b>训练/推理分布必须一致:</b> 训练用原始cmd但推理用降噪cmd → neg相似度升高0.124 → RR崩 (V3教训); 解决: 训练数据cmd也走降噪预处理",
    "3. <b>降噪是双刃剑:</b> 3model zscore 组合下降噪误杀106条pos (raw接受→拒识), 净效果为负 → 最终方案必须整体调优而非单独选最优降噪",
    "4. <b>调参空间:</b> NoiseReduce 底层 n_std_thresh/n_fft 默认即可, 最优组合 stationary=False + prop=0.8 对空调/风扇噪声最有效",
    "5. <b>工程接线:</b> 分支代码若未接入 denoiser.py 工厂则 config 设 renoise 会静默直通 (等于没降噪) — 审查时务必验证工厂返回真实对象",
]
for l in lessons:
    story.append(P(l))

# ============ 四、最终性能 ============
story.append(Paragraph("四、降噪在最终流水线中的作用", h2_style))
final_tbl = [
    ["指标", "数值", "说明"],
    ["最终80分 (V8)", "64.66", "Renoise+分离+双微调融合, 全量1838条"],
    ["验证折无偏", "60.91", "双fold0融合, 模型未见数据 (预测比赛参考)"],
    ["RR", "0.9958", "neg假接受仅2/474"],
    ["CER", "0.3793", "官方micro口径"],
    ["推理时间", "1427s (1838条)", "本地4050实测, 单条0.776s"],
    ["峰值内存", "7.14 GB", "三模型+ASR+分离加载峰值"],
]
ft = Table(final_tbl, colWidths=[32*mm, 40*mm, 76*mm])
ft.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#854F0B")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("ALIGN", (0,0), (-1,-1), "CENTER"),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#FAC775")),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#FAEEDA")]),
    ("FONTNAME", (0,0), (-1,-1), font_name),
    ("FONTSIZE", (0,0), (-1,-1), 8.5),
    ("TOPPADDING", (0,0), (-1,-1), 4),
    ("BOTTOMPADDING", (0,0), (-1,-1), 4),
]))
story.append(ft)
story.append(Spacer(1, 8))

story.append(Paragraph("注: 全量验证口径为模型见过全部datasetA数据 (偏乐观); 验证折无偏口径为模型未见过fold_0验证折数据 (预测比赛更诚实)。推理时间/内存为本地 RTX 4050 实测, 官方 RTX 3090 预计快2.5-3倍。", note_style))

doc.build(story)
print("✅ PDF 已生成: results/降噪组更新记录汇总.pdf")
