# -*- coding: utf-8 -*-
"""生成流水线版本结果对比表 PDF"""
import json, os, sys
sys.path.insert(0, '.')
from utils.metrics import compute_micro_cer

RESULT_DIR = 'results'

# (文件名, 版本名, 说明/配置)
VERSIONS = [
    ("full_inference_v2.json", "V2", "无分离基线 (降噪+声纹+ASR)"),
    ("full_inference_v3.json", "V3", "SepFormer16k全量分离 (能量选轨)"),
    ("full_inference_v4_1.json", "V4.1", "自适应分离+声纹选轨"),
    ("full_inference_v5_finetuned.json", "V5", "CAM++微调 (fold0)"),
    ("full_inference_v6_full.json", "V6", "V6全量 (Renoise+分离+微调)"),
    ("full_inference_camplus_v7.json", "V7", "CAM++v7微调 (Renoise+分离)"),
    ("final_inference_dual_full.json", "V8 最终", "双微调fold_full (CAM++v7+ERes2NetV2v7)"),
]

def extract(p):
    try:
        d = json.load(open(os.path.join(RESULT_DIR, p), encoding='utf-8'))
        r = d['result']
        results = r.get('results', [])
        pos = [x for x in results if x.get('label') and x.get('label') is not True
               and (isinstance(x.get('label'), bool) or str(x.get('label')).strip())]
        # 兼容: label 可能是 bool(True/False) 或 文本(非空=pos)
        pos = [x for x in results if (isinstance(x.get('label'), bool) and x.get('label'))
               or (not isinstance(x.get('label'), bool) and str(x.get('label', '')).strip())]
        neg = [x for x in results if x not in pos]
        cer = float(r.get('final_cer', compute_micro_cer(pos) if pos else 0))
        accn = [x for x in neg if x.get('content')]
        rr = 1 - len(accn) / len(neg) if neg else 0
        dur = float(r.get('duration', 0))
        peak = r.get('peak_memory_gb')
        n = len(results)
        return {'cer': cer, 'rr': rr, 'dur': dur, 'peak': peak, 'n': n,
                'pos_acc': len([x for x in pos if x.get('content')]) / len(pos) if pos else 0,
                'neg_fa': len(accn)}
    except Exception as e:
        return None

rows = []
for fn, name, desc in VERSIONS:
    data = extract(fn)
    if data:
        score = (1 - data['cer']) * 40 + data['rr'] * 40
        rows.append({
            'ver': name, 'desc': desc, 'cer': data['cer'], 'rr': data['rr'],
            'score': score, 'dur': data['dur'], 'peak': data['peak'],
            'n': data['n'], 'pos_acc': data['pos_acc'], 'neg_fa': data['neg_fa'],
        })
        print(f"{name}: CER={data['cer']:.4f} RR={data['rr']:.4f} Score={score:.2f} dur={data['dur']:.0f}s peak={data['peak']} n={data['n']}")

# 保存汇总数据
with open('results/summary_table.json', 'w', encoding='utf-8') as f:
    json.dump(rows, f, ensure_ascii=False, indent=1, default=str)
print(f"\n共 {len(rows)} 个版本")
