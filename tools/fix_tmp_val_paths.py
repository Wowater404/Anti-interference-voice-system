# -*- coding: utf-8 -*-
"""把 tmp_val_fold0 的 jsonl 音频路径改为绝对路径"""
import json, os

tmp_root = 'F:/龙虾/2026-07-18-13-57-00/tmp_val_fold0'
src = 'F:/挑杯资料/datasetA'

for split in ['pos', 'neg']:
    p = os.path.join(tmp_root, f'{split}.jsonl')
    with open(p, encoding='utf-8') as f:
        recs = [json.loads(l) for l in f if l.strip()]
    for r in recs:
        r['唤醒音频'] = os.path.join(src, r['唤醒音频']).replace('\\', '/')
        r['识别音频'] = os.path.join(src, r['识别音频']).replace('\\', '/')
    with open(p, 'w', encoding='utf-8') as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'{split}.jsonl: {len(recs)} 条, 样例 kws={recs[0]["唤醒音频"]}')
