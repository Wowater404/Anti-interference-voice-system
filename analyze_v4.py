"""
V4 推理结果分析: 对比 V2(无分离) vs V4(自适应分离+声纹选轨)
用法: python analyze_v4.py
"""
import json
import unicodedata
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def normalize(text):
    if text is None:
        return ''
    text = unicodedata.normalize('NFKC', str(text))
    text = text.lower()
    text = ''.join(ch for ch in text if not unicodedata.category(ch).startswith('P'))
    return text


def editdist(a, b):
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb]


def cer_for(ref, hyp):
    r = normalize(ref)
    h = normalize(hyp)
    if len(r) == 0:
        return 0, 0
    return editdist(h, r), len(r)


def load_results(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    results = data['result']['results']
    for x in results:
        if x.get('content') in ('null', 'Null', 'NULL'):
            x['content'] = ''
    return results


def load_id_sets():
    pos_ids = set()
    neg_ids = set()
    with open('F:/挑杯资料/datasetA/pos.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            pos_ids.add(str(d.get('id', '')))
    with open('F:/挑杯资料/datasetA/neg.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            neg_ids.add(str(d.get('id', '')))
    return pos_ids, neg_ids


def analyze(results, pos_ids, neg_ids, label):
    pe = pl = 0
    acc = 0
    nr = 0
    for r in results:
        iid = r['id']
        c = r.get('content', '')
        lab = r.get('label', '')
        e, l = cer_for(lab, c)
        if iid in pos_ids:
            pe += e
            pl += l
            if c != '':
                acc += 1
        if iid in neg_ids:
            if c == '':
                nr += 1
    cer = pe / pl if pl > 0 else 0
    accept_rate = acc / len(pos_ids) if pos_ids else 0
    rr = nr / len(neg_ids) if neg_ids else 0
    score = 0.4 * (1 - cer) + 0.4 * rr
    print(f"\n=== {label} ===")
    print(f"  CER (micro): {cer:.4f}  ({pe}/{pl})")
    print(f"  pos接受率: {acc}/{len(pos_ids)} = {accept_rate:.3f}")
    print(f"  neg拒识率: {nr}/{len(neg_ids)} = {rr:.3f}")
    print(f"  得分(CER40+RR40): {score:.4f}")
    return {'cer': cer, 'accept': acc, 'accept_rate': accept_rate,
            'rr': rr, 'score': score}


def compare(v2, v4, pos_ids, neg_ids):
    better = worse = same = 0
    pos_better = pos_worse = 0
    rescued = 0  # V2拒识但V4接受
    lost = 0     # V2接受但V4拒识
    diffs = []

    for a, b in zip(v2, v4):
        iid = a['id']
        ispos = iid in pos_ids
        c2 = a.get('content', '')
        c4 = b.get('content', '')
        lab = a.get('label', '')
        e2, l2 = cer_for(lab, c2)
        e4, l4 = cer_for(lab, c4)
        if ispos:
            if c2 == '' and c4 != '':
                rescued += 1
            if c2 != '' and c4 == '':
                lost += 1
        if normalize(c2) != normalize(c4):
            if e4 < e2:
                better += 1
                pos_better += ispos
            elif e4 > e2:
                worse += 1
                pos_worse += ispos
            else:
                same += 1
            diffs.append((iid, 'pos' if ispos else 'neg', c2, c4, lab,
                          e2 / max(l2, 1), e4 / max(l4, 1)))

    print(f"\n=== V2 -> V4 变化 ===")
    print(f"  变好: {better} (pos {pos_better})")
    print(f"  变差: {worse} (pos {pos_worse})")
    print(f"  CER持平(文本不同): {same}")
    print(f"  救回(V2拒识->V4接受): {rescued}")
    print(f"  丢失(V2接受->V4拒识): {lost}")

    print(f"\n--- 变差样本 (前15) ---")
    cnt = 0
    for d in diffs:
        if d[6] > d[5]:
            print(f'  [{d[0]}|{d[1]}] V2="{d[2][:30]}" CER={d[5]:.3f} -> V4="{d[3][:30]}" CER={d[6]:.3f} | label="{d[4][:25]}"')
            cnt += 1
            if cnt >= 15:
                break

    print(f"\n--- 救回样本 (V2拒识->V4接受, 前15) ---")
    cnt = 0
    for a, b in zip(v2, v4):
        if a['id'] in pos_ids and a.get('content', '') == '' and b.get('content', '') != '':
            lab = a.get('label', '')
            e4, l4 = cer_for(lab, b.get('content', ''))
            print(f'  [{a["id"]}] V4="{b["content"][:35]}" CER={e4/max(l4,1):.3f} | label="{lab[:30]}"')
            cnt += 1
            if cnt >= 15:
                break


def main():
    v2 = load_results(os.path.join(PROJECT_ROOT, 'results', 'full_inference_v2.json'))
    v4 = load_results(os.path.join(PROJECT_ROOT, 'results', 'full_inference_v4.json'))
    pos_ids, neg_ids = load_id_sets()

    print(f"V2 样本数: {len(v2)}, V4 样本数: {len(v4)}")
    print(f"pos: {len(pos_ids)}, neg: {len(neg_ids)}")

    m2 = analyze(v2, pos_ids, neg_ids, "V2 无分离")
    m4 = analyze(v4, pos_ids, neg_ids, "V4 自适应分离+声纹选轨")

    print(f"\n=== 改进对比 ===")
    print(f"  CER: {m2['cer']:.4f} -> {m4['cer']:.4f}  ({'改善' if m4['cer'] < m2['cer'] else '变差'} {abs(m4['cer']-m2['cer']):.4f})")
    print(f"  pos接受率: {m2['accept_rate']:.3f} -> {m4['accept_rate']:.3f}  (+{m4['accept']-m2['accept']}条)")
    print(f"  得分: {m2['score']:.4f} -> {m4['score']:.4f}  ({'提升' if m4['score'] > m2['score'] else '下降'} {abs(m4['score']-m2['score']):.4f})")

    compare(v2, v4, pos_ids, neg_ids)


if __name__ == '__main__':
    main()
