# -*- coding: utf-8 -*-
"""
ASR 输出后处理: 数字归一化 (阿拉伯数字 → 中文数字)

背景: 官方 CER 不做数字转中文 (NFKC只转全半角),
      ASR 输出 "26度" 而 label 是 "二十六度" → 被算成错误
      (三份报告 + 我方验证均确认此问题)

规则:
  1. 纯数字串 (26) → 中文读数 (二十六)
  2. 数字+汉字单位 (26度) → 二十六度
  3. 百分比 (30% / 百分之30) → 百分之三十
  4. 数字+字母 (4G) → 四G
  5. 小数 (3.5) → 三点五
  6. 数字在词中间 (W25) → 保留? 默认转 (但英文/品牌词不转, 见 _should_skip)
"""
import re

# 中文数字
_CN_DIGITS = "零一二三四五六七八九"
_CN_UNITS = ["", "十", "百", "千", "万", "亿"]


def _num_to_cn(n: int) -> str:
    """整数 → 中文数字 (支持 0~99999999, 超过返回空串由调用方保留原文)"""
    if n == 0:
        return "零"
    if n < 0:
        return "负" + _num_to_cn(-n)
    # 安全保护: 超过 8 位 (1亿) 超出本函数支持范围, 返回 None 表示不转换
    if n >= 100000000:
        return None

    def _under_10000(x: int) -> str:
        if x == 0:
            return ""
        result = ""
        # 千位
        q = x // 1000
        if q:
            result += _CN_DIGITS[q] + "千"
        x %= 1000
        # 百位
        b = x // 100
        if b:
            result += _CN_DIGITS[b] + "百"
        elif result and x % 100:
            result += "零"
        x %= 100
        # 十位
        t = x // 10
        if t:
            if t == 1 and not result:
                result += "十"
            else:
                result += _CN_DIGITS[t] + "十"
        elif result and x % 10:
            result += "零"
        x %= 10
        # 个位
        if x:
            result += _CN_DIGITS[x]
        return result

    result = ""
    # 万位以上
    w = n // 10000
    if w:
        result += _under_10000(w) + "万"
    n %= 10000
    rest = _under_10000(n)
    if rest:
        result += rest
    return result


def _num_to_cn_decimal(num_str: str) -> str:
    """小数 '3.5' → '三点五' (超大数字保留原文)"""
    if "." in num_str:
        int_part, frac_part = num_str.split(".", 1)
        cn = _num_to_cn(int(int_part)) if int_part else "零"
        if cn is None:
            return num_str  # 超大整数部分不转换
        result = cn
        if frac_part:
            result += "点" + "".join(_CN_DIGITS[int(d)] for d in frac_part if d.isdigit())
        return result
    cn = _num_to_cn(int(num_str))
    return cn if cn is not None else num_str


def _should_skip(context: str, match_start: int, match_end: int) -> bool:
    """判断数字是否在不需要转换的上下文中 (如英文单词、品牌名、型号)"""
    before = context[:match_start]
    after = context[match_end:]

    # 前面紧跟【英文字母】 (如 W25, V2) → 是型号/编号, 跳过
    # 注意: 用 ASCII 判断, 中文汉字不触发 (否则 '风量调到30' 会被误判为型号)
    if before and ('A' <= before[-1] <= 'Z' or 'a' <= before[-1] <= 'z'):
        return True
    return False


def _cn_or_orig(n: int, orig: str) -> str:
    """_num_to_cn 的安全封装: 超大数字返回原文"""
    cn = _num_to_cn(n)
    return cn if cn is not None else orig


def normalize_digits(text: str) -> str:
    """
    阿拉伯数字 → 中文数字归一化

    Args:
        text: ASR 识别文本 (已去标点)

    Returns:
        归一化后的文本 (数字转中文)
    """
    if not text:
        return text

    # 1. 百分比: 30% / 百分之30 → 百分之三十
    text = re.sub(r"百分之(\d+)", lambda m: "百分之" + _cn_or_orig(int(m.group(1)), m.group(0)), text)
    text = re.sub(r"(\d+)%", lambda m: "百分之" + _cn_or_orig(int(m.group(1)), m.group(0)), text)

    # 2. 小数: 3.5 → 三点五
    text = re.sub(r"(\d+\.\d+)", lambda m: _num_to_cn_decimal(m.group(1)), text)

    # 3. 整数: 26 → 二十六 (跳过型号场景: 数字前紧跟字母如 W25)
    #    数字后跟单位/字母也转换: 26度→二十六度, 9点→九点, 4G→四G
    text = re.sub(r"(\d+)([度%倍个只台盏件套双点G])", lambda m: _cn_or_orig(int(m.group(1)), m.group(1)) + m.group(2), text)

    # 4. 剩余纯数字 (前面不是字母才转, 避免 W25 这类型号)
    def _int_repl(m: re.Match) -> str:
        if _should_skip(m.string, m.start(), m.end()):
            return m.group(0)
        return _cn_or_orig(int(m.group(1)), m.group(1))

    text = re.sub(r"(\d+)", _int_repl, text)

    return text


if __name__ == "__main__":
    # 自测
    tests = [
        ("空调调到26度", "空调调到二十六度"),
        ("二十六度", "二十六度"),           # 已中文, 不变
        ("百分之30", "百分之三十"),
        ("风量30%", "百分之三十"),
        ("温度3.5", "温度三点五"),
        ("打开4G", "打开四G"),
        ("W25型号", "W25型号"),            # 型号跳过
        ("预约明天9点", "预约明天九点"),
        ("100", "一百"),
        ("26", "二十六"),
    ]
    for inp, exp in tests:
        got = normalize_digits(inp)
        status = "✅" if got == exp else f"❌ (得 {got})"
        print(f"{status} {inp!r} → {got!r} (期望 {exp!r})")
