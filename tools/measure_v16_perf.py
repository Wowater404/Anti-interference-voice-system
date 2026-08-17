# -*- coding: utf-8 -*-
"""
V16 流水线推理时间 + 峰值内存测量
跑 pos 前 N 条 + neg 前 M 条, 统计:
  - 每样本平均耗时, 外推全量
  - RAM 峰值 (ctypes GetProcessMemoryInfo)
  - GPU 显存峰值 (torch.cuda.max_memory_allocated)
用法:
  python tools/measure_v16_perf.py --n_pos 80 --n_neg 20
"""
import os
import sys
import json
import time
import ctypes
from ctypes import wintypes

# [重要] 先 import torch 再写 os.environ
import torch  # noqa: F401
_lib_bin = os.path.abspath(os.path.join(os.path.dirname(sys.executable), 'Library', 'bin'))
if os.path.isdir(_lib_bin):
    os.environ['PATH'] = _lib_bin + os.pathsep + os.environ.get('PATH', '')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PipelineConfig
from pipeline import VoicePipeline

# ---- Windows 峰值内存 (无需 psutil) ----
class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
    ]

def peak_ram_mb():
    try:
        psapi = ctypes.WinDLL('psapi')
        c = PROCESS_MEMORY_COUNTERS()
        c.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return c.PeakWorkingSetSize / (1024 * 1024)
    except Exception:
        return -1.0

def load_samples(jsonl, n):
    with open(jsonl, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()][:n]

def main():
    n_pos, n_neg = 80, 20
    data_root = r"F:/挑杯资料/datasetA"
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "configs", "verify_dual_full.yaml")
    pos = load_samples(os.path.join(data_root, "pos.jsonl"), n_pos)
    neg = load_samples(os.path.join(data_root, "neg.jsonl"), n_neg)

    t0 = time.time()
    print("加载 V16 流水线模型...", flush=True)
    cfg = PipelineConfig(cfg_path)
    pipe = VoicePipeline(cfg)
    pipe.load_models()
    t_load = time.time() - t0
    mem_after_load = peak_ram_mb()
    print(f"模型加载耗时: {t_load:.1f}s | 加载后 RAM: {mem_after_load:.0f} MB", flush=True)

    times = []
    t_start = time.time()
    # pos
    for i, s in enumerate(pos):
        t1 = time.time()
        pipe.process_sample(
            os.path.join(data_root, s["唤醒音频"]),
            os.path.join(data_root, s["识别音频"]),
            label=s.get("识别文本"),
            sample_id=str(s["id"]),
            kws_text=s.get("唤醒文本"),
        )
        times.append(time.time() - t1)
        if (i + 1) % 20 == 0:
            print(f"  pos {i+1}/{n_pos} 累计{(time.time()-t_start):.0f}s 峰值RAM {peak_ram_mb():.0f}MB", flush=True)
    # neg
    for i, s in enumerate(neg):
        t1 = time.time()
        pipe.process_sample(
            os.path.join(data_root, s["唤醒音频"]),
            os.path.join(data_root, s["识别音频"]),
            label=s.get("识别文本"),
            sample_id=str(s["id"]),
            kws_text=s.get("唤醒文本"),
        )
        times.append(time.time() - t1)
    t_total = time.time() - t_start
    n = len(times)
    avg = t_total / n
    mem_peak = peak_ram_mb()
    gpu_mem = 0
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)

    print("\n========== V16 推理性能测量结果 ==========", flush=True)
    print(f"样本数: pos {n_pos} + neg {n_neg} = {n}")
    print(f"总推理耗时: {t_total:.1f}s | 平均 {avg:.3f}s/样本")
    print(f"外推全量 1838 条: {avg*1838/60:.1f} 分钟 ({avg*1838:.0f}s)")
    print(f"峰值 RAM: {mem_peak:.0f} MB")
    print(f"GPU 显存峰值: {gpu_mem:.0f} MB")
    print(f"模型加载耗时: {t_load:.1f}s")
    print(f"==========================================", flush=True)

if __name__ == "__main__":
    main()
