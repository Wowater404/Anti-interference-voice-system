"""
下载 WeSpeaker ResNet34 声纹模型 (sherpa-onnx ONNX 格式)

模型: wespeaker_zh_cnceleb_resnet34.onnx
来源: https://github.com/k2-fsa/sherpa-onnx/releases
用途: 三模型 Z-score 集成中的 ResNetSE 声纹提取器

用法:
  python tools/download_wespeaker.py
"""
import os
import sys
import urllib.request

MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/wespeaker_zh_cnceleb_resnet34.onnx"
)
EXPECTED_SIZE = 26_552_559  # ~26.5 MB
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "pretrained", "wespeaker_resnet34",
)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "wespeaker_zh_cnceleb_resnet34.onnx")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if os.path.exists(OUTPUT_PATH):
        size = os.path.getsize(OUTPUT_PATH)
        if size == EXPECTED_SIZE:
            print(f"[OK] 模型已存在: {OUTPUT_PATH} ({size:,} bytes)")
            return 0
        else:
            print(f"[WARN] 文件大小不匹配 (期望 {EXPECTED_SIZE:,}, 实际 {size:,}), 重新下载")

    print(f"正在下载 WeSpeaker ResNet34 模型...")
    print(f"  URL: {MODEL_URL}")
    print(f"  输出: {OUTPUT_PATH}")

    try:
        urllib.request.urlretrieve(MODEL_URL, OUTPUT_PATH)
    except Exception as e:
        print(f"[ERROR] 下载失败: {e}")
        print("  请手动下载并放置到上述路径")
        return 1

    size = os.path.getsize(OUTPUT_PATH)
    if size != EXPECTED_SIZE:
        print(f"[WARN] 下载文件大小不匹配 (期望 {EXPECTED_SIZE:,}, 实际 {size:,})")
        print("  文件可能不完整, 请重新运行或手动下载")

    print(f"[OK] 下载完成: {OUTPUT_PATH} ({size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
