# -*- coding: utf-8 -*-
"""
下载微调声纹模型权重 (CAM++v7 / ERes2NetV2v7)

替代 Git LFS 分发方案：权重托管在 Hugging Face Hub + GitHub Releases 双镜像，
组员 clone 代码后运行本脚本即可拉取权重，无需安装 git-lfs、无需 git lfs pull。

双镜像说明：
  - Hugging Face Hub: https://huggingface.co/{HF_REPO_ID}/resolve/main/{文件名}
  - GitHub Releases:  https://github.com/{GH_REPO}/releases/download/{GH_TAG}/{文件名}
  auto 模式默认先试 Hugging Face（国内通常更快），失败自动回退 GitHub Releases。

默认只下载生产必需的 2 个 full 权重（约 96MB）；fold0 交叉验证权重用 --all 下载。

用法:
  python tools/download_finetuned.py                  # 默认下 full 权重 (auto 双源)
  python tools/download_finetuned.py --source hf      # 强制 Hugging Face
  python tools/download_finetuned.py --source github  # 强制 GitHub Releases
  python tools/download_finetuned.py --all            # 下载全部 5 个权重
  python tools/download_finetuned.py --list           # 只看清单，不下载
"""
import argparse
import hashlib
import os
import sys
import urllib.request

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "finetuned_models")

# ============ 上传后按需修改这两个常量 ============
HF_REPO_ID = "Wowater404/anti-interference-voice-weights"  # Hugging Face 仓库 id
GH_REPO = "Wowater404/Anti-interference-voice-system"      # GitHub 仓库 (owner/repo)
GH_TAG = "v8-weights"                                      # GitHub Release tag
# =================================================

# 权重清单：文件名 -> (字节大小, SHA256, 是否生产必需)
WEIGHTS = {
    "camplus_v7_full.pt": {
        "size": 28098798,
        "sha256": "ee0e4ea937c6b39306571ac84ceb8d57e1e8139bb606d7d259d5a551fb91a6e3",
        "required": True,
    },
    "eres2netv2_v7_full.pt": {
        "size": 71777826,
        "sha256": "7874576b4ab2ac86d3c672dc930eb06f7fb628c877f804537b79a220da3e30d0",
        "required": True,
    },
    "camplus_v6_fold0.pt": {
        "size": 28083616,
        "sha256": "80d00dece558e35c5a8850102845bf2cbef0e10b4fd6540e1e20f2333ad4ba9d",
        "required": False,
    },
    "camplus_v7_fold0.pt": {
        "size": 28098798,
        "sha256": "cae99b65fca7a75f0b46c43dfebedffd6bf5d18bf06042919e417dd478915f7d",
        "required": False,
    },
    "eres2netv2_v7_fold0.pt": {
        "size": 71776073,
        "sha256": "f1769bcd76f0b24b85988c459381656379446a6ed3733f61606cabf3c2d75f64",
        "required": False,
    },
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _progress(block_num, block_size, total_size):
    """urllib 下载进度回调（每 10% 打印一次）"""
    if total_size <= 0:
        return
    pct = block_num * block_size * 100.0 / total_size
    if int(pct) % 10 == 0 and int(pct) != _progress.last:
        _progress.last = int(pct)
        sys.stdout.write(f"\r  下载进度: {pct:3.0f}%")
        sys.stdout.flush()
    if pct >= 100:
        _progress.last = -1
        sys.stdout.write("\r  下载完成，正在校验...      \n")


_progress.last = -1


def download(url, dest, expected_size, expected_sha256):
    """下载单个文件到 dest，并校验大小 + SHA256。已存在且校验通过则跳过。"""
    # 已存在且校验通过 → 跳过
    if os.path.isfile(dest):
        if os.path.getsize(dest) == expected_size and sha256(dest) == expected_sha256:
            print(f"  [跳过] 已存在且校验通过: {os.path.basename(dest)}")
            return True
        print(f"  [重新下载] 文件校验不通过: {os.path.basename(dest)}")

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    print(f"  下载 {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        urllib.request.urlretrieve(req, tmp, reporthook=_progress)
    except Exception as e:
        if os.path.isfile(tmp):
            os.remove(tmp)
        print(f"  [失败] {e}")
        return False

    # 大小校验
    if os.path.getsize(tmp) != expected_size:
        os.remove(tmp)
        print(f"  [失败] 大小不匹配 (期望 {expected_size}, 实际 {os.path.getsize(tmp)})")
        return False

    # SHA256 校验
    actual = sha256(tmp)
    if actual != expected_sha256:
        os.remove(tmp)
        print(f"  [失败] SHA256 校验失败\n    期望 {expected_sha256}\n    实际 {actual}")
        return False

    os.replace(tmp, dest)
    print(f"  [OK] {os.path.basename(dest)}")
    return True


def urls_for(name, source):
    """按 source 返回候选下载 URL 列表（auto 时 HF 在前、GitHub 在后）。"""
    if source == "hf":
        return [f"https://huggingface.co/{HF_REPO_ID}/resolve/main/{name}"]
    if source == "github":
        return [f"https://github.com/{GH_REPO}/releases/download/{GH_TAG}/{name}"]
    # auto
    return [
        f"https://huggingface.co/{HF_REPO_ID}/resolve/main/{name}",
        f"https://github.com/{GH_REPO}/releases/download/{GH_TAG}/{name}",
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["auto", "hf", "github"], default="auto",
                    help="下载源 (默认 auto: HF 优先, 失败回退 GitHub)")
    ap.add_argument("--all", action="store_true",
                    help="下载全部权重 (含 fold0 交叉验证权重, 默认只下生产必需)")
    ap.add_argument("--list", action="store_true", help="仅打印权重清单")
    ap.add_argument("--output-dir", default=OUTPUT_DIR,
                    help="权重输出目录 (默认 finetuned_models/)")
    args = ap.parse_args()

    if args.list:
        print("微调权重清单:")
        total = 0
        for name, meta in WEIGHTS.items():
            tag = "生产必需" if meta["required"] else "fold0可选"
            print(f"  {name:<28} {meta['size']/1e6:6.1f}MB  [{tag}]")
            total += meta["size"]
        print(f"  合计: {total/1e6:.1f}MB (生产必需 {(sum(m['size'] for m in WEIGHTS.values() if m['required']))/1e6:.1f}MB)")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    targets = {n: m for n, m in WEIGHTS.items() if args.all or m["required"]}
    print(f"下载源: {args.source}  |  目标: {len(targets)} 个权重")
    if not args.all:
        print("  (仅生产必需 full 权重；fold0 权重用 --all 下载)")

    fail = []
    for name, meta in targets.items():
        dest = os.path.join(args.output_dir, name)
        ok = False
        for url in urls_for(name, args.source):
            if download(url, dest, meta["size"], meta["sha256"]):
                ok = True
                break
        if not ok:
            fail.append(name)

    if fail:
        print(f"\n[ERROR] 以下权重下载失败: {fail}")
        print("  请检查 HF_REPO_ID / GH_REPO / GH_TAG 是否已上传对应文件，")
        print("  或手动下载后放入 finetuned_models/ 目录。")
        sys.exit(1)
    print("\n全部权重就绪。")


if __name__ == "__main__":
    main()
