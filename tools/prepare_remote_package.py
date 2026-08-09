# -*- coding: utf-8 -*-
"""
声纹微调异地打包脚本
把 流水线代码 + 训练脚本 + 数据 + 权重 + 方案文档 打包到指定目录
用法: python tools/prepare_remote_package.py --out D:/remote_package
"""
import os, sys, shutil, argparse, json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="打包输出目录")
    args = ap.parse_args()

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # voice_pipeline 目录
    OUT = args.out
    PKG = os.path.join(OUT, "remote_package")
    os.makedirs(PKG, exist_ok=True)

    def copy(src, dst):
        if not os.path.exists(src):
            print(f"  ⚠️ 缺失: {src}")
            return
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
        print(f"  ✅ {src} → {dst}")

    print("=" * 60)
    print("打包开始 →", PKG)
    print("=" * 60)

    # 1. 流水线代码 (排除体积大的 pretrained 和 results)
    print("\n[1/6] 流水线代码...")
    code_dst = os.path.join(PKG, "voice_pipeline")
    os.makedirs(code_dst, exist_ok=True)
    for item in ["config.py", "pipeline.py", "run_inference.py", "requirements.txt",
                 "configs", "modules", "utils", "tools"]:
        copy(os.path.join(ROOT, item), os.path.join(code_dst, item))
    # tools 里只需要训练相关 + 下载脚本
    keep_tools = {"train_camplus_finetune.py", "train_eres2netv2_finetune.py",
                  "augment_dataset.py", "make_folds.py", "train_all_folds.py",
                  "download_spexplus.py", "download_wespeaker.py",
                  "prepare_processed_train_data.py"}
    tools_dir = os.path.join(code_dst, "tools")
    if os.path.isdir(tools_dir):
        for f in os.listdir(tools_dir):
            if f not in keep_tools:
                p = os.path.join(tools_dir, f)
                if os.path.isfile(p):
                    os.remove(p)
        print(f"  ✅ tools/ 裁剪完成 (保留: {sorted(keep_tools)})")

    # 2. 权重
    print("\n[2/6] 模型权重...")
    pre_dst = os.path.join(code_dst, "pretrained")
    os.makedirs(pre_dst, exist_ok=True)
    copy(os.path.join(ROOT, "pretrained", "spex_plus"), os.path.join(pre_dst, "spex_plus"))
    copy(os.path.join(ROOT, "pretrained", "gtcrn"), os.path.join(pre_dst, "gtcrn"))
    copy(os.path.join(ROOT, "pretrained", "wespeaker_resnet34"), os.path.join(pre_dst, "wespeaker_resnet34"))
    copy(os.path.join(ROOT, "pretrained", "modelscope_cache"), os.path.join(pre_dst, "modelscope_cache"))

    # 3. 微调权重
    print("\n[3/6] 微调权重...")
    fm_dst = os.path.join(code_dst, "finetuned_models")
    os.makedirs(fm_dst, exist_ok=True)
    for f in os.listdir(os.path.join(ROOT, "finetuned_models")):
        if f.endswith(".pt"):
            copy(os.path.join(ROOT, "finetuned_models", f), os.path.join(fm_dst, f))

    # 4. 训练数据 (增强数据集)
    print("\n[4/6] 训练数据 (datasetA_aug, 2.2G)...")
    copy(os.path.join(os.path.dirname(ROOT), "datasetA_aug"), os.path.join(PKG, "datasetA_aug"))

    # 5. 原始数据集
    print("\n[5/6] 原始数据集 (datasetA)...")
    copy("F:/挑杯资料/datasetA", os.path.join(PKG, "datasetA"))

    # 6. 关键测试结果 (仅保留全量json结果, 不带走断点/大文件)
    print("\n[6/6] 关键测试结果...")
    res_dst = os.path.join(PKG, "results_backup")
    os.makedirs(res_dst, exist_ok=True)
    for f in os.listdir(os.path.join(ROOT, "results")):
        if f.startswith("full_inference") and f.endswith(".json"):
            copy(os.path.join(ROOT, "results", f), os.path.join(res_dst, f))

    # 7. 方案文档
    copy(os.path.join(os.path.dirname(ROOT), "声纹微调打包方案.md"), os.path.join(PKG, "声纹微调打包方案.md"))
    copy(os.path.join(os.path.dirname(ROOT), "声纹微调任务分配说明.md"), os.path.join(PKG, "声纹微调任务分配说明.md"))

    print("\n" + "=" * 60)
    print("打包完成!")
    print("总目录:", PKG)
    total = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(PKG) for f in fs)
    print(f"总大小: {total/1024**3:.2f} GB")
    print("=" * 60)

if __name__ == "__main__":
    main()
