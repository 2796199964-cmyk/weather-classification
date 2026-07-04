import os
import sys
import glob
import random
import time
import csv
import numpy as np
import cv2
from datetime import datetime
from collections import defaultdict

CONFIG = {
    "source": r"train",  # 测试图片根目录
    #"source": r"datasets/6a39ed934d7b489daf5f80a4-momodel/train",  # 测试图片根目录2
    "report_dir": r"results/infer",
    "sample_size": 100,  # 设为 None 则全量测试，设为整数则随机抽样
    "seed": 42,
}

# ================= 1. 修正导入 =================
# 使用 as 别名来兼容下方代码，并手动实现 abs_from_script
from main import (
    predict as main_predict,
    WEATHER_CLASSES_EN as CLASS_NAMES_EN,
    WEATHER_CLASSES_CN as CLASS_NAMES_CN,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    SCRIPT_DIR
)


def abs_from_script(relative_path):
    """替代 main.py 中缺失的 abs_from_script 函数"""
    if os.path.isabs(relative_path):
        return relative_path
    return os.path.normpath(os.path.join(SCRIPT_DIR, relative_path))


# ================= 配置 =================


def set_seed(seed):
    random.seed(seed);
    np.random.seed(seed)


def safe_imread_bgr(path):
    """读取原始图片"""
    if path is None or not os.path.exists(path): return None
    try:
        arr = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def collect_images(source, sample_size=None):
    source = abs_from_script(source)
    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        images.extend(glob.glob(os.path.join(source, ext)))
        images.extend(glob.glob(os.path.join(source, "*", ext)))
    images = sorted(set(images))
    if sample_size and sample_size < len(images):
        images = sorted(random.sample(images, sample_size))
    print(f"[INFO] predict.py 收集图片: {len(images)} 张")
    return images


def infer_gt_from_path(img_path):
    parent = os.path.basename(os.path.dirname(img_path)).lower()
    if parent in CLASS_TO_IDX: return CLASS_TO_IDX[parent]
    base = os.path.splitext(os.path.basename(img_path))[0].lower()
    for name, idx in CLASS_TO_IDX.items():
        if base.startswith(name) or f"_{name}" in base or f"-{name}" in base:
            return idx
    return None


def generate_confusion_matrix(results, report_dir):
    """计算并保存混淆矩阵及报告"""
    cm = np.zeros((4, 4), dtype=np.int64)
    valid_count = 0

    for r in results:
        if r["gt"] >= 0:
            cm[r["gt"], r["pred_idx"]] += 1
            valid_count += 1

    total = int(cm.sum())
    correct = int(np.trace(cm))
    acc = correct / total if total else 0.0

    os.makedirs(report_dir, exist_ok=True)

    # 1. 保存详细 CSV
    csv_path = os.path.join(report_dir, "infer_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["文件名", "预测标签", "真实标签", "是否正确"])
        for r in results:
            gt_name = CLASS_NAMES_EN[r["gt"]] if r["gt"] >= 0 else "Unknown"
            writer.writerow([os.path.basename(r["path"]), r["pred"], gt_name, "Yes" if r["correct"] else "No"])

    # 2. 保存 TXT 报告 (含混淆矩阵)
    report_path = os.path.join(report_dir, "infer_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("ConvNeXtCrossMask 推理与混淆矩阵报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"验证样本数: {total}\n")
        f.write(f"总体准确率: {correct}/{total} = {acc * 100:.2f}%\n\n")

        for i, name in enumerate(CLASS_NAMES_CN):
            row_total = int(cm[i].sum())
            row_ok = int(cm[i, i])
            f.write(f"{name} 准确率: {row_ok}/{row_total} = {(row_ok / row_total * 100 if row_total else 0):.2f}%\n")

        f.write("\n" + "=" * 40 + "\n")
        f.write("混淆矩阵 (行=真实，列=预测)\n")
        f.write("        " + "    ".join(f"{name:<6}" for name in CLASS_NAMES_CN) + "\n")
        for i, name in enumerate(CLASS_NAMES_CN):
            f.write(f"{name:<6}  " + "    ".join(f"{int(x):<6}" for x in cm[i]) + "\n")

    print(f"\n[INFO] 报告已保存: {report_path}")
    print("\n" + open(report_path, encoding="utf-8").read())


def main():
    set_seed(CONFIG["seed"])

    # 1. 收集并读取图片
    img_paths = collect_images(CONFIG["source"], CONFIG["sample_size"])
    if not img_paths:
        print("❌ 未找到任何图片，退出。")
        return

    results = []
    start_time = time.time()

    print("\n🔍 开始逐张读取图片并推理...")
    for i, path in enumerate(img_paths):
        # 读取原始图片
        img_bgr_raw = safe_imread_bgr(path)
        if img_bgr_raw is None:
            continue

        # ================= 核心修改：图像预处理 =================
        # main.py 的 predict(X) 严格要求输入是 224x224 的 BGR 图像
        img_bgr_224 = cv2.resize(img_bgr_raw, (224, 224), interpolation=cv2.INTER_LINEAR)

        # 获取真实标签 (Ground Truth)
        gt_idx = infer_gt_from_path(path)

        # 调用 main.py 的全局 predict 函数
        pred_label_str = main_predict(img_bgr_224)

        # 防止返回的标签不在字典中
        if pred_label_str not in CLASS_TO_IDX:
            print(f"[WARN] 未知预测标签: {pred_label_str}，跳过")
            continue

        pred_idx = CLASS_TO_IDX[pred_label_str]

        is_correct = (gt_idx >= 0 and pred_idx == gt_idx)
        results.append({
            "path": path,
            "pred": pred_label_str,
            "pred_idx": pred_idx,
            "gt": gt_idx,
            "correct": is_correct
        })

        if (i + 1) % 50 == 0:
            print(f"  已处理 {i + 1}/{len(img_paths)} 张...")

    elapsed = time.time() - start_time
    if results:
        print(f"\n✅ 推理完成！耗时: {elapsed:.2f}s ({elapsed / len(results):.3f}s/张)")
        # 生成混淆矩阵与报告
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = os.path.join(abs_from_script(CONFIG["report_dir"]), stamp)
        generate_confusion_matrix(results, report_dir)
    else:
        print("❌ 没有成功处理任何图片。")


if __name__ == "__main__":
    main()