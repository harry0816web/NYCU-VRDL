"""
analyze_dataset.py - 統計 train/test dataset 各種特性

統計項目：
  - 每張圖的圖片尺寸分佈
  - GT instance 總數、per-class 分佈
  - 每張圖 instance 數量（mean / std / min / max / 分位數）
  - Instance 面積分佈（bbox area、mask area）
  - Instance 長寬比分佈
  - 各 class 出現在多少張圖中（class co-occurrence）
  - 小/中/大物件比例（依 COCO 定義）

Usage:
    python analyze_dataset.py
    python analyze_dataset.py --data_root data --split train
"""

import os
import argparse
import numpy as np
import tifffile
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings("ignore")

# ── 顏色輸出 ─────────────────────────────────────────────────────────────
BOLD  = "\033[1m"
CYAN  = "\033[96m"
GREEN = "\033[92m"
YELLOW= "\033[93m"
RESET = "\033[0m"

def header(text): print(f"\n{BOLD}{CYAN}{'═'*60}{RESET}\n{BOLD}{CYAN}  {text}{RESET}\n{BOLD}{CYAN}{'═'*60}{RESET}")
def section(text): print(f"\n{BOLD}{GREEN}── {text} ──{RESET}")
def info(text):    print(f"  {text}")

# ── 統計輔助 ─────────────────────────────────────────────────────────────
def stats_str(arr, unit=""):
    if len(arr) == 0:
        return "N/A"
    a = np.array(arr)
    return (f"mean={a.mean():.1f}{unit}  std={a.std():.1f}{unit}  "
            f"min={a.min():.0f}{unit}  p25={np.percentile(a,25):.0f}{unit}  "
            f"median={np.median(a):.0f}{unit}  p75={np.percentile(a,75):.0f}{unit}  "
            f"max={a.max():.0f}{unit}")

def hist_bar(counter, total, width=30):
    """橫向長條圖，counter: {label: count}"""
    if not counter:
        return
    max_val = max(counter.values())
    for label, cnt in sorted(counter.items()):
        bar_len = int(cnt / max_val * width) if max_val > 0 else 0
        pct = cnt / total * 100 if total > 0 else 0
        bar = "█" * bar_len
        print(f"    {str(label):>6}  {bar:<{width}}  {cnt:>6}  ({pct:5.1f}%)")

# ── 讀取一個 sample ───────────────────────────────────────────────────────
def parse_sample(sample_dir):
    """
    Returns:
        img_h, img_w  : 圖片高/寬
        instances     : list of dict {class_id, area_mask, bbox_w, bbox_h, area_bbox}
    """
    img_path = os.path.join(sample_dir, "image.tif")
    if not os.path.exists(img_path):
        return None, None, []

    img = tifffile.imread(img_path)
    if img.ndim == 3:
        img_h, img_w = img.shape[:2]
    else:
        img_h, img_w = img.shape

    instances = []
    for cls_id in range(1, 5):
        mask_path = os.path.join(sample_dir, f"class{cls_id}.tif")
        if not os.path.exists(mask_path):
            continue
        mask = tifffile.imread(mask_path)
        # instance IDs（0 = background）
        ids = np.unique(mask)
        ids = ids[ids != 0]
        for inst_id in ids:
            region = (mask == inst_id)
            area_mask = int(region.sum())
            if area_mask < 5:
                continue
            rows = np.any(region, axis=1)
            cols = np.any(region, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            bbox_h = int(rmax - rmin + 1)
            bbox_w = int(cmax - cmin + 1)
            instances.append({
                "class_id":  cls_id,
                "area_mask": area_mask,
                "bbox_h":    bbox_h,
                "bbox_w":    bbox_w,
                "area_bbox": bbox_h * bbox_w,
            })

    return img_h, img_w, instances

# ── 主程式 ────────────────────────────────────────────────────────────────
def analyze(data_root: str, split: str):
    split_dir = os.path.join(data_root, split)
    if not os.path.isdir(split_dir):
        print(f"[ERROR] 找不到 {split_dir}")
        return

    sample_ids = sorted(os.listdir(split_dir))
    sample_ids = [s for s in sample_ids if os.path.isdir(os.path.join(split_dir, s))]

    header(f"Dataset Analysis — {split}  ({len(sample_ids)} images)")

    # ── 收集資料 ──────────────────────────────────────────────────────────
    img_sizes      = []          # (h, w)
    inst_per_img   = []          # total instances per image
    cls_per_img    = defaultdict(int)   # per-class instance count per image
    cls_img_count  = Counter()   # 幾張圖含有 class x
    all_area_mask  = []
    all_area_bbox  = []
    all_aspect     = []          # bbox_w / bbox_h
    per_cls_area   = defaultdict(list)
    per_cls_count  = Counter()   # total instance count per class

    # COCO 小/中/大: area < 32^2, 32^2~96^2, >96^2
    size_bins = {"small (<32²)": 0, "medium (32²–96²)": 0, "large (>96²)": 0}

    for sid in sample_ids:
        sample_dir = os.path.join(split_dir, sid)
        img_h, img_w, instances = parse_sample(sample_dir)
        if img_h is None:
            continue

        img_sizes.append((img_h, img_w))
        inst_per_img.append(len(instances))

        classes_in_img = set()
        for inst in instances:
            c = inst["class_id"]
            per_cls_count[c] += 1
            cls_per_img[c]   += 1          # 累計（across all images）
            all_area_mask.append(inst["area_mask"])
            all_area_bbox.append(inst["area_bbox"])
            per_cls_area[c].append(inst["area_mask"])
            aspect = inst["bbox_w"] / inst["bbox_h"] if inst["bbox_h"] > 0 else 1.0
            all_aspect.append(aspect)
            classes_in_img.add(c)

            a = inst["area_mask"]
            if a < 32**2:
                size_bins["small (<32²)"] += 1
            elif a < 96**2:
                size_bins["medium (32²–96²)"] += 1
            else:
                size_bins["large (>96²)"] += 1

        for c in classes_in_img:
            cls_img_count[c] += 1

    total_inst = sum(inst_per_img)
    n_imgs = len(img_sizes)

    # ── 圖片尺寸 ──────────────────────────────────────────────────────────
    section("圖片尺寸")
    heights = [s[0] for s in img_sizes]
    widths  = [s[1] for s in img_sizes]
    info(f"Height : {stats_str(heights, 'px')}")
    info(f"Width  : {stats_str(widths,  'px')}")
    size_counter = Counter(img_sizes)
    info(f"尺寸種類數: {len(size_counter)}")
    if len(size_counter) <= 10:
        for sz, cnt in size_counter.most_common():
            info(f"  {sz[0]}×{sz[1]} → {cnt} 張")
    else:
        info("Top-5 常見尺寸：")
        for sz, cnt in size_counter.most_common(5):
            info(f"  {sz[0]}×{sz[1]} → {cnt} 張")

    # ── GT 數量 ───────────────────────────────────────────────────────────
    section("GT Instance 總覽")
    info(f"Total instances : {BOLD}{total_inst}{RESET}")
    info(f"Instances/image : {stats_str(inst_per_img)}")

    section("Per-Class Instance 數量")
    for c in sorted(per_cls_count.keys()):
        cnt = per_cls_count[c]
        img_c = cls_img_count[c]
        areas = per_cls_area[c]
        info(f"  Class {c}: {cnt:>6} instances  ({img_c}/{n_imgs} 張圖有此 class)  "
             f"area: mean={np.mean(areas):.0f}  median={np.median(areas):.0f}  max={np.max(areas):.0f}")

    section("每張圖 Instance 數量分佈（直方圖）")
    bins = [0, 1, 5, 10, 20, 50, 100, 200, 500, 9999]
    labels = ["0", "1–4", "5–9", "10–19", "20–49", "50–99", "100–199", "200–499", "500+"]
    bucket = Counter()
    for n in inst_per_img:
        for i in range(len(bins)-1):
            if bins[i] <= n < bins[i+1]:
                bucket[labels[i]] += 1
                break
    hist_bar(dict(zip(labels, [bucket[l] for l in labels])), n_imgs)

    # ── Instance 面積 ─────────────────────────────────────────────────────
    section("Instance Mask 面積（pixel²）")
    info(stats_str(all_area_mask, "px²"))

    section("小/中/大物件比例（COCO 定義，以 mask area）")
    total_inst_counted = sum(size_bins.values())
    for k, v in size_bins.items():
        pct = v / total_inst_counted * 100 if total_inst_counted else 0
        bar = "█" * int(pct / 2)
        info(f"  {k:<22} {v:>7} ({pct:5.1f}%)  {bar}")

    # ── 長寬比 ────────────────────────────────────────────────────────────
    section("Bounding Box 長寬比（w/h）")
    info(stats_str(all_aspect))
    ar_counter = Counter()
    for ar in all_aspect:
        if ar < 0.5:   ar_counter["<0.5 (tall)"] += 1
        elif ar < 0.8: ar_counter["0.5–0.8"] += 1
        elif ar < 1.25:ar_counter["0.8–1.25 (square)"] += 1
        elif ar < 2.0: ar_counter["1.25–2.0"] += 1
        else:          ar_counter[">2.0 (wide)"] += 1
    hist_bar(ar_counter, total_inst)

    # ── Class 共現 ────────────────────────────────────────────────────────
    section("Class 出現的圖片數")
    for c in sorted(cls_img_count.keys()):
        pct = cls_img_count[c] / n_imgs * 100
        bar = "█" * int(pct / 3)
        info(f"  Class {c}: {cls_img_count[c]:>4} / {n_imgs} 張  ({pct:5.1f}%)  {bar}")

    # ── 總結 ──────────────────────────────────────────────────────────────
    section("快速摘要")
    info(f"  圖片數         : {n_imgs}")
    info(f"  GT instances   : {total_inst}")
    info(f"  Avg inst/img   : {np.mean(inst_per_img):.1f}")
    info(f"  Max inst/img   : {max(inst_per_img)}")
    info(f"  Avg mask area  : {np.mean(all_area_mask):.1f} px²")
    info(f"  % small objs   : {size_bins['small (<32²)'] / total_inst * 100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--split", default="train", choices=["train", "test_release"])
    args = parser.parse_args()
    analyze(args.data_root, args.split)
