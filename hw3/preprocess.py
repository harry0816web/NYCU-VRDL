"""
preprocess.py — 將原始 TIF 資料集預處理並快取為 .pkl 檔案

為什麼需要這個前處理？
═══════════════════════

1. TIF 解碼慢且有警告
   原始 mask 是 float64 TIFF 格式，OpenCV 每次讀取都會觸發
   「TIFFReadDirectory: Sum of Photometric type-related color channels
   and ExtraSamples doesn't match SamplesPerPixel」的警告。
   每張圖要讀 1 張 image.tif + 最多 4 張 classN.tif = 5 次磁碟 I/O。
   209 張圖 × 5 ≈ 1,045 次 TIF 解碼，光是啟動就要耗費 60–90 秒。

2. Instance 解析是 CPU 密集運算
   每張 mask 需要 np.unique() 找出所有 instance ID，
   再對每個 ID 做 np.where() 取得像素座標、計算 bbox。
   Class 1/2 平均有 100~150 個 instance，最多到 734 個，
   單張圖可能要跑幾百次 np.where()。

3. 重複工作浪費訓練時間
   每個 epoch 的每個 __getitem__ 都會重新執行上述流程。
   30 epochs × 209 張 = 6,270 次完整 TIF 解析，
   其中 mask 解析的結果每次都完全一樣（資料集是固定的）。

快取策略
════════
- 圖片：壓縮為 PNG bytes（無損，比原始 numpy array 小 5~10 倍）
- Mask：編碼為 pycocotools RLE 格式（極度緊湊，734 個 mask
  從 ~3GB numpy → 幾 MB RLE strings）
- Bbox/Label：直接存 numpy array
- 全部打包成一個 .pkl 檔，之後載入只需 < 2 秒

Usage:
    python preprocess.py --data_root data --output data/train_cache.pkl
"""

import os
import gc
import time
import pickle
import argparse
import numpy as np
import cv2
from pycocotools import mask as mask_utils

# 最小 instance 面積（像素數），低於此值視為雜訊
MIN_INSTANCE_AREA = 5


def encode_image_to_png(image_rgb: np.ndarray) -> bytes:
    """將 RGB numpy array 壓縮為 PNG bytes（無損壓縮，節省記憶體）。"""
    # cv2.imencode 需要 BGR
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    success, buf = cv2.imencode(".png", image_bgr)
    assert success, "PNG encoding failed"
    return buf.tobytes()


def load_and_convert_image(img_path: str) -> np.ndarray:
    """讀取 TIF 圖片並轉為 RGB uint8。"""
    image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Cannot read: {img_path}")

    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    return image


def parse_mask_tif(mask_path: str, class_idx: int):
    """
    解析單張 class mask TIF，回傳該 class 的所有 instance 資訊。

    Returns:
        list of dict, each containing:
            - 'rle': RLE encoded binary mask (compact string format)
            - 'bbox': [x_min, y_min, x_max, y_max]
            - 'label': class_idx (1-4)
            - 'area': pixel count
    """
    mask_data = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask_data is None:
        return []

    # float64 → int32
    if mask_data.dtype in (np.float64, np.float32):
        mask_data = mask_data.astype(np.int32)

    if mask_data.ndim == 3:
        mask_data = mask_data[:, :, 0]

    instance_ids = np.unique(mask_data)
    instance_ids = instance_ids[instance_ids > 0]

    instances = []
    for inst_id in instance_ids:
        ys, xs = np.where(mask_data == inst_id)
        area = len(xs)

        if area < MIN_INSTANCE_AREA:
            continue

        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())

        if x_max <= x_min or y_max <= y_min:
            continue

        # 編碼為 RLE — 比存 binary mask 小幾百倍
        binary_mask = (mask_data == inst_id).astype(np.uint8)
        rle = mask_utils.encode(np.asfortranarray(binary_mask))
        rle["counts"] = rle["counts"].decode("utf-8")

        instances.append({
            "rle": rle,
            "bbox": [x_min, y_min, x_max, y_max],
            "label": class_idx,
            "area": area,
        })

    return instances


def preprocess_dataset(data_root: str, output_path: str):
    """
    前處理整個訓練集，將結果存為 pickle 快取。

    Cache 結構:
        {
            "samples": [
                {
                    "sample_id": str,
                    "image_png": bytes,         # PNG 壓縮的圖片
                    "height": int,
                    "width": int,
                    "instances": [
                        {
                            "rle": {"size": [H,W], "counts": str},
                            "bbox": [x1, y1, x2, y2],
                            "label": int,       # 1-4
                            "area": int,
                        },
                        ...
                    ]
                },
                ...
            ]
        }
    """
    train_dir = os.path.join(data_root, "train")
    sample_ids = sorted(os.listdir(train_dir))
    class_names = ["class1", "class2", "class3", "class4"]

    print(f"Preprocessing {len(sample_ids)} samples from {train_dir}")
    print(f"Min instance area: {MIN_INSTANCE_AREA} pixels")

    samples = []
    total_instances = 0
    t_start = time.time()

    for i, sample_id in enumerate(sample_ids):
        sample_dir = os.path.join(train_dir, sample_id)

        # 1. 讀取並壓縮圖片
        image = load_and_convert_image(
            os.path.join(sample_dir, "image.tif")
        )
        h, w = image.shape[:2]
        image_png = encode_image_to_png(image)
        del image  # 釋放原始 array

        # 2. 解析每個 class 的 mask
        all_instances = []
        for class_idx, class_name in enumerate(class_names, start=1):
            mask_path = os.path.join(sample_dir, f"{class_name}.tif")
            if os.path.exists(mask_path):
                instances = parse_mask_tif(mask_path, class_idx)
                all_instances.extend(instances)

        total_instances += len(all_instances)

        samples.append({
            "sample_id": sample_id,
            "image_png": image_png,
            "height": h,
            "width": w,
            "instances": all_instances,
        })

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t_start
            print(f"  [{i+1}/{len(sample_ids)}] "
                  f"{total_instances} instances so far, "
                  f"{elapsed:.1f}s elapsed")

        # 每 50 張做一次 GC
        if (i + 1) % 50 == 0:
            gc.collect()

    elapsed = time.time() - t_start

    cache = {"samples": samples}

    # 寫入檔案
    print(f"\nSaving cache to {output_path}...")
    with open(output_path, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    print(f"\n{'='*50}")
    print(f"Preprocessing complete!")
    print(f"  Samples:   {len(samples)}")
    print(f"  Instances: {total_instances}")
    print(f"  Cache size: {file_size_mb:.1f} MB")
    print(f"  Time:      {elapsed:.1f}s")
    print(f"  Saved to:  {output_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess TIF dataset into cached pickle"
    )
    parser.add_argument("--data_root", type=str, default="data",
                        help="Root data directory containing train/")
    parser.add_argument("--output", type=str, default="data/train_cache.pkl",
                        help="Output cache file path")
    args = parser.parse_args()

    preprocess_dataset(args.data_root, args.output)
