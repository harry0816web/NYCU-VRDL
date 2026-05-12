"""
dataset.py - Custom PyTorch Dataset for Cell Instance Segmentation

支援兩種載入模式：
    1. 快取模式（預設）：從 preprocess.py 產生的 .pkl 快取載入
       - 圖片以 PNG bytes 儲存，masks 以 RLE 編碼
       - 載入整個快取 < 2 秒，__getitem__ 只做解碼
    2. 即時模式（fallback）：直接讀取原始 TIF 檔案
       - 首次使用或快取不存在時自動 fallback

Usage:
    # 先跑一次前處理（只需一次）
    python preprocess.py --data_root data --output data/train_cache.pkl

    # 之後訓練時自動使用快取
    from dataset import get_train_val_datasets, collate_fn
    train_ds, val_ds = get_train_val_datasets("data", val_ratio=0.15)
"""

import os
import time
import pickle
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

from pycocotools import mask as mask_utils

# 最小 instance 面積
MIN_INSTANCE_AREA = 5


# ═══════════════════════════════════════════════════════════════
# 快取模式 Dataset — 從 .pkl 快取載入
# ═══════════════════════════════════════════════════════════════

class CellDatasetCached(Dataset):
    """
    從預處理快取載入的 Dataset。

    快取中：
        - 圖片 = PNG bytes（需 cv2.imdecode 解碼，~1ms/張）
        - Masks = RLE strings（需 mask_utils.decode 解碼，~0.5ms/個）
        - Bbox/Label = 已解析好的 list

    相比即時模式，省去了：
        1. 5 次 TIF I/O（image + 4 class masks）
        2. float64→int32 型別轉換
        3. np.unique + np.where 遍歷所有 instance ID
    """

    def __init__(
        self,
        cached_samples: list,
        transforms=None,
        enable_dense_tiling: bool = False,
        dense_instance_threshold: int = 256,
        tile_size: int = 512,
        tile_overlap: int = 128,
        max_instances_per_tile: int = 192,
        min_instances_per_tile: int = 1,
        seed: int = 42,
        augment: bool = False,
        copy_paste: bool = False,
        copy_paste_prob: float = 0.5,
        copy_paste_max_instances: int = 30,
    ):
        """
        Args:
            cached_samples: 從 pickle 載入的 sample list，
                            每個元素包含 image_png, instances, height, width
            transforms:     Optional transforms
            augment:        啟用隨機翻轉等 data augmentation
            copy_paste:     啟用 Copy-Paste augmentation
            copy_paste_prob: Copy-Paste 觸發機率
            copy_paste_max_instances: 最多從 donor 圖片複製幾個 instances
        """
        self.samples = cached_samples
        self.transforms = transforms
        self.enable_dense_tiling = enable_dense_tiling
        self.dense_instance_threshold = dense_instance_threshold
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.max_instances_per_tile = max_instances_per_tile
        self.min_instances_per_tile = min_instances_per_tile
        self.rng = np.random.RandomState(seed)
        self.augment = augment
        self.copy_paste = copy_paste
        self.copy_paste_prob = copy_paste_prob
        self.copy_paste_max_instances = copy_paste_max_instances

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        # ── 解碼圖片（PNG bytes → numpy RGB）──
        png_bytes = sample["image_png"]
        buf = np.frombuffer(png_bytes, dtype=np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = sample["height"], sample["width"]
        instances = sample["instances"]

        # 判斷是否需要 tiling：instance 數量超標 或 mask 總記憶體超標
        n_inst = len(instances)
        mask_mem_mb = n_inst * h * w / (1024 * 1024)  # 估計全解碼記憶體
        needs_tiling = (
            self.enable_dense_tiling
            and (n_inst > self.dense_instance_threshold or mask_mem_mb > 200)
        )

        if needs_tiling:
            image, target = self._build_dense_tiled_sample(image, instances, idx)
        else:
            target = self._build_full_target(instances, h, w, idx)

        # ── Copy-Paste augmentation ──
        if self.copy_paste and self.augment and self.rng.rand() < self.copy_paste_prob:
            image, target = self._apply_copy_paste(image, target)

        # ── Data augmentation（翻轉 image + masks + boxes）──
        if self.augment:
            image, target = self._apply_augmentation(image, target)

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        else:
            image = F.to_tensor(image)

        return image, target

    def _apply_copy_paste(self, image, target):
        """Copy-Paste augmentation: 從隨機 donor 圖片複製 instances 貼到當前圖片上。

        參考：Simple Copy-Paste is a Strong Data Augmentation Method for
        Instance Segmentation (Ghiasi et al., CVPR 2021)

        步驟：
        1. 隨機選一張 donor 圖片
        2. 從 donor 隨機選取部分 instances
        3. 將 donor instances 的 mask 區域像素貼到當前圖片上
        4. 更新被遮蓋的原始 instances（移除被完全遮蓋的）
        5. 合併 donor instances 到 target
        """
        h, w = image.shape[:2]

        # 隨機選 donor 圖片（不同於自己）
        donor_idx = self.rng.randint(0, len(self.samples))
        donor_sample = self.samples[donor_idx]

        # 解碼 donor 圖片
        donor_buf = np.frombuffer(donor_sample["image_png"], dtype=np.uint8)
        donor_image = cv2.imdecode(donor_buf, cv2.IMREAD_COLOR)
        donor_image = cv2.cvtColor(donor_image, cv2.COLOR_BGR2RGB)
        dh, dw = donor_sample["height"], donor_sample["width"]

        donor_instances = donor_sample["instances"]
        if len(donor_instances) == 0:
            return image, target

        # 隨機選取 donor instances（最多 copy_paste_max_instances 個）
        n_paste = min(len(donor_instances), self.copy_paste_max_instances)
        n_paste = self.rng.randint(1, n_paste + 1)
        paste_indices = self.rng.choice(
            len(donor_instances), size=n_paste, replace=False
        )

        # 解碼選中的 donor masks 並 resize 到當前圖片尺寸
        paste_masks = []
        paste_boxes = []
        paste_labels = []

        for pi in paste_indices:
            inst = donor_instances[pi]
            mask = mask_utils.decode(inst["rle"])  # (dh, dw)

            # 如果 donor 和當前圖片尺寸不同，resize mask 和 image crop
            if dh != h or dw != w:
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            if mask.sum() < MIN_INSTANCE_AREA:
                continue

            ys, xs = np.where(mask > 0)
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            if x1 <= x0 or y1 <= y0:
                continue

            paste_masks.append(mask.astype(np.uint8))
            paste_boxes.append([x0, y0, x1, y1])
            paste_labels.append(inst["label"])

        if len(paste_masks) == 0:
            return image, target

        # Resize donor image if needed
        if dh != h or dw != w:
            donor_image = cv2.resize(
                donor_image, (w, h), interpolation=cv2.INTER_LINEAR
            )

        # 構建 composite paste mask（所有 paste instances 的聯集）
        composite_mask = np.zeros((h, w), dtype=np.uint8)
        for m in paste_masks:
            composite_mask = np.maximum(composite_mask, m)

        # 將 donor 像素貼到當前圖片上
        paste_region = composite_mask > 0
        image[paste_region] = donor_image[paste_region]

        # 更新原始 instances：移除被嚴重遮蓋的
        orig_masks = target["masks"]  # (N, H, W) tensor
        orig_boxes = target["boxes"]
        orig_labels = target["labels"]

        if len(orig_masks) > 0:
            # 計算每個原始 mask 被遮蓋的比例
            composite_tensor = torch.as_tensor(composite_mask, dtype=torch.uint8)
            orig_area = orig_masks.sum(dim=(1, 2)).float()
            comp = composite_tensor.unsqueeze(0)
            occluded_area = (orig_masks & comp).sum(dim=(1, 2)).float()
            # 保留被遮蓋 < 70% 的 instances
            keep_ratio = 1.0 - occluded_area / (orig_area + 1e-6)
            keep_mask = keep_ratio > 0.3

            if keep_mask.any():
                # 更新保留的原始 masks：扣除被 paste 遮蓋的區域
                surviving_masks = orig_masks[keep_mask]
                neg_comp = ~composite_tensor.unsqueeze(0).bool()
                surviving_masks = surviving_masks & neg_comp
                surviving_masks = surviving_masks.to(torch.uint8)

                # 重新計算 boxes
                surviving_boxes = []
                valid_idx = []
                for i in range(len(surviving_masks)):
                    m = surviving_masks[i]
                    if m.sum() < MIN_INSTANCE_AREA:
                        continue
                    ys_t, xs_t = torch.where(m > 0)
                    surviving_boxes.append([
                        int(xs_t.min()), int(ys_t.min()),
                        int(xs_t.max()), int(ys_t.max()),
                    ])
                    valid_idx.append(i)

                if valid_idx:
                    surviving_masks = surviving_masks[valid_idx]
                    surviving_labels = orig_labels[keep_mask][valid_idx]
                    surviving_boxes = torch.as_tensor(
                        surviving_boxes, dtype=torch.float32
                    )
                else:
                    surviving_masks = torch.zeros((0, h, w), dtype=torch.uint8)
                    surviving_labels = torch.zeros((0,), dtype=torch.int64)
                    surviving_boxes = torch.zeros((0, 4), dtype=torch.float32)
            else:
                surviving_masks = torch.zeros((0, h, w), dtype=torch.uint8)
                surviving_labels = torch.zeros((0,), dtype=torch.int64)
                surviving_boxes = torch.zeros((0, 4), dtype=torch.float32)
        else:
            surviving_masks = orig_masks
            surviving_labels = orig_labels
            surviving_boxes = orig_boxes

        # 合併 surviving + pasted instances
        stacked_paste = np.stack(paste_masks, axis=0)
        paste_masks_tensor = torch.as_tensor(stacked_paste, dtype=torch.uint8)
        paste_boxes_tensor = torch.as_tensor(paste_boxes, dtype=torch.float32)
        paste_labels_tensor = torch.as_tensor(paste_labels, dtype=torch.int64)

        all_masks = torch.cat([surviving_masks, paste_masks_tensor], dim=0)
        all_boxes = torch.cat([surviving_boxes, paste_boxes_tensor], dim=0)
        all_labels = torch.cat([surviving_labels, paste_labels_tensor], dim=0)
        all_area = (all_boxes[:, 3] - all_boxes[:, 1]) * (
            all_boxes[:, 2] - all_boxes[:, 0]
        )

        target = {
            "boxes": all_boxes,
            "labels": all_labels,
            "masks": all_masks,
            "image_id": target["image_id"],
            "area": all_area,
            "iscrowd": torch.zeros(len(all_labels), dtype=torch.int64),
        }
        return image, target

    def _apply_augmentation(self, image, target):
        """Color Jitter + 90° 旋轉 + 隨機水平/垂直翻轉，同步更新 masks 和 boxes。"""
        h, w = image.shape[:2]
        boxes = target["boxes"]     # (N, 4) tensor
        masks = target["masks"]     # (N, H, W) tensor

        # ── 1. 90° 隨機旋轉（僅限正方形圖片，避免 box 軸交換時 H≠W 對不齊）──
        if h == w and self.rng.rand() < 0.5:
            k = int(self.rng.randint(1, 4))   # 1, 2, 或 3 個 90°
            image = np.ascontiguousarray(np.rot90(image, k=k))
            if len(masks) > 0:
                masks = torch.rot90(masks, k=k, dims=(1, 2)).contiguous()
                boxes = self._rotate_boxes_k90(boxes, k, h, w)

        # ── 2. Color Jitter（只動像素值，不影響 mask/box）──
        if self.rng.rand() < 0.5:
            image = self._color_jitter(
                image,
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
            )

        # ── 3. 隨機水平翻轉 ──
        if self.rng.rand() < 0.5:
            image = image[:, ::-1, :].copy()
            if len(masks) > 0:
                masks = masks.flip(dims=[2])
                new_x1 = w - boxes[:, 2]
                new_x2 = w - boxes[:, 0]
                boxes = boxes.clone()
                boxes[:, 0] = new_x1
                boxes[:, 2] = new_x2

        # ── 4. 隨機垂直翻轉 ──
        if self.rng.rand() < 0.5:
            image = image[::-1, :, :].copy()
            if len(masks) > 0:
                masks = masks.flip(dims=[1])
                new_y1 = h - boxes[:, 3]
                new_y2 = h - boxes[:, 1]
                boxes = boxes.clone()
                boxes[:, 1] = new_y1
                boxes[:, 3] = new_y2

        target["boxes"] = boxes
        target["masks"] = masks
        if len(boxes) > 0:
            target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        return image, target

    @staticmethod
    def _rotate_boxes_k90(boxes: torch.Tensor, k: int, h: int, w: int) -> torch.Tensor:
        """旋轉 xyxy boxes k * 90° (假設 h == w)。"""
        if len(boxes) == 0:
            return boxes
        k = k % 4
        if k == 0:
            return boxes.clone()

        x1, y1, x2, y2 = boxes.unbind(dim=1)
        if k == 1:        # 90° CCW (np.rot90 預設方向)
            new_x1 = y1
            new_y1 = w - x2
            new_x2 = y2
            new_y2 = w - x1
        elif k == 2:      # 180°
            new_x1 = w - x2
            new_y1 = h - y2
            new_x2 = w - x1
            new_y2 = h - y1
        else:             # 270° CCW (= 90° CW)
            new_x1 = h - y2
            new_y1 = x1
            new_x2 = h - y1
            new_y2 = x2

        rotated = torch.stack(
            [
                torch.minimum(new_x1, new_x2),
                torch.minimum(new_y1, new_y2),
                torch.maximum(new_x1, new_x2),
                torch.maximum(new_y1, new_y2),
            ],
            dim=1,
        )
        # 旋轉後寬高仍為原 w / h；若旋轉 90/270 軸交換但 h==w，bound 相同
        rotated[:, 0::2] = rotated[:, 0::2].clamp(min=0, max=w)
        rotated[:, 1::2] = rotated[:, 1::2].clamp(min=0, max=h)
        return rotated

    def _color_jitter(
        self,
        image: np.ndarray,
        brightness: float,
        contrast: float,
        saturation: float,
        hue: float,
    ) -> np.ndarray:
        """以隨機順序套用 brightness/contrast/saturation/hue。

        輸入/輸出都是 uint8 RGB numpy。
        """
        tensor = F.to_tensor(image)  # (3, H, W) float in [0, 1]

        ops = [0, 1, 2, 3]
        self.rng.shuffle(ops)
        for op in ops:
            if op == 0 and brightness > 0:
                factor = float(self.rng.uniform(max(0, 1 - brightness), 1 + brightness))
                tensor = F.adjust_brightness(tensor, factor)
            elif op == 1 and contrast > 0:
                factor = float(self.rng.uniform(max(0, 1 - contrast), 1 + contrast))
                tensor = F.adjust_contrast(tensor, factor)
            elif op == 2 and saturation > 0:
                factor = float(self.rng.uniform(max(0, 1 - saturation), 1 + saturation))
                tensor = F.adjust_saturation(tensor, factor)
            elif op == 3 and hue > 0:
                factor = float(self.rng.uniform(-hue, hue))
                tensor = F.adjust_hue(tensor, factor)

        tensor = tensor.clamp(0.0, 1.0)
        return (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)

    def _build_full_target(self, instances, h, w, idx):
        masks = []
        boxes = []
        labels = []

        for inst in instances:
            binary_mask = mask_utils.decode(inst["rle"])
            masks.append(binary_mask)
            boxes.append(inst["bbox"])
            labels.append(inst["label"])

        if len(masks) > 0:
            masks_tensor = torch.as_tensor(np.stack(masks, axis=0), dtype=torch.uint8)
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
            area = (boxes_tensor[:, 3] - boxes_tensor[:, 1]) * (
                boxes_tensor[:, 2] - boxes_tensor[:, 0]
            )
        else:
            masks_tensor = torch.zeros((0, h, w), dtype=torch.uint8)
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)

        return {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "masks": masks_tensor,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": torch.zeros(len(labels), dtype=torch.int64),
        }

    def _generate_tile_windows(self, h, w):
        tile = max(32, int(self.tile_size))
        stride = max(1, tile - int(self.tile_overlap))

        y_starts = list(range(0, max(1, h - tile + 1), stride))
        x_starts = list(range(0, max(1, w - tile + 1), stride))
        if not y_starts or y_starts[-1] != max(0, h - tile):
            y_starts.append(max(0, h - tile))
        if not x_starts or x_starts[-1] != max(0, w - tile):
            x_starts.append(max(0, w - tile))

        windows = []
        for y0 in y_starts:
            for x0 in x_starts:
                x1 = min(w, x0 + tile)
                y1 = min(h, y0 + tile)
                windows.append((x0, y0, x1, y1))
        return windows

    @staticmethod
    def _bbox_intersects_window(bbox, window):
        bx0, by0, bx1, by1 = bbox
        wx0, wy0, wx1, wy1 = window
        return (bx0 < wx1) and (bx1 > wx0) and (by0 < wy1) and (by1 > wy0)

    def _build_dense_tiled_sample(self, image, instances, idx):
        h, w = image.shape[:2]
        windows = self._generate_tile_windows(h, w)

        if not windows:
            return image, self._build_full_target(instances, h, w, idx)

        window_to_instance_ids = []
        for window in windows:
            candidate_ids = []
            for inst_id, inst in enumerate(instances):
                if self._bbox_intersects_window(inst["bbox"], window):
                    candidate_ids.append(inst_id)
            window_to_instance_ids.append(candidate_ids)

        valid_windows = [
            i for i, ids in enumerate(window_to_instance_ids)
            if len(ids) >= self.min_instances_per_tile
        ]
        if not valid_windows:
            return image, self._build_full_target(instances, h, w, idx)

        chosen_window_idx = valid_windows[self.rng.randint(0, len(valid_windows))]
        x0, y0, x1, y1 = windows[chosen_window_idx]
        selected_ids = window_to_instance_ids[chosen_window_idx]

        if len(selected_ids) > self.max_instances_per_tile:
            selected_ids = self.rng.choice(
                selected_ids,
                size=self.max_instances_per_tile,
                replace=False,
            ).tolist()

        tile_image = image[y0:y1, x0:x1, :]
        tile_h, tile_w = tile_image.shape[:2]

        tile_masks = []
        tile_boxes = []
        tile_labels = []

        for inst_id in selected_ids:
            inst = instances[inst_id]
            binary_mask = mask_utils.decode(inst["rle"])
            tile_mask = binary_mask[y0:y1, x0:x1]
            if tile_mask.sum() < MIN_INSTANCE_AREA:
                continue

            ys, xs = np.where(tile_mask > 0)
            if len(xs) == 0 or len(ys) == 0:
                continue
            local_x0, local_x1 = int(xs.min()), int(xs.max())
            local_y0, local_y1 = int(ys.min()), int(ys.max())
            if local_x1 <= local_x0 or local_y1 <= local_y0:
                continue

            tile_masks.append(tile_mask.astype(np.uint8))
            tile_boxes.append([local_x0, local_y0, local_x1, local_y1])
            tile_labels.append(inst["label"])

        if len(tile_masks) == 0:
            return image, self._build_full_target(instances, h, w, idx)

        masks_tensor = torch.as_tensor(np.stack(tile_masks, axis=0), dtype=torch.uint8)
        boxes_tensor = torch.as_tensor(tile_boxes, dtype=torch.float32)
        labels_tensor = torch.as_tensor(tile_labels, dtype=torch.int64)
        area = (boxes_tensor[:, 3] - boxes_tensor[:, 1]) * (
            boxes_tensor[:, 2] - boxes_tensor[:, 0]
        )
        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "masks": masks_tensor,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": torch.zeros(len(tile_labels), dtype=torch.int64),
        }
        return tile_image, target


# ═══════════════════════════════════════════════════════════════
# 即時模式 Dataset — 直接讀 TIF（fallback）
# ═══════════════════════════════════════════════════════════════

class CellDataset(Dataset):
    """直接從 TIF 檔案載入的 Dataset（無快取時的 fallback）。"""

    def __init__(self, data_dir: str, sample_ids: list, transforms=None):
        self.data_dir = data_dir
        self.sample_ids = sample_ids
        self.transforms = transforms
        self.class_names = ["class1", "class2", "class3", "class4"]

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int):
        sample_id = self.sample_ids[idx]
        sample_dir = os.path.join(self.data_dir, sample_id)

        # Load image
        img_path = os.path.join(sample_dir, "image.tif")
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        h, w = image.shape[:2]

        masks = []
        labels = []
        boxes = []

        for class_idx, class_name in enumerate(self.class_names, start=1):
            mask_path = os.path.join(sample_dir, f"{class_name}.tif")
            if not os.path.exists(mask_path):
                continue

            mask_data = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if mask_data is None:
                continue

            if mask_data.dtype in (np.float64, np.float32):
                mask_data = mask_data.astype(np.int32)
            if mask_data.ndim == 3:
                mask_data = mask_data[:, :, 0]

            instance_ids = np.unique(mask_data)
            instance_ids = instance_ids[instance_ids > 0]

            for inst_id in instance_ids:
                ys, xs = np.where(mask_data == inst_id)
                if len(xs) < MIN_INSTANCE_AREA:
                    continue

                x_min, x_max = int(xs.min()), int(xs.max())
                y_min, y_max = int(ys.min()), int(ys.max())
                if x_max <= x_min or y_max <= y_min:
                    continue

                binary_mask = (mask_data == inst_id).astype(np.uint8)
                boxes.append([x_min, y_min, x_max, y_max])
                masks.append(binary_mask)
                labels.append(class_idx)

        if len(masks) > 0:
            masks_tensor = torch.as_tensor(
                np.stack(masks, axis=0), dtype=torch.uint8
            )
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
            area = (boxes_tensor[:, 3] - boxes_tensor[:, 1]) * \
                   (boxes_tensor[:, 2] - boxes_tensor[:, 0])
        else:
            masks_tensor = torch.zeros((0, h, w), dtype=torch.uint8)
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "masks": masks_tensor,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": torch.zeros(len(labels), dtype=torch.int64),
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        else:
            image = F.to_tensor(image)

        return image, target


# ═══════════════════════════════════════════════════════════════
# Test Dataset
# ═══════════════════════════════════════════════════════════════

class CellTestDataset(Dataset):
    """Dataset for test images (no masks)."""

    def __init__(self, test_dir: str, image_info: list):
        self.test_dir = test_dir
        self.image_info = image_info

    def __len__(self) -> int:
        return len(self.image_info)

    def __getitem__(self, idx: int):
        info = self.image_info[idx]
        img_path = os.path.join(self.test_dir, info["file_name"])

        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        image = F.to_tensor(image)
        return image, info


# ═══════════════════════════════════════════════════════════════
# Dataset Factory — 自動選擇快取 / 即時模式
# ═══════════════════════════════════════════════════════════════

def get_train_val_datasets(
    data_root: str = "data",
    val_ratio: float = 0.15,
    seed: int = 42,
    cache_path: str = None,
    enable_dense_tiling: bool = False,
    dense_instance_threshold: int = 256,
    tile_size: int = 512,
    tile_overlap: int = 128,
    max_instances_per_tile: int = 192,
    min_instances_per_tile: int = 1,
):
    """
    建立 train/val datasets，自動偵測是否有快取檔案。

    Args:
        data_root:  資料根目錄
        val_ratio:  驗證集比例
        seed:       隨機種子
        cache_path: 快取檔案路徑（預設自動偵測 data/train_cache.pkl）

    Returns:
        train_dataset, val_dataset
    """
    if cache_path is None:
        cache_path = os.path.join(data_root, "train_cache.pkl")

    # ── 嘗試載入快取 ──
    if os.path.exists(cache_path):
        print(f"Loading cached dataset from {cache_path}...")
        t0 = time.time()
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        all_samples = cache["samples"]
        t1 = time.time()
        print(f"  Loaded {len(all_samples)} samples in {t1-t0:.1f}s")

        # 用 sample_id 排序以確保 split 一致
        all_samples.sort(key=lambda x: x["sample_id"])

        # Train/Val split
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(all_samples))
        val_size = int(len(all_samples) * val_ratio)

        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        train_samples = [all_samples[i] for i in train_indices]
        val_samples = [all_samples[i] for i in val_indices]

        print(f"  Train: {len(train_samples)}, Val: {len(val_samples)}")

        total_train_inst = sum(len(s["instances"]) for s in train_samples)
        total_val_inst = sum(len(s["instances"]) for s in val_samples)
        print(f"  Train instances: {total_train_inst}, "
              f"Val instances: {total_val_inst}")

        train_dataset = CellDatasetCached(
            train_samples,
            enable_dense_tiling=enable_dense_tiling,
            dense_instance_threshold=dense_instance_threshold,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            max_instances_per_tile=max_instances_per_tile,
            min_instances_per_tile=min_instances_per_tile,
            seed=seed,
            augment=True,   # 訓練集啟用 augmentation
            copy_paste=True,
            copy_paste_prob=0.5,
            copy_paste_max_instances=30,
        )
        val_dataset = CellDatasetCached(
            val_samples,
            enable_dense_tiling=False,
            dense_instance_threshold=dense_instance_threshold,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            max_instances_per_tile=max_instances_per_tile,
            min_instances_per_tile=min_instances_per_tile,
            seed=seed,
        )

        return train_dataset, val_dataset

    # ── Fallback: 即時讀取 TIF ──
    print(f"Cache not found at {cache_path}, loading TIF files directly.")
    print(f"  (Run 'python preprocess.py' to create cache for faster loading)")

    train_dir = os.path.join(data_root, "train")
    all_sample_ids = sorted(os.listdir(train_dir))

    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(all_sample_ids))
    val_size = int(len(all_sample_ids) * val_ratio)

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_ids = [all_sample_ids[i] for i in train_indices]
    val_ids = [all_sample_ids[i] for i in val_indices]

    print(f"  Train: {len(train_ids)}, Val: {len(val_ids)}")

    train_dataset = CellDataset(train_dir, train_ids)
    val_dataset = CellDataset(train_dir, val_ids)

    return train_dataset, val_dataset


def collate_fn(batch):
    """Custom collate function for variable-size images."""
    return tuple(zip(*batch))


if __name__ == "__main__":
    train_ds, val_ds = get_train_val_datasets("data")
    print(f"\nTrain: {len(train_ds)}, Val: {len(val_ds)}")

    t0 = time.time()
    img, target = train_ds[0]
    t1 = time.time()
    print(f"First sample load time: {(t1-t0)*1000:.1f} ms")
    print(f"Image shape: {img.shape}")
    print(f"Num instances: {target['masks'].shape[0]}")
    print(f"Boxes shape: {target['boxes'].shape}")
