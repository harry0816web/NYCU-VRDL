"""
inference.py - Generate submission file for CodaBench competition

V6.1 (Route A) — 預設行為對齊 V2 + 訓練時的 image scale：
    - 預設 **不切 tile**（單張 full-image inference），對齊 train.py 的 val 流程。
      這樣 val AP 才能準確預測 test 行為。
    - 模型建構時傳入 `min_size = MODEL_CONFIG["multi_scale_min_sizes"]`、
      `max_size = MODEL_CONFIG["multi_scale_max_size"]`，讓 torchvision 在 eval
      模式下用 `min_size[-1]`（=800）resize，與訓練時 val 完全一致。
    - 仍保留 V6 引入的「逐個 instance 處理 mask」(避免 WSL OOM)，且 tiled
      sliding-window + box-IoU Soft-NMS 的程式碼仍然在，但只在 `--tiling` 時啟用。

Output format per prediction:
    {
        "image_id": int,
        "category_id": int,
        "bbox": [x, y, w, h],
        "score": float,
        "segmentation": {"size": [H, W], "counts": str}
    }

Usage:
    # 預設（推薦，等同 V2 inference 行為）：
    python inference.py --checkpoint checkpoints/<run>/best_model.pth

    # Ablation：開啟 tiled inference + box-IoU Soft-NMS
    python inference.py --tiling --checkpoint checkpoints/<run>/best_model.pth
"""

import os
import json
import argparse
from collections import defaultdict, Counter

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    # When running as a script from within `detection/`
    from model import get_baseline_model
    from dataset import CellTestDataset, collate_fn
    from config import MODEL_CONFIG
except ModuleNotFoundError:
    # When running from repo root: `python detection/inference.py ...`
    from detection.model import get_baseline_model
    from detection.dataset import CellTestDataset, collate_fn
    from detection.config import MODEL_CONFIG

from pycocotools import mask as mask_utils


# ════════════════════════════════════════════════════════════════════
# Tile generation
# ════════════════════════════════════════════════════════════════════

def generate_tile_windows(h: int, w: int, tile: int, overlap: int):
    """產生 (x0, y0, x1, y1) sliding windows，最後一格會貼齊邊界。

    若圖比 tile 還小，就回傳整張圖一個 window。
    """
    tile = max(32, int(tile))
    if h <= tile and w <= tile:
        return [(0, 0, w, h)]

    stride = max(1, tile - int(overlap))

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


# ════════════════════════════════════════════════════════════════════
# Box-IoU Soft-NMS (Gaussian)
# ════════════════════════════════════════════════════════════════════

def _box_iou_one_to_many(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    """Pairwise IoU between a single xyxy box and (M, 4) xyxy boxes."""
    if len(others) == 0:
        return np.zeros((0,), dtype=np.float32)

    x1 = np.maximum(box[0], others[:, 0])
    y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2])
    y2 = np.minimum(box[3], others[:, 3])

    inter_w = np.clip(x2 - x1, a_min=0.0, a_max=None)
    inter_h = np.clip(y2 - y1, a_min=0.0, a_max=None)
    inter = inter_w * inter_h

    area_box = max(0.0, (box[2] - box[0])) * max(0.0, (box[3] - box[1]))
    area_others = (
        np.clip(others[:, 2] - others[:, 0], a_min=0.0, a_max=None)
        * np.clip(others[:, 3] - others[:, 1], a_min=0.0, a_max=None)
    )

    union = area_box + area_others - inter
    iou = np.where(union > 0, inter / union, 0.0)
    return iou.astype(np.float32)


def soft_nms_box(
    boxes: np.ndarray,
    scores: np.ndarray,
    sigma: float = 0.5,
    score_thresh: float = 0.05,
) -> np.ndarray:
    """Soft-NMS (Gaussian) on box-IoU.

    Args:
        boxes: (N, 4) xyxy float32
        scores: (N,) float32 (caller's array is not modified in-place)
        sigma: Gaussian decay sigma
        score_thresh: 最後 score 過濾門檻

    Returns:
        kept_indices: ndarray of original indices to keep（按抽出順序排列）
    """
    n = len(scores)
    if n == 0:
        return np.zeros((0,), dtype=np.int64)

    scores = scores.astype(np.float32, copy=True)
    remaining = np.arange(n, dtype=np.int64)
    kept = []

    while remaining.size > 0:
        local_max = int(np.argmax(scores[remaining]))
        chosen = int(remaining[local_max])

        # 篩選低分前先加入，最後再過濾
        kept.append(chosen)

        # 從 remaining 移除目前選中的
        remaining = np.delete(remaining, local_max)
        if remaining.size == 0:
            break

        ious = _box_iou_one_to_many(boxes[chosen], boxes[remaining])
        decay = np.exp(-(ious ** 2) / max(1e-6, sigma))
        scores[remaining] = scores[remaining] * decay

        # 提前剔除分數已掉到 thresh 以下的，加快收斂
        alive_mask = scores[remaining] >= score_thresh
        remaining = remaining[alive_mask]

    kept_arr = np.asarray(kept, dtype=np.int64)
    final_mask = scores[kept_arr] >= score_thresh
    return kept_arr[final_mask]


# ════════════════════════════════════════════════════════════════════
# Inference
# ════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _infer_on_window(
    model,
    full_image: torch.Tensor,
    window: tuple,
) -> dict:
    """Run model on a single (x0,y0,x1,y1) crop. Returns CPU-friendly tensors."""
    x0, y0, x1, y1 = window
    crop = full_image[:, y0:y1, x0:x1].contiguous()
    output = model([crop])[0]
    return output


@torch.no_grad()
def _infer_single_image(
    model,
    image: torch.Tensor,
    img_h: int,
    img_w: int,
    score_threshold: float,
    use_tiling: bool,
    tile_size: int,
    tile_overlap: int,
):
    """對單張圖片（可能是翻轉後的）推論，回傳 per_image predictions list。

    每個 prediction = {bbox_xyxy, score, label, binary_mask (H,W numpy)}
    注意：回傳的是 binary_mask 而非 RLE，方便 TTA 翻轉回原圖座標後再編碼。
    """
    if use_tiling:
        windows = generate_tile_windows(img_h, img_w, tile_size, tile_overlap)
    else:
        windows = [(0, 0, img_w, img_h)]

    per_image = []

    for window in windows:
        output = _infer_on_window(model, image, window)
        wx0, wy0, wx1, wy1 = window

        boxes = output["boxes"].cpu().numpy()
        scores = output["scores"].cpu().numpy()
        labels = output["labels"].cpu().numpy()
        masks_tensor = output["masks"]

        for j in range(len(scores)):
            s = float(scores[j])
            if s < score_threshold:
                continue

            binary = (
                (masks_tensor[j, 0] > 0.5).cpu().numpy().astype(np.uint8)
            )

            if binary.shape != (wy1 - wy0, wx1 - wx0):
                import cv2
                binary = cv2.resize(
                    binary,
                    (wx1 - wx0, wy1 - wy0),
                    interpolation=cv2.INTER_NEAREST,
                )

            full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            full_mask[wy0:wy1, wx0:wx1] = binary

            x1, y1, x2, y2 = boxes[j]
            gx1 = max(0.0, min(img_w, float(x1) + wx0))
            gy1 = max(0.0, min(img_h, float(y1) + wy0))
            gx2 = max(0.0, min(img_w, float(x2) + wx0))
            gy2 = max(0.0, min(img_h, float(y2) + wy0))
            if gx2 <= gx1 or gy2 <= gy1:
                del binary, full_mask
                continue

            per_image.append({
                "bbox_xyxy": (gx1, gy1, gx2, gy2),
                "score": s,
                "label": int(labels[j]),
                "binary_mask": full_mask,
            })

            del binary

        del output, masks_tensor, boxes, scores, labels
        torch.cuda.empty_cache()

    return per_image


def _flip_predictions_h(preds: list, img_w: int) -> list:
    """將水平翻轉的預測結果映射回原圖座標。"""
    flipped = []
    for p in preds:
        gx1, gy1, gx2, gy2 = p["bbox_xyxy"]
        new_gx1 = img_w - gx2
        new_gx2 = img_w - gx1
        mask = p["binary_mask"][:, ::-1].copy()
        flipped.append({
            "bbox_xyxy": (new_gx1, gy1, new_gx2, gy2),
            "score": p["score"],
            "label": p["label"],
            "binary_mask": mask,
        })
    return flipped


def _flip_predictions_v(preds: list, img_h: int) -> list:
    """將垂直翻轉的預測結果映射回原圖座標。"""
    flipped = []
    for p in preds:
        gx1, gy1, gx2, gy2 = p["bbox_xyxy"]
        new_gy1 = img_h - gy2
        new_gy2 = img_h - gy1
        mask = p["binary_mask"][::-1, :].copy()
        flipped.append({
            "bbox_xyxy": (gx1, new_gy1, gx2, new_gy2),
            "score": p["score"],
            "label": p["label"],
            "binary_mask": mask,
        })
    return flipped


def _pad_to_square(image: torch.Tensor, max_side: int = 1024):
    """Pad image (C,H,W) to square with zeros on right/bottom.

    If max(H,W) > max_side, downscale proportionally first to prevent OOM
    in paste_masks_in_image (which allocates masks at the input resolution).

    Returns (padded_image, square_size, resized_h, resized_w, inv_scale).
        - resized_h, resized_w: image dims after optional resize, before padding
        - inv_scale: multiply coordinates by this to map back to original image
    """
    _, h, w = image.shape
    inv_scale = 1.0

    if max_side > 0 and max(h, w) > max_side:
        scale = max_side / max(h, w)
        inv_scale = 1.0 / scale
        new_h = round(h * scale)
        new_w = round(w * scale)
        image = torch.nn.functional.interpolate(
            image.unsqueeze(0), size=(new_h, new_w),
            mode='bilinear', align_corners=False,
        ).squeeze(0)
        h, w = new_h, new_w

    s = max(h, w)
    if h == s and w == s:
        return image, s, h, w, inv_scale

    padded = torch.zeros(3, s, s, dtype=image.dtype, device=image.device)
    padded[:, :h, :w] = image
    return padded, s, h, w, inv_scale


def _rot90_predictions(preds: list, k: int, square_size: int,
                       resized_h: int, resized_w: int,
                       img_h: int, img_w: int,
                       inv_scale: float = 1.0) -> list:
    """Map predictions from k*90° CCW rotated padded-square back to original coords.

    Steps:
        1. Inverse-rotate mask and bbox from rotated (S×S) space.
        2. Crop mask to (resized_h, resized_w) region (discard padding).
        3. If image was downscaled (inv_scale != 1), resize mask and scale bbox
           back to the original (img_h, img_w) resolution.
        4. Filter out predictions outside the original image.

    Coordinate derivation (for square size S):
        k=1 (90° CCW): pixel (x,y) → (y, S-1-x) in rotated space.
            Inverse: (rx, ry) → orig_x = S-1-ry, orig_y = rx.
        k=3 (270° CCW): pixel (x,y) → (S-1-y, x) in rotated space.
            Inverse: (rx, ry) → orig_x = ry, orig_y = S-1-rx.
    """
    inv_k = (4 - k) % 4
    S = float(square_size)
    needs_rescale = abs(inv_scale - 1.0) > 1e-6
    result = []

    for p in preds:
        rx1, ry1, rx2, ry2 = p["bbox_xyxy"]
        mask_rot = p["binary_mask"]  # (square_size, square_size)

        # Inverse rotate mask back to padded original space
        mask_orig = np.rot90(mask_rot, k=inv_k)  # (S, S)

        # Inverse rotate bbox (still in resized-padded space)
        if k == 1:
            ox1, oy1 = S - 1 - ry2, rx1
            ox2, oy2 = S - 1 - ry1, rx2
        elif k == 3:
            ox1, oy1 = ry1, S - 1 - rx2
            ox2, oy2 = ry2, S - 1 - rx1
        else:
            ox1, oy1, ox2, oy2 = rx1, ry1, rx2, ry2

        # Clip to resized image area (before padding)
        ox1 = max(0.0, min(float(resized_w), float(ox1)))
        oy1 = max(0.0, min(float(resized_h), float(oy1)))
        ox2 = max(0.0, min(float(resized_w), float(ox2)))
        oy2 = max(0.0, min(float(resized_h), float(oy2)))

        if ox2 <= ox1 or oy2 <= oy1:
            continue

        # Crop mask to resized image area (discard padding)
        mask_crop = mask_orig[:resized_h, :resized_w]
        if mask_crop.sum() == 0:
            continue

        # Scale back to original image resolution if downscaled
        if needs_rescale:
            import cv2
            mask_crop = cv2.resize(
                mask_crop.copy().astype(np.uint8),
                (img_w, img_h),
                interpolation=cv2.INTER_NEAREST,
            )
            ox1 = max(0.0, min(float(img_w), ox1 * inv_scale))
            oy1 = max(0.0, min(float(img_h), oy1 * inv_scale))
            ox2 = max(0.0, min(float(img_w), ox2 * inv_scale))
            oy2 = max(0.0, min(float(img_h), oy2 * inv_scale))
            if ox2 <= ox1 or oy2 <= oy1:
                continue
        else:
            mask_crop = mask_crop.copy()

        result.append({
            "bbox_xyxy": (ox1, oy1, ox2, oy2),
            "score": p["score"],
            "label": p["label"],
            "binary_mask": mask_crop,
        })

    return result


def _fuse_tta_mask_iou(
    base_preds: list,
    aug_preds_list: list,
    img_h: int,
    img_w: int,
    mask_iou_thresh: float = 0.3,
    add_unmatched_thresh: float = 0.7,
) -> list:
    """Fuse TTA predictions using mask IoU matching.

    Algorithm:
        1. Use original-image predictions as the "base" set.
        2. For each augmented view, match its predictions to base predictions
           using per-class mask IoU (greedy, highest IoU first).
        3. Matched: accumulate score → final_score = sum(scores) / total_views.
           This naturally boosts consistent detections and suppresses sporadic FPs.
        4. Unmatched augmented predictions with score >= add_unmatched_thresh
           are added as new instances (catches objects missed by original view).
        5. Final mask: always use the base prediction's mask (highest quality,
           no inverse-transform artifacts).

    Args:
        base_preds: predictions from original image
        aug_preds_list: list of prediction lists from augmented views
                        (already mapped back to original coordinates)
        img_h, img_w: image dimensions
        mask_iou_thresh: minimum mask IoU to consider a match
        add_unmatched_thresh: score threshold for adding unmatched augmented preds

    Returns:
        fused_preds: list of prediction dicts with fused scores
    """
    total_views = 1 + len(aug_preds_list)  # original + augmented views

    if len(base_preds) == 0:
        # No base predictions — collect high-confidence unmatched from augmented
        fused = []
        for aug_preds in aug_preds_list:
            for p in aug_preds:
                if p["score"] >= add_unmatched_thresh:
                    fused.append({
                        "bbox_xyxy": p["bbox_xyxy"],
                        "score": p["score"] / total_views,
                        "label": p["label"],
                        "binary_mask": p["binary_mask"],
                    })
        return fused

    # ── Encode base masks to RLE for fast IoU computation ──
    base_rles = []
    for p in base_preds:
        rle = mask_utils.encode(np.asfortranarray(p["binary_mask"]))
        base_rles.append(rle)

    # Track score accumulation: score_sums[i] = sum of matched scores
    score_sums = np.array([p["score"] for p in base_preds], dtype=np.float64)
    match_counts = np.ones(len(base_preds), dtype=np.int32)  # count original as 1

    # Group base predictions by class for efficient matching
    base_by_class = defaultdict(list)  # class -> list of base indices
    for idx, p in enumerate(base_preds):
        base_by_class[p["label"]].append(idx)

    # ── Match each augmented view to base ──
    all_unmatched = []

    for aug_preds in aug_preds_list:
        # Group augmented preds by class
        aug_by_class = defaultdict(list)
        for aug_idx, p in enumerate(aug_preds):
            aug_by_class[p["label"]].append(aug_idx)

        matched_base_set = set()  # base indices already matched in this view
        matched_aug_set = set()   # aug indices already matched

        for cls_id in aug_by_class:
            if cls_id not in base_by_class:
                continue

            base_indices = base_by_class[cls_id]
            aug_indices = aug_by_class[cls_id]

            if len(base_indices) == 0 or len(aug_indices) == 0:
                continue

            # Compute mask IoU matrix: (len(aug_indices), len(base_indices))
            aug_rles = []
            for ai in aug_indices:
                rle = mask_utils.encode(
                    np.asfortranarray(aug_preds[ai]["binary_mask"])
                )
                aug_rles.append(rle)

            cls_base_rles = [base_rles[bi] for bi in base_indices]

            # pycocotools mask_utils.iou(dt, gt, iscrowd)
            # Returns (len(dt), len(gt)) IoU matrix
            iscrowd = [0] * len(cls_base_rles)
            iou_matrix = mask_utils.iou(aug_rles, cls_base_rles, iscrowd)
            iou_matrix = np.array(iou_matrix, dtype=np.float32)

            if iou_matrix.size == 0:
                continue

            # Greedy matching: pick highest IoU pairs first
            while True:
                max_val = iou_matrix.max()
                if max_val < mask_iou_thresh:
                    break

                aug_local, base_local = np.unravel_index(
                    iou_matrix.argmax(), iou_matrix.shape
                )
                aug_global = aug_indices[aug_local]
                base_global = base_indices[base_local]

                # Accumulate score
                score_sums[base_global] += aug_preds[aug_global]["score"]
                match_counts[base_global] += 1

                matched_base_set.add(base_global)
                matched_aug_set.add(aug_global)

                # Invalidate this row and column
                iou_matrix[aug_local, :] = 0
                iou_matrix[:, base_local] = 0

        # Collect unmatched high-confidence augmented predictions
        for aug_idx, p in enumerate(aug_preds):
            if aug_idx not in matched_aug_set and p["score"] >= add_unmatched_thresh:
                all_unmatched.append(p)

    # ── Build fused output ──
    fused = []
    for i, p in enumerate(base_preds):
        fused_score = score_sums[i] / total_views
        fused.append({
            "bbox_xyxy": p["bbox_xyxy"],
            "score": float(fused_score),
            "label": p["label"],
            "binary_mask": p["binary_mask"],
        })

    # Add unmatched high-confidence augmented detections
    # (These are instances the original view missed)
    # De-duplicate among unmatched using a simple mask IoU check
    if len(all_unmatched) > 0:
        # Encode all unmatched + fused for de-duplication
        for up in all_unmatched:
            up_rle = mask_utils.encode(np.asfortranarray(up["binary_mask"]))

            # Check if this unmatched pred overlaps with any existing fused pred
            # of the same class
            is_duplicate = False
            for fp in fused:
                if fp["label"] != up["label"]:
                    continue
                fp_rle = mask_utils.encode(np.asfortranarray(fp["binary_mask"]))
                iou_val = mask_utils.iou([up_rle], [fp_rle], [0])[0][0]
                if iou_val >= mask_iou_thresh:
                    is_duplicate = True
                    break

            if not is_duplicate:
                fused.append({
                    "bbox_xyxy": up["bbox_xyxy"],
                    "score": up["score"] / total_views,
                    "label": up["label"],
                    "binary_mask": up["binary_mask"],
                })

    return fused


@torch.no_grad()
def run_inference(
    model,
    data_loader,
    device,
    score_threshold: float = 0.05,
    tile_size: int = 512,
    tile_overlap: int = 128,
    use_tiling: bool = True,
    soft_nms_sigma: float = 0.5,
    soft_nms_score_thresh: float = 0.05,
    use_tta: bool = False,
    tta_mask_iou_thresh: float = 0.3,
    tta_add_unmatched_thresh: float = 0.7,
    tta_scales: list = None,
):
    """Run inference on the test set.

    Args:
        use_tta: 啟用 Test-Time Augmentation，包含：
                 - 幾何：hflip + vflip + rot90 + rot270（訓練時用過的 augmentation）
                 - 多尺度：額外的 min_size resize（模型訓練過的 scale range）
                 使用 mask IoU fusion 合併：以原圖為 base，augmented views 做
                 per-class greedy mask IoU matching，matched 取 score 平均，
                 unmatched 高信心預測加入新 instance。
        tta_mask_iou_thresh: Mask IoU threshold for matching TTA predictions.
        tta_add_unmatched_thresh: Score threshold for adding unmatched augmented
                                  predictions as new instances.
        tta_scales: Additional min_size values for multi-scale TTA.
                    e.g., [640] = run an extra inference at min_size=640.
                    Default eval uses min_size[-1] (=800 from config).
    """
    if tta_scales is None:
        tta_scales = []

    model.eval()
    all_predictions = []

    for batch_idx, (images, infos) in enumerate(data_loader):
        for i, image in enumerate(images):
            image = image.to(device)  # (3, H, W)
            info = infos[i]
            image_id = int(info["id"])
            img_h = int(info["height"])
            img_w = int(info["width"])

            # Common kwargs (without img dimensions — rotation uses different dims)
            common_kwargs = dict(
                model=model,
                score_threshold=score_threshold,
                use_tiling=use_tiling,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
            )

            # ── 原圖推論 (base) ──
            base_preds = _infer_single_image(
                image=image, img_h=img_h, img_w=img_w, **common_kwargs)

            # ── TTA: mask IoU fusion ──
            if use_tta:
                aug_preds_list = []

                # ── 幾何 augmentation（訓練時用過） ──

                # 水平翻轉
                image_hflip = image.flip(dims=[2])
                preds_hflip = _infer_single_image(
                    image=image_hflip, img_h=img_h, img_w=img_w, **common_kwargs)
                preds_hflip = _flip_predictions_h(preds_hflip, img_w)
                aug_preds_list.append(preds_hflip)
                del image_hflip
                torch.cuda.empty_cache()

                # 垂直翻轉
                image_vflip = image.flip(dims=[1])
                preds_vflip = _infer_single_image(
                    image=image_vflip, img_h=img_h, img_w=img_w, **common_kwargs)
                preds_vflip = _flip_predictions_v(preds_vflip, img_h)
                aug_preds_list.append(preds_vflip)
                del image_vflip
                torch.cuda.empty_cache()

                # 90° / 270° 旋轉
                # pad to square（若 max(H,W) > 1024 先縮小，防止 paste_masks OOM）
                padded, S, rh, rw, inv_s = _pad_to_square(image, max_side=1024)

                image_rot90 = torch.rot90(padded, k=1, dims=(1, 2))
                preds_rot90 = _infer_single_image(
                    image=image_rot90, img_h=S, img_w=S, **common_kwargs)
                preds_rot90 = _rot90_predictions(
                    preds_rot90, k=1, square_size=S,
                    resized_h=rh, resized_w=rw,
                    img_h=img_h, img_w=img_w, inv_scale=inv_s)
                aug_preds_list.append(preds_rot90)
                del image_rot90
                torch.cuda.empty_cache()

                image_rot270 = torch.rot90(padded, k=3, dims=(1, 2))
                preds_rot270 = _infer_single_image(
                    image=image_rot270, img_h=S, img_w=S, **common_kwargs)
                preds_rot270 = _rot90_predictions(
                    preds_rot270, k=3, square_size=S,
                    resized_h=rh, resized_w=rw,
                    img_h=img_h, img_w=img_w, inv_scale=inv_s)
                aug_preds_list.append(preds_rot270)
                del image_rot270, padded
                torch.cuda.empty_cache()

                # ── Multi-scale augmentation ──
                # 訓練時 multi_scale_min_sizes=(640–800)，模型學過這些 scale。
                # 預設 eval 用 800（min_size[-1]），額外的 scale 提供不同解析度視角。
                # torchvision postprocessing 會自動將 predictions 映射回原圖座標。
                if tta_scales and hasattr(model, 'transform') and \
                        hasattr(model.transform, 'min_size'):
                    original_min_size = model.transform.min_size
                    for scale in tta_scales:
                        model.transform.min_size = (scale,)
                        preds_scale = _infer_single_image(
                            image=image, img_h=img_h, img_w=img_w,
                            **common_kwargs)
                        aug_preds_list.append(preds_scale)
                        del preds_scale
                        torch.cuda.empty_cache()
                    model.transform.min_size = original_min_size

                # Mask IoU fusion
                per_image = _fuse_tta_mask_iou(
                    base_preds=base_preds,
                    aug_preds_list=aug_preds_list,
                    img_h=img_h,
                    img_w=img_w,
                    mask_iou_thresh=tta_mask_iou_thresh,
                    add_unmatched_thresh=tta_add_unmatched_thresh,
                )

                del aug_preds_list
                torch.cuda.empty_cache()
            else:
                per_image = base_preds

            # ── Tiling merge (Soft-NMS, only when tiling without TTA) ──
            # TTA already does fusion; tiling-only uses Soft-NMS for tile overlap
            if not use_tta and use_tiling and len(per_image) > 0:
                by_class = defaultdict(list)
                for idx_pred, p in enumerate(per_image):
                    by_class[p["label"]].append(idx_pred)

                survivors_idx = []
                for cat_id, idxs in by_class.items():
                    sub_boxes = np.array(
                        [per_image[k]["bbox_xyxy"] for k in idxs],
                        dtype=np.float32,
                    )
                    sub_scores = np.array(
                        [per_image[k]["score"] for k in idxs],
                        dtype=np.float32,
                    )
                    keep_local = soft_nms_box(
                        sub_boxes,
                        sub_scores,
                        sigma=soft_nms_sigma,
                        score_thresh=soft_nms_score_thresh,
                    )
                    survivors_idx.extend(idxs[k] for k in keep_local.tolist())

                per_image = [per_image[k] for k in survivors_idx]

            # ── Emit predictions ──
            for p in per_image:
                gx1, gy1, gx2, gy2 = p["bbox_xyxy"]
                # 編碼 binary mask → RLE
                rle = mask_utils.encode(np.asfortranarray(p["binary_mask"]))
                rle["counts"] = rle["counts"].decode("utf-8")
                bbox_xywh = [
                    float(gx1),
                    float(gy1),
                    float(gx2 - gx1),
                    float(gy2 - gy1),
                ]
                all_predictions.append({
                    "image_id": image_id,
                    "category_id": p["label"],
                    "bbox": bbox_xywh,
                    "score": p["score"],
                    "segmentation": {
                        "size": [img_h, img_w],
                        "counts": rle["counts"],
                    },
                })

            # 釋放本張圖記憶體
            del per_image, base_preds, image
            torch.cuda.empty_cache()

        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed {batch_idx + 1}/{len(data_loader)} batches "
                  f"(total predictions so far: {len(all_predictions)})")

    return all_predictions


def main(args):
    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Load test image info ──
    json_path = os.path.join(args.data_root, "test_image_name_to_ids.json")
    with open(json_path) as f:
        test_image_info = json.load(f)
    print(f"Test images: {len(test_image_info)}")

    # ── Test dataset ──
    test_dir = os.path.join(args.data_root, "test_release")
    test_dataset = CellTestDataset(test_dir, test_image_info)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    # ── Load model ──
    # 必須與 train.py 用同一組 min_size / max_size，否則 torchvision 在 eval()
    # 會用預設的 min_size[-1] = 512 做 resize → 與訓練時 val 的 800 對不上。
    model = get_baseline_model(
        num_classes=args.num_classes,
        min_size=tuple(args.multi_scale_min_sizes),
        max_size=args.multi_scale_max_size,
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    print(f"Loaded checkpoint from: {args.checkpoint}")
    if "ap50" in checkpoint:
        print(f"  Checkpoint AP50: {checkpoint['ap50']:.4f}")

    # ── Run inference ──
    use_tiling = bool(args.tiling)
    use_tta = bool(args.tta)
    print(
        "Inference config | "
        f"tiling={use_tiling}, tta={use_tta}, "
        f"resize_min={tuple(args.multi_scale_min_sizes)} "
        f"(eval uses last={args.multi_scale_min_sizes[-1]}), "
        f"resize_max={args.multi_scale_max_size}, "
        f"score_thresh={args.score_threshold}, "
        f"tile_size={args.tile_size}, overlap={args.tile_overlap}, "
        f"soft_nms_sigma={args.soft_nms_sigma}, "
        f"soft_nms_score_thresh={args.soft_nms_score_thresh} "
        "(soft-NMS used for tiling overlap only)"
    )
    tta_scales = list(args.tta_scales) if args.tta_scales else []
    if use_tta:
        n_views = 5 + len(tta_scales)  # orig + hflip + vflip + rot90 + rot270 + scales
        scale_str = f"+scales{tta_scales}" if tta_scales else ""
        print(
            f"TTA config | views={n_views} "
            f"(orig+hflip+vflip+rot90+rot270{scale_str}), "
            f"fusion=mask_iou, "
            f"mask_iou_thresh={args.tta_mask_iou_thresh}, "
            f"add_unmatched_thresh={args.tta_add_unmatched_thresh}"
        )
    print("Running inference...")
    predictions = run_inference(
        model,
        test_loader,
        device,
        score_threshold=args.score_threshold,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        use_tiling=use_tiling,
        soft_nms_sigma=args.soft_nms_sigma,
        soft_nms_score_thresh=args.soft_nms_score_thresh,
        use_tta=use_tta,
        tta_mask_iou_thresh=args.tta_mask_iou_thresh,
        tta_add_unmatched_thresh=args.tta_add_unmatched_thresh,
        tta_scales=tta_scales,
    )
    print(f"Total predictions: {len(predictions)}")

    # ── Save results ──
    output_path = args.output
    with open(output_path, "w") as f:
        json.dump(predictions, f)
    print(f"Results saved to: {output_path}")

    # ── Print summary ──
    class_counts = Counter(p["category_id"] for p in predictions)
    print("\nPrediction summary per class:")
    for cls_id in sorted(class_counts.keys()):
        scores = [p["score"] for p in predictions if p["category_id"] == cls_id]
        print(f"  Class {cls_id}: {class_counts[cls_id]} predictions, "
              f"avg score: {np.mean(scores):.3f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference for cell instance segmentation",
    )

    parser.add_argument("--data_root", type=str, default="data",
                        help="Root directory of dataset")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth",
                        help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="test-results.json",
                        help="Output JSON file path")
    parser.add_argument("--num_classes", type=int, default=5,
                        help="Number of classes (4 + background)")
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.05,
        help=(
            "Minimum score before Soft-NMS / before emitting predictions"
        ),
    )
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers")

    # ── V6.1: Image scale (must match train.py to avoid scale mismatch) ──
    parser.add_argument(
        "--multi_scale_min_sizes",
        type=int,
        nargs="+",
        default=list(MODEL_CONFIG["multi_scale_min_sizes"]),
        help=("Resize shortest-edge candidates. torchvision uses the LAST value "
              "in eval() — keep this aligned with train.py."),
    )
    parser.add_argument(
        "--multi_scale_max_size",
        type=int,
        default=MODEL_CONFIG["multi_scale_max_size"],
        help="Longest-edge resize cap (must match train.py).",
    )

    # ── V6: Tiled inference (opt-in) ──
    parser.add_argument(
        "--tiling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=("Enable sliding-window tiled inference + box-IoU Soft-NMS. "
              "Default OFF; this matches the V2 / val pipeline."),
    )
    parser.add_argument("--tile_size", type=int, default=512,
                        help="Sliding-window tile size (matches training tile_size).")
    parser.add_argument("--tile_overlap", type=int, default=128,
                        help="Sliding-window overlap.")

    # ── V6: Box-IoU Soft-NMS (only used when --tiling or --tta) ──
    parser.add_argument("--soft_nms_sigma", type=float, default=0.5,
                        help="Gaussian Soft-NMS sigma.")
    parser.add_argument("--soft_nms_score_thresh", type=float, default=0.05,
                        help="Final score threshold after Soft-NMS.")

    # ── TTA (Mask IoU Fusion) ──
    parser.add_argument(
        "--tta",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=("Enable Test-Time Augmentation with mask IoU fusion. "
              "Geometric: hflip + vflip + rot90 + rot270 (5 views). "
              "Add --tta_scales for multi-scale views."),
    )
    parser.add_argument("--tta_mask_iou_thresh", type=float, default=0.3,
                        help="Mask IoU threshold for matching TTA predictions to base.")
    parser.add_argument("--tta_add_unmatched_thresh", type=float, default=0.7,
                        help=("Score threshold for adding unmatched augmented "
                              "predictions as new instances."))
    parser.add_argument(
        "--tta_scales",
        type=int,
        nargs="*",
        default=[640],
        help=("Additional min_size scales for multi-scale TTA. "
              "Default [640]. Set empty (--tta_scales) for geometric only. "
              "Training used min_sizes=(640–800), eval default=800."),
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
