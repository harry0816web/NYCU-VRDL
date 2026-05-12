"""
Local validation for HW3 using the bundled cocoeval.py.

Runs inference on the held-out val split (hw3_val), builds a COCO-format
ground-truth dict from the same split, then scores predictions with the
reference COCOeval implementation (cocoeval.py).

Usage:
    python validate.py --output output/r50 --weights output/r50/model_final.pth
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List

import numpy as np
from pycocotools.coco import COCO

from cocoeval import COCOeval
from dataset import (
    DATASET_NAME_VAL,
    THING_CLASSES,
    register_datasets,
)
from detectron2.data import DatasetCatalog
from inference import _instances_to_coco_records, load_predictor
from dataset import read_image_rgb


def build_val_gt_coco(val_records: List[Dict]) -> Dict:
    """Convert Detectron2-style val records to a COCO GT dict."""
    images, annotations = [], []
    categories = [
        {"id": i + 1, "name": name} for i, name in enumerate(THING_CLASSES)
    ]
    ann_id = 1
    for rec in val_records:
        images.append(
            {
                "id": int(rec["image_id"]),
                "file_name": os.path.basename(rec["file_name"]),
                "height": int(rec["height"]),
                "width": int(rec["width"]),
            }
        )
        for a in rec["annotations"]:
            seg = a["segmentation"]
            # cocoeval needs size[H,W] + counts (bytes or str accepted)
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": int(rec["image_id"]),
                    "category_id": int(a["category_id"]) + 1,  # 1-based
                    "bbox": [float(x) for x in a["bbox"]],
                    "area": float(a["area"]),
                    "iscrowd": int(a.get("iscrowd", 0)),
                    "segmentation": {
                        "size": list(seg["size"]),
                        "counts": seg["counts"],
                    },
                }
            )
            ann_id += 1
    return {"images": images, "annotations": annotations, "categories": categories}


def run_val_inference(
    val_records: List[Dict],
    cfg_path: str,
    weights: str,
    score_thresh: float,
) -> List[Dict]:
    predictor = load_predictor(cfg_path, weights, score_thresh)
    results: List[Dict] = []
    t0 = time.time()
    for i, rec in enumerate(val_records, 1):
        img = read_image_rgb(rec["file_name"])
        outputs = predictor(img)
        results.extend(
            _instances_to_coco_records(outputs["instances"], rec["image_id"])
        )
        if i % 10 == 0 or i == len(val_records):
            print(f"  [{i}/{len(val_records)}] "
                  f"elapsed={time.time() - t0:.1f}s "
                  f"total_instances={len(results)}")
    return results


def evaluate(gt_dict: Dict, dt_records: List[Dict], iou_type: str) -> np.ndarray:
    """Run COCOeval for the given iouType (expects 'segm' or 'bbox')."""
    coco_gt = COCO()
    coco_gt.dataset = gt_dict
    coco_gt.createIndex()

    if len(dt_records) == 0:
        print(f"[{iou_type}] no detections — skipping")
        return np.zeros(12)

    coco_dt = coco_gt.loadRes(dt_records)
    ev = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return ev.stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data")
    p.add_argument("--output", default="output/r50")
    p.add_argument("--weights", default=None)
    p.add_argument("--cfg", default=None)
    p.add_argument("--score-thresh", type=float, default=0.05)
    p.add_argument("--predictions-out", default=None,
                   help="Optional path to dump the val predictions JSON.")
    p.add_argument("--gt-out", default=None,
                   help="Optional path to dump the val GT COCO JSON.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    weights = args.weights or os.path.join(args.output, "model_final.pth")
    cfg_path = args.cfg or os.path.join(args.output, "config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"config.yaml not found at {cfg_path}")
    if not os.path.exists(weights):
        raise FileNotFoundError(f"weights not found at {weights}")

    register_datasets(data_root=args.data_root)
    val_records = DatasetCatalog.get(DATASET_NAME_VAL)
    print(f"[validate] {len(val_records)} val images")

    gt_dict = build_val_gt_coco(val_records)
    n_ann = len(gt_dict["annotations"])
    print(f"[validate] GT: {n_ann} annotations, "
          f"{len(gt_dict['categories'])} categories")

    dt_records = run_val_inference(
        val_records, cfg_path, weights, args.score_thresh
    )
    print(f"[validate] predictions: {len(dt_records)} instances")

    if args.predictions_out:
        os.makedirs(os.path.dirname(args.predictions_out) or ".", exist_ok=True)
        with open(args.predictions_out, "w") as f:
            json.dump(dt_records, f)
    if args.gt_out:
        os.makedirs(os.path.dirname(args.gt_out) or ".", exist_ok=True)
        with open(args.gt_out, "w") as f:
            json.dump(gt_dict, f)

    print("\n=========== Segmentation AP ===========")
    segm_stats = evaluate(gt_dict, dt_records, "segm")
    print("\n=========== Bounding-box AP ===========")
    bbox_stats = evaluate(gt_dict, dt_records, "bbox")

    # HW3 is scored on AP50
    print("\n[validate] summary")
    print(f"  segm  AP     = {segm_stats[0]:.4f}")
    print(f"  segm  AP50   = {segm_stats[1]:.4f}   <-- HW3 leaderboard metric")
    print(f"  segm  AP75   = {segm_stats[2]:.4f}")
    print(f"  bbox  AP50   = {bbox_stats[1]:.4f}")


if __name__ == "__main__":
    main()
