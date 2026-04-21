"""
WBF (Weighted Boxes Fusion) ensemble inference for DINO.

Loads multiple checkpoints (trained with different seeds), runs inference
on each, then fuses predictions with WBF to produce a single pred.json.

Requires:  pip install ensemble-boxes

Usage:
    python ensemble_inference.py \
        --config config.json \
        --checkpoints output/seed_42/.../best.pth \
                      output/seed_123/.../best.pth \
                      output/seed_7/.../best.pth \
        --output pred.json

    # With custom WBF parameters
    python ensemble_inference.py \
        --config config.json \
        --checkpoints ckpt1.pth ckpt2.pth ckpt3.pth \
        --iou_thr 0.55 \
        --skip_box_thr 0.001 \
        --score_threshold 0.01
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from model import build_model
from dataset import Compose, ToTensor, Normalize, RandomResize
from util.box_ops import box_cxcywh_to_xyxy

try:
    from ensemble_boxes import weighted_boxes_fusion
except ImportError:
    raise ImportError(
        "ensemble-boxes is required for WBF ensemble.\n"
        "Install it with:  pip install ensemble-boxes")


def get_val_transforms(config):
    short_edge = config.get('val_short_edge', 320)
    max_size = config.get('val_max_size', 640)
    return Compose([
        RandomResize([short_edge], max_size=max_size),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def load_model(config, checkpoint_path, device):
    """Build model and load a single checkpoint."""
    model, _, postprocessors = build_model(config)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    return model, postprocessors


@torch.no_grad()
def get_raw_predictions(model, tensor_list, num_select):
    """
    Run model forward and extract normalised boxes [0,1] in xyxy format,
    scores, and labels — ready for WBF.
    """
    outputs = model(tensor_list)
    # outputs['pred_logits']: (B, num_queries, num_classes)
    # outputs['pred_boxes']:  (B, num_queries, 4)  cxcywh in [0,1]

    pred_logits = outputs['pred_logits']  # (B, Q, C)
    pred_boxes = outputs['pred_boxes']    # (B, Q, 4)

    prob = pred_logits.sigmoid()  # (B, Q, C)
    B = prob.shape[0]

    all_boxes, all_scores, all_labels = [], [], []
    for b in range(B):
        # Top-k selection (same logic as PostProcess)
        values, indexes = torch.topk(
            prob[b].flatten(), num_select, dim=0)
        scores = values  # (num_select,)
        topk_boxes_idx = indexes // prob.shape[2]
        labels = indexes % prob.shape[2]

        # Get boxes in cxcywh [0,1], convert to xyxy [0,1]
        boxes_cxcywh = pred_boxes[b][topk_boxes_idx]  # (num_select, 4)
        boxes_xyxy = box_cxcywh_to_xyxy(boxes_cxcywh)  # (num_select, 4)

        # Clamp to [0, 1] for WBF
        boxes_xyxy = boxes_xyxy.clamp(min=0.0, max=1.0)

        all_boxes.append(boxes_xyxy.cpu().numpy())
        all_scores.append(scores.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    return all_boxes, all_scores, all_labels


def wbf_ensemble_per_image(models_boxes, models_scores, models_labels,
                           iou_thr=0.55, skip_box_thr=0.0001,
                           weights=None):
    """
    Apply WBF to merge predictions from multiple models for a single image.

    Args:
        models_boxes:  list of np.array (N_i, 4) in xyxy [0,1]
        models_scores: list of np.array (N_i,)
        models_labels: list of np.array (N_i,)
        iou_thr:       WBF IoU threshold for merging
        skip_box_thr:  minimum score to consider a box
        weights:       per-model weights (None = equal)

    Returns:
        fused_boxes:  np.array (M, 4) xyxy [0,1]
        fused_scores: np.array (M,)
        fused_labels: np.array (M,)
    """
    boxes_list = [b.tolist() for b in models_boxes]
    scores_list = [s.tolist() for s in models_scores]
    labels_list = [label.tolist() for label in models_labels]

    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list,
        weights=weights,
        iou_thr=iou_thr,
        skip_box_thr=skip_box_thr,
    )
    return fused_boxes, fused_scores, fused_labels


@torch.no_grad()
def main(config_path, checkpoint_paths, test_dir=None, output_path="pred.json",
         score_threshold=0.0, iou_thr=0.55, skip_box_thr=0.0001,
         weights=None, batch_size=1, num_select_override=None):

    with open(config_path, 'r') as f:
        config = json.load(f)

    device = torch.device(config["device"])
    num_models = len(checkpoint_paths)

    # Default to config's num_select (= num_queries), since each query
    # corresponds to one box position. Using more than num_queries just
    # duplicates the same box with different class labels — pure noise.
    if num_select_override is not None:
        num_select = num_select_override
    else:
        num_select = config.get('num_select', 300)

    print(f"num_select per model: {num_select}")

    # --- Load all models ---
    print(f"Loading {num_models} models...")
    models = []
    for i, ckpt_path in enumerate(checkpoint_paths):
        print(f"  [{i + 1}/{num_models}] {ckpt_path}")
        model, _ = load_model(config, ckpt_path, device)
        models.append(model)

    # --- Prepare test images ---
    if test_dir is None:
        test_dir = os.path.join(config["data_path"], "test")

    transforms = get_val_transforms(config)

    image_files = sorted(Path(test_dir).glob("*.png"))
    if not image_files:
        image_files = sorted(Path(test_dir).glob("*.jpg"))
    print(f"Found {len(image_files)} test images in {test_dir}")

    # --- Inference + WBF ---
    predictions = []

    for batch_start in tqdm(range(0, len(image_files), batch_size),
                            desc="Ensemble inference"):
        batch_paths = image_files[batch_start: batch_start + batch_size]

        tensors, orig_sizes, image_ids = [], [], []
        for img_path in batch_paths:
            image_ids.append(int(img_path.stem))
            img = Image.open(img_path).convert("RGB")
            orig_w, orig_h = img.size
            orig_sizes.append((orig_h, orig_w))
            img_tensor, _ = transforms(img, None)
            tensors.append(img_tensor)

        tensor_list = [t.to(device) for t in tensors]

        # Collect predictions from all models
        # per_model_preds[model_idx] = (boxes_list_per_img, scores, labels)
        per_model_preds = []
        for model in models:
            boxes, scores, labels = get_raw_predictions(
                model, tensor_list, num_select)
            per_model_preds.append((boxes, scores, labels))

        # Apply WBF per image in batch
        for img_idx in range(len(batch_paths)):
            image_id = image_ids[img_idx]
            orig_h, orig_w = orig_sizes[img_idx]

            # Gather this image's predictions across all models
            m_boxes = [per_model_preds[m][0][img_idx]
                       for m in range(num_models)]
            m_scores = [per_model_preds[m][1][img_idx]
                        for m in range(num_models)]
            m_labels = [per_model_preds[m][2][img_idx]
                        for m in range(num_models)]

            fused_boxes, fused_scores, fused_labels = wbf_ensemble_per_image(
                m_boxes, m_scores, m_labels,
                iou_thr=iou_thr,
                skip_box_thr=skip_box_thr,
                weights=weights,
            )

            # Filter by score threshold
            keep = fused_scores > score_threshold
            fused_boxes = fused_boxes[keep]
            fused_scores = fused_scores[keep]
            fused_labels = fused_labels[keep]

            # Convert from normalised xyxy [0,1] to absolute pixel xywh
            for box, score, label in zip(
                    fused_boxes, fused_scores, fused_labels):
                x1, y1, x2, y2 = box
                abs_x1 = x1 * orig_w
                abs_y1 = y1 * orig_h
                abs_w = (x2 - x1) * orig_w
                abs_h = (y2 - y1) * orig_h

                predictions.append({
                    "image_id": image_id,
                    "bbox": [round(abs_x1, 4), round(abs_y1, 4),
                             round(abs_w, 4), round(abs_h, 4)],
                    "score": round(float(score), 6),
                    "category_id": int(label),
                })

    with open(output_path, 'w') as f:
        json.dump(predictions, f)

    print(f"\nEnsemble complete!")
    print(f"  Models used:    {num_models}")
    print(f"  WBF iou_thr:    {iou_thr}")
    print(f"  WBF skip_thr:   {skip_box_thr}")
    print(f"  Score threshold: {score_threshold}")
    print(f"  Total predictions: {len(predictions)}")
    print(f"  Saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO WBF Ensemble Inference')
    parser.add_argument('--config', default='config.json', type=str,
                        help='Path to config.json')
    parser.add_argument(
        '--checkpoints',
        nargs='+',
        required=True,
        type=str,
        help='Paths to checkpoint files (e.g. best.pth from each seed)')
    parser.add_argument('--test_dir', default=None, type=str,
                        help='Test images directory (default: data_path/test)')
    parser.add_argument('--output', default='pred.json', type=str,
                        help='Output prediction JSON path')
    parser.add_argument('--num_select', default=None, type=int,
                        help='Top-k predictions per model per image '
                             '(default: from config num_select)')
    parser.add_argument(
        '--score_threshold',
        default=0.0,
        type=float,
        help='Final score threshold for output (default: 0.0, keep all)')
    parser.add_argument('--iou_thr', default=0.55, type=float,
                        help='WBF IoU threshold for merging overlapping boxes')
    parser.add_argument('--skip_box_thr', default=0.0001, type=float,
                        help='WBF minimum score to keep a box before fusion')
    parser.add_argument(
        '--weights',
        nargs='+',
        type=float,
        default=None,
        help='Per-model weights for WBF (default: equal weights)')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Inference batch size')

    args = parser.parse_args()
    main(args.config, args.checkpoints, args.test_dir, args.output,
         args.score_threshold, args.iou_thr, args.skip_box_thr,
         args.weights, args.batch_size, args.num_select)
