"""
DETR inference: run trained model on test images and output pred.json in COCO format.

Output format:
[
    {"image_id": int, "bbox": [x_min, y_min, w, h], "score": float, "category_id": int},
    ...
]
bbox is in absolute pixel coordinates (no normalization), category_id starts from 1.
"""
import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from model import build_model
from dataset import Compose, ToTensor, Normalize, RandomResize


def get_val_transforms():
    return Compose([
        RandomResize([150]),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def main(config_path, checkpoint_path, test_dir=None, output_path="pred.json",
         score_threshold=0.0, batch_size=1):
    with open(config_path, 'r') as f:
        config = json.load(f)

    device = torch.device(config["device"])

    model, _, postprocessors = build_model(config)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    if test_dir is None:
        test_dir = os.path.join(config["data_path"], "test")

    transforms = get_val_transforms()

    image_files = sorted(Path(test_dir).glob("*.png"))
    if not image_files:
        image_files = sorted(Path(test_dir).glob("*.jpg"))
    print(f"Found {len(image_files)} test images in {test_dir}")

    predictions = []

    samples = []
    for img_path in tqdm(image_files, desc="Loading images"):
        image_id = int(img_path.stem)
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size
        img_tensor, _ = transforms(img, None)
        samples.append((img_tensor, (orig_h, orig_w), image_id))

    for batch_start in tqdm(
            range(
                0,
                len(samples),
                batch_size),
            desc="Running inference"):
        batch_samples = samples[batch_start: batch_start + batch_size]
        tensors, orig_sizes, image_ids = zip(*batch_samples)

        # Pass as list of individual tensors so that
        # nested_tensor_from_tensor_list generates a correct padding mask.
        tensor_list = [t.to(device) for t in tensors]
        outputs = model(tensor_list)

        target_sizes = torch.tensor(list(orig_sizes), device=device)
        results = postprocessors['bbox'](outputs, target_sizes)

        for result, image_id in zip(results, image_ids):
            scores = result['scores']
            labels = result['labels']
            boxes = result['boxes']

            keep = scores > score_threshold
            scores = scores[keep]
            labels = labels[keep]
            boxes = boxes[keep]

            for s, l, b in zip(scores, labels, boxes):
                x0, y0, x1, y1 = b.tolist()
                predictions.append({
                    "image_id": image_id,
                    "bbox": [round(x0, 4), round(y0, 4),
                             round(x1 - x0, 4), round(y1 - y0, 4)],
                    "score": round(s.item(), 6),
                    "category_id": l.item(),
                })

    with open(output_path, 'w') as f:
        json.dump(predictions, f)
    print(f"Saved {len(predictions)} predictions to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR Inference')
    parser.add_argument('--config', default='config.json', type=str)
    parser.add_argument(
        '--checkpoint',
        required=True,
        type=str,
        help='path to trained checkpoint (e.g. output/best.pth)')
    parser.add_argument(
        '--test_dir',
        default=None,
        type=str,
        help='path to test images directory (default: data_path/test)')
    parser.add_argument('--output', default='pred.json', type=str,
                        help='output json file path')
    parser.add_argument('--score_threshold', default=0.01, type=float,
                        help='minimum score threshold for predictions')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='number of images per inference batch')
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.test_dir, args.output,
         args.score_threshold, args.batch_size)
