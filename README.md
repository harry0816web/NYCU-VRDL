# NYCU Visual Recognition Using Deep Learning

Coursework for **Visual Recognition Using Deep Learning** at National Yang Ming Chiao Tung University. This repository contains four computer-vision assignments covering image classification, object detection, instance segmentation, and image restoration.

## Assignments

| Assignment | Task | Approach | Submission | Resources |
| --- | --- | --- | --- | --- |
| **HW1** | 100-class image classification | ImageNet-pretrained ResNeXt-50, data augmentation, and a three-seed bagging ensemble | `prediction.csv` | [Code](hw1/) · [README](hw1/README.md) · [Report](hw1/112550097_HW1.pdf) · [Brief](<hw1/(114-2) HW1 Slides.pdf>) |
| **HW2** | Digit object detection | DINO with a ResNet-50 backbone; DETR is included as a baseline | COCO-format `pred.json` | [Code](hw2/) · [README](hw2/README.md) · [Report](hw2/112550097_HW2.pdf) · [Brief](<hw2/(114-2) HW2 Slides.pdf>) |
| **HW3** | Cell instance segmentation | Mask R-CNN with a ConvNeXt-Small backbone and FPN | COCO RLE `test-results.json` | [Code](hw3/) · [README](hw3/README.md) · [Report](hw3/112550097_HW3.pdf) · [Brief](<hw3/(114-2) HW3 Slides.pdf>) |
| **HW4** | Rain and snow image restoration | PromptIR with spatial prompts, EMA, L1 + SSIM loss, and test-time augmentation | `pred.npz` | [Code](hw4/) · [README](hw4/readme.md) · [Report](hw4/112550097_HW4.pdf) · [Brief](<hw4/(114-2) HW4 Slides.pdf>) |

## Highlights

### HW1 — Image Classification

Classifies approximately 23,000 images across 100 categories. The final system fine-tunes a ResNeXt-50 model with strong augmentation and label smoothing, then combines models trained with different random seeds through soft-voting bagging.

### HW2 — Object Detection

Detects digit instances in COCO-format data. The main solution replaces vanilla DETR with DINO, uses multi-scale deformable attention, reduces the transformer depth for the dataset, and increases input resolution to preserve small digit features.

### HW3 — Instance Segmentation

Segments four categories of cells in densely populated microscopy images. The solution combines Mask R-CNN, a ConvNeXt-Small backbone, and FPN, with increased proposal and detection budgets to improve recall in crowded scenes.

### HW4 — Image Restoration

Restores images degraded by rain or snow using a single PromptIR model. The final recipe adds spatially gated prompts, EMA weights, a combined L1 and SSIM objective, and eight-way geometric self-ensemble inference.

## Repository Layout

```text
NYCU-VRDL/
├── hw1/   # Image classification
├── hw2/   # Object detection (DINO and DETR)
├── hw3/   # Cell instance segmentation
└── hw4/   # Rain and snow image restoration
```

Each homework directory contains its implementation, assignment brief, final report, and a dedicated README with setup, dataset layout, training, and inference instructions.

## Usage

The assignments use different environments and dataset formats. Enter the relevant `hw*` directory and follow its README before running the training or inference scripts. Datasets, checkpoints, experiment logs, and generated submissions are intentionally excluded from version control unless otherwise noted.

## Author

**Hung-I Yang** · Student ID `112550097`
