# NYCU Computer Vision 2026 HW3

- **Student ID:** 112550097
- **Name:** Hung-I Yang

---

## Introduction

This homework is an **instance segmentation** task on colored microscopy images: many cell instances per image, four semantic categories (`class1`–`class4`), with instance masks given as TIFF label maps (each distinct non-zero pixel value is one instance). The goal is to predict instance masks on the test set and submit them in **COCO-style RLE** format; the leaderboard metric is **AP50**.

The implementation follows a **Mask R-CNN + FPN** pipeline in PyTorch/torchvision, with adaptations for **dense scenes** (higher proposal/detection budgets so crowded cells are not truncated), **data augmentation** (geometric flips/90° rotations, color jitter, optional Copy-Paste) for the small labeled set, and **memory-aware** mask handling during evaluation. The report describes a **ConvNeXt-Small** ImageNet backbone with light regularization (partial backbone freeze, dropout before the FPN). Details are in `Report/main.tex`.

---

## Project layout

Place the course dataset under the repository root so the default `data_root` is `**./data`** (same level as `train.py`). Typical layout:

```text
hw3/
├── config.py              # Shared defaults (paths, model, training)
├── dataset.py             # Train/val/test PyTorch datasets
├── model.py               # Mask R-CNN construction
├── preprocess.py          # Optional: build train_cache.pkl from raw TIFFs
├── train.py
├── inference.py           # Test inference → submission JSON
├── validate.py            # Optional: COCOeval on val (see Usage)
├── requirements.txt
├── data/                  # ← put the official dataset here
│   ├── train/             # Labeled samples (see Data layout)
│   ├── test_release/      # Test images
│   ├── test_image_name_to_ids.json
│   └── train_cache.pkl    # Optional; created by preprocess.py
├── checkpoints/           # Created by train.py (best + latest weights)
└── Report/
    └── main.tex           # Write-up
```

Training and inference read paths relative to `--data_root` (default `**data**`), i.e. `**hw3/data/**` when you run scripts from the `hw3` directory.

---

## Data layout

Under `**data/train/**`, each training sample is a folder (one folder per image), for example `data/train/00001/`:


| File                        | Role                                                                                  |
| --------------------------- | ------------------------------------------------------------------------------------- |
| `image.tif`                 | RGB (or grayscale) input image                                                        |
| `class1.tif` … `class4.tif` | Per-class **instance ID maps** (0 = background; each positive integer = one instance) |


**Test data:** images live under `**data/test_release/`**. Filenames and `image_id` mapping come from `**data/test_image_name_to_ids.json**` (read by `inference.py`).

**Optional cache:** run `preprocess.py` once to build `**data/train_cache.pkl`** for faster epoch startup; `dataset.py` will use the cache automatically when this file exists.

---

## Environment setup

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Training

Default configuration is controlled by `config.py` and can be overridden on the command line.

```bash
python train.py
```

Example with explicit data path:

```bash
python train.py --data_root data
```

### (Optional) Preprocess cache

```bash
python preprocess.py --data_root data --output data/train_cache.pkl
```

### Inference

Run from the project root with `data/` populated as above. This writes a **JSON list** of COCO-style predictions (`image_id`, `category_id`, `bbox` in **xywh**, `score`, `segmentation` with `size` and RLE `counts`).

```bash
python inference.py --checkpoint checkpoints/<run>/best_model.pth --output submission.json
```

**Submission format check:** before uploading, you can run `**validate.py`** on the held-out validation split: it runs COCO-style evaluation and can **save the prediction JSON** for inspection with the same record layout as the leaderboard. Use `--predictions-out` to dump predictions; see the docstring at the top of `validate.py` for arguments (e.g. `--output`, `--weights`, `--cfg`, `--data-root`). *Note:* that script targets a **Detectron2-style** config/weights layout as documented there; ensure the dependencies and files it imports (e.g. `cocoeval.py`, Detectron2 dataset registration) match your local setup if you use it.

---

## Performance snapshot

## 1778560236452

