# NYCU Computer Vision 2026 HW1

- Student ID: `112550097`
- Name: `Hung-I Yang`

## Introduction

This project trains an image classification model with a bagging ensemble strategy based on `ResNeXt50` pre-trained models.
The final output is a `prediction.csv` file for the test set.

## Environment Setup

1. Create and activate a virtual environment:
  ```bash
   python3 -m venv .venv
   source .venv/bin/activate
  ```
2. Install required packages:
  ```bash
   pip install -r requirements.txt
  ```
3. **Dataset directory `data/`**

   Place the course image data under `data/` at the project root. `train.py` reads from this path. The expected layout is:

   ```text
   data/
   ├── train/
   │   ├── <class0>/
   │   │   └── img1
   │   ├── <class1>/
   │   └── ...
   ├── val/
   │   ├── <class0>/
   │   ├── <class1>/
   │   └── ...
   └── test/
       ├── 001.jpg
       ├── 002.png
       └── ...                    # no label
   ```

   - **train / val**: Must follow the PyTorch `ImageFolder` layout—images of the same class live in one subfolder; class folder names must match between `train` and `val` so labels align.
   - **test**: Put all images to predict directly under `data/test/` (no subfolders by class).

4. Configure Weights & Biases API key:
  - Replace `os.environ['WANDB_API_KEY']` in `train.py`, or
  - Export environment variable before running:

## Usage

### Training + Inference

Run the main training script:

```bash
python train.py
```

Expected outputs:

- Model checkpoints in `resnext_ensemble_models/`
- Test predictions in `prediction.csv`

## Performance Snapshot

![Performance snapshot](image/README/1774599568108.png)
