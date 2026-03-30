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

3. Configure Weights & Biases API key:
   - Replace `os.environ['WANDB_API_KEY']` in `train.py`, or
   - Export environment variable before running:

   ```bash
   export WANDB_API_KEY="your_wandb_api_key"
   ```

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
![1774599568108](image/README/1774599568108.png)
