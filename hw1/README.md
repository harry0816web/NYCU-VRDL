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
3. **資料集目錄 `data/`**

   請將課程提供的影像資料放在專案根目錄底下的 `data/`。`train.py` 會從此路徑讀取資料，目錄架構如下：

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

   - **train / val**：須符合 PyTorch `ImageFolder` 慣例，同一類別的影像放在同一子資料夾；`train` 與 `val` 的類別子資料夾名稱需一致，以便標籤對應。
   - **test**：所有待預測影像直接放在 `data/test/` 根層。
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
