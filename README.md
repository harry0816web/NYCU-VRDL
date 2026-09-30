# NYCU Visual Recognition using Deep Learning

本 repository 收錄國立陽明交通大學 **Visual Recognition using Deep Learning** 課程的作業實作。四個 project 分別涵蓋影像分類、物件偵測、實例分割與影像復原。

## Projects

| Project | 主題 | 主要方法 | 輸出 |
| --- | --- | --- | --- |
| [HW1](hw1/) | 影像分類 | ResNeXt-50、Bagging Ensemble | `prediction.csv` |
| [HW2](hw2/) | 物件偵測 | DINO、DETR | COCO 格式的預測 JSON |
| [HW3](hw3/) | 細胞實例分割 | Mask R-CNN、FPN | COCO RLE 格式的 `submission.json` |
| [HW4](hw4/) | 去雨、去雪影像復原 | PromptIR | `pred.npz` |

### [HW1: Image Classification](hw1/)

針對 100 個類別進行影像分類。以 ImageNet 預訓練的 **ResNeXt-50** 為 backbone，搭配資料增強、label smoothing 與三組不同 random seed 的 Bagging 訓練；推論時以 soft voting 整合三個模型的結果，產生測試集分類結果。

- 訓練與推論入口：`hw1/train.py`
- 詳細說明：[hw1/README.md](hw1/README.md)

### [HW2: Object Detection](hw2/)

在 11 類物件資料集上進行物件偵測。專案包含 **DETR** baseline 與主要使用的 **DINO** 實作，負責預測物件類別、bounding box 與 confidence score；DINO 版本另提供 ensemble 訓練與推論工具。

- DINO 實作：`hw2/dino/`
- DETR baseline：`hw2/detr/`
- 詳細說明：[hw2/README.md](hw2/README.md)

### [HW3: Instance Segmentation](hw3/)

對顯微影像中的細胞進行四類 **instance segmentation**。模型採用 **Mask R-CNN + FPN**，並針對單張影像包含大量細胞的情況調整 proposals 與 detections 數量；同時使用資料增強與快取機制改善小型資料集的訓練效率。最終預測以 COCO-style RLE 格式提交，評估指標為 AP50。

- 訓練入口：`hw3/train.py`
- 推論入口：`hw3/inference.py`
- 詳細說明：[hw3/README.md](hw3/README.md)

### [HW4: Image Restoration](hw4/)

使用單一模型移除 RGB 影像中的雨與雪。專案以 **PromptIR** 為基礎，使用 PyTorch Lightning 從頭訓練，並支援 spatial prompt、EMA、複合 loss 與 test-time augmentation 等選項。模型輸出復原後的影像，評估指標為 PSNR。

- 訓練入口：`hw4/train.py`
- 推論入口：`hw4/inference.py`
- 詳細說明：[hw4/readme.md](hw4/readme.md)

## Repository Structure

```text
NYCU-VRDL/
├── hw1/                    # Image classification
├── hw2/                    # Object detection
│   ├── dino/               # DINO implementation
│   └── detr/               # DETR baseline
├── hw3/                    # Cell instance segmentation
├── hw4/                    # Rain and snow image restoration
├── 2. DNN_CNN.pdf          # Course material
└── 3. CNN_Object_Recognition.pdf
```

## Usage

各 project 使用不同的資料格式與相依套件。請進入對應的 `hw*` 目錄，依該目錄 README 的說明準備環境與資料集後，再執行訓練或推論程式。資料集、模型 checkpoint 與預測結果不一定包含在 repository 中。
