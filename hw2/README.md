# NYCU Computer Vision 2026 HW2

- **Student ID** 112550097
- **Name**: Your Name

## Introduction

Brief introduction of this homework and your implementation.

## Environment Setup

How to install dependencies.

```bash
git clone <your-repo-url>
cd hw2
pip install -r requirement.txt
cd dino/ops
python setup.py build install
cd ../..
```

## Dataset Structure

Put `data/` at the same level as `dino/` (in the repository root).

```text
hw2/
├── dino/
├── detr/
├── data/
│   ├── train/
│   ├── valid/
│   ├── test/
│   ├── train.json
│   └── valid.json
└── ...
```

## Usage

### Training / Inference

For best result, use `dino/`.

For vanilla DETR baseline, use `detr/`.

```
# checkpoint/log would be stored in output/${date_time}/
python train.py

python inference.py --checkpoint <your_checkpoint>
```

## Performance Snapshot

Insert a screenshot of your leaderboard or validation performance here.

## References

This homework codebase is modified from the following repositories:

- [DINO](https://github.com/IDEA-Research/DINO)
- [DETR](https://github.com/facebookresearch/detr)
