# PromptIR HW4 — Architecture & Dataset Analysis

## Task Overview

**Image Restoration**: Given degraded images (rain or snow), restore them to clean images using a single PromptIR model. Evaluated by PSNR on a private test set via CodaBench.

| Target | PSNR | Score |
|--------|------|-------|
| Weak baseline | ~26 dB | 60 |
| Strong baseline | ~30 dB | 80 |
| Rank 3 | unknown | 100 |

---

## Dataset Analysis

### Structure

```
data/
├── train/
│   ├── degraded/          # 3200 images
│   │   ├── rain-1.png ~ rain-1600.png
│   │   └── snow-1.png ~ snow-1600.png
│   └── clean/             # 3200 matching targets
│       ├── rain_clean-1.png ~ rain_clean-1600.png
│       └── snow_clean-1.png ~ snow_clean-1600.png
└── test/
    └── degraded/          # 100 images (type unknown)
        └── 0.png ~ 99.png
```

### Image Properties

| Property | Value |
|----------|-------|
| Resolution | 256 × 256 |
| Color | RGB (3 channels) |
| Dtype | uint8 (0–255) |
| Train pairs | 3200 (1600 rain + 1600 snow) |
| Test images | 100 (mixed, unlabeled) |

### Degradation Statistics

| Type | Degraded PSNR | Degraded Mean | Clean Mean | Observation |
|------|---------------|---------------|------------|-------------|
| Rain | ~13.8 dB | 119.6 | 91.7 | Heavy degradation; rain streaks add significant intensity |
| Snow | ~21.1 dB | 145.8 | 133.5 | Moderate degradation; snow particles are subtler |

Rain degradation is substantially heavier than snow (~7 dB gap). Both degradation types shift pixel intensity upward (additive artifacts).

---

## Model Architecture: PromptIR

Reference: *PromptIR: Prompting for All-in-One Blind Image Restoration* (Potlapalli et al., 2023, [arXiv:2306.13090](https://arxiv.org/abs/2306.13090))

### High-Level Design

PromptIR is a U-Net style Transformer with **learnable prompt blocks** injected in the decoder. The prompts allow a single model to handle multiple degradation types without explicit type labels at inference time.

```
Input (3, 256, 256)
  │
  ├─ OverlapPatchEmbed (Conv 3×3) → (48, 256, 256)
  │
  ├─ Encoder Level 1: 4× TransformerBlock   (48 ch,  1 head)  → skip₁
  ├─ Downsample (PixelUnshuffle)             (96 ch,  128×128)
  ├─ Encoder Level 2: 6× TransformerBlock   (96 ch,  2 heads) → skip₂
  ├─ Downsample                              (192 ch, 64×64)
  ├─ Encoder Level 3: 6× TransformerBlock   (192 ch, 4 heads) → skip₃
  ├─ Downsample                              (384 ch, 32×32)
  │
  ├─ Latent: 8× TransformerBlock            (384 ch, 8 heads)
  │   └─ PromptGenBlock₃ → concat + reduce
  │
  ├─ Upsample + skip₃ → Decoder Level 3: 6× TransformerBlock
  │   └─ PromptGenBlock₂ → concat + reduce
  ├─ Upsample + skip₂ → Decoder Level 2: 6× TransformerBlock
  │   └─ PromptGenBlock₁ → concat + reduce
  ├─ Upsample + skip₁ → Decoder Level 1: 4× TransformerBlock
  │
  ├─ Refinement: 4× TransformerBlock        (96 ch,  1 head)
  ├─ Output Conv (3×3) → (3, 256, 256)
  │
  └─ + Input (residual connection)
```

**Total parameters**: 35.6M (all trainable)

### Key Components

#### TransformerBlock

Each block contains:
- **MDTA** (Multi-DConv Head Transposed Self-Attention): channel-wise attention with depthwise convolutions for local context. Uses normalized Q/K with learnable temperature.
- **GDFN** (Gated-Dconv Feed-Forward Network): gated mechanism with depthwise conv — `GELU(DWConv(x₁)) × DWConv(x₂)`.
- LayerNorm (WithBias variant) before each sub-layer.

#### PromptGenBlock

The core innovation of PromptIR. Each block:
1. Maintains a **learnable prompt bank**: `(1, prompt_len, prompt_dim, H, W)` — 5 prompt components stored as parameters.
2. Computes **content-adaptive weights** via global average pooling → linear layer → softmax.
3. Weighted-sums the prompt components and interpolates to match feature resolution.
4. Concatenated with decoder features, then reduced back via 1×1 conv.

Three PromptGenBlocks are placed at different decoder levels:

| Block | Prompt Dim | Prompt Size | Linear Dim | Feature Resolution |
|-------|-----------|-------------|------------|-------------------|
| prompt3 | 320 | 16×16 | 384 | 32×32 (latent) |
| prompt2 | 128 | 32×32 | 192 | 64×64 |
| prompt1 | 64 | 64×64 | 96 | 128×128 |

#### Residual Learning

The final output is `Conv(features) + input_image`, so the network learns to predict the **residual** (degradation to be removed) rather than the clean image directly.

#### Down/Upsample

- **Downsample**: Conv 3×3 (halve channels) → PixelUnshuffle(2) — spatial ÷2, channels ×4, net ×2.
- **Upsample**: Conv 3×3 (double channels) → PixelShuffle(2) — spatial ×2, channels ÷4, net ÷2.

---

## Training Configuration

| Parameter | Default | Notes |
|-----------|---------|-------|
| Optimizer | AdamW | weight_decay=1e-4 |
| Learning rate | 2e-4 | |
| Scheduler | LinearWarmup + CosineAnnealing | warmup=15 epochs, eta_min=1e-6 |
| Epochs | 150 | |
| Batch size | 8 | |
| Patch size | 128×128 | Random crop from 256×256 |
| Loss | L1 | Alternatives: Charbonnier, L1+SSIM |
| Precision | 16-mixed (AMP) | |
| Augmentation | Random flip + rotation (8 modes) | |
| Val split | 5% (160 images) | Stratified random |

---

## Code Structure

```
hw4/
├── config.py       # Dataclass-based configuration (ModelConfig, TrainConfig, DataConfig)
├── model.py        # PromptIR architecture (self-contained, no external dependencies)
├── dataset.py      # TrainDataset (paired rain/snow) + TestDataset (unlabeled)
├── train.py        # PyTorch Lightning training with wandb logging
├── inference.py    # Checkpoint → pred.npz for CodaBench submission
├── pyproject.toml  # uv project dependencies
└── data/           # Dataset (not in repo)
```

### Dependencies Between Files

```
config.py ← (standalone)
model.py  ← (standalone, only torch + einops)
dataset.py ← (standalone, only torch + PIL + numpy)
train.py  ← config.py, model.py, dataset.py
inference.py ← config.py, model.py, dataset.py
```

---

## Submission Format

Output file: `pred.npz` containing a dictionary:
- **Keys**: filenames (`"0.png"`, `"1.png"`, ..., `"99.png"`)
- **Values**: numpy arrays of shape `(3, H, W)`, dtype `uint8` (0–255)

Package into a `.zip` file with `pred.npz` inside for CodaBench upload.
