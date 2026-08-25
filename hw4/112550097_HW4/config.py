"""Centralized configuration for PromptIR training and inference."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """PromptIR architecture hyperparameters."""

    inp_channels: int = 3
    out_channels: int = 3
    dim: int = 64
    num_blocks: List[int] = field(default_factory=lambda: [4, 6, 6, 8])
    num_refinement_blocks: int = 4
    heads: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    ffn_expansion_factor: float = 2.66
    bias: bool = False
    LayerNorm_type: str = "WithBias"  # "WithBias" or "BiasFree"
    decoder: bool = True  # enable PromptGenBlock in decoder

    # --- Optimization 1: Spatial-aware Prompt --- #
    spatial_prompt: bool = False  # add spatial gate to PromptGenBlock

    # --- Optimization 2: Enhanced Transformer Block --- #
    large_kernel: bool = False   # use larger dwconv instead of 3x3
    large_kernel_size: int = 7   # kernel size when large_kernel=True
    simple_gate: bool = False    # use SimpleGate instead of GELU in GDFN

    # --- C2: Decoder Dropout --- #
    decoder_dropout: float = 0.0  # dropout rate in decoder blocks (e.g. 0.05)

    # --- C3: Skip Attention --- #
    skip_attention: bool = False  # channel attention on skip connections


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    # Optimizer
    lr: float = 2e-4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"  # "adamw" or "adam"

    # Scheduler
    warmup_epochs: int = 15
    max_epochs: int = 150
    eta_min: float = 1e-6

    # Data
    batch_size: int = 8
    patch_size: int = 128
    num_workers: int = 4

    # Augmentation
    use_augmentation: bool = True  # random flip + rotation
    channel_shuffle: bool = False  # A1: randomly permute RGB channels
    cutmix: bool = False           # A2: CutMix between same-type samples
    cutmix_alpha: float = 1.0      # Beta distribution param for CutMix
    random_grayscale: float = 0.0  # A3: probability of converting to grayscale (e.g. 0.1)

    # Validation
    val_ratio: float = 0.05  # fraction of training data for validation
    val_every_n_epochs: int = 1

    # Loss
    loss_type: str = "l1"  # "l1", "charbonnier", "l1_ssim"
    charbonnier_eps: float = 1e-3
    ssim_weight: float = 0.1  # weight for SSIM when loss_type="l1_ssim"

    # --- B3: Masked Loss Weighting --- #
    masked_loss: bool = False        # weight loss by per-pixel degradation severity
    masked_loss_min: float = 0.2     # minimum weight for clean pixels

    # --- Optimization 3: Auxiliary losses --- #
    fft_loss: bool = False       # add frequency domain loss
    fft_loss_weight: float = 0.05
    edge_loss: bool = False      # add Sobel edge loss
    edge_loss_weight: float = 0.05

    # --- Optimization 5: Type-aware loss balancing --- #
    type_balanced_loss: bool = False  # up-weight snow to balance rain/snow
    snow_loss_weight: float = 3.0    # multiplier for snow samples

    # --- EMA (Exponential Moving Average) --- #
    ema: bool = False          # maintain EMA shadow weights
    ema_decay: float = 0.999   # EMA decay rate (0.999 typical for ~150 epochs)
    ema_start_epoch: int = 0   # start EMA after this epoch (0 = from beginning)

    # --- Data loading --- #
    cache: bool = False  # pre-load all images into RAM

    # --- Optimization 4: Progressive Learning --- #
    progressive: bool = False    # enable progressive patch size
    progressive_milestones: List[int] = field(
        default_factory=lambda: [0, 50, 100],  # epoch milestones
    )
    progressive_patch_sizes: List[int] = field(
        default_factory=lambda: [96, 128, 192],  # corresponding patch sizes
    )
    progressive_batch_sizes: List[int] = field(
        default_factory=lambda: [24, 16, 6],  # corresponding batch sizes (tuned for 5090 32GB)
    )


@dataclass
class DataConfig:
    """Dataset paths."""

    train_degraded_dir: str = "data/train/degraded"
    train_clean_dir: str = "data/train/clean"
    test_degraded_dir: str = "data/test/degraded"


@dataclass
class Config:
    """Top-level configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # Infrastructure
    seed: int = 42
    num_gpus: int = 1
    precision: str = "16-mixed"  # "32", "16-mixed", "bf16-mixed"
    # default; train.py uses checkpoints/{YYYYMMDD_HHMMSS}/ under cwd unless --ckpt_dir
    ckpt_dir: str = "checkpoints"
    ckpt_path: Optional[str] = None  # resume from checkpoint
    output_npz: str = "pred.npz"

    # Logging
    logger: str = "wandb"  # "tensorboard", "wandb", or "none"
    wandb_project: str = "promptir-hw4"
    log_dir: str = "logs"


def get_config() -> Config:
    """Create default config. Modify fields as needed before use."""
    return Config()
