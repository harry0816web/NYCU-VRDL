"""
Centralized configuration for training/inference scripts.

All values can still be overridden via CLI arguments in train.py.
"""

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_CONFIG = {
    "data_root": "data",
    "val_ratio": 0.15,
    "seed": 42,
}

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_CONFIG = {
    "num_classes": 5,
    "multi_scale_min_sizes": (640, 672, 704, 736, 768, 800),
    "multi_scale_max_size": 1024,
}

# ── Training ──────────────────────────────────────────────────────────────────
TRAIN_CONFIG = {
    "epochs": 48,
    "batch_size": 1,
    "grad_accum_steps": 2,   # 等效 batch_size = batch_size × grad_accum_steps
    "lr": 0.005,
    "momentum": 0.9,
    "weight_decay": 0.001,
    "warmup_epochs": 3,
    "lr_step_size": 20,
    "lr_gamma": 0.1,
    "num_workers": 0,
    "pin_memory": False,
    "persistent_workers": False,
    "prefetch_factor": 2,
    "max_grad_norm": 5.0,
    "use_amp": True,
}

# ── Memory Safety (Dense instances) ───────────────────────────────────────────
MEMORY_CONFIG = {
    # RTX 5090（32GB）可直接訓練整張大圖，消除 train/test domain gap
    "enable_dense_tiling": False,
    "dense_instance_threshold": 100,
    "tile_size": 512,
    "tile_overlap": 128,
    "max_instances_per_tile": 192,
    "min_instances_per_tile": 1,
}

# ── Evaluation ────────────────────────────────────────────────────────────────
EVAL_CONFIG = {
    "eval_interval": 2,
}

# ── Output / Logging ──────────────────────────────────────────────────────────
LOG_CONFIG = {
    "output_dir": "checkpoints",
    "wandb_project": "hw3-cell-segmentation",
    "run_name": None,
    "no_wandb": False,
}
