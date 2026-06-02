"""Inference script — produce pred.npz for CodaBench submission.

Usage
-----
    python inference.py --ckpt_path checkpoints/20260123_120000/last.ckpt

    # With TTA (8-fold geometric self-ensemble, typically +0.2~0.5 dB)
    python inference.py --ckpt_path checkpoints/last.ckpt --tta

    # Custom test dir / output path
    python inference.py --ckpt_path checkpoints/20260123_120000/last.ckpt \
                        --test_dir data/test/degraded \
                        --output pred.npz

    # Create submission zip
    python inference.py --ckpt_path checkpoints/last.ckpt --tta --zip
"""

import argparse
import os
import zipfile

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import get_config, Config, ModelConfig, TrainConfig, DataConfig
from model import PromptIR
from dataset import TestDataset

import lightning.pytorch as pl


# --------------------------------------------------------------------------- #
#  Lightning Module (inference only — mirrors train.py)
# --------------------------------------------------------------------------- #

class PromptIRModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.net = PromptIR.from_config(cfg.model)

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------- #
#  Test-Time Augmentation (TTA) — 8-fold geometric self-ensemble
# --------------------------------------------------------------------------- #

def _apply_augment(x, mode):
    """Apply one of 8 geometric transforms to a (B, C, H, W) tensor."""
    if mode == 0:
        return x
    elif mode == 1:  # h-flip
        return x.flip(3)
    elif mode == 2:  # v-flip
        return x.flip(2)
    elif mode == 3:  # rot90
        return x.transpose(2, 3).flip(3)
    elif mode == 4:  # rot90 + h-flip
        return x.transpose(2, 3)
    elif mode == 5:  # rot180
        return x.flip(2).flip(3)
    elif mode == 6:  # rot270
        return x.transpose(2, 3).flip(2)
    elif mode == 7:  # rot270 + h-flip
        return x.flip(2).transpose(2, 3)
    return x


def _apply_augment_inv(x, mode):
    """Inverse of _apply_augment — undo the geometric transform."""
    if mode == 0:
        return x
    elif mode == 1:  # inv h-flip
        return x.flip(3)
    elif mode == 2:  # inv v-flip
        return x.flip(2)
    elif mode == 3:  # inv rot90 = rot270
        return x.flip(3).transpose(2, 3)
    elif mode == 4:  # inv (rot90 + h-flip) = transpose
        return x.transpose(2, 3)
    elif mode == 5:  # inv rot180
        return x.flip(3).flip(2)
    elif mode == 6:  # inv rot270 = rot90
        return x.flip(2).transpose(2, 3)
    elif mode == 7:  # inv (rot270 + h-flip)
        return x.transpose(2, 3).flip(2)
    return x


def tta_inference(model, img_tensor):
    """Run 8-fold geometric TTA and average the results."""
    accum = torch.zeros_like(img_tensor)
    for mode in range(8):
        aug_input = _apply_augment(img_tensor, mode)
        aug_output = model(aug_input)
        accum += _apply_augment_inv(aug_output, mode)
    return accum / 8.0


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(description="PromptIR Inference")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help=(
            "Path to .ckpt (e.g. checkpoints/<timestamp>/last.ckpt or best epoch file)"
        ),
    )
    parser.add_argument("--test_dir", type=str, default=None,
                        help="Path to test degraded images")
    parser.add_argument("--output", type=str, default=None,
                        help="Output .npz file path")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--zip", action="store_true",
                        help="Also create a zip file for submission")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: 'cuda', 'cpu', or 'cuda:0'")

    # TTA
    parser.add_argument("--tta", action="store_true",
                        help="Enable 8-fold geometric test-time augmentation")

    # EMA
    parser.add_argument("--ema", action="store_true",
                        help="Load EMA weights (ema.ckpt) instead of regular checkpoint")

    # Architecture flags — must match training config
    parser.add_argument("--spatial_prompt", action="store_true")
    parser.add_argument("--large_kernel", action="store_true")
    parser.add_argument("--large_kernel_size", type=int, default=None)
    parser.add_argument("--simple_gate", action="store_true")
    parser.add_argument("--decoder_dropout", type=float, default=None)
    parser.add_argument("--skip_attention", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = get_config()

    # Apply architecture overrides to match training config
    if args.spatial_prompt:
        cfg.model.spatial_prompt = True
    if args.large_kernel:
        cfg.model.large_kernel = True
    if args.large_kernel_size is not None:
        cfg.model.large_kernel_size = args.large_kernel_size
    if args.simple_gate:
        cfg.model.simple_gate = True
    if args.decoder_dropout is not None:
        cfg.model.decoder_dropout = args.decoder_dropout
    if args.skip_attention:
        cfg.model.skip_attention = True

    test_dir = args.test_dir or cfg.data.test_degraded_dir
    output_npz = args.output or cfg.output_npz
    use_ema = args.ema

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve checkpoint path — if --ema, look for ema.ckpt in same directory
    ckpt_path = os.path.abspath(args.ckpt_path)
    if use_ema:
        ema_path = os.path.join(os.path.dirname(ckpt_path), "ema.ckpt")
        if os.path.isfile(ema_path):
            ckpt_path = ema_path
            print(f"[EMA] Using EMA weights: {ema_path}")
        else:
            raise FileNotFoundError(
                f"EMA checkpoint not found: {ema_path}\n"
                "Train with --ema to generate ema.ckpt."
            )
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Hint: training saves under ./checkpoints/<YYYYMMDD_HHMMSS>/; "
            "pass that folder's last.ckpt or the best epoch=*.ckpt file."
        )
    ckpt_size = os.path.getsize(ckpt_path)
    if ckpt_size < 1024:
        raise ValueError(
            f"Checkpoint file is too small ({ckpt_size} bytes), "
            f"likely empty or truncated: {ckpt_path}\n"
            "If training was interrupted during save, use an older .ckpt or re-train."
        )

    use_tta = args.tta

    print(f"Loading checkpoint: {ckpt_path} ({ckpt_size / 1e6:.1f} MB)")
    print(f"Test directory: {test_dir}")
    print(f"Output: {output_npz}")
    print(f"Device: {device}")
    print(f"TTA: {'ON (8-fold)' if use_tta else 'OFF'}")
    print(f"EMA: {'ON' if use_ema else 'OFF'}")

    # Allow custom config classes when loading checkpoint (PyTorch 2.6+ safe unpickler)
    torch.serialization.add_safe_globals([Config, ModelConfig, TrainConfig, DataConfig])

    if use_ema:
        # EMA checkpoint is a pure state_dict saved by EMACallback
        model = PromptIRModel(cfg)
        ema_state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.net.load_state_dict(ema_state, strict=False)
        print(f"[EMA] Loaded EMA state dict ({len(ema_state)} keys)")
    else:
        try:
            model = PromptIRModel.load_from_checkpoint(
                ckpt_path,
                cfg=cfg,
                strict=False,
                map_location=device,
                weights_only=False,
            )
        except TypeError:
            # Older Lightning without ``weights_only`` on load_from_checkpoint
            model = PromptIRModel.load_from_checkpoint(
                ckpt_path,
                cfg=cfg,
                strict=False,
                map_location=device,
            )
    model = model.to(device)
    model.eval()

    # Dataset
    test_dataset = TestDataset(test_dir)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Inference
    images_dict = {}
    desc = "Inference (TTA×8)" if use_tta else "Inference"

    with torch.no_grad():
        for img_tensor, filenames in tqdm(test_loader, desc=desc):
            img_tensor = img_tensor.to(device)
            if use_tta:
                restored = tta_inference(model, img_tensor)
            else:
                restored = model(img_tensor)
            restored = torch.clamp(restored, 0.0, 1.0)

            # Convert to uint8 numpy, shape: (3, H, W) as required by spec
            restored_np = (restored * 255.0).round().byte().cpu().numpy()

            for i, fname in enumerate(filenames):
                images_dict[fname] = restored_np[i]  # shape: (3, H, W), uint8

    # Save npz
    np.savez(output_npz, **images_dict)
    print(f"\nSaved {len(images_dict)} images to {output_npz}")

    # Optional: create submission zip
    if args.zip:
        zip_name = output_npz.replace(".npz", ".zip")
        with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(output_npz, "pred.npz")
        print(f"Submission zip created: {zip_name}")


if __name__ == "__main__":
    main()
