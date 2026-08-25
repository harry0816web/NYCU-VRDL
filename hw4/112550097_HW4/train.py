"""Training script for PromptIR image restoration (HW4).

Usage
-----
    # Train with default config
    python train.py

    # Override hyperparameters via CLI
    python train.py --epochs 200 --lr 1e-4 --batch_size 4 --patch_size 256

    # Resume from checkpoint (path is independent of per-run checkpoints/ dir)
    python train.py --ckpt_path checkpoints/20260123_120000/last.ckpt
"""

import argparse
import math
import os
import warnings
from contextlib import nullcontext
from datetime import datetime
from typing import List

from dotenv import load_dotenv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

from config import get_config
from model import PromptIR
from dataset import TrainDataset, ValDataset, train_val_split


# --------------------------------------------------------------------------- #
#  Learning rate scheduler
# --------------------------------------------------------------------------- #

class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """Linear warmup followed by cosine annealing."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        if self.last_epoch < self.warmup_epochs:
            return [
                group["lr"]
                + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        if self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        if (self.last_epoch - 1 - self.max_epochs) % (
            2 * (self.max_epochs - self.warmup_epochs)
        ) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min)
                * (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs)))
                / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        return [
            (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            / (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs - 1)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]


# --------------------------------------------------------------------------- #
#  Loss functions
# --------------------------------------------------------------------------- #

class CharbonnierLoss(nn.Module):
    """Charbonnier loss (a smooth variant of L1)."""

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps2))


class L1SSIMLoss(nn.Module):
    """Combined L1 + SSIM loss."""

    def __init__(self, ssim_weight=0.1):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.ssim_weight = ssim_weight

    def forward(self, pred, target):
        l1_loss = self.l1(pred, target)
        ssim_val = self._ssim(pred, target)
        return l1_loss + self.ssim_weight * (1.0 - ssim_val)

    @staticmethod
    def _ssim(img1, img2, window_size=11):
        """Simplified SSIM on channel-averaged images."""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        pool = torch.nn.functional.avg_pool2d
        pad = window_size // 2
        mu1 = pool(img1, window_size, stride=1, padding=pad)
        mu2 = pool(img2, window_size, stride=1, padding=pad)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = pool(img1 * img1, window_size, stride=1, padding=pad) - mu1_sq
        sigma2_sq = pool(img2 * img2, window_size, stride=1, padding=pad) - mu2_sq
        sigma12 = pool(img1 * img2, window_size, stride=1, padding=pad) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )
        return ssim_map.mean()


class FFTLoss(nn.Module):
    """頻域損失 (Frequency Domain Loss)"""

    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()

    def forward(self, pred, target):
        # Mixed precision + rfft2 on half-width tensors can hit CUDA driver bugs;
        # run FFT in FP32 outside autocast.
        p, t = pred.float(), target.float()
        amp_ctx = (
            torch.amp.autocast("cuda", enabled=False)
            if pred.is_cuda
            else nullcontext()
        )
        with amp_ctx:
            # norm="ortho" divides by sqrt(H*W), keeping magnitudes in a
            # comparable range to pixel-space losses (~0.x).  The original
            # "backward" norm left them at H*W scale (~16 384×), which
            # dwarfed L1 and caused NaN after ~25 k steps.
            pred_fft = torch.fft.rfft2(p, norm="ortho")
            target_fft = torch.fft.rfft2(t, norm="ortho")
            diff = pred_fft - target_fft
            # torch.abs(complex) can hit buggy CUDA kernels; |z| from real/imag is stable.
            mag = torch.hypot(diff.real, diff.imag)
        return mag.mean()


class EdgeLoss(nn.Module):
    """邊緣增強損失 (Edge/Gradient Loss using Sobel Filters)"""

    def __init__(self):
        super().__init__()
        # 定義水平與垂直的 Sobel Filter
        k_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        k_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

        # 擴展維度以適應 Conv2d (out_channels, in_channels/groups, kH, kW)
        k_x = k_x.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
        k_y = k_y.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)

        self.weight_x = nn.Parameter(k_x, requires_grad=False)
        self.weight_y = nn.Parameter(k_y, requires_grad=False)
        self.criterion = nn.L1Loss()

    def forward(self, pred, target):
        # Sobel weights are FP32; match activations to avoid conv dtype issues under AMP.
        p, t = pred.float(), target.float()
        # 對 3 個 Channel 分別做 convolution (groups=3)
        pred_grad_x = F.conv2d(p, self.weight_x, padding=1, groups=3)
        pred_grad_y = F.conv2d(p, self.weight_y, padding=1, groups=3)
        target_grad_x = F.conv2d(t, self.weight_x, padding=1, groups=3)
        target_grad_y = F.conv2d(t, self.weight_y, padding=1, groups=3)

        loss_x = self.criterion(pred_grad_x, target_grad_x)
        loss_y = self.criterion(pred_grad_y, target_grad_y)
        return loss_x + loss_y


class CompositeLoss(nn.Module):
    """Modular composite loss: base_loss + optional FFT + optional Edge.

    The auxiliary losses (FFT, Edge) can be combined with ANY base loss
    via the --fft_loss and --edge_loss CLI flags.
    """

    def __init__(self, base_loss, fft_weight=0.0, edge_weight=0.0):
        super().__init__()
        self.base_loss = base_loss
        self.fft_weight = fft_weight
        self.edge_weight = edge_weight
        if fft_weight > 0:
            self.fft_loss = FFTLoss()
        if edge_weight > 0:
            self.edge_loss = EdgeLoss()

    def forward(self, pred, target):
        loss = self.base_loss(pred, target)
        if self.fft_weight > 0:
            loss = loss + self.fft_weight * self.fft_loss(pred, target)
        if self.edge_weight > 0:
            loss = loss + self.edge_weight * self.edge_loss(pred, target)
        return loss


def build_loss(cfg):
    """Build loss function from config.

    Base loss is selected by ``cfg.loss_type``.  Auxiliary FFT / Edge losses
    are toggled independently via ``cfg.fft_loss`` and ``cfg.edge_loss``.
    """
    # Base loss
    if cfg.loss_type == "l1":
        base = nn.L1Loss()
    elif cfg.loss_type == "charbonnier":
        base = CharbonnierLoss(eps=cfg.charbonnier_eps)
    elif cfg.loss_type == "l1_ssim":
        base = L1SSIMLoss(ssim_weight=cfg.ssim_weight)
    else:
        raise ValueError(f"Unknown loss type: {cfg.loss_type}")

    # Wrap with auxiliary losses if enabled
    fft_w = cfg.fft_loss_weight if getattr(cfg, 'fft_loss', False) else 0.0
    edge_w = cfg.edge_loss_weight if getattr(cfg, 'edge_loss', False) else 0.0

    if fft_w > 0 or edge_w > 0:
        return CompositeLoss(base, fft_weight=fft_w, edge_weight=edge_w)
    return base


# --------------------------------------------------------------------------- #
#  PSNR metric
# --------------------------------------------------------------------------- #

def compute_psnr(restored, clean):
    """Compute PSNR between two batched tensors [0, 1]."""
    mse = torch.mean((restored - clean) ** 2, dim=[1, 2, 3])
    psnr = 10.0 * torch.log10(1.0 / (mse + 1e-10))
    return psnr.mean().item()


# --------------------------------------------------------------------------- #
#  Lightning Module
# --------------------------------------------------------------------------- #

class PromptIRModel(pl.LightningModule):
    """Lightning wrapper for PromptIR."""

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.net = PromptIR.from_config(cfg.model)
        self.loss_fn = build_loss(cfg.train)

        # Collect per-type PSNR during validation
        self._val_psnr_rain = []
        self._val_psnr_snow = []

    # ---- Dynamic dataloader (for progressive batch size) ---- #

    def setup_train_data(self, train_set, num_workers):
        """Store dataset reference so train_dataloader() can rebuild with new batch_size."""
        self._train_set = train_set
        self._num_workers = num_workers
        self._train_batch_size = self.cfg.train.batch_size

    def train_dataloader(self):
        return DataLoader(
            self._train_set,
            batch_size=self._train_batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def _per_pixel_loss(self, pred, target):
        """Compute per-pixel loss (no reduction) for masked loss weighting.

        Returns a tensor of shape (B, 1, H, W) representing per-pixel loss.
        Works with Charbonnier, L1, or L1+SSIM (falls back to L1 for the
        per-pixel component when using composite losses).
        """
        diff = pred - target
        loss_type = self.cfg.train.loss_type
        if loss_type == "charbonnier":
            eps2 = self.cfg.train.charbonnier_eps ** 2
            per_pixel = torch.sqrt(diff * diff + eps2)
        else:
            per_pixel = diff.abs()
        # Average over channels → (B, 1, H, W)
        return per_pixel.mean(dim=1, keepdim=True)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        degrad_patch, clean_patch, deg_types = batch
        restored = self.net(degrad_patch)

        tc = self.cfg.train

        # --- B3: Masked Loss Weighting --- #
        # Weight each pixel's loss by how degraded it is.
        # Heavy degradation → high weight; clean pixels → low weight (min_w).
        if tc.masked_loss:
            # Compute per-pixel residual magnitude as weight map
            with torch.no_grad():
                # (B,1,H,W)
                residual_map = (degrad_patch - clean_patch).abs().mean(
                    dim=1, keepdim=True
                )
                weight_map = residual_map / (residual_map.mean() + 1e-6)  # normalize
                # keep minimum weight for clean pixels
                weight_map = weight_map.clamp(min=tc.masked_loss_min)

            # Per-pixel loss (no reduction)
            loss_per_pixel = self._per_pixel_loss(restored, clean_patch)  # (B,C,H,W) or (B,1,H,W)
            loss = (loss_per_pixel * weight_map).mean()
        elif tc.type_balanced_loss:
            # Per-sample loss with type-aware weighting
            B = degrad_patch.size(0)
            loss = 0.0
            for i in range(B):
                sample_loss = self.loss_fn(
                    restored[i:i+1], clean_patch[i:i+1],
                )
                w = tc.snow_loss_weight if deg_types[i] == "snow" else 1.0
                loss = loss + w * sample_loss
            loss = loss / B
        else:
            loss = self.loss_fn(restored, clean_patch)

        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            sync_dist=True,
            batch_size=degrad_patch.size(0),
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        degrad_patch, clean_patch, deg_types = batch
        restored = self.net(degrad_patch)
        restored = torch.clamp(restored, 0.0, 1.0)

        loss = self.loss_fn(restored, clean_patch)

        # Per-sample PSNR
        mse = torch.mean((restored - clean_patch) ** 2, dim=[1, 2, 3])
        psnr_per_sample = 10.0 * torch.log10(1.0 / (mse + 1e-10))
        psnr_avg = psnr_per_sample.mean().item()

        # Accumulate per-type PSNR
        for i, dt in enumerate(deg_types):
            p = psnr_per_sample[i].item()
            if dt == "rain":
                self._val_psnr_rain.append(p)
            elif dt == "snow":
                self._val_psnr_snow.append(p)

        bs = degrad_patch.size(0)
        self.log(
            "val/loss",
            loss,
            prog_bar=True,
            sync_dist=True,
            batch_size=bs,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val/psnr",
            psnr_avg,
            prog_bar=True,
            sync_dist=True,
            batch_size=bs,
            on_step=False,
            on_epoch=True,
        )

        # Log sample images to wandb (first batch only)
        if batch_idx == 0 and self.logger and hasattr(self.logger, "experiment"):
            self._log_sample_images(degrad_patch, restored, clean_patch)

        return {"val_loss": loss, "val_psnr": psnr_avg}

    def on_validation_epoch_end(self):
        """Log per-type PSNR at the end of each validation epoch."""
        if self._val_psnr_rain:
            avg_rain = sum(self._val_psnr_rain) / len(self._val_psnr_rain)
            self.log("val/psnr_rain", avg_rain, sync_dist=True, on_epoch=True)
        if self._val_psnr_snow:
            avg_snow = sum(self._val_psnr_snow) / len(self._val_psnr_snow)
            self.log("val/psnr_snow", avg_snow, sync_dist=True, on_epoch=True)

        self._val_psnr_rain.clear()
        self._val_psnr_snow.clear()

    def _log_sample_images(self, degraded, restored, clean, max_images=4):
        """Log comparison images to wandb."""
        try:
            import wandb
            n = min(max_images, degraded.size(0))
            images = []
            for i in range(n):
                deg_img = degraded[i].cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                res_img = restored[i].cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                cln_img = clean[i].cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                images.append(wandb.Image(
                    np.concatenate([deg_img, res_img, cln_img], axis=1),
                    caption=f"Sample {i}: Degraded | Restored | Clean",
                ))
            self.logger.experiment.log(
                {"val/samples": images, "epoch": self.current_epoch},
            )
        except Exception:
            pass  # silently skip if wandb not available

    def lr_scheduler_step(self, scheduler, metric):
        # PyTorch 2.x: pass no epoch; closed-form epoch=... path is deprecated.
        scheduler.step()

    def configure_optimizers(self):
        tc = self.cfg.train
        if tc.optimizer == "adamw":
            optimizer = optim.AdamW(
                self.parameters(), lr=tc.lr, weight_decay=tc.weight_decay,
            )
        else:
            optimizer = optim.Adam(self.parameters(), lr=tc.lr)

        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_epochs=tc.warmup_epochs,
            max_epochs=tc.max_epochs,
            eta_min=tc.eta_min,
        )
        return [optimizer], [scheduler]


# --------------------------------------------------------------------------- #
#  Progressive Learning Callback
# --------------------------------------------------------------------------- #

class ProgressiveLearningCallback(pl.Callback):
    """Gradually increase patch size and decrease batch size during training.

    At each milestone epoch, updates the dataset's ``patch_size`` attribute
    directly.  Because the dataset is shared between the DataLoader and
    this callback, any worker process that lazily reads ``self.patch_size``
    will pick up the new value automatically (persistent workers may cache
    the old value for one epoch at most, which is acceptable).

    Batch-size changes are handled by storing the schedule on the
    LightningModule and overriding ``train_dataloader`` — see
    ``PromptIRModel.train_dataloader``.

    Args:
        dataset: The underlying TrainDataset (must have mutable ``patch_size``).
        milestones: List of epoch numbers where changes happen.
        patch_sizes: Corresponding patch sizes for each milestone.
        batch_sizes: Corresponding batch sizes for each milestone.
    """

    def __init__(self, dataset, milestones, patch_sizes, batch_sizes):
        super().__init__()
        self.dataset = dataset
        self.milestones = milestones
        self.patch_sizes = patch_sizes
        self.batch_sizes = batch_sizes
        self._current_stage = -1

    @staticmethod
    def _reload_train_dataloader(trainer, pl_module):
        """Rebuild the train DataLoader mid-training (compatible across Lightning 2.x)."""
        new_dl = pl_module.train_dataloader()
        try:
            from lightning.pytorch.utilities.combined_loader import CombinedLoader
            combined = CombinedLoader(new_dl, mode="min_size")
            trainer.fit_loop._combined_loader = combined
            # setup_data() will call iter() internally and set num_training_batches
            trainer.fit_loop.setup_data()
        except Exception:
            # Fallback: just replace the data source; Lightning will re-iter at epoch start
            try:
                trainer.fit_loop._data_source.instance = pl_module
            except AttributeError:
                pass

    def _get_stage(self, epoch):
        """Find which stage we're in based on current epoch."""
        stage = 0
        for i, m in enumerate(self.milestones):
            if epoch >= m:
                stage = i
        return stage

    def on_train_epoch_start(self, trainer, pl_module):
        stage = self._get_stage(trainer.current_epoch)
        if stage != self._current_stage:
            self._current_stage = stage
            new_ps = self.patch_sizes[stage]
            new_bs = self.batch_sizes[stage]

            # Update dataset patch_size (workers see this on next __getitem__)
            self.dataset.patch_size = new_ps

            # Update batch size and rebuild the DataLoader
            if hasattr(pl_module, '_train_batch_size'):
                old_bs = pl_module._train_batch_size
                pl_module._train_batch_size = new_bs
                if old_bs != new_bs:
                    self._reload_train_dataloader(trainer, pl_module)

            print(f"\n[Progressive] Epoch {trainer.current_epoch}: "
                  f"patch_size={new_ps}, batch_size={new_bs}")

            if pl_module.logger:
                pl_module.log("train/patch_size", float(new_ps))
                pl_module.log("train/batch_size", float(new_bs))


# --------------------------------------------------------------------------- #
#  EMA (Exponential Moving Average) Callback
# --------------------------------------------------------------------------- #

class EMACallback(pl.Callback):
    """Maintain an exponential moving average of model weights.

    - Shadow weights are updated after every training step.
    - Before validation the shadow weights are swapped in so that
      val metrics reflect EMA performance.
    - After validation the original (online) weights are restored so
      training continues unaffected.
    - At every checkpoint save, an additional ``ema.ckpt`` is written
      containing *only* the EMA state dict (ready for inference).

    Args:
        decay: EMA coefficient.  shadow ← decay * shadow + (1-decay) * param.
        start_epoch: Don't accumulate EMA before this epoch (avoids
                     contaminating the average with early unstable weights).
    """

    def __init__(self, decay: float = 0.999, start_epoch: int = 0):
        super().__init__()
        self.decay = decay
        self.start_epoch = start_epoch
        self.shadow: dict = {}      # EMA parameters (detached clones)
        self.backup: dict = {}      # online weights stashed during val
        self._active = False        # True once we've started collecting

    # ---- lifecycle -------------------------------------------------------- #

    def _init_shadow(self, pl_module):
        """Copy current params as the initial shadow state."""
        self.shadow = {
            name: p.data.clone()
            for name, p in pl_module.net.named_parameters()
            if p.requires_grad
        }
        self._active = True
        print(f"[EMA] Initialized shadow weights "
              f"({len(self.shadow)} params, decay={self.decay})")

    @torch.no_grad()
    def _update(self, pl_module):
        """One EMA update step."""
        d = self.decay
        for name, p in pl_module.net.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.shadow[name].mul_(d).add_(p.data, alpha=1.0 - d)

    def _swap_to_ema(self, pl_module):
        """Replace online weights with EMA weights (stash originals)."""
        self.backup = {}
        for name, p in pl_module.net.named_parameters():
            if name in self.shadow:
                self.backup[name] = p.data.clone()
                p.data.copy_(self.shadow[name])

    def _swap_to_online(self, pl_module):
        """Restore online weights after validation."""
        for name, p in pl_module.net.named_parameters():
            if name in self.backup:
                p.data.copy_(self.backup[name])
        self.backup = {}

    # ---- hooks ----------------------------------------------------------- #

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self._active:
            if trainer.current_epoch >= self.start_epoch:
                self._init_shadow(pl_module)
            else:
                return
        self._update(pl_module)

    def on_validation_epoch_start(self, trainer, pl_module):
        if self._active:
            self._swap_to_ema(pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        if self._active and self.backup:
            # Save EMA checkpoint before swapping back
            self._save_ema_ckpt(trainer, pl_module)
            self._swap_to_online(pl_module)

    def _save_ema_ckpt(self, trainer, pl_module):
        """Save EMA weights as a standalone state dict."""
        ckpt_dir = trainer.checkpoint_callback.dirpath if trainer.checkpoint_callback else None
        if ckpt_dir is None:
            return
        ema_path = os.path.join(ckpt_dir, "ema.ckpt")
        # Save the net's state dict (currently holding EMA weights)
        torch.save(pl_module.net.state_dict(), ema_path)


# --------------------------------------------------------------------------- #
#  CLI + main
# --------------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(description="Train PromptIR")

    # Frequently tuned
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--num_gpus", type=int, default=None)
    parser.add_argument("--precision", type=str, default=None)
    parser.add_argument("--loss_type", type=str, default=None,
                        choices=["l1", "charbonnier", "l1_ssim"])
    parser.add_argument("--val_ratio", type=float, default=None)

    # --- Optimization 1: Spatial-aware Prompt --- #
    parser.add_argument("--spatial_prompt", action="store_true",
                        help="Enable spatial gate in PromptGenBlock")

    # --- Optimization 2: Enhanced Transformer Block --- #
    parser.add_argument("--large_kernel", action="store_true",
                        help="Use larger depthwise conv (default 7x7) in Transformer blocks")
    parser.add_argument("--large_kernel_size", type=int, default=None,
                        help="Kernel size when --large_kernel is set (default: 7)")
    parser.add_argument("--simple_gate", action="store_true",
                        help="Use SimpleGate (NAFNet) instead of GELU in GDFN")

    # --- Optimization 3: Auxiliary losses --- #
    parser.add_argument("--fft_loss", action="store_true",
                        help="Add frequency domain (FFT) loss")
    parser.add_argument("--fft_loss_weight", type=float, default=None,
                        help="Weight for FFT loss (default: 0.05)")
    parser.add_argument("--edge_loss", action="store_true",
                        help="Add Sobel edge loss")
    parser.add_argument("--edge_loss_weight", type=float, default=None,
                        help="Weight for edge loss (default: 0.05)")

    # --- Optimization 4: Progressive Learning --- #
    parser.add_argument("--progressive", action="store_true",
                        help="Enable progressive patch size training")

    # --- Optimization 5: Type-aware loss balancing --- #
    parser.add_argument("--type_balanced_loss", action="store_true",
                        help="Up-weight snow samples to balance rain/snow loss")
    parser.add_argument("--snow_loss_weight", type=float, default=None,
                        help="Loss multiplier for snow samples (default: 3.0)")

    # --- Augmentation enhancements (方案 A) --- #
    parser.add_argument("--channel_shuffle", action="store_true",
                        help="A1: Randomly permute RGB channels")
    parser.add_argument("--cutmix", action="store_true",
                        help="A2: CutMix between same-type samples")
    parser.add_argument("--cutmix_alpha", type=float, default=None,
                        help="Beta distribution alpha for CutMix (default: 1.0)")
    parser.add_argument("--random_grayscale", type=float, default=None,
                        help="A3: Probability of random grayscale (e.g. 0.1)")

    # --- B3: Masked Loss Weighting --- #
    parser.add_argument("--masked_loss", action="store_true",
                        help="Weight loss by per-pixel degradation severity")
    parser.add_argument("--masked_loss_min", type=float, default=None,
                        help="Minimum weight for clean pixels (default: 0.2)")

    # --- C2: Decoder Dropout --- #
    parser.add_argument("--decoder_dropout", type=float, default=None,
                        help="Dropout rate in decoder blocks (e.g. 0.05)")

    # --- C3: Skip Attention --- #
    parser.add_argument("--skip_attention", action="store_true",
                        help="Channel attention on encoder skip connections")

    # --- EMA --- #
    parser.add_argument("--ema", action="store_true",
                        help="Enable Exponential Moving Average of model weights")
    parser.add_argument("--ema_decay", type=float, default=None,
                        help="EMA decay rate (default: 0.999)")
    parser.add_argument("--ema_start_epoch", type=int, default=None,
                        help="Start EMA after this epoch (default: 0)")

    # --- Data loading --- #
    parser.add_argument("--cache", action="store_true",
                        help="Pre-load all images into RAM (~600MB)")

    # Paths
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--train_degraded_dir", type=str, default=None)
    parser.add_argument("--train_clean_dir", type=str, default=None)

    # Logging
    parser.add_argument("--logger", type=str, default=None,
                        choices=["tensorboard", "wandb", "none"])
    parser.add_argument("--wandb_project", type=str, default=None)

    # Seed
    parser.add_argument("--seed", type=int, default=None)

    return parser.parse_args()


def apply_overrides(cfg, args):
    """Apply CLI overrides to config."""
    if args.epochs is not None:
        cfg.train.max_epochs = args.epochs
    if args.lr is not None:
        cfg.train.lr = args.lr
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.patch_size is not None:
        cfg.train.patch_size = args.patch_size
    if args.num_workers is not None:
        cfg.train.num_workers = args.num_workers
    if args.num_gpus is not None:
        cfg.num_gpus = args.num_gpus
    if args.precision is not None:
        cfg.precision = args.precision
    if args.loss_type is not None:
        cfg.train.loss_type = args.loss_type
    if args.val_ratio is not None:
        cfg.train.val_ratio = args.val_ratio

    # Optimization 1: Spatial prompt
    if args.spatial_prompt:
        cfg.model.spatial_prompt = True

    # Optimization 2: Enhanced Transformer
    if args.large_kernel:
        cfg.model.large_kernel = True
    if args.large_kernel_size is not None:
        cfg.model.large_kernel_size = args.large_kernel_size
    if args.simple_gate:
        cfg.model.simple_gate = True

    # Optimization 3: Auxiliary losses
    if args.fft_loss:
        cfg.train.fft_loss = True
    if args.fft_loss_weight is not None:
        cfg.train.fft_loss_weight = args.fft_loss_weight
    if args.edge_loss:
        cfg.train.edge_loss = True
    if args.edge_loss_weight is not None:
        cfg.train.edge_loss_weight = args.edge_loss_weight

    # Optimization 4: Progressive learning
    if args.progressive:
        cfg.train.progressive = True

    # Optimization 5: Type-balanced loss
    if args.type_balanced_loss:
        cfg.train.type_balanced_loss = True
    if args.snow_loss_weight is not None:
        cfg.train.snow_loss_weight = args.snow_loss_weight

    # Augmentation enhancements (方案 A)
    if args.channel_shuffle:
        cfg.train.channel_shuffle = True
    if args.cutmix:
        cfg.train.cutmix = True
    if args.cutmix_alpha is not None:
        cfg.train.cutmix_alpha = args.cutmix_alpha
    if args.random_grayscale is not None:
        cfg.train.random_grayscale = args.random_grayscale

    # B3: Masked loss weighting
    if args.masked_loss:
        cfg.train.masked_loss = True
    if args.masked_loss_min is not None:
        cfg.train.masked_loss_min = args.masked_loss_min

    # C2: Decoder dropout
    if args.decoder_dropout is not None:
        cfg.model.decoder_dropout = args.decoder_dropout

    # C3: Skip attention
    if args.skip_attention:
        cfg.model.skip_attention = True

    # EMA
    if args.ema:
        cfg.train.ema = True
    if args.ema_decay is not None:
        cfg.train.ema_decay = args.ema_decay
    if args.ema_start_epoch is not None:
        cfg.train.ema_start_epoch = args.ema_start_epoch

    # Data loading
    if args.cache:
        cfg.train.cache = True

    # Paths & misc
    if args.ckpt_path is not None:
        cfg.ckpt_path = args.ckpt_path
    if args.ckpt_dir is not None:
        cfg.ckpt_dir = args.ckpt_dir
    if args.train_degraded_dir is not None:
        cfg.data.train_degraded_dir = args.train_degraded_dir
    if args.train_clean_dir is not None:
        cfg.data.train_clean_dir = args.train_clean_dir
    if args.logger is not None:
        cfg.logger = args.logger
    if args.wandb_project is not None:
        cfg.wandb_project = args.wandb_project
    if args.seed is not None:
        cfg.seed = args.seed
    return cfg


def main():
    # Load .env (e.g. WANDB_API_KEY) before WandbLogger or wandb init
    load_dotenv()

    args = parse_args()
    cfg = get_config()
    cfg = apply_overrides(cfg, args)

    # Checkpoints: ./checkpoints/{YYYYMMDD_HHMMSS}/ unless --ckpt_dir is set
    if args.ckpt_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg.ckpt_dir = os.path.join(os.getcwd(), "checkpoints", stamp)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    pl.seed_everything(cfg.seed, workers=True)

    # Tensor Cores optimization (RTX 30xx/40xx/50xx)
    torch.set_float32_matmul_precision("high")

    # Print config
    print("=" * 60)
    print("Config:")
    print(f"  Model dim={cfg.model.dim}, blocks={cfg.model.num_blocks}")
    print(f"  Train lr={cfg.train.lr}, epochs={cfg.train.max_epochs}, "
          f"batch={cfg.train.batch_size}, patch={cfg.train.patch_size}")
    print(f"  Loss: {cfg.train.loss_type}"
          f"{' +FFT' if cfg.train.fft_loss else ''}"
          f"{' +Edge' if cfg.train.edge_loss else ''}"
          f"{' +MaskedLoss' if cfg.train.masked_loss else ''}")
    print(f"  Spatial prompt: {cfg.model.spatial_prompt}")
    aug_extras = []
    if cfg.train.channel_shuffle:
        aug_extras.append("ChannelShuffle")
    if cfg.train.cutmix:
        aug_extras.append(f"CutMix(α={cfg.train.cutmix_alpha})")
    if cfg.train.random_grayscale > 0:
        aug_extras.append(f"Grayscale(p={cfg.train.random_grayscale})")
    if aug_extras:
        print(f"  Extra augmentation: {', '.join(aug_extras)}")
    print(f"  Decoder dropout: {cfg.model.decoder_dropout}")
    print(f"  Skip attention: {cfg.model.skip_attention}")
    lk_info = f" (k={cfg.model.large_kernel_size})" if cfg.model.large_kernel else ""
    print(f"  Large kernel: {cfg.model.large_kernel}{lk_info}")
    print(f"  Simple gate: {cfg.model.simple_gate}")
    print(f"  Progressive: {cfg.train.progressive}")
    bal_info = f" (snow×{cfg.train.snow_loss_weight})" if cfg.train.type_balanced_loss else ""
    print(f"  Type-balanced loss: {cfg.train.type_balanced_loss}{bal_info}")
    ema_decay = cfg.train.ema_decay
    ema_start = cfg.train.ema_start_epoch
    ema_info = (
        f" (decay={ema_decay}, start={ema_start})" if cfg.train.ema else ""
    )
    print(f"  EMA: {cfg.train.ema}{ema_info}")
    print(f"  Cache: {cfg.train.cache}")
    print(f"  Precision: {cfg.precision}, GPUs: {cfg.num_gpus}")
    print(f"  Checkpoints: {cfg.ckpt_dir}")
    print("=" * 60)

    # Dataset
    full_dataset = TrainDataset(
        degraded_dir=cfg.data.train_degraded_dir,
        clean_dir=cfg.data.train_clean_dir,
        patch_size=cfg.train.patch_size,
        augment=cfg.train.use_augmentation,
        cache=cfg.train.cache,
        channel_shuffle=cfg.train.channel_shuffle,
        cutmix=cfg.train.cutmix,
        cutmix_alpha=cfg.train.cutmix_alpha,
        random_grayscale=cfg.train.random_grayscale,
    )

    val_set = None
    if cfg.train.val_ratio > 0:
        # Stratified split → separate ValDataset (no crop, no augment)
        from torch.utils.data import Subset
        train_indices, val_pairs, val_cache = train_val_split(
            full_dataset, val_ratio=cfg.train.val_ratio, seed=cfg.seed,
        )
        train_set = Subset(full_dataset, train_indices)
        val_set = ValDataset(val_pairs, cache_from=val_cache)
        print(f"Train: {len(train_set)}, Val: {len(val_set)}")
    else:
        train_set = full_dataset

    val_loader = None
    if val_set is not None:
        val_loader = DataLoader(
            val_set,
            batch_size=cfg.train.batch_size,
            shuffle=False,
            num_workers=cfg.train.num_workers,
            pin_memory=True,
        )

    # Model
    model = PromptIRModel(cfg)
    model.setup_train_data(train_set, cfg.train.num_workers)

    # Logger
    if cfg.logger == "wandb":
        logger = WandbLogger(project=cfg.wandb_project, name="PromptIR-Train")
    elif cfg.logger == "tensorboard":
        logger = TensorBoardLogger(save_dir=cfg.log_dir, name="promptir")
    else:
        logger = None

    # Callbacks: keep only the single best checkpoint by lowest val loss (plus last.ckpt)
    if val_set is not None:
        ckpt_monitor = "val/loss"
        ckpt_filename = "epoch={epoch:03d}-val_loss={val/loss:.4f}"
        save_on_train_epoch_end = False
    else:
        ckpt_monitor = "train/loss"
        ckpt_filename = "epoch={epoch:03d}-train_loss={train/loss:.4f}"
        save_on_train_epoch_end = True

    callbacks = [
        ModelCheckpoint(
            dirpath=cfg.ckpt_dir,
            filename=ckpt_filename,
            monitor=ckpt_monitor,
            mode="min",
            save_top_k=1,
            every_n_epochs=1,
            auto_insert_metric_name=False,
            save_last=True,
            save_on_train_epoch_end=save_on_train_epoch_end,
        ),
    ]
    if logger is not None:
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    # EMA callback
    if cfg.train.ema:
        ema_cb = EMACallback(
            decay=cfg.train.ema_decay,
            start_epoch=cfg.train.ema_start_epoch,
        )
        callbacks.append(ema_cb)

    # Progressive learning callback
    progressive_cb = None
    if cfg.train.progressive:
        progressive_cb = ProgressiveLearningCallback(
            dataset=full_dataset,
            milestones=cfg.train.progressive_milestones,
            patch_sizes=cfg.train.progressive_patch_sizes,
            batch_sizes=cfg.train.progressive_batch_sizes,
        )
        callbacks.append(progressive_cb)

    # Trainer (log_every_n_steps: high so loggers like wandb are not flooded per-step)
    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=cfg.num_gpus if torch.cuda.is_available() else 1,
        strategy=(
            "ddp_find_unused_parameters_true" if cfg.num_gpus > 1 else "auto"
        ),
        precision=cfg.precision,
        gradient_clip_val=1.0,        # safety net against loss spikes
        gradient_clip_algorithm="norm",
        logger=logger,
        callbacks=callbacks,
        check_val_every_n_epoch=cfg.train.val_every_n_epochs,
        log_every_n_steps=1_000_000,
    )

    # Train (train dataloader comes from model.train_dataloader() for dynamic batch size)
    trainer.fit(
        model=model,
        val_dataloaders=val_loader,
        ckpt_path=cfg.ckpt_path,
    )

    print(f"\nTraining complete. Checkpoints saved to: {cfg.ckpt_dir}/")


if __name__ == "__main__":
    main()
