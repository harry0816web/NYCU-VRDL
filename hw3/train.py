"""
train.py - Training script for Mask R-CNN cell instance segmentation

Features:
    - tqdm progress bars for training / evaluation / GT building
    - Wandb logging (reads WANDB_API_KEY from .env)
    - COCO-style mAP evaluation
    - Learning rate scheduler (StepLR)
    - Checkpoint saving (best + latest)
    - Gradient clipping
    - Memory-efficient COCO GT building

Usage:
    python train.py --data_root data --epochs 30 --batch_size 2 --lr 0.005
"""

import os
import sys
import gc
import time
import argparse
import datetime
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    DATA_CONFIG,
    MODEL_CONFIG,
    TRAIN_CONFIG,
    EVAL_CONFIG,
    LOG_CONFIG,
    MEMORY_CONFIG,
)
from model import get_baseline_model
from dataset import CellDataset, get_train_val_datasets, collate_fn

# ── COCO evaluation utilities ──
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils


def load_wandb_key():
    """Load WANDB_API_KEY from .env file if present."""
    env_path = ".env"
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("WANDB_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    os.environ["WANDB_API_KEY"] = key
                    print("Loaded WANDB_API_KEY from .env")
                    return True
    return False


def build_coco_gt(dataset):
    """
    Build a COCO-format ground truth object from the dataset.

    零記憶體解碼版：如果 dataset 是 CellDatasetCached，直接從 cache
    中的 RLE strings 構建 COCO GT，完全不需要解碼 mask 為 numpy array。
    這避免了密集圖片（700+ instances × 2000×2000）導致的數 GB 記憶體峰值。
    """
    from dataset import CellDatasetCached

    coco_dict = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "class1"},
            {"id": 2, "name": "class2"},
            {"id": 3, "name": "class3"},
            {"id": 4, "name": "class4"},
        ],
    }

    ann_id = 1

    # ── 快速路徑：直接從 cache RLE 構建（零 mask 解碼）──
    if isinstance(dataset, CellDatasetCached):
        pbar = tqdm(range(len(dataset)), desc="Building COCO GT (from cache)",
                    unit="img", leave=False)

        for idx in pbar:
            sample = dataset.samples[idx]
            h, w = sample["height"], sample["width"]
            image_id = idx + 1

            coco_dict["images"].append({
                "id": image_id, "height": h, "width": w,
            })

            for inst in sample["instances"]:
                rle = inst["rle"]  # 已經是 {"size": [H,W], "counts": str}
                area = float(mask_utils.area(rle))
                x_min, y_min, x_max, y_max = inst["bbox"]

                coco_dict["annotations"].append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": int(inst["label"]),
                    "segmentation": rle,
                    "area": area,
                    "bbox": [float(x_min), float(y_min),
                             float(x_max - x_min), float(y_max - y_min)],
                    "iscrowd": 0,
                })
                ann_id += 1

            pbar.set_postfix(annotations=ann_id - 1)
        pbar.close()

    else:
        # ── Fallback：從 __getitem__ 解碼（非 cache 模式）──
        pbar = tqdm(range(len(dataset)), desc="Building COCO GT",
                    unit="img", leave=False)

        for idx in pbar:
            img, target = dataset[idx]
            _, h, w = img.shape
            image_id = idx + 1

            coco_dict["images"].append({
                "id": image_id, "height": h, "width": w,
            })

            boxes = target["boxes"].numpy()
            labels = target["labels"].numpy()
            masks = target["masks"].numpy()

            for i in range(len(labels)):
                rle = mask_utils.encode(
                    np.asfortranarray(masks[i].astype(np.uint8))
                )
                rle["counts"] = rle["counts"].decode("utf-8")
                area = float(mask_utils.area(rle))

                x_min, y_min, x_max, y_max = boxes[i]
                coco_dict["annotations"].append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": int(labels[i]),
                    "segmentation": rle,
                    "area": area,
                    "bbox": [float(x_min), float(y_min),
                             float(x_max - x_min), float(y_max - y_min)],
                    "iscrowd": 0,
                })
                ann_id += 1

            pbar.set_postfix(annotations=ann_id - 1)

            del img, target, boxes, labels, masks
            if idx % 10 == 0:
                gc.collect()

        pbar.close()

    tmp_path = "/tmp/coco_gt_val.json"
    with open(tmp_path, "w") as f:
        json.dump(coco_dict, f)

    del coco_dict
    gc.collect()

    coco_gt = COCO(tmp_path)
    return coco_gt


@torch.no_grad()
def evaluate(model, data_loader, device, coco_gt=None):
    """Run evaluation and compute COCO mAP.

    為避免 WSL 在密集場景一次性把 (N, 1, H, W) mask tensor 轉成 numpy
    導致記憶體爆掉，這裡逐個 instance 處理 mask。
    """
    model.eval()
    coco_results = []

    pbar = tqdm(data_loader, desc="Evaluating", unit="batch", leave=False)

    for batch_idx, (images, targets) in enumerate(pbar):
        images = [img.to(device) for img in images]
        outputs = model(images)

        for i, output in enumerate(outputs):
            image_id = targets[i]["image_id"].item() + 1

            boxes = output["boxes"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            labels = output["labels"].cpu().numpy()
            masks_tensor = output["masks"]  # 留在 GPU，逐個搬到 CPU

            for j in range(len(scores)):
                binary_mask = (
                    (masks_tensor[j, 0] > 0.5).cpu().numpy().astype(np.uint8)
                )
                rle = mask_utils.encode(np.asfortranarray(binary_mask))
                rle["counts"] = rle["counts"].decode("utf-8")

                x_min, y_min, x_max, y_max = boxes[j]
                coco_results.append({
                    "image_id": image_id,
                    "category_id": int(labels[j]),
                    "segmentation": rle,
                    "score": float(scores[j]),
                    "bbox": [float(x_min), float(y_min),
                             float(x_max - x_min), float(y_max - y_min)],
                })
                del binary_mask

            del output, masks_tensor
        del outputs
        torch.cuda.empty_cache()

        if (batch_idx % 50) == 0:
            gc.collect()

        pbar.set_postfix(preds=len(coco_results))

    pbar.close()
    gc.collect()

    if len(coco_results) == 0 or coco_gt is None:
        print("No predictions or no GT for evaluation.")
        return {"AP": 0.0, "AP50": 0.0}

    tmp_path = "/tmp/coco_pred_val.json"
    with open(tmp_path, "w") as f:
        json.dump(coco_results, f)

    coco_dt = coco_gt.loadRes(tmp_path)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="segm")
    # V6: maxDets 1000 for dense-scene recall
    coco_eval.params.maxDets = [100, 300, 1000]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    metrics = {
        "AP": coco_eval.stats[0],
        "AP50": coco_eval.stats[1],
        "AP75": coco_eval.stats[2],
        "AP_small": coco_eval.stats[3],
        "AP_medium": coco_eval.stats[4],
        "AP_large": coco_eval.stats[5],
    }
    return metrics


@torch.no_grad()
def compute_val_loss(model, data_loader, device, use_amp=False):
    """在 val set 上計算 loss（不更新參數）。

    Mask R-CNN 必須在 train mode + targets 存在時才會回傳 loss dict，
    所以這裡將 model 設為 train()，但用 torch.no_grad() 包住，確保
    BN/Dropout 行為和 train 時一致，但不會做反向傳播。
    """
    was_training = model.training
    model.train()

    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(data_loader, desc="Val loss", unit="batch", leave=False)
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=use_amp):
            loss_dict = model(images, targets)
            losses = sum(v for v in loss_dict.values())
        total_loss += float(losses.item())
        num_batches += 1

        del images, targets, loss_dict, losses
        torch.cuda.empty_cache()
    pbar.close()
    gc.collect()

    if not was_training:
        model.eval()

    return total_loss / max(num_batches, 1)


def train_one_epoch(
    model,
    optimizer,
    data_loader,
    device,
    epoch,
    scaler=None,
    use_amp=False,
    max_grad_norm=5.0,
    grad_accum_steps=1,
    warmup_scheduler=None,
    in_warmup=False,
):
    """Train for one epoch with gradient accumulation and tqdm progress bar.

    Args:
        warmup_scheduler: LinearLR scheduler，在 warmup 階段每個 optimizer.step()
                          之後呼叫一次。
        in_warmup:        當前 epoch 是否仍在 warmup 範圍內。
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(data_loader,
                desc=f"Epoch {epoch:>2d}",
                unit="batch",
                bar_format="{l_bar}{bar:20}{r_bar}",
                leave=True)

    optimizer.zero_grad()

    for batch_idx, (images, targets) in enumerate(pbar):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward (optional mixed precision)
        with torch.cuda.amp.autocast(enabled=use_amp):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            # 除以 accumulation steps 以維持正確的梯度量級
            losses = losses / grad_accum_steps

        # Backward (累積梯度)
        if use_amp and scaler is not None:
            scaler.scale(losses).backward()
        else:
            losses.backward()

        # 每 grad_accum_steps 步或 epoch 結束時更新參數
        stepped = False
        at_accum = (batch_idx + 1) % grad_accum_steps == 0
        at_end = (batch_idx + 1) == len(data_loader)
        if at_accum or at_end:
            if use_amp and scaler is not None:
                if max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            optimizer.zero_grad()
            stepped = True

        # warmup：每次真正 optimizer.step() 之後 step LR
        if stepped and in_warmup and warmup_scheduler is not None:
            warmup_scheduler.step()

        batch_loss = losses.item() * grad_accum_steps  # 還原真實 loss 值
        total_loss += batch_loss
        num_batches += 1

        # ── tqdm postfix: 即時顯示各項 loss ──
        postfix = {
            "loss": f"{batch_loss:.3f}",
            "rpn_cls": f"{loss_dict['loss_objectness'].item():.3f}",
            "rpn_box": f"{loss_dict['loss_rpn_box_reg'].item():.3f}",
            "roi_cls": f"{loss_dict['loss_classifier'].item():.3f}",
            "roi_box": f"{loss_dict['loss_box_reg'].item():.3f}",
            "mask": f"{loss_dict['loss_mask'].item():.3f}",
        }
        pbar.set_postfix(postfix)

        del images, targets, loss_dict, losses
        torch.cuda.empty_cache()

    pbar.close()

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def main(args):
    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    use_amp = bool(args.use_amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"AMP enabled: {use_amp}")

    # ── Wandb ──
    use_wandb = False
    if not args.no_wandb:
        load_wandb_key()
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.run_name,
                config=vars(args),
            )
            use_wandb = True
            print("Wandb initialized successfully.")
        except Exception as e:
            print(f"Wandb init failed: {e}. Training without wandb.")

    # ── Data ──
    train_dataset, val_dataset = get_train_val_datasets(
        args.data_root,
        val_ratio=args.val_ratio,
        seed=args.seed,
        enable_dense_tiling=args.enable_dense_tiling,
        dense_instance_threshold=args.dense_instance_threshold,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        max_instances_per_tile=args.max_instances_per_tile,
        min_instances_per_tile=args.min_instances_per_tile,
    )

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "persistent_workers": (
            args.persistent_workers if args.num_workers > 0 else False
        ),
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        **loader_kwargs,
    )

    # ── Build COCO GT for validation ──
    print("Building COCO ground truth for validation set...")
    coco_gt = build_coco_gt(val_dataset)
    print(f"COCO GT: {len(coco_gt.getImgIds())} images, "
          f"{len(coco_gt.getAnnIds())} annotations")
    gc.collect()

    # ── Model ──
    model = get_baseline_model(
        num_classes=args.num_classes,
        min_size=args.multi_scale_min_sizes,
        max_size=args.multi_scale_max_size,
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters()
                          if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, "
          f"{trainable_params:,} trainable")
    print(
        "Train config | "
        f"resize_min={args.multi_scale_min_sizes}, "
        f"resize_max={args.multi_scale_max_size}, "
        f"amp={use_amp}, "
        f"dense_tiling={args.enable_dense_tiling}, "
        f"dense_threshold={args.dense_instance_threshold}, "
        f"tile={args.tile_size}, overlap={args.tile_overlap}, "
        f"max_inst_tile={args.max_instances_per_tile}, "
        f"workers={args.num_workers}, pin_memory={args.pin_memory}, "
        f"persistent_workers={args.persistent_workers}, "
        f"prefetch_factor={args.prefetch_factor if args.num_workers > 0 else 'n/a'}"
    )
    assert trainable_params < 200_000_000, \
        f"Model exceeds 200M parameter limit: {trainable_params:,}"

    # ── Optimizer & Scheduler ──
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # V6: LinearLR warmup + StepLR。
    # warmup 在前 warmup_epochs 期間「每次 optimizer.step()」逐步把 LR 拉滿；
    # warmup 結束後 StepLR 接手，按 epoch 衰減。
    updates_per_epoch = max(
        1, (len(train_loader) + args.grad_accum_steps - 1) // args.grad_accum_steps
    )
    warmup_total_iters = max(1, args.warmup_epochs * updates_per_epoch)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-3,
        end_factor=1.0,
        total_iters=warmup_total_iters,
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_step_size,
        gamma=args.lr_gamma,
    )
    print(
        f"LR schedule | warmup_epochs={args.warmup_epochs}, "
        f"warmup_total_iters={warmup_total_iters}, "
        f"step_size={args.lr_step_size}, gamma={args.lr_gamma}"
    )

    # ── Training loop ──
    os.makedirs(args.output_dir, exist_ok=True)
    run_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = os.path.join(args.output_dir, run_stamp)
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {ckpt_dir}")
    if use_wandb:
        try:
            wandb.config.update({"run_stamp": run_stamp, "ckpt_dir": ckpt_dir},
                                allow_val_change=True)
        except Exception:
            pass

    best_ap50 = 0.0

    # Epoch-level progress bar
    epoch_pbar = tqdm(range(1, args.epochs + 1),
                      desc="Training", unit="epoch",
                      bar_format="{l_bar}{bar:25}{r_bar}")

    for epoch in epoch_pbar:
        t_start = time.time()

        in_warmup = epoch <= args.warmup_epochs

        # Train
        avg_loss = train_one_epoch(
            model,
            optimizer,
            train_loader,
            device,
            epoch,
            scaler=scaler,
            use_amp=use_amp,
            max_grad_norm=args.max_grad_norm,
            grad_accum_steps=args.grad_accum_steps,
            warmup_scheduler=warmup_scheduler,
            in_warmup=in_warmup,
        )

        # Step scheduler：warmup 結束後才讓 StepLR 接手
        if not in_warmup:
            lr_scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        t_train = time.time() - t_start

        # Evaluate
        metrics = {}
        val_loss = None
        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            metrics = evaluate(model, val_loader, device, coco_gt)

            # Val loss monitor（用來判斷 overfitting）
            try:
                val_loss = compute_val_loss(model, val_loader, device, use_amp=use_amp)
            except Exception as e:
                tqdm.write(f"  [warn] compute_val_loss failed: {e}")
                val_loss = None

            # Save best model
            if metrics.get("AP50", 0) > best_ap50:
                best_ap50 = metrics["AP50"]
                save_path = os.path.join(ckpt_dir, "best_model.pth")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "ap50": best_ap50,
                }, save_path)

        # Save latest checkpoint
        save_path = os.path.join(ckpt_dir, "latest_model.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            "warmup_scheduler_state_dict": warmup_scheduler.state_dict(),
        }, save_path)

        # Update epoch-level progress bar
        postfix = {
            "loss": f"{avg_loss:.3f}",
            "lr": f"{current_lr:.1e}",
            "best": f"{best_ap50:.3f}",
        }
        if metrics:
            postfix["AP50"] = f"{metrics['AP50']:.3f}"
        if val_loss is not None:
            postfix["val_loss"] = f"{val_loss:.3f}"
        epoch_pbar.set_postfix(postfix)

        # Print epoch summary
        summary = (f"  Epoch {epoch}/{args.epochs} │ "
                   f"Loss: {avg_loss:.4f} │ LR: {current_lr:.1e} │ "
                   f"Time: {t_train:.0f}s")
        if metrics:
            summary += (f" │ AP50: {metrics['AP50']:.4f} │ "
                        f"AP: {metrics['AP']:.4f}")
        if val_loss is not None:
            summary += f" │ ValLoss: {val_loss:.4f}"
        tqdm.write(summary)

        # Log to wandb
        if use_wandb:
            log_dict = {
                "epoch": epoch,
                "train/loss": avg_loss,
                "train/lr": current_lr,
                "train/in_warmup": int(in_warmup),
            }
            if val_loss is not None:
                log_dict["val/loss"] = val_loss
            for k, v in metrics.items():
                log_dict[f"val/{k}"] = v
            wandb.log(log_dict)

    epoch_pbar.close()

    print(f"\n{'='*55}")
    print(f"  Training complete!  Best AP50: {best_ap50:.4f}")
    print(f"  Checkpoints saved to: {ckpt_dir}/")
    print(f"{'='*55}")

    if use_wandb:
        wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Mask R-CNN for cell segmentation"
    )

    # Data
    parser.add_argument("--data_root", type=str, default=DATA_CONFIG["data_root"])
    parser.add_argument("--val_ratio", type=float, default=DATA_CONFIG["val_ratio"])
    parser.add_argument("--seed", type=int, default=DATA_CONFIG["seed"])

    # Model
    parser.add_argument("--num_classes", type=int, default=MODEL_CONFIG["num_classes"])
    min_sizes_default = MODEL_CONFIG["multi_scale_min_sizes"]
    if isinstance(min_sizes_default, int):
        min_sizes_default = (min_sizes_default,)

    parser.add_argument(
        "--multi_scale_min_sizes",
        type=int,
        nargs="+",
        default=list(min_sizes_default),
        help="Shortest-edge multi-scale resize candidates.",
    )
    parser.add_argument(
        "--multi_scale_max_size",
        type=int,
        default=MODEL_CONFIG["multi_scale_max_size"],
        help="Longest-edge resize cap.",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=TRAIN_CONFIG["epochs"])
    parser.add_argument("--batch_size", type=int, default=TRAIN_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=TRAIN_CONFIG["lr"])
    parser.add_argument("--momentum", type=float, default=TRAIN_CONFIG["momentum"])
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=TRAIN_CONFIG["weight_decay"],
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=TRAIN_CONFIG.get("warmup_epochs", 0),
        help="Linear LR warmup epochs (0 = no warmup).",
    )
    parser.add_argument(
        "--lr_step_size",
        type=int,
        default=TRAIN_CONFIG["lr_step_size"],
    )
    parser.add_argument("--lr_gamma", type=float, default=TRAIN_CONFIG["lr_gamma"])
    parser.add_argument(
        "--num_workers",
        type=int,
        default=TRAIN_CONFIG["num_workers"],
    )
    parser.add_argument(
        "--pin_memory",
        action=argparse.BooleanOptionalAction,
        default=TRAIN_CONFIG["pin_memory"],
    )
    parser.add_argument(
        "--persistent_workers",
        action=argparse.BooleanOptionalAction,
        default=TRAIN_CONFIG["persistent_workers"],
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=TRAIN_CONFIG["prefetch_factor"],
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=TRAIN_CONFIG["max_grad_norm"],
    )
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=TRAIN_CONFIG.get("grad_accum_steps", 1),
        help="Gradient accumulation steps (effective batch = batch_size × this).",
    )
    parser.add_argument(
        "--use_amp",
        action=argparse.BooleanOptionalAction,
        default=TRAIN_CONFIG["use_amp"],
        help="Enable AMP mixed precision training on CUDA.",
    )
    parser.add_argument(
        "--enable_dense_tiling",
        action=argparse.BooleanOptionalAction,
        default=MEMORY_CONFIG["enable_dense_tiling"],
        help="Tile images with very dense instances to avoid mask OOM spikes.",
    )
    parser.add_argument(
        "--dense_instance_threshold",
        type=int,
        default=MEMORY_CONFIG["dense_instance_threshold"],
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=MEMORY_CONFIG["tile_size"],
    )
    parser.add_argument(
        "--tile_overlap",
        type=int,
        default=MEMORY_CONFIG["tile_overlap"],
    )
    parser.add_argument(
        "--max_instances_per_tile",
        type=int,
        default=MEMORY_CONFIG["max_instances_per_tile"],
    )
    parser.add_argument(
        "--min_instances_per_tile",
        type=int,
        default=MEMORY_CONFIG["min_instances_per_tile"],
    )

    # Evaluation
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=EVAL_CONFIG["eval_interval"],
    )

    # Output
    parser.add_argument("--output_dir", type=str, default=LOG_CONFIG["output_dir"])

    # Wandb
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=LOG_CONFIG["wandb_project"],
    )
    parser.add_argument("--run_name", type=str, default=LOG_CONFIG["run_name"])

    args = parser.parse_args()
    args.multi_scale_min_sizes = tuple(args.multi_scale_min_sizes)
    return args


if __name__ == "__main__":
    main(parse_args())
