"""
Multi-seed DINO training wrapper.

Trains N models with different random seeds (all other hyper-parameters
come from a shared config.json), then automatically launches WBF
ensemble inference on the test set.

Usage examples:
    # Train 3 seeds then ensemble (defaults: 42 123 7)
    python train_ensemble.py --config config.json

    # Custom seeds
    python train_ensemble.py --config config.json --seeds 0 1 2

    # Skip training, only run ensemble on existing checkpoints
    python train_ensemble.py --config config.json --seeds 42 123 7 --skip_train

    # Custom ensemble parameters
    python train_ensemble.py --config config.json --seeds 42 123 7 \
        --iou_thr 0.55 --skip_box_thr 0.01 --score_threshold 0.01
"""
import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def find_best_checkpoint(output_root: Path, seed: int) -> Path:
    """Find the best.pth for a given seed under output_root/seed_<seed>/."""
    seed_dir = output_root / f"seed_{seed}"
    if not seed_dir.exists():
        raise FileNotFoundError(
            f"Seed directory not found: {seed_dir}\n"
            f"Make sure training completed for seed={seed}.")

    # The training script creates a timestamped sub-directory;
    # look for the most-recent one that contains best.pth.
    candidates = sorted(seed_dir.iterdir(), reverse=True)
    for d in candidates:
        best = d / "best.pth"
        if best.exists():
            return best

    # Fallback: best.pth directly inside seed_dir
    if (seed_dir / "best.pth").exists():
        return seed_dir / "best.pth"

    raise FileNotFoundError(
        f"best.pth not found under {seed_dir}. "
        f"Available contents: {[p.name for p in seed_dir.iterdir()]}")


def train_single_seed(config_path: str, seed: int, base_output_dir: Path,
                      wandb_project: str = None):
    """Train one model by spawning a subprocess with a temp config."""
    with open(config_path, "r") as f:
        config = json.load(f)

    config["seed"] = seed
    config["output_dir"] = str(base_output_dir / f"seed_{seed}")

    # Set up wandb logging for this seed
    if wandb_project:
        config["wandb_project"] = wandb_project
        config["wandb_run_name"] = f"seed_{seed}"
    elif config.get("wandb_project"):
        config["wandb_run_name"] = f"seed_{seed}"

    # Write a temporary config file
    tmp_config = base_output_dir / f"config_seed_{seed}.json"
    tmp_config.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_config, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  Training seed={seed}  |  output -> {config['output_dir']}")
    print(f"{'=' * 60}\n")

    script_dir = Path(__file__).resolve().parent
    cmd = [sys.executable, str(script_dir / "train.py"),
           "--config", str(tmp_config)]

    result = subprocess.run(cmd, cwd=str(script_dir))
    if result.returncode != 0:
        print(
            f"[ERROR] Training failed for seed={seed} (exit code {
                result.returncode})")
        sys.exit(result.returncode)

    print(f"[OK] Training completed for seed={seed}")


def run_ensemble_inference(
        config_path: str,
        seeds: list,
        base_output_dir: Path,
        iou_thr: float,
        skip_box_thr: float,
        score_threshold: float,
        ensemble_output: str,
        test_dir: str = None,
        batch_size: int = 1,
        num_select: int = None):
    """Launch ensemble_inference.py with the checkpoints from all seeds."""
    checkpoint_paths = []
    for seed in seeds:
        ckpt = find_best_checkpoint(base_output_dir, seed)
        checkpoint_paths.append(str(ckpt))
        print(f"  seed={seed} -> {ckpt}")

    script_dir = Path(__file__).resolve().parent
    cmd = [
        sys.executable, str(script_dir / "ensemble_inference.py"),
        "--config", config_path,
        "--checkpoints", *checkpoint_paths,
        "--output", ensemble_output,
        "--iou_thr", str(iou_thr),
        "--skip_box_thr", str(skip_box_thr),
        "--score_threshold", str(score_threshold),
        "--batch_size", str(batch_size),
    ]
    if num_select is not None:
        cmd.extend(["--num_select", str(num_select)])
    if test_dir:
        cmd.extend(["--test_dir", test_dir])

    print(f"\n{'=' * 60}")
    print(f"  Running WBF Ensemble Inference  ({len(seeds)} models)")
    print(f"{'=' * 60}\n")

    result = subprocess.run(cmd, cwd=str(script_dir))
    if result.returncode != 0:
        print(
            f"[ERROR] Ensemble inference failed (exit code {
                result.returncode})")
        sys.exit(result.returncode)

    print(f"\n[OK] Ensemble predictions saved to {ensemble_output}")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-seed DINO training + WBF ensemble inference")

    # --- Training args ---
    parser.add_argument(
        "--config",
        default="config.json",
        type=str,
        help="Path to base config.json (shared across all seeds)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 7],
                        help="Random seeds to train (default: 42 123 7)")
    parser.add_argument("--output_dir", default=None, type=str,
                        help="Root output dir (default: config's output_dir)")
    parser.add_argument(
        "--skip_train",
        action="store_true",
        help="Skip training, only run ensemble on existing checkpoints")

    # --- Wandb args ---
    parser.add_argument(
        "--wandb_project",
        default=None,
        type=str,
        help="Wandb project name (enables wandb logging for all seeds)")

    # --- Ensemble inference args ---
    parser.add_argument("--iou_thr", type=float, default=0.55,
                        help="WBF IoU threshold (default: 0.55)")
    parser.add_argument(
        "--skip_box_thr",
        type=float,
        default=0.0001,
        help="WBF minimum score to keep a box (default: 0.0001)")
    parser.add_argument("--score_threshold", type=float, default=0.01,
                        help="Final score threshold for output predictions")
    parser.add_argument("--ensemble_output", default="pred.json", type=str,
                        help="Output path for ensemble predictions")
    parser.add_argument("--num_select", type=int, default=None,
                        help="Top-k predictions per model for WBF "
                             "(default: num_queries × num_classes)")
    parser.add_argument("--test_dir", default=None, type=str,
                        help="Test images directory (default: data_path/test)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Inference batch size")

    args = parser.parse_args()

    # Resolve output directory
    with open(args.config, "r") as f:
        base_config = json.load(f)
    base_output_dir = Path(
        args.output_dir or base_config.get(
            "output_dir", "./output"))
    base_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Seeds: {args.seeds}")
    print(f"Base config: {args.config}")
    print(f"Output root: {base_output_dir}")

    # ---- Phase 1: Train each seed ----
    if not args.skip_train:
        for i, seed in enumerate(args.seeds):
            print(f"\n>>> Model {i + 1}/{len(args.seeds)}  (seed={seed})")
            train_single_seed(args.config, seed, base_output_dir,
                              wandb_project=args.wandb_project)
    else:
        print("\n[SKIP] Training skipped (--skip_train). Using existing checkpoints.")

    # ---- Phase 2: Ensemble inference ----
    run_ensemble_inference(
        config_path=args.config,
        seeds=args.seeds,
        base_output_dir=base_output_dir,
        iou_thr=args.iou_thr,
        skip_box_thr=args.skip_box_thr,
        score_threshold=args.score_threshold,
        ensemble_output=args.ensemble_output,
        test_dir=args.test_dir,
        batch_size=args.batch_size,
        num_select=args.num_select,
    )


if __name__ == "__main__":
    main()
