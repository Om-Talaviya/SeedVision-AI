"""Model training script for SeedVision AI."""

import argparse
from pathlib import Path
import sys
import pandas as pd
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from seedvision.config import load_config
from seedvision.data.dataset import create_dataloaders
from seedvision.evaluation.plots import plot_training_curves
from seedvision.training.trainer import SeedTrainer


def main():
    parser = argparse.ArgumentParser(description="Train deep learning models for SeedVision AI.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cnn.yaml",
        help="Path to YAML configuration file (e.g. configs/cnn.yaml or configs/transfer_learning.yaml).",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["cnn", "transfer"],
        default=None,
        help="Model architecture override (cnn: CustomSeedCNN, transfer: MobileNetV2).",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cpu', 'cuda', 'auto').")
    parser.add_argument(
        "--splits-dir",
        type=str,
        default=None,
        help="Path to directory containing train_manifest.csv and val_manifest.csv (e.g. data/processed/splits_real).",
    )
    parser.add_argument(
        "--name-suffix",
        type=str,
        default=None,
        help="Optional suffix to append to model name (e.g. 'Real' -> 'SeedNet-Baseline-Real').",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a 1-epoch quick smoke test to verify training pipeline.",
    )
    args = parser.parse_args()

    # Determine default config file based on model-type if provided
    config_file = args.config
    if args.model_type == "transfer" and config_file == "configs/cnn.yaml":
        config_file = "configs/transfer_learning.yaml"

    cfg = load_config(config_file)

    # Overrides
    if args.model_type == "cnn":
        cfg.model.architecture = "custom_cnn"
        cfg.model.name = "SeedNet-Baseline"
    elif args.model_type == "transfer":
        cfg.model.architecture = "mobilenet_v2"
        cfg.model.name = "MobileNetV2-Transfer"

    suffix = args.name_suffix
    if not suffix and args.splits_dir and "real" in str(args.splits_dir).lower():
        suffix = "Real"
    if suffix:
        cfg.model.name = f"{cfg.model.name}-{suffix}"

    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.lr is not None:
        cfg.training.learning_rate = args.lr
    if args.device is not None:
        cfg.training.device = args.device
    if args.smoke_test:
        cfg.training.epochs = 1
        cfg.training.early_stopping_patience = 1
        print("[SMOKE TEST MODE ENABLED] Setting epochs=1 for pipeline verification.")

    device = cfg.get_device()
    print(f"=== SeedVision AI Model Training ===")
    print(f"Model Architecture: {cfg.model.architecture} ({cfg.model.name})")
    print(f"Device:             {device}")
    print(f"Epochs:             {cfg.training.epochs}")
    print(f"Batch Size:         {cfg.training.batch_size}")
    print(f"Learning Rate:      {cfg.training.learning_rate}")
    print(f"Classes:            {cfg.dataset.classes}")

    # Check for split manifests
    splits_dir = Path(args.splits_dir) if args.splits_dir else Path(cfg.dataset.splits_dir)
    train_manifest = splits_dir / "train_manifest.csv"
    val_manifest = splits_dir / "val_manifest.csv"

    if not train_manifest.is_file() or not val_manifest.is_file():
        print(f"[ERROR] Split manifests not found in {splits_dir}.")
        print("Please run `python scripts/prepare_dataset.py` or `python scripts/ingest_real_data.py` first to generate splits.")
        sys.exit(1)

    train_df = pd.read_csv(train_manifest)
    val_df = pd.read_csv(val_manifest)
    print(f"Loaded datasets: Train={len(train_df)}, Val={len(val_df)}")

    # Build DataLoaders
    train_loader, val_loader, _ = create_dataloaders(
        config=cfg,
        train_df=train_df,
        val_df=val_df,
    )

    # Train model
    trainer = SeedTrainer(config=cfg)
    best_ckpt, history = trainer.fit(train_loader=train_loader, val_loader=val_loader)

    print(f"\n[SUCCESS] Training completed successfully!")
    print(f"Best checkpoint saved: {best_ckpt.resolve()}")

    # Render training curve figure
    figures_dir = Path(cfg.paths.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    curve_path = figures_dir / f"training_curves_{cfg.model.name}.png"
    plot_training_curves(history, curve_path, title=f"Training Convergence - {cfg.model.name}")
    print(f"Training convergence plot saved: {curve_path.resolve()}")


if __name__ == "__main__":
    main()
