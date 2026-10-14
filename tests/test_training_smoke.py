"""Smoke test for CPU-based model training and checkpoint persistence."""

from pathlib import Path
import tempfile
import pandas as pd
from PIL import Image
import pytest
import torch

from seedvision.config import SeedVisionConfig
from seedvision.data.dataset import create_dataloaders
from seedvision.training.checkpoints import load_checkpoint
from seedvision.training.trainer import SeedTrainer


def test_cpu_training_smoke_and_checkpoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        classes = ["sound_healthy", "discolored_spotted", "broken_cracked", "shrivelled_damaged"]

        # Generate 16 dummy images (4 per class)
        rows = []
        for c_idx, c_name in enumerate(classes):
            c_dir = tmp_path / c_name
            c_dir.mkdir(parents=True, exist_ok=True)
            for i in range(4):
                img_path = c_dir / f"{c_name}_{i}.jpg"
                img = Image.new("RGB", (32, 32), color=(c_idx * 50, 100, 150))
                img.save(img_path)
                rows.append({"filepath": str(img_path), "class_name": c_name})

        df = pd.DataFrame(rows)
        train_df = df.iloc[:12]
        val_df = df.iloc[12:]

        # Configure small fast training
        cfg = SeedVisionConfig()
        cfg.dataset.classes = classes
        cfg.dataset.image_size = [32, 32]
        cfg.training.epochs = 1
        cfg.training.batch_size = 4
        cfg.training.early_stopping_patience = 1
        cfg.paths.checkpoints_dir = str(tmp_path / "checkpoints")
        cfg.paths.metrics_dir = str(tmp_path / "metrics")

        train_loader, val_loader, _ = create_dataloaders(cfg, train_df, val_df)

        trainer = SeedTrainer(cfg)
        best_ckpt, history = trainer.fit(train_loader, val_loader)

        assert best_ckpt.is_file()
        assert len(history.epochs) == 1
        assert not torch.isnan(torch.tensor(history.train_loss[0]))
        assert not torch.isnan(torch.tensor(history.val_loss[0]))

        # Reload checkpoint
        loaded_model, payload = load_checkpoint(best_ckpt, device="cpu")
        assert payload["epoch"] == 1
        assert payload["classes"] == classes
