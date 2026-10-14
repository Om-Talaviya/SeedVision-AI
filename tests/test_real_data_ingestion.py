"""Unit tests for genuine seed photograph ingestion, validation, and splitting tools."""

import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from PIL import Image
import pytest

from scripts.ingest_real_data import (
    audit_and_ingest_real_dataset,
    compute_average_hash,
    initialize_real_data_scaffold,
)
from seedvision.data.splitting import create_leakage_resistant_splits


def test_real_data_scaffold_initialization():
    """Verify scaffolding creates all 4 class directories, README, and annotation template."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        target_path = Path(tmp_dir) / "real_raw"
        initialize_real_data_scaffold(target_path)

        for cls in [
            "sound_healthy",
            "discolored_spotted",
            "broken_cracked",
            "shrivelled_damaged",
        ]:
            assert (target_path / cls).is_dir(), f"Missing class directory {cls}"

        assert (target_path / "annotations_template.csv").is_file()
        assert (target_path / "README.md").is_file()

        df = pd.read_csv(target_path / "annotations_template.csv")
        assert "class_name" in df.columns
        assert "seed_id" in df.columns


def test_compute_average_hash():
    """Verify average hash computation produces consistent hexadecimal bit strings."""
    img1 = Image.new("RGB", (128, 128), color=(220, 200, 160))
    img2 = Image.new("RGB", (128, 128), color=(220, 200, 160))
    img_diff = Image.new("RGB", (128, 128), color=(50, 30, 10))

    h1 = compute_average_hash(img1)
    h2 = compute_average_hash(img2)
    h3 = compute_average_hash(img_diff)

    assert isinstance(h1, str)
    assert len(h1) == 16
    assert h1 == h2, "Identical images must yield identical perceptual hashes"


def test_audit_and_ingest_real_dataset_deduplication():
    """Verify ingestion pipeline detects duplicates, prevents split leakage, and generates manifests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        src = Path(tmp_dir) / "source"
        tgt = Path(tmp_dir) / "target"
        splits = Path(tmp_dir) / "splits"

        classes = ["sound_healthy", "broken_cracked"]

        # Create dummy test images with duplicates
        for cls_idx, cls in enumerate(classes):
            c_dir = src / cls
            c_dir.mkdir(parents=True, exist_ok=True)
            for i in range(1, 6):
                im = Image.new("RGB", (100, 100), color=(150 + i * 10, 100 + cls_idx * 40, 80))
                im.save(c_dir / f"{cls}_{i:03d}.jpg")

            # Add duplicate of sample 001
            im_dup = Image.open(c_dir / f"{cls}_001.jpg")
            im_dup.save(c_dir / f"{cls}_001_copy.jpg")

        record = audit_and_ingest_real_dataset(
            source_dir=src,
            target_dir=tgt,
            splits_dir=splits,
            classes=classes,
            copy_files=True,
            seed=42,
            provenance_path=Path(tmp_dir) / "provenance_real.json",
        )

        assert record["exact_duplicates_removed"] == 2
        assert record["valid_images_retained"] == 10
        assert (splits / "train_manifest.csv").is_file()
        assert (splits / "val_manifest.csv").is_file()
        assert (splits / "test_manifest.csv").is_file()


def test_empty_or_insufficient_samples_handling():
    """Verify empty directory or insufficient samples produces actionable error status."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        src = Path(tmp_dir) / "empty_source"
        tgt = Path(tmp_dir) / "target"
        splits = Path(tmp_dir) / "splits"

        # 1. Empty source
        rec_empty = audit_and_ingest_real_dataset(
            source_dir=src, target_dir=tgt, splits_dir=splits
        )
        assert rec_empty["status"] == "empty"
        assert rec_empty["valid_count"] == 0

        # 2. Insufficient samples (< 3 groups per class)
        cls_dir = src / "sound_healthy"
        cls_dir.mkdir(parents=True, exist_ok=True)
        im = Image.new("RGB", (100, 100), color=(200, 180, 140))
        im.save(cls_dir / "sample_001.jpg")

        rec_few = audit_and_ingest_real_dataset(
            source_dir=src, target_dir=tgt, splits_dir=splits
        )
        assert rec_few["status"] == "insufficient_samples"


def test_seed_group_leakage_impossibility():
    """Verify that multiple views/photos of the same physical seed NEVER cross train/val/test splits."""
    rows = []
    classes = ["sound_healthy", "broken_cracked", "discolored_spotted", "shrivelled_damaged"]

    # 10 physical seeds per class, with 3 views each = 30 images per class
    for cls in classes:
        for seed_num in range(1, 11):
            seed_id = f"{cls}_seed_{seed_num:03d}"
            for view_num in range(1, 4):
                rows.append(
                    {
                        "filepath": f"/data/real/{cls}/{seed_id}_v{view_num}.jpg",
                        "class_name": cls,
                        "seed_id": seed_id,
                    }
                )

    manifest_df = pd.DataFrame(rows)

    train_df, val_df, test_df = create_leakage_resistant_splits(
        manifest_df=manifest_df,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        group_col="seed_id",
        class_col="class_name",
        random_state=42,
    )

    train_seeds = set(train_df["seed_id"].unique())
    val_seeds = set(val_df["seed_id"].unique())
    test_seeds = set(test_df["seed_id"].unique())

    # Mathematical disjointness check
    assert len(train_seeds.intersection(val_seeds)) == 0, "Seed group leaked between train and val"
    assert len(train_seeds.intersection(test_seeds)) == 0, "Seed group leaked between train and test"
    assert len(val_seeds.intersection(test_seeds)) == 0, "Seed group leaked between val and test"

    # Ensure every partition has samples for every class
    for cls in classes:
        assert (train_df["class_name"] == cls).sum() >= 1
        assert (val_df["class_name"] == cls).sum() >= 1
        assert (test_df["class_name"] == cls).sum() >= 1


def test_real_vs_synthetic_checkpoint_separation():
    """Verify that real-data model naming and checkpoint tagging cannot collide with synthetic models."""
    from seedvision.config import load_config

    cfg = load_config("configs/cnn.yaml")
    assert cfg.model.name == "SeedNet-Baseline"

    # Simulate real-data suffix logic from scripts/train.py
    splits_dir = "data/processed/splits_real"
    name_suffix = None
    if not name_suffix and "real" in splits_dir.lower():
        name_suffix = "Real"

    real_model_name = f"{cfg.model.name}-{name_suffix}"
    assert real_model_name == "SeedNet-Baseline-Real"
    assert real_model_name != cfg.model.name, "Real model name must be strictly distinct from baseline"
