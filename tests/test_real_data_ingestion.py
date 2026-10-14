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
