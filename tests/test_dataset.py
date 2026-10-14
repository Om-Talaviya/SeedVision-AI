"""Unit tests for dataset loading, validation, and splitting."""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch

from seedvision.data.dataset import SeedDataset
from seedvision.data.splitting import create_leakage_resistant_splits
from seedvision.data.validation import ImageValidator, validate_dataset_structure


@pytest.fixture
def temp_dataset_dir():
    """Create a temporary directory structure with dummy images for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        classes = ["sound_healthy", "discolored_spotted", "broken_cracked", "shrivelled_damaged"]
        for c_idx, cls in enumerate(classes):
            cls_dir = root / cls
            cls_dir.mkdir(parents=True, exist_ok=True)
            for i in range(5):
                color = ((c_idx * 60 + i * 10) % 256, (c_idx * 30 + 100) % 256, (i * 40 + 50) % 256)
                img = Image.new("RGB", (64, 64), color=color)
                img.save(cls_dir / f"{cls}_{i}.jpg")
        yield root, classes


def test_image_validator(temp_dataset_dir):
    root, classes = temp_dataset_dir
    validator = ImageValidator(min_width=32, min_height=32)

    sample_img = root / classes[0] / f"{classes[0]}_0.jpg"
    res = validator.validate_file(str(sample_img))

    assert res.is_valid is True
    assert res.width == 64
    assert res.height == 64
    assert res.channels == 3
    assert res.file_hash is not None


def test_corrupt_image_detection():
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(b"NOT_A_VALID_IMAGE_DATA_BYTES")
        corrupt_path = f.name

    validator = ImageValidator()
    res = validator.validate_file(corrupt_path)
    assert res.is_valid is False
    assert "Corrupted" in (res.error_message or "")


def test_dataset_structure_validation(temp_dataset_dir):
    root, classes = temp_dataset_dir
    report = validate_dataset_structure(str(root), classes)

    assert report.total_images_found == 20
    assert report.valid_images_count == 20
    assert report.corrupt_images_count == 0
    assert report.is_ready_for_training is True


def test_leakage_resistant_splitting():
    rows = []
    classes = ["sound_healthy", "discolored_spotted"]
    for i in range(40):
        rows.append(
            {
                "filepath": f"/fake/path/img_{i}.jpg",
                "class_name": classes[i % 2],
                "seed_id": f"seed_{i // 2}",  # pair of photos per seed
            }
        )
    df = pd.DataFrame(rows)

    train_df, val_df, test_df = create_leakage_resistant_splits(
        df,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        group_col="seed_id",
        random_state=42,
    )

    # Verify no seed_id leaks between train and val/test
    train_seeds = set(train_df["seed_id"].unique())
    val_seeds = set(val_df["seed_id"].unique())
    test_seeds = set(test_df["seed_id"].unique())

    assert len(train_seeds.intersection(val_seeds)) == 0
    assert len(train_seeds.intersection(test_seeds)) == 0
    assert len(val_seeds.intersection(test_seeds)) == 0


def test_seed_dataset_item_retrieval(temp_dataset_dir):
    root, classes = temp_dataset_dir
    ds = SeedDataset(root, classes=classes)

    assert len(ds) == 20
    tensor, label_idx, meta = ds[0]

    assert isinstance(tensor, torch.Tensor)
    assert tensor.dim() == 3  # [3, H, W]
    assert 0 <= label_idx < len(classes)
    assert "filepath" in meta
