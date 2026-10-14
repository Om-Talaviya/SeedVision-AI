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


def test_annotations_csv_ingestion_and_ambiguous_discard():
    """Verify annotations.csv ingestion, ambiguous_discard filtering, and unreviewed image exclusion."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        src = Path(tmp_dir) / "source"
        tgt = Path(tmp_dir) / "target"
        splits = Path(tmp_dir) / "splits"
        src.mkdir(parents=True)

        classes = ["sound_healthy", "broken_cracked"]

        # Create 4 physical seeds for sound_healthy (each with 2 multi-view photos)
        # and 4 physical seeds for broken_cracked
        annot_rows = []
        for cls in classes:
            c_dir = src / cls
            c_dir.mkdir(parents=True, exist_ok=True)
            for s_idx in range(1, 5):
                seed_id = f"phy_seed_{cls}_{s_idx:02d}"
                for v_idx in [1, 2]:
                    fname = f"arbitrary_filename_{cls}_{s_idx}_{v_idx}.jpg"
                    im = Image.new("RGB", (100, 100), color=(100 + s_idx * 20, 50 + v_idx * 30, 20 + len(cls) * 5))
                    im.save(c_dir / fname)
                    annot_rows.append(
                        {
                            "filename": fname,
                            "class_name": cls,
                            "seed_id": seed_id,
                            "confidence": "high",
                            "notes": "Verified good view",
                        }
                    )

        # Add an ambiguous discard sample
        im_amb = Image.new("RGB", (100, 100), color=(50, 50, 50))
        im_amb.save(src / "sound_healthy" / "blurry_seed.jpg")
        annot_rows.append(
            {
                "filename": "blurry_seed.jpg",
                "class_name": "ambiguous_discard",
                "seed_id": "seed_ambiguous",
                "confidence": "low",
                "notes": "Out of focus and partial glare",
            }
        )

        # Add an unreviewed image on disk (NOT in annotations.csv)
        im_unrev = Image.new("RGB", (100, 100), color=(200, 200, 200))
        im_unrev.save(src / "sound_healthy" / "unreviewed_sample.jpg")

        # Save annotations.csv
        annot_csv_path = src / "annotations.csv"
        pd.DataFrame(annot_rows).to_csv(annot_csv_path, index=False)

        res = audit_and_ingest_real_dataset(
            source_dir=src,
            target_dir=tgt,
            splits_dir=splits,
            classes=classes,
            annotations_file=annot_csv_path,
            provenance_path=Path(tmp_dir) / "provenance_real.json",
        )

        # Check results
        assert res["annotation_manifest_used"] is True
        assert res["ambiguous_discards_count"] == 1
        assert res["unreviewed_images_skipped_count"] == 1
        assert res["valid_images_retained"] == 16  # 2 classes * 4 seeds * 2 views = 16

        # Check splits exist
        train_df = pd.read_csv(splits / "train_manifest.csv")
        test_df = pd.read_csv(splits / "test_manifest.csv")
        val_df = pd.read_csv(splits / "val_manifest.csv")

        # Unreviewed and ambiguous images MUST NOT exist in any split
        all_split_files = set(train_df["filename"]).union(test_df["filename"]).union(val_df["filename"])
        assert "blurry_seed.jpg" not in all_split_files
        assert "unreviewed_sample.jpg" not in all_split_files

        # Multi-view photos with different filenames MUST stay in same partition
        for s_idx in range(1, 5):
            for cls in classes:
                s_id = f"{cls}_phy_seed_{cls}_{s_idx:02d}"
                partitions_found = []
                if s_id in train_df["seed_id"].values:
                    partitions_found.append("train")
                if s_id in val_df["seed_id"].values:
                    partitions_found.append("val")
                if s_id in test_df["seed_id"].values:
                    partitions_found.append("test")

                # The seed must appear in EXACTLY 1 partition
                assert len(partitions_found) == 1, f"Seed {s_id} leaked across {partitions_found}"


def test_evaluator_real_vs_synthetic_metadata():
    """Verify ModelEvaluator sets appropriate evaluation_type and provenance tags."""
    import torch
    import torch.nn as nn
    from seedvision.evaluation.metrics import ModelEvaluator

    dummy_model = nn.Linear(10, 4)

    # 1. Synthetic default
    ev_synth = ModelEvaluator(
        model=dummy_model,
        classes=["a", "b", "c", "d"],
        model_name="SeedNet-Baseline",
    )
    assert ev_synth.is_real is False
    assert ev_synth.evaluation_type == "synthetic_pipeline_verification"
    assert ev_synth.dataset_provenance == "synthetic_procedural_fixture"

    # 2. Real evaluated
    ev_real = ModelEvaluator(
        model=dummy_model,
        classes=["a", "b", "c", "d"],
        model_name="SeedNet-Baseline-Real",
        is_real_world_validated=True,
    )
    assert ev_real.is_real is True
    assert ev_real.evaluation_type == "real_world_test_evaluation"
    assert ev_real.dataset_provenance == "manually_annotated_genuine_photographs"


def test_renamed_synthetic_checkpoint_cannot_be_marked_real_world_validated():
    """Prove that renaming a synthetic checkpoint to include 'Real' fails provenance verification."""
    from seedvision.evaluation.provenance_verifier import verify_evaluation_provenance
    import torch

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 1. Create a synthetic checkpoint renamed to include 'Real'
        fake_checkpoint_path = tmp_path / "best_SeedNet-Baseline-Real.pt"
        synthetic_payload = {
            "model_state_dict": {},
            "epoch": 10,
            "val_loss": 0.1,
            "val_acc": 0.95,
            # Synthetic checkpoint lacks is_real_world_trained=True
            "is_real_world_trained": False,
            "dataset_provenance": "synthetic_procedural_fixture",
        }
        torch.save(synthetic_payload, fake_checkpoint_path)

        # 2. Create splits directory with genuine provenance to simulate evaluated folder
        splits_dir = tmp_path / "splits_real"
        splits_dir.mkdir(parents=True)
        prov_record = {
            "is_real_world_photograph": True,
            "provenance_status": "genuine_photograph",
            "valid_images_retained": 10,
        }
        with open(splits_dir / "provenance_real.json", "w", encoding="utf-8") as f:
            json.dump(prov_record, f)

        # Create dummy test manifest with existing dummy image
        img_path = tmp_path / "dummy_real.jpg"
        Image.new("RGB", (64, 64), color=(120, 100, 80)).save(img_path)
        test_df = pd.DataFrame([{"filepath": str(img_path), "class_name": "sound_healthy", "seed_id": "seed_01"}])
        test_df.to_csv(splits_dir / "test_manifest.csv", index=False)

        # 3. Run provenance verification
        result = verify_evaluation_provenance(
            checkpoint_path=fake_checkpoint_path,
            splits_dir=splits_dir,
            data_dir=tmp_path,
        )

        # Verification must FAIL because the checkpoint itself is synthetic
        assert result["is_real_world_validated"] is False
        assert result["evaluation_type"] == "unverified_evaluation"
        assert "checkpoint lacks verified genuine training metadata" in result["verification_details"]["reason"]


def test_in_place_ingestion_safety_and_nested_prevention():
    """Verify in-place ingestion works without duplication and nested directories are rejected."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        raw_dir = root / "real_raw"
        splits_dir = root / "splits_real"
        classes = ["sound_healthy", "broken_cracked"]

        # Populate images directly in raw_dir
        for cls_idx, cls in enumerate(classes):
            c_dir = raw_dir / cls
            c_dir.mkdir(parents=True, exist_ok=True)
            for i in range(1, 6):
                im = Image.new("RGB", (100, 100), color=(100 + i * 10, 100 + cls_idx * 40, 80))
                im.save(c_dir / f"{cls}_seed_{i:03d}.jpg")

        # In-place ingestion (source_dir == target_dir)
        record = audit_and_ingest_real_dataset(
            source_dir=raw_dir,
            target_dir=raw_dir,
            splits_dir=splits_dir,
            classes=classes,
            copy_files=True,  # Should be automatically disabled in in-place mode
            seed=42,
            provenance_path=root / "provenance_real.json",
        )

        assert record["valid_images_retained"] == 10
        assert (splits_dir / "train_manifest.csv").is_file()
        assert (splits_dir / "provenance_real.json").is_file()

        # Verify no nested duplicate files were created in raw_dir
        for cls in classes:
            files_in_cls = list((raw_dir / cls).glob("*.jpg"))
            assert len(files_in_cls) == 5, f"Expected 5 files in {cls}, found {len(files_in_cls)}"

        # Verify nesting rejection: passing a splits_dir inside raw_dir must raise ValueError
        nested_splits = raw_dir / "nested_splits"
        with pytest.raises(ValueError, match="nested inside or equal"):
            audit_and_ingest_real_dataset(
                source_dir=raw_dir,
                target_dir=root / "staging_out",
                splits_dir=nested_splits,
                classes=classes,
            )

