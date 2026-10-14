"""Automated tests for dataset provenance, split leakage absence, and manifest integrity."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import pandas as pd
import pytest


def test_provenance_metadata_exists_and_valid():
    """Verify data/provenance.json exists, identifies synthetic fixtures, and forbids unvalidated real-world claims."""
    prov_file = Path("data/provenance.json")
    assert prov_file.is_file(), "data/provenance.json must exist in repository"

    with open(prov_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["is_real_world_photograph"] is False, "Synthetic fixtures must not be marked as real-world"
    assert meta["provenance_status"] == "synthetic_procedural_fixture"
    assert meta["total_images"] == 240
    assert len(meta["classes"]) == 4
    assert meta["splits"]["overlap_count"] == 0
    assert "CRITICAL" in meta["scientific_notice"]
    assert "germination" in meta["biological_boundary"]


def test_split_leakage_absence():
    """Verify train, val, and test manifests have zero sample overlap."""
    splits_dir = Path("data/processed/splits")
    train_csv = splits_dir / "train_manifest.csv"
    val_csv = splits_dir / "val_manifest.csv"
    test_csv = splits_dir / "test_manifest.csv"

    assert train_csv.is_file()
    assert val_csv.is_file()
    assert test_csv.is_file()

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    train_files = set(train_df["filename"])
    val_files = set(val_df["filename"])
    test_files = set(test_df["filename"])

    # Assert mutual disjointness (zero leakage)
    assert len(train_files.intersection(val_files)) == 0, "Train and Val splits share filenames"
    assert len(train_files.intersection(test_files)) == 0, "Train and Test splits share filenames"
    assert len(val_files.intersection(test_files)) == 0, "Val and Test splits share filenames"

    # Assert seed_id disjointness
    train_ids = set(train_df["seed_id"])
    val_ids = set(val_df["seed_id"])
    test_ids = set(test_df["seed_id"])
    assert len(train_ids.intersection(val_ids)) == 0, "Train and Val share physical seed_id"
    assert len(train_ids.intersection(test_ids)) == 0, "Train and Test share physical seed_id"
    assert len(val_ids.intersection(test_ids)) == 0, "Val and Test share physical seed_id"


def test_manifest_schema_and_file_existence():
    """Verify all files recorded in split manifests actually exist on disk with valid dimensions."""
    full_manifest = Path("data/processed/splits/full_split_manifest.csv")
    assert full_manifest.is_file()

    df = pd.read_csv(full_manifest)
    required_cols = {"filepath", "filename", "class_name", "seed_id", "split"}
    assert required_cols.issubset(set(df.columns)), f"Missing required columns in manifest: {set(df.columns)}"

    for _, row in df.iterrows():
        img_path = Path(row["filepath"])
        assert img_path.is_file(), f"Image file missing on disk: {img_path}"
        assert row["class_name"] in [
            "sound_healthy",
            "discolored_spotted",
            "broken_cracked",
            "shrivelled_damaged",
        ]


def test_hash_integrity_and_duplicate_isolation():
    """Verify no cryptographic MD5 duplicates exist across split boundaries."""
    splits_dir = Path("data/processed/splits")
    train_df = pd.read_csv(splits_dir / "train_manifest.csv")
    val_df = pd.read_csv(splits_dir / "val_manifest.csv")
    test_df = pd.read_csv(splits_dir / "test_manifest.csv")

    def get_hashes(df):
        hashes = set()
        for p in df["filepath"]:
            path = Path(p)
            if path.is_file():
                h = hashlib.md5(path.read_bytes()).hexdigest()
                hashes.add(h)
        return hashes

    train_hashes = get_hashes(train_df)
    val_hashes = get_hashes(val_df)
    test_hashes = get_hashes(test_df)

    assert len(train_hashes.intersection(val_hashes)) == 0, "Identical image hash crosses train/val"
    assert len(train_hashes.intersection(test_hashes)) == 0, "Identical image hash crosses train/test"
    assert len(val_hashes.intersection(test_hashes)) == 0, "Identical image hash crosses val/test"


def test_checkpoint_metadata_dataset_consistency():
    """Verify that checkpoint metadata classes match dataset configuration."""
    ckpt_dir = Path("artifacts/checkpoints")
    meta_files = list(ckpt_dir.glob("*.meta.json"))
    assert len(meta_files) >= 1, "At least one checkpoint metadata file must exist"

    expected_classes = [
        "sound_healthy",
        "discolored_spotted",
        "broken_cracked",
        "shrivelled_damaged",
    ]

    for mf in meta_files:
        with open(mf, "r", encoding="utf-8") as f:
            meta = json.load(f)
        classes = meta.get("classes", [])
        assert set(classes) == set(expected_classes), f"Checkpoint classes mismatch in {mf.name}"
