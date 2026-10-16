"""Unit tests for the real-image staging, validation, and multi-view specimen grouping workflow."""

import json
from pathlib import Path

import pandas as pd
from PIL import Image
import pytest

from scripts.stage_real_intake import (
    ALLOWED_TARGET_CLASSES,
    initialize_staging_scaffold,
    is_synthetic_fixture,
    validate_staging_dataset,
)


@pytest.fixture
def mock_staging_dir(tmp_path: Path) -> Path:
    staging_dir = tmp_path / "test_staging"
    initialize_staging_scaffold(staging_dir)
    return staging_dir


def test_staging_scaffold_initialization(mock_staging_dir: Path):
    assert (mock_staging_dir / "incoming").is_dir()
    assert (mock_staging_dir / "staging_manifest_template.csv").is_file()
    assert (mock_staging_dir / "README.md").is_file()

    df_template = pd.read_csv(mock_staging_dir / "staging_manifest_template.csv")
    required_cols = {
        "filename",
        "specimen_id",
        "view_id",
        "candidate_label",
        "review_status",
        "confidence",
        "source_provenance",
        "variety",
        "reviewer",
        "diagnostic_notes",
    }
    assert required_cols.issubset(set(df_template.columns))


def test_validate_staging_empty(mock_staging_dir: Path):
    report = validate_staging_dataset(staging_dir=mock_staging_dir)
    assert report["total_files_found"] == 0
    assert report["valid_images"] == 0
    assert report["status"] == "valid"
    assert report["approved_count"] == 0
    assert report["rejected_count"] == 0
    assert report["synthetic_fixtures"] == []
    assert report["issues"] == []


def test_validate_staging_empty_incoming_with_clean_manifest(mock_staging_dir: Path):
    # Manifest with only header and 0 data rows
    manifest_path = mock_staging_dir / "staging_manifest.csv"
    manifest_path.write_text(
        "filename,specimen_id,view_id,candidate_label,review_status,confidence,source_provenance,variety,reviewer,diagnostic_notes\n",
        encoding="utf-8",
    )
    report = validate_staging_dataset(staging_dir=mock_staging_dir, manifest_path=manifest_path)
    assert report["total_files_found"] == 0
    assert report["valid_images"] == 0
    assert report["status"] == "valid"
    assert report["issues"] == []


def test_manifest_references_missing_incoming_files_flagged(mock_staging_dir: Path):
    # Manifest lists a file that does not exist in incoming
    manifest_path = mock_staging_dir / "staging_manifest.csv"
    df = pd.DataFrame(
        [
            {
                "filename": "non_existent_seed.jpg",
                "specimen_id": "SPECIMEN-404",
                "view_id": "v1",
                "candidate_label": "sound_healthy",
                "review_status": "pending",
                "confidence": "high",
                "source_provenance": "Field Station",
                "variety": "Desi",
                "reviewer": "test_reviewer",
                "diagnostic_notes": "Sample",
            }
        ]
    )
    df.to_csv(manifest_path, index=False)
    report = validate_staging_dataset(staging_dir=mock_staging_dir, manifest_path=manifest_path)
    assert any("not present in incoming directory" in iss for iss in report["issues"])


def test_validate_staging_with_multiview_specimen(mock_staging_dir: Path):
    incoming = mock_staging_dir / "incoming"

    # Create 2 valid 128x128 images for 1 specimen
    img1_path = incoming / "seed_A_v1.jpg"
    img2_path = incoming / "seed_A_v2.jpg"
    Image.new("RGB", (128, 128), "beige").save(img1_path)
    Image.new("RGB", (128, 128), "wheat").save(img2_path)

    # Manifest with pending review status
    manifest_path = mock_staging_dir / "staging_manifest.csv"
    df = pd.DataFrame(
        [
            {
                "filename": "seed_A_v1.jpg",
                "specimen_id": "SPECIMEN-001",
                "view_id": "v1",
                "candidate_label": "sound_healthy",
                "review_status": "pending",
                "confidence": "high",
                "source_provenance": "Trial Field",
                "variety": "Desi JG-11",
                "reviewer": "test_reviewer",
                "diagnostic_notes": "Intact testa, zero damage",
            },
            {
                "filename": "seed_A_v2.jpg",
                "specimen_id": "SPECIMEN-001",
                "view_id": "v2",
                "candidate_label": "sound_healthy",
                "review_status": "pending",
                "confidence": "high",
                "source_provenance": "Trial Field",
                "variety": "Desi JG-11",
                "reviewer": "test_reviewer",
                "diagnostic_notes": "Intact testa, opposite view",
            },
        ]
    )
    df.to_csv(manifest_path, index=False)

    report = validate_staging_dataset(staging_dir=mock_staging_dir, manifest_path=manifest_path)
    assert report["total_files_found"] == 2
    assert report["valid_images"] == 2
    assert report["approved_count"] == 0
    assert report["unreviewed_pending_count"] == 2
    assert "SPECIMEN-001" in report["specimen_groups"]
    assert report["specimen_groups"]["SPECIMEN-001"]["count"] == 2


def test_validate_staging_rejects_unapproved_classes_and_flags_corrupt(mock_staging_dir: Path):
    incoming = mock_staging_dir / "incoming"

    # Corrupt zero-byte file
    corrupt_path = incoming / "bad_image.jpg"
    corrupt_path.write_bytes(b"")

    # Image with unapproved class
    valid_path = incoming / "unapproved_class.jpg"
    Image.new("RGB", (128, 128), "tan").save(valid_path)

    manifest_path = mock_staging_dir / "staging_manifest.csv"
    df = pd.DataFrame(
        [
            {
                "filename": "unapproved_class.jpg",
                "specimen_id": "SPECIMEN-999",
                "view_id": "v1",
                "candidate_label": "foreign_weed_seed",
                "review_status": "pending",
                "confidence": "high",
                "source_provenance": "Trial Field",
                "variety": "Unknown",
                "reviewer": "test_reviewer",
                "diagnostic_notes": "Not a chickpea",
            }
        ]
    )
    df.to_csv(manifest_path, index=False)

    report = validate_staging_dataset(staging_dir=mock_staging_dir, manifest_path=manifest_path)
    assert report["status"] == "has_errors"
    assert len(report["corrupt_files"]) == 1
    assert any("unapproved class 'foreign_weed_seed'" in iss for iss in report["issues"])


def test_synthetic_fixtures_cannot_be_approved(mock_staging_dir: Path):
    incoming = mock_staging_dir / "incoming"

    # Create synthetic test image
    syn_img = incoming / "synthetic_seed_demo.jpg"
    Image.new("RGB", (128, 128), "peru").save(syn_img)

    # Attempt to approve the synthetic fixture via manifest
    manifest_path = mock_staging_dir / "staging_manifest.csv"
    df = pd.DataFrame(
        [
            {
                "filename": "synthetic_seed_demo.jpg",
                "specimen_id": "SYNTHETIC-DEMO-001",
                "view_id": "v1",
                "candidate_label": "sound_healthy",
                "review_status": "approved",  # ILLEGAL APPROVAL ATTEMPT
                "confidence": "high",
                "source_provenance": "Synthetic Test Fixture Generator",
                "variety": "Synthetic Demo",
                "reviewer": "lead_agronomist",
                "diagnostic_notes": "Synthetic rendering",
                "is_synthetic": "True",
            }
        ]
    )
    df.to_csv(manifest_path, index=False)

    report = validate_staging_dataset(staging_dir=mock_staging_dir, manifest_path=manifest_path)

    # Must be intercepted, rejected, and flagged with errors
    assert report["status"] == "has_errors"
    assert report["approved_count"] == 0
    assert report["rejected_count"] == 1
    assert "synthetic_seed_demo.jpg" in report["synthetic_fixtures"]
    assert any(
        "Synthetic fixture 'synthetic_seed_demo.jpg' cannot be approved" in iss
        for iss in report["issues"]
    )


def test_synthetic_fixture_helper_detection():
    # Test explicit flag
    assert is_synthetic_fixture({"is_synthetic": "True"}, "sample.jpg") is True
    assert is_synthetic_fixture({"is_synthetic": "1"}, "sample.jpg") is True
    assert is_synthetic_fixture({"is_synthetic": "false"}, "sample.jpg") is False

    # Test provenance keyword
    assert is_synthetic_fixture({"source_provenance": "Synthetic Test Fixture"}, "sample.jpg") is True
    assert is_synthetic_fixture({"source_provenance": "University Field Trial"}, "sample.jpg") is False

    # Test filename indicator
    assert is_synthetic_fixture({}, "chickpea_synthetic_fixture_v1.jpg") is True
    assert is_synthetic_fixture({}, "genuine_chickpea_photo.jpg") is False


def test_isolated_synthetic_staging_fixtures_integrity():
    fixtures_dir = Path("d:/AI/Projects/SeedVision AI/tests/fixtures/synthetic_staging")
    assert fixtures_dir.is_dir()

    v1 = fixtures_dir / "chickpea_desi_JG11_SPECIMEN001_v1.jpg"
    v2 = fixtures_dir / "chickpea_desi_JG11_SPECIMEN001_v2.jpg"
    assert v1.is_file()
    assert v2.is_file()

    # Validate image file integrity
    with Image.open(v1) as im1:
        assert im1.size[0] >= 128 and im1.size[1] >= 128
    with Image.open(v2) as im2:
        assert im2.size[0] >= 128 and im2.size[1] >= 128

    # Validate sanitized manifest
    manifest_p = fixtures_dir / "synthetic_staging_manifest.csv"
    assert manifest_p.is_file()
    df = pd.read_csv(manifest_p)
    assert len(df) == 2
    assert all(df["is_synthetic"] == True)
    # Fabricated claims must NOT be present
    for prov in df["source_provenance"]:
        assert "ICRISAT Genebank" not in str(prov)
    for var in df["variety"]:
        assert "Desi Chickpea (JG-11)" not in str(var)
    for rev in df["reviewer"]:
        assert "lead_agronomist" not in str(rev)


def test_real_staging_incoming_is_clean():
    incoming_dir = Path("d:/AI/Projects/SeedVision AI/data/real_staging/incoming")
    assert incoming_dir.is_dir()

    # Verify no synthetic demonstration images remain in incoming/
    incoming_files = [p.name for p in incoming_dir.iterdir() if p.name != ".gitkeep"]
    assert "chickpea_desi_JG11_SPECIMEN001_v1.jpg" not in incoming_files
    assert "chickpea_desi_JG11_SPECIMEN001_v2.jpg" not in incoming_files

    # Verify staging manifest contains valid records matching incoming files
    manifest_p = Path("d:/AI/Projects/SeedVision AI/data/real_staging/staging_manifest.csv")
    assert manifest_p.is_file()
    df = pd.read_csv(manifest_p)
    if len(df) > 0:
        for fname in df["filename"]:
            assert (incoming_dir / fname).is_file()
        assert (df["review_status"].isin(["pending", "approved", "rejected"])).all()

