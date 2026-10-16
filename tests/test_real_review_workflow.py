"""Tests for human review workflow, specimen grouping, validation, and rollback.

Covers:
- Label validation (4 allowed target classes + pending_review)
- Specimen-level consistent multi-view updates
- Rejection of low-confidence approvals (must stay pending)
- Reviewer identity and diagnostic note mandates
- Automatic backup creation and rollback restoration
- Missing image tracking and error handling
"""

from pathlib import Path
import shutil
import tempfile
import pandas as pd
from PIL import Image
import pytest

from seedvision.data.real_review import (
    ALLOWED_CONFIDENCES,
    ALLOWED_REVIEW_STATUSES,
    ALLOWED_TARGET_LABELS,
    REQUIRED_MANIFEST_COLUMNS,
    create_manifest_backup,
    load_staging_manifest,
    restore_manifest_backup,
    revert_specimen_review,
    save_specimen_review,
    validate_review_decision,
)


@pytest.fixture
def staged_review_fixture(tmp_path: Path):
    """Create a temporary staged manifest with two 3-view specimens and real image files."""
    staging_dir = tmp_path / "real_staging"
    incoming_dir = staging_dir / "incoming"
    incoming_dir.mkdir(parents=True)

    manifest_path = staging_dir / "staging_manifest.csv"

    # Create dummy images for REAL_SPECIMEN_A (3 views) and REAL_SPECIMEN_B (3 views)
    rows = []
    for spec_id in ["SPECIMEN_A", "SPECIMEN_B"]:
        for v_num in [1, 2, 3]:
            fname = f"{spec_id}_view_{v_num:02d}.jpg"
            img_p = incoming_dir / fname
            # Generate small valid RGB image
            im = Image.new("RGB", (320, 240), color=(180, 160, 120))
            im.save(img_p, format="JPEG")

            rows.append({
                "filename": fname,
                "specimen_id": spec_id,
                "view_id": f"view_{v_num:02d}",
                "candidate_label": "pending_review",
                "review_status": "pending",
                "confidence": "",
                "source_provenance": "ICAR_Test_Station",
                "variety": "Desi_Chickpea",
                "reviewer": "",
                "diagnostic_notes": "Awaiting human review",
            })

    df = pd.DataFrame(rows)
    df.to_csv(manifest_path, index=False)

    return {
        "staging_dir": staging_dir,
        "incoming_dir": incoming_dir,
        "manifest_path": manifest_path,
        "df": df,
    }


def test_allowed_labels_and_statuses():
    """Verify standard allowed values match the SeedVision AI taxonomy."""
    assert "sound_healthy" in ALLOWED_TARGET_LABELS
    assert "discolored_spotted" in ALLOWED_TARGET_LABELS
    assert "broken_cracked" in ALLOWED_TARGET_LABELS
    assert "shrivelled_damaged" in ALLOWED_TARGET_LABELS
    assert "pending_review" in ALLOWED_TARGET_LABELS

    assert "pending" in ALLOWED_REVIEW_STATUSES
    assert "approved" in ALLOWED_REVIEW_STATUSES
    assert "rejected" in ALLOWED_REVIEW_STATUSES

    assert "high" in ALLOWED_CONFIDENCES
    assert "medium" in ALLOWED_CONFIDENCES
    assert "low" in ALLOWED_CONFIDENCES


def test_validate_review_decision_valid_approval():
    """Valid approval with reviewer, notes, and high confidence must succeed."""
    dec = validate_review_decision(
        candidate_label="sound_healthy",
        review_status="approved",
        confidence="high",
        reviewer="evaluator_01",
        diagnostic_notes="Uniform smooth testa, no lesions or fractures.",
    )
    assert dec["candidate_label"] == "sound_healthy"
    assert dec["review_status"] == "approved"
    assert dec["confidence"] == "high"
    assert dec["reviewer"] == "evaluator_01"


def test_validate_review_decision_low_confidence_forces_pending():
    """Uncertain specimen with confidence='low' cannot be approved."""
    with pytest.raises(ValueError, match="Safety constraint violation: Low-confidence specimens cannot be marked 'approved'"):
        validate_review_decision(
            candidate_label="broken_cracked",
            review_status="approved",
            confidence="low",
            reviewer="evaluator_01",
            diagnostic_notes="Looks ambiguous from one angle.",
        )


def test_validate_review_decision_pending_label_cannot_be_approved():
    """Specimen with candidate_label='pending_review' cannot be marked approved."""
    with pytest.raises(ValueError, match="Cannot approve a specimen with candidate_label='pending_review'"):
        validate_review_decision(
            candidate_label="pending_review",
            review_status="approved",
            confidence="high",
            reviewer="evaluator_01",
            diagnostic_notes="Not sure yet",
        )


def test_validate_review_decision_missing_reviewer_or_notes():
    """Approval requires reviewer identity and diagnostic justification."""
    # Missing reviewer
    with pytest.raises(ValueError, match="Reviewer identity is required"):
        validate_review_decision(
            candidate_label="discolored_spotted",
            review_status="approved",
            confidence="high",
            reviewer="",
            diagnostic_notes="Dark spots on dorsal side",
        )

    # Missing notes
    with pytest.raises(ValueError, match="Diagnostic notes are required"):
        validate_review_decision(
            candidate_label="discolored_spotted",
            review_status="approved",
            confidence="high",
            reviewer="evaluator_01",
            diagnostic_notes="",
        )


def test_validate_review_decision_rejection_requires_notes():
    """Rejection must include notes specifying why the specimen was rejected."""
    with pytest.raises(ValueError, match="Diagnostic notes / rejection rationale are required"):
        validate_review_decision(
            candidate_label="pending_review",
            review_status="rejected",
            confidence="high",
            reviewer="evaluator_01",
            diagnostic_notes="",
        )


def test_validate_review_decision_invalid_label():
    """Unknown label must be rejected."""
    with pytest.raises(ValueError, match="Invalid candidate_label"):
        validate_review_decision(
            candidate_label="moldy_rotten",
            review_status="approved",
            confidence="high",
            reviewer="evaluator_01",
            diagnostic_notes="Invalid class",
        )


def test_load_staging_manifest(staged_review_fixture):
    """Manifest loader groups rows by specimen_id and checks image dimensions."""
    manifest_p = staged_review_fixture["manifest_path"]
    incoming_dir = staged_review_fixture["incoming_dir"]

    result = load_staging_manifest(manifest_p, incoming_dir)

    assert result["total_specimens"] == 2
    assert result["total_views"] == 6
    assert len(result["missing_images"]) == 0
    assert "SPECIMEN_A" in result["specimens"]
    assert "SPECIMEN_B" in result["specimens"]

    spec_a = result["specimens"]["SPECIMEN_A"]
    assert len(spec_a["views"]) == 3
    for v in spec_a["views"]:
        assert v["image_exists"] is True
        assert v["width"] == 320
        assert v["height"] == 240


def test_load_staging_manifest_detects_missing_image(staged_review_fixture):
    """Manifest loader flags rows whose image files do not exist on disk."""
    manifest_p = staged_review_fixture["manifest_path"]
    incoming_dir = staged_review_fixture["incoming_dir"]

    # Delete one view image
    target_img = incoming_dir / "SPECIMEN_A_view_01.jpg"
    target_img.unlink()

    result = load_staging_manifest(manifest_p, incoming_dir)
    assert "SPECIMEN_A_view_01.jpg" in result["missing_images"]


def test_save_specimen_review_updates_all_views_and_creates_backup(staged_review_fixture):
    """Saving a decision updates all 3 views of the physical specimen consistently and creates backup."""
    manifest_p = staged_review_fixture["manifest_path"]

    res = save_specimen_review(
        manifest_path=manifest_p,
        specimen_id="SPECIMEN_A",
        candidate_label="discolored_spotted",
        review_status="approved",
        confidence="high",
        reviewer="dr_agronomist",
        diagnostic_notes="Observed necrotic spot on chalazal region across lateral and dorsal views.",
    )

    assert res["status"] == "success"
    assert res["views_updated"] == 3
    assert res["backup_path"] is not None
    assert Path(res["backup_path"]).is_file()

    # Verify CSV file contents
    df_updated = pd.read_csv(manifest_p)
    spec_a_rows = df_updated[df_updated["specimen_id"] == "SPECIMEN_A"]
    assert len(spec_a_rows) == 3
    assert (spec_a_rows["candidate_label"] == "discolored_spotted").all()
    assert (spec_a_rows["review_status"] == "approved").all()
    assert (spec_a_rows["confidence"] == "high").all()
    assert (spec_a_rows["reviewer"] == "dr_agronomist").all()

    # SPECIMEN_B should remain untouched
    spec_b_rows = df_updated[df_updated["specimen_id"] == "SPECIMEN_B"]
    assert len(spec_b_rows) == 3
    assert (spec_b_rows["candidate_label"] == "pending_review").all()
    assert (spec_b_rows["review_status"] == "pending").all()


def test_save_specimen_review_dry_run_does_not_modify(staged_review_fixture):
    """Dry run validates without writing to CSV or creating backup."""
    manifest_p = staged_review_fixture["manifest_path"]
    mtime_before = manifest_p.stat().st_mtime

    res = save_specimen_review(
        manifest_path=manifest_p,
        specimen_id="SPECIMEN_A",
        candidate_label="broken_cracked",
        review_status="approved",
        confidence="high",
        reviewer="test_eval",
        diagnostic_notes="Test crack",
        dry_run=True,
    )

    assert res["dry_run"] is True
    assert res["backup_path"] is None
    assert manifest_p.stat().st_mtime == mtime_before

    df = pd.read_csv(manifest_p)
    spec_a_rows = df[df["specimen_id"] == "SPECIMEN_A"]
    assert (spec_a_rows["candidate_label"] == "pending_review").all()


def test_revert_specimen_review(staged_review_fixture):
    """Reverting a specimen returns it to pending_review across all views."""
    manifest_p = staged_review_fixture["manifest_path"]

    # First approve SPECIMEN_A
    save_specimen_review(
        manifest_path=manifest_p,
        specimen_id="SPECIMEN_A",
        candidate_label="shrivelled_damaged",
        review_status="approved",
        confidence="high",
        reviewer="reviewer_1",
        diagnostic_notes="Severe cotyledon shrinkage.",
    )

    # Revert SPECIMEN_A
    rev_res = revert_specimen_review(manifest_p, "SPECIMEN_A")
    assert rev_res["views_updated"] == 3

    df = pd.read_csv(manifest_p)
    spec_a_rows = df[df["specimen_id"] == "SPECIMEN_A"]
    assert (spec_a_rows["candidate_label"] == "pending_review").all()
    assert (spec_a_rows["review_status"] == "pending").all()


def test_restore_manifest_backup(staged_review_fixture):
    """Restoring from backup undoes all manual modifications."""
    manifest_p = staged_review_fixture["manifest_path"]

    # 1. Test custom named snapshot restore
    custom_backup = manifest_p.parent / "initial_state.backup.csv"
    create_manifest_backup(manifest_p, custom_backup)
    assert custom_backup.is_file()

    # Modify SPECIMEN_A
    save_specimen_review(
        manifest_path=manifest_p,
        specimen_id="SPECIMEN_A",
        candidate_label="sound_healthy",
        review_status="approved",
        confidence="high",
        reviewer="reviewer_1",
        diagnostic_notes="Perfect seed",
    )

    df_mod1 = pd.read_csv(manifest_p)
    assert (df_mod1[df_mod1["specimen_id"] == "SPECIMEN_A"]["review_status"] == "approved").all()

    # Automatic backup was created at manifest_p.with_suffix(".backup.csv")
    auto_backup = manifest_p.with_suffix(".backup.csv")
    assert auto_backup.is_file()

    # Restoring the automatic backup undoes the SPECIMEN_A modification
    restore_manifest_backup(manifest_p)
    df_restored = pd.read_csv(manifest_p)
    assert (df_restored["review_status"] == "pending").all()
    assert (df_restored["candidate_label"] == "pending_review").all()

    # 2. Modify both specimens, then restore from the custom initial snapshot
    save_specimen_review(
        manifest_path=manifest_p,
        specimen_id="SPECIMEN_A",
        candidate_label="sound_healthy",
        review_status="approved",
        confidence="high",
        reviewer="reviewer_1",
        diagnostic_notes="Perfect seed",
    )
    save_specimen_review(
        manifest_path=manifest_p,
        specimen_id="SPECIMEN_B",
        candidate_label="broken_cracked",
        review_status="approved",
        confidence="high",
        reviewer="reviewer_1",
        diagnostic_notes="Cracked seed",
    )
    df_both = pd.read_csv(manifest_p)
    assert (df_both["review_status"] == "approved").all()

    # Restore from initial custom snapshot
    restore_manifest_backup(manifest_p, custom_backup)
    df_restored_custom = pd.read_csv(manifest_p)
    assert (df_restored_custom["review_status"] == "pending").all()
    assert (df_restored_custom["candidate_label"] == "pending_review").all()


def test_save_specimen_review_nonexistent_specimen(staged_review_fixture):
    """Specifying an unknown specimen_id raises ValueError."""
    manifest_p = staged_review_fixture["manifest_path"]

    with pytest.raises(ValueError, match="Specimen ID 'UNKNOWN_SPECIMEN' not found"):
        save_specimen_review(
            manifest_path=manifest_p,
            specimen_id="UNKNOWN_SPECIMEN",
            candidate_label="sound_healthy",
            review_status="approved",
            confidence="high",
            reviewer="rev",
            diagnostic_notes="note",
        )
