"""Comprehensive unit and regression tests for Roboflow review gallery, metadata, and triage."""

import csv
import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

from seedvision.data.roboflow_review import (
    CONFIRMED_FOREIGN_MATERIAL_CLASSES,
    SEEDVISION_TARGET_LABELS,
    extract_source_annotations,
    generate_crop_metadata,
    get_crop_dimensions,
    get_crop_triage_flags,
    sync_source_images,
    triage_foreign_material,
    validate_source_image_linkage,
)


@pytest.fixture
def staged_review_env(tmp_path: Path):
    """Create a mock Roboflow manual_review environment with varied crops."""
    review_dir = tmp_path / "roboflow_review"
    crops_dir = review_dir / "crops"
    crops_dir.mkdir(parents=True)

    # 1. Foreign material inorganic (30x20)
    img1 = crops_dir / "crop_inorg.png"
    Image.new("RGB", (30, 20), "gray").save(img1)

    # 2. Foreign material organic (15x12 - low res < 28px)
    img2 = crops_dir / "crop_org_lowres.png"
    Image.new("RGB", (15, 12), "brown").save(img2)

    # 3. Saludable standard (64x64)
    img3 = crops_dir / "crop_saludable.png"
    Image.new("RGB", (64, 64), "green").save(img3)

    # 4. Roto small (<28px) - Should NOT be rejected because it is pulse defect, not confirmed foreign matter!
    img4 = crops_dir / "crop_roto_lowres.png"
    Image.new("RGB", (20, 22), "yellow").save(img4)

    # 5. Danado (40x40)
    img5 = crops_dir / "crop_danado.png"
    Image.new("RGB", (40, 40), "orange").save(img5)

    manifest_path = review_dir / "manual_review.csv"
    rows = [
        {
            "crop_path": "crops/crop_inorg.png",
            "source_image": "train/images/img1.jpg",
            "source_group_id": "group_1",
            "source_split": "train",
            "source_class_id": "5",
            "source_class_name": "Material extrano inorganico",
            "annotation_line": "1",
            "geometry": "bbox",
            "review_status": "pending",
            "reviewed_label": "",
            "reviewer": "",
            "review_notes": "",
        },
        {
            "crop_path": "crops/crop_org_lowres.png",
            "source_image": "train/images/img1.jpg",
            "source_group_id": "group_1",
            "source_split": "train",
            "source_class_id": "6",
            "source_class_name": "Material extrano organico",
            "annotation_line": "2",
            "geometry": "bbox",
            "review_status": "pending",
            "reviewed_label": "",
            "reviewer": "",
            "review_notes": "",
        },
        {
            "crop_path": "crops/crop_saludable.png",
            "source_image": "train/images/img2.jpg",
            "source_group_id": "group_2",
            "source_split": "train",
            "source_class_id": "8",
            "source_class_name": "Saludable",
            "annotation_line": "1",
            "geometry": "bbox",
            "review_status": "pending",
            "reviewed_label": "",
            "reviewer": "",
            "review_notes": "",
        },
        {
            "crop_path": "crops/crop_roto_lowres.png",
            "source_image": "valid/images/img3.jpg",
            "source_group_id": "group_3",
            "source_split": "valid",
            "source_class_id": "7",
            "source_class_name": "Roto",
            "annotation_line": "1",
            "geometry": "bbox",
            "review_status": "pending",
            "reviewed_label": "",
            "reviewer": "",
            "review_notes": "",
        },
        {
            "crop_path": "crops/crop_danado.png",
            "source_image": "test/images/img4.jpg",
            "source_group_id": "group_4",
            "source_split": "test",
            "source_class_id": "1",
            "source_class_name": "Danado",
            "annotation_line": "1",
            "geometry": "bbox",
            "review_status": "pending",
            "reviewed_label": "",
            "reviewer": "",
            "review_notes": "",
        },
    ]

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return review_dir, manifest_path


def test_get_crop_dimensions(staged_review_env):
    review_dir, _ = staged_review_env
    dim1 = get_crop_dimensions(review_dir / "crops/crop_inorg.png")
    dim2 = get_crop_dimensions(review_dir / "crops/crop_org_lowres.png")
    assert dim1 == (30, 20)
    assert dim2 == (15, 12)


def test_metadata_generation_preserves_annotations_and_grouping(staged_review_env):
    review_dir, manifest_path = staged_review_env
    metadata_out = review_dir / "crop_metadata.csv"

    metadata = generate_crop_metadata(manifest_path, output_path=metadata_out)

    assert len(metadata) == 5
    assert metadata_out.is_file()

    with metadata_out.open(newline="", encoding="utf-8") as f:
        saved_rows = list(csv.DictReader(f))

    assert len(saved_rows) == 5
    # Verify dimensions added
    assert saved_rows[0]["width"] == "30"
    assert saved_rows[0]["height"] == "20"
    assert saved_rows[0]["aspect_ratio"] == "1.5"
    assert saved_rows[0]["resolution_flag"] == "low_resolution"  # 20 < 28

    # Verify source annotations preserved
    assert saved_rows[0]["source_group_id"] == "group_1"
    assert saved_rows[0]["source_split"] == "train"
    assert saved_rows[0]["source_class_name"] == "Material extrano inorganico"
    assert saved_rows[0]["review_status"] == "pending"

    # Check non-foreign low-res row
    roto_row = next(r for r in saved_rows if r["source_class_name"] == "Roto")
    assert roto_row["is_low_resolution"].lower() == "true"
    assert roto_row["source_group_id"] == "group_3"


def test_low_resolution_flag_does_not_auto_reject(staged_review_env):
    """Crops below 28px must NOT be automatically rejected unless confirmed foreign material."""
    _, manifest_path = staged_review_env

    triage_result = triage_foreign_material(manifest_path)
    assert triage_result["rows_changed"] == 2  # Only the 2 foreign material crops!

    with manifest_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # crop_roto_lowres is 20x22 (< 28px), but it is 'Roto' (pulse seed defect)
    # It must remain pending, NOT rejected!
    roto_row = next(r for r in rows if r["source_class_name"] == "Roto")
    assert roto_row["review_status"] == "pending"
    assert roto_row["reviewed_label"] == ""


def test_reversible_foreign_material_triage_and_audit_backup(staged_review_env):
    review_dir, manifest_path = staged_review_env
    backup_file = review_dir / "manual_review_audit_backup.csv"

    # Step 1: Triage
    triage_res = triage_foreign_material(manifest_path, backup_path=backup_file)
    assert triage_res["rows_changed"] == 2
    assert triage_res["rejected_count"] == 2
    assert triage_res["pending_count"] == 3
    assert backup_file.is_file()

    # Verify rows in manifest
    with manifest_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        if r["source_class_name"] in CONFIRMED_FOREIGN_MATERIAL_CLASSES:
            assert r["review_status"] == "rejected"
            assert r["review_notes"] == "foreign_material"
            assert r["reviewed_label"] == ""
        else:
            assert r["review_status"] == "pending"

    # Step 2: Idempotency check (triage again)
    idempotent_res = triage_foreign_material(manifest_path)
    assert idempotent_res["rows_changed"] == 0
    assert idempotent_res["idempotent"] is True

    # Step 3: Revert
    revert_res = triage_foreign_material(manifest_path, revert=True)
    assert revert_res["rows_changed"] == 2
    assert revert_res["rejected_count"] == 0
    assert revert_res["pending_count"] == 5

    with manifest_path.open(newline="", encoding="utf-8") as f:
        reverted_rows = list(csv.DictReader(f))
    assert all(r["review_status"] == "pending" for r in reverted_rows)
    assert all(r["review_notes"] == "" for r in reverted_rows)

    # Step 4: Revert idempotency
    revert_idempotent_res = triage_foreign_material(manifest_path, revert=True)
    assert revert_idempotent_res["rows_changed"] == 0


def test_never_maps_roboflow_to_seedvision_target_labels(staged_review_env):
    _, manifest_path = staged_review_env
    triage_foreign_material(manifest_path)

    with manifest_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        assert r["reviewed_label"] not in SEEDVISION_TARGET_LABELS
        assert r["reviewed_label"] == ""


def test_source_group_integrity_preserved(staged_review_env):
    _, manifest_path = staged_review_env
    with manifest_path.open(newline="", encoding="utf-8") as f:
        original_groups = [r["source_group_id"] for r in csv.DictReader(f)]

    triage_foreign_material(manifest_path)

    with manifest_path.open(newline="", encoding="utf-8") as f:
        updated_groups = [r["source_group_id"] for r in csv.DictReader(f)]

    assert original_groups == updated_groups


def test_gallery_generation_contains_pagination_filters_and_dimensions(staged_review_env, monkeypatch):
    from scripts.build_roboflow_review_gallery import main as gallery_main

    review_dir, manifest_path = staged_review_env
    output_html = review_dir / "test_gallery.html"

    # Run gallery builder via CLI entrypoint
    cli_args = [
        "build_roboflow_review_gallery.py",
        "--manifest",
        str(manifest_path),
        "--output",
        str(output_html),
        "--page-size",
        "2",
        "--batch-size",
        "2",
    ]
    monkeypatch.setattr("sys.argv", cli_args)
    assert gallery_main() == 0
    assert output_html.is_file()

    content = output_html.read_text(encoding="utf-8")

    # Client-side pagination controls
    assert 'id="btn-prev"' in content
    assert 'id="btn-next"' in content
    assert 'id="page-indicator"' in content
    assert 'id="page-size-select"' in content

    # Batch navigation controls
    assert 'id="btn-batch-prev"' in content
    assert 'id="btn-batch-next"' in content
    assert 'id="batch-indicator"' in content

    # Source-class and review-status filters
    assert 'id="source-class-filter"' in content
    assert 'id="review-status-filter"' in content
    assert 'id="resolution-filter"' in content
    assert "Material extrano inorganico" in content
    assert "Saludable" in content

    # Crop dimensions and resolution tags
    assert "Dimensions:" in content
    assert "px" in content
    assert "AR:" in content
    assert "Low Res" in content

    # JSON dataset embedded for client-side pagination
    assert 'id="crop-data"' in content

    # Human decision fields and export controls
    assert "Human Verification Decision" in content
    assert 'class="input-status"' in content
    assert 'class="input-label"' in content
    assert 'id="btn-export-decisions"' in content
    assert 'id="global-reviewer"' in content

    # Image preview container scaling and visibility
    assert "height: 240px;" in content
    assert "object-fit: contain;" in content
    assert "max-width: 100%;" in content
    assert "max-height: 100%;" in content
    assert "linear-gradient(45deg" in content  # Checkerboard pattern for RGBA contrast

    # Click-to-zoom / lightbox functionality
    assert 'id="lightbox-modal"' in content
    assert 'id="lightbox-close"' in content
    assert 'class="lightbox-img-wrap"' in content
    assert 'id="lightbox-img"' in content


def test_validate_human_decision_schema():
    from seedvision.data.roboflow_review import validate_human_decision

    # Valid approved
    validate_human_decision("approved", "sound_healthy", "om", "clear cotyledon")

    # Invalid approved label
    with pytest.raises(ValueError, match="Invalid reviewed_label"):
        validate_human_decision("approved", "saludable_invalid", "om", "note")

    # Approved without reviewer
    with pytest.raises(ValueError, match="Reviewer identifier is required"):
        validate_human_decision("approved", "broken_cracked", "", "note")

    # Approved without notes
    with pytest.raises(ValueError, match="Diagnostic review_notes are required"):
        validate_human_decision("approved", "broken_cracked", "om", "")

    # Valid rejected
    validate_human_decision("rejected", "", "om", "clipped_boundary")

    # Invalid rejected with non-empty label
    with pytest.raises(ValueError, match="Rejected crop cannot have a reviewed_label"):
        validate_human_decision("rejected", "broken_cracked", "om", "note")

    # Rejected without notes
    with pytest.raises(ValueError, match="Rejection reason/notes are required"):
        validate_human_decision("rejected", "", "om", "")


def test_apply_manual_review_decisions_reversible_and_audited(staged_review_env):
    from seedvision.data.roboflow_review import (
        apply_manual_review_decisions,
        revert_manual_review_decisions,
    )

    review_dir, manifest_path = staged_review_env
    backup_file = review_dir / "audit_before_manual_decisions.csv"

    decisions = [
        {
            "crop_path": "crops/crop_saludable.png",
            "review_status": "approved",
            "reviewed_label": "sound_healthy",
            "reviewer": "expert_evaluator",
            "review_notes": "intact testa, uniform coloration",
        },
        {
            "crop_path": "crops/crop_roto_lowres.png",
            "review_status": "rejected",
            "reviewed_label": "",
            "reviewer": "expert_evaluator",
            "review_notes": "sub_resolution_insufficient_fissure_detail",
        },
    ]

    # Apply decisions
    res = apply_manual_review_decisions(manifest_path, decisions, backup_path=backup_file)
    assert res["rows_changed"] == 2
    assert res["approved_count"] == 1
    assert res["rejected_count"] == 1
    assert backup_file.is_file()

    with manifest_path.open(newline="", encoding="utf-8") as f:
        rows = {r["crop_path"]: r for r in csv.DictReader(f)}

    assert rows["crops/crop_saludable.png"]["review_status"] == "approved"
    assert rows["crops/crop_saludable.png"]["reviewed_label"] == "sound_healthy"
    assert rows["crops/crop_saludable.png"]["reviewer"] == "expert_evaluator"

    assert rows["crops/crop_roto_lowres.png"]["review_status"] == "rejected"
    assert rows["crops/crop_roto_lowres.png"]["reviewed_label"] == ""

    # Revert decisions
    revert_res = revert_manual_review_decisions(
        manifest_path, ["crops/crop_saludable.png", "crops/crop_roto_lowres.png"]
    )
    assert revert_res["rows_changed"] == 2

    with manifest_path.open(newline="", encoding="utf-8") as f:
        reverted_rows = {r["crop_path"]: r for r in csv.DictReader(f)}

    assert reverted_rows["crops/crop_saludable.png"]["review_status"] == "pending"
    assert reverted_rows["crops/crop_saludable.png"]["reviewed_label"] == ""
    assert reverted_rows["crops/crop_roto_lowres.png"]["review_status"] == "pending"


def test_apply_manual_review_cli(staged_review_env, monkeypatch):
    from scripts.apply_manual_review import main as apply_cli_main

    _, manifest_path = staged_review_env

    # Run single crop approval
    cli_args = [
        "apply_manual_review.py",
        "--manifest",
        str(manifest_path),
        "--crop-path",
        "crops/crop_saludable.png",
        "--status",
        "approved",
        "--label",
        "sound_healthy",
        "--reviewer",
        "test_reviewer",
        "--notes",
        "verified intact seed",
    ]
    monkeypatch.setattr("sys.argv", cli_args)
    assert apply_cli_main() == 0

    with manifest_path.open(newline="", encoding="utf-8") as f:
        row = next(r for r in csv.DictReader(f) if r["crop_path"] == "crops/crop_saludable.png")
    assert row["review_status"] == "approved"
    assert row["reviewed_label"] == "sound_healthy"

    # Revert via CLI
    cli_revert = [
        "apply_manual_review.py",
        "--manifest",
        str(manifest_path),
        "--crop-path",
        "crops/crop_saludable.png",
        "--revert",
    ]
    monkeypatch.setattr("sys.argv", cli_revert)
    assert apply_cli_main() == 0

    with manifest_path.open(newline="", encoding="utf-8") as f:
        row = next(r for r in csv.DictReader(f) if r["crop_path"] == "crops/crop_saludable.png")
    assert row["review_status"] == "pending"
    assert row["reviewed_label"] == ""


def test_source_image_linkage_validation(staged_review_env):
    review_dir, manifest_path = staged_review_env

    # Without source images copied yet
    res = validate_source_image_linkage(manifest_path)
    assert res["total_crops"] == 5
    assert res["linked_crops"] == 5
    assert res["invalid_linkage_count"] == 0
    assert res["local_images_available"] is False

    # Create local source images to satisfy linkage
    img_dir_train = review_dir / "source_images" / "train" / "images"
    img_dir_train.mkdir(parents=True)
    Image.new("RGB", (640, 640), "white").save(img_dir_train / "img1.jpg")
    Image.new("RGB", (640, 640), "white").save(img_dir_train / "img2.jpg")

    img_dir_valid = review_dir / "source_images" / "valid" / "images"
    img_dir_valid.mkdir(parents=True)
    Image.new("RGB", (640, 640), "white").save(img_dir_valid / "img3.jpg")

    img_dir_test = review_dir / "source_images" / "test" / "images"
    img_dir_test.mkdir(parents=True)
    Image.new("RGB", (640, 640), "white").save(img_dir_test / "img4.jpg")

    res_after = validate_source_image_linkage(manifest_path)
    assert res_after["linked_crops"] == 5
    assert res_after["local_images_available"] is True
    assert res_after["missing_local_images_count"] == 0


def test_annotation_overlays_extraction_and_distinction(tmp_path: Path):
    # Create mock Roboflow YOLO dataset
    dataset_dir = tmp_path / "mock_dataset"
    train_lbl_dir = dataset_dir / "train" / "labels"
    train_img_dir = dataset_dir / "train" / "images"
    train_lbl_dir.mkdir(parents=True)
    train_img_dir.mkdir(parents=True)

    # data.yaml
    (dataset_dir / "data.yaml").write_text(
        "names:\n  0: Saludable\n  1: Danado\n  2: Roto\n", encoding="utf-8"
    )

    # Image
    Image.new("RGB", (640, 640), "gray").save(train_img_dir / "sample_img.jpg")

    # Annotations: 1 bbox, 1 polygon
    lbl_content = (
        "0 0.25 0.25 0.1 0.1\n"
        "1 0.6 0.6 0.05 0.05\n"
        "2 0.7 0.7 0.75 0.7 0.75 0.75 0.7 0.75\n"
    )
    (train_lbl_dir / "sample_img.txt").write_text(lbl_content, encoding="utf-8")

    anns_map = extract_source_annotations(dataset_dir)
    assert "train/images/sample_img.jpg" in anns_map
    anns = anns_map["train/images/sample_img.jpg"]
    assert len(anns) == 3

    # Check line 1 (selected seed candidate)
    line1 = next(a for a in anns if a["line"] == 1)
    assert line1["class_name"] == "Saludable"
    assert line1["geometry"] == "bbox"
    assert len(line1["bbox"]) == 4
    assert line1["bbox"] == [0.2, 0.2, 0.3, 0.3]

    # Check line 3 (polygon candidate)
    line3 = next(a for a in anns if a["line"] == 3)
    assert line3["class_name"] == "Roto"
    assert line3["geometry"] == "polygon"
    assert len(line3["polygon"]) == 4

    # Verify nearby seeds are distinct from selected seed
    nearby = [a for a in anns if a["line"] != 1]
    assert len(nearby) == 2
    assert all(a["line"] != 1 for a in nearby)


def test_missing_source_image_graceful_handling(staged_review_env, monkeypatch):
    from scripts.build_roboflow_review_gallery import main as gallery_main

    review_dir, manifest_path = staged_review_env
    output_html = review_dir / "test_missing_gallery.html"

    # Run without source-dataset argument and with nonexistent dataset
    cli_args = [
        "build_roboflow_review_gallery.py",
        "--manifest",
        str(manifest_path),
        "--output",
        str(output_html),
        "--source-dataset",
        str(review_dir / "nonexistent_dataset"),
        "--page-size",
        "5",
    ]
    monkeypatch.setattr("sys.argv", cli_args)
    # Must exit cleanly with 0 and write the gallery with fallback error messaging
    assert gallery_main() == 0
    assert output_html.is_file()

    content = output_html.read_text(encoding="utf-8")
    assert 'id="source-image-modal"' in content
    assert 'id="source-error-msg"' in content
    assert "Source Image Not Found" in content


def test_image_resolution_warnings_and_upscaled_labels(staged_review_env, monkeypatch):
    from scripts.build_roboflow_review_gallery import main as gallery_main

    review_dir, manifest_path = staged_review_env
    output_html = review_dir / "test_resolution_gallery.html"

    cli_args = [
        "build_roboflow_review_gallery.py",
        "--manifest",
        str(manifest_path),
        "--output",
        str(output_html),
        "--page-size",
        "5",
    ]
    monkeypatch.setattr("sys.argv", cli_args)
    assert gallery_main() == 0
    assert output_html.is_file()

    content = output_html.read_text(encoding="utf-8")

    # Check upscaling badge
    assert "badge-upscaled" in content
    assert "Upscaled" in content

    # Check resolution triage flags and warnings
    assert "Small Crop (&lt;28px)" in content
    assert 'id="quality-filter"' in content
    assert 'data-dims="15&times;12"' in content or 'data-dims="30&times;20"' in content

    # Triage flags utility
    small_flags = get_crop_triage_flags(20, 22, 0.909)
    assert "small_crop" in small_flags
    assert "extreme_aspect_ratio" not in small_flags

    extreme_flags = get_crop_triage_flags(45, 15, 3.0)
    assert "extreme_aspect_ratio" in extreme_flags

    edge_flags = get_crop_triage_flags(30, 30, 1.0, bbox=(0.0, 0.2, 0.05, 0.25))
    assert "edge_crop" in edge_flags


