"""Tests for the specimen-safe review-to-ingest bridge (``review_bridge.py``).

Covers eligibility rules, annotation generation, atomic writing, image staging
collision handling, the CLI dry-run contract, and the leakage regression proving
that all views of one specimen stay inside a single split partition.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from seedvision.data.real_review import REQUIRED_MANIFEST_COLUMNS, load_staging_manifest
from seedvision.data.review_bridge import (
    INGEST_ANNOTATION_COLUMNS,
    build_annotation_dataframe,
    build_bridge_report,
    build_ingest_annotations,
    plan_image_staging,
    select_approved_specimens,
    specimen_seed_id,
    stage_images,
    validate_no_duplicate_filenames,
    write_annotations_atomic,
)
from seedvision.data.splitting import create_leakage_resistant_splits


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _valid_view(filename: str, **overrides) -> dict:
    view = {
        "filename": filename,
        "view_id": "v01",
        "image_path": "/tmp/" + filename,
        "image_exists": True,
        "width": 128,
        "height": 96,
        "candidate_label": "sound_healthy",
        "review_status": "approved",
        "confidence": "high",
        "source_provenance": "ICAR_Test_Station",
        "variety": "Desi_Chickpea",
        "reviewer": "evaluator_01",
        "diagnostic_notes": "Reviewed specimen.",
    }
    view.update(overrides)
    return view


def _staging(specimens: dict) -> dict:
    built = {
        sid: {"specimen_id": sid, "views": list(views)}
        for sid, views in specimens.items()
    }
    return {
        "specimens": built,
        "total_specimens": len(built),
        "total_views": sum(len(v) for v in specimens.values()),
    }


def _default_row(filename: str, specimen_id: str, view_id: str, **overrides) -> dict:
    row = {
        "filename": filename,
        "specimen_id": specimen_id,
        "view_id": view_id,
        "candidate_label": "sound_healthy",
        "review_status": "approved",
        "confidence": "high",
        "source_provenance": "ICAR_Test_Station",
        "variety": "Desi_Chickpea",
        "reviewer": "evaluator_01",
        "diagnostic_notes": "Reviewed specimen.",
    }
    row.update(overrides)
    return row


def _stage_on_disk(tmp_path: Path, rows, *, make_images: bool = True):
    staging_dir = tmp_path / "real_staging"
    incoming = staging_dir / "incoming"
    incoming.mkdir(parents=True, exist_ok=True)
    if make_images:
        for row in rows:
            Image.new("RGB", (128, 96), color=(120, 150, 90)).save(
                incoming / row["filename"], format="JPEG"
            )
    manifest = staging_dir / "staging_manifest.csv"
    pd.DataFrame(rows, columns=REQUIRED_MANIFEST_COLUMNS).to_csv(manifest, index=False)
    return staging_dir, incoming, manifest


# --------------------------------------------------------------------------- #
# specimen_seed_id
# --------------------------------------------------------------------------- #
def test_specimen_seed_id_is_verbatim_and_class_independent():
    assert specimen_seed_id("REAL_CHICKPEA_001") == "REAL_CHICKPEA_001"
    assert specimen_seed_id("  REAL_CHICKPEA_002  ") == "REAL_CHICKPEA_002"


@pytest.mark.parametrize("value", ["", "   ", None])
def test_specimen_seed_id_rejects_empty(value):
    with pytest.raises(ValueError):
        specimen_seed_id(value)


# --------------------------------------------------------------------------- #
# Eligibility: select_approved_specimens
# --------------------------------------------------------------------------- #
def test_fully_approved_specimen_is_eligible():
    staging = _staging(
        {"REAL_CHICKPEA_001": [_valid_view("REAL_CHICKPEA_001_v01.jpg")]}
    )
    selection = select_approved_specimens(staging)
    assert [s["specimen_id"] for s in selection["eligible"]] == ["REAL_CHICKPEA_001"]
    assert selection["excluded"] == []


def test_pending_view_excludes_specimen():
    staging = _staging(
        {"S1": [_valid_view("S1_v01.jpg", review_status="pending")]}
    )
    selection = select_approved_specimens(staging)
    assert selection["eligible"] == []
    assert selection["excluded"] == [{"specimen_id": "S1", "reason": "has_pending_view"}]


def test_rejected_view_excludes_specimen():
    staging = _staging(
        {"S1": [_valid_view("S1_v01.jpg", review_status="rejected")]}
    )
    selection = select_approved_specimens(staging)
    assert selection["excluded"][0]["reason"] == "has_rejected_view"


def test_mixed_status_specimen_is_excluded():
    staging = _staging(
        {
            "S1": [
                _valid_view("S1_v01.jpg", review_status="approved"),
                _valid_view("S1_v02.jpg", review_status="pending"),
            ]
        }
    )
    selection = select_approved_specimens(staging)
    assert selection["eligible"] == []
    assert selection["excluded"][0]["reason"] == "has_pending_view"


def test_inconsistent_labels_exclude_specimen():
    staging = _staging(
        {
            "S1": [
                _valid_view("S1_v01.jpg", candidate_label="sound_healthy"),
                _valid_view("S1_v02.jpg", candidate_label="broken_cracked"),
            ]
        }
    )
    selection = select_approved_specimens(staging)
    assert selection["excluded"][0]["reason"] == "inconsistent_labels"


def test_pending_review_label_is_unresolved():
    staging = _staging(
        {"S1": [_valid_view("S1_v01.jpg", candidate_label="pending_review")]}
    )
    selection = select_approved_specimens(staging)
    assert selection["excluded"][0]["reason"] == "unresolved_label"


def test_unsupported_label_is_excluded():
    staging = _staging(
        {"S1": [_valid_view("S1_v01.jpg", candidate_label="not_a_real_class")]}
    )
    selection = select_approved_specimens(staging)
    assert selection["excluded"][0]["reason"] == "unsupported_label"


def test_inconsistent_variety_excludes_specimen():
    staging = _staging(
        {
            "S1": [
                _valid_view("S1_v01.jpg", variety="Desi_Chickpea"),
                _valid_view("S1_v02.jpg", variety="Kabuli_Chickpea"),
            ]
        }
    )
    selection = select_approved_specimens(staging)
    assert selection["excluded"][0]["reason"] == "inconsistent_variety"


def test_missing_variety_excludes_specimen():
    staging = _staging({"S1": [_valid_view("S1_v01.jpg", variety="")]})
    selection = select_approved_specimens(staging)
    assert selection["excluded"][0]["reason"] == "missing_variety"


def test_inconsistent_provenance_excludes_specimen():
    staging = _staging(
        {
            "S1": [
                _valid_view("S1_v01.jpg", source_provenance="Station_A"),
                _valid_view("S1_v02.jpg", source_provenance="Station_B"),
            ]
        }
    )
    selection = select_approved_specimens(staging)
    assert selection["excluded"][0]["reason"] == "inconsistent_provenance"


def test_missing_provenance_excludes_specimen():
    staging = _staging({"S1": [_valid_view("S1_v01.jpg", source_provenance="")]})
    selection = select_approved_specimens(staging)
    assert selection["excluded"][0]["reason"] == "missing_provenance"


def test_missing_reviewer_excludes_specimen():
    staging = _staging({"S1": [_valid_view("S1_v01.jpg", reviewer="")]})
    selection = select_approved_specimens(staging)
    assert selection["excluded"][0]["reason"] == "missing_reviewer"


def test_missing_diagnostic_notes_excludes_specimen():
    staging = _staging({"S1": [_valid_view("S1_v01.jpg", diagnostic_notes="")]})
    selection = select_approved_specimens(staging)
    assert selection["excluded"][0]["reason"] == "missing_diagnostic_notes"


def test_missing_image_excludes_specimen():
    staging = _staging({"S1": [_valid_view("S1_v01.jpg", image_exists=False)]})
    selection = select_approved_specimens(staging)
    assert selection["excluded"][0]["reason"] == "missing_image"


def test_missing_filename_excludes_specimen():
    staging = _staging({"S1": [_valid_view("")]})
    selection = select_approved_specimens(staging)
    assert selection["excluded"][0]["reason"] == "missing_filename"


def test_duplicate_view_filenames_exclude_specimen():
    staging = _staging(
        {
            "S1": [
                _valid_view("S1_v01.jpg"),
                _valid_view("S1_v01.jpg", view_id="v02"),
            ]
        }
    )
    selection = select_approved_specimens(staging)
    assert selection["excluded"][0]["reason"] == "duplicate_view_filenames"


def test_no_views_excludes_specimen():
    staging = _staging({"S1": []})
    selection = select_approved_specimens(staging)
    assert selection["excluded"][0]["reason"] == "no_views"


def test_missing_specimen_id_excludes_specimen():
    staging = {"specimens": {"": {"specimen_id": "", "views": [_valid_view("a.jpg")]}}}
    selection = select_approved_specimens(staging)
    assert selection["excluded"][0]["reason"] == "missing_specimen_id"


# --------------------------------------------------------------------------- #
# build_ingest_annotations
# --------------------------------------------------------------------------- #
def test_every_view_receives_the_same_seed_id():
    specimen = {
        "specimen_id": "REAL_CHICKPEA_001",
        "views": [
            _valid_view("REAL_CHICKPEA_001_v01.jpg"),
            _valid_view("REAL_CHICKPEA_001_v02.jpg"),
            _valid_view("REAL_CHICKPEA_001_v03.jpg"),
        ],
    }
    rows = build_ingest_annotations([specimen])
    assert len(rows) == 3
    assert {row["seed_id"] for row in rows} == {"REAL_CHICKPEA_001"}
    assert {row["class_name"] for row in rows} == {"sound_healthy"}


def test_distinct_specimens_get_distinct_seed_ids():
    specimens = [
        {"specimen_id": "S1", "views": [_valid_view("S1_v01.jpg")]},
        {"specimen_id": "S2", "views": [_valid_view("S2_v01.jpg")]},
    ]
    rows = build_ingest_annotations(specimens)
    assert {row["seed_id"] for row in rows} == {"S1", "S2"}


def test_build_ingest_annotations_is_deterministically_ordered():
    specimens = [
        {"specimen_id": "S2", "views": [_valid_view("S2_v01.jpg")]},
        {"specimen_id": "S1", "views": [_valid_view("S1_v02.jpg"), _valid_view("S1_v01.jpg")]},
    ]
    rows = build_ingest_annotations(specimens)
    assert [row["filename"] for row in rows] == ["S1_v01.jpg", "S1_v02.jpg", "S2_v01.jpg"]


def test_duplicate_source_filenames_fail_closed():
    specimens = [
        {"specimen_id": "S1", "views": [_valid_view("shared.jpg")]},
        {"specimen_id": "S2", "views": [_valid_view("shared.jpg")]},
    ]
    with pytest.raises(ValueError):
        build_ingest_annotations(specimens)


def test_validate_no_duplicate_filenames_direct():
    validate_no_duplicate_filenames(
        [{"filename": "a.jpg"}, {"filename": "b.jpg"}]
    )
    with pytest.raises(ValueError):
        validate_no_duplicate_filenames(
            [{"filename": "a.jpg"}, {"filename": "a.jpg"}]
        )


# --------------------------------------------------------------------------- #
# build_annotation_dataframe
# --------------------------------------------------------------------------- #
def test_annotation_dataframe_uses_canonical_schema():
    rows = build_ingest_annotations(
        [{"specimen_id": "S1", "views": [_valid_view("S1_v01.jpg")]}]
    )
    df = build_annotation_dataframe(rows)
    assert list(df.columns) == INGEST_ANNOTATION_COLUMNS
    assert len(df) == 1


# --------------------------------------------------------------------------- #
# write_annotations_atomic
# --------------------------------------------------------------------------- #
def test_write_annotations_atomic_creates_new_file(tmp_path: Path):
    df = build_annotation_dataframe(
        [{"filename": "a.jpg", "class_name": "sound_healthy", "seed_id": "S1", "batch_id": "", "variety": "Desi", "notes": "n"}]
    )
    out = tmp_path / "nested" / "annotations.csv"
    written = write_annotations_atomic(df, out)
    assert written == out
    assert out.is_file()
    reloaded = pd.read_csv(out, dtype=str)
    assert list(reloaded.columns) == INGEST_ANNOTATION_COLUMNS


def test_write_annotations_atomic_refuses_silent_overwrite(tmp_path: Path):
    out = tmp_path / "annotations.csv"
    out.write_text("original\n", encoding="utf-8")
    df = build_annotation_dataframe(
        [{"filename": "a.jpg", "class_name": "sound_healthy", "seed_id": "S1", "batch_id": "", "variety": "Desi", "notes": "n"}]
    )
    with pytest.raises(FileExistsError):
        write_annotations_atomic(df, out)
    assert out.read_text(encoding="utf-8") == "original\n"


def test_write_annotations_atomic_overwrite_is_explicit(tmp_path: Path):
    out = tmp_path / "annotations.csv"
    out.write_text("original\n", encoding="utf-8")
    df = build_annotation_dataframe(
        [{"filename": "a.jpg", "class_name": "sound_healthy", "seed_id": "S1", "batch_id": "", "variety": "Desi", "notes": "n"}]
    )
    write_annotations_atomic(df, out, overwrite=True)
    assert "seed_id" in out.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# plan_image_staging / stage_images
# --------------------------------------------------------------------------- #
def _staging_scenario(tmp_path: Path):
    incoming = tmp_path / "incoming"
    images = tmp_path / "images"
    incoming.mkdir()

    (incoming / "a.jpg").write_bytes(b"AAA")
    (incoming / "b.jpg").write_bytes(b"BBB")
    (incoming / "c.jpg").write_bytes(b"CCC")

    dest_b = images / "sound_healthy" / "b.jpg"
    dest_b.parent.mkdir(parents=True)
    dest_b.write_bytes(b"BBB")

    dest_c = images / "sound_healthy" / "c.jpg"
    dest_c.write_bytes(b"DIFFERENT")

    return incoming, images


def test_plan_image_staging_reports_all_statuses(tmp_path: Path):
    incoming, images = _staging_scenario(tmp_path)
    rows = [
        {"filename": "a.jpg", "class_name": "sound_healthy"},
        {"filename": "b.jpg", "class_name": "sound_healthy"},
        {"filename": "c.jpg", "class_name": "sound_healthy"},
        {"filename": "missing.jpg", "class_name": "sound_healthy"},
    ]
    plan = plan_image_staging(rows, incoming, images)
    statuses = {entry["filename"]: entry["status"] for entry in plan}
    assert statuses == {
        "a.jpg": "copy",
        "b.jpg": "identical",
        "c.jpg": "conflict",
        "missing.jpg": "missing_source",
    }


def test_plan_image_staging_is_read_only(tmp_path: Path):
    incoming, images = _staging_scenario(tmp_path)
    rows = [{"filename": "a.jpg", "class_name": "sound_healthy"}]
    plan_image_staging(rows, incoming, images)
    assert not (images / "sound_healthy" / "a.jpg").exists()


def test_stage_images_copies_new_and_skips_identical(tmp_path: Path):
    incoming, images = _staging_scenario(tmp_path)
    rows = [
        {"filename": "a.jpg", "class_name": "sound_healthy"},
        {"filename": "b.jpg", "class_name": "sound_healthy"},
    ]
    plan = plan_image_staging(rows, incoming, images)
    result = stage_images(plan)
    assert result["copied"] == 1
    assert result["skipped_identical"] == 1
    assert (images / "sound_healthy" / "a.jpg").read_bytes() == b"AAA"


def test_stage_images_refuses_conflicting_overwrite(tmp_path: Path):
    incoming, images = _staging_scenario(tmp_path)
    rows = [{"filename": "c.jpg", "class_name": "sound_healthy"}]
    plan = plan_image_staging(rows, incoming, images)
    with pytest.raises(FileExistsError):
        stage_images(plan)
    assert (images / "sound_healthy" / "c.jpg").read_bytes() == b"DIFFERENT"


def test_stage_images_raises_on_missing_source(tmp_path: Path):
    incoming, images = _staging_scenario(tmp_path)
    rows = [{"filename": "missing.jpg", "class_name": "sound_healthy"}]
    plan = plan_image_staging(rows, incoming, images)
    with pytest.raises(FileNotFoundError):
        stage_images(plan)


# --------------------------------------------------------------------------- #
# build_bridge_report
# --------------------------------------------------------------------------- #
def test_build_bridge_report_counts():
    staging = _staging(
        {
            "S1": [_valid_view("a.jpg")],
            "S2": [_valid_view("b.jpg", review_status="pending")],
        }
    )
    selection = select_approved_specimens(staging)
    rows = build_ingest_annotations(selection["eligible"])
    plan = [
        {
            "filename": "a.jpg",
            "class_name": "sound_healthy",
            "source": "a",
            "destination": "d",
            "status": "copy",
        }
    ]
    report = build_bridge_report(staging, selection, rows, plan)
    assert report["total_specimens"] == 2
    assert report["eligible_specimens"] == ["S1"]
    assert report["excluded_specimens"] == [{"specimen_id": "S2", "reason": "has_pending_view"}]
    assert report["annotation_rows"] == 1
    assert report["seed_ids"] == ["S1"]
    assert report["class_distribution"] == {"sound_healthy": 1}
    assert report["image_plan_counts"] == {"copy": 1}


# --------------------------------------------------------------------------- #
# Integration via load_staging_manifest
# --------------------------------------------------------------------------- #
def test_bridge_integration_selects_only_fully_approved(tmp_path: Path):
    rows = [
        _default_row(f"REAL_CHICKPEA_001_v{n:02d}.jpg", "REAL_CHICKPEA_001", f"v{n:02d}")
        for n in (1, 2, 3)
    ]
    rows += [
        _default_row(
            f"REAL_CHICKPEA_002_v{n:02d}.jpg",
            "REAL_CHICKPEA_002",
            f"v{n:02d}",
            candidate_label="pending_review",
            review_status="pending",
        )
        for n in (1, 2, 3)
    ]
    _, incoming, manifest = _stage_on_disk(tmp_path, rows)

    staging = load_staging_manifest(manifest, incoming)
    selection = select_approved_specimens(staging)
    assert [s["specimen_id"] for s in selection["eligible"]] == ["REAL_CHICKPEA_001"]
    assert selection["excluded"] == [
        {"specimen_id": "REAL_CHICKPEA_002", "reason": "has_pending_view"}
    ]

    out_rows = build_ingest_annotations(selection["eligible"])
    assert len(out_rows) == 3
    assert {row["seed_id"] for row in out_rows} == {"REAL_CHICKPEA_001"}


def test_bridge_excludes_specimen_with_missing_image_on_disk(tmp_path: Path):
    rows = [_default_row("REAL_CHICKPEA_003_v01.jpg", "REAL_CHICKPEA_003", "v01")]
    _, incoming, manifest = _stage_on_disk(tmp_path, rows, make_images=False)
    staging = load_staging_manifest(manifest, incoming)
    selection = select_approved_specimens(staging)
    assert selection["eligible"] == []
    assert selection["excluded"][0]["reason"] == "missing_image"


def test_bridge_excludes_inconsistent_labels_from_csv(tmp_path: Path):
    rows = [
        _default_row("S1_v01.jpg", "SPEC1", "v01", candidate_label="sound_healthy"),
        _default_row("S1_v02.jpg", "SPEC1", "v02", candidate_label="broken_cracked"),
    ]
    _, incoming, manifest = _stage_on_disk(tmp_path, rows)
    staging = load_staging_manifest(manifest, incoming)
    selection = select_approved_specimens(staging)
    assert selection["excluded"][0]["reason"] == "inconsistent_labels"


def test_bridge_excludes_missing_metadata_from_csv(tmp_path: Path):
    rows = [_default_row("S1_v01.jpg", "SPEC1", "v01", variety="")]
    _, incoming, manifest = _stage_on_disk(tmp_path, rows)
    staging = load_staging_manifest(manifest, incoming)
    selection = select_approved_specimens(staging)
    assert selection["excluded"][0]["reason"] == "missing_variety"


# --------------------------------------------------------------------------- #
# Leakage regression: specimen grouping must partition all views together
# --------------------------------------------------------------------------- #
def test_group_split_keeps_all_views_of_a_specimen_together():
    specimens = {}
    for i in range(1, 5):
        sid = f"REAL_CHICKPEA_A{i}"
        specimens[sid] = [
            _valid_view(f"{sid}_v{n:02d}.jpg", candidate_label="sound_healthy")
            for n in (1, 2, 3)
        ]
    for i in range(1, 5):
        sid = f"REAL_CHICKPEA_B{i}"
        specimens[sid] = [
            _valid_view(f"{sid}_v{n:02d}.jpg", candidate_label="broken_cracked")
            for n in (1, 2, 3)
        ]

    staging = _staging(specimens)
    selection = select_approved_specimens(staging)
    assert len(selection["eligible"]) == 8

    rows = build_ingest_annotations(selection["eligible"])
    df = build_annotation_dataframe(rows)
    assert len(df) == 24

    train_df, val_df, test_df = create_leakage_resistant_splits(df)

    # Rows are preserved across partitions.
    assert len(train_df) + len(val_df) + len(test_df) == 24

    # Every seed_id (physical specimen) lives in exactly one split.
    seed_to_splits: dict[str, set] = {}
    for partition, part_df in (("train", train_df), ("val", val_df), ("test", test_df)):
        for seed_id in part_df["seed_id"].unique():
            seed_to_splits.setdefault(seed_id, set()).add(partition)

    for seed_id, splits in seed_to_splits.items():
        assert len(splits) == 1, f"specimen {seed_id} crossed splits: {splits}"

    # All three views of each specimen remain co-located.
    for seed_id in df["seed_id"].unique():
        home = seed_to_splits[seed_id]
        rows_in = {
            name: int((part_df["seed_id"] == seed_id).sum())
            for name, part_df in (("train", train_df), ("val", val_df), ("test", test_df))
        }
        assert rows_in[next(iter(home))] == 3
        assert sum(rows_in.values()) == 3


# --------------------------------------------------------------------------- #
# CLI contract
# --------------------------------------------------------------------------- #
def test_cli_dry_run_writes_nothing_then_apply_writes(tmp_path, monkeypatch, capsys):
    import scripts.build_ingest_annotations as bridge_cli

    rows = [
        _default_row(f"REAL_CHICKPEA_001_v{n:02d}.jpg", "REAL_CHICKPEA_001", f"v{n:02d}")
        for n in (1, 2, 3)
    ]
    _, incoming, manifest = _stage_on_disk(tmp_path, rows)
    out = tmp_path / "annotations.csv"

    base = [
        "build_ingest_annotations.py",
        "--manifest",
        str(manifest),
        "--staging-dir",
        str(incoming),
        "--out",
        str(out),
    ]

    monkeypatch.setattr(sys, "argv", base)
    assert bridge_cli.main() == 0
    capsys.readouterr()
    assert not out.exists()

    monkeypatch.setattr(sys, "argv", base + ["--apply"])
    assert bridge_cli.main() == 0
    assert out.is_file()
    written = pd.read_csv(out, dtype=str)
    assert list(written.columns) == INGEST_ANNOTATION_COLUMNS
    assert set(written["seed_id"]) == {"REAL_CHICKPEA_001"}
    assert len(written) == 3

    # Re-applying without --overwrite must refuse rather than clobber.
    monkeypatch.setattr(sys, "argv", base + ["--apply"])
    assert bridge_cli.main() == 1


def test_cli_reports_action_required_when_no_specimen_is_eligible(
    tmp_path, monkeypatch, capsys
):
    import scripts.build_ingest_annotations as bridge_cli

    rows = [
        _default_row(
            "REAL_CHICKPEA_009_v01.jpg",
            "REAL_CHICKPEA_009",
            "v01",
            candidate_label="pending_review",
            review_status="pending",
        )
    ]
    _, incoming, manifest = _stage_on_disk(tmp_path, rows)
    out = tmp_path / "annotations.csv"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_ingest_annotations.py",
            "--manifest",
            str(manifest),
            "--staging-dir",
            str(incoming),
            "--out",
            str(out),
            "--apply",
        ],
    )
    assert bridge_cli.main() == 1
    assert not out.exists()


def test_cli_blocks_on_image_destination_conflict(tmp_path, monkeypatch, capsys):
    import scripts.build_ingest_annotations as bridge_cli

    rows = [_default_row("S1_v01.jpg", "SPEC1", "v01")]
    _, incoming, manifest = _stage_on_disk(tmp_path, rows)

    images_dir = tmp_path / "real_raw"
    dest = images_dir / "sound_healthy" / "S1_v01.jpg"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"DIFFERENT-BYTES")

    out = tmp_path / "annotations.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_ingest_annotations.py",
            "--manifest",
            str(manifest),
            "--staging-dir",
            str(incoming),
            "--out",
            str(out),
            "--images-dir",
            str(images_dir),
            "--stage-images",
            "--apply",
        ],
    )
    assert bridge_cli.main() == 1
    assert dest.read_bytes() == b"DIFFERENT-BYTES"
    assert not out.exists()
