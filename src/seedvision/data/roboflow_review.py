"""Roboflow review gallery generation, metadata enrichment, and auditable triage.

This module provides data-quality utilities for the Roboflow v8 crop manual-review
workflow, enforcing strict boundaries:
- Never automatically maps Roboflow source classes to SeedVision's four target classes.
- Does not automatically reject crops based solely on resolution (<28px); flags them instead.
- Provides reversible, auditable, and idempotent triage for confirmed foreign material.
- Preserves all original annotations, source splits, and source group IDs.
"""

from __future__ import annotations

import csv
import json
import shutil
import struct
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image

from seedvision.data.roboflow_preprocessing import (
    IMAGE_SUFFIXES,
    _load_dataset_config,
    _parse_annotation_line,
)

# Roboflow classes confirmed as non-pulse foreign matter (dockage/debris)
CONFIRMED_FOREIGN_MATERIAL_CLASSES = {
    "Material extrano inorganico",
    "Material extrano organico",
}

# The four official SeedVision classification target labels
SEEDVISION_TARGET_LABELS = {
    "sound_healthy",
    "discolored_spotted",
    "broken_cracked",
    "shrivelled_damaged",
}

DEFAULT_MIN_RESOLUTION = 28


def get_crop_dimensions(file_path: Path | str) -> tuple[int, int]:
    """Read image width and height directly from PNG IHDR header or via PIL fallback."""
    path = Path(file_path)
    try:
        with path.open("rb") as handle:
            header = handle.read(24)
            if len(header) >= 24 and header[:8] == b"\x89PNG\r\n\x1a\n":
                width, height = struct.unpack(">LL", header[16:24])
                return int(width), int(height)
    except Exception:
        pass

    with Image.open(path) as img:
        return img.size


def generate_crop_metadata(
    manifest_path: str | Path,
    output_path: str | Path | None = None,
    min_resolution: int = DEFAULT_MIN_RESOLUTION,
    max_workers: int = 16,
) -> list[dict[str, Any]]:
    """Compute crop dimensions and triage flags while preserving all source annotations.

    Adds width, height, aspect_ratio, is_low_resolution, and resolution_flag.
    Never modifies existing review decisions or original source annotations.
    """
    manifest = Path(manifest_path).expanduser().absolute()
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    root_dir = manifest.parent
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        return []

    def _enrich_row(row: dict[str, str]) -> dict[str, Any]:
        crop_rel = row["crop_path"]
        crop_full = root_dir / crop_rel
        try:
            width, height = get_crop_dimensions(crop_full)
        except Exception:
            width, height = 0, 0

        aspect_ratio = round(width / height, 3) if height > 0 else 0.0
        is_low_res = width < min_resolution or height < min_resolution
        resolution_flag = "low_resolution" if is_low_res else "standard"

        enriched = dict(row)
        enriched["width"] = width
        enriched["height"] = height
        enriched["aspect_ratio"] = aspect_ratio
        enriched["is_low_resolution"] = is_low_res
        enriched["resolution_flag"] = resolution_flag
        return enriched

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        enriched_rows = list(executor.map(_enrich_row, rows))

    if output_path is not None:
        out_file = Path(output_path).expanduser().absolute()
        out_file.parent.mkdir(parents=True, exist_ok=True)
        if out_file.suffix.lower() == ".json":
            out_file.write_text(json.dumps(enriched_rows, indent=2), encoding="utf-8")
        else:
            fieldnames = list(enriched_rows[0].keys())
            with out_file.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(enriched_rows)

    return enriched_rows


def triage_foreign_material(
    manifest_path: str | Path,
    backup_path: str | Path | None = None,
    revert: bool = False,
    dry_run: bool = False,
    reviewer: str = "policy:foreign_material_triage",
    reason: str = "foreign_material",
) -> dict[str, Any]:
    """Perform a reversible, auditable, idempotent triage operation on foreign material.

    Criteria:
    - Only rows with source_class_name in CONFIRMED_FOREIGN_MATERIAL_CLASSES are triaged.
    - Status is marked as 'rejected' with reason 'foreign_material'.
    - reviewed_label remains strictly blank (never mapped to SeedVision target labels).
    - Rows below 28px are NOT auto-rejected; they remain pending unless they are confirmed foreign material.
    - All other rows remain in their previous state ('pending').
    - A backup of the manifest is preserved before any modifications.
    - Idempotent: re-running on an already triaged or reverted manifest reports 0 rows changed.
    """
    manifest = Path(manifest_path).expanduser().absolute()
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    required_fields = {
        "crop_path",
        "source_image",
        "source_group_id",
        "source_split",
        "source_class_name",
        "review_status",
        "reviewed_label",
        "reviewer",
        "review_notes",
    }
    if not required_fields.issubset(fieldnames):
        raise ValueError(f"Manifest missing required review fields: {required_fields - set(fieldnames)}")

    rows_changed = 0
    foreign_material_count = 0
    updated_rows: list[dict[str, str]] = []

    for row in rows:
        row_copy = dict(row)
        source_class = row_copy["source_class_name"]
        is_foreign = source_class in CONFIRMED_FOREIGN_MATERIAL_CLASSES

        if is_foreign:
            foreign_material_count += 1

        if not revert:
            # Triage mode: Mark confirmed foreign material as rejected
            if is_foreign:
                needs_update = (
                    row_copy["review_status"] != "rejected"
                    or row_copy["review_notes"] != reason
                    or row_copy["reviewer"] != reviewer
                )
                if needs_update:
                    row_copy["review_status"] = "rejected"
                    row_copy["reviewed_label"] = ""  # Never map to SeedVision labels
                    row_copy["reviewer"] = reviewer
                    row_copy["review_notes"] = reason
                    rows_changed += 1
        else:
            # Revert mode: Restore previously triaged foreign material back to pending
            if is_foreign and row_copy["review_status"] == "rejected" and row_copy["review_notes"] == reason:
                row_copy["review_status"] = "pending"
                row_copy["reviewed_label"] = ""
                row_copy["reviewer"] = ""
                row_copy["review_notes"] = ""
                rows_changed += 1

        # Strict safety check: Never allow any row to have a SeedVision target label automatically
        if row_copy["reviewed_label"] in SEEDVISION_TARGET_LABELS:
            raise RuntimeError(
                f"Safety violation: Roboflow class '{source_class}' was assigned "
                f"target label '{row_copy['reviewed_label']}'."
            )

        updated_rows.append(row_copy)

    # Determine backup destination
    backup_file: Path | None = None
    if not dry_run and rows_changed > 0:
        if backup_path is not None:
            backup_file = Path(backup_path).expanduser().absolute()
        else:
            backup_file = manifest.with_name(f"{manifest.name}.bak")
        backup_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest, backup_file)

        # Write updated rows back atomically
        temp_file = manifest.with_suffix(".tmp")
        with temp_file.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(updated_rows)
        temp_file.replace(manifest)

    status_counts = Counter(r["review_status"] for r in updated_rows)
    return {
        "manifest": str(manifest),
        "action": "revert" if revert else "triage",
        "dry_run": dry_run,
        "backup_path": str(backup_file) if backup_file else None,
        "total_rows": len(rows),
        "rows_changed": rows_changed,
        "foreign_material_count": foreign_material_count,
        "pending_count": status_counts.get("pending", 0),
        "rejected_count": status_counts.get("rejected", 0),
        "approved_count": status_counts.get("approved", 0),
        "other_status_count": sum(c for s, c in status_counts.items() if s not in ("pending", "rejected", "approved")),
        "idempotent": rows_changed == 0 if (
            (not revert and status_counts.get("rejected", 0) >= foreign_material_count)
            or (revert and status_counts.get("rejected", 0) == 0)
        ) else False,
    }


def validate_human_decision(
    review_status: str,
    reviewed_label: str,
    reviewer: str,
    review_notes: str,
) -> None:
    """Validate a human-entered review decision against SeedVision's strict schema."""
    if review_status not in ("approved", "rejected", "pending"):
        raise ValueError(
            f"Invalid review_status '{review_status}'. Must be 'approved', 'rejected', or 'pending'."
        )

    if review_status == "approved":
        if reviewed_label not in SEEDVISION_TARGET_LABELS:
            raise ValueError(
                f"Invalid reviewed_label '{reviewed_label}' for approved crop. "
                f"Must be one of {sorted(SEEDVISION_TARGET_LABELS)}."
            )
        if not reviewer or not reviewer.strip():
            raise ValueError("Reviewer identifier is required when approving a crop.")
        if not review_notes or not review_notes.strip():
            raise ValueError("Diagnostic review_notes are required when approving a crop.")

    elif review_status == "rejected":
        if reviewed_label and reviewed_label.strip():
            raise ValueError(
                f"Rejected crop cannot have a reviewed_label ('{reviewed_label}'). It must be blank."
            )
        if not reviewer or not reviewer.strip():
            raise ValueError("Reviewer identifier is required when rejecting a crop.")
        if not review_notes or not review_notes.strip():
            raise ValueError("Rejection reason/notes are required when rejecting a crop.")

    elif review_status == "pending":
        if reviewed_label and reviewed_label.strip():
            raise ValueError("Pending crop must have a blank reviewed_label.")


def apply_manual_review_decisions(
    manifest_path: str | Path,
    decisions: list[dict[str, str]],
    backup_path: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Reversibly apply validated human decisions to manual_review.csv.

    Preserves source-group IDs, splits, and all unreferenced rows.
    Creates an automatic backup before writing any changes.
    """
    manifest = Path(manifest_path).expanduser().absolute()
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    if not decisions:
        return {"manifest": str(manifest), "rows_changed": 0, "decisions_applied": 0}

    decisions_by_crop: dict[str, dict[str, str]] = {}
    for d in decisions:
        crop_path = d.get("crop_path")
        if not crop_path:
            raise ValueError("Every decision must specify 'crop_path'.")
        status = d.get("review_status", "pending")
        label = d.get("reviewed_label", "")
        reviewer = d.get("reviewer", "")
        notes = d.get("review_notes", "")
        validate_human_decision(status, label, reviewer, notes)
        decisions_by_crop[crop_path] = {
            "review_status": status,
            "reviewed_label": label,
            "reviewer": reviewer,
            "review_notes": notes,
        }

    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    rows_changed = 0
    updated_rows = []
    matched_crops = set()

    for row in rows:
        crop_path = row["crop_path"]
        row_copy = dict(row)
        if crop_path in decisions_by_crop:
            matched_crops.add(crop_path)
            dec = decisions_by_crop[crop_path]
            needs_update = (
                row_copy["review_status"] != dec["review_status"]
                or row_copy["reviewed_label"] != dec["reviewed_label"]
                or row_copy["reviewer"] != dec["reviewer"]
                or row_copy["review_notes"] != dec["review_notes"]
            )
            if needs_update:
                row_copy["review_status"] = dec["review_status"]
                row_copy["reviewed_label"] = dec["reviewed_label"]
                row_copy["reviewer"] = dec["reviewer"]
                row_copy["review_notes"] = dec["review_notes"]
                rows_changed += 1
        updated_rows.append(row_copy)

    unmatched = set(decisions_by_crop.keys()) - matched_crops
    if unmatched:
        raise ValueError(f"Some crop paths in decisions were not found in manifest: {list(unmatched)[:5]}")

    backup_file: Path | None = None
    if not dry_run and rows_changed > 0:
        if backup_path is not None:
            backup_file = Path(backup_path).expanduser().absolute()
        else:
            backup_file = manifest.with_name(f"{manifest.name}.bak")
        backup_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest, backup_file)

        temp_file = manifest.with_suffix(".tmp")
        with temp_file.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(updated_rows)
        temp_file.replace(manifest)

    status_counts = Counter(r["review_status"] for r in updated_rows)
    return {
        "manifest": str(manifest),
        "dry_run": dry_run,
        "backup_path": str(backup_file) if backup_file else None,
        "total_rows": len(rows),
        "decisions_applied": len(decisions_by_crop),
        "rows_changed": rows_changed,
        "status_counts": dict(status_counts),
        "approved_count": status_counts.get("approved", 0),
        "rejected_count": status_counts.get("rejected", 0),
        "pending_count": status_counts.get("pending", 0),
    }


def revert_manual_review_decisions(
    manifest_path: str | Path,
    crop_paths: list[str],
    backup_path: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Reversibly reset decisions on specified crop paths back to pending."""
    decisions = [
        {
            "crop_path": p,
            "review_status": "pending",
            "reviewed_label": "",
            "reviewer": "",
            "review_notes": "",
        }
        for p in crop_paths
    ]
    return apply_manual_review_decisions(
        manifest_path=manifest_path,
        decisions=decisions,
        backup_path=backup_path,
        dry_run=dry_run,
    )


def extract_source_annotations(dataset_dir: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Parse YOLO label files across splits and map each relative image path to its annotations."""
    dataset = Path(dataset_dir).expanduser().absolute()
    if not dataset.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset}")

    names, splits = _load_dataset_config(dataset)
    source_annotations: dict[str, list[dict[str, Any]]] = {}

    for split in splits:
        label_root = dataset / split / "labels"
        if not label_root.is_dir():
            continue
        for label_path in sorted(label_root.glob("*.txt")):
            stem = label_path.stem
            image_dir = dataset / split / "images"
            matches = [p for p in image_dir.glob(f"{stem}.*") if p.suffix.lower() in IMAGE_SUFFIXES]
            if not matches:
                continue
            image_rel = f"{split}/images/{matches[0].name}"

            annotations: list[dict[str, Any]] = []
            try:
                lines = label_path.read_text(encoding="utf-8").splitlines()
            except Exception:
                continue

            for line_number, line in enumerate(lines, start=1):
                if not line.strip():
                    continue
                try:
                    ann = _parse_annotation_line(line, len(names))
                    annotations.append({
                        "line": line_number,
                        "class_id": ann.class_id,
                        "class_name": names[ann.class_id],
                        "geometry": ann.geometry,
                        "bbox": [round(float(v), 4) for v in ann.bbox],
                        "polygon": [[round(float(x), 4), round(float(y), 4)] for x, y in ann.polygon]
                        if ann.geometry == "polygon"
                        else [],
                    })
                except ValueError:
                    continue

            source_annotations[image_rel] = annotations

    return source_annotations


def sync_source_images(
    dataset_dir: str | Path,
    review_dir: str | Path,
    target_subfolder: str = "source_images",
) -> dict[str, int]:
    """Mirror source images from Roboflow dataset to review gallery directory.

    Allows browsers to load original images via relative paths without cross-drive security blocks.
    """
    dataset = Path(dataset_dir).expanduser().absolute()
    review = Path(review_dir).expanduser().absolute()
    if not dataset.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset}")

    dest_root = review / target_subfolder
    dest_root.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0
    total = 0

    _, splits = _load_dataset_config(dataset)
    for split in splits:
        image_dir = dataset / split / "images"
        if not image_dir.is_dir():
            continue
        for img_path in sorted(image_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            total += 1
            rel_dest = dest_root / split / "images" / img_path.name
            rel_dest.parent.mkdir(parents=True, exist_ok=True)

            if rel_dest.is_file() and rel_dest.stat().st_size == img_path.stat().st_size:
                skipped += 1
                continue

            shutil.copy2(img_path, rel_dest)
            copied += 1

    return {"copied": copied, "skipped": skipped, "total": total}


def get_crop_triage_flags(
    width: int,
    height: int,
    aspect_ratio: float,
    bbox: tuple[float, float, float, float] | None = None,
    min_resolution: int = DEFAULT_MIN_RESOLUTION,
) -> list[str]:
    """Compute non-destructive advisory data-quality flags for a crop."""
    flags: list[str] = []
    if width < min_resolution or height < min_resolution:
        flags.append("small_crop")
    if aspect_ratio < 0.7 or aspect_ratio > 1.5:
        flags.append("extreme_aspect_ratio")
    if bbox is not None and len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        if x1 <= 0.01 or y1 <= 0.01 or x2 >= 0.99 or y2 >= 0.99:
            flags.append("edge_crop")
    return flags


def validate_source_image_linkage(
    manifest_path: str | Path,
    dataset_dir: str | Path | None = None,
    source_images_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Verify that crops in manifest cleanly link to source images, group IDs, and annotation lines."""
    manifest = Path(manifest_path).expanduser().absolute()
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    dataset = Path(dataset_dir).expanduser().absolute() if dataset_dir else None
    local_source_images = Path(source_images_dir).expanduser().absolute() if source_images_dir else manifest.parent / "source_images"

    missing_local_images = []
    missing_dataset_images = []
    invalid_rows = []

    for r in rows:
        source_img = r.get("source_image", "")
        source_group = r.get("source_group_id", "")
        ann_line = r.get("annotation_line", "")

        if not source_img or not source_group or not ann_line:
            invalid_rows.append(r.get("crop_path", "unknown"))
            continue

        if local_source_images.is_dir():
            local_path = local_source_images / source_img
            if not local_path.is_file():
                missing_local_images.append(source_img)

        if dataset and dataset.is_dir():
            dataset_img_path = dataset / source_img
            if not dataset_img_path.is_file():
                missing_dataset_images.append(source_img)

    return {
        "manifest": str(manifest),
        "total_crops": len(rows),
        "linked_crops": len(rows) - len(invalid_rows),
        "invalid_linkage_count": len(invalid_rows),
        "missing_local_images_count": len(set(missing_local_images)),
        "missing_dataset_images_count": len(set(missing_dataset_images)),
        "local_images_available": local_source_images.is_dir() and len(missing_local_images) == 0,
        "dataset_images_available": dataset is not None and dataset.is_dir() and len(missing_dataset_images) == 0,
    }


