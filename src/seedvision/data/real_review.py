"""Human review and verification workflow for staged real seed photographs.

Provides validation, specimen grouping, consistent specimen-level label assignment,
auditable backup generation, and rollback functionality.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional

import pandas as pd
from PIL import Image

ALLOWED_TARGET_LABELS: list[str] = [
    "sound_healthy",
    "discolored_spotted",
    "broken_cracked",
    "shrivelled_damaged",
    "pending_review",
]

ALLOWED_REVIEW_STATUSES: list[str] = [
    "pending",
    "approved",
    "rejected",
]

ALLOWED_CONFIDENCES: list[str] = [
    "high",
    "medium",
    "low",
]

REQUIRED_MANIFEST_COLUMNS: list[str] = [
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
]


def load_staging_manifest(
    manifest_path: str | Path,
    incoming_dir: str | Path,
) -> dict[str, Any]:
    """Load and validate the staging manifest and verify image existence on disk.
    
    Returns:
        Dictionary containing:
        - 'df': the loaded pandas DataFrame
        - 'specimens': dict grouped by specimen_id with views and image metadata
        - 'total_specimens': count of unique specimens
        - 'total_views': count of individual image rows
        - 'status_counts': counts by review_status
        - 'class_counts': counts by candidate_label
        - 'missing_images': list of files in manifest missing from disk
    """
    manifest_p = Path(manifest_path).resolve()
    incoming_p = Path(incoming_dir).resolve()

    if not manifest_p.is_file():
        raise FileNotFoundError(f"Staging manifest not found: {manifest_p}")

    df = pd.read_csv(manifest_p, dtype=str).fillna("")

    # Verify column presence
    missing_cols = set(REQUIRED_MANIFEST_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Manifest missing required columns: {sorted(missing_cols)}")

    specimens: dict[str, dict[str, Any]] = {}
    missing_images: list[str] = []

    for _, row in df.iterrows():
        fname = str(row["filename"]).strip()
        specimen_id = str(row["specimen_id"]).strip()
        view_id = str(row["view_id"]).strip()
        cand_label = str(row["candidate_label"]).strip().lower()
        rev_status = str(row["review_status"]).strip().lower()
        confidence = str(row["confidence"]).strip().lower()
        provenance = str(row["source_provenance"]).strip()
        variety = str(row["variety"]).strip()
        reviewer = str(row["reviewer"]).strip()
        notes = str(row["diagnostic_notes"]).strip()

        img_file = incoming_p / fname
        img_exists = img_file.is_file()
        width, height = (0, 0)

        if img_exists:
            try:
                with Image.open(img_file) as im:
                    width, height = im.size
            except Exception:
                width, height = (0, 0)
        else:
            missing_images.append(fname)

        view_entry = {
            "filename": fname,
            "view_id": view_id,
            "image_path": str(img_file) if img_exists else None,
            "image_exists": img_exists,
            "width": width,
            "height": height,
            "candidate_label": cand_label,
            "review_status": rev_status,
            "confidence": confidence,
            "source_provenance": provenance,
            "variety": variety,
            "reviewer": reviewer,
            "diagnostic_notes": notes,
        }

        if specimen_id not in specimens:
            specimens[specimen_id] = {
                "specimen_id": specimen_id,
                "candidate_label": cand_label,
                "review_status": rev_status,
                "confidence": confidence,
                "source_provenance": provenance,
                "variety": variety,
                "reviewer": reviewer,
                "diagnostic_notes": notes,
                "views": [],
            }
        specimens[specimen_id]["views"].append(view_entry)

    status_counts = df["review_status"].str.strip().str.lower().value_counts().to_dict()
    class_counts = df["candidate_label"].str.strip().str.lower().value_counts().to_dict()

    return {
        "df": df,
        "manifest_path": str(manifest_p),
        "incoming_dir": str(incoming_p),
        "specimens": specimens,
        "total_specimens": len(specimens),
        "total_views": len(df),
        "status_counts": status_counts,
        "class_counts": class_counts,
        "missing_images": missing_images,
    }


def validate_review_decision(
    candidate_label: str,
    review_status: str,
    confidence: str,
    reviewer: str,
    diagnostic_notes: str,
) -> dict[str, str]:
    """Validate a candidate review decision against SeedVision AI safety rules.
    
    Rules:
    - candidate_label must be in ALLOWED_TARGET_LABELS.
    - If candidate_label == "pending_review", review_status MUST be "pending".
    - If review_status == "approved":
        - candidate_label must be one of the 4 target defect classes.
        - reviewer must not be empty.
        - confidence must be 'high' or 'medium' (low confidence cannot be approved).
        - diagnostic_notes must not be empty.
    - If confidence is "low", review_status is forced to "pending" (uncertain specimens stay pending).
    - If review_status == "rejected":
        - diagnostic_notes must provide the rejection reason.
    """
    cand_label = str(candidate_label or "").strip().lower()
    rev_status = str(review_status or "").strip().lower()
    conf = str(confidence or "").strip().lower()
    rev_user = str(reviewer or "").strip()
    notes = str(diagnostic_notes or "").strip()

    if cand_label not in ALLOWED_TARGET_LABELS:
        raise ValueError(
            f"Invalid candidate_label '{cand_label}'. Must be one of {ALLOWED_TARGET_LABELS}"
        )

    if rev_status not in ALLOWED_REVIEW_STATUSES:
        raise ValueError(
            f"Invalid review_status '{rev_status}'. Must be one of {ALLOWED_REVIEW_STATUSES}"
        )

    if conf and conf not in ALLOWED_CONFIDENCES:
        raise ValueError(
            f"Invalid confidence '{conf}'. Must be one of {ALLOWED_CONFIDENCES}"
        )

    # Uncertainty rule: low confidence must never be approved
    if conf == "low" and rev_status == "approved":
        raise ValueError(
            "Safety constraint violation: Low-confidence specimens cannot be marked 'approved'. "
            "Keep review_status as 'pending' for uncertain samples."
        )

    # Pending label rule
    if cand_label == "pending_review":
        if rev_status == "approved":
            raise ValueError(
                "Cannot approve a specimen with candidate_label='pending_review'. "
                "Select one of the 4 target defect classes or keep review_status='pending'."
            )
        if rev_status != "rejected":
            rev_status = "pending"

    # Approval requirements
    if rev_status == "approved":
        if not rev_user:
            raise ValueError("Reviewer identity is required to approve a specimen.")
        if not notes:
            raise ValueError("Diagnostic notes are required to approve a specimen.")
        if not conf:
            conf = "high"

    # Rejection requirements
    if rev_status == "rejected":
        if not notes:
            raise ValueError("Diagnostic notes / rejection rationale are required when rejecting a specimen.")

    return {
        "candidate_label": cand_label,
        "review_status": rev_status,
        "confidence": conf,
        "reviewer": rev_user,
        "diagnostic_notes": notes,
    }


def create_manifest_backup(manifest_path: Path, backup_path: Optional[Path] = None) -> Path:
    """Create a backup of the staging manifest before modifying it."""
    manifest_p = Path(manifest_path).resolve()
    if not manifest_p.is_file():
        raise FileNotFoundError(f"Manifest not found for backup: {manifest_p}")

    if backup_path:
        dest_backup = Path(backup_path).resolve()
        dest_backup.parent.mkdir(parents=True, exist_ok=True)
    else:
        dest_backup = manifest_p.with_suffix(".backup.csv")

    shutil.copy2(manifest_p, dest_backup)
    return dest_backup


def restore_manifest_backup(manifest_path: str | Path, backup_path: Optional[str | Path] = None) -> Path:
    """Restore the staging manifest CSV from a previously created backup file."""
    manifest_p = Path(manifest_path).resolve()
    if backup_path:
        backup_p = Path(backup_path).resolve()
    else:
        backup_p = manifest_p.with_suffix(".backup.csv")

    if not backup_p.is_file():
        raise FileNotFoundError(f"Backup manifest not found: {backup_p}")

    shutil.copy2(backup_p, manifest_p)
    return manifest_p


def save_specimen_review(
    manifest_path: str | Path,
    specimen_id: str,
    candidate_label: str,
    review_status: str,
    confidence: str,
    reviewer: str,
    diagnostic_notes: str,
    backup_path: Optional[str | Path] = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Apply a validated specimen-level review decision across all views of a specimen.
    
    Enforces that all views of the specimen receive the exact same consistent label,
    creates a backup before modification, and writes the updated CSV atomically.
    """
    manifest_p = Path(manifest_path).resolve()
    if not manifest_p.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_p}")

    specimen_id = str(specimen_id).strip()
    if not specimen_id:
        raise ValueError("specimen_id cannot be empty.")

    # Validate decision
    validated = validate_review_decision(
        candidate_label=candidate_label,
        review_status=review_status,
        confidence=confidence,
        reviewer=reviewer,
        diagnostic_notes=diagnostic_notes,
    )

    df = pd.read_csv(manifest_p, dtype=str).fillna("")
    mask = df["specimen_id"].str.strip() == specimen_id
    matching_count = int(mask.sum())

    if matching_count == 0:
        raise ValueError(f"Specimen ID '{specimen_id}' not found in manifest {manifest_p.name}")

    backup_file = None
    if not dry_run:
        backup_file = create_manifest_backup(manifest_p, Path(backup_path) if backup_path else None)

        # Update all views of this specimen consistently
        df.loc[mask, "candidate_label"] = validated["candidate_label"]
        df.loc[mask, "review_status"] = validated["review_status"]
        df.loc[mask, "confidence"] = validated["confidence"]
        df.loc[mask, "reviewer"] = validated["reviewer"]
        df.loc[mask, "diagnostic_notes"] = validated["diagnostic_notes"]

        # Atomic write
        temp_file = manifest_p.with_suffix(".tmp.csv")
        df.to_csv(temp_file, index=False)
        temp_file.replace(manifest_p)

    return {
        "status": "success",
        "specimen_id": specimen_id,
        "views_updated": matching_count,
        "applied_decision": validated,
        "backup_path": str(backup_file) if backup_file else None,
        "dry_run": dry_run,
    }


def revert_specimen_review(
    manifest_path: str | Path,
    specimen_id: str,
    backup_path: Optional[str | Path] = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Revert a specimen's decision back to pending_review across all views."""
    return save_specimen_review(
        manifest_path=manifest_path,
        specimen_id=specimen_id,
        candidate_label="pending_review",
        review_status="pending",
        confidence="",
        reviewer="",
        diagnostic_notes="Reset to pending review",
        backup_path=backup_path,
        dry_run=dry_run,
    )
