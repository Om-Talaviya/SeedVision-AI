"""Bridge from specimen-level staged reviews to leakage-safe ingestion annotations.

Consumes only existing, valid human review decisions recorded in ``staging_manifest.csv``
and maps every view of a physically-identical specimen onto a single stable ``seed_id``.

Design guarantees:
- ``seed_id`` equals the reviewed ``specimen_id`` verbatim, so all views of one specimen
  share one grouping key and cannot straddle train/val/test partitions.
- A specimen is eligible only when *every* view is approved and all views agree on
  specimen-level metadata and label. Otherwise the specimen is excluded (fail closed).
- No label is ever inferred from image appearance or filename. Downstream ingestion
  re-verifies seed identity; see ``--strict-annotations`` in ``scripts/ingest_real_data.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import hashlib
import pandas as pd

INGEST_ANNOTATION_COLUMNS: list[str] = [
    "filename",
    "class_name",
    "seed_id",
    "batch_id",
    "variety",
    "notes",
]

TARGET_LABELS: list[str] = [
    "sound_healthy",
    "discolored_spotted",
    "broken_cracked",
    "shrivelled_damaged",
]


def specimen_seed_id(specimen_id: str | Any) -> str:
    """Return the canonical ``seed_id`` for a specimen.

    The reviewed ``specimen_id`` is used verbatim: it is already unique and stable per
    physical specimen and independent of the assigned quality class. Deriving a seed
    identity from filenames or view indices is explicitly not permitted.
    """
    value = str(specimen_id or "").strip()
    if not value:
        raise ValueError("specimen_id cannot be empty; a stable seed identity is required.")
    return value


def _specimen_exclusion_reason(specimen: Dict[str, Any]) -> Optional[str]:
    """Return the first reason a specimen is ineligible, or ``None`` when eligible."""
    if not str(specimen.get("specimen_id", "")).strip():
        return "missing_specimen_id"

    views: List[Dict[str, Any]] = list(specimen.get("views", []))
    if not views:
        return "no_views"

    statuses = {str(v.get("review_status", "")).strip().lower() for v in views}
    if "pending" in statuses:
        return "has_pending_view"
    if "rejected" in statuses:
        return "has_rejected_view"
    if statuses != {"approved"}:
        return "not_all_approved"

    labels = {str(v.get("candidate_label", "")).strip().lower() for v in views}
    if len(labels) != 1:
        return "inconsistent_labels"
    label = next(iter(labels))
    if label == "pending_review":
        return "unresolved_label"
    if label not in TARGET_LABELS:
        return "unsupported_label"

    varieties = {str(v.get("variety", "")).strip() for v in views}
    if len(varieties) != 1:
        return "inconsistent_variety"
    if not next(iter(varieties)):
        return "missing_variety"

    provenances = {str(v.get("source_provenance", "")).strip() for v in views}
    if len(provenances) != 1:
        return "inconsistent_provenance"
    if not next(iter(provenances)):
        return "missing_provenance"

    if any(not str(v.get("reviewer", "")).strip() for v in views):
        return "missing_reviewer"
    if any(not str(v.get("diagnostic_notes", "")).strip() for v in views):
        return "missing_diagnostic_notes"

    if any(not v.get("image_exists") for v in views):
        return "missing_image"

    filenames = [str(v.get("filename", "")).strip() for v in views]
    if any(not name for name in filenames):
        return "missing_filename"
    if len(set(filenames)) != len(filenames):
        return "duplicate_view_filenames"

    return None


def select_approved_specimens(staging: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Partition staged specimens into eligible (fully approved) and excluded ones.

    Args:
        staging: The dictionary returned by
            :func:`seedvision.data.real_review.load_staging_manifest`.

    Returns:
        ``{"eligible": [...specimens...], "excluded": [{"specimen_id", "reason"}, ...]}``
        Deterministically ordered by ``specimen_id``.
    """
    specimens: Dict[str, Dict[str, Any]] = staging.get("specimens", {}) or {}

    eligible: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []

    for specimen_id in sorted(specimens):
        specimen = specimens[specimen_id]
        reason = _specimen_exclusion_reason(specimen)
        if reason is None:
            eligible.append(specimen)
        else:
            excluded.append({"specimen_id": str(specimen.get("specimen_id", specimen_id)).strip(), "reason": reason})

    return {"eligible": eligible, "excluded": excluded}


def build_ingest_annotations(eligible_specimens: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Build ingestion annotation rows from eligible specimens.

    Every view of a specimen receives the identical ``seed_id`` (the reviewed
    ``specimen_id``), guaranteeing that no views of one physical specimen can be
    separated across splits.

    Raises:
        ValueError: if the same source filename is claimed by more than one row.
    """
    rows: List[Dict[str, str]] = []
    for specimen in sorted(eligible_specimens, key=lambda s: str(s.get("specimen_id", ""))):
        specimen_id = str(specimen["specimen_id"]).strip()
        seed_id = specimen_seed_id(specimen_id)
        views = sorted(specimen.get("views", []), key=lambda v: str(v.get("filename", "")))
        for view in views:
            rows.append(
                {
                    "filename": str(view.get("filename", "")).strip(),
                    "class_name": str(view.get("candidate_label", "")).strip().lower(),
                    "seed_id": seed_id,
                    "batch_id": "",
                    "variety": str(view.get("variety", "")).strip(),
                    "notes": str(view.get("diagnostic_notes", "")).strip(),
                }
            )

    validate_no_duplicate_filenames(rows)
    return rows


def validate_no_duplicate_filenames(rows: List[Dict[str, str]]) -> None:
    """Fail closed if any source filename appears in more than one annotation row."""
    seen: Dict[str, int] = {}
    for row in rows:
        name = row.get("filename", "")
        seen[name] = seen.get(name, 0) + 1

    duplicates = sorted(name for name, count in seen.items() if count > 1)
    if duplicates:
        raise ValueError(
            f"Duplicate source filenames would produce conflicting annotations: {duplicates}"
        )


def build_annotation_dataframe(rows: List[Dict[str, str]]) -> pd.DataFrame:
    """Return a DataFrame with the canonical ingestion annotation schema."""
    return pd.DataFrame(rows, columns=INGEST_ANNOTATION_COLUMNS)


def write_annotations_atomic(
    df: pd.DataFrame,
    out_path: str | Path,
    overwrite: bool = False,
) -> Path:
    """Atomically write the annotation manifest, refusing silent overwrite by default."""
    out = Path(out_path)
    if out.exists() and not overwrite:
        raise FileExistsError(
            f"Annotation manifest already exists: {out}. Pass overwrite=True to replace it explicitly."
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    temp_file = out.with_suffix(out.suffix + ".tmp")
    df.to_csv(temp_file, index=False)
    temp_file.replace(out)
    return out


def plan_image_staging(
    rows: List[Dict[str, str]],
    incoming_dir: str | Path,
    images_dir: str | Path,
) -> List[Dict[str, str]]:
    """Plan copying each approved view into its class folder without writing anything.

    Each plan entry reports a ``status`` of ``copy`` (destination absent), ``identical``
    (destination already holds the same bytes), or ``conflict`` (destination exists with
    different bytes). Conflicts must be resolved before any write occurs.
    """
    incoming = Path(incoming_dir)
    images = Path(images_dir)

    plan: List[Dict[str, str]] = []
    for row in sorted(rows, key=lambda r: (r["class_name"], r["filename"])):
        src = incoming / row["filename"]
        dest = images / row["class_name"] / row["filename"]

        if not src.is_file():
            status = "missing_source"
        elif not dest.exists():
            status = "copy"
        elif hashlib.md5(dest.read_bytes()).hexdigest() == hashlib.md5(src.read_bytes()).hexdigest():
            status = "identical"
        else:
            status = "conflict"

        plan.append(
            {
                "filename": row["filename"],
                "class_name": row["class_name"],
                "source": str(src),
                "destination": str(dest),
                "status": status,
            }
        )

    return plan


def stage_images(plan: List[Dict[str, str]]) -> Dict[str, Any]:
    """Execute an image staging plan after verifying no destination conflicts exist.

    Raises:
        FileExistsError: when any planned destination exists with differing content.
        FileNotFoundError: when a planned source image is missing.
    """
    conflicts = [entry for entry in plan if entry["status"] == "conflict"]
    if conflicts:
        raise FileExistsError(
            "Refusing to overwrite existing images with differing content: "
            + ", ".join(sorted(e["destination"] for e in conflicts))
        )

    missing = [entry for entry in plan if entry["status"] == "missing_source"]
    if missing:
        raise FileNotFoundError(
            "Missing source images: " + ", ".join(sorted(e["source"] for e in missing))
        )

    copied = 0
    skipped = 0
    for entry in plan:
        if entry["status"] == "copy":
            dest = Path(entry["destination"])
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(Path(entry["source"]).read_bytes())
            copied += 1
        elif entry["status"] == "identical":
            skipped += 1

    return {
        "copied": copied,
        "skipped_identical": skipped,
        "total_planned": len(plan),
        "conflicts": len(conflicts),
    }


def build_bridge_report(
    staging: Dict[str, Any],
    selection: Dict[str, List[Dict[str, Any]]],
    rows: List[Dict[str, str]],
    plan: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Assemble a JSON-serializable report describing the bridge decision."""
    eligible = selection.get("eligible", [])
    excluded = selection.get("excluded", [])

    report: Dict[str, Any] = {
        "total_specimens": staging.get("total_specimens", 0),
        "total_views": staging.get("total_views", 0),
        "eligible_specimens": [str(s["specimen_id"]) for s in eligible],
        "excluded_specimens": excluded,
        "annotation_rows": len(rows),
        "seed_ids": sorted({row["seed_id"] for row in rows}),
        "class_distribution": {},
    }

    for row in rows:
        report["class_distribution"][row["class_name"]] = (
            report["class_distribution"].get(row["class_name"], 0) + 1
        )

    if plan is not None:
        statuses: Dict[str, int] = {}
        for entry in plan:
            statuses[entry["status"]] = statuses.get(entry["status"], 0) + 1
        report["image_plan_counts"] = statuses

    return report
