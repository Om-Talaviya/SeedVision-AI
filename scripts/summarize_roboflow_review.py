"""Summarize the current state of a Roboflow manual-review manifest, enriched with resolution & triage stats."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Write an enriched review-progress report without changing manifest rows.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata", type=Path, help="Optional crop_metadata.csv or crop_metadata.json path")
    parser.add_argument("--rejected-example-limit", type=int, default=10)
    args = parser.parse_args()
    manifest = args.manifest.expanduser().absolute()
    output = args.output.expanduser().absolute()
    if not manifest.is_file():
        parser.error(f"Manifest does not exist: {manifest}")
    if output == manifest:
        parser.error("Output must not overwrite the review manifest")

    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"crop_path", "source_class_name", "source_group_id", "review_status", "reviewed_label"}
    if not rows or not required.issubset(rows[0]):
        parser.error("Manifest does not have the required manual-review columns")

    groups = {}
    for row in rows:
        groups.setdefault(row["source_group_id"], set()).add(row["source_split"])
    rejected = [row for row in rows if row["review_status"] == "rejected"]

    # Resolution & metadata enrichment if available
    metadata_file = args.metadata
    if metadata_file is None:
        candidate = manifest.parent / "crop_metadata.csv"
        if candidate.is_file():
            metadata_file = candidate

    resolution_stats = None
    if metadata_file and metadata_file.is_file():
        try:
            if metadata_file.suffix.lower() == ".json":
                meta_rows = json.loads(metadata_file.read_text(encoding="utf-8"))
            else:
                with metadata_file.open(newline="", encoding="utf-8") as m_f:
                    meta_rows = list(csv.DictReader(m_f))

            widths = [int(m["width"]) for m in meta_rows if m.get("width") and int(m["width"]) > 0]
            heights = [int(m["height"]) for m in meta_rows if m.get("height") and int(m["height"]) > 0]
            aspect_ratios = [float(m["aspect_ratio"]) for m in meta_rows if m.get("aspect_ratio")]
            low_res_rows = [m for m in meta_rows if str(m.get("is_low_resolution", "")).lower() in ("true", "1")]

            if widths and heights:
                resolution_stats = {
                    "crop_metadata_file": str(metadata_file),
                    "min_width": min(widths),
                    "max_width": max(widths),
                    "median_width": int(statistics.median(widths)),
                    "min_height": min(heights),
                    "max_height": max(heights),
                    "median_height": int(statistics.median(heights)),
                    "min_aspect_ratio": min(aspect_ratios) if aspect_ratios else 0.0,
                    "max_aspect_ratio": max(aspect_ratios) if aspect_ratios else 0.0,
                    "median_aspect_ratio": round(statistics.median(aspect_ratios), 3) if aspect_ratios else 0.0,
                    "low_resolution_threshold_px": 28,
                    "low_resolution_crop_count": len(low_res_rows),
                    "low_resolution_by_source_class": dict(
                        sorted(Counter(m.get("source_class_name", "") for m in low_res_rows).items())
                    ),
                    "policy_note": "Crops <28px are flagged for review and never auto-rejected.",
                }
        except Exception as exc:
            resolution_stats = {"error": f"Failed reading metadata: {exc}"}

    foreign_material_rejected = sum(
        1 for r in rejected
        if r.get("source_class_name") in ("Material extrano inorganico", "Material extrano organico")
        and r.get("review_notes") == "foreign_material"
    )

    report = {
        "manifest": str(manifest),
        "crop_count": len(rows),
        "crops_by_source_class": dict(sorted(Counter(row["source_class_name"] for row in rows).items())),
        "review_status_counts": dict(sorted(Counter(row["review_status"] for row in rows).items())),
        "reviewed_label_counts": dict(sorted(Counter(row["reviewed_label"] or "(blank)" for row in rows).items())),
        "source_group_count": len(groups),
        "source_groups_crossing_export_splits": sorted(group for group, splits in groups.items() if len(splits) > 1),
        "rejected_count": len(rejected),
        "rejected_examples": rejected[: args.rejected_example_limit],
        "future_split_requirement": "Keep every source_group_id within one partition after human review.",
    }

    if resolution_stats:
        report["resolution_statistics"] = resolution_stats

    report["triage_summary"] = {
        "foreign_material_rejected": foreign_material_rejected,
        "pending_count": sum(1 for r in rows if r["review_status"] == "pending"),
        "rejected_count": len(rejected),
        "target_label_mappings": 0,
        "policy_note": "Zero Roboflow classes auto-mapped to SeedVision target labels.",
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
