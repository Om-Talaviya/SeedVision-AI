"""Generate leakage-safe ingestion annotations from fully approved staged specimens.

Reads ``staging_manifest.csv``, selects specimens whose every view has been approved with
consistent specimen-level metadata, and emits an ``annotations.csv`` in the schema consumed
by ``scripts/ingest_real_data.py``. Every view of one physical specimen receives the same
stable ``seed_id`` (the reviewed ``specimen_id``), preventing cross-split leakage.

Safety: dry-run by default (writes nothing). Image staging is opt-in and refuses to
overwrite existing images with differing content. Source photographs and staging records
are never modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from seedvision.data.real_review import load_staging_manifest
from seedvision.data.review_bridge import (
    build_annotation_dataframe,
    build_bridge_report,
    build_ingest_annotations,
    plan_image_staging,
    select_approved_specimens,
    stage_images,
    write_annotations_atomic,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build ingestable annotations.csv from approved staged specimens (specimen-safe)."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/real_staging/staging_manifest.csv"),
        help="Path to staging manifest CSV.",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=Path("data/real_staging/incoming"),
        help="Directory containing staged photographic views.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/real_raw/annotations.csv"),
        help="Destination annotation manifest to generate.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("data/real_raw"),
        help="Destination root for class folders when --stage-images is used.",
    )
    parser.add_argument(
        "--stage-images",
        action="store_true",
        help="Copy approved views into <images-dir>/<class_name>/ (opt-in; refuses conflicting overwrites).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually write outputs. Without this flag the tool runs in dry-run mode and writes nothing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing annotation manifest.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional path to write the JSON report (only written with --apply).",
    )

    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    staging_dir = args.staging_dir.resolve()
    out_path = args.out.resolve()
    images_dir = args.images_dir.resolve()

    if not manifest_path.is_file():
        print(f"Error: Manifest file not found: {manifest_path}", file=sys.stderr)
        return 1

    try:
        staging = load_staging_manifest(manifest_path, staging_dir)
    except Exception as exc:
        print(f"Error loading manifest: {exc}", file=sys.stderr)
        return 1

    selection = select_approved_specimens(staging)
    rows = build_ingest_annotations(selection["eligible"])
    plan = plan_image_staging(rows, staging_dir, images_dir) if args.stage_images else None
    report = build_bridge_report(staging, selection, rows, plan)

    print(json.dumps(report, indent=2))

    if not rows:
        print(
            "\n[ACTION REQUIRED] No fully approved specimens are eligible. "
            "Approve every view of a specimen before bridging (see review_staging_specimens.py).",
            file=sys.stderr,
        )
        return 1

    conflicts = [entry for entry in (plan or []) if entry["status"] == "conflict"]
    if conflicts:
        print(
            "\n[ERROR] Destination images exist with differing content; resolve before applying:",
            file=sys.stderr,
        )
        for entry in conflicts:
            print(f"  - {entry['destination']}", file=sys.stderr)
        return 1

    if not args.apply:
        print("\n[DRY-RUN] No files were written. Re-run with --apply to commit changes.")
        print(f"         Would write annotation manifest: {out_path}")
        if plan is not None:
            print(f"         Would stage {report.get('image_plan_counts', {}).get('copy', 0)} image(s) into {images_dir}")
        return 0

    try:
        if plan is not None:
            stage_result = stage_images(plan)
            print(f"\n[INFO] Image staging: {json.dumps(stage_result)}")

        df = build_annotation_dataframe(rows)
        written = write_annotations_atomic(df, out_path, overwrite=args.overwrite)
        print(f"\n[SUCCESS] Wrote {len(df)} annotation row(s) to {written}")
    except FileExistsError as exc:
        print(f"\n[ERROR] Refusing to overwrite: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"\n[ERROR] Bridge failed: {exc}", file=sys.stderr)
        return 1

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"[INFO] Report written to {args.report.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
