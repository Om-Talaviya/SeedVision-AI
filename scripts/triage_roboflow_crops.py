"""CLI tool for reversible, auditable triage and metadata enrichment of Roboflow crops."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from seedvision.data.roboflow_review import (
    generate_crop_metadata,
    triage_foreign_material,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reversibly triage confirmed foreign material and generate crop metadata."
    )
    parser.add_argument("--manifest", required=True, type=Path, help="Path to manual_review.csv")
    parser.add_argument(
        "--action",
        choices=["triage", "revert", "status"],
        default="triage",
        help="Action to perform (triage, revert, or status)",
    )
    parser.add_argument("--backup-path", type=Path, help="Optional custom path for manifest backup")
    parser.add_argument("--generate-metadata", action="store_true", help="Generate crop_metadata.csv")
    parser.add_argument("--metadata-output", type=Path, help="Output path for crop metadata (CSV or JSON)")
    parser.add_argument("--dry-run", action="store_true", help="Report planned changes without modifying files")
    parser.add_argument("--min-resolution", type=int, default=28, help="Low-resolution threshold in pixels (default: 28)")
    args = parser.parse_args()

    manifest = args.manifest.expanduser().absolute()
    if not manifest.is_file():
        parser.error(f"Manifest not found: {manifest}")

    print(f"=== Roboflow Crop Review Triage & Metadata CLI ===")
    print(f"Manifest: {manifest}")
    print(f"Action: {args.action}")
    if args.dry_run:
        print("Dry Run: YES (no files will be written)")

    # Metadata generation if requested
    if args.generate_metadata or args.metadata_output is not None:
        metadata_dest = (
            args.metadata_output.expanduser().absolute()
            if args.metadata_output is not None
            else manifest.parent / "crop_metadata.csv"
        )
        print(f"\nGenerating crop metadata -> {metadata_dest} ...")
        metadata = generate_crop_metadata(
            manifest_path=manifest,
            output_path=metadata_dest,
            min_resolution=args.min_resolution,
        )
        low_res = sum(1 for m in metadata if m.get("is_low_resolution"))
        print(f"Metadata generated: {len(metadata)} crops.")
        print(f"Resolution triage flags (<{args.min_resolution}px): {low_res} crops flagged (not auto-rejected).")

    if args.action == "status":
        import csv
        with manifest.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        from collections import Counter
        status_counts = Counter(r["review_status"] for r in rows)
        foreign = sum(1 for r in rows if r["source_class_name"] in ("Material extrano inorganico", "Material extrano organico"))
        foreign_rejected = sum(1 for r in rows if r["source_class_name"] in ("Material extrano inorganico", "Material extrano organico") and r["review_status"] == "rejected")
        print("\nCurrent Status:")
        print(f"Total rows: {len(rows)}")
        print(f"Foreign material rows: {foreign} (rejected: {foreign_rejected}, pending: {foreign - foreign_rejected})")
        print(f"Status distribution: {dict(status_counts)}")
        return 0

    revert = args.action == "revert"
    result = triage_foreign_material(
        manifest_path=manifest,
        backup_path=args.backup_path,
        revert=revert,
        dry_run=args.dry_run,
    )

    print("\nTriage Operation Results:")
    print(json.dumps(result, indent=2))
    print(f"\nSummary:")
    print(f"Rows changed: {result['rows_changed']}")
    print(f"Rows rejected: {result['rejected_count']}")
    print(f"Rows pending: {result['pending_count']}")
    if result.get("backup_path"):
        print(f"Audit backup saved at: {result['backup_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
