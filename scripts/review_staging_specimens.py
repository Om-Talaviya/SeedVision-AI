"""CLI tool for reviewing and validating staged real-seed photographs by specimen.

Allows human annotators to inspect multi-view specimens, assign consistent specimen-level
labels, keep uncertain specimens pending, and generate reversible backups.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from seedvision.data.real_review import (
    ALLOWED_CONFIDENCES,
    ALLOWED_REVIEW_STATUSES,
    ALLOWED_TARGET_LABELS,
    load_staging_manifest,
    restore_manifest_backup,
    revert_specimen_review,
    save_specimen_review,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect and review staged real seed photographs grouped by physical specimen."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/real_staging/staging_manifest.csv"),
        help="Path to staging manifest CSV.",
    )
    parser.add_argument(
        "--incoming-dir",
        type=Path,
        default=Path("data/real_staging/incoming"),
        help="Directory containing incoming photographic views.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all physical specimens, view counts, and their current review status.",
    )
    parser.add_argument(
        "--specimen-id",
        type=str,
        help="Target physical specimen ID to review (e.g. REAL_CHICKPEA_001).",
    )
    parser.add_argument(
        "--label",
        type=str,
        choices=ALLOWED_TARGET_LABELS,
        help="Target label to assign across all views of this physical specimen.",
    )
    parser.add_argument(
        "--status",
        type=str,
        choices=ALLOWED_REVIEW_STATUSES,
        default="approved",
        help="Review status to assign (default: approved).",
    )
    parser.add_argument(
        "--confidence",
        type=str,
        choices=ALLOWED_CONFIDENCES,
        default="high",
        help="Reviewer confidence (high, medium, low). Low confidence forces review status to pending.",
    )
    parser.add_argument(
        "--reviewer",
        type=str,
        default="",
        help="Reviewer identity / username.",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Diagnostic morphological notes supporting the classification decision.",
    )
    parser.add_argument(
        "--revert",
        action="store_true",
        help="Revert the specified specimen back to pending_review.",
    )
    parser.add_argument(
        "--backup-path",
        type=Path,
        default=None,
        help="Custom backup path for the manifest before changes are written.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform validation without saving modifications to the CSV.",
    )

    parser.add_argument(
        "--restore-backup",
        action="store_true",
        help="Restore the staging manifest from its .backup.csv file, undoing all manual review changes.",
    )

    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    incoming_dir = args.incoming_dir.resolve()

    if args.restore_backup:
        try:
            restored = restore_manifest_backup(manifest_path, args.backup_path)
            print(f"\n[SUCCESS] Manifest successfully restored from backup: {restored}")
            return 0
        except Exception as exc:
            print(f"\n[ERROR] Failed to restore backup: {exc}", file=sys.stderr)
            return 1

    if not manifest_path.is_file():
        print(f"Error: Manifest file not found: {manifest_path}", file=sys.stderr)
        return 1

    try:
        data = load_staging_manifest(manifest_path, incoming_dir)
    except Exception as exc:
        print(f"Error loading manifest: {exc}", file=sys.stderr)
        return 1

    if args.list:
        print("\n=== Staged Specimens Summary ===")
        print(f"Manifest:        {manifest_path}")
        print(f"Total Specimens: {data['total_specimens']}")
        print(f"Total Views:     {data['total_views']}")
        print(f"Status Counts:   {data['status_counts']}")
        print(f"Class Counts:    {data['class_counts']}")
        if data["missing_images"]:
            print(f"[WARNING] Missing image files on disk: {data['missing_images']}")

        print("\nSpecimens:")
        for spec_id, info in data["specimens"].items():
            print(f"  - [{info['review_status'].upper()}] {spec_id}:")
            print(f"      Label:      {info['candidate_label']}")
            print(f"      Provenance: {info['source_provenance']}")
            print(f"      Variety:    {info['variety']}")
            print(f"      Views:      {len(info['views'])} view(s)")
            for v in info["views"]:
                dim_str = f"{v['width']}x{v['height']} px" if v["image_exists"] else "MISSING"
                print(f"        * {v['view_id']} -> {v['filename']} ({dim_str})")
        return 0

    if not args.specimen_id:
        parser.print_help()
        print("\nTip: Run with --list to view all available specimens.", file=sys.stderr)
        return 1

    specimen_id = args.specimen_id.strip()
    if specimen_id not in data["specimens"]:
        print(f"Error: Specimen '{specimen_id}' not found in manifest.", file=sys.stderr)
        return 1

    if args.revert:
        try:
            res = revert_specimen_review(
                manifest_path=manifest_path,
                specimen_id=specimen_id,
                backup_path=args.backup_path,
                dry_run=args.dry_run,
            )
            print(f"\n[SUCCESS] Reverted specimen '{specimen_id}' to pending_review.")
            print(f"Views updated: {res['views_updated']}")
            if res["backup_path"]:
                print(f"Backup saved to: {res['backup_path']}")
            return 0
        except Exception as exc:
            print(f"\n[ERROR] Failed to revert specimen: {exc}", file=sys.stderr)
            return 1

    if not args.label:
        print("Error: --label is required when reviewing a specimen.", file=sys.stderr)
        return 1

    try:
        res = save_specimen_review(
            manifest_path=manifest_path,
            specimen_id=specimen_id,
            candidate_label=args.label,
            review_status=args.status,
            confidence=args.confidence,
            reviewer=args.reviewer,
            diagnostic_notes=args.notes,
            backup_path=args.backup_path,
            dry_run=args.dry_run,
        )
        print(f"\n[SUCCESS] Specimen '{specimen_id}' updated successfully across all {res['views_updated']} view(s)!")
        print("Applied Decision:")
        print(json.dumps(res["applied_decision"], indent=2))
        if res["backup_path"]:
            print(f"Pre-modification backup created at: {res['backup_path']}")
        if args.dry_run:
            print("[INFO] DRY RUN: No files were modified.")
        return 0
    except Exception as exc:
        print(f"\n[VALIDATION ERROR] Could not save review decision: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
