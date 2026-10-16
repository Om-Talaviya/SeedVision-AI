"""CLI tool to validate and reversibly apply human review decisions to manual_review.csv."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from seedvision.data.roboflow_review import (
    apply_manual_review_decisions,
    revert_manual_review_decisions,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply or revert validated human review decisions on Roboflow crops."
    )
    parser.add_argument("--manifest", required=True, type=Path, help="Path to manual_review.csv")
    parser.add_argument("--decisions", type=Path, help="CSV or JSON file containing human review decisions")
    parser.add_argument("--crop-path", help="Single crop path to review (e.g., crops/ff52bace81fd91d5.png)")
    parser.add_argument("--status", choices=["approved", "rejected", "pending"], help="Review status for single crop")
    parser.add_argument(
        "--label",
        choices=["sound_healthy", "discolored_spotted", "broken_cracked", "shrivelled_damaged", ""],
        default="",
        help="Target classification label (required if approved; must be blank if rejected)",
    )
    parser.add_argument("--reviewer", default="", help="Reviewer identifier/initials (required for approved/rejected)")
    parser.add_argument("--notes", default="", help="Diagnostic rationale or rejection reason (required)")
    parser.add_argument("--revert", action="store_true", help="Revert decisions on specified crop(s) back to pending")
    parser.add_argument("--backup-path", type=Path, help="Custom path for manifest backup before applying changes")
    parser.add_argument("--dry-run", action="store_true", help="Validate decisions without modifying files")
    args = parser.parse_args()

    manifest = args.manifest.expanduser().absolute()
    if not manifest.is_file():
        parser.error(f"Manifest not found: {manifest}")

    decisions: list[dict[str, str]] = []

    if args.decisions:
        dec_file = args.decisions.expanduser().absolute()
        if not dec_file.is_file():
            parser.error(f"Decisions file not found: {dec_file}")
        if dec_file.suffix.lower() == ".json":
            decisions = json.loads(dec_file.read_text(encoding="utf-8"))
        else:
            with dec_file.open(newline="", encoding="utf-8") as f:
                decisions = list(csv.DictReader(f))

    elif args.crop_path:
        if args.revert:
            decisions = [
                {
                    "crop_path": args.crop_path,
                    "review_status": "pending",
                    "reviewed_label": "",
                    "reviewer": "",
                    "review_notes": "",
                }
            ]
        else:
            if not args.status:
                parser.error("--status is required when specifying --crop-path without --revert")
            decisions = [
                {
                    "crop_path": args.crop_path,
                    "review_status": args.status,
                    "reviewed_label": args.label,
                    "reviewer": args.reviewer,
                    "review_notes": args.notes,
                }
            ]
    else:
        parser.error("Either --decisions file or --crop-path must be specified.")

    print(f"=== SeedVision AI Manual Review Applicator ===")
    print(f"Manifest: {manifest}")
    print(f"Decisions to process: {len(decisions)}")
    if args.dry_run:
        print("Dry Run: YES (no files will be modified)")

    try:
        if args.revert and args.decisions:
            crop_paths = [d["crop_path"] for d in decisions]
            result = revert_manual_review_decisions(
                manifest_path=manifest,
                crop_paths=crop_paths,
                backup_path=args.backup_path,
                dry_run=args.dry_run,
            )
        else:
            result = apply_manual_review_decisions(
                manifest_path=manifest,
                decisions=decisions,
                backup_path=args.backup_path,
                dry_run=args.dry_run,
            )

        print("\nResult:")
        print(json.dumps(result, indent=2))
        print(f"\nSuccessfully applied {result['rows_changed']} row change(s).")
        return 0
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"\nVALIDATION ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
