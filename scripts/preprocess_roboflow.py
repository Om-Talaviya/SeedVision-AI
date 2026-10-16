"""Validate a Roboflow YOLO export and optionally stage crops for review."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from seedvision.data.roboflow_preprocessing import stage_review_crops, validate_roboflow_dataset


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report-only YOLO box/polygon validation; optionally stage pending manual-review crops."
    )
    parser.add_argument("--dataset", required=True, type=Path, help="Roboflow YOLO export directory")
    parser.add_argument("--report-json", type=Path, help="Optional path for a validation report JSON")
    parser.add_argument("--stage-crops", action="store_true", help="Write per-annotation crops for manual review")
    parser.add_argument("--output", type=Path, help="Empty output directory, required with --stage-crops")
    args = parser.parse_args()
    if args.stage_crops and args.output is None:
        parser.error("--output is required with --stage-crops")
    if args.output is not None and not args.stage_crops:
        parser.error("--output is only used with --stage-crops")

    try:
        report = validate_roboflow_dataset(args.dataset)
        if args.report_json:
            report_path = args.report_json.expanduser().absolute()
            dataset_path = args.dataset.expanduser().absolute()
            if report_path == dataset_path or dataset_path in report_path.parents:
                raise ValueError("Report JSON must be written outside the Roboflow export")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        if args.stage_crops:
            result = stage_review_crops(args.dataset, args.output)
            print("\nManual-review staging:")
            print(json.dumps(result, indent=2))
        return 0 if report["status"] == "valid" else 1
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
