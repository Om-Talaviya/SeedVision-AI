"""Staging, intake auditing, and validation workflow for genuine high-resolution seed photographs.

This module provides a separate, fully reversible staging workflow for raw photographic
acquisitions prior to approval and ingestion into SeedVision AI's core dataset.

Rules:
- Never automatically maps or approves labels.
- Ambiguous samples or low-confidence decisions remain pending.
- Enforces specimen IDs, provenance, dimensions, reviewer identity, and diagnostic notes.
- Detects exact (MD5) and near (aHash) duplicates.
- Groups multi-view images of the same physical seed to prevent cross-split leakage.
- Preserves all existing data and manifests.
- Enforces strict safeguard: Synthetic test fixtures can NEVER be approved as real samples.
"""

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from seedvision.data.validation import ImageValidator

ALLOWED_TARGET_CLASSES = [
    "sound_healthy",
    "discolored_spotted",
    "broken_cracked",
    "shrivelled_damaged",
]

ALLOWED_REVIEW_STATUSES = [
    "pending",
    "approved",
    "rejected",
]

MIN_RECOMMENDED_DIM = 128
MIN_STRICT_DIM = 64

SYNTHETIC_INDICATORS = {
    "synthetic",
    "simulation",
    "simulated",
    "fixture",
    "test_fixture",
    "mock",
    "placeholder",
    "demo",
    "demonstration",
    "fabricated",
}


def is_synthetic_fixture(meta: Optional[Dict[str, Any]], filename: str) -> bool:
    """Determine if a file or manifest entry represents a synthetic test fixture."""
    if meta:
        # Check explicit boolean / string flag
        syn_flag = str(meta.get("is_synthetic", "")).strip().lower()
        if syn_flag in ("true", "1", "yes", "synthetic"):
            return True

        # Check metadata fields for synthetic markers
        for field in (
            "source_provenance",
            "diagnostic_notes",
            "variety",
            "reviewer",
            "candidate_label",
            "specimen_id",
        ):
            val = str(meta.get(field, "")).strip().lower()
            if any(ind in val for ind in SYNTHETIC_INDICATORS):
                return True

    # Check filename
    fn_lower = filename.lower()
    if any(ind in fn_lower for ind in ("synthetic", "test_fixture", "fixture", "mock")):
        return True

    return False


def compute_average_hash(image: Image.Image, hash_size: int = 8) -> str:
    """Compute perceptual average hash (aHash) for near-duplicate detection."""
    resized = image.convert("L").resize((hash_size, hash_size), Image.Resampling.BILINEAR)
    pixels = np.array(resized, dtype=np.float32).flatten()
    avg = pixels.mean()
    binary_bits = (pixels > avg).astype(int)
    return "".join(
        f"{int(''.join(map(str, byte)), 2):02x}" for byte in np.split(binary_bits, hash_size)
    )


def initialize_staging_scaffold(staging_dir: Path) -> None:
    """Initialize a separate staging directory with incoming folder and manifest template."""
    staging_dir = Path(staging_dir)
    incoming_dir = staging_dir / "incoming"
    incoming_dir.mkdir(parents=True, exist_ok=True)
    (incoming_dir / ".gitkeep").touch()

    # Template manifest CSV with illustrative placeholder values (no fabricated claims)
    template_path = staging_dir / "staging_manifest_template.csv"
    template_df = pd.DataFrame(
        [
            {
                "filename": "sample_seed_001_v1.jpg",
                "specimen_id": "SPECIMEN-2026-001",
                "view_id": "v1",
                "candidate_label": "sound_healthy",
                "review_status": "pending",
                "confidence": "high",
                "source_provenance": "[Example] Research Station / Field Plot 1A",
                "variety": "[Example] Desi Chickpea Cultivar",
                "reviewer": "[Example] lead_agronomist",
                "diagnostic_notes": "[Example] Intact seed coat, normal angular morphology, uniform beige pigmentation, no fissures.",
                "is_synthetic": "False",
            },
            {
                "filename": "sample_seed_002_v1.jpg",
                "specimen_id": "SPECIMEN-2026-002",
                "view_id": "v1",
                "candidate_label": "broken_cracked",
                "review_status": "pending",
                "confidence": "high",
                "source_provenance": "[Example] Research Station / Field Plot 1A",
                "variety": "[Example] Desi Chickpea Cultivar",
                "reviewer": "[Example] lead_agronomist",
                "diagnostic_notes": "[Example] Prominent 2.2mm mechanical fissure across left cotyledon following threshing; testa split.",
                "is_synthetic": "False",
            },
            {
                "filename": "sample_seed_003_v1.jpg",
                "specimen_id": "SPECIMEN-2026-003",
                "view_id": "v1",
                "candidate_label": "discolored_spotted",
                "review_status": "pending",
                "confidence": "high",
                "source_provenance": "[Example] University Agricultural Farm",
                "variety": "[Example] Kabuli Chickpea Cultivar",
                "reviewer": "[Example] pathologist_evaluator",
                "diagnostic_notes": "[Example] Fungal necrotic lesions (>8% testa surface area) with pycnidial centers.",
                "is_synthetic": "False",
            },
            {
                "filename": "sample_seed_004_v1.jpg",
                "specimen_id": "SPECIMEN-2026-004",
                "view_id": "v1",
                "candidate_label": "shrivelled_damaged",
                "review_status": "pending",
                "confidence": "high",
                "source_provenance": "[Example] Research Station / Field Plot 1A",
                "variety": "[Example] Desi Chickpea Cultivar",
                "reviewer": "[Example] lead_agronomist",
                "diagnostic_notes": "[Example] Premature desiccation causing severe wrinkling and hollow cotyledon structure; >30% volume reduction.",
                "is_synthetic": "False",
            },
        ]
    )
    template_df.to_csv(template_path, index=False)

    # Scaffolding documentation
    readme_path = staging_dir / "README.md"
    readme_content = """# SeedVision AI — Real-Image Staging & Intake Scaffolding

This directory serves as the **isolated, reversible staging area** for genuine agricultural seed photographs.
No images placed here are automatically imported into approved training data.

## Photography Submission Requirements
1. **Single-Seed Framing**: Single seed occupying $\\ge 60\\%$ of the visual frame.
2. **Resolution**: Minimum 128x128 pixels (native macro capture recommended $\\ge 512\\times 512$ px).
3. **Lighting**: Diffuse 5000K—6500K LED illumination (ring light or diffuser box) to prevent harsh specular highlights.
4. **Background**: Neutral non-reflective matte gray (18% gray card) or white laboratory tray.
5. **Multi-View Grouping**: When capturing multiple views of the same seed, assign an identical `specimen_id` (e.g. `SPECIMEN-2026-001`) with distinct `view_id` (`v1`, `v2`, `v3`).

## Mandatory Metadata Fields in `staging_manifest.csv`
- `filename`: Local image file name inside `incoming/`
- `specimen_id`: Unique identifier for the individual physical seed
- `view_id`: Camera angle / rotation view index (`v1`, `v2`, `v3`)
- `candidate_label`: Target class (`sound_healthy`, `discolored_spotted`, `broken_cracked`, `shrivelled_damaged`)
- `review_status`: `pending`, `approved`, or `rejected` (must default to `pending`)
- `confidence`: `high`, `medium`, or `low`
- `source_provenance`: Collection institute, field plot, or genebank origin
- `variety`: Botanical cultivar / variety name
- `reviewer`: Human expert / technician identifier
- `diagnostic_notes`: Documented morphological evidence supporting the label
- `is_synthetic`: Boolean flag (default `False`). Synthetic fixtures cannot be approved.

## Validation Command
```powershell
python scripts/stage_real_intake.py --staging-dir data/real_staging --manifest data/real_staging/staging_manifest.csv --dry-run
```
"""
    readme_path.write_text(readme_content, encoding="utf-8")


def validate_staging_dataset(
    staging_dir: Path,
    manifest_path: Optional[Path] = None,
    min_width: int = MIN_STRICT_DIM,
    min_height: int = MIN_STRICT_DIM,
    recommended_min_dim: int = MIN_RECOMMENDED_DIM,
) -> Dict[str, Any]:
    """Validate all images and metadata in staging directory.
    
    Checks:
    1. Manifest presence and schema completeness.
    2. Image file existence, format, and readable pixel integrity.
    3. Minimum dimension requirements and low-resolution flags.
    4. Exact (MD5) and near (aHash) duplicate detection.
    5. Specimen grouping integrity (multi-view linkage).
    6. Candidate label compliance with the 4 allowed target classes.
    7. Review status compliance and automatic pending retention for ambiguous/unreviewed files.
    8. Synthetic fixture isolation: Synthetic fixtures cannot be approved as real samples.
    """
    staging_dir = Path(staging_dir)
    incoming_dir = staging_dir / "incoming" if (staging_dir / "incoming").is_dir() else staging_dir

    if manifest_path is None:
        potential_manifest = staging_dir / "staging_manifest.csv"
        if potential_manifest.is_file():
            manifest_path = potential_manifest

    report: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "staging_dir": str(staging_dir.resolve()),
        "incoming_dir": str(incoming_dir.resolve()),
        "manifest_path": str(manifest_path.resolve()) if manifest_path else None,
        "status": "valid",
        "total_files_found": 0,
        "valid_images": 0,
        "corrupt_files": [],
        "low_resolution_warnings": [],
        "exact_duplicates": [],
        "near_duplicates": [],
        "specimen_groups": {},
        "class_distribution": defaultdict(int),
        "review_status_counts": defaultdict(int),
        "unreviewed_pending_count": 0,
        "ambiguous_pending_count": 0,
        "approved_count": 0,
        "rejected_count": 0,
        "synthetic_fixtures": [],
        "issues": [],
    }

    # Discover images in incoming directory
    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
        image_paths.extend(incoming_dir.glob(ext))
        image_paths.extend(incoming_dir.glob(ext.upper()))
    image_paths = sorted(set(image_paths))
    report["total_files_found"] = len(image_paths)

    # Load manifest if available
    manifest_records: Dict[str, Dict[str, Any]] = {}
    if manifest_path and Path(manifest_path).is_file():
        try:
            df_manifest = pd.read_csv(manifest_path)
        except Exception as e:
            df_manifest = pd.DataFrame()
            report["issues"].append(f"Failed to read manifest CSV: {e}")
            report["status"] = "has_errors"

        if not df_manifest.empty:
            required_cols = {"filename", "specimen_id", "candidate_label"}
            missing_cols = required_cols - set(df_manifest.columns)
            if missing_cols:
                report["issues"].append(f"Manifest missing required columns: {sorted(missing_cols)}")
                report["status"] = "invalid"

            for _, row in df_manifest.iterrows():
                fn = str(row.get("filename", "")).strip()
                if not fn or fn.lower() == "nan":
                    continue
                manifest_records[fn] = {
                    "specimen_id": str(row.get("specimen_id", "")).strip(),
                    "view_id": str(row.get("view_id", "v1")).strip(),
                    "candidate_label": str(row.get("candidate_label", "")).strip().lower(),
                    "review_status": str(row.get("review_status", "pending")).strip().lower(),
                    "confidence": str(row.get("confidence", "high")).strip().lower(),
                    "source_provenance": str(row.get("source_provenance", "")).strip(),
                    "variety": str(row.get("variety", "")).strip(),
                    "reviewer": str(row.get("reviewer", "")).strip(),
                    "diagnostic_notes": str(row.get("diagnostic_notes", "")).strip(),
                    "is_synthetic": str(row.get("is_synthetic", "")).strip(),
                }

    # Verify if manifest references files missing from incoming directory
    if manifest_records:
        incoming_file_set = {p.name for p in image_paths}
        missing_from_incoming = sorted(set(manifest_records.keys()) - incoming_file_set)
        if missing_from_incoming:
            for mf in missing_from_incoming:
                report["issues"].append(
                    f"Manifest references file '{mf}' which is not present in incoming directory."
                )

    validator = ImageValidator(min_width=min_width, min_height=min_height)
    md5_map = defaultdict(list)
    ahash_map = defaultdict(list)
    specimen_to_images = defaultdict(list)

    for img_path in image_paths:
        v_res = validator.validate_file(str(img_path))
        if not v_res.is_valid:
            report["corrupt_files"].append({"file": img_path.name, "reason": v_res.error_message})
            report["issues"].append(f"Corrupt/invalid file: {img_path.name} - {v_res.error_message}")
            continue

        report["valid_images"] += 1

        # Check resolution
        if v_res.width < recommended_min_dim or v_res.height < recommended_min_dim:
            report["low_resolution_warnings"].append(
                {
                    "file": img_path.name,
                    "width": v_res.width,
                    "height": v_res.height,
                    "threshold": recommended_min_dim,
                }
            )

        # Hash calculations
        file_bytes = img_path.read_bytes()
        md5 = hashlib.md5(file_bytes).hexdigest()
        md5_map[md5].append(img_path.name)

        try:
            with Image.open(img_path) as im:
                ahash = compute_average_hash(im)
        except Exception:
            ahash = "unknown"
        if ahash != "unknown":
            ahash_map[ahash].append(img_path.name)

        # Manifest metadata linkage
        meta = manifest_records.get(img_path.name)
        synthetic_flag = is_synthetic_fixture(meta, img_path.name)
        if synthetic_flag:
            report["synthetic_fixtures"].append(img_path.name)

        if meta:
            specimen_id = meta["specimen_id"] or f"UNSPECIFIED_{img_path.stem}"
            cand_label = meta["candidate_label"]
            rev_status = meta["review_status"]
            conf = meta["confidence"]
            reviewer = meta["reviewer"]
            notes = meta["diagnostic_notes"]

            # CRITICAL SAFEGUARD: Synthetic fixtures cannot be approved as real samples
            if synthetic_flag and rev_status == "approved":
                rev_status = "rejected"
                report["issues"].append(
                    f"CRITICAL: Synthetic fixture '{img_path.name}' cannot be approved as a real sample in real-image staging."
                )
                report["status"] = "has_errors"

            # Label validation
            if cand_label not in ALLOWED_TARGET_CLASSES and cand_label != "pending_review":
                report["issues"].append(
                    f"File '{img_path.name}' has unapproved class '{cand_label}'. Allowed: {ALLOWED_TARGET_CLASSES}"
                )

            # Review status validation
            if rev_status not in ALLOWED_REVIEW_STATUSES:
                report["issues"].append(
                    f"File '{img_path.name}' has invalid review_status '{rev_status}'"
                )

            # Ambiguity policy: low confidence or missing reviewer keeps item pending
            if (conf == "low" or not reviewer or not notes) and rev_status != "rejected":
                rev_status = "pending"
                report["ambiguous_pending_count"] += 1

            if rev_status == "approved":
                report["approved_count"] += 1
            elif rev_status == "rejected":
                report["rejected_count"] += 1
            else:
                report["unreviewed_pending_count"] += 1

            report["class_distribution"][cand_label] += 1
            report["review_status_counts"][rev_status] += 1
        else:
            # File exists in incoming but not in manifest -> strictly pending
            specimen_id = f"UNMANIFESTED_{img_path.stem}"
            report["unreviewed_pending_count"] += 1
            report["review_status_counts"]["pending"] += 1
            report["class_distribution"]["unassigned_pending"] += 1

        specimen_to_images[specimen_id].append(img_path.name)

    # Collate duplicate clusters
    for md5, files in md5_map.items():
        if len(files) > 1:
            report["exact_duplicates"].append({"md5": md5, "files": files})

    for ahash, files in ahash_map.items():
        if len(files) > 1 and len(set(files)) > 1:
            report["near_duplicates"].append({"ahash": ahash, "files": files})

    report["specimen_groups"] = {
        specimen: {"views": imgs, "count": len(imgs)}
        for specimen, imgs in specimen_to_images.items()
    }

    # Format defaultdicts to standard dicts
    report["class_distribution"] = dict(report["class_distribution"])
    report["review_status_counts"] = dict(report["review_status_counts"])

    if report["corrupt_files"]:
        report["status"] = "has_errors"

    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage and validate genuine seed photographs before dataset approval."
    )
    parser.add_argument(
        "--staging-dir",
        type=str,
        default="data/real_staging",
        help="Path to real staging intake directory.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to staging manifest CSV.",
    )
    parser.add_argument(
        "--init",
        action="store_true",
        help="Initialize directory scaffold and manifest template.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Audit staging files without moving or modifying dataset files.",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default=None,
        help="Optional path to write JSON audit report.",
    )

    args = parser.parse_args()
    staging_dir = Path(args.staging_dir)

    if args.init:
        initialize_staging_scaffold(staging_dir)
        print(f"[SUCCESS] Staging scaffold initialized at {staging_dir.resolve()}")
        return

    manifest_path = Path(args.manifest) if args.manifest else None
    report = validate_staging_dataset(staging_dir=staging_dir, manifest_path=manifest_path)

    print("\n=== Real-Image Staging Intake Audit ===")
    print(f"Staging directory:      {report['staging_dir']}")
    print(f"Status:                 {report['status']}")
    print(f"Total files found:      {report['total_files_found']}")
    print(f"Valid images:           {report['valid_images']}")
    print(f"Approved for intake:    {report['approved_count']}")
    print(f"Pending review:         {report['unreviewed_pending_count']}")
    print(f"Ambiguous / pending:    {report['ambiguous_pending_count']}")
    print(f"Rejected:               {report['rejected_count']}")
    print(f"Synthetic fixtures:     {len(report.get('synthetic_fixtures', []))}")
    print(f"Exact duplicate sets:   {len(report['exact_duplicates'])}")
    print(f"Near duplicate sets:    {len(report['near_duplicates'])}")
    print(f"Specimen seed groups:   {len(report['specimen_groups'])}")
    print("\nClass distribution in staging:")
    for cls, cnt in report["class_distribution"].items():
        print(f"  - {cls}: {cnt}")

    if report["low_resolution_warnings"]:
        print(f"\n[WARNING] {len(report['low_resolution_warnings'])} images below recommended 128x128px")

    if report["issues"]:
        print("\n[ISSUES DETECTED]:")
        for iss in report["issues"]:
            print(f"  ! {iss}")

    if args.output_report:
        out_p = Path(args.output_report)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n[INFO] Wrote staging validation report to {out_p.resolve()}")


if __name__ == "__main__":
    main()
