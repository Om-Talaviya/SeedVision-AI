"""Ingestion, duplicate auditing, and leakage-resistant splitting pipeline for genuine seed photographs."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from seedvision.config import load_config
from seedvision.data.splitting import create_leakage_resistant_splits, save_splits
from seedvision.data.validation import ImageValidator

EXPECTED_CLASSES = [
    "sound_healthy",
    "discolored_spotted",
    "broken_cracked",
    "shrivelled_damaged",
]


def compute_average_hash(image: Image.Image, hash_size: int = 8) -> str:
    """Compute perceptual average hash (aHash) for near-duplicate detection."""
    resized = image.convert("L").resize((hash_size, hash_size), Image.Resampling.BILINEAR)
    pixels = np.array(resized, dtype=np.float32).flatten()
    avg = pixels.mean()
    binary_bits = (pixels > avg).astype(int)
    hex_str = "".join([f"{int(''.join(map(str, byte)), 2):02x}" for byte in np.split(binary_bits, hash_size)])
    return hex_str


def initialize_real_data_scaffold(target_dir: Path) -> None:
    """Scaffold the real dataset directory with class folders, annotation CSV template, and guide."""
    target_dir.mkdir(parents=True, exist_ok=True)
    for cls in EXPECTED_CLASSES:
        cls_dir = target_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        (cls_dir / ".gitkeep").touch()

    # Annotation template CSV
    template_path = target_dir / "annotations_template.csv"
    if not template_path.exists():
        template_df = pd.DataFrame(
            [
                {
                    "filename": "sample_healthy_001.jpg",
                    "class_name": "sound_healthy",
                    "seed_id": "seed_001",
                    "batch_id": "batch_2026_fieldA",
                    "variety": "Desi Chickpea (JG-11)",
                    "notes": "Intact testa, uniform beige pigmentation",
                },
                {
                    "filename": "sample_cracked_001.jpg",
                    "class_name": "broken_cracked",
                    "seed_id": "seed_002",
                    "batch_id": "batch_2026_fieldA",
                    "variety": "Desi Chickpea (JG-11)",
                    "notes": "Mechanical thresher fissure across cotyledon",
                },
            ]
        )
        template_df.to_csv(template_path, index=False)

    # Ingestion guide README
    readme_path = target_dir / "README.md"
    readme_content = """# Genuine Agricultural Seed Photography Ingestion

This directory is designated exclusively for **genuine, real-world photographs of chickpea seeds** (*Cicer arietinum L.*).

## Classification Schema
Place single-seed macro photographs into the corresponding quality class directory:
1. `sound_healthy/`: Intact testa, normal angular/spherical morphology, uniform color, zero cracks or necrotic spots.
2. `discolored_spotted/`: Fungal lesions, necrotic stains, moisture damage, or dark weathered patches.
3. `broken_cracked/`: Mechanical cracks, cotyledon split, or chipped edge caused by threshing or handling stress.
4. `shrivelled_damaged/`: Immature, wrinkled, hollowed, or insect-bored seeds (e.g. pulse beetle *Callosobruchus*).

## Recommended Photography Setup
- **Lighting**: Diffuse 5000K–6500K LED ring light to eliminate harsh shadows and specular hotspots.
- **Background**: Neutral matte 18% gray card or non-reflective white laboratory tray.
- **Resolution**: Single-seed macro capture with seed occupying at least 60% of frame (minimum 128x128 px).
- **Naming Convention**: `{class_name}_{batch_id}_{seed_id}_{view_idx}.jpg` (e.g., `broken_cracked_B1_S042_v1.jpg`).

## Ingestion Command
To validate, audit duplicates, and generate leakage-free splits:
```bash
python scripts/ingest_real_data.py --source-dir data/real_raw --target-dir data/real_raw --provenance-source "University Trial Farm"
```
"""
    readme_path.write_text(readme_content, encoding="utf-8")
    print(f"[SUCCESS] Initialized real-data scaffold in {target_dir.resolve()}")


def audit_and_ingest_real_dataset(
    source_dir: Path,
    target_dir: Path,
    splits_dir: Path,
    classes: List[str] = EXPECTED_CLASSES,
    min_size: int = 64,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    provenance_source: str = "Unknown/User Supplied",
    license_name: str = "Unspecified / Research Use Only",
    copy_files: bool = True,
    seed: int = 42,
    provenance_path: Optional[Path] = None,
) -> Dict:
    """Audit images in source_dir, detect duplicates, copy to target_dir, and partition splits."""
    validator = ImageValidator(min_width=min_size, min_height=min_size)

    md5_to_paths = defaultdict(list)
    ahash_to_paths = defaultdict(list)
    valid_records = []
    invalid_records = []

    print(f"\n=== Auditing Real Seed Dataset in: {source_dir.resolve()} ===")
    print(f"Target classes: {classes}")

    if not source_dir.exists():
        print(f"[NOTE] Source directory does not exist: {source_dir}")
        return {"status": "empty", "valid_count": 0, "classes_represented": {}}

    # Detect unrecognized directories in source_dir
    unrecognized_dirs = []
    for d in sorted(source_dir.iterdir()):
        if d.is_dir() and d.name not in classes and not d.name.startswith("."):
            unrecognized_dirs.append(d.name)

    if unrecognized_dirs:
        print(f"\n[WARNING] Unrecognized subdirectories found in {source_dir.name}: {unrecognized_dirs}")
        print(f"Only images inside the 4 target quality classes will be processed: {classes}")

    # Discover images across class subfolders
    for cls in classes:
        cls_folder = source_dir / cls
        if not cls_folder.is_dir():
            print(f"[WARN] Class folder not found: {cls_folder}")
            continue

        for img_path in sorted(cls_folder.glob("*.*")):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
                continue

            # Validate file integrity
            v_res = validator.validate_file(str(img_path))
            if not v_res.is_valid:
                invalid_records.append({"path": str(img_path), "reason": v_res.error_message})
                continue

            # Compute MD5
            file_bytes = img_path.read_bytes()
            md5_hash = hashlib.md5(file_bytes).hexdigest()
            md5_to_paths[md5_hash].append(img_path)

            # Compute aHash
            try:
                with Image.open(img_path) as im:
                    ahash = compute_average_hash(im)
            except Exception:
                ahash = "unknown"
            ahash_to_paths[ahash].append(img_path)

            # Extract physical seed ID from stem (strip class prefix if present)
            stem = img_path.stem
            if stem.startswith(cls + "_"):
                remainder = stem[len(cls) + 1 :]
                # If multi-view, e.g. "seed01_v1", isolate seed01
                seed_id = f"{cls}_" + (remainder.split("_v")[0] if "_v" in remainder else remainder)
            else:
                seed_id = f"{cls}_{stem}"

            valid_records.append(
                {
                    "source_path": str(img_path.resolve()),
                    "filename": img_path.name,
                    "class_name": cls,
                    "seed_id": seed_id,
                    "md5": md5_hash,
                    "ahash": ahash,
                    "width": v_res.width,
                    "height": v_res.height,
                    "file_size_bytes": v_res.file_size_bytes,
                }
            )

    total_scanned = len(valid_records) + len(invalid_records)
    exact_duplicate_clusters = {k: v for k, v in md5_to_paths.items() if len(v) > 1}
    near_duplicate_clusters = {k: v for k, v in ahash_to_paths.items() if len(v) > 1 and k != "unknown"}

    print(f"\n--- Real-Data Audit Summary ---")
    print(f"Total scanned files:          {total_scanned}")
    print(f"Valid images:                 {len(valid_records)}")
    print(f"Corrupt / invalid images:     {len(invalid_records)}")
    print(f"Exact duplicate clusters:     {len(exact_duplicate_clusters)}")
    print(f"Suspected near-duplicates:    {len(near_duplicate_clusters)}")

    class_counts = defaultdict(int)
    for r in valid_records:
        class_counts[r["class_name"]] += 1
    for cls in classes:
        print(f"  - Class '{cls}': {class_counts[cls]} samples")

    if len(valid_records) == 0:
        print("\n[NOTE] No valid images were found in source directory.")
        return {
            "status": "empty",
            "valid_count": 0,
            "classes_represented": dict(class_counts),
        }

    # Dedup: Keep only the first occurrence for exact duplicates
    seen_md5 = set()
    deduped_records = []
    duplicates_dropped = 0
    for r in valid_records:
        if r["md5"] in seen_md5:
            duplicates_dropped += 1
            continue
        seen_md5.add(r["md5"])
        deduped_records.append(r)

    print(f"\nRetained {len(deduped_records)} unique records after dropping {duplicates_dropped} exact duplicates.")

    # Copy files to target directory if requested
    target_dir.mkdir(parents=True, exist_ok=True)
    for r in deduped_records:
        src = Path(r["source_path"])
        dest_folder = target_dir / r["class_name"]
        dest_folder.mkdir(parents=True, exist_ok=True)
        dest = dest_folder / r["filename"]
        if copy_files and src.resolve() != dest.resolve():
            shutil.copy2(src, dest)
        r["filepath"] = str(dest.resolve())

    # Create DataFrame manifest
    df_manifest = pd.DataFrame(deduped_records)

    # Check minimum class representation
    classes_with_data = [cls for cls, cnt in class_counts.items() if cnt > 0]
    unsupported_classes = [cls for cls in classes if class_counts[cls] == 0]

    if unsupported_classes:
        print(f"\n[WARNING] The following classes have ZERO genuine samples: {unsupported_classes}")
        print("Model training requires at least 2 samples per class to construct stratified splits.")

    if len(classes_with_data) < 2 or any(class_counts[cls] < 3 for cls in classes_with_data):
        print("\n[ERROR] Sample count too small to create reliable train/val/test splits.")
        return {
            "status": "insufficient_samples",
            "valid_count": len(deduped_records),
            "classes_represented": dict(class_counts),
            "unsupported_classes": unsupported_classes,
        }

    # Partition splits
    train_df, val_df, test_df = create_leakage_resistant_splits(
        manifest_df=df_manifest,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        group_col="seed_id",
        class_col="class_name",
        random_state=seed,
    )

    splits_dir.mkdir(parents=True, exist_ok=True)
    save_splits(train_df, val_df, test_df, splits_dir)

    # Provenance JSON
    provenance_record = {
        "dataset_name": "SeedVision-Real-Agricultural-Pulses",
        "provenance_status": "genuine_photograph",
        "is_real_world_photograph": True,
        "source_organization": provenance_source,
        "license": license_name,
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_images_scanned": total_scanned,
        "valid_images_retained": len(deduped_records),
        "exact_duplicates_removed": duplicates_dropped,
        "class_distribution": dict(class_counts),
        "unsupported_classes": unsupported_classes,
        "splits": {
            "train_count": len(train_df),
            "val_count": len(val_df),
            "test_count": len(test_df),
            "splitting_method": "Class-stratified group-aware split (seed_id isolated)",
        },
        "biological_boundary": (
            "Visual RGB inspection evaluates external seed coat morphology only. "
            "It does NOT evaluate physiological germination rate, internal embryo viability (Tetrazolium), "
            "or moisture content."
        ),
    }

    prov_file = Path(provenance_path) if provenance_path else Path("data/provenance_real.json")
    prov_file.parent.mkdir(parents=True, exist_ok=True)
    with open(prov_file, "w", encoding="utf-8") as f:
        json.dump(provenance_record, f, indent=2)

    print(f"\n[SUCCESS] Real dataset ingestion and split creation complete!")
    print(f"  - Train split: {len(train_df)} samples -> {splits_dir / 'train_manifest.csv'}")
    print(f"  - Val split:   {len(val_df)} samples -> {splits_dir / 'val_manifest.csv'}")
    print(f"  - Test split:  {len(test_df)} samples -> {splits_dir / 'test_manifest.csv'}")
    print(f"  - Provenance:  {prov_file.resolve()}")

    return provenance_record


def main():
    parser = argparse.ArgumentParser(
        description="Ingest, validate, dedup, and split genuine chickpea seed photographs."
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default="data/real_raw",
        help="Directory containing real seed photographs organized by class (default: data/real_raw).",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default="data/real_raw",
        help="Destination directory for validated real dataset (default: data/real_raw).",
    )
    parser.add_argument(
        "--splits-dir",
        type=str,
        default="data/processed/splits_real",
        help="Destination directory for real data train/val/test splits (default: data/processed/splits_real).",
    )
    parser.add_argument(
        "--init-structure",
        action="store_true",
        help="Create scaffold directories and annotation guide in target-dir.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Audit dataset integrity and duplicates without writing splits.",
    )
    parser.add_argument(
        "--provenance-source",
        type=str,
        default="Field & Laboratory Pulse Collection",
        help="Metadata source identifier for provenance record.",
    )
    parser.add_argument(
        "--license",
        type=str,
        default="CC BY 4.0",
        help="Licensing information for the genuine dataset.",
    )
    args = parser.parse_args()

    target_dir = Path(args.target_dir)

    if args.init_structure:
        initialize_real_data_scaffold(target_dir)
        return

    source_dir = Path(args.source_dir)
    splits_dir = Path(args.splits_dir)

    if not source_dir.exists():
        print(f"[INFO] Source directory does not exist: {source_dir.resolve()}")
        print("Initializing standard directory structure for genuine seed photos...")
        initialize_real_data_scaffold(source_dir)
        return

    res = audit_and_ingest_real_dataset(
        source_dir=source_dir,
        target_dir=target_dir,
        splits_dir=splits_dir,
        provenance_source=args.provenance_source,
        license_name=args.license,
        copy_files=not args.verify_only,
    )

    if res.get("status") in ["empty", "insufficient_samples"]:
        print(f"\n[ACTION REQUIRED] Real data ingestion halted: status='{res.get('status')}'.")
        print("Please review data/real_raw/README.md and add genuine single-seed photos for all classes.")
        sys.exit(1)


if __name__ == "__main__":
    main()
