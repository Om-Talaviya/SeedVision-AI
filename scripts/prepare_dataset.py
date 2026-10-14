"""Dataset preparation, validation, and leakage-resistant splitting script."""

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from seedvision.config import load_config
from seedvision.data.splitting import create_leakage_resistant_splits, save_splits
from seedvision.data.validation import ImageValidator, validate_dataset_structure


def generate_seed_sample_image(
    output_path: Path,
    class_name: str,
    seed_idx: int,
    image_size: int = 128,
) -> None:
    """Generate a realistic synthetic seed image for smoke-tests and verification fixtures.

    Simulates chickpea (Cicer arietinum) morphology, natural pigmentation, and defect visual features:
    - sound_healthy: Creamy beige, smooth convex angular chickpea shape with micro-texture.
    - discolored_spotted: Fungal dark necrotic spots and moisture stains on the coat.
    - broken_cracked: Mechanical fissure lines, cotyledon cleavage, fractured seed coat.
    - shrivelled_damaged: Wrinkled non-convex perimeter, insect borehole, stunted morphology.
    """
    np.random.seed(seed_idx + hash(class_name) % 10000)

    # 1. Background: Neutral laboratory tray surface (off-white / pale gray) with subtle sensor noise
    bg_color = np.random.randint(235, 245)
    img_arr = np.full((image_size, image_size, 3), bg_color, dtype=np.uint8)
    noise = np.random.randint(-4, 5, (image_size, image_size, 3))
    img_arr = np.clip(img_arr.astype(int) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(img)

    center_x = image_size // 2 + np.random.randint(-4, 5)
    center_y = image_size // 2 + np.random.randint(-4, 5)
    base_radius = int(image_size * 0.32) + np.random.randint(-3, 4)

    # Base chickpea color: warm creamy beige/light ochre
    r = int(np.random.uniform(205, 225))
    g = int(np.random.uniform(180, 198))
    b = int(np.random.uniform(130, 155))
    seed_rgb = (r, g, b)

    if class_name == "shrivelled_damaged":
        base_radius = int(base_radius * 0.78)

    # Construct angular chickpea polygon
    num_points = 24
    points = []
    for i in range(num_points):
        angle = (2 * np.pi / num_points) * i
        # Chickpea has a characteristic beak/keel at one end
        radial_jitter = np.random.uniform(0.88, 1.12)
        if 0.2 < angle < 0.6:  # simulate the chickpea beak
            radial_jitter *= 1.25

        if class_name == "shrivelled_damaged":
            # high irregular waviness for wrinkled skin
            radial_jitter *= 1.0 + 0.18 * np.sin(5 * angle)

        px = center_x + int(base_radius * radial_jitter * np.cos(angle))
        py = center_y + int(base_radius * radial_jitter * np.sin(angle))
        points.append((px, py))

    # Draw soft shadow beneath seed
    shadow_offset = 3
    shadow_points = [(p[0] + shadow_offset, p[1] + shadow_offset) for p in points]
    draw.polygon(shadow_points, fill=(185, 185, 185))

    # Draw seed body
    draw.polygon(points, fill=seed_rgb, outline=(r - 35, g - 35, b - 35))

    # Add class-specific defect signatures
    if class_name == "discolored_spotted":
        # Draw dark necrotic spots / fungal lesions
        num_spots = np.random.randint(3, 7)
        for _ in range(num_spots):
            sx = center_x + np.random.randint(-base_radius // 2, base_radius // 2)
            sy = center_y + np.random.randint(-base_radius // 2, base_radius // 2)
            s_rad = np.random.randint(3, 8)
            spot_color = (
                np.random.randint(50, 90),
                np.random.randint(40, 70),
                np.random.randint(25, 45),
            )
            draw.ellipse(
                [sx - s_rad, sy - s_rad, sx + s_rad, sy + s_rad],
                fill=spot_color,
            )

    elif class_name == "broken_cracked":
        # Draw dark jagged fracture line representing mechanical cleavage
        fx1 = center_x + np.random.randint(-base_radius // 2, -base_radius // 4)
        fy1 = center_y + np.random.randint(-base_radius // 2, base_radius // 2)
        fx2 = center_x + np.random.randint(base_radius // 4, base_radius // 2)
        fy2 = center_y + np.random.randint(-base_radius // 2, base_radius // 2)

        mid_x = (fx1 + fx2) // 2 + np.random.randint(-4, 5)
        mid_y = (fy1 + fy2) // 2 + np.random.randint(-4, 5)

        crack_color = (60, 45, 30)
        draw.line([fx1, fy1, mid_x, mid_y, fx2, fy2], fill=crack_color, width=3)
        # Cotyledon exposed inner yellow/cream line
        draw.line(
            [fx1 + 1, fy1 + 1, mid_x + 1, mid_y + 1, fx2 + 1, fy2 + 1],
            fill=(245, 230, 160),
            width=1,
        )

    elif class_name == "shrivelled_damaged":
        # Draw wrinkled contour lines and insect puncture borehole
        wrinkle_color = (r - 40, g - 40, b - 40)
        for _ in range(4):
            wx = center_x + np.random.randint(-base_radius // 3, base_radius // 3)
            wy = center_y + np.random.randint(-base_radius // 3, base_radius // 3)
            draw.arc([wx - 6, wy - 6, wx + 6, wy + 6], 0, 180, fill=wrinkle_color, width=2)
        # Insect hole
        hx = center_x + np.random.randint(-6, 6)
        hy = center_y + np.random.randint(-6, 6)
        draw.ellipse([hx - 3, hy - 3, hx + 3, hy + 3], fill=(30, 20, 15))

    # Apply subtle smoothing
    img = img.filter(ImageFilter.SMOOTH_MORE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, quality=95)


def generate_fixture_dataset(
    output_dir: Path,
    classes: list,
    samples_per_class: int = 40,
) -> None:
    """Generate balanced sample fixture dataset for software verification and CPU smoke-training."""
    print(
        f"[INFO] Generating verification fixture dataset: {samples_per_class} samples/class across {len(classes)} classes..."
    )
    for class_name in classes:
        cls_dir = output_dir / class_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(1, samples_per_class + 1):
            img_path = cls_dir / f"{class_name}_{i:03d}.jpg"
            generate_seed_sample_image(img_path, class_name, seed_idx=i)
    print(f"[SUCCESS] Generated {samples_per_class * len(classes)} fixture images in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Prepare, validate, and split SeedVision AI dataset.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Custom raw dataset directory. Defaults to data/raw or config setting.",
    )
    parser.add_argument(
        "--generate-fixtures",
        action="store_true",
        help="Generate synthetic smoke-test dataset in data/raw if no images exist.",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=50,
        help="Number of samples per class when generating fixtures (default: 50).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir = Path(args.data_dir or cfg.dataset.raw_dir)
    classes = cfg.dataset.classes

    print(f"=== SeedVision AI Dataset Preparation ===")
    print(f"Target crop: {cfg.dataset.seed_type}")
    print(f"Target classes: {classes}")
    print(f"Raw directory: {raw_dir.resolve()}")

    # 1. Check if raw directory has images or needs generation
    raw_dir.mkdir(parents=True, exist_ok=True)
    report = validate_dataset_structure(str(raw_dir), classes)

    if report.valid_images_count == 0 or args.generate_fixtures:
        print("\n[NOTE] No pre-existing images found in data/raw (or --generate-fixtures specified).")
        generate_fixture_dataset(
            raw_dir,
            classes,
            samples_per_class=args.samples_per_class,
        )
        report = validate_dataset_structure(str(raw_dir), classes)

    print("\n--- Dataset Validation Report ---")
    print(f"Total images found: {report.total_images_found}")
    print(f"Valid images count: {report.valid_images_count}")
    print(f"Corrupt images count: {report.corrupt_images_count}")
    print(f"Duplicates count: {report.duplicate_images_count}")
    print(f"Class distribution: {report.class_counts}")
    print(f"Readiness status: {report.is_ready_for_training}")
    print(f"Summary: {report.summary_message}")

    if not report.is_ready_for_training:
        print("\n[ERROR] Dataset validation failed. Please address missing or empty classes.")
        sys.exit(1)

    # 2. Build full dataset manifest
    rows = []
    validator = ImageValidator()
    for class_name in classes:
        cls_dir = raw_dir / class_name
        for p in cls_dir.iterdir():
            if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
                v_res = validator.validate_file(str(p))
                if v_res.is_valid:
                    # Unique physical seed sample ID from filename stem
                    seed_id = p.stem
                    rows.append(
                        {
                            "filepath": str(p.resolve()),
                            "filename": p.name,
                            "class_name": class_name,
                            "seed_id": seed_id,
                            "width": v_res.width,
                            "height": v_res.height,
                            "file_size_bytes": v_res.file_size_bytes,
                        }
                    )

    manifest_df = pd.DataFrame(rows)
    manifest_out = Path(cfg.dataset.manifest_path)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(manifest_out, index=False)
    print(f"\n[INFO] Dataset manifest saved to: {manifest_out} ({len(manifest_df)} records)")

    # 3. Create leakage-resistant splits
    splits_dir = Path(cfg.dataset.splits_dir)
    ratios = cfg.dataset.split_ratios
    train_df, val_df, test_df = create_leakage_resistant_splits(
        manifest_df=manifest_df,
        train_ratio=ratios["train"],
        val_ratio=ratios["val"],
        test_ratio=ratios["test"],
        group_col="seed_id",
        class_col="class_name",
        random_state=cfg.training.seed,
    )

    save_splits(train_df, val_df, test_df, splits_dir)
    print(f"\n[SUCCESS] Leakage-resistant train/val/test splits created successfully!")
    print(f"  - Train: {len(train_df)} samples ({ratios['train']*100:.0f}%)")
    print(f"  - Val:   {len(val_df)} samples ({ratios['val']*100:.0f}%)")
    print(f"  - Test:  {len(test_df)} samples ({ratios['test']*100:.0f}%)")


if __name__ == "__main__":
    main()
