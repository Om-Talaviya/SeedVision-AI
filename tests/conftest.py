"""Pytest configuration and session-scoped fixtures for SeedVision AI."""

from pathlib import Path
import sys
import pandas as pd
import pytest

# Ensure repository root is on sys.path across all platforms
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


@pytest.fixture(scope="session", autouse=True)
def ensure_verification_fixtures():
    """Ensure synthetic procedural fixtures and split manifests exist for testing.

    In a clean checkout or fresh CI runner where procedural artifacts were not committed,
    this generates the standard 240 procedural verification images and split manifests.
    """
    manifest_file = _PROJECT_ROOT / "data" / "processed" / "splits" / "train_manifest.csv"
    raw_sample = _PROJECT_ROOT / "data" / "raw" / "sound_healthy" / "sound_healthy_001.jpg"

    if not manifest_file.is_file() or not raw_sample.is_file():
        from scripts.prepare_dataset import generate_fixture_dataset
        from seedvision.config import load_config
        from seedvision.data.splitting import create_leakage_resistant_splits, save_splits
        from seedvision.data.validation import ImageValidator

        cfg = load_config()
        classes = cfg.dataset.classes
        raw_dir = _PROJECT_ROOT / cfg.dataset.raw_dir

        generate_fixture_dataset(raw_dir, classes, samples_per_class=60)

        rows = []
        validator = ImageValidator()
        for class_name in classes:
            cls_dir = raw_dir / class_name
            for p in cls_dir.iterdir():
                if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
                    v_res = validator.validate_file(str(p))
                    if v_res.is_valid:
                        rows.append(
                            {
                                "filepath": str(p.resolve()),
                                "filename": p.name,
                                "class_name": class_name,
                                "seed_id": p.stem,
                                "width": v_res.width,
                                "height": v_res.height,
                                "file_size_bytes": v_res.file_size_bytes,
                            }
                        )

        manifest_df = pd.DataFrame(rows)
        manifest_out = _PROJECT_ROOT / cfg.dataset.manifest_path
        manifest_out.parent.mkdir(parents=True, exist_ok=True)
        manifest_df.to_csv(manifest_out, index=False)

        splits_dir = _PROJECT_ROOT / cfg.dataset.splits_dir
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
