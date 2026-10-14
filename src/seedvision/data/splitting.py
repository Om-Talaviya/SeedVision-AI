"""Leakage-resistant dataset splitting utilities for SeedVision AI."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

logger = logging.getLogger(__name__)


def create_leakage_resistant_splits(
    manifest_df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    group_col: Optional[str] = "seed_id",
    class_col: str = "class_name",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into train, val, and test subsets without data leakage.

    If `group_col` is present and contains distinct group IDs (e.g. physical seed identifier,
    tray ID, or harvest batch), a Group-based split is enforced so that all views or duplicates
    of the same seed stay strictly within one split partition.

    Otherwise, stratified sampling across class labels is performed.

    Args:
        manifest_df: DataFrame containing at least 'filepath' and `class_col`.
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.
        test_ratio: Fraction for test set.
        group_col: Column name indicating physical seed / session grouping.
        class_col: Column name indicating categorical target class.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0, atol=1e-4):
        raise ValueError(
            f"Split ratios must sum to 1.0 (got {train_ratio} + {val_ratio} + {test_ratio} = {total})"
        )

    df = manifest_df.copy().reset_index(drop=True)

    # Stratified Group Splitting across each class individually
    train_subsets = []
    val_subsets = []
    test_subsets = []

    unique_classes = df[class_col].unique()
    if len(unique_classes) == 0:
        raise ValueError("Cannot split an empty dataset manifest.")

    for cls in unique_classes:
        cls_df = df[df[class_col] == cls].copy().reset_index(drop=True)

        # Determine grouping identifier
        if group_col and group_col in cls_df.columns:
            unique_groups = cls_df[group_col].dropna().unique().tolist()
        else:
            # If no group column, each row is an independent sample
            cls_df["_auto_sample_id"] = [f"{cls}_s{i}" for i in range(len(cls_df))]
            group_col_effective = "_auto_sample_id"
            unique_groups = cls_df[group_col_effective].unique().tolist()

        k = len(unique_groups)
        if k < 3:
            raise ValueError(
                f"Class '{cls}' contains only {k} physical sample/group(s). "
                f"At least 3 distinct physical sample groups are required per class to construct "
                f"non-empty, leakage-free train, validation, and test partitions."
            )

        # Deterministically shuffle the unique groups
        rng = np.random.RandomState(random_state)
        shuffled_groups = list(unique_groups)
        rng.shuffle(shuffled_groups)

        # Compute partition group counts ensuring at least 1 group per split
        n_val = max(1, int(round(k * val_ratio)))
        n_test = max(1, int(round(k * test_ratio)))

        if n_val + n_test >= k:
            n_val = 1
            n_test = 1

        n_train = k - n_val - n_test
        if n_train < 1:
            n_train = 1
            if n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1

        val_groups = set(shuffled_groups[:n_val])
        test_groups = set(shuffled_groups[n_val : n_val + n_test])
        train_groups = set(shuffled_groups[n_val + n_test :])

        # Enforce mathematical disjointness
        assert len(train_groups.intersection(val_groups)) == 0
        assert len(train_groups.intersection(test_groups)) == 0
        assert len(val_groups.intersection(test_groups)) == 0

        # Subsetting
        active_col = group_col if (group_col and group_col in cls_df.columns) else "_auto_sample_id"
        cls_train = cls_df[cls_df[active_col].isin(train_groups)].copy()
        cls_val = cls_df[cls_df[active_col].isin(val_groups)].copy()
        cls_test = cls_df[cls_df[active_col].isin(test_groups)].copy()

        if "_auto_sample_id" in cls_df.columns:
            cls_train.drop(columns=["_auto_sample_id"], inplace=True)
            cls_val.drop(columns=["_auto_sample_id"], inplace=True)
            cls_test.drop(columns=["_auto_sample_id"], inplace=True)

        train_subsets.append(cls_train)
        val_subsets.append(cls_val)
        test_subsets.append(cls_test)

    train_df = pd.concat(train_subsets, ignore_index=True)
    val_df = pd.concat(val_subsets, ignore_index=True)
    test_df = pd.concat(test_subsets, ignore_index=True)

    # Tag partitions
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    # Final split sanity assertions
    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise RuntimeError("Splitting algorithm produced an unexpectedly empty partition.")

    return train_df, val_df, test_df


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Union[str, Path],
) -> Dict[str, str]:
    """Save split DataFrames into CSV files inside output_dir."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = {
        "train": str(out / "train_manifest.csv"),
        "val": str(out / "val_manifest.csv"),
        "test": str(out / "test_manifest.csv"),
        "all": str(out / "full_split_manifest.csv"),
    }

    train_df.to_csv(paths["train"], index=False)
    val_df.to_csv(paths["val"], index=False)
    test_df.to_csv(paths["test"], index=False)

    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    full_df.to_csv(paths["all"], index=False)

    logger.info(
        f"Splits saved: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)} to {output_dir}"
    )
    return paths
