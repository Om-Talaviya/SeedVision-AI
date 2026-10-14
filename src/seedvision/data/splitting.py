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

    for cls in unique_classes:
        cls_df = df[df[class_col] == cls].copy().reset_index(drop=True)

        if group_col and group_col in cls_df.columns and cls_df[group_col].nunique() > 2:
            groups = cls_df[group_col].values
            test_val_fraction = val_ratio + test_ratio

            # First split: train vs (val + test)
            gss_1 = GroupShuffleSplit(
                n_splits=1, test_size=test_val_fraction, random_state=random_state
            )
            train_idx, temp_idx = next(gss_1.split(cls_df, groups=groups))

            cls_train = cls_df.iloc[train_idx].copy()
            cls_temp = cls_df.iloc[temp_idx].copy().reset_index(drop=True)

            # Second split: val vs test
            test_fraction_of_temp = test_ratio / (val_ratio + test_ratio)
            temp_groups = cls_temp[group_col].values

            if cls_temp[group_col].nunique() > 1:
                gss_2 = GroupShuffleSplit(
                    n_splits=1, test_size=test_fraction_of_temp, random_state=random_state
                )
                val_idx, test_idx = next(gss_2.split(cls_temp, groups=temp_groups))
                cls_val = cls_temp.iloc[val_idx].copy()
                cls_test = cls_temp.iloc[test_idx].copy()
            else:
                half = len(cls_temp) // 2
                cls_val = cls_temp.iloc[:half].copy()
                cls_test = cls_temp.iloc[half:].copy()

        else:
            # Fallback simple partition per class
            n = len(cls_df)
            shuffled = cls_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)

            cls_train = shuffled.iloc[:n_train].copy()
            cls_val = shuffled.iloc[n_train : n_train + n_val].copy()
            cls_test = shuffled.iloc[n_train + n_val :].copy()

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
