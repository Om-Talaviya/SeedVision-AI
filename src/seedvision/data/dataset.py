"""PyTorch Dataset implementation and DataLoader factories for SeedVision AI."""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from seedvision.config import SeedVisionConfig
from seedvision.data.preprocessing import get_eval_transforms, get_train_transforms


class SeedDataset(Dataset):
    """Agricultural seed image dataset supporting both CSV manifests and class folders."""

    def __init__(
        self,
        data_source: Union[pd.DataFrame, str, Path],
        classes: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        is_training: bool = False,
    ):
        """Initialize SeedDataset.

        Args:
            data_source: DataFrame with 'filepath' and 'class_name' columns, or path to CSV/directory.
            classes: Ordered list of class names. If None, inferred alphabetically from source.
            transform: PyTorch torchvision transform callable.
            is_training: Boolean flag for training mode.
        """
        self.transform = transform
        self.is_training = is_training

        # 1. Parse data_source
        if isinstance(data_source, pd.DataFrame):
            self.df = data_source.copy().reset_index(drop=True)
        elif isinstance(data_source, (str, Path)):
            src_path = Path(data_source)
            if src_path.is_file() and src_path.suffix.lower() == ".csv":
                self.df = pd.read_csv(src_path)
            elif src_path.is_dir():
                self.df = self._scan_directory(src_path)
            else:
                raise ValueError(f"Invalid data_source: {data_source}")
        else:
            raise TypeError(f"Unsupported data_source type: {type(data_source)}")

        if "filepath" not in self.df.columns or "class_name" not in self.df.columns:
            raise ValueError("Dataset DataFrame must contain 'filepath' and 'class_name' columns.")

        # 2. Determine classes and mapping
        if classes is not None:
            self.classes = list(classes)
        else:
            self.classes = sorted(self.df["class_name"].unique().tolist())

        self.class_to_idx: Dict[str, int] = {c: i for i, c in enumerate(self.classes)}
        self.idx_to_class: Dict[int, str] = {i: c for i, c in enumerate(self.classes)}

        # Filter out rows whose class_name is not in self.classes
        valid_mask = self.df["class_name"].isin(self.class_to_idx)
        self.df = self.df[valid_mask].reset_index(drop=True)

    def _scan_directory(self, root_dir: Path) -> pd.DataFrame:
        """Scan directory with class subfolders and return a manifest DataFrame."""
        rows = []
        supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for class_dir in root_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                for img_file in class_dir.iterdir():
                    if img_file.is_file() and img_file.suffix.lower() in supported_exts:
                        rows.append(
                            {
                                "filepath": str(img_file.resolve()),
                                "class_name": class_name,
                                "filename": img_file.name,
                            }
                        )
        return pd.DataFrame(rows)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Union[str, int]]]:
        row = self.df.iloc[idx]
        filepath = str(row["filepath"])
        class_name = str(row["class_name"])
        label_idx = self.class_to_idx[class_name]

        try:
            with Image.open(filepath) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                else:
                    img = img.copy()
        except Exception as e:
            raise IOError(f"Failed to read image at {filepath}: {str(e)}") from e

        if self.transform is not None:
            image_tensor = self.transform(img)
        else:
            image_tensor = get_eval_transforms()(img)

        metadata = {
            "filepath": filepath,
            "class_name": class_name,
            "index": idx,
        }

        return image_tensor, label_idx, metadata

    def get_class_counts(self) -> Dict[str, int]:
        """Return counts of images per class."""
        counts = self.df["class_name"].value_counts().to_dict()
        return {c: counts.get(c, 0) for c in self.classes}


def create_dataloaders(
    config: SeedVisionConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Build PyTorch DataLoaders with appropriate transforms and worker settings."""
    classes = config.dataset.classes
    img_size = config.dataset.image_size
    norm = config.dataset.normalization
    aug = config.augmentation

    train_transform = get_train_transforms(image_size=img_size, normalization=norm, augmentation=aug)
    eval_transform = get_eval_transforms(image_size=img_size, normalization=norm)

    train_ds = SeedDataset(train_df, classes=classes, transform=train_transform, is_training=True)
    val_ds = SeedDataset(val_df, classes=classes, transform=eval_transform, is_training=False)

    batch_size = config.training.batch_size
    num_workers = config.dataset.num_workers
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=len(train_ds) > batch_size,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    test_loader = None
    if test_df is not None:
        test_ds = SeedDataset(test_df, classes=classes, transform=eval_transform, is_training=False)
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    return train_loader, val_loader, test_loader
