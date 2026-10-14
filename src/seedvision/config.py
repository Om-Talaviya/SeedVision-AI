"""Configuration management module for SeedVision AI."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import os
import torch
import yaml


@dataclass
class NormalizationConfig:
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class DatasetConfig:
    seed_type: str = "Chickpea"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    manifest_path: str = "data/processed/dataset_manifest.csv"
    splits_dir: str = "data/processed/splits"
    classes: List[str] = field(
        default_factory=lambda: [
            "sound_healthy",
            "discolored_spotted",
            "broken_cracked",
            "shrivelled_damaged",
        ]
    )
    class_descriptions: Dict[str, str] = field(
        default_factory=lambda: {
            "sound_healthy": "Physically intact, uniform coloration, no mechanical fissures or surface lesions.",
            "discolored_spotted": "Necrotic spots, fungal blemishes, or severe discoloration from moisture.",
            "broken_cracked": "Mechanical fractures, split cotyledons, or severed seed coat.",
            "shrivelled_damaged": "Wrinkled, underdeveloped morphology, or insect penetration holes.",
        }
    )
    split_ratios: Dict[str, float] = field(
        default_factory=lambda: {"train": 0.70, "val": 0.15, "test": 0.15}
    )
    image_size: List[int] = field(default_factory=lambda: [128, 128])
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    num_workers: int = 0


@dataclass
class AugmentationConfig:
    enabled: bool = True
    horizontal_flip: bool = True
    vertical_flip: bool = True
    rotation_degrees: int = 15
    brightness_contrast: bool = True
    brightness_factor: float = 0.15
    contrast_factor: float = 0.15


@dataclass
class ModelConfig:
    architecture: str = "custom_cnn"
    name: str = "SeedNet-Baseline"
    num_classes: int = 4
    input_channels: int = 3
    conv_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    kernel_size: int = 3
    pool_size: int = 2
    dropout_rate: float = 0.35
    dense_units: int = 256
    target_layer: str = "features.10"
    pretrained: bool = True
    freeze_backbone: bool = False
    unfreeze_layers: int = 4


@dataclass
class TrainingConfig:
    epochs: int = 25
    batch_size: int = 16
    learning_rate: float = 0.001
    optimizer: str = "adam"
    weight_decay: float = 0.0001
    scheduler: str = "cosine"
    min_lr: float = 1e-5
    early_stopping_patience: int = 6
    device: str = "auto"
    seed: int = 42


@dataclass
class PathsConfig:
    checkpoints_dir: str = "artifacts/checkpoints"
    metrics_dir: str = "artifacts/metrics"
    figures_dir: str = "artifacts/figures"


@dataclass
class SeedVisionConfig:
    project_name: str = "SeedVision AI"
    version: str = "1.0.0"
    description: str = "Deep Learning-Based Agricultural Seed Quality Grader"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    def get_device(self) -> torch.device:
        """Resolve target device (CUDA if requested & available, else CPU)."""
        target = self.training.device.lower()
        if target in ("auto", "cuda"):
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        return torch.device("cpu")


def _deep_update(base_dict: dict, update_dict: dict) -> dict:
    """Recursively update a dictionary."""
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            _deep_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict


def load_config(
    config_path: Optional[str] = None,
    default_config_path: str = "configs/default.yaml",
) -> SeedVisionConfig:
    """Load configuration from YAML file, cascading over defaults.

    Args:
        config_path: Optional path to specific configuration file (e.g., cnn.yaml).
        default_config_path: Path to default base configuration file.

    Returns:
        SeedVisionConfig: Structured configuration object.
    """
    raw_config: Dict[str, Any] = {}

    # 1. Load default YAML if it exists
    if os.path.exists(default_config_path):
        with open(default_config_path, "r", encoding="utf-8") as f:
            default_data = yaml.safe_load(f) or {}
            _deep_update(raw_config, default_data)

    # 2. Layer specific config if provided
    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            specific_data = yaml.safe_load(f) or {}
            _deep_update(raw_config, specific_data)

    # 3. Build dataclass hierarchy
    dataset_dict = raw_config.get("dataset", {})
    norm_dict = dataset_dict.get("normalization", {})
    norm_cfg = NormalizationConfig(
        mean=norm_dict.get("mean", [0.485, 0.456, 0.406]),
        std=norm_dict.get("std", [0.229, 0.224, 0.225]),
    )

    dataset_cfg = DatasetConfig(
        seed_type=dataset_dict.get("seed_type", "Chickpea"),
        raw_dir=dataset_dict.get("raw_dir", "data/raw"),
        processed_dir=dataset_dict.get("processed_dir", "data/processed"),
        manifest_path=dataset_dict.get("manifest_path", "data/processed/dataset_manifest.csv"),
        splits_dir=dataset_dict.get("splits_dir", "data/processed/splits"),
        classes=dataset_dict.get(
            "classes",
            ["sound_healthy", "discolored_spotted", "broken_cracked", "shrivelled_damaged"],
        ),
        class_descriptions=dataset_dict.get("class_descriptions", {}),
        split_ratios=dataset_dict.get(
            "split_ratios", {"train": 0.70, "val": 0.15, "test": 0.15}
        ),
        image_size=dataset_dict.get("image_size", [128, 128]),
        normalization=norm_cfg,
        num_workers=dataset_dict.get("num_workers", 0),
    )

    aug_dict = raw_config.get("augmentation", {})
    aug_cfg = AugmentationConfig(
        enabled=aug_dict.get("enabled", True),
        horizontal_flip=aug_dict.get("horizontal_flip", True),
        vertical_flip=aug_dict.get("vertical_flip", True),
        rotation_degrees=aug_dict.get("rotation_degrees", 15),
        brightness_contrast=aug_dict.get("brightness_contrast", True),
        brightness_factor=aug_dict.get("brightness_factor", 0.15),
        contrast_factor=aug_dict.get("contrast_factor", 0.15),
    )

    model_dict = raw_config.get("model", {})
    model_cfg = ModelConfig(
        architecture=model_dict.get("architecture", "custom_cnn"),
        name=model_dict.get("name", "SeedNet-Baseline"),
        num_classes=model_dict.get("num_classes", len(dataset_cfg.classes)),
        input_channels=model_dict.get("input_channels", 3),
        conv_channels=model_dict.get("conv_channels", [32, 64, 128, 256]),
        kernel_size=model_dict.get("kernel_size", 3),
        pool_size=model_dict.get("pool_size", 2),
        dropout_rate=model_dict.get("dropout_rate", 0.35),
        dense_units=model_dict.get("dense_units", 256),
        target_layer=model_dict.get("target_layer", "features.10"),
        pretrained=model_dict.get("pretrained", True),
        freeze_backbone=model_dict.get("freeze_backbone", False),
        unfreeze_layers=model_dict.get("unfreeze_layers", 4),
    )

    train_dict = raw_config.get("training", {})
    train_cfg = TrainingConfig(
        epochs=train_dict.get("epochs", 25),
        batch_size=train_dict.get("batch_size", 16),
        learning_rate=train_dict.get("learning_rate", 0.001),
        optimizer=train_dict.get("optimizer", "adam"),
        weight_decay=train_dict.get("weight_decay", 0.0001),
        scheduler=train_dict.get("scheduler", "cosine"),
        min_lr=train_dict.get("min_lr", 1e-5),
        early_stopping_patience=train_dict.get("early_stopping_patience", 6),
        device=train_dict.get("device", "auto"),
        seed=train_dict.get("seed", 42),
    )

    paths_dict = raw_config.get("paths", {})
    paths_cfg = PathsConfig(
        checkpoints_dir=paths_dict.get("checkpoints_dir", "artifacts/checkpoints"),
        metrics_dir=paths_dict.get("metrics_dir", "artifacts/metrics"),
        figures_dir=paths_dict.get("figures_dir", "artifacts/figures"),
    )

    project_dict = raw_config.get("project", {})

    return SeedVisionConfig(
        project_name=project_dict.get("name", "SeedVision AI"),
        version=project_dict.get("version", "1.0.0"),
        description=project_dict.get(
            "description", "Deep Learning-Based Agricultural Seed Quality Grader"
        ),
        dataset=dataset_cfg,
        augmentation=aug_cfg,
        model=model_cfg,
        training=train_cfg,
        paths=paths_cfg,
    )
