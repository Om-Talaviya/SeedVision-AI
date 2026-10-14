"""Unit tests for configuration loading and validation."""

from pathlib import Path
import pytest

from seedvision.config import SeedVisionConfig, load_config


def test_default_config_loading():
    """Verify default configuration values and device resolution."""
    cfg = load_config("configs/default.yaml")

    assert isinstance(cfg, SeedVisionConfig)
    assert cfg.project_name == "SeedVision AI"
    assert cfg.dataset.seed_type == "Chickpea"
    assert len(cfg.dataset.classes) == 4
    assert "sound_healthy" in cfg.dataset.classes
    assert cfg.dataset.image_size == [128, 128]

    # Test device resolution
    device = cfg.get_device()
    assert device.type in ("cpu", "cuda")


def test_cnn_config_overlay():
    """Verify CNN configuration overlays default configuration."""
    cfg = load_config("configs/cnn.yaml")

    assert cfg.model.architecture == "custom_cnn"
    assert cfg.model.name == "SeedNet-Baseline"
    assert cfg.model.dropout_rate == 0.35
    assert len(cfg.model.conv_channels) == 4


def test_transfer_learning_config_overlay():
    """Verify Transfer Learning configuration overlays default configuration."""
    cfg = load_config("configs/transfer_learning.yaml")

    assert cfg.model.architecture == "mobilenet_v2"
    assert cfg.model.name == "MobileNetV2-Transfer"
    assert cfg.training.optimizer.lower() == "adamw"
