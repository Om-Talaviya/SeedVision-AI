"""Unit tests for deep learning model architectures."""

import pytest
import torch

from seedvision.config import ModelConfig
from seedvision.models.cnn import CustomSeedCNN, build_custom_cnn
from seedvision.models.factory import build_model, get_model_info
from seedvision.models.transfer_learning import SeedTransferModel


def test_custom_cnn_forward():
    model = CustomSeedCNN(num_classes=4, input_channels=3)
    dummy_input = torch.randn(2, 3, 128, 128)

    logits = model(dummy_input)
    assert logits.shape == (2, 4)


def test_custom_cnn_backward():
    model = CustomSeedCNN(num_classes=4)
    dummy_input = torch.randn(2, 3, 64, 64)
    logits = model(dummy_input)
    loss = logits.sum()
    loss.backward()

    # Check gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None


def test_transfer_model_architecture():
    model = SeedTransferModel(num_classes=4, pretrained=False)
    dummy_input = torch.randn(2, 3, 128, 128)

    logits = model(dummy_input)
    assert logits.shape == (2, 4)


def test_model_factory_and_info():
    cfg = ModelConfig(architecture="custom_cnn", num_classes=4)
    model = build_model(cfg)
    assert isinstance(model, CustomSeedCNN)

    info = get_model_info(model)
    assert info["total_parameters"] > 0
    assert info["trainable_parameters"] > 0
    assert info["estimated_size_mb"] > 0
