"""Unit tests for Grad-CAM explainability and overlay generation."""

import numpy as np
from PIL import Image
import pytest
import torch

from seedvision.explainability.gradcam import GradCAM, generate_gradcam_overlay
from seedvision.models.cnn import CustomSeedCNN


def test_gradcam_heatmap_generation():
    model = CustomSeedCNN(num_classes=4)
    target_layer = model.get_gradcam_target_layer()

    with GradCAM(model, target_layer) as gradcam:
        dummy_tensor = torch.randn(1, 3, 64, 64, requires_grad=True)
        heatmap = gradcam.generate_heatmap(dummy_tensor, target_class_idx=1)

        assert isinstance(heatmap, np.ndarray)
        assert heatmap.shape == (64, 64)
        assert 0.0 <= heatmap.min() <= heatmap.max() <= 1.0


def test_gradcam_overlay_blending():
    img = Image.new("RGB", (128, 128), color=(200, 180, 140))
    heatmap = np.random.uniform(0.0, 1.0, size=(128, 128)).astype(np.float32)

    colored_rgb, overlay_rgb = generate_gradcam_overlay(img, heatmap, alpha=0.5)

    assert colored_rgb.shape == (128, 128, 3)
    assert overlay_rgb.shape == (128, 128, 3)
    assert overlay_rgb.dtype == np.uint8


def test_gradcam_hook_cleanup():
    model = CustomSeedCNN(num_classes=4)
    target_layer = model.get_gradcam_target_layer()

    gradcam = GradCAM(model, target_layer)
    assert len(gradcam.handles) == 2

    gradcam.remove_hooks()
    assert len(gradcam.handles) == 0
    assert gradcam.activations is None
    assert gradcam.gradients is None


def test_gradcam_mobilenetv2_support():
    from seedvision.models.transfer_learning import SeedTransferModel

    model = SeedTransferModel(num_classes=4, pretrained=False)
    target_layer = model.get_gradcam_target_layer()

    with GradCAM(model, target_layer) as gradcam:
        dummy_tensor = torch.randn(1, 3, 64, 64, requires_grad=True)
        heatmap = gradcam.generate_heatmap(dummy_tensor, target_class_idx=2)

        assert isinstance(heatmap, np.ndarray)
        assert heatmap.shape == (64, 64)
        assert 0.0 <= heatmap.min() <= heatmap.max() <= 1.0

