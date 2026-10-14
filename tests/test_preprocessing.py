"""Unit tests for image preprocessing and normalization transforms."""

import numpy as np
from PIL import Image
import pytest
import torch

from seedvision.config import AugmentationConfig, NormalizationConfig
from seedvision.data.preprocessing import (
    denormalize_tensor,
    get_eval_transforms,
    get_train_transforms,
    preprocess_pil_image,
)


def test_eval_transforms_output_shape():
    img = Image.new("RGB", (200, 150), color=(120, 180, 200))
    transform = get_eval_transforms(image_size=(128, 128))
    tensor = transform(img)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 128, 128)
    assert tensor.dtype == torch.float32


def test_train_transforms_with_augmentation():
    img = Image.new("RGB", (100, 100), color=(200, 100, 50))
    aug_cfg = AugmentationConfig(enabled=True, horizontal_flip=True, rotation_degrees=20)
    transform = get_train_transforms(image_size=(64, 64), augmentation=aug_cfg)
    tensor = transform(img)

    assert tensor.shape == (3, 64, 64)


def test_preprocess_pil_image_batch_dim():
    img = Image.new("RGB", (80, 80), color=(50, 150, 50))
    tensor = preprocess_pil_image(img, image_size=(128, 128))

    assert tensor.dim() == 4  # [1, 3, 128, 128]
    assert tensor.shape == (1, 3, 128, 128)


def test_denormalize_tensor():
    norm = NormalizationConfig(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    # A normalized tensor of all 0.0 corresponds to an unnormalized pixel of 0.5
    norm_tensor = torch.zeros((3, 64, 64), dtype=torch.float32)
    img_np = denormalize_tensor(norm_tensor, normalization=norm)

    assert img_np.shape == (64, 64, 3)
    assert np.allclose(img_np, 0.5, atol=1e-3)
