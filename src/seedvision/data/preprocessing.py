"""Preprocessing and data augmentation pipelines for SeedVision AI."""

from typing import List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

from seedvision.config import AugmentationConfig, NormalizationConfig


def get_train_transforms(
    image_size: Union[int, Tuple[int, int], List[int]] = (128, 128),
    normalization: Optional[NormalizationConfig] = None,
    augmentation: Optional[AugmentationConfig] = None,
) -> T.Compose:
    """Construct data augmentation and preprocessing pipeline for training.

    Preserves defect characteristics (e.g. cracks, spots) without unrealistic distortion.
    """
    if isinstance(image_size, list):
        size = tuple(image_size)
    elif isinstance(image_size, int):
        size = (image_size, image_size)
    else:
        size = image_size

    norm = normalization or NormalizationConfig()
    aug = augmentation or AugmentationConfig()

    transforms_list: List[object] = [
        T.Resize(size, interpolation=T.InterpolationMode.BILINEAR),
    ]

    if aug.enabled:
        if aug.horizontal_flip:
            transforms_list.append(T.RandomHorizontalFlip(p=0.5))
        if aug.vertical_flip:
            transforms_list.append(T.RandomVerticalFlip(p=0.5))
        if aug.rotation_degrees > 0:
            transforms_list.append(
                T.RandomRotation(
                    degrees=aug.rotation_degrees,
                    interpolation=T.InterpolationMode.BILINEAR,
                )
            )
        if aug.brightness_contrast:
            transforms_list.append(
                T.ColorJitter(
                    brightness=aug.brightness_factor,
                    contrast=aug.contrast_factor,
                    saturation=0.05,
                    hue=0.02,
                )
            )

    transforms_list.extend(
        [
            T.ToTensor(),
            T.Normalize(mean=norm.mean, std=norm.std),
        ]
    )

    return T.Compose(transforms_list)


def get_eval_transforms(
    image_size: Union[int, Tuple[int, int], List[int]] = (128, 128),
    normalization: Optional[NormalizationConfig] = None,
) -> T.Compose:
    """Construct deterministic preprocessing pipeline for validation, test, and inference."""
    if isinstance(image_size, list):
        size = tuple(image_size)
    elif isinstance(image_size, int):
        size = (image_size, image_size)
    else:
        size = image_size

    norm = normalization or NormalizationConfig()

    return T.Compose(
        [
            T.Resize(size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=norm.mean, std=norm.std),
        ]
    )


def preprocess_pil_image(
    image: Image.Image,
    image_size: Union[int, Tuple[int, int], List[int]] = (128, 128),
    normalization: Optional[NormalizationConfig] = None,
) -> torch.Tensor:
    """Preprocess a single PIL Image for inference.

    Ensures RGB mode conversion and returns batch-dimensioned tensor [1, 3, H, W].
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    transform = get_eval_transforms(image_size=image_size, normalization=normalization)
    tensor = transform(image)
    return tensor.unsqueeze(0)  # Shape: [1, 3, H, W]


def denormalize_tensor(
    tensor: torch.Tensor,
    normalization: Optional[NormalizationConfig] = None,
) -> np.ndarray:
    """Reverse ImageNet/custom normalization for display and Grad-CAM overlay.

    Args:
        tensor: Tensor of shape [3, H, W] or [1, 3, H, W].
        normalization: NormalizationConfig.

    Returns:
        np.ndarray: Image array [H, W, 3] with float values in [0, 1].
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    norm = normalization or NormalizationConfig()
    mean = torch.tensor(norm.mean, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(norm.std, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)

    denorm = tensor * std + mean
    denorm = torch.clamp(denorm, 0.0, 1.0)
    # Convert CHW to HWC
    img_np = denorm.detach().cpu().permute(1, 2, 0).numpy()
    return img_np
