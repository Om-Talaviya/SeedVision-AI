"""Model factory and introspection utilities for SeedVision AI."""

from typing import Any, Dict
import torch
import torch.nn as nn

from seedvision.config import ModelConfig
from seedvision.models.cnn import CustomSeedCNN, build_custom_cnn
from seedvision.models.transfer_learning import SeedTransferModel, build_transfer_model


def build_model(model_config: ModelConfig) -> nn.Module:
    """Build and return model based on architecture specified in config."""
    arch = model_config.architecture.lower()
    if arch in ("custom_cnn", "baseline_cnn", "cnn"):
        return build_custom_cnn(model_config)
    elif arch in ("mobilenet_v2", "mobilenet", "transfer_learning"):
        return build_transfer_model(model_config)
    else:
        raise ValueError(
            f"Unsupported architecture '{model_config.architecture}'. Supported: 'custom_cnn', 'mobilenet_v2'."
        )


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """Calculate parameter counts and approximate memory size."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    # Estimate size in MB (float32 = 4 bytes per param)
    size_mb = (total_params * 4) / (1024 * 1024)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": frozen_params,
        "estimated_size_mb": round(size_mb, 2),
    }
