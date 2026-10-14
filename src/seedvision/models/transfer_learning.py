"""Transfer learning architectures (MobileNetV2) for SeedVision AI."""

from typing import Optional
import logging
import torch
import torch.nn as nn
import torchvision.models as models

from seedvision.config import ModelConfig

logger = logging.getLogger(__name__)


class SeedTransferModel(nn.Module):
    """Transfer Learning model based on MobileNetV2 with configurable backbone fine-tuning."""

    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        unfreeze_layers: int = 4,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        self.unfreeze_layers = unfreeze_layers

        # Load MobileNetV2
        try:
            if pretrained:
                weights = models.MobileNet_V2_Weights.DEFAULT
                self.backbone = models.mobilenet_v2(weights=weights)
                logger.info("Loaded pretrained MobileNetV2 weights.")
            else:
                self.backbone = models.mobilenet_v2(weights=None)
                logger.info("Initialized MobileNetV2 without pretrained weights.")
        except Exception as e:
            logger.warning(
                f"Failed to load pretrained weights from network ({str(e)}). Falling back to random initialization."
            )
            self.backbone = models.mobilenet_v2(weights=None)

        # Backbone freezing strategy
        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False

            # Optionally unfreeze the top N layers of the feature extractor
            if unfreeze_layers > 0:
                for layer in list(self.backbone.features.children())[-unfreeze_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True

        # Replace classification head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2.0),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_gradcam_target_layer(self) -> nn.Module:
        """Return final convolutional layer of MobileNetV2 feature extractor."""
        # features[18] is the 1x1 conv layer right before global pool
        return self.backbone.features[-1]


def build_transfer_model(config: ModelConfig) -> SeedTransferModel:
    """Factory helper to build SeedTransferModel from ModelConfig."""
    return SeedTransferModel(
        num_classes=config.num_classes,
        pretrained=config.pretrained,
        freeze_backbone=config.freeze_backbone,
        unfreeze_layers=config.unfreeze_layers,
        dropout_rate=config.dropout_rate,
    )
