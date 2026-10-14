"""Custom Baseline Convolutional Neural Network for SeedVision AI."""

from typing import List, Optional
import torch
import torch.nn as nn

from seedvision.config import ModelConfig


class ConvBlock(nn.Module):
    """Convolutional block: Conv2d -> BatchNorm2d -> LeakyReLU -> MaxPool2d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pool_size: int = 2,
        use_pool: bool = True,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size) if use_pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        return x


class CustomSeedCNN(nn.Module):
    """Custom 4-stage convolutional neural network for seed defect classification.

    Designed for fast CPU and GPU execution with strong spatial feature extraction,
    batch normalization for stable convergence, and clear target layers for Grad-CAM.
    """

    def __init__(
        self,
        num_classes: int = 4,
        input_channels: int = 3,
        conv_channels: Optional[List[int]] = None,
        dropout_rate: float = 0.35,
        dense_units: int = 256,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.conv_channels = conv_channels or [32, 64, 128, 256]

        # Feature extractor stages
        layers = []
        in_ch = input_channels
        for i, out_ch in enumerate(self.conv_channels):
            # Use pooling for all stages
            block = ConvBlock(in_channels=in_ch, out_channels=out_ch, use_pool=True)
            layers.append(block)
            in_ch = out_ch

        self.features = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier head
        last_ch = self.conv_channels[-1]
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(last_ch, dense_units),
            nn.BatchNorm1d(dense_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(dense_units, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming normal distribution."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)
        logits = self.classifier(x)
        return logits

    def get_gradcam_target_layer(self) -> nn.Module:
        """Return the final convolutional layer for Grad-CAM gradient tracking."""
        # Last conv block inside features
        last_block = self.features[-1]
        return last_block.conv


def build_custom_cnn(config: ModelConfig) -> CustomSeedCNN:
    """Factory helper to build CustomSeedCNN from ModelConfig."""
    return CustomSeedCNN(
        num_classes=config.num_classes,
        input_channels=config.input_channels,
        conv_channels=config.conv_channels,
        dropout_rate=config.dropout_rate,
        dense_units=config.dense_units,
    )
