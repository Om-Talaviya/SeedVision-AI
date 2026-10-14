"""Grad-CAM (Gradient-weighted Class Activation Mapping) for SeedVision AI."""

from typing import Optional, Tuple, Union
import logging
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class GradCAM:
    """Computes Grad-CAM saliency heatmaps for convolutional neural networks."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """Initialize GradCAM hook manager.

        Args:
            model: PyTorch model (eval mode recommended).
            target_layer: The target convolutional layer (e.g., model.features[-1]).
        """
        self.model = model
        self.target_layer = target_layer

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.handles: list = []

        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on the target layer."""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            # grad_output[0] contains gradients with respect to output feature map
            self.gradients = grad_output[0].detach()

        h_fwd = self.target_layer.register_forward_hook(forward_hook)
        h_bwd = self.target_layer.register_full_backward_hook(backward_hook)
        self.handles.extend([h_fwd, h_bwd])

    def remove_hooks(self) -> None:
        """Remove hooks to release temporary computational graphs and memory."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.activations = None
        self.gradients = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """Generate normalized 2D Grad-CAM heatmap for a given input and class.

        Args:
            input_tensor: Tensor of shape [1, 3, H, W].
            target_class_idx: Integer class index to explain. If None, uses top predicted class.

        Returns:
            np.ndarray: 2D array of shape [H, W] with float values in [0.0, 1.0].
        """
        self.model.eval()
        self.model.zero_grad()

        # Ensure tensor requires gradient
        input_tensor = input_tensor.clone()

        # Forward pass
        logits = self.model(input_tensor)

        if target_class_idx is None:
            target_class_idx = int(torch.argmax(logits, dim=1).item())

        score = logits[0, target_class_idx]

        # Backward pass to compute gradients at target layer
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError(
                "Gradients or activations were not captured. Ensure target_layer is executed during forward pass."
            )

        # Global average pooling of gradients: [1, C, H, W] -> [1, C, 1, 1]
        pooled_gradients = torch.mean(self.gradients, dim=[2, 3], keepdim=True)

        # Weight activations by pooled gradients
        cam = torch.sum(pooled_gradients * self.activations, dim=1, keepdim=True)

        # Apply ReLU to retain only features having positive influence on target class
        cam = F.relu(cam)

        # Resize heatmap to input tensor dimensions
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)

        cam_np = cam.squeeze().cpu().numpy()

        # Min-max normalization
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max > cam_min:
            cam_normalized = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_normalized = np.zeros_like(cam_np)

        return cam_normalized


def generate_gradcam_overlay(
    image: Union[Image.Image, np.ndarray],
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> Tuple[np.ndarray, np.ndarray]:
    """Overlay a normalized 2D heatmap on an RGB image.

    Args:
        image: Original PIL Image or RGB numpy array in [0, 255].
        heatmap: 2D float array in [0, 1] from GradCAM.
        alpha: Blending weight for heatmap overlay (0.0 = only image, 1.0 = only heatmap).
        colormap: OpenCV colormap enum (default COLORMAP_JET).

    Returns:
        Tuple of (colored_heatmap_rgb, blended_overlay_rgb) as uint8 arrays [H, W, 3].
    """
    if isinstance(image, Image.Image):
        img_np = np.array(image.convert("RGB"))
    else:
        img_np = image.copy()
        if img_np.dtype != np.uint8:
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)

    h, w = img_np.shape[:2]

    # Resize heatmap to match image dimensions
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Convert to 8-bit [0, 255]
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    # Apply color map (OpenCV outputs BGR)
    colored_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)

    # Blend original and heatmap
    overlay_rgb = np.uint8(alpha * colored_rgb + (1 - alpha) * img_np)

    return colored_rgb, overlay_rgb
