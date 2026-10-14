"""End-to-end inference and explainability engine for SeedVision AI."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import time
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from seedvision.config import NormalizationConfig
from seedvision.data.preprocessing import preprocess_pil_image
from seedvision.explainability.gradcam import GradCAM, generate_gradcam_overlay
from seedvision.training.checkpoints import load_checkpoint

logger = logging.getLogger(__name__)

SCIENTIFIC_LIMITATION_NOTICE = (
    "NOTICE: This assessment reflects visually observable external physical features only. "
    "It does not evaluate germination capacity, embryo viability, moisture percentage, internal "
    "infestation, or regulatory seed lot certification. Do not rely on visual inference alone "
    "for official agricultural trade or planting certification."
)


@dataclass
class TopKPrediction:
    rank: int
    class_name: str
    probability: float
    percentage_str: str


@dataclass
class PredictionResult:
    predicted_class: str
    confidence_score: float
    is_uncertain: bool
    uncertainty_reason: Optional[str]
    top_k: List[TopKPrediction]
    raw_probabilities: Dict[str, float]
    inference_latency_ms: float
    model_name: str
    architecture: str
    heatmap_2d: Optional[np.ndarray] = None
    overlay_rgb: Optional[np.ndarray] = None
    colored_heatmap_rgb: Optional[np.ndarray] = None
    gradcam_target_class: Optional[str] = None
    gradcam_error: Optional[str] = None
    scientific_notice: str = SCIENTIFIC_LIMITATION_NOTICE


class SeedPredictor:
    """Loads a trained checkpoint and provides safe, explainable seed quality inference."""

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: str = "cpu",
        uncertainty_threshold: float = 0.50,
    ):
        self.device = torch.device(device)
        self.checkpoint_path = Path(checkpoint_path)
        self.uncertainty_threshold = uncertainty_threshold

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

        # Load checkpoint payload and model
        self.model, self.payload = load_checkpoint(
            checkpoint_path=self.checkpoint_path,
            device=self.device,
        )
        self.model.eval()

        self.architecture = self.payload.get("architecture", "custom_cnn")
        self.model_name = self.payload.get("model_name", self.checkpoint_path.stem)
        self.classes: List[str] = self.payload.get(
            "classes",
            ["sound_healthy", "discolored_spotted", "broken_cracked", "shrivelled_damaged"],
        )
        self.image_size: List[int] = self.payload.get("image_size", [128, 128])

        norm_dict = self.payload.get("normalization", {})
        self.normalization = NormalizationConfig(
            mean=norm_dict.get("mean", [0.485, 0.456, 0.406]),
            std=norm_dict.get("std", [0.229, 0.224, 0.225]),
        )

        # Resolve target layer for Grad-CAM
        self.target_layer = self._resolve_target_layer()

    def _resolve_target_layer(self) -> Optional[nn.Module]:
        """Identify the target layer for Grad-CAM from model architecture."""
        try:
            if hasattr(self.model, "get_gradcam_target_layer"):
                return self.model.get_gradcam_target_layer()

            # Fallback inspection for Sequential models
            if hasattr(self.model, "features"):
                last_child = list(self.model.features.children())[-1]
                if hasattr(last_child, "conv"):
                    return last_child.conv
                return last_child

            return None
        except Exception as e:
            logger.warning(f"Could not automatically resolve Grad-CAM layer: {e}")
            return None

    def predict(
        self,
        image: Union[Image.Image, str, Path],
        top_k_count: int = 4,
        explain_class: Optional[str] = None,
        generate_cam: bool = True,
    ) -> PredictionResult:
        """Run complete inference and Grad-CAM explanation.

        Args:
            image: PIL Image or path to image file.
            top_k_count: Number of top classes to include.
            explain_class: Specific class name to explain via Grad-CAM (defaults to top-1).
            generate_cam: Boolean whether to compute Grad-CAM heatmap.

        Returns:
            PredictionResult object.
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB") if image.mode != "RGB" else image

        # Preprocess tensor
        input_tensor = preprocess_pil_image(
            image=pil_image,
            image_size=self.image_size,
            normalization=self.normalization,
        ).to(self.device)

        # Enable grad for tensor to support Grad-CAM
        input_tensor.requires_grad = True

        start_time = time.perf_counter()
        with torch.enable_grad():
            logits = self.model(input_tensor)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Calculate softmax probabilities
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]

        top_indices = np.argsort(probs)[::-1]
        top1_idx = int(top_indices[0])
        top1_class = self.classes[top1_idx]
        top1_conf = float(probs[top1_idx])

        # Raw probabilities dictionary
        raw_probs = {cls_name: float(probs[i]) for i, cls_name in enumerate(self.classes)}

        # Format top-k predictions
        top_k_list: List[TopKPrediction] = []
        for rank, idx in enumerate(top_indices[:top_k_count], start=1):
            top_k_list.append(
                TopKPrediction(
                    rank=rank,
                    class_name=self.classes[idx],
                    probability=round(float(probs[idx]), 4),
                    percentage_str=f"{probs[idx] * 100:.1f}%",
                )
            )

        # Uncertainty evaluation
        is_uncertain = False
        uncertainty_reason = None

        if top1_conf < self.uncertainty_threshold:
            is_uncertain = True
            uncertainty_reason = (
                f"Confidence ({top1_conf * 100:.1f}%) is below decision threshold "
                f"({self.uncertainty_threshold * 100:.1f}%). The sample may have ambiguous defect patterns."
            )
        elif len(top_indices) > 1:
            margin = top1_conf - float(probs[top_indices[1]])
            if margin < 0.15:
                is_uncertain = True
                uncertainty_reason = (
                    f"Small margin ({margin * 100:.1f}%) between top class '{top1_class}' "
                    f"and '{self.classes[top_indices[1]]}'. Sample exhibits multi-class visual ambiguity."
                )

        # Grad-CAM generation
        heatmap_2d = None
        overlay_rgb = None
        colored_cam_rgb = None
        cam_target_class = explain_class or top1_class
        cam_error = None

        if generate_cam and self.target_layer is not None:
            try:
                target_idx = (
                    self.classes.index(cam_target_class)
                    if cam_target_class in self.classes
                    else top1_idx
                )
                with GradCAM(self.model, self.target_layer) as gradcam:
                    heatmap_2d = gradcam.generate_heatmap(
                        input_tensor=input_tensor,
                        target_class_idx=target_idx,
                    )
                colored_cam_rgb, overlay_rgb = generate_gradcam_overlay(
                    image=pil_image,
                    heatmap=heatmap_2d,
                    alpha=0.45,
                )
            except Exception as e:
                logger.warning(f"Grad-CAM generation failed: {e}")
                cam_error = f"Grad-CAM could not be generated for this input: {str(e)}"
        elif generate_cam and self.target_layer is None:
            cam_error = "Target convolutional layer not identified for this architecture."

        return PredictionResult(
            predicted_class=top1_class,
            confidence_score=round(top1_conf, 4),
            is_uncertain=is_uncertain,
            uncertainty_reason=uncertainty_reason,
            top_k=top_k_list,
            raw_probabilities=raw_probs,
            inference_latency_ms=round(latency_ms, 2),
            model_name=self.model_name,
            architecture=self.architecture,
            heatmap_2d=heatmap_2d,
            overlay_rgb=overlay_rgb,
            colored_heatmap_rgb=colored_cam_rgb,
            gradcam_target_class=cam_target_class,
            gradcam_error=cam_error,
        )
