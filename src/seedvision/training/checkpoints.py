"""Model checkpoint saving, loading, and metadata management for SeedVision AI."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import logging
import torch
import torch.nn as nn

from seedvision.models.factory import build_model, get_model_info

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages saving, versioning, and reloading model checkpoints with explicit metadata."""

    def __init__(self, checkpoints_dir: Union[str, Path], model_name: str):
        self.checkpoints_dir = Path(checkpoints_dir)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name

        self.best_checkpoint_path = self.checkpoints_dir / f"best_{model_name}.pt"
        self.last_checkpoint_path = self.checkpoints_dir / f"last_{model_name}.pt"

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        val_loss: float,
        val_acc: float,
        is_best: bool,
        classes: List[str],
        image_size: List[int],
        architecture: str,
        normalization: Dict[str, List[float]],
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save model checkpoint and accompanying metadata JSON.

        Args:
            model: PyTorch model.
            optimizer: PyTorch optimizer.
            epoch: Current epoch number.
            val_loss: Current validation loss.
            val_acc: Current validation accuracy.
            is_best: True if this checkpoint achieves the best validation metric.
            classes: List of target class names.
            image_size: Input image resolution [H, W].
            architecture: Architecture string identifier.
            normalization: Mean and std lists.
            extra_metadata: Optional auxiliary details.

        Returns:
            Path to the saved checkpoint file.
        """
        model_info = get_model_info(model)

        checkpoint_payload = {
            "epoch": epoch,
            "architecture": architecture,
            "model_name": self.model_name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "classes": classes,
            "image_size": image_size,
            "normalization": normalization,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "extra_metadata": extra_metadata or {},
        }

        # Metadata dictionary for quick inspection without loading torch tensors
        meta_dict = {
            "epoch": epoch,
            "architecture": architecture,
            "model_name": self.model_name,
            "val_loss": round(float(val_loss), 5),
            "val_acc": round(float(val_acc), 5),
            "is_best": bool(is_best),
            "classes": classes,
            "num_classes": len(classes),
            "image_size": image_size,
            "normalization": normalization,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "parameters": model_info,
            "is_real_world_trained": (extra_metadata or {}).get("is_real_world_trained", False),
            "dataset_provenance": (extra_metadata or {}).get("dataset_provenance", "synthetic_procedural_fixture"),
        }

        # 1. Always save as last checkpoint
        torch.save(checkpoint_payload, self.last_checkpoint_path)
        with open(self.last_checkpoint_path.with_suffix(".pt.meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta_dict, f, indent=2)

        # 2. If best, save to best checkpoint
        if is_best:
            torch.save(checkpoint_payload, self.best_checkpoint_path)
            with open(self.best_checkpoint_path.with_suffix(".pt.meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta_dict, f, indent=2)
            logger.info(f"Saved new best checkpoint to {self.best_checkpoint_path} (Val Acc: {val_acc:.4f})")
            return self.best_checkpoint_path

        return self.last_checkpoint_path


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: Optional[nn.Module] = None,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load a model checkpoint and reconstruct the architecture if model is not provided.

    Args:
        checkpoint_path: Path to .pt checkpoint file.
        model: Optional pre-instantiated model instance.
        device: Device to load tensors onto.

    Returns:
        Tuple of (model, checkpoint_payload)
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    device = torch.device(device)
    payload = torch.load(path, map_location=device, weights_only=False)

    if model is None:
        from seedvision.config import ModelConfig

        arch = payload.get("architecture", "custom_cnn")
        classes = payload.get("classes", [])
        m_cfg = ModelConfig(
            architecture=arch,
            num_classes=len(classes) if classes else 4,
            pretrained=False,
        )
        model = build_model(m_cfg)

    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info(
        f"Loaded checkpoint from {path} (Epoch: {payload.get('epoch')}, Val Acc: {payload.get('val_acc', 0):.4f})"
    )
    return model, payload
