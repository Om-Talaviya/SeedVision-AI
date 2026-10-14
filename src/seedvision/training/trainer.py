"""Training loop and model fitting coordinator for SeedVision AI."""

from pathlib import Path
from typing import Optional, Tuple
import logging
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from seedvision.config import SeedVisionConfig
from seedvision.models.factory import build_model
from seedvision.training.checkpoints import CheckpointManager
from seedvision.training.history import TrainingHistory

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds across random, numpy, and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class SeedTrainer:
    """Orchestrates model training, validation, learning rate scheduling, and early stopping."""

    def __init__(
        self,
        config: SeedVisionConfig,
        model: Optional[nn.Module] = None,
    ):
        self.config = config
        set_seed(config.training.seed)

        self.device = config.get_device()
        self.model = model or build_model(config.model)
        self.model.to(self.device)

        # Loss function with mild label smoothing to improve generalization
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

        # Optimizer selection
        lr = config.training.learning_rate
        weight_decay = config.training.weight_decay
        opt_name = config.training.optimizer.lower()

        if opt_name == "adamw":
            self.optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=lr,
                weight_decay=weight_decay,
            )
        else:
            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=lr,
                weight_decay=weight_decay,
            )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(config.training.epochs, 1),
            eta_min=config.training.min_lr,
        )

        self.history = TrainingHistory()
        self.checkpoint_manager = CheckpointManager(
            checkpoints_dir=config.paths.checkpoints_dir,
            model_name=config.model.name,
        )

    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Execute one training epoch.

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels, _ in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)

        epoch_loss = running_loss / max(total, 1)
        epoch_acc = correct / max(total, 1)
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Evaluate model on validation dataloader.

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels, _ in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)

        val_loss = running_loss / max(total, 1)
        val_acc = correct / max(total, 1)
        return val_loss, val_acc

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Tuple[Path, TrainingHistory]:
        """Train model with early stopping and best checkpoint persistence.

        Returns:
            Tuple of (best_checkpoint_path, training_history)
        """
        epochs = self.config.training.epochs
        patience = self.config.training.early_stopping_patience

        best_val_loss = float("inf")
        best_val_acc = 0.0
        epochs_without_improvement = 0
        best_checkpoint_path = self.checkpoint_manager.best_checkpoint_path

        logger.info(
            f"Starting training '{self.config.model.name}' on {self.device} for {epochs} epochs..."
        )

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            current_lr = self.optimizer.param_groups[0]["lr"]

            self.scheduler.step()
            duration = time.time() - start_time

            self.history.record_epoch(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_acc=train_acc,
                val_acc=val_acc,
                lr=current_lr,
                duration_sec=duration,
            )

            # Check improvement based on validation accuracy (or val_loss)
            is_best = bool(val_acc > best_val_acc or (np.isclose(val_acc, best_val_acc) and val_loss < best_val_loss))

            if is_best:
                best_val_acc = val_acc
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Save checkpoint
            saved_path = self.checkpoint_manager.save(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                val_loss=val_loss,
                val_acc=val_acc,
                is_best=is_best,
                classes=self.config.dataset.classes,
                image_size=self.config.dataset.image_size,
                architecture=self.config.model.architecture,
                normalization={
                    "mean": self.config.dataset.normalization.mean,
                    "std": self.config.dataset.normalization.std,
                },
                extra_metadata={"seed_type": self.config.dataset.seed_type},
            )
            if is_best:
                best_checkpoint_path = saved_path

            logger.info(
                f"Epoch [{epoch:02d}/{epochs:02d}] "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                f"LR: {current_lr:.6f} ({duration:.1f}s)"
                f"{' *BEST*' if is_best else ''}"
            )

            # Early stopping check
            if epochs_without_improvement >= patience:
                logger.info(
                    f"Early stopping triggered after {epoch} epochs (no improvement for {patience} epochs)."
                )
                break

        # Save training history files
        metrics_dir = Path(self.config.paths.metrics_dir)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        history_save_path = metrics_dir / f"history_{self.config.model.name}"
        self.history.save(history_save_path)

        return best_checkpoint_path, self.history
