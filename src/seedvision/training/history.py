"""Training history tracking and persistence for SeedVision AI."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Union
import json
import pandas as pd


@dataclass
class TrainingHistory:
    """Stores epoch-by-epoch training and validation metrics."""

    epochs: List[int] = field(default_factory=list)
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    epoch_times_sec: List[float] = field(default_factory=list)

    def record_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_acc: float,
        val_acc: float,
        lr: float,
        duration_sec: float,
    ) -> None:
        """Record metrics for a single completed epoch."""
        self.epochs.append(epoch)
        self.train_loss.append(round(train_loss, 5))
        self.val_loss.append(round(val_loss, 5))
        self.train_acc.append(round(train_acc, 5))
        self.val_acc.append(round(val_acc, 5))
        self.learning_rates.append(round(lr, 7))
        self.epoch_times_sec.append(round(duration_sec, 2))

    def to_dataframe(self) -> pd.DataFrame:
        """Convert history into a pandas DataFrame."""
        return pd.DataFrame(
            {
                "epoch": self.epochs,
                "train_loss": self.train_loss,
                "val_loss": self.val_loss,
                "train_acc": self.train_acc,
                "val_acc": self.val_acc,
                "learning_rate": self.learning_rates,
                "duration_sec": self.epoch_times_sec,
            }
        )

    def save(self, filepath: Union[str, Path]) -> None:
        """Save history as both JSON and CSV for interoperability."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        json_path = path.with_suffix(".json")
        csv_path = path.with_suffix(".csv")

        data = {
            "epochs": self.epochs,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "train_acc": self.train_acc,
            "val_acc": self.val_acc,
            "learning_rates": self.learning_rates,
            "epoch_times_sec": self.epoch_times_sec,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        df = self.to_dataframe()
        df.to_csv(csv_path, index=False)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "TrainingHistory":
        """Load history from JSON file."""
        json_path = Path(filepath).with_suffix(".json")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            epochs=data.get("epochs", []),
            train_loss=data.get("train_loss", []),
            val_loss=data.get("val_loss", []),
            train_acc=data.get("train_acc", []),
            val_acc=data.get("val_acc", []),
            learning_rates=data.get("learning_rates", []),
            epoch_times_sec=data.get("epoch_times_sec", []),
        )
