"""Comprehensive scientific evaluation metrics for SeedVision AI."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import logging
import time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from seedvision.models.factory import get_model_info

logger = logging.getLogger(__name__)


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate Expected Calibration Error (ECE) and reliability bin statistics.

    Args:
        probs: Array of predicted class probabilities [N, num_classes].
        labels: Array of true class integer labels [N].
        n_bins: Number of confidence bins between 0 and 1.

    Returns:
        Tuple of (ece_value, bin_accuracies, bin_confidences, bin_counts)
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    bin_accs = []
    bin_confs = []
    bin_counts = []

    n_samples = len(labels)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_count = np.sum(in_bin)
        bin_counts.append(int(bin_count))

        if bin_count > 0:
            avg_acc = np.mean(accuracies[in_bin])
            avg_conf = np.mean(confidences[in_bin])
            bin_accs.append(float(avg_acc))
            bin_confs.append(float(avg_conf))
            ece += (bin_count / n_samples) * np.abs(avg_acc - avg_conf)
        else:
            bin_accs.append(0.0)
            bin_confs.append(float((bin_lower + bin_upper) / 2.0))

    return float(ece), np.array(bin_accs), np.array(bin_confs), np.array(bin_counts)


class ModelEvaluator:
    """Evaluates trained models on held-out test datasets with scientific metrics."""

    def __init__(
        self,
        model: nn.Module,
        classes: List[str],
        device: Union[str, torch.device] = "cpu",
        model_name: str = "Model",
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.classes = classes
        self.model_name = model_name

    @torch.no_grad()
    def evaluate(
        self,
        test_loader: DataLoader,
        warmup_batches: int = 2,
    ) -> Dict[str, Any]:
        """Perform evaluation and calculate complete metric suite.

        Returns:
            Dictionary containing metrics, confusion matrix, per-class stats, latency, and calibration.
        """
        all_preds = []
        all_labels = []
        all_probs = []
        latencies_ms = []

        total_batches = len(test_loader)

        for batch_idx, (images, labels, _) in enumerate(test_loader):
            images = images.to(self.device)

            # Measure inference latency accurately
            start_t = time.perf_counter()
            outputs = self.model(images)
            duration_ms = (time.perf_counter() - start_t) * 1000.0

            if batch_idx >= warmup_batches:
                # Per-sample latency in milliseconds
                per_sample_latency = duration_ms / images.size(0)
                latencies_ms.append(per_sample_latency)

            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_probs.append(probs)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_prob = np.concatenate(all_probs, axis=0)

        # Standard classification metrics
        acc = accuracy_score(y_true, y_pred)
        macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        # Per-class metrics
        prec_per_class, rec_per_class, f1_per_class, support = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(len(self.classes))), zero_division=0
        )

        per_class_dict = {}
        for idx, cls_name in enumerate(self.classes):
            per_class_dict[cls_name] = {
                "precision": round(float(prec_per_class[idx]), 4),
                "recall": round(float(rec_per_class[idx]), 4),
                "f1_score": round(float(f1_per_class[idx]), 4),
                "support": int(support[idx]),
            }

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(self.classes))))
        cm_normalized = cm.astype("float") / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1)

        # Expected Calibration Error (ECE)
        ece, bin_accs, bin_confs, bin_counts = compute_ece(y_prob, y_true, n_bins=10)

        # Latency statistics
        avg_latency = float(np.mean(latencies_ms)) if latencies_ms else 0.0
        p95_latency = float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0
        throughput = (1000.0 / avg_latency) if avg_latency > 0 else 0.0

        model_params = get_model_info(self.model)

        evaluation_results = {
            "model_name": self.model_name,
            "num_test_samples": len(y_true),
            "accuracy": round(float(acc), 4),
            "macro_precision": round(float(macro_prec), 4),
            "macro_recall": round(float(macro_rec), 4),
            "macro_f1": round(float(macro_f1), 4),
            "expected_calibration_error": round(float(ece), 4),
            "latency_ms_per_sample": round(avg_latency, 2),
            "p95_latency_ms": round(p95_latency, 2),
            "throughput_samples_per_sec": round(throughput, 1),
            "parameters": model_params,
            "classes": self.classes,
            "per_class_metrics": per_class_dict,
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_normalized": np.round(cm_normalized, 4).tolist(),
            "calibration_bins": {
                "accuracies": np.round(bin_accs, 4).tolist(),
                "confidences": np.round(bin_confs, 4).tolist(),
                "counts": bin_counts.tolist(),
            },
        }

        return evaluation_results

    def save_results(
        self,
        results: Dict[str, Any],
        output_dir: Union[str, Path],
    ) -> Path:
        """Save evaluation results to JSON and per-class CSV."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        json_path = out / f"eval_{self.model_name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        # Also write per-class CSV
        per_class_rows = []
        for cls_name, vals in results.get("per_class_metrics", {}).items():
            row = {"class_name": cls_name, **vals}
            per_class_rows.append(row)

        df_per_class = pd.DataFrame(per_class_rows)
        csv_path = out / f"per_class_{self.model_name}.csv"
        df_per_class.to_csv(csv_path, index=False)

        logger.info(f"Saved evaluation results to {json_path} and {csv_path}")
        return json_path
