"""Matplotlib-based scientific publication visualization for SeedVision AI."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Use non-interactive Agg backend for safe server/CLI rendering
matplotlib.use("Agg")

# Scientific theme styling
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.titlesize": 14,
        "figure.dpi": 200,
    }
)


def plot_confusion_matrix(
    cm: Union[np.ndarray, List[List[int]]],
    classes: List[str],
    output_path: Union[str, Path],
    title: str = "Confusion Matrix",
    normalize: bool = False,
) -> Path:
    """Render and save a high-resolution confusion matrix heatmap."""
    cm_arr = np.array(cm)
    if normalize:
        cm_arr = cm_arr.astype("float") / np.maximum(cm_arr.sum(axis=1)[:, np.newaxis], 1)
        fmt = ".2f"
    else:
        fmt = "d"

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_arr, interpolation="nearest", cmap="Greens")
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Normalized Rate" if normalize else "Count", rotation=-90, va="bottom")

    ax.set(
        xticks=np.arange(cm_arr.shape[1]),
        yticks=np.arange(cm_arr.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="Ground Truth Class",
        xlabel="Predicted Class",
    )

    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    thresh = cm_arr.max() / 2.0
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            val_str = f"{cm_arr[i, j]:{fmt}}"
            color = "white" if cm_arr[i, j] > thresh else "#1e3a1e"
            ax.text(j, i, val_str, ha="center", va="center", color=color, fontweight="bold")

    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_training_curves(
    history_data: Union[Dict[str, List[float]], Any],
    output_path: Union[str, Path],
    title: str = "Training & Validation Convergence",
) -> Path:
    """Render dual-axis loss and accuracy curves across training epochs."""
    if hasattr(history_data, "epochs"):
        epochs = history_data.epochs
        train_loss = history_data.train_loss
        val_loss = history_data.val_loss
        train_acc = history_data.train_acc
        val_acc = history_data.val_acc
    else:
        epochs = history_data.get("epochs", list(range(1, len(history_data.get("train_loss", [])) + 1)))
        train_loss = history_data.get("train_loss", [])
        val_loss = history_data.get("val_loss", [])
        train_acc = history_data.get("train_acc", [])
        val_acc = history_data.get("val_acc", [])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Loss plot
    ax1.plot(epochs, train_loss, label="Train Loss", color="#2e7d32", linewidth=2)
    ax1.plot(epochs, val_loss, label="Val Loss", color="#c62828", linewidth=2, linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Loss Trajectory")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right")

    # Accuracy plot
    ax2.plot(epochs, [a * 100 for a in train_acc], label="Train Acc (%)", color="#1565c0", linewidth=2)
    ax2.plot(
        epochs,
        [a * 100 for a in val_acc],
        label="Val Acc (%)",
        color="#f57f17",
        linewidth=2,
        linestyle="--",
    )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy Trajectory")
    ax2.set_ylim([0, 105])
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right")

    fig.suptitle(title, y=1.02)
    fig.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_per_class_metrics(
    per_class_metrics: Dict[str, Dict[str, float]],
    output_path: Union[str, Path],
    title: str = "Per-Class Precision, Recall & F1-Score",
) -> Path:
    """Render grouped bar chart showing per-class precision, recall, and F1."""
    classes = list(per_class_metrics.keys())
    precisions = [per_class_metrics[c]["precision"] * 100 for c in classes]
    recalls = [per_class_metrics[c]["recall"] * 100 for c in classes]
    f1_scores = [per_class_metrics[c]["f1_score"] * 100 for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width, precisions, width, label="Precision", color="#388e3c")
    ax.bar(x, recalls, width, label="Recall", color="#1976d2")
    ax.bar(x + width, f1_scores, width, label="F1-Score", color="#fbc02d")

    ax.set_ylabel("Score (%)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=20, ha="right")
    ax.set_ylim([0, 110])
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="lower right")

    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_calibration_curve(
    calibration_bins: Dict[str, List[float]],
    ece: float,
    output_path: Union[str, Path],
    title: str = "Confidence Calibration Reliability Diagram",
) -> Path:
    """Render reliability diagram comparing confidence against empirical accuracy."""
    confs = np.array(calibration_bins["confidences"])
    accs = np.array(calibration_bins["accuracies"])

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    ax.plot(confs, accs, marker="o", color="#2e7d32", linewidth=2, label=f"Model (ECE={ece:.4f})")

    ax.set_xlabel("Predicted Confidence")
    ax.set_ylabel("Observed Empirical Accuracy")
    ax.set_title(title)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left")

    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_model_comparison(
    models_data: List[Dict[str, Any]],
    output_path: Union[str, Path],
    title: str = "Deep Learning Architecture Comparison",
) -> Path:
    """Render comparison chart between baseline CNN and transfer learning architectures."""
    model_names = [m["model_name"] for m in models_data]
    accuracies = [m["accuracy"] * 100 for m in models_data]
    macro_f1s = [m["macro_f1"] * 100 for m in models_data]
    latencies = [m["latency_ms_per_sample"] for m in models_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    x = np.arange(len(model_names))
    w = 0.35

    # Acc & F1
    ax1.bar(x - w / 2, accuracies, w, label="Accuracy (%)", color="#2e7d32")
    ax1.bar(x + w / 2, macro_f1s, w, label="Macro F1 (%)", color="#1976d2")
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names)
    ax1.set_ylabel("Percentage (%)")
    ax1.set_title("Classification Performance")
    ax1.set_ylim([0, 105])
    ax1.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right")

    # Latency
    bars = ax2.bar(model_names, latencies, color="#e65100", width=0.45)
    ax2.set_ylabel("Inference Latency (ms/sample)")
    ax2.set_title("CPU Latency Benchmark")
    ax2.grid(True, axis="y", linestyle=":", alpha=0.6)

    for bar in bars:
        h = bar.get_height()
        ax2.annotate(
            f"{h:.1f}ms",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    fig.suptitle(title, y=1.02)
    fig.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out
