"""Evaluation metrics, visualization, and model comparison for SeedVision AI."""

from seedvision.evaluation.compare import compare_models
from seedvision.evaluation.metrics import ModelEvaluator, compute_ece
from seedvision.evaluation.plots import (
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_per_class_metrics,
    plot_training_curves,
)

from seedvision.evaluation.provenance_verifier import verify_evaluation_provenance

__all__ = [
    "ModelEvaluator",
    "compute_ece",
    "plot_confusion_matrix",
    "plot_training_curves",
    "plot_per_class_metrics",
    "plot_calibration_curve",
    "plot_model_comparison",
    "compare_models",
    "verify_evaluation_provenance",
]
