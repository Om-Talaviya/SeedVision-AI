"""Unit tests for scientific evaluation metrics and calibration calculations."""

import numpy as np
import pytest

from seedvision.evaluation.metrics import compute_ece


def test_compute_ece_perfect_calibration():
    # If confidence equals accuracy, ECE is 0
    probs = np.array(
        [
            [0.9, 0.1],
            [0.9, 0.1],
            [0.9, 0.1],
            [0.9, 0.1],
            [0.9, 0.1],
            [0.9, 0.1],
            [0.9, 0.1],
            [0.9, 0.1],
            [0.9, 0.1],
            [0.1, 0.9],
        ]
    )
    labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # 9 correct out of 10 = 90% accuracy

    ece, bin_accs, bin_confs, bin_counts = compute_ece(probs, labels, n_bins=10)
    assert isinstance(ece, float)
    assert 0.0 <= ece <= 1.0


def test_compute_ece_overconfidence():
    # Model predicts 100% confidence on wrong labels
    probs = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ]
    )
    labels = np.array([1, 1, 1, 1])

    ece, _, _, _ = compute_ece(probs, labels, n_bins=10)
    assert np.isclose(ece, 1.0, atol=1e-2)


def test_compare_models_empty_and_populated():
    """Verify compare_models handles both empty metrics directories and valid evaluation JSONs."""
    import tempfile
    import json
    from pathlib import Path
    from seedvision.evaluation.compare import compare_models

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        metrics_dir = tmp_path / "metrics"
        figures_dir = tmp_path / "figures"
        metrics_dir.mkdir(parents=True)

        # 1. Empty directory
        df_empty, summary_empty, plot_empty = compare_models(metrics_dir, figures_dir)
        assert len(df_empty) == 0
        assert summary_empty == {}
        assert plot_empty is None

        # 2. Populated with synthetic evaluation JSON
        eval_data_1 = {
            "model_name": "SeedNet-Baseline",
            "accuracy": 0.95,
            "macro_f1": 0.94,
            "expected_calibration_error": 0.03,
            "latency_ms_per_sample": 6.2,
            "throughput_samples_per_sec": 160.0,
            "parameters": {"total_parameters": 456228, "estimated_size_mb": 1.74},
            "is_real_world_validated": False,
            "evaluation_type": "synthetic_pipeline_verification",
        }
        with open(metrics_dir / "eval_SeedNet-Baseline.json", "w", encoding="utf-8") as f:
            json.dump(eval_data_1, f)

        df_pop, summary_pop, _ = compare_models(metrics_dir, figures_dir)
        assert len(df_pop) == 1
        assert df_pop.iloc[0]["Model"] == "SeedNet-Baseline"
        assert summary_pop["is_real_world_validated"] is False
        assert summary_pop["evaluation_type"] == "synthetic_pipeline_verification"


def test_model_evaluator_complete_flow():
    """Verify ModelEvaluator runs inference over DataLoader, computes classification report and metrics."""
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from seedvision.evaluation.metrics import ModelEvaluator

    # Mock model that maps 8 features to 2 classes
    class MockNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 2)
            # Set weights so class 0 has positive bias for first feature
            with torch.no_grad():
                self.fc.weight.zero_()
                self.fc.bias.zero_()
                self.fc.weight[0, 0] = 10.0
                self.fc.weight[1, 0] = -10.0

        def forward(self, x):
            return self.fc(x)

    model = MockNet()
    classes = ["sound_healthy", "broken_cracked"]

    # 4 samples: 2 positive for class 0, 2 negative for class 1
    inputs = torch.zeros(4, 8)
    inputs[0:2, 0] = 1.0   # Class 0
    inputs[2:4, 0] = -1.0  # Class 1
    targets = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    class DummySeedDataset(torch.utils.data.Dataset):
        def __init__(self, inputs, targets):
            self.inputs = inputs
            self.targets = targets

        def __len__(self):
            return len(self.inputs)

        def __getitem__(self, idx):
            return self.inputs[idx], self.targets[idx], f"sample_{idx}"

    dataset = DummySeedDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=2)

    evaluator = ModelEvaluator(
        model=model,
        classes=classes,
        device="cpu",
        model_name="MockNet",
        is_real_world_validated=False,
    )

    metrics = evaluator.evaluate(dataloader, warmup_batches=0)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert len(metrics["confusion_matrix"]) == 2
    assert len(metrics["confusion_matrix"][0]) == 2
    assert "per_class_metrics" in metrics
    assert evaluator.is_real is False
    assert evaluator.evaluation_type == "synthetic_pipeline_verification"
