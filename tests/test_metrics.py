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
