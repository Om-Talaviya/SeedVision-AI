"""Unit tests for inference validation and prediction engine."""

import io
from pathlib import Path
import tempfile
from PIL import Image
import pytest
import torch

from seedvision.config import SeedVisionConfig
from seedvision.inference.predictor import PredictionResult, SeedPredictor
from seedvision.inference.validation import InferenceInputValidator
from seedvision.models.cnn import CustomSeedCNN
from seedvision.training.checkpoints import CheckpointManager


@pytest.fixture
def dummy_checkpoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        classes = ["sound_healthy", "discolored_spotted", "broken_cracked", "shrivelled_damaged"]
        model = CustomSeedCNN(num_classes=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        mgr = CheckpointManager(checkpoints_dir=tmp_path, model_name="TestModel")
        ckpt_path = mgr.save(
            model=model,
            optimizer=optimizer,
            epoch=1,
            val_loss=0.5,
            val_acc=0.85,
            is_best=True,
            classes=classes,
            image_size=[64, 64],
            architecture="custom_cnn",
            normalization={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        )
        yield ckpt_path


def test_inference_input_validator():
    validator = InferenceInputValidator(min_dim=32, max_dim=512)

    # Valid image
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=(100, 150, 200)).save(buf, format="JPEG")
    res = validator.validate_upload(buf)
    assert res.is_valid is True
    assert res.width == 64
    assert res.height == 64

    # Corrupt upload
    corrupt_res = validator.validate_upload(b"GARBAGE_BYTES")
    assert corrupt_res.is_valid is False
    assert "Corrupted" in (corrupt_res.error_message or "")


def test_seed_predictor(dummy_checkpoint):
    predictor = SeedPredictor(checkpoint_path=dummy_checkpoint, device="cpu")

    dummy_image = Image.new("RGB", (64, 64), color=(150, 120, 80))
    res = predictor.predict(dummy_image, top_k_count=4, generate_cam=True)

    assert isinstance(res, PredictionResult)
    assert res.predicted_class in predictor.classes
    assert 0.0 <= res.confidence_score <= 1.0
    assert len(res.top_k) == 4
    assert res.overlay_rgb is not None
    assert res.inference_latency_ms > 0
