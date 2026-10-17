"""Tests for the advisory AI-assisted review utilities (review_assist).

These tests exercise quality screening, optional model suggestions, specimen-level
aggregation, duplicate detection, auto-approval safeguards, report persistence, and
the real-staging integration path. No test writes to the staging manifest and no
test overwrites or deletes the protected real photographs.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from seedvision.data import review_assist as ra
from seedvision.data.real_review import load_staging_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_MANIFEST = REPO_ROOT / "data" / "real_staging" / "staging_manifest.csv"
REAL_INCOMING = REPO_ROOT / "data" / "real_staging" / "incoming"
REAL_ANNOTATIONS = REPO_ROOT / "data" / "real_raw" / "annotations.csv"
MOBILENET_CHECKPOINT = REPO_ROOT / "artifacts" / "checkpoints" / "best_MobileNetV2-Transfer.pt"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _solid(path, size=(160, 160), color=(128, 128, 128)):
    Image.new("RGB", size, color=color).save(path, format="PNG")
    return path


def _noise(path, size=(160, 160), seed=0):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(arr, "RGB").save(path, format="PNG")
    return path


def _garbage(path):
    Path(path).write_bytes(b"this is not a valid image payload")
    return path


def _view(filename, image_path, view_id="v01", image_exists=True):
    return {
        "view_id": view_id,
        "filename": filename,
        "image_path": str(image_path) if image_path is not None else None,
        "image_exists": image_exists,
    }


def _specimen(specimen_id, views, candidate_label="pending_review"):
    return {
        "specimen_id": specimen_id,
        "candidate_label": candidate_label,
        "views": views,
    }


class _FakeResult:
    def __init__(
        self,
        predicted_class,
        confidence,
        *,
        is_uncertain=False,
        uncertainty_reason="",
        model_name="fake_model",
        architecture="fake_arch",
    ):
        self.predicted_class = predicted_class
        self.confidence_score = confidence
        self.is_uncertain = is_uncertain
        self.uncertainty_reason = uncertainty_reason
        self.top_k = []
        self.model_name = model_name
        self.architecture = architecture
        self.scientific_notice = "fake scientific notice"


class _FakePredictor:
    def __init__(
        self,
        mapping=None,
        *,
        default="sound_healthy",
        confidence=0.9,
        uncertain=False,
    ):
        self.mapping = mapping or {}
        self.default = default
        self.confidence = confidence
        self.uncertain = uncertain
        self.model_name = "fake_model"
        self.architecture = "fake_arch"
        self.calls = []

    def predict(self, *, image, top_k_count=ra.DEFAULT_TOP_K, generate_cam=False):
        name = Path(image).name
        self.calls.append((name, generate_cam))
        predicted = self.mapping.get(name, self.default)
        return _FakeResult(
            predicted,
            self.confidence,
            is_uncertain=self.uncertain,
        )


def _assist(**overrides):
    base = dict(
        specimen_id="S1",
        human_candidate_label="pending_review",
        quality_status="ok",
        quality_flags=[],
        views=[{"suggestion": {"is_uncertain": False}}],
        suggestion_available=True,
        model_suggested_label="sound_healthy",
        mean_confidence=0.9,
        views_agree=True,
        conflicting_classes=[],
        recommendation="accept_candidate",
        recommendation_reason="",
        is_auto_approvable=False,
        auto_approval_reason="",
    )
    base.update(overrides)
    return ra.SpecimenAssist(**base)


# --------------------------------------------------------------------------- #
# Quality screening
# --------------------------------------------------------------------------- #
def test_screen_missing_file_is_unusable_unreadable(tmp_path):
    finding = ra.screen_image_quality(tmp_path / "absent.jpg", filename="absent.jpg")
    assert finding.readable is False
    assert finding.status == "unusable"
    assert finding.flags == ["unreadable"]
    assert finding.is_usable is False


def test_screen_none_path_is_unusable():
    finding = ra.screen_image_quality(None)
    assert finding.readable is False
    assert finding.status == "unusable"
    assert finding.flags == ["unreadable"]


def test_screen_corrupted_image_is_unusable(tmp_path):
    path = _garbage(tmp_path / "corrupt.jpg")
    finding = ra.screen_image_quality(path)
    assert finding.readable is False
    assert finding.status == "unusable"
    assert "unreadable" in finding.flags
    assert finding.is_usable is False


def test_screen_too_small_is_unusable(tmp_path):
    path = _solid(tmp_path / "tiny.png", size=(8, 8))
    finding = ra.screen_image_quality(path)
    assert finding.readable is True
    assert "too_small" in finding.flags
    assert finding.status == "unusable"
    assert finding.is_usable is False
    assert finding.metrics["width"] == 8.0


def test_screen_low_resolution_is_warning(tmp_path):
    path = _noise(tmp_path / "small.png", size=(64, 64))
    finding = ra.screen_image_quality(path)
    assert "low_resolution" in finding.flags
    assert finding.status == "warning"
    assert finding.is_usable is True


def test_screen_low_brightness(tmp_path):
    path = _solid(tmp_path / "dark.png", color=(10, 10, 10))
    finding = ra.screen_image_quality(path)
    assert "low_brightness" in finding.flags
    assert finding.metrics["brightness"] < ra.DARK_BRIGHTNESS_THRESHOLD


def test_screen_high_brightness(tmp_path):
    path = _solid(tmp_path / "bright.png", color=(250, 250, 250))
    finding = ra.screen_image_quality(path)
    assert "high_brightness" in finding.flags
    assert finding.metrics["brightness"] > ra.BRIGHT_BRIGHTNESS_THRESHOLD


def test_screen_blur_and_low_contrast_on_flat_image(tmp_path):
    path = _solid(tmp_path / "flat.png", color=(128, 128, 128))
    finding = ra.screen_image_quality(path)
    assert "excessive_blur" in finding.flags
    assert "low_contrast" in finding.flags
    assert finding.metrics["blur_variance"] < ra.BLUR_VARIANCE_THRESHOLD


def test_screen_high_detail_image_is_ok(tmp_path):
    path = _noise(tmp_path / "detail.png")
    finding = ra.screen_image_quality(path)
    assert finding.status == "ok"
    assert finding.flags == []
    assert finding.is_usable is True
    assert finding.metrics["blur_variance"] >= ra.BLUR_VARIANCE_THRESHOLD


def test_screen_near_duplicate_flag(tmp_path):
    path = _noise(tmp_path / "dup.png", seed=1)
    finding = ra.screen_image_quality(path, near_duplicate=True)
    assert "near_duplicate" in finding.flags


def test_quality_finding_to_dict_includes_is_usable(tmp_path):
    path = _noise(tmp_path / "ok.png", seed=2)
    payload = ra.screen_image_quality(path).to_dict()
    assert payload["is_usable"] is True
    assert set(payload) >= {"filename", "status", "flags", "metrics", "reasons", "is_usable"}


# --------------------------------------------------------------------------- #
# Hashes and duplicates
# --------------------------------------------------------------------------- #
def test_compute_average_hash_deterministic(tmp_path):
    path = _noise(tmp_path / "a.png", seed=3)
    with Image.open(path) as im:
        first = ra.compute_average_hash(im)
        second = ra.compute_average_hash(im)
    assert first == second
    assert isinstance(first, str) and first


def test_compute_average_hash_rejects_bad_size(tmp_path):
    path = _noise(tmp_path / "b.png", seed=4)
    with Image.open(path) as im:
        with pytest.raises(ValueError):
            ra.compute_average_hash(im, hash_size=0)


def test_hamming_distance_identical_is_zero():
    assert ra.hamming_distance("ffff", "ffff") == 0


def test_hamming_distance_length_mismatch_raises():
    with pytest.raises(ValueError):
        ra.hamming_distance("ff", "ffff")


def test_hamming_distance_empty_raises():
    with pytest.raises(ValueError):
        ra.hamming_distance("", "ff")


def test_find_near_duplicate_groups():
    hashes = {"a.png": "ffff", "b.png": "ffff", "c.png": "0000"}
    groups = ra.find_near_duplicate_groups(hashes, threshold=5)
    assert groups == [["a.png", "b.png"]]


def test_find_near_duplicate_groups_respects_threshold():
    hashes = {"a.png": "ffff", "b.png": "0000"}
    assert ra.find_near_duplicate_groups(hashes, threshold=1) == []


# --------------------------------------------------------------------------- #
# Checkpoint compatibility
# --------------------------------------------------------------------------- #
def test_checkpoint_not_configured():
    result = ra.inspect_checkpoint_compatibility(None)
    assert result["compatible"] is False
    assert result["reason"] == "checkpoint_not_configured"


def test_checkpoint_not_found(tmp_path):
    result = ra.inspect_checkpoint_compatibility(tmp_path / "missing.pt")
    assert result["exists"] is False
    assert result["compatible"] is False
    assert result["reason"] == "checkpoint_not_found"


def test_checkpoint_unknown_class_mapping(tmp_path):
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"dummy")
    (tmp_path / "model.pt.meta.json").write_text(
        json.dumps({"classes": ["a", "b"], "model_name": "m", "architecture": "custom_cnn"}),
        encoding="utf-8",
    )
    result = ra.inspect_checkpoint_compatibility(ckpt)
    assert result["exists"] is True
    assert result["compatible"] is False
    assert result["reason"] == "unknown_class_mapping"


def test_checkpoint_missing_class_mapping(tmp_path):
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"dummy")
    (tmp_path / "model.pt.meta.json").write_text(
        json.dumps({"classes": [], "model_name": "m"}),
        encoding="utf-8",
    )
    result = ra.inspect_checkpoint_compatibility(ckpt)
    assert result["compatible"] is False
    assert result["reason"] == "checkpoint_missing_class_mapping"


def test_checkpoint_compatible_with_target_classes(tmp_path):
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"dummy")
    (tmp_path / "model.pt.meta.json").write_text(
        json.dumps(
            {
                "classes": list(ra.TARGET_CLASSES),
                "model_name": "SeedNet-Baseline",
                "architecture": "custom_cnn",
            }
        ),
        encoding="utf-8",
    )
    result = ra.inspect_checkpoint_compatibility(ckpt)
    assert result["compatible"] is True
    assert result["reason"] == "compatible"
    assert set(result["classes"]) == set(ra.TARGET_CLASSES)


# --------------------------------------------------------------------------- #
# Specimen aggregation
# --------------------------------------------------------------------------- #
def test_assist_specimen_quality_only_no_model(tmp_path):
    path = _noise(tmp_path / "S1_v01.png", seed=5)
    specimen = _specimen("S1", [_view("S1_v01.png", path)])
    assist = ra.assist_specimen(
        "S1",
        specimen,
        compatibility={"reason": "checkpoint_not_configured", "model_name": "", "architecture": ""},
    )
    assert assist.suggestion_available is False
    assert assist.views_agree is True
    assert assist.quality_status == "ok"
    assert assist.recommendation in ra.REVIEW_RECOMMENDATIONS
    assert assist.is_auto_approvable is False


def test_assist_specimen_unusable_view_recommends_do_not_use(tmp_path):
    path = _garbage(tmp_path / "S1_v01.jpg")
    specimen = _specimen("S1", [_view("S1_v01.jpg", path)])
    assist = ra.assist_specimen(
        "S1", specimen, compatibility={"reason": "model_unavailable"}
    )
    assert assist.quality_status == "unusable"
    assert assist.recommendation == "do_not_use"


def test_assist_specimen_warning_recommends_retake(tmp_path):
    path = _solid(tmp_path / "S1_v01.png", color=(128, 128, 128))
    specimen = _specimen("S1", [_view("S1_v01.png", path)])
    assist = ra.assist_specimen(
        "S1", specimen, compatibility={"reason": "model_unavailable"}
    )
    assert assist.quality_status == "warning"
    assert assist.recommendation == "retake_image"


def test_assist_specimen_agreeing_views_accept_candidate(tmp_path):
    p1 = _noise(tmp_path / "S1_v01.png", seed=6)
    p2 = _noise(tmp_path / "S1_v02.png", seed=7)
    predictor = _FakePredictor(default="sound_healthy", confidence=0.95)
    specimen = _specimen("S1", [_view("S1_v01.png", p1), _view("S1_v02.png", p2, "v02")])
    assist = ra.assist_specimen("S1", specimen, predictor=predictor)
    assert assist.suggestion_available is True
    assert assist.views_agree is True
    assert assist.model_suggested_label == "sound_healthy"
    assert assist.recommendation == "accept_candidate"
    # Model suggestion never authorizes approval on its own.
    assert assist.is_auto_approvable is False


def test_assist_specimen_conflicting_views(tmp_path):
    p1 = _noise(tmp_path / "S1_v01.png", seed=8)
    p2 = _noise(tmp_path / "S1_v02.png", seed=9)
    predictor = _FakePredictor(
        mapping={"S1_v01.png": "sound_healthy", "S1_v02.png": "broken_cracked"}
    )
    specimen = _specimen("S1", [_view("S1_v01.png", p1), _view("S1_v02.png", p2, "v02")])
    assist = ra.assist_specimen("S1", specimen, predictor=predictor)
    assert assist.views_agree is False
    assert set(assist.conflicting_classes) == {"sound_healthy", "broken_cracked"}
    assert assist.recommendation == "inspect_closely"


def test_assist_specimen_uncertain_suggestion_not_accept(tmp_path):
    path = _noise(tmp_path / "S1_v01.png", seed=10)
    predictor = _FakePredictor(default="sound_healthy", uncertain=True)
    specimen = _specimen("S1", [_view("S1_v01.png", path)])
    assist = ra.assist_specimen("S1", specimen, predictor=predictor)
    assert assist.recommendation == "inspect_closely"


def test_assist_specimen_preserves_human_label(tmp_path):
    path = _noise(tmp_path / "S1_v01.png", seed=11)
    predictor = _FakePredictor(default="broken_cracked")
    specimen = _specimen("S1", [_view("S1_v01.png", path)], candidate_label="sound_healthy")
    assist = ra.assist_specimen("S1", specimen, predictor=predictor)
    # The human label is retained untouched and kept separate from the suggestion.
    assert assist.human_candidate_label == "sound_healthy"
    payload = assist.to_dict()
    assert payload["human_candidate_label"] == "sound_healthy"
    assert payload["model_suggested_label"] == "broken_cracked"


def test_assist_specimen_near_duplicate_filenames(tmp_path):
    path = _noise(tmp_path / "S1_v01.png", seed=12)
    specimen = _specimen("S1", [_view("S1_v01.png", path)])
    assist = ra.assist_specimen(
        "S1",
        specimen,
        compatibility={"reason": "model_unavailable"},
        near_duplicate_filenames=["S1_v01.png"],
    )
    assert "near_duplicate" in assist.quality_flags


# --------------------------------------------------------------------------- #
# Auto-approval safeguards
# --------------------------------------------------------------------------- #
def test_auto_approval_disabled_by_default():
    result = ra.evaluate_auto_approval(_assist())
    assert result["allowed"] is False
    assert result["reason"] == "auto_approval_module_disabled"


def test_auto_approval_module_switch_gates(monkeypatch):
    monkeypatch.setattr(ra, "AUTO_APPROVAL_ENABLED", True)
    # No config provided.
    result = ra.evaluate_auto_approval(_assist())
    assert result["allowed"] is False
    assert result["reason"] == "auto_approval_not_enabled_in_config"


def test_auto_approval_requires_documented_validation(monkeypatch):
    monkeypatch.setattr(ra, "AUTO_APPROVAL_ENABLED", True)
    result = ra.evaluate_auto_approval(
        _assist(), config={"auto_approval": {"enabled": True, "validation_documented": False}}
    )
    assert result["allowed"] is False
    assert result["reason"] == "documented_validation_missing"


@pytest.mark.parametrize(
    "overrides,expected_reason",
    [
        ({"quality_status": "warning"}, "quality_not_acceptable"),
        ({"quality_status": "unusable"}, "quality_not_acceptable"),
        ({"human_candidate_label": "sound_healthy"}, "specimen_already_reviewed"),
        ({"suggestion_available": False, "model_suggested_label": None}, "no_model_suggestion"),
        ({"views_agree": False, "conflicting_classes": ["a", "b"]}, "conflicting_view_suggestions"),
        ({"views": [{"suggestion": {"is_uncertain": True}}]}, "uncertain_suggestion"),
    ],
)
def test_auto_approval_denies_when_safeguard_fails(monkeypatch, overrides, expected_reason):
    monkeypatch.setattr(ra, "AUTO_APPROVAL_ENABLED", True)
    config = {"auto_approval": {"enabled": True, "validation_documented": True}}
    result = ra.evaluate_auto_approval(_assist(**overrides), config=config)
    assert result["allowed"] is False
    assert result["reason"] == expected_reason


def test_auto_approval_allowed_when_every_safeguard_holds(monkeypatch):
    monkeypatch.setattr(ra, "AUTO_APPROVAL_ENABLED", True)
    config = {"auto_approval": {"enabled": True, "validation_documented": True}}
    result = ra.evaluate_auto_approval(_assist(), config=config)
    assert result["allowed"] is True
    assert result["reason"] == "all_safeguards_satisfied"


# --------------------------------------------------------------------------- #
# Report building and persistence
# --------------------------------------------------------------------------- #
def test_build_assist_report_shape(tmp_path):
    path = _noise(tmp_path / "S1_v01.png", seed=13)
    predictor = _FakePredictor(default="sound_healthy")
    assist = ra.assist_specimen(
        "S1", _specimen("S1", [_view("S1_v01.png", path)]), predictor=predictor
    )
    compatibility = {"compatible": True, "reason": "compatible", "classes": list(ra.TARGET_CLASSES)}
    report = ra.build_assist_report([assist], compatibility=compatibility)
    assert report["model"]["compatible"] is True
    assert report["auto_approval_enabled"] is False
    assert "S1" in report["specimens"]
    summary = report["summary"]
    assert summary["total_specimens"] == 1
    assert summary["specimens_with_suggestions"] == 1
    assert summary["auto_approvable_specimens"] == 0


def test_write_and_load_assist_report_roundtrip(tmp_path):
    report = {"generated_at": "now", "specimens": {}, "model": {}}
    out = tmp_path / "nested" / "assist.json"
    written = ra.write_assist_report(report, out)
    assert written == out and out.is_file()
    assert ra.load_assist_report(out) == report


def test_write_assist_report_refuses_overwrite(tmp_path):
    out = tmp_path / "assist.json"
    ra.write_assist_report({"a": 1}, out)
    with pytest.raises(FileExistsError):
        ra.write_assist_report({"a": 2}, out)
    ra.write_assist_report({"a": 3}, out, overwrite=True)
    assert ra.load_assist_report(out) == {"a": 3}


def test_load_assist_report_missing_returns_none(tmp_path):
    assert ra.load_assist_report(tmp_path / "nope.json") is None


# --------------------------------------------------------------------------- #
# Whole-staging assistance (no checkpoint -> never imports torch)
# --------------------------------------------------------------------------- #
def test_assist_staging_without_checkpoint(tmp_path):
    p1 = _noise(tmp_path / "S1_v01.png", seed=14)
    duplicate = tmp_path / "S2_v01.png"
    duplicate.write_bytes(p1.read_bytes())
    staging = {
        "specimens": {
            "S1": _specimen("S1", [_view("S1_v01.png", p1)]),
            "S2": _specimen("S2", [_view("S2_v01.png", duplicate)]),
        }
    }
    report = ra.assist_staging(staging)
    assert report["model"]["compatible"] is False
    assert report["model"]["reason"] == "checkpoint_not_configured"
    assert report["summary"]["total_specimens"] == 2
    assert report["summary"]["specimens_with_suggestions"] == 0
    # Identical image bytes are grouped as near-duplicates.
    assert report["near_duplicate_groups"] == [["S1_v01.png", "S2_v01.png"]]
    s1 = report["specimens"]["S1"]
    assert "near_duplicate" in s1["quality_flags"]


def test_assist_staging_never_writes_manifest(tmp_path):
    p1 = _noise(tmp_path / "S1_v01.png", seed=15)
    staging = {"specimens": {"S1": _specimen("S1", [_view("S1_v01.png", p1)])}}
    before = p1.read_bytes()
    ra.assist_staging(staging)
    assert p1.read_bytes() == before


# --------------------------------------------------------------------------- #
# Real staging integration (read-only)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not REAL_MANIFEST.is_file(), reason="real staging manifest not present")
def test_real_staging_stays_pending_and_read_only():
    staging = load_staging_manifest(REAL_MANIFEST, REAL_INCOMING)
    assert staging["total_specimens"] == 2
    assert staging["total_views"] == 6
    assert set(staging["status_counts"]) == {"pending"}

    report = ra.assist_staging(staging)
    assert report["summary"]["total_specimens"] == 2
    assert report["summary"]["auto_approvable_specimens"] == 0
    for specimen in report["specimens"].values():
        assert specimen["human_candidate_label"] == "pending_review"
        assert specimen["is_auto_approvable"] is False
        assert specimen["suggestion_available"] is False
    # Assistance never creates the downstream annotation file.
    assert not REAL_ANNOTATIONS.exists()


@pytest.mark.skipif(
    not MOBILENET_CHECKPOINT.is_file(), reason="compatible checkpoint not present"
)
def test_assist_real_specimen_with_real_checkpoint():
    staging = load_staging_manifest(REAL_MANIFEST, REAL_INCOMING)
    specimen_id, specimen = next(iter(staging["specimens"].items()))
    assist = ra.assist_specimen(
        specimen_id,
        specimen,
        checkpoint_path=MOBILENET_CHECKPOINT,
        device="cpu",
    )
    assert assist.suggestion_available is True
    assert assist.model_suggested_label in ra.TARGET_CLASSES
    # Advisory only: still pending, still not auto-approvable.
    assert assist.human_candidate_label == "pending_review"
    assert assist.is_auto_approvable is False
