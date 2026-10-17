"""Advisory AI-assisted review utilities for staged real-world seed photographs.

This module provides decision support only. It screens image quality, produces
optional model class suggestions, and aggregates findings at the physical
specimen level so that multiple views of one physical seed are never treated as
independent specimens.

Safety contract:
    - Image quality findings are advisory and are NOT evidence of a seed class.
    - Model class suggestions never overwrite human labels or staging decisions.
    - Automatic approval is disabled by default and requires an explicit
      configuration flag plus documented validation evidence to be enabled.
    - No function in this module writes to the staging manifest.
"""

import json
import os
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

TARGET_CLASSES: Tuple[str, ...] = (
    "sound_healthy",
    "discolored_spotted",
    "broken_cracked",
    "shrivelled_damaged",
)

QUALITY_STATUSES: Tuple[str, ...] = ("ok", "warning", "unusable")

QUALITY_FLAG_DESCRIPTIONS: Dict[str, str] = {
    "unreadable": "Image file could not be located or decoded.",
    "too_small": "Image is below the absolute minimum usable dimension.",
    "low_resolution": "Image is below the recommended inspection resolution.",
    "excessive_blur": "Image appears out of focus (low edge detail).",
    "low_brightness": "Image appears underexposed (too dark).",
    "high_brightness": "Image appears overexposed (too bright).",
    "low_contrast": "Image appears washed out (low tonal variation).",
    "near_duplicate": "Image is visually near-identical to another staged view.",
}

REVIEW_RECOMMENDATIONS: Tuple[str, ...] = (
    "accept_candidate",
    "inspect_closely",
    "retake_image",
    "do_not_use",
)

RECOMMENDED_MIN_DIMENSION = 128
ABSOLUTE_MIN_DIMENSION = 16
BLUR_VARIANCE_THRESHOLD = 50.0
DARK_BRIGHTNESS_THRESHOLD = 40.0
BRIGHT_BRIGHTNESS_THRESHOLD = 220.0
LOW_CONTRAST_STD_THRESHOLD = 20.0
NEAR_DUPLICATE_HAMMING_THRESHOLD = 5
HASH_SIZE = 8
DEFAULT_UNCERTAINTY_THRESHOLD = 0.50
DEFAULT_TOP_K = 4

AUTO_APPROVAL_ENABLED = False
AUTO_APPROVAL_CONFIG_KEY = "auto_approval"
AUTO_APPROVAL_REQUIREMENTS: Tuple[str, ...] = (
    "Auto-approval is disabled by default.",
    "Enabling it requires an explicit configuration flag AND documented validation evidence.",
    "Model confidence alone must never authorize approval.",
    "All views must agree, image quality must be acceptable, and no suggestion may be uncertain.",
    "Human review stays authoritative for every real-world specimen.",
)

ADVISORY_NOTICE = (
    "AI-assisted findings are advisory decision support only. Image quality is not "
    "evidence of a seed's class, and model suggestions never overwrite human labels."
)

MODEL_SUGGESTION_NOTICE = (
    "Model class suggestions derive from checkpoints trained on procedural synthetic "
    "fixtures. Reported training accuracy does not represent real-world grading "
    "accuracy, and suggestions must be confirmed by a human reviewer."
)

DEFAULT_ASSIST_REPORT_PATH = Path("data/real_staging/staging_review_assist.json")

_RESAMPLE_LANCZOS = getattr(Image, "Resampling", Image).LANCZOS


@dataclass
class QualityFinding:
    """Advisory image-quality screening result for a single photographic view."""

    filename: str
    image_path: Optional[str]
    readable: bool
    status: str
    flags: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)

    @property
    def is_usable(self) -> bool:
        return self.readable and self.status != "unusable"

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["is_usable"] = self.is_usable
        return payload


@dataclass
class ClassSuggestion:
    """Optional model class suggestion for a single photographic view."""

    available: bool
    image_path: Optional[str] = None
    predicted_class: Optional[str] = None
    confidence: Optional[float] = None
    is_uncertain: bool = False
    uncertainty_reason: str = ""
    top_k: List[Dict[str, Any]] = field(default_factory=list)
    model_name: str = ""
    architecture: str = ""
    reason: str = ""
    error: str = ""
    scientific_notice: str = MODEL_SUGGESTION_NOTICE

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SpecimenAssist:
    """Advisory aggregation of all views belonging to one physical specimen."""

    specimen_id: str
    human_candidate_label: str
    quality_status: str
    quality_flags: List[str]
    views: List[Dict[str, Any]]
    suggestion_available: bool
    model_suggested_label: Optional[str]
    mean_confidence: Optional[float]
    views_agree: bool
    conflicting_classes: List[str]
    recommendation: str
    recommendation_reason: str
    is_auto_approvable: bool
    auto_approval_reason: str
    auto_approval_requirements: List[str] = field(default_factory=lambda: list(AUTO_APPROVAL_REQUIREMENTS))
    advisory_notice: str = ADVISORY_NOTICE

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _grayscale_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.float32)


def compute_blur_score(image: Image.Image) -> float:
    """Return the variance of the Laplacian (higher means sharper)."""
    gray = _grayscale_array(image)
    if gray.ndim != 2 or gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    padded = np.pad(gray, 1, mode="edge")
    laplacian = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        - 4.0 * padded[1:-1, 1:-1]
    )
    return float(laplacian.var())


def compute_brightness(image: Image.Image) -> float:
    """Return the mean grayscale intensity (0-255)."""
    return float(_grayscale_array(image).mean())


def compute_contrast(image: Image.Image) -> float:
    """Return the grayscale intensity standard deviation (0-255-ish)."""
    return float(_grayscale_array(image).std())


def compute_average_hash(image: Image.Image, hash_size: int = HASH_SIZE) -> str:
    """Return an average-hash (aHash) hexadecimal string for near-duplicate checks."""
    if hash_size < 1:
        raise ValueError("hash_size must be >= 1")
    small = image.convert("L").resize((hash_size, hash_size), _RESAMPLE_LANCZOS)
    arr = np.asarray(small, dtype=np.float32)
    bits = (arr > arr.mean()).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bool(bit))
    return format(value, "0{}x".format(max(1, (hash_size * hash_size) // 4)))


def hamming_distance(hash_a: str, hash_b: str) -> int:
    """Return the Hamming distance between two equal-length hexadecimal hashes."""
    if not hash_a or not hash_b:
        raise ValueError("Both hashes must be non-empty hexadecimal strings.")
    if len(hash_a) != len(hash_b):
        raise ValueError("Hashes must have equal length.")
    return bin(int(hash_a, 16) ^ int(hash_b, 16)).count("1")


def find_near_duplicate_groups(
    hashes: Dict[str, str],
    threshold: int = NEAR_DUPLICATE_HAMMING_THRESHOLD,
) -> List[List[str]]:
    """Group identifiers whose aHashes are within the Hamming threshold."""
    names = list(hashes.keys())
    parent = {name: name for name in names}

    def _find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            if hamming_distance(hashes[a], hashes[b]) <= threshold:
                root_a, root_b = _find(a), _find(b)
                if root_a != root_b:
                    parent[root_b] = root_a

    buckets: Dict[str, List[str]] = defaultdict(list)
    for name in names:
        buckets[_find(name)].append(name)
    return [sorted(group) for group in buckets.values() if len(group) > 1]


def _probe_image(image_path: Path) -> Image.Image:
    with Image.open(image_path) as probe:
        probe.verify()
    with Image.open(image_path) as reopened:
        image = reopened.convert("RGB")
        image.load()
        return image


def screen_image_quality(
    image_path: Optional[Union[str, Path]],
    *,
    filename: Optional[str] = None,
    near_duplicate: bool = False,
    recommended_min_dimension: int = RECOMMENDED_MIN_DIMENSION,
    absolute_min_dimension: int = ABSOLUTE_MIN_DIMENSION,
) -> QualityFinding:
    """Screen one image for corruption, size, blur, exposure, and duplication."""
    path = Path(image_path) if image_path is not None else None
    resolved_name = filename or (path.name if path is not None else "unknown")

    if path is None or not path.is_file():
        return QualityFinding(
            filename=resolved_name,
            image_path=str(path) if path is not None else None,
            readable=False,
            status="unusable",
            flags=["unreadable"],
            metrics={},
            reasons=[f"Image file not found: {resolved_name}"],
        )

    try:
        image = _probe_image(path)
    except Exception as exc:
        return QualityFinding(
            filename=resolved_name,
            image_path=str(path),
            readable=False,
            status="unusable",
            flags=["unreadable"],
            metrics={},
            reasons=[f"Image could not be decoded: {exc}"],
        )

    width, height = image.size
    blur = compute_blur_score(image)
    brightness = compute_brightness(image)
    contrast = compute_contrast(image)
    metrics = {
        "width": float(width),
        "height": float(height),
        "file_size_bytes": float(path.stat().st_size),
        "blur_variance": blur,
        "brightness": brightness,
        "contrast": contrast,
    }

    flags: List[str] = []
    reasons: List[str] = []

    if width < absolute_min_dimension or height < absolute_min_dimension:
        flags.append("too_small")
        reasons.append(
            f"Image is {width}x{height}px, below the absolute minimum of "
            f"{absolute_min_dimension}px."
        )
    elif width < recommended_min_dimension or height < recommended_min_dimension:
        flags.append("low_resolution")
        reasons.append(
            f"Image is {width}x{height}px, below the recommended "
            f"{recommended_min_dimension}px inspection resolution."
        )

    if blur < BLUR_VARIANCE_THRESHOLD:
        flags.append("excessive_blur")
        reasons.append(
            f"Edge detail is low (blur variance {blur:.1f} < {BLUR_VARIANCE_THRESHOLD:.0f})."
        )

    if brightness < DARK_BRIGHTNESS_THRESHOLD:
        flags.append("low_brightness")
        reasons.append(
            f"Mean brightness {brightness:.1f} is below {DARK_BRIGHTNESS_THRESHOLD:.0f} (underexposed)."
        )
    elif brightness > BRIGHT_BRIGHTNESS_THRESHOLD:
        flags.append("high_brightness")
        reasons.append(
            f"Mean brightness {brightness:.1f} is above {BRIGHT_BRIGHTNESS_THRESHOLD:.0f} (overexposed)."
        )

    if contrast < LOW_CONTRAST_STD_THRESHOLD:
        flags.append("low_contrast")
        reasons.append(
            f"Tonal variation {contrast:.1f} is below {LOW_CONTRAST_STD_THRESHOLD:.0f} (washed out)."
        )

    if near_duplicate:
        flags.append("near_duplicate")
        reasons.append(QUALITY_FLAG_DESCRIPTIONS["near_duplicate"])

    if "too_small" in flags:
        status = "unusable"
    elif flags:
        status = "warning"
    else:
        status = "ok"

    if not reasons:
        reasons.append("No image-quality issues detected.")

    return QualityFinding(
        filename=resolved_name,
        image_path=str(path),
        readable=True,
        status=status,
        flags=flags,
        metrics=metrics,
        reasons=reasons,
    )


def _read_checkpoint_metadata(checkpoint_path: Path) -> Optional[Dict[str, Any]]:
    sidecar = Path(str(checkpoint_path) + ".meta.json")
    if not sidecar.is_file():
        return None
    try:
        with open(sidecar, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def inspect_checkpoint_compatibility(
    checkpoint_path: Optional[Union[str, Path]],
    *,
    target_classes: Sequence[str] = TARGET_CLASSES,
) -> Dict[str, Any]:
    """Report whether a checkpoint exists and maps exactly onto the target classes."""
    result: Dict[str, Any] = {
        "path": str(checkpoint_path) if checkpoint_path else None,
        "exists": False,
        "compatible": False,
        "classes": [],
        "model_name": "",
        "architecture": "",
        "reason": "",
    }

    if not checkpoint_path:
        result["reason"] = "checkpoint_not_configured"
        return result

    path = Path(checkpoint_path)
    if not path.is_file():
        result["reason"] = "checkpoint_not_found"
        return result
    result["exists"] = True

    meta = _read_checkpoint_metadata(path)
    if meta is not None:
        classes = list(meta.get("classes") or [])
        result["classes"] = classes
        result["model_name"] = str(meta.get("model_name", path.stem))
        result["architecture"] = str(meta.get("architecture", "custom_cnn"))
    else:
        try:
            import torch

            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            result["reason"] = "checkpoint_unreadable"
            result["error"] = str(exc)
            return result
        if not isinstance(payload, dict):
            result["reason"] = "checkpoint_payload_invalid"
            return result
        classes = list(payload.get("classes") or [])
        result["classes"] = classes
        result["model_name"] = str(payload.get("model_name", path.stem))
        result["architecture"] = str(payload.get("architecture", "custom_cnn"))

    if not classes:
        result["reason"] = "checkpoint_missing_class_mapping"
        return result
    if len(classes) != len(list(target_classes)) or set(classes) != set(target_classes):
        result["reason"] = "unknown_class_mapping"
        return result

    result["compatible"] = True
    result["reason"] = "compatible"
    return result


def _build_predictor(
    checkpoint_path: Optional[Union[str, Path]],
    *,
    device: str,
    uncertainty_threshold: float,
) -> Tuple[Optional[Any], Dict[str, Any]]:
    compatibility = inspect_checkpoint_compatibility(checkpoint_path)
    if not compatibility["compatible"]:
        return None, compatibility
    try:
        from seedvision.inference.predictor import SeedPredictor

        predictor = SeedPredictor(
            checkpoint_path=str(checkpoint_path),
            device=device,
            uncertainty_threshold=uncertainty_threshold,
        )
    except Exception as exc:
        compatibility = dict(compatibility)
        compatibility["compatible"] = False
        compatibility["reason"] = "predictor_initialization_failed"
        compatibility["error"] = str(exc)
        return None, compatibility
    return predictor, compatibility


def _suggest_with_predictor(
    predictor: Any,
    image_path: Optional[Union[str, Path]],
    *,
    top_k_count: int = DEFAULT_TOP_K,
) -> ClassSuggestion:
    model_name = getattr(predictor, "model_name", "")
    architecture = getattr(predictor, "architecture", "")

    if image_path is None or not Path(image_path).is_file():
        return ClassSuggestion(
            available=False,
            image_path=str(image_path) if image_path is not None else None,
            reason="image_unavailable",
            model_name=model_name,
            architecture=architecture,
        )

    try:
        result = predictor.predict(
            image=str(image_path),
            top_k_count=top_k_count,
            generate_cam=False,
        )
    except Exception as exc:
        return ClassSuggestion(
            available=False,
            image_path=str(image_path),
            reason="inference_failed",
            model_name=model_name,
            architecture=architecture,
            error=str(exc),
        )

    return ClassSuggestion(
        available=True,
        image_path=str(image_path),
        predicted_class=result.predicted_class,
        confidence=float(result.confidence_score),
        is_uncertain=bool(result.is_uncertain),
        uncertainty_reason=getattr(result, "uncertainty_reason", "") or "",
        top_k=[
            {
                "rank": item.rank,
                "class_name": item.class_name,
                "probability": item.probability,
                "percentage": item.percentage_str,
            }
            for item in (result.top_k or [])
        ],
        model_name=getattr(result, "model_name", "") or model_name,
        architecture=getattr(result, "architecture", "") or architecture,
        reason="ok",
        scientific_notice=getattr(result, "scientific_notice", "") or MODEL_SUGGESTION_NOTICE,
    )


def _unavailable_suggestion(
    image_path: Optional[Union[str, Path]],
    compatibility: Dict[str, Any],
) -> ClassSuggestion:
    return ClassSuggestion(
        available=False,
        image_path=str(image_path) if image_path is not None else None,
        reason=compatibility.get("reason", "model_unavailable"),
        model_name=compatibility.get("model_name", ""),
        architecture=compatibility.get("architecture", ""),
        error=compatibility.get("error", ""),
    )


def _assemble_specimen(
    specimen_id: str,
    specimen: Dict[str, Any],
    view_results: List[Dict[str, Any]],
    *,
    human_candidate_label: str,
) -> SpecimenAssist:
    quality_flags: List[str] = []
    statuses: List[str] = []
    for view in view_results:
        quality = view["quality"]
        statuses.append(quality["status"])
        for flag in quality["flags"]:
            if flag not in quality_flags:
                quality_flags.append(flag)

    if "unusable" in statuses:
        quality_status = "unusable"
    elif "warning" in statuses:
        quality_status = "warning"
    else:
        quality_status = "ok"

    per_class: Dict[str, List[float]] = defaultdict(list)
    available_classes: List[str] = []
    uncertain_present = False
    for view in view_results:
        suggestion = view["suggestion"]
        if not suggestion.get("available"):
            continue
        predicted = suggestion.get("predicted_class")
        if not predicted:
            continue
        available_classes.append(predicted)
        if suggestion.get("confidence") is not None:
            per_class[predicted].append(float(suggestion["confidence"]))
        if suggestion.get("is_uncertain"):
            uncertain_present = True

    suggestion_available = bool(available_classes)
    conflicts: List[str] = []
    model_suggested_label: Optional[str] = None
    mean_confidence: Optional[float] = None

    if suggestion_available:
        ordered = sorted(
            per_class.items(),
            key=lambda item: (len(item[1]), float(np.mean(item[1]) if item[1] else 0.0)),
            reverse=True,
        )
        model_suggested_label = ordered[0][0]
        all_confidences = [c for values in per_class.values() for c in values]
        mean_confidence = float(np.mean(all_confidences)) if all_confidences else None
        conflicts = sorted(per_class.keys())

    views_agree = len(conflicts) <= 1

    reasons: List[str] = []
    if quality_status == "unusable":
        recommendation = "do_not_use"
        reasons.append(
            "At least one view is unreadable or below the absolute usable resolution; "
            "the specimen cannot be reliably inspected."
        )
    elif quality_status == "warning":
        blocking = [
            flag
            for flag in quality_flags
            if flag in ("low_resolution", "excessive_blur", "low_brightness", "high_brightness", "low_contrast")
        ]
        if blocking:
            recommendation = "retake_image"
            reasons.append(
                "Image-quality issues detected ({}); retake photographs before making a decision.".format(
                    ", ".join(sorted(blocking))
                )
            )
        else:
            recommendation = "inspect_closely"
            reasons.append(
                "Views are usable but warrant closer inspection ({}).".format(
                    ", ".join(sorted(quality_flags))
                )
            )
    elif suggestion_available and views_agree and not uncertain_present:
        recommendation = "accept_candidate"
        reasons.append(
            "Views are readable with acceptable quality and model suggestions agree. "
            "Advisory only: a human reviewer must still confirm the label."
        )
    else:
        recommendation = "inspect_closely"
        if not suggestion_available:
            reasons.append("No compatible model suggestion is available; classify by human inspection.")
        if not views_agree:
            reasons.append(
                "Model suggestions conflict across views ({}).".format(", ".join(conflicts))
            )
        if uncertain_present:
            reasons.append("One or more model suggestions are uncertain.")

    provisional = SpecimenAssist(
        specimen_id=specimen_id,
        human_candidate_label=human_candidate_label,
        quality_status=quality_status,
        quality_flags=quality_flags,
        views=view_results,
        suggestion_available=suggestion_available,
        model_suggested_label=model_suggested_label,
        mean_confidence=mean_confidence,
        views_agree=views_agree,
        conflicting_classes=conflicts if not views_agree else [],
        recommendation=recommendation,
        recommendation_reason=" ".join(reasons).strip(),
        is_auto_approvable=False,
        auto_approval_reason="",
    )

    approval = evaluate_auto_approval(provisional)
    provisional.is_auto_approvable = bool(approval["allowed"])
    provisional.auto_approval_reason = approval["reason"] + ": " + approval["detail"]
    return provisional


def evaluate_auto_approval(
    assist: SpecimenAssist,
    *,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Decide whether automatic approval is permitted.

    Automatic approval is disabled by default. Even when the module master switch
    and configuration flag are enabled, documented validation evidence must be
    present and every safeguard must hold; model confidence alone never suffices.
    """
    enabled = False
    documented = False
    if isinstance(config, dict):
        block = config.get(AUTO_APPROVAL_CONFIG_KEY, {})
        if isinstance(block, dict):
            enabled = bool(block.get("enabled", False))
            documented = bool(block.get("validation_documented", False))

    def _deny(reason: str, detail: str) -> Dict[str, Any]:
        return {"allowed": False, "reason": reason, "detail": detail}

    if not AUTO_APPROVAL_ENABLED:
        return _deny(
            "auto_approval_module_disabled",
            "The module master switch AUTO_APPROVAL_ENABLED is False.",
        )
    if not enabled:
        return _deny(
            "auto_approval_not_enabled_in_config",
            "No explicit auto_approval.enabled configuration flag was provided.",
        )
    if not documented:
        return _deny(
            "documented_validation_missing",
            "Documented validation evidence (auto_approval.validation_documented) is required.",
        )
    if assist.quality_status != "ok":
        return _deny("quality_not_acceptable", f"Specimen quality status is '{assist.quality_status}'.")
    if assist.human_candidate_label != "pending_review":
        return _deny(
            "specimen_already_reviewed",
            "Auto-approval never applies to specimens with an existing human label.",
        )
    if not assist.suggestion_available or not assist.model_suggested_label:
        return _deny("no_model_suggestion", "No model class suggestion is available.")
    if not assist.views_agree:
        return _deny("conflicting_view_suggestions", "Model suggestions conflict across views.")
    if any(view["suggestion"].get("is_uncertain") for view in assist.views):
        return _deny("uncertain_suggestion", "At least one model suggestion is uncertain.")
    return {
        "allowed": True,
        "reason": "all_safeguards_satisfied",
        "detail": "Explicit configuration, documented validation, and all safeguards are satisfied.",
    }


def assist_specimen(
    specimen_id: str,
    specimen: Dict[str, Any],
    *,
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: str = "cpu",
    uncertainty_threshold: float = DEFAULT_UNCERTAINTY_THRESHOLD,
    predictor: Optional[Any] = None,
    compatibility: Optional[Dict[str, Any]] = None,
    near_duplicate_filenames: Optional[Sequence[str]] = None,
    recommended_min_dimension: int = RECOMMENDED_MIN_DIMENSION,
    top_k_count: int = DEFAULT_TOP_K,
) -> SpecimenAssist:
    """Screen every view of one specimen and aggregate advisory findings."""
    if predictor is None and compatibility is None:
        predictor, compatibility = _build_predictor(
            checkpoint_path,
            device=device,
            uncertainty_threshold=uncertainty_threshold,
        )
    if compatibility is None:
        compatibility = {"reason": "model_unavailable", "model_name": "", "architecture": ""}

    duplicate_names = set(near_duplicate_filenames or [])
    views = specimen.get("views", []) or []
    view_results: List[Dict[str, Any]] = []

    for view in views:
        image_path = view.get("image_path")
        filename = view.get("filename") or (Path(image_path).name if image_path else "unknown")
        finding = screen_image_quality(
            image_path,
            filename=filename,
            near_duplicate=filename in duplicate_names,
            recommended_min_dimension=recommended_min_dimension,
        )
        if predictor is not None:
            suggestion = _suggest_with_predictor(predictor, image_path, top_k_count=top_k_count)
        else:
            suggestion = _unavailable_suggestion(image_path, compatibility)
        view_results.append(
            {
                "view_id": view.get("view_id"),
                "filename": filename,
                "image_path": str(image_path) if image_path else None,
                "image_exists": bool(view.get("image_exists", Path(image_path).is_file() if image_path else False)),
                "quality": finding.to_dict(),
                "suggestion": suggestion.to_dict(),
            }
        )

    human_label = specimen.get("candidate_label", "pending_review")
    return _assemble_specimen(
        specimen_id,
        specimen,
        view_results,
        human_candidate_label=human_label,
    )


def _collect_hashes(
    specimens: Dict[str, Dict[str, Any]],
    *,
    hash_size: int = HASH_SIZE,
) -> Tuple[Dict[str, str], List[List[str]]]:
    hashes: Dict[str, str] = {}
    for specimen in specimens.values():
        for view in specimen.get("views", []) or []:
            image_path = view.get("image_path")
            filename = view.get("filename") or (Path(image_path).name if image_path else None)
            if not image_path or not filename or filename in hashes:
                continue
            path = Path(image_path)
            if not path.is_file():
                continue
            try:
                hashes[filename] = compute_average_hash(_probe_image(path), hash_size=hash_size)
            except Exception:
                continue
    return hashes, find_near_duplicate_groups(hashes)


def build_assist_report(
    assists: Sequence[SpecimenAssist],
    *,
    compatibility: Dict[str, Any],
    near_duplicate_groups: Optional[List[List[str]]] = None,
) -> Dict[str, Any]:
    """Build a serializable advisory report from aggregated specimen findings."""
    from datetime import datetime

    specimens = {assist.specimen_id: assist.to_dict() for assist in assists}
    recommendations: Dict[str, int] = defaultdict(int)
    for assist in assists:
        recommendations[assist.recommendation] += 1

    return {
        "generated_at": datetime.now().isoformat(),
        "advisory_notice": ADVISORY_NOTICE,
        "auto_approval_enabled": AUTO_APPROVAL_ENABLED,
        "auto_approval_requirements": list(AUTO_APPROVAL_REQUIREMENTS),
        "model": {
            "compatible": bool(compatibility.get("compatible", False)),
            "reason": compatibility.get("reason", ""),
            "model_name": compatibility.get("model_name", ""),
            "architecture": compatibility.get("architecture", ""),
            "classes": compatibility.get("classes", []),
            "path": compatibility.get("path"),
        },
        "near_duplicate_groups": near_duplicate_groups or [],
        "specimens": specimens,
        "summary": {
            "total_specimens": len(assists),
            "recommendations": dict(recommendations),
            "specimens_with_suggestions": sum(1 for a in assists if a.suggestion_available),
            "specimens_with_suggestion_conflicts": sum(1 for a in assists if not a.views_agree),
            "auto_approvable_specimens": sum(1 for a in assists if a.is_auto_approvable),
        },
    }


def assist_staging(
    staging_data: Dict[str, Any],
    *,
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: str = "cpu",
    uncertainty_threshold: float = DEFAULT_UNCERTAINTY_THRESHOLD,
    recommended_min_dimension: int = RECOMMENDED_MIN_DIMENSION,
    top_k_count: int = DEFAULT_TOP_K,
) -> Dict[str, Any]:
    """Screen and suggest across every specimen in a staging manifest payload."""
    specimens = staging_data.get("specimens", {}) or {}
    predictor, compatibility = _build_predictor(
        checkpoint_path,
        device=device,
        uncertainty_threshold=uncertainty_threshold,
    )
    _, near_duplicate_groups = _collect_hashes(specimens)
    duplicate_names = {name for group in near_duplicate_groups for name in group}

    assists = [
        assist_specimen(
            specimen_id,
            specimen,
            predictor=predictor,
            compatibility=compatibility,
            near_duplicate_filenames=duplicate_names,
            recommended_min_dimension=recommended_min_dimension,
            top_k_count=top_k_count,
        )
        for specimen_id, specimen in specimens.items()
    ]

    return build_assist_report(
        assists,
        compatibility=compatibility,
        near_duplicate_groups=near_duplicate_groups,
    )


def write_assist_report(
    report: Dict[str, Any],
    output_path: Union[str, Path] = DEFAULT_ASSIST_REPORT_PATH,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write the advisory report without touching the staging manifest."""
    path = Path(output_path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Advisory report already exists (pass overwrite=True): {path}")
    path.parent.mkdir(parents=True, exist_ok=True)

    handle, temp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as temp_handle:
            json.dump(report, temp_handle, indent=2)
            temp_handle.flush()
            os.fsync(temp_handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        if os.path.exists(temp_name):
            os.remove(temp_name)
        raise
    return path


def load_assist_report(report_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Load a previously written advisory report, returning None when absent."""
    path = Path(report_path)
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
