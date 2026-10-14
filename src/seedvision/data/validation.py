"""Dataset and image validation utilities for SeedVision AI."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import hashlib
import logging
import os
from PIL import Image

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class ImageValidationResult:
    is_valid: bool
    filepath: str
    error_message: Optional[str] = None
    width: int = 0
    height: int = 0
    channels: int = 0
    format: Optional[str] = None
    file_size_bytes: int = 0
    file_hash: Optional[str] = None


@dataclass
class DatasetValidationReport:
    total_images_found: int = 0
    valid_images_count: int = 0
    corrupt_images_count: int = 0
    duplicate_images_count: int = 0
    class_counts: Dict[str, int] = field(default_factory=dict)
    corrupted_files: List[Tuple[str, str]] = field(default_factory=list)
    duplicate_pairs: List[Tuple[str, str]] = field(default_factory=list)
    missing_classes: List[str] = field(default_factory=list)
    is_ready_for_training: bool = False
    summary_message: str = ""


class ImageValidator:
    """Validates single image files against corruption, dimensions, and size constraints."""

    def __init__(
        self,
        min_width: int = 16,
        min_height: int = 16,
        max_file_size_bytes: int = 25 * 1024 * 1024,  # 25 MB
        allowed_extensions: Optional[Set[str]] = None,
    ):
        self.min_width = min_width
        self.min_height = min_height
        self.max_file_size_bytes = max_file_size_bytes
        self.allowed_extensions = allowed_extensions or SUPPORTED_EXTENSIONS

    def validate_file(self, filepath: str) -> ImageValidationResult:
        """Validate an image file on disk.

        Args:
            filepath: Path to the image file.

        Returns:
            ImageValidationResult with status and metadata.
        """
        path = Path(filepath)

        if not path.is_file():
            return ImageValidationResult(
                is_valid=False,
                filepath=str(path),
                error_message=f"File does not exist: {filepath}",
            )

        # Check extension
        ext = path.suffix.lower()
        if ext not in self.allowed_extensions:
            return ImageValidationResult(
                is_valid=False,
                filepath=str(path),
                error_message=f"Unsupported image extension '{ext}'. Allowed: {sorted(list(self.allowed_extensions))}",
            )

        # Check file size
        file_size = path.stat().st_size
        if file_size == 0:
            return ImageValidationResult(
                is_valid=False,
                filepath=str(path),
                error_message="Image file is completely empty (0 bytes).",
                file_size_bytes=0,
            )

        if file_size > self.max_file_size_bytes:
            return ImageValidationResult(
                is_valid=False,
                filepath=str(path),
                error_message=f"File size ({file_size / 1024 / 1024:.2f} MB) exceeds maximum allowed ({self.max_file_size_bytes / 1024 / 1024:.2f} MB).",
                file_size_bytes=file_size,
            )

        # Validate image decoding
        try:
            with Image.open(path) as img:
                img.verify()  # Fast structural verification

            # Re-open for actual decompression and attribute extraction
            with Image.open(path) as img:
                img.load()
                w, h = img.size
                fmt = img.format
                channels = len(img.getbands())

                if w < self.min_width or h < self.min_height:
                    return ImageValidationResult(
                        is_valid=False,
                        filepath=str(path),
                        error_message=f"Image resolution {w}x{h} is below minimum requirement {self.min_width}x{self.min_height}.",
                        width=w,
                        height=h,
                        channels=channels,
                        format=fmt,
                        file_size_bytes=file_size,
                    )

                # Compute content hash
                hasher = hashlib.sha256()
                with open(path, "rb") as f:
                    for chunk in iter(lambda: f.read(65536), b""):
                        hasher.update(chunk)
                file_hash = hasher.hexdigest()

                return ImageValidationResult(
                    is_valid=True,
                    filepath=str(path),
                    width=w,
                    height=h,
                    channels=channels,
                    format=fmt,
                    file_size_bytes=file_size,
                    file_hash=file_hash,
                )

        except Exception as e:
            return ImageValidationResult(
                is_valid=False,
                filepath=str(path),
                error_message=f"Corrupted image or decoding failure: {str(e)}",
                file_size_bytes=file_size,
            )


def validate_dataset_structure(
    dataset_dir: str,
    expected_classes: List[str],
    validator: Optional[ImageValidator] = None,
) -> DatasetValidationReport:
    """Validate full dataset directory containing class subdirectories.

    Args:
        dataset_dir: Path to root dataset directory.
        expected_classes: List of expected class folder names.
        validator: ImageValidator instance or None.

    Returns:
        DatasetValidationReport with complete validation statistics.
    """
    v = validator or ImageValidator()
    report = DatasetValidationReport()
    root = Path(dataset_dir)

    if not root.exists():
        report.summary_message = f"Dataset directory '{dataset_dir}' does not exist."
        return report

    report.class_counts = {c: 0 for c in expected_classes}
    seen_hashes: Dict[str, str] = {}

    for cls in expected_classes:
        class_folder = root / cls
        if not class_folder.is_dir():
            report.missing_classes.append(cls)
            continue

        for file_path in class_folder.iterdir():
            if not file_path.is_file():
                continue

            report.total_images_found += 1
            res = v.validate_file(str(file_path))

            if not res.is_valid:
                report.corrupt_images_count += 1
                report.corrupted_files.append((str(file_path), res.error_message or "Unknown error"))
            else:
                # Check duplicate
                if res.file_hash and res.file_hash in seen_hashes:
                    report.duplicate_images_count += 1
                    report.duplicate_pairs.append((str(file_path), seen_hashes[res.file_hash]))
                else:
                    if res.file_hash:
                        seen_hashes[res.file_hash] = str(file_path)
                    report.valid_images_count += 1
                    report.class_counts[cls] += 1

    # Check readiness
    has_all_classes = len(report.missing_classes) == 0
    all_classes_have_data = all(report.class_counts.get(c, 0) > 0 for c in expected_classes)
    report.is_ready_for_training = has_all_classes and all_classes_have_data and report.valid_images_count >= 10

    if report.is_ready_for_training:
        report.summary_message = (
            f"Dataset is valid: {report.valid_images_count} valid samples across {len(expected_classes)} classes."
        )
    else:
        reasons = []
        if report.missing_classes:
            reasons.append(f"Missing class directories: {report.missing_classes}")
        zero_classes = [c for c, count in report.class_counts.items() if count == 0]
        if zero_classes:
            reasons.append(f"Empty classes: {zero_classes}")
        if report.valid_images_count < 10:
            reasons.append(f"Insufficient total samples ({report.valid_images_count} < 10)")
        report.summary_message = "Dataset validation incomplete: " + "; ".join(reasons)

    return report
