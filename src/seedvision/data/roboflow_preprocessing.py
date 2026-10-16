"""Safe, review-first preprocessing for Roboflow YOLO detection exports.

This module is intentionally separate from SeedVision's classification pipeline.
It validates YOLO boxes and polygons and can stage individual crops for manual
review without assigning SeedVision defect labels.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "valid", "test")


@dataclass(frozen=True)
class Annotation:
    class_id: int
    geometry: str
    bbox: tuple[float, float, float, float]
    polygon: tuple[tuple[float, float], ...] = ()


def source_group_id(image_path: Path) -> str:
    """Group Roboflow augmented exports by the pre-augmentation image stem."""
    stem = image_path.stem
    stem = re.sub(r"\.rf\.[0-9a-fA-F]+$", "", stem)
    # Roboflow commonly encodes the original suffix in names such as IMG_1_jpg.
    stem = re.sub(r"_(?:jpg|jpeg|png|bmp|tif|tiff|webp)$", "", stem, flags=re.I)
    return stem.casefold()


def _parse_annotation_line(line: str, class_count: int) -> Annotation:
    fields = line.split()
    if not fields:
        raise ValueError("empty annotation")
    try:
        class_id = int(fields[0])
    except ValueError as exc:
        raise ValueError("class id must be an integer") from exc
    if not 0 <= class_id < class_count:
        raise ValueError(f"class id {class_id} is outside [0, {class_count - 1}]")
    try:
        values = [float(value) for value in fields[1:]]
    except ValueError as exc:
        raise ValueError("coordinates must be numeric") from exc
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in values):
        raise ValueError("coordinates must be finite and normalized to [0, 1]")

    if len(values) == 4:
        cx, cy, width, height = values
        if width <= 0 or height <= 0:
            raise ValueError("box width and height must be positive")
        x1, y1, x2, y2 = cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2
        if x1 < 0 or y1 < 0 or x2 > 1 or y2 > 1:
            raise ValueError("box extends outside normalized image bounds")
        return Annotation(class_id, "bbox", (x1, y1, x2, y2))

    if len(values) < 6 or len(values) % 2:
        raise ValueError("polygon needs at least three x,y coordinate pairs")
    points = tuple(zip(values[::2], values[1::2]))
    twice_area = sum(
        points[i][0] * points[(i + 1) % len(points)][1]
        - points[(i + 1) % len(points)][0] * points[i][1]
        for i in range(len(points))
    )
    if math.isclose(twice_area, 0.0, abs_tol=1e-12):
        raise ValueError("polygon area must be non-zero")
    xs, ys = zip(*points)
    return Annotation(class_id, "polygon", (min(xs), min(ys), max(xs), max(ys)), points)


def _load_dataset_config(dataset: Path) -> tuple[list[str], list[str]]:
    config_path = dataset / "data.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing dataset config: {config_path}")
    try:
        import yaml

        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise ValueError(f"Could not parse {config_path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError("data.yaml must contain a mapping at its root")
    names_value = config.get("names")
    if isinstance(names_value, dict):
        names = [str(names_value[key]) for key in sorted(names_value, key=lambda item: int(item))]
    elif isinstance(names_value, list):
        names = [str(name) for name in names_value]
    else:
        raise ValueError("data.yaml must define names as a list or numeric-key mapping")
    if not names or len(set(names)) != len(names):
        raise ValueError("data.yaml class names must be non-empty and unique")
    declared_count = config.get("nc")
    if declared_count is not None and int(declared_count) != len(names):
        raise ValueError(f"data.yaml nc={declared_count} but defines {len(names)} names")
    return names, list(SPLITS)


def validate_roboflow_dataset(dataset_dir: str | Path) -> dict[str, Any]:
    """Return a JSON-serializable audit; never writes to the dataset."""
    dataset = Path(dataset_dir).expanduser().absolute()
    if not dataset.is_dir():
        raise NotADirectoryError(dataset)
    names, splits = _load_dataset_config(dataset)
    issues: list[dict[str, str]] = []
    split_stats: dict[str, dict[str, int]] = {}
    class_counts = {name: 0 for name in names}
    geometry_counts = {"bbox": 0, "polygon": 0}
    grouped_splits: dict[str, set[str]] = {}
    annotation_rows: list[tuple[Path, str, int, Annotation]] = []

    for split in splits:
        split_root = dataset / split
        image_root, label_root = split_root / "images", split_root / "labels"
        if not split_root.is_dir():
            issues.append({"kind": "missing_split_directory", "path": str(split_root)})
        if not image_root.is_dir():
            issues.append({"kind": "missing_image_directory", "path": str(image_root)})
        if not label_root.is_dir():
            issues.append({"kind": "missing_label_directory", "path": str(label_root)})
        images = sorted(p for p in image_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES) if image_root.is_dir() else []
        labels = sorted(label_root.rglob("*.txt")) if label_root.is_dir() else []
        image_stems = [p.relative_to(image_root).with_suffix("").as_posix() for p in images]
        label_stems = [p.relative_to(label_root).with_suffix("").as_posix() for p in labels]
        for stem, count in sorted(Counter(image_stems).items()):
            if count > 1:
                issues.append({"kind": "duplicate_image_stem", "path": str(image_root / stem)})
        for stem, count in sorted(Counter(label_stems).items()):
            if count > 1:
                issues.append({"kind": "duplicate_label_stem", "path": str(label_root / stem)})
        image_map = {p.relative_to(image_root).with_suffix("").as_posix(): p for p in images}
        label_map = {p.relative_to(label_root).with_suffix("").as_posix(): p for p in labels}
        for key in sorted(image_map.keys() - label_map.keys()):
            issues.append({"kind": "missing_label", "path": str(image_map[key])})
        for key in sorted(label_map.keys() - image_map.keys()):
            issues.append({"kind": "orphan_label", "path": str(label_map[key])})

        valid_images = 0
        for image_path in images:
            group_id = source_group_id(image_path)
            grouped_splits.setdefault(group_id, set()).add(split)
            try:
                with Image.open(image_path) as image:
                    image.verify()
                valid_images += 1
            except Exception as exc:
                issues.append({"kind": "unreadable_image", "path": str(image_path), "detail": str(exc)})
            key = image_path.relative_to(image_root).with_suffix("").as_posix()
            label_path = label_map.get(key)
            if label_path is None:
                continue
            try:
                lines = label_path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeError) as exc:
                issues.append({"kind": "unreadable_label", "path": str(label_path), "detail": str(exc)})
                continue
            for line_number, line in enumerate(lines, 1):
                if not line.strip():
                    continue
                try:
                    annotation = _parse_annotation_line(line, len(names))
                except ValueError as exc:
                    issues.append({"kind": "invalid_annotation", "path": str(label_path), "line": str(line_number), "detail": str(exc)})
                    continue
                class_counts[names[annotation.class_id]] += 1
                geometry_counts[annotation.geometry] += 1
                annotation_rows.append((image_path, split, line_number, annotation))

        split_stats[split] = {
            "images": len(images), "readable_images": valid_images, "label_files": len(labels),
            "annotations": sum(1 for row in annotation_rows if row[1] == split),
        }

    leaked_groups = [group for group, group_splits in grouped_splits.items() if len(group_splits) > 1]
    for group in leaked_groups:
        issues.append({"kind": "source_group_crosses_splits", "source_group_id": group,
                       "splits": ",".join(sorted(grouped_splits[group]))})
    return {
        "dataset": str(dataset), "class_names": names, "split_stats": split_stats,
        "class_annotation_counts": class_counts, "geometry_counts": geometry_counts,
        "source_group_count": len(grouped_splits), "cross_split_source_group_count": len(leaked_groups),
        "annotation_count": sum(geometry_counts.values()), "issue_count": len(issues),
        "label_mapping_status": "unverified_manual_review_required",
        "issues": issues,
        "status": "valid" if not issues else "review_required",
    }


def stage_review_crops(dataset_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Stage crops and a pending-only review CSV; never assign defect labels."""
    dataset = Path(dataset_dir).expanduser().absolute()
    output = Path(output_dir).expanduser().absolute()
    if output == dataset or dataset in output.parents or output in dataset.parents:
        raise ValueError("Output directory must be separate from and outside the Roboflow export")
    if output.is_symlink():
        raise ValueError("Output directory must not be a symbolic link")
    report = validate_roboflow_dataset(dataset)
    names, splits = _load_dataset_config(dataset)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    crop_count = 0
    for split in splits:
        image_root, label_root = dataset / split / "images", dataset / split / "labels"
        for label_path in sorted(label_root.rglob("*.txt")) if label_root.is_dir() else []:
            relative = label_path.relative_to(label_root).with_suffix("")
            image_matches = [p for p in (image_root / relative.parent).glob(relative.name + ".*") if p.suffix.lower() in IMAGE_SUFFIXES]
            if len(image_matches) != 1:
                continue
            image_path = image_matches[0]
            try:
                with Image.open(image_path) as source:
                    image = source.convert("RGB")
            except Exception:
                continue
            width, height = image.size
            for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    ann = _parse_annotation_line(line, len(names))
                except ValueError:
                    continue
                x1, y1, x2, y2 = ann.bbox
                left, top = max(0, math.floor(x1 * width)), max(0, math.floor(y1 * height))
                right, bottom = min(width, math.ceil(x2 * width)), min(height, math.ceil(y2 * height))
                if right <= left or bottom <= top:
                    continue
                crop = image.crop((left, top, right, bottom))
                if ann.geometry == "polygon":
                    mask = Image.new("L", crop.size, 0)
                    draw = ImageDraw.Draw(mask)
                    points = [((x * width) - left, (y * height) - top) for x, y in ann.polygon]
                    draw.polygon(points, fill=255)
                    crop.putalpha(mask)
                source_key = f"{split}/{relative.as_posix()}:{line_number}"
                crop_id = hashlib.sha256(source_key.encode("utf-8")).hexdigest()[:16]
                crop_rel = Path("crops") / f"{crop_id}.png"
                destination = output / crop_rel
                destination.parent.mkdir(parents=True, exist_ok=True)
                crop.save(destination)
                rows.append({
                    "crop_path": crop_rel.as_posix(), "source_image": image_path.relative_to(dataset).as_posix(),
                    "source_group_id": source_group_id(image_path), "source_split": split,
                    "source_class_id": str(ann.class_id), "source_class_name": names[ann.class_id],
                    "annotation_line": str(line_number), "geometry": ann.geometry,
                    "review_status": "pending", "reviewed_label": "", "reviewer": "", "review_notes": "",
                })
                crop_count += 1
    manifest = output / "manual_review.csv"
    fields = list(rows[0]) if rows else ["crop_path", "source_image", "source_group_id", "source_split", "source_class_id", "source_class_name", "annotation_line", "geometry", "review_status", "reviewed_label", "reviewer", "review_notes"]
    with manifest.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"output_dir": str(output), "crop_count": crop_count, "manifest": str(manifest), "validation_status": report["status"]}
