import csv
import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

from seedvision.data.roboflow_preprocessing import (
    _parse_annotation_line,
    source_group_id,
    stage_review_crops,
    validate_roboflow_dataset,
)


def _dataset(root: Path) -> Path:
    root.mkdir()
    (root / "data.yaml").write_text(
        yaml.safe_dump({"nc": 2, "names": ["Garbanzo", "Calibre 7"]}), encoding="utf-8"
    )
    for split in ("train", "valid", "test"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir()
    image = root / "train" / "images" / "IMG_1_jpg.rf.abcd1234.jpg"
    Image.new("RGB", (100, 80), "white").save(image)
    (root / "train" / "labels" / f"{image.stem}.txt").write_text(
        "0 0.5 0.5 0.4 0.5\n1 0.1 0.1 0.3 0.1 0.2 0.3\n", encoding="utf-8"
    )
    return root


@pytest.mark.parametrize(
    "line",
    [
        "2 0.5 0.5 0.2 0.2",
        "x 0.5 0.5 0.2 0.2",
        "0 0.5 0.5 0 0.2",
        "0 0.05 0.5 0.2 0.2",
        "0 0.1 0.1 0.2",
        "0 0 0 0.5 0.5 1 1",
        "0 nan 0.5 0.2 0.2",
    ],
)
def test_rejects_invalid_bbox_and_polygon_lines(line):
    with pytest.raises(ValueError):
        _parse_annotation_line(line, class_count=2)


def test_validates_boxes_polygons_pairs_and_image_integrity(tmp_path):
    report = validate_roboflow_dataset(_dataset(tmp_path / "dataset"))

    assert report["status"] == "valid"
    assert report["annotation_count"] == 2
    assert report["geometry_counts"] == {"bbox": 1, "polygon": 1}
    assert report["split_stats"]["train"]["readable_images"] == 1


def test_reports_missing_orphan_bad_labels_and_cross_split_source_group(tmp_path):
    root = _dataset(tmp_path / "dataset")
    other = root / "valid" / "images" / "IMG_1_jpg.rf.ffff9999.jpg"
    Image.new("RGB", (100, 80), "white").save(other)
    (root / "valid" / "labels" / f"{other.stem}.txt").write_text(
        "0 0.5 0.5 1.2 0.2\n", encoding="utf-8"
    )
    (root / "test" / "labels" / "orphan.txt").write_text("", encoding="utf-8")
    report = validate_roboflow_dataset(root)
    kinds = {issue["kind"] for issue in report["issues"]}

    assert "source_group_crosses_splits" in kinds
    assert "invalid_annotation" in kinds
    assert "orphan_label" in kinds
    assert report["cross_split_source_group_count"] == 1


def test_staged_crops_remain_pending_and_export_is_read_only(tmp_path):
    root = _dataset(tmp_path / "dataset")
    label = root / "train" / "labels" / "IMG_1_jpg.rf.abcd1234.txt"
    before = label.read_bytes()
    output = tmp_path / "review"

    result = stage_review_crops(root, output)

    assert result["crop_count"] == 2
    with (output / "manual_review.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2
    assert all(row["review_status"] == "pending" and row["reviewed_label"] == "" for row in rows)
    assert {row["source_group_id"] for row in rows} == {"img_1"}
    assert all((output / row["crop_path"]).is_file() for row in rows)
    assert json.loads((output / "validation_report.json").read_text(encoding="utf-8"))["status"] == "valid"
    assert label.read_bytes() == before


def test_crop_output_must_not_modify_dataset(tmp_path):
    root = _dataset(tmp_path / "dataset")

    with pytest.raises(ValueError, match="separate"):
        stage_review_crops(root, root / "review")


def test_source_group_strips_roboflow_variant_suffix():
    assert source_group_id(Path("IMG_4151_jpg.rf.92c64eddcfe26ae6aa4553fc2a645e9e.jpg")) == "img_4151"
