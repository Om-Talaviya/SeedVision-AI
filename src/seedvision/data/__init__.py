"""Data ingestion, validation, splitting, and preprocessing for SeedVision AI."""

from seedvision.data.dataset import SeedDataset, create_dataloaders
from seedvision.data.preprocessing import get_eval_transforms, get_train_transforms
from seedvision.data.roboflow_review import (
    CONFIRMED_FOREIGN_MATERIAL_CLASSES,
    SEEDVISION_TARGET_LABELS,
    apply_manual_review_decisions,
    extract_source_annotations,
    generate_crop_metadata,
    get_crop_dimensions,
    get_crop_triage_flags,
    revert_manual_review_decisions,
    sync_source_images,
    triage_foreign_material,
    validate_human_decision,
    validate_source_image_linkage,
)
from seedvision.data.splitting import create_leakage_resistant_splits
from seedvision.data.validation import ImageValidator, validate_dataset_structure

__all__ = [
    "SeedDataset",
    "create_dataloaders",
    "get_train_transforms",
    "get_eval_transforms",
    "create_leakage_resistant_splits",
    "ImageValidator",
    "validate_dataset_structure",
    "generate_crop_metadata",
    "triage_foreign_material",
    "get_crop_dimensions",
    "get_crop_triage_flags",
    "extract_source_annotations",
    "sync_source_images",
    "validate_source_image_linkage",
    "CONFIRMED_FOREIGN_MATERIAL_CLASSES",
    "SEEDVISION_TARGET_LABELS",
    "validate_human_decision",
    "apply_manual_review_decisions",
    "revert_manual_review_decisions",
    "ALLOWED_TARGET_LABELS",
    "load_staging_manifest",
    "validate_review_decision",
    "save_specimen_review",
    "revert_specimen_review",
    "restore_manifest_backup",
    "INGEST_ANNOTATION_COLUMNS",
    "specimen_seed_id",
    "select_approved_specimens",
    "build_ingest_annotations",
    "build_annotation_dataframe",
    "write_annotations_atomic",
    "plan_image_staging",
    "stage_images",
    "build_bridge_report",
]

from seedvision.data.real_review import (
    ALLOWED_TARGET_LABELS,
    load_staging_manifest,
    restore_manifest_backup,
    revert_specimen_review,
    save_specimen_review,
    validate_review_decision,
)
from seedvision.data.review_bridge import (
    INGEST_ANNOTATION_COLUMNS,
    build_annotation_dataframe,
    build_bridge_report,
    build_ingest_annotations,
    plan_image_staging,
    select_approved_specimens,
    specimen_seed_id,
    stage_images,
    write_annotations_atomic,
)


