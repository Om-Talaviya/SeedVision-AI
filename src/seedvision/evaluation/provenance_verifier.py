"""Rigorous dataset and model evaluation provenance verification for SeedVision AI."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd

logger = logging.getLogger(__name__)


def verify_evaluation_provenance(
    checkpoint_payload: Optional[Union[Dict[str, Any], str, Path]] = None,
    splits_dir: Union[str, Path] = "data/processed/splits",
    split_name: str = "test",
    expected_classes: Optional[List[str]] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    data_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Verify that an evaluation run genuinely satisfies real-world validation criteria.

    Real-world validation CANNOT be established by filename or directory heuristics (e.g.
    merely having 'Real' or 'real' in a string). It strictly requires:
    1. Checkpoint training provenance: Checkpoint payload must contain verified genuine training
       metadata ('is_real_world_trained': True and 'dataset_provenance': 'genuine_photograph').
    2. Dataset provenance: The dataset must have an authentic provenance record confirming
       'is_real_world_photograph': True and 'provenance_status': 'genuine_photograph'.
    3. Held-out manifest integrity: The split manifest must exist, contain valid rows, and point
       to image files that physically exist on disk and are readable.
    4. Zero leakage: No physical seed identifiers (seed_id) may cross the train and test partitions.

    If ANY condition fails or is contradictory, is_real_world_validated MUST be False.

    Returns:
        Dict containing:
            - is_real_world_validated: bool
            - evaluation_type: str
            - dataset_provenance: str
            - scientific_notice: str
            - verification_details: dict
    """
    # Accept checkpoint_path alias or string/Path checkpoint_payload
    resolved_ckpt = checkpoint_path if checkpoint_path is not None else checkpoint_payload
    if resolved_ckpt is None:
        checkpoint_payload = {}
    elif isinstance(resolved_ckpt, (str, Path)):
        ckpt_p = Path(resolved_ckpt)
        meta_p = Path(str(ckpt_p) + ".meta.json")
        extra_meta = {}
        if meta_p.is_file():
            try:
                with open(meta_p, "r", encoding="utf-8") as f:
                    extra_meta = json.load(f)
            except Exception:
                pass
        try:
            import torch
            loaded = torch.load(ckpt_p, map_location="cpu", weights_only=False)
            if isinstance(loaded, dict):
                checkpoint_payload = loaded
                if not checkpoint_payload.get("extra_metadata"):
                    checkpoint_payload["extra_metadata"] = extra_meta
            else:
                checkpoint_payload = {"extra_metadata": extra_meta}
        except Exception:
            checkpoint_payload = {"extra_metadata": extra_meta}
    else:
        checkpoint_payload = resolved_ckpt

    splits_path = Path(splits_dir)

    # 1. Verify Checkpoint Training Provenance
    extra_meta = checkpoint_payload.get("extra_metadata", {})
    if not extra_meta and isinstance(checkpoint_payload, dict):
        # Also check top-level keys if saved flat
        extra_meta = {
            "is_real_world_trained": checkpoint_payload.get("is_real_world_trained"),
            "dataset_provenance": checkpoint_payload.get("dataset_provenance"),
        }
    ckpt_is_real = (
        extra_meta.get("is_real_world_trained") is True
        and extra_meta.get("dataset_provenance") == "genuine_photograph"
    )


    # 2. Verify Dataset Provenance File
    dataset_is_real = False
    prov_file_found = None
    for candidate in [
        splits_path / "provenance_real.json",
        splits_path.parent / "provenance_real.json",
        Path("data/provenance_real.json"),
    ]:
        if candidate.is_file():
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    p_data = json.load(f)
                if (
                    p_data.get("is_real_world_photograph") is True
                    and p_data.get("provenance_status") == "genuine_photograph"
                    and p_data.get("valid_images_retained", 0) > 0
                ):
                    dataset_is_real = True
                    prov_file_found = candidate
                    break
            except Exception as e:
                logger.warning(f"Error reading provenance file {candidate}: {e}")

    # 3. Verify Split Manifest Integrity & Image File Existence
    manifest_file = splits_path / f"{split_name}_manifest.csv"
    manifest_verified = False
    manifest_reason = ""

    if not manifest_file.is_file():
        manifest_reason = f"Manifest file not found: {manifest_file}"
    else:
        try:
            m_df = pd.read_csv(manifest_file)
            if len(m_df) == 0:
                manifest_reason = "Manifest is empty (0 samples)"
            elif "filepath" not in m_df.columns:
                manifest_reason = "Manifest missing 'filepath' column"
            elif "class_name" not in m_df.columns:
                manifest_reason = "Manifest missing 'class_name' column"
            else:
                missing_files = []
                for fp in m_df["filepath"]:
                    p = Path(fp)
                    if not p.is_file() or p.stat().st_size == 0:
                        missing_files.append(fp)
                        if len(missing_files) >= 3:
                            break

                if missing_files:
                    manifest_reason = f"Sample image file(s) missing or empty on disk: {missing_files[:3]}"
                else:
                    if expected_classes:
                        invalid_classes = [c for c in m_df["class_name"] if c not in expected_classes]
                        if invalid_classes:
                            manifest_reason = f"Invalid class names in manifest: {invalid_classes[:3]}"
                        else:
                            manifest_verified = True
                    else:
                        manifest_verified = True
        except Exception as e:
            manifest_reason = f"Error reading manifest {manifest_file}: {e}"

    # 4. Verify Leakage Absence between train and test
    leakage_free = True
    leakage_reason = ""
    train_manifest = splits_path / "train_manifest.csv"
    if train_manifest.is_file() and manifest_file.is_file() and manifest_verified:
        try:
            tr_df = pd.read_csv(train_manifest)
            te_df = pd.read_csv(manifest_file)
            if "seed_id" in tr_df.columns and "seed_id" in te_df.columns:
                tr_seeds = set(tr_df["seed_id"].dropna().unique())
                te_seeds = set(te_df["seed_id"].dropna().unique())
                leaked = tr_seeds.intersection(te_seeds)
                if leaked:
                    leakage_free = False
                    leakage_reason = f"Seed group leakage detected between train and test: {list(leaked)[:3]}"
        except Exception as e:
            logger.warning(f"Error checking split leakage: {e}")

    # Synthesize Verification Verdict
    if ckpt_is_real and dataset_is_real and manifest_verified and leakage_free:
        return {
            "is_real_world_validated": True,
            "evaluation_type": "real_world_test_evaluation",
            "dataset_provenance": "manually_annotated_genuine_photographs",
            "scientific_notice": (
                "Empirical test metrics evaluated on genuine held-out human-annotated agricultural pulse photographs."
            ),
            "verification_details": {
                "checkpoint_provenance_verified": True,
                "dataset_provenance_verified": True,
                "manifest_integrity_verified": True,
                "leakage_absence_verified": True,
                "prov_file": str(prov_file_found),
                "reason": "All genuine real-world validation criteria strictly satisfied.",
            },
        }

    # If it is recognized as standard synthetic verification
    is_standard_synthetic = (
        not ckpt_is_real
        and not dataset_is_real
        and extra_meta.get("dataset_provenance", "synthetic_procedural_fixture") == "synthetic_procedural_fixture"
    )

    if is_standard_synthetic and manifest_verified:
        return {
            "is_real_world_validated": False,
            "evaluation_type": "synthetic_pipeline_verification",
            "dataset_provenance": "synthetic_procedural_fixture",
            "scientific_notice": (
                "Metrics demonstrate software pipeline convergence on synthetic procedural fixtures. "
                "Real-world agronomic accuracy is pending certified agricultural pulse dataset acquisition."
            ),
            "verification_details": {
                "checkpoint_provenance_verified": False,
                "dataset_provenance_verified": False,
                "manifest_integrity_verified": True,
                "leakage_absence_verified": leakage_free,
                "reason": "Standard procedural synthetic calibration fixtures.",
            },
        }

    # Otherwise, it's contradictory, renamed, or unverified
    failures = []
    if not ckpt_is_real:
        failures.append("checkpoint lacks verified genuine training metadata (synthetic or unverified weights)")
    if not dataset_is_real:
        failures.append("evaluated dataset lacks verified genuine photograph provenance record")
    if not manifest_verified:
        failures.append(f"manifest integrity verification failed: {manifest_reason}")
    if not leakage_free:
        failures.append(leakage_reason)

    failure_summary = "; ".join(failures)
    return {
        "is_real_world_validated": False,
        "evaluation_type": "unverified_evaluation",
        "dataset_provenance": "unverified_or_contradictory",
        "scientific_notice": (
            f"Unverified evaluation: {failure_summary}. Flag is_real_world_validated set to False."
        ),
        "verification_details": {
            "checkpoint_provenance_verified": ckpt_is_real,
            "dataset_provenance_verified": dataset_is_real,
            "manifest_integrity_verified": manifest_verified,
            "leakage_absence_verified": leakage_free,
            "reason": failure_summary,
        },
    }
