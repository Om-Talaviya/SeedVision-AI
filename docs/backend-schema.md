# SeedVision AI: Backend Data & Artifact Schemas

This document formally specifies the schemas, data structures, and serializations utilized across the SeedVision AI platform.

---

## 1. Configuration Schema (`configs/*.yaml`)

```yaml
project:
  name: str              # e.g., "SeedVision AI"
  version: str           # Semantic version, e.g., "1.0.0"
  description: str       # High-level project summary

dataset:
  seed_type: str         # Target pulse crop, e.g., "Chickpea"
  raw_dir: str           # Path to un-split raw images
  processed_dir: str     # Path for generated manifests and splits
  manifest_path: str     # Path to unified dataset manifest CSV
  splits_dir: str        # Directory containing train/val/test CSVs
  classes: list[str]     # Discrete quality class identifiers
  class_descriptions: map[str, str] # Agronomic class definitions
  split_ratios:
    train: float         # e.g., 0.70
    val: float           # e.g., 0.15
    test: float          # e.g., 0.15
  image_size: [int, int] # Height, Width (e.g. [128, 128])
  normalization:
    mean: list[float]    # RGB mean channels
    std: list[float]     # RGB std channels
  num_workers: int       # DataLoader worker count (0 for Windows)

augmentation:
  enabled: bool
  horizontal_flip: bool
  vertical_flip: bool
  rotation_degrees: int
  brightness_contrast: bool
  brightness_factor: float
  contrast_factor: float

model:
  architecture: str      # "custom_cnn" | "mobilenet_v2"
  name: str              # Identifier string, e.g. "SeedNet-Baseline"
  num_classes: int       # Number of output classes (4)
  input_channels: int    # 3 for RGB
  conv_channels: list[int] # [32, 64, 128, 256]
  dropout_rate: float    # 0.35
  dense_units: int       # 256
  target_layer: str      # Submodule name for Grad-CAM hooks
  pretrained: bool       # For transfer learning backbones
  freeze_backbone: bool

training:
  epochs: int            # Total epochs (e.g. 15 or 25)
  batch_size: int        # Mini-batch size (e.g. 16)
  learning_rate: float   # Optimizer learning rate
  optimizer: str         # "adam" | "adamw"
  weight_decay: float    # L2 regularization factor
  scheduler: str         # "cosine"
  min_lr: float
  early_stopping_patience: int
  device: str            # "cpu" | "cuda" | "auto"
  seed: int              # Deterministic random seed (42)

paths:
  checkpoints_dir: str   # artifacts/checkpoints
  metrics_dir: str       # artifacts/metrics
  figures_dir: str       # artifacts/figures
```

---

## 2. Dataset Split Manifest Schema (`.csv`)

| Column Name | Type | Description | Example |
| :--- | :---: | :--- | :--- |
| `filepath` | String | Absolute path to validated image file | `/data/raw/sound_healthy/sound_healthy_001.jpg` |
| `filename` | String | Base image file name | `sound_healthy_001.jpg` |
| `class_name` | String | Target agricultural quality class | `sound_healthy` |
| `seed_id` | String | Unique physical seed entity identifier | `sound_healthy_001` |
| `width` | Integer | Decoded image width in pixels | `128` |
| `height` | Integer | Decoded image height in pixels | `128` |
| `file_size_bytes` | Integer | File size on storage | `3601` |
| `split` | String | Dataset partition assignment | `train` \| `val` \| `test` |

---

## 3. Checkpoint Metadata Schema (`*.pt.meta.json`)

Saved alongside every PyTorch `.pt` file for fast introspection without loading tensor weights:

```json
{
  "epoch": 12,
  "architecture": "custom_cnn",
  "model_name": "SeedNet-Baseline",
  "val_loss": 0.32145,
  "val_acc": 1.0,
  "is_best": true,
  "classes": [
    "sound_healthy",
    "discolored_spotted",
    "broken_cracked",
    "shrivelled_damaged"
  ],
  "num_classes": 4,
  "image_size": [128, 128],
  "normalization": {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225]
  },
  "saved_at": "2026-10-14T16:53:51.123456+00:00",
  "parameters": {
    "total_parameters": 456228,
    "trainable_parameters": 456228,
    "frozen_parameters": 0,
    "estimated_size_mb": 1.74
  }
}
```

---

## 4. Model Evaluation Schema (`eval_*.json`)

```json
{
  "model_name": "SeedNet-Baseline",
  "num_test_samples": 36,
  "accuracy": 1.0,
  "macro_precision": 1.0,
  "macro_recall": 1.0,
  "macro_f1": 1.0,
  "expected_calibration_error": 0.0251,
  "latency_ms_per_sample": 5.72,
  "p95_latency_ms": 7.15,
  "throughput_samples_per_sec": 174.8,
  "parameters": {
    "total_parameters": 456228,
    "trainable_parameters": 456228,
    "frozen_parameters": 0,
    "estimated_size_mb": 1.74
  },
  "classes": [
    "sound_healthy",
    "discolored_spotted",
    "broken_cracked",
    "shrivelled_damaged"
  ],
  "per_class_metrics": {
    "sound_healthy": { "precision": 1.0, "recall": 1.0, "f1_score": 1.0, "support": 9 },
    "discolored_spotted": { "precision": 1.0, "recall": 1.0, "f1_score": 1.0, "support": 9 },
    "broken_cracked": { "precision": 1.0, "recall": 1.0, "f1_score": 1.0, "support": 9 },
    "shrivelled_damaged": { "precision": 1.0, "recall": 1.0, "f1_score": 1.0, "support": 9 }
  },
  "confusion_matrix": [[9, 0, 0, 0], [0, 9, 0, 0], [0, 0, 9, 0], [0, 0, 0, 9]],
  "confusion_matrix_normalized": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
  "calibration_bins": {
    "accuracies": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    "confidences": [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.975],
    "counts": [0, 0, 0, 0, 0, 0, 0, 0, 0, 36]
  }
}
```

---

## 5. Live Inference Prediction Result Schema (`PredictionResult`)

```python
class PredictionResult:
    predicted_class: str             # e.g., "broken_cracked"
    confidence_score: float          # e.g., 0.9915 (0.0 to 1.0)
    is_uncertain: bool               # True if confidence < 0.50 or margin < 0.15
    uncertainty_reason: str | None   # Explanatory warning message if uncertain
    top_k: list[TopKPrediction]      # Ranked array with rank, class, probability, string
    raw_probabilities: dict[str, float] # Unsorted class name -> float mapping
    inference_latency_ms: float      # Latency in ms for forward pass
    model_name: str                  # e.g. "SeedNet-Baseline"
    architecture: str                # e.g. "custom_cnn"
    heatmap_2d: np.ndarray | None    # 2D float array [H, W] normalized [0, 1]
    overlay_rgb: np.ndarray | None   # 3D uint8 array [H, W, 3] blended with original
    colored_heatmap_rgb: np.ndarray | None # Colormapped heatmap
    gradcam_target_class: str | None # Class hypothesis explained
    gradcam_error: str | None        # Error message if Grad-CAM could not be computed
    scientific_notice: str           # Mandatory disclaimer on optical limits
```
