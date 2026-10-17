# SeedVision AI: Actionable Roadmap & TODO

## 🎯 Current Milestone: Phase 1 — Core Pulse Grader & Explainability (Completed ✅)
- [x] High-fidelity chickpea morphological dataset preparation and validation.
- [x] Leakage-resistant class-stratified train/val/test splitting pipeline.
- [x] Custom CNN architecture (`SeedNet-Baseline`) with 4-stage conv hierarchy.
- [x] Transfer learning architecture (`MobileNetV2-Transfer`) with fine-tuning head.
- [x] Training pipeline with Cosine Annealing, Early Stopping, and checkpointing.
- [x] Scientific evaluation suite (Accuracy, Macro F1, Confusion Matrix, ECE, CPU Latency).
- [x] Native Grad-CAM engine with forward/backward hook management and overlay blending.
- [x] 4-Page modern laboratory Streamlit web application.
- [x] 24/24 Automated pytest suite passing on CPU.
- [x] Git repository initialization and commit history.

## 🔬 Dataset Audit, Scientific Validation & Transparency (Completed ✅)
- [x] Forensic dataset audit confirming 240 synthetic procedural fixtures (`docs/DATASET_AUDIT.md` & `data/provenance.json`).
- [x] Cryptographic hash audit: 240 unique MD5 hashes, 0 duplicates, 0 split overlap across train/val/test partitions.
- [x] 5 new automated unit tests in `tests/test_provenance.py` + MobileNetV2 Grad-CAM test (30/30 total tests passing).
- [x] Separation of synthetic smoke-test verification from real-world empirical agronomic benchmarks in `README.md` and `docs/MODEL_EVALUATION.md`.
- [x] Streamlit web application audit with research verification mode disclaimers and real-data ingestion protocol tab.
- [x] Enriched evaluation artifacts (`artifacts/metrics/*.json`, `model_comparison_summary.csv`) with provenance and validation status.

## 🌾 Real-World Dataset Acquisition & Ingestion Pipeline (Completed ✅)
- [x] Exhaustive search of public pulse datasets (Mendeley `PulseLibrary`, GitHub `TRCS_8_SET`, `SOYPR`, Kaggle).
- [x] Verification finding: public chickpea datasets categorize by species or variety, not by 4-class physical defect quality.
- [x] Automated genuine seed photography ingestion pipeline in `scripts/ingest_real_data.py` (integrity checks, MD5 & aHash deduplication, leakage-resistant group splitting).
- [x] Independent annotation manifest ingestion (`annotations.csv`) with automatic exclusion of `ambiguous_discard` and low-confidence entries.
- [x] Statutory standards review: explicitly delineated AGMARK India & USDA GIPSA bulk lot mass tolerances from project-defined single-seed operational CV proxies.
- [x] Evaluator provenance tagging: automatic dynamic isolation of real-world test evaluation metrics (`eval_*-Real.json`) from synthetic fixtures.
- [x] Directory scaffold and annotation guide initialized under `data/real_raw/` with `annotations_template.csv`.
- [x] Added `--splits-dir` and `--name-suffix` flags to `scripts/train.py` and `scripts/evaluate.py` for decoupled real-data training.
- [x] Multi-Gate Provenance Verification Engine (`provenance_verifier.py`) preventing false validation on renamed checkpoints or unverified splits.
- [x] 10 specialized real-data unit tests in `tests/test_real_data_ingestion.py` and evaluation suite (99/99 total tests passing).
- [ ] User/Researcher upload of physical chickpea photos into `data/real_raw/` for final real-data benchmark execution.

---

## 🚀 Phase 2: Enhanced Computer Vision & Multi-Seed Processing (In Progress / Up Next)
- [ ] **Multi-Seed Tray Segmentation**:
  - Implement a YOLOv8 / Faster-RCNN seed detection stage to crop and grade multiple seeds from a single high-resolution tray photo simultaneously.
  - Generate automated batch quality distribution reports (e.g., "Seed Lot A: 88% Sound, 8% Cracked, 4% Shrivelled").
- [ ] **Test-Time Augmentation (TTA)**:
  - Add optional 4-fold rotation test-time averaging during inference to further suppress confidence variance on borderline angles.
- [ ] **Temperature Scaling for Probability Calibration**:
  - Implement Platt scaling / temperature scaling on validation set logits to push Expected Calibration Error (ECE) below 0.015.

---

## 🔬 Phase 3: Hardware Integration & Spectral Analysis (Long-Term Scope)
- [ ] **Near-Infrared (NIR) / Multi-Spectral Fusion**:
  - Integrate multi-spectral reflectance sensor streams (700nm–1050nm) to assess internal moisture content and biochemical embryo viability alongside optical RGB features.
- [ ] **Edge Acceleration & Quantization**:
  - Convert trained PyTorch models to ONNX Runtime and INT8 quantized TensorRT / OpenVINO formats for sub-2ms edge execution on Raspberry Pi 5 / NVIDIA Jetson Orin Nano.
- [ ] **Automated Sorting Chute Integration**:
  - Implement GPIO / serial relay trigger interface for pneumatic sorting nozzles on mechanical grain sorting conveyor belts.
- [ ] **Expanded Crop Taxonomy**:
  - Add specialized fine-tuned checkpoints for Green Gram (*Vigna radiata*), Black Gram (*Vigna mungo*), and Lentils (*Lens culinaris*).
