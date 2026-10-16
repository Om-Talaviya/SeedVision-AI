# Changelog — SeedVision AI

All notable changes to the SeedVision AI project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Fixed
- **Streamlit UI Readability & Contrast (WCAG 2.1 AA/AAA Compliance)**:
  - Created `.streamlit/config.toml` configuring official Streamlit theme tokens (`primaryColor: #2e7d32`, `backgroundColor: #f8fafc`, `secondaryBackgroundColor: #ffffff`, `textColor: #0f172a`).
  - Removed unscoped hardcoded `background-color: #ffffff` on `section[data-testid="stSidebar"]` that previously caused white-on-white text in dark mode and low contrast on radio buttons and headers.
  - Implemented theme-adaptive CSS rules in `app.py` for sidebar headings, radio navigation labels, selected navigation state, and model checkpoints.
  - Added `@media (prefers-color-scheme: dark)` overrides ensuring accessible contrast across both light and dark modes without hardcoded clashing colors.
  - Converted inline styles for quality class descriptions, notices, and tags to semantic CSS classes (`feature-card-desc`, `sidebar-notice-box`, `sidebar-tag-real`, `sidebar-tag-synth`, `benchmark-notice-*`, `safeguard-notice`).
- **Streamlit Top Header Spacing & Layout Optimization**:
  - Overrode default Streamlit `.block-container` / `[data-testid="stMainBlockContainer"]` top padding from `6rem` (96px) down to `1.5rem` (24px), eliminating dead blank vertical space above the hero banner.
  - Constrained top header height (`header[data-testid="stHeader"]`) to `2.25rem` with transparent background and non-blocking pointer events (`pointer-events: none` on header container, `pointer-events: auto` on interactive children).
  - Cleaned up top-right navigation by hiding the deployment promotion widget (`[data-testid="stDeployButton"]`) while preserving full accessibility of hamburger settings, sidebar toggles, and main controls.
  - Optimized `.hero-card` dimensions (`padding: 1.75rem 2.25rem`, `margin-top: 0 !important`, `margin-bottom: 1.5rem`) and ensured second-page `<h2>` titles have zero top margin.
  - Added responsive viewport rules (`@media (max-width: 768px)`) for compact mobile/tablet viewports.
  - Added `test_app_header_and_main_padding_optimization` to `tests/test_ui_contrast_and_theme.py` (99/99 tests passing repository-wide).

### Added

- **Human-Review Workflow for Real Data Staging**:
  - Implemented core review and specimen-grouping module `src/seedvision/data/real_review.py`:
    - `load_staging_manifest`: Groups photos by `specimen_id`, verifies image existence and dimensions ($W \times H$ px).
    - `validate_review_decision`: Enforces SeedVision 4-class taxonomy + `pending_review`, mandates reviewer identity and diagnostic notes on approval, and strictly preserves `pending` status for low-confidence or ambiguous specimens.
    - `save_specimen_review`: Applies uniform specimen-level labels across all views of a physical seed with automatic backup creation (`staging_manifest.backup.csv`) and atomic file replacement.
    - `revert_specimen_review` & `restore_manifest_backup`: Provides one-click specimen reset and complete manifest undo capabilities.
  - Integrated dedicated **"Real Data Staging Review"** page into the Streamlit laboratory (`app.py`):
    - Displays high-level specimen and view metrics.
    - Specimen selector with status badges and view counts.
    - Multi-view gallery displaying all photographic angles side-by-side.
    - Human review form with target defect selection, confidence rating, reviewer identity, and diagnostic notes.
    - Specimen revert button and manifest restore-from-backup control with confirmation safeguards.
  - Created CLI review interface `scripts/review_staging_specimens.py` supporting `--list`, `--specimen-id`, `--label`, `--status`, `--confidence`, `--reviewer`, `--notes`, `--revert`, `--restore-backup`, and `--dry-run`.
  - Added unit test suite `tests/test_real_review_workflow.py` (14 new test cases covering validation, multi-view consistency, backup creation, and rollback; 92 total tests passing across repository).

  - Upgraded `scripts/build_roboflow_review_gallery.py` with client-side pagination, batch navigation (200 crops/batch), dynamic source-class and review-status filters, search, and crop dimensions display (`width x height`, aspect ratio, low-res indicator).
  - Implemented `src/seedvision/data/roboflow_review.py` providing `get_crop_dimensions`, `generate_crop_metadata`, and `triage_foreign_material`.
  - Generated `crop_metadata.csv` recording dimensions (`width`, `height`, `aspect_ratio`), low-resolution triage flags (<28px), while strictly preserving source annotations, splits, and grouping.
  - Implemented reversible, auditable, and idempotent triage (`scripts/triage_roboflow_crops.py`):
    - Confirmed non-pulse foreign matter (`Material extrano inorganico` and `Material extrano organico`, totaling 2,028 crops) and triaged as `rejected` with reason `foreign_material`.
    - Zero auto-mapping of Roboflow classes to SeedVision's four target classification labels.
    - Low-resolution crops (<28px) are flagged for human review and never auto-rejected.
    - Preserved automatic pre-update backup (`manual_review.backup.csv`).
    - Maintained all 28,428 remaining rows in `pending` review status.
  - Enriched `scripts/summarize_roboflow_review.py` with resolution distribution statistics and triage audit metrics.
  - Added on-card human verification forms and CSV decision export to `scripts/build_roboflow_review_gallery.py`.
  - Implemented `scripts/apply_manual_review.py` and core validators (`validate_human_decision`, `apply_manual_review_decisions`, `revert_manual_review_decisions`) with strict 4-class target validation, mandatory reviewer rationale, and pre-modification backup.
  - Expanded automated test suite to 64 passing tests across the repository.

## [1.4.0] - 2026-10-15

### Added
- **Multi-Gate Provenance Verification Engine (`provenance_verifier.py`)**:
  - Implemented 4-stage strict verification gate (`verify_evaluation_provenance`) requiring checkpoint training metadata, authentic dataset provenance record, physical image existence, and zero split leakage.
  - Eliminated naive substring heuristics (`"real" in model_name`), guaranteeing that renamed synthetic checkpoints evaluate as `is_real_world_validated = False` and `evaluation_type = "unverified_evaluation"`.
- **Safe In-Place Real Data Ingestion**:
  - Enhanced `scripts/ingest_real_data.py` with automatic in-place mode detection (`source_dir == target_dir`), disabling file duplication while enforcing leakage-resistant splits and duplicate auditing.
  - Added strict directory nesting checks preventing `splits_dir` or `target_dir` from being nested inside `source_dir`.
  - Added hash collision protection preventing overwrite of modified destination images during staging.
- **Automated Regression Suite Expansion (42 Tests)**:
  - Added `test_renamed_synthetic_checkpoint_cannot_be_marked_real_world_validated` proving renamed synthetic models cannot be marked real-world validated.
  - Added `test_in_place_ingestion_safety_and_nested_prevention` verifying in-place auditing and nested directory rejection.
  - Added `test_compare_models_empty_and_populated` and `test_model_evaluator_complete_flow` (42/42 tests passing with 73% package coverage).
- **Packaging & Dependency Harmonization**:
  - Synchronized `pyproject.toml` dependencies with `requirements.txt` (added `scipy` and `pydantic`).
  - Added `data/real_raw/**/*.jpg` and test outputs to `.gitignore` to prevent large binary dumps or temporary files from git tracking.

---

## [1.3.0] - 2026-10-14

### Added
- **Final Pre-Training Audit & Statutory Standards Review**:
  - Delineated official agricultural grain inspection standards (AGMARK India Pulses Rules 2004, USDA GIPSA Garbanzo Bean Standards) based on bulk sample weight percentages from project-defined single-seed operational CV proxies (1.5mm crack, 5% necrotic area).
  - Documented independent dual-annotator scoring protocol, Cohen's Kappa target ($\kappa \ge 0.85$), and Senior Agronomist arbitration in `docs/REAL_DATA_ACQUISITION_WORKFLOW.md`.
- **Independent Annotation Manifest Ingestion (`annotations.csv`)**:
  - Added support for `--annotations` in `scripts/ingest_real_data.py`, enabling strict ingestion of human-reviewed annotations.
  - Implemented automatic exclusion of `ambiguous_discard` labels and low-confidence entries from training/validation/test splits.
  - Implemented strict filtering of unreviewed images on disk to prevent unlabeled data leakage.
  - Grouped multi-view captures sharing identical `seed_id` across distinct filenames into the same partition.
- **Evaluation Provenance Isolation**:
  - Parameterized `ModelEvaluator` and `scripts/evaluate.py` to auto-tag real-data evaluation runs (`evaluation_type: real_world_test_evaluation`, `is_real_world_validated: True`).
  - Updated `compare_models` and `app.py` to render distinct green status badges for genuine real-world evaluations and blue badges for synthetic verification fixtures.
- **Automated Regression Suite Expansion**:
  - Added 5 new unit tests in `tests/test_real_data_ingestion.py` covering `annotations.csv` ingestion, ambiguous discard filtering, unreviewed image skipping, and evaluator metadata isolation (38/38 tests passing).
- **GitHub Publication Preparation**:
  - Configured repository metadata, GitHub Actions CI workflow, and verified zero secret leakage.

---

## [1.2.0] - 2026-10-14

### Added
- **Public Pulse Dataset Due Diligence**:
  - Investigated publicly available chickpea and pulse datasets across Mendeley Data (`PulseLibrary` DOI `10.17632/kw7x5xyprh.1`, `Cicer Arietinum Germination`), GitHub (`TRCS_8_SET`, `SOYPR`), and Kaggle (`Fusarium Wilt`).
  - Documented forensic finding: public repositories classify by pulse species or botanical variety, but no ready-to-use dataset exists pre-annotated with the 4 physical defect classes.
- **Genuine Seed Photography Ingestion Engine (`scripts/ingest_real_data.py`)**:
  - Implemented automated ingestion CLI supporting image integrity validation, perceptual average hashing (aHash), cryptographic MD5 duplicate purging, and group-aware leakage-resistant splitting.
  - Implemented `--init-structure` scaffolding generating `data/real_raw/` with class directories, annotation template CSV, and laboratory capture guidelines.
  - Generates `data/provenance_real.json` recording source metadata, license, and class balance.
- **Training & Evaluation Flexibility**:
  - Added `--splits-dir` argument to both `scripts/train.py` and `scripts/evaluate.py`, enabling immediate training and evaluation on genuine pulse splits (`data/processed/splits_real/`) without editing configuration files.
  - Added `--name-suffix` argument to `scripts/train.py` for tagging real-data model checkpoints (e.g. `SeedNet-Baseline-Real`).
- **Automated Ingestion Test Suite**:
  - Created `tests/test_real_data_ingestion.py` covering directory scaffolding, perceptual hash consistency, and duplicate purging (33/33 tests passing).

---

## [1.1.0] - 2026-10-14

### Added
- **Dataset Provenance & Cryptographic Audit**:
  - Conducted full forensic provenance audit of 240 images in `data/raw/`, formally establishing their status as procedural synthetic fixtures generated by PIL algorithms for software testing (`docs/DATASET_AUDIT.md` and `data/provenance.json`).
  - Cryptographic hash audit confirming 240 unique MD5 hashes, zero duplicates, and zero cross-partition split leakage across train, val, and test manifests.
  - Automated provenance test suite in `tests/test_provenance.py` covering provenance metadata integrity, split leakage absence, file existence, and checkpoint consistency.
  - New test `test_gradcam_mobilenetv2_support` in `tests/test_gradcam.py` confirming Grad-CAM hook management on MobileNetV2 architectures (30 total tests passing).
- **Scientific Validation & Transparency**:
  - Explicitly separated "Software Pipeline Verification on Synthetic Calibration Fixtures (100% Convergence)" from "Real-World Agronomic Benchmarks (Pending Certified Pulse Collection)".
  - Added statistical Wilson confidence interval analysis for small sample size evaluation (36-sample test split).
  - Enriched evaluation metrics JSON and CSV summaries with `evaluation_type: synthetic_pipeline_verification` and scientific notice fields.
- **Web Application Refinement**:
  - Added prominent Software Verification Mode banners in `app.py` across sidebar, Home page, and Model Evaluation benchmark tab.
  - Added dedicated 5th tab to Methodology & Limitations explaining dataset provenance and providing step-by-step instructions for researchers to ingest genuine agricultural pulse photographs.

---

## [1.0.0] - 2026-10-14

### Added
- **Core Deep Learning Models**:
  - `CustomSeedCNN` (`SeedNet-Baseline`): 4-stage convolutional hierarchy with Batch Normalization, LeakyReLU, and Dropout-regularized classification head (456K parameters).
  - `SeedTransferModel` (`MobileNetV2-Transfer`): Fine-tuned inverted residual bottleneck backbone with configurable layer unfreezing (2.39M parameters).
- **Explainability**:
  - Native `GradCAM` implementation in `src/seedvision/explainability/gradcam.py` supporting forward/backward hook registration, gradient pooling, ReLU attribution, and alpha overlay generation.
- **Dataset Pipeline**:
  - Comprehensive chickpea defect dataset (240 samples, 60 per class across `sound_healthy`, `discolored_spotted`, `broken_cracked`, and `shrivelled_damaged`).
  - `ImageValidator` with corruption detection, dimension limits, and SHA-256 content hashing to eliminate duplicates.
  - Leakage-resistant class-stratified group splitting algorithm in `src/seedvision/data/splitting.py`.
- **Training Engine**:
  - `SeedTrainer` with CrossEntropyLoss label smoothing ($\epsilon=0.05$), Cosine Annealing learning rate schedule, gradient norm clipping ($\le 1.0$), and early stopping.
  - `CheckpointManager` generating atomic `.pt` checkpoints and `.pt.meta.json` inspection companions.
- **Scientific Evaluation**:
  - Evaluation suite computing Accuracy, Macro Precision, Macro Recall, Macro F1, Per-Class metrics, Confusion Matrix, Expected Calibration Error (ECE), and CPU Latency.
  - Matplotlib plotting suite for dual-axis training curves, confusion matrix heatmaps, per-class bars, and reliability diagrams.
  - Cross-architecture benchmark comparison tool (`compare_models`).
- **Interactive Web Application**:
  - 4-Page Streamlit dashboard (`app.py`) on port 8501 featuring Home, Seed Quality Analysis, Model Evaluation benchmarks, and Methodology documentation.
  - Real-time demo sample loader and interactive Grad-CAM class hypothesis selector.
- **Automated Testing & CI/CD**:
  - 24-test pytest suite passing with 100% success rate on CPU.
  - GitHub Actions CI workflow in `.github/workflows/tests.yml`.
- **Governance & Documentation**:
  - Comprehensive PRD, TRD, Architecture Blueprint, Backend Data Schemas, Flow Sequence Diagrams, ADRs, UI/UX Design System, and College Presentation Viva Guide.

### Changed
- Standardized `seed_id` granularity to avoid group collapsing during dataset partitioning.
- Refactored `CheckpointManager` to guarantee native boolean casting for JSON metadata serialization in Python 3.13.

### Fixed
- Resolved duplicate detection false-positives in synthetic unit test fixtures by generating unique RGB offsets per class.
- Fixed raw string escape sequences in Streamlit methodology tab for LaTeX formulas.
