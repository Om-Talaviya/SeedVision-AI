# Product Requirements Document (PRD) — SeedVision AI

## 1. Document Control
- **Product Name**: SeedVision AI
- **Document Version**: 1.0.0
- **Document Status**: Approved / Baseline Implemented
- **Owner**: SeedVision AI Product & Engineering Team
- **Target Audience**: Agricultural researchers, grain quality inspectors, seed supply chain managers, academic evaluators.

---

## 2. Product Vision & Problem Statement

### 2.1 The Problem
In commercial pulse trade (specifically Chickpeas, *Cicer arietinum*), grain lots are priced based on physical quality grades established by agricultural authorities (e.g. AGMARK, USDA GIPSA). Manual inspection is:
- **Subjective**: High inter-grader variance depending on fatigue, lighting, and operator bias.
- **Unscalable**: Sample inspection of thousands of seeds per truckload is slow.
- **Opaque**: Lacks visual audit trails to justify lot downgrades or rejections to farmers.

### 2.2 The Solution
SeedVision AI provides an automated, explainable computer vision system that:
1. Validates and preprocesses single-seed macro photographs.
2. Predicts physical quality grades across 4 canonical defect categories.
3. Quantifies predictive confidence and raises actionable uncertainty alerts on borderline samples.
4. Generates visual Grad-CAM overlays isolating the exact physical defect (crack, lesion, shrivelled skin).
5. Provides a clear scientific distinction: optical appearance vs physiological seed viability.

---

## 3. User Personas

| Persona | Primary Goal | Critical Need in SeedVision AI |
| :--- | :--- | :--- |
| **Dr. Ananya (Pulse Researcher)** | Evaluate visual defect patterns across harvest seasons. | Statistical metrics (Macro F1, ECE), reproducible splits, confusion matrices. |
| **Rajesh (Grain Procurement Officer)** | Rapidly inspect incoming seed bags at procurement hubs. | Fast CPU inference ($< 15\text{ ms}$), simple web UI, clear grade badges. |
| **Priya (DLA Course Evaluator / Professor)** | Verify that deep learning models, Grad-CAM, and data splits are authentic. | Real training history, clean modular code, no synthetic benchmark fabrication. |

---

## 4. Functional Requirements

### FR-1: Image Acquisition & Ingestion
- **FR-1.1**: Accept uploaded images in `.jpg`, `.jpeg`, `.png`, `.webp`, and `.bmp` formats up to 10 MB.
- **FR-1.2**: Reject zero-byte, corrupted, or undersized images ($< 32 \times 32\text{ px}$) with descriptive user-facing errors.
- **FR-1.3**: Provide a built-in repository of verified demo samples for zero-friction demonstrations.

### FR-2: Deep Learning Quality Classification
- **FR-2.1**: Support 4 discrete quality classes: `sound_healthy`, `discolored_spotted`, `broken_cracked`, `shrivelled_damaged`.
- **FR-2.2**: Return top-1 predicted class and top-$k$ Softmax probability distribution.
- **FR-2.3**: Compare at least two trained architectures: Custom Baseline CNN (`SeedNet-Baseline`) and Fine-Tuned `MobileNetV2`.

### FR-3: Calibrated Uncertainty Assessment
- **FR-3.1**: Flag predictions as `is_uncertain = True` when top-1 confidence is $< 50\%$.
- **FR-3.2**: Flag ambiguous predictions when the difference between rank 1 and rank 2 probabilities is $< 15\%$.

### FR-4: Explainable AI with Grad-CAM
- **FR-4.1**: Compute activation heatmaps at the final convolutional feature layer for the predicted class.
- **FR-4.2**: Allow users to interactively generate Grad-CAM explanations for alternative candidate classes.
- **FR-4.3**: Render a 3-image comparison triplet: Original image, Normalized Heatmap, and Blended Overlay ($45\%$ alpha).

### FR-5: Model Evaluation & Benchmarking
- **FR-5.1**: Evaluate both checkpoints on an identical held-out test split (15% partition).
- **FR-5.2**: Display Accuracy, Macro F1, Per-class Precision/Recall/F1, Confusion Matrix, Expected Calibration Error (ECE), and CPU Latency.

---

## 5. Non-Functional Requirements

- **NFR-1 (Performance)**: CPU inference latency must remain $< 25\text{ ms}$ per sample on modern standard student hardware.
- **NFR-2 (Portability)**: Zero mandatory GPU dependencies; full execution supported on CPU with automatic CUDA detection.
- **NFR-3 (Reproducibility)**: Deterministic random seeds (`seed = 42`) for training and data splitting.
- **NFR-4 (Accessibility)**: Adhere to WCAG 2.1 AA contrast ratios ($> 4.5:1$) across all Streamlit UI components.
- **NFR-5 (Scientific Integrity)**: Permanent disclaimer clarifying that optical classification does not certify germination or internal embryo viability.

---

## 6. Success Metrics & KPIs
1. **Model Performance**: Test Macro F1-score $\ge 90\%$ on balanced test dataset.
2. **Explainability Validity**: Grad-CAM heatmaps localize the physical defect on cracked, spotted, and shrivelled samples.
3. **Inference Speed**: Single-image prediction with Grad-CAM completed in $< 100\text{ ms}$ end-to-end on CPU.
4. **Test Reliability**: $100\%$ pass rate on the 24-case automated test suite.
