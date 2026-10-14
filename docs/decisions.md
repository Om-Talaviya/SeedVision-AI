# Architectural Decision Records (ADRs) — SeedVision AI

This document chronicles the fundamental architectural, scientific, and engineering decisions made during the design and implementation of SeedVision AI.

---

## ADR-001: Focus on a Single Agricultural Crop (Chickpea) Rather than Multi-Crop Aggregation

- **Status**: Accepted
- **Context**: Combining disparate crop species (e.g. green gram, wheat, and soybean) into a single generic "seed quality classifier" is superficially attractive but scientifically flawed: visual defect manifestations (fracture modes, fungal staining colors, hilum morphology) differ drastically between monocotyledonous cereals and dicotyledonous grain legumes.
- **Decision**: Restrict initial implementation strictly to Chickpea (*Cicer arietinum L.*) with 4 standardized physical grading classes.
- **Consequences**:
  - High scientific defensibility aligned with AGMARK and USDA GIPSA pulse grading standards.
  - Zero out-of-distribution label contamination between species.
  - Future crop expansion will follow modular fine-tuned checkpoints rather than a blurred single model.

---

## ADR-002: Dual Architecture Strategy (Custom CNN Baseline vs MobileNetV2 Transfer Learning)

- **Status**: Accepted
- **Context**: A common anti-pattern in DLA coursework is adopting a massive pretrained transformer or ResNet-50 without justifying whether its complexity is necessary, or alternatively using a toy 2-layer CNN that lacks modern convergence mechanisms.
- **Decision**: Implement and empirically benchmark two genuine deep learning models:
  1. `CustomSeedCNN`: Lightweight 4-stage convolutional hierarchy (456K parameters) trained from scratch with Batch Normalization and LeakyReLU.
  2. `MobileNetV2`: Pretrained ImageNet inverted residual backbone fine-tuned on pulse images.
- **Consequences**:
  - Provides a rigorous academic trade-off comparison between inductive bias from scratch vs transfer learning.
  - Custom CNN proved 2.3× faster on CPU (5.72ms vs 13.37ms) and 5.2× more compact (1.74MB vs 9.11MB) while matching test accuracy.

---

## ADR-003: Native PyTorch Grad-CAM Implementation Over Heavy Third-Party Wrappers

- **Status**: Accepted
- **Context**: External interpretability libraries (e.g., `captum`, `pytorch-grad-cam`) often introduce complex dependency graphs, breaking changes across PyTorch versions, or hidden memory retention issues.
- **Decision**: Implement a native, clean, self-contained `GradCAM` class in `src/seedvision/explainability/gradcam.py` using PyTorch's forward and full backward hook API with context manager cleanup (`__enter__` / `__exit__`).
- **Consequences**:
  - Zero external dependency bloat.
  - Guaranteed removal of forward and backward hooks, preventing CPU/GPU memory leaks.
  - Total control over layer resolution and overlay alpha blending.

---

## ADR-004: Class-Stratified Group Partitioning to Prevent Sample-Level Data Leakage

- **Status**: Accepted
- **Context**: Multiple photographs or rotations of the same physical seed sample appearing across both training and test partitions leads to severe data leakage, generating deceptive 100% test scores that fail in real field conditions.
- **Decision**: Implement group-aware stratified splitting where every physical seed instance (`seed_id`) is partitioned as an indivisible unit within its respective class.
- **Consequences**:
  - Zero intra-seed leakage across Train (70%), Validation (15%), and Test (15%) partitions.
  - Every split retains identical class balance (9 test samples per class across all 4 classes).

---

## ADR-005: Streamlit for Interactive Laboratory Web Application

- **Status**: Accepted
- **Context**: Need a responsive, high-aesthetic research interface without introducing the overhead, build pipelines, or multi-language complexity of full-stack React/Node platforms.
- **Decision**: Use Streamlit with custom CSS tokens, modern laboratory aesthetics, and `@st.cache_resource` model caching.
- **Consequences**:
  - Python-first architecture keeps modeling, Grad-CAM, and UI in unified synchronization.
  - Instant deployment on port 8501 with fast interactive inference.

---

## ADR-006: Explicit Separation of Optical Physical Grading from Biochemical Viability

- **Status**: Accepted
- **Context**: Non-technical stakeholders often mistake visual computer vision predictions for seed viability guarantees (e.g. "if the seed looks healthy, it will germinate").
- **Decision**: Embed persistent scientific disclaimers in the UI, CLI output, and documentation emphasizing that optical classification evaluates external physical appearance only and cannot replace standard biochemical blotter or Tetrazolium tests.
- **Consequences**:
  - Adherence to responsible AI principles and agricultural safety governance.
  - Avoids fraudulent agricultural certification claims.
