# SeedVision AI: College Presentation & Viva Guide

## 1. Project Presentation Outline (10-12 Slides)
- **Slide 1: Title & Authors**: SeedVision AI — Deep Learning-Based Seed Quality Grader with Grad-CAM Explainability.
- **Slide 2: Background & Motivation**: Pulse grading challenges in agricultural supply chains; manual inspection limitations.
- **Slide 3: Objectives & Scientific Scope**: Physical optical grading vs physiological seed viability.
- **Slide 4: Dataset & Quality Classes**: 4 distinct classes: Healthy, Spotted, Cracked, Shrivelled; leakage-resistant group splitting.
- **Slide 5: System Architecture**: Modular pipeline from image acquisition to Grad-CAM heatmap overlays.
- **Slide 6: Dual Deep Learning Architectures**: SeedNet-Baseline (Custom CNN) vs MobileNetV2 (Transfer Learning).
- **Slide 7: Training Methodology**: Label smoothing, Cosine Annealing learning rate schedule, early stopping.
- **Slide 8: Explainable AI with Grad-CAM**: Mathematical derivation, gradient pooling, feature map weighting.
- **Slide 9: Quantitative Experimental Results**: Accuracy, Macro F1, Confusion Matrix, and Calibration (ECE).
- **Slide 10: Streamlit Interactive Demonstration**: Live image grading, top-k probabilities, uncertainty advisory.
- **Slide 11: Scientific Limitations & Ethical Guardrails**: Clarifying why optical CV cannot certify germination rate.
- **Slide 12: Future Work & Conclusion**: Multi-spectral imaging, automated grading conveyors, edge deployment.

---

## 2. Frequently Asked Viva Questions & Model Answers

### Q1: Why did you choose Chickpea rather than combining multiple crops into one dataset?
**Answer:** In agricultural seed science, physical grading standards and defect morphologies differ drastically between crops (e.g., green gram vs wheat vs chickpea). Mixing unrelated crop species without distinct species-level classifiers creates spurious label noise and class imbalance. Focus on a single pulse crop allows scientifically valid class definitions aligned with commercial standards.

### Q2: What is data leakage, and how does SeedVision AI prevent it?
**Answer:** Data leakage occurs when images of the same physical seed sample (or augmentations of the same photograph) appear in both training and test sets, artificially inflating test scores. SeedVision AI tracks physical `seed_id` entities and applies `GroupShuffleSplit` so that all instances or views of a physical seed remain strictly isolated within one partition.

### Q3: Why is accuracy alone insufficient for model evaluation?
**Answer:** Accuracy fails to account for class imbalance and asymmetric error costs. In agricultural seed sorting, confusing a broken or mouldy seed with a healthy seed carries severe commercial contamination penalties. We compute **Macro Precision, Macro Recall, Macro F1-score**, per-class confusion matrices, and **Expected Calibration Error (ECE)**.

### Q4: How does Grad-CAM work mathematically?
**Answer:** Grad-CAM calculates the gradient of the score for target class $c$ with respect to the feature map activations $A^k$ of the final convolutional layer. By performing global average pooling over spatial dimensions $(i, j)$, it derives importance weights $\alpha_k^c$. These weights scale the activation maps, followed by a ReLU operation to preserve only features that positively support class $c$.

### Q5: Can this system certify that seeds will germinate in the field?
**Answer:** **No, absolutely not.** Computer vision detects external physical defects (cracks, discoloration, shrivelling). Germination depends on internal physiological and biochemical embryo vitality, moisture levels, and enzymatic activity, which require standard laboratory blotter tests or Tetrazolium assays. We explicitly document this boundary in our application interface and documentation.

### Q6: Why compare a Custom CNN against MobileNetV2?
**Answer:** To evaluate the trade-off between architectural complexity, pre-trained feature representation, and inference latency. MobileNetV2 leverages depthwise separable convolutions pre-trained on ImageNet, whereas our Custom CNN provides a lightweight, transparent baseline trained from scratch without external dependencies.
