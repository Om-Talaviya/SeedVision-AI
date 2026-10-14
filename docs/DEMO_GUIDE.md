# SeedVision AI — Academic Demonstration & Viva Guide
**Course / Track:** Deep Learning Applications (DLA)  
**Project:** SeedVision AI — Explainable Deep Learning-Based Agricultural Seed Quality Grader  
**Author:** Om Talaviya  
**Repository:** [https://github.com/Om-Talaviya/SeedVision-AI](https://github.com/Om-Talaviya/SeedVision-AI)  

---

## 1. Agricultural Problem & Motivation

Grain legumes—specifically chickpeas (*Cicer arietinum L.*)—represent a staple protein source and vital crop in global agriculture. In commercial grain procurement, export lot grading, and post-harvest seed handling, physical quality grading dictates economic valuation, food processing grade, and storage viability.

### The Problem with Manual Inspection:
- **Subjective & Inconsistent:** Traditional grading relies on manual visual examination of seed samples by grain inspectors under varied ambient illumination.
- **Fatigue & Human Error:** Inspecting thousands of seeds causes visual fatigue, leading to misclassification of micro-fissures and subtle fungal spots.
- **Throughput Bottlenecks:** Manual physical sorting cannot scale to industrial processing chutes or high-speed intake conveyors.

### The SeedVision AI Objective:
Develop an end-to-end, reproducible deep learning framework that standardizes physical quality inspection, automates 4-class defect categorization, provides transparent spatial attribution through Gradient-weighted Class Activation Mapping (Grad-CAM), and guarantees rigorous scientific evaluation.

---

## 2. Why Image Classification is Suitable

Macro single-seed photography provides high spatial resolution of the external seed coat (testa):
- **Defects are Spatially Localized:** Cotyledon cracks, necrotic fungal lesions, insect puncture holes, and wrinkling have distinct edge, texture, and color features observable at the surface.
- **High-Throughput Non-Destructive Testing (NDT):** RGB optical capture is non-destructive, instantaneous, and does not alter or damage seed lots.
- **Cost-Effective Hardware:** Can be deployed using inexpensive CMOS camera modules and standard LED ring lights on processing conveyor belts.

---

## 3. Dataset Classes & Scientific Distinction

### 3.1 Four Quality Classes
| Class Name | Visual Morphological Characteristic | Agronomic / Handling Cause |
| :--- | :--- | :--- |
| **`sound_healthy`** | Intact testa, characteristic angular/spherical shape, uniform creamy beige color, zero cracks or necrotic spots. | Optimal harvest maturity, proper drying, disease-free. |
| **`discolored_spotted`** | Dark brown/black necrotic lesions, fungal stains, weathered seed coat patches. | Pre-harvest moisture exposure, fungal pathogens (*Aspergillus*, *Fusarium*, *Ascochyta*). |
| **`broken_cracked`** | Longitudinal or transverse fissures in testa, chipped margins, cotyledon cleavage. | Combine thresher impact, mechanical drop shear, rapid heat drying stress. |
| **`shrivelled_damaged`** | Wrinkled, hollowed, underdeveloped seed mass, insect boreholes. | Terminal drought stress, pulse beetle (*Callosobruchus*) infestation. |

### 3.2 Current Dataset Status: Synthetic Fixtures vs. Real Photographs
- **Verification Dataset (`data/raw/`):** 240 procedural synthetic fixtures (60 samples per class) generated via PIL geometry algorithms to mathematically verify the convolutional pipeline, backpropagation, and Grad-CAM hooks.
- **Scientific Honesty:** 100% test accuracy on synthetic fixtures proves software and architectural convergence. It **does NOT** prove real-world agricultural pulse grading accuracy.
- **Real-World Acquisition Protocol:** The repository scaffolds `data/real_raw/` with a dual-annotator scoring protocol, Cohen's Kappa arbitration ($\kappa \ge 0.85$), and multi-gate provenance verification (`provenance_verifier.py`). Real-world agronomic validation is **pending** until verified photographs are ingested and evaluated.

---

## 4. Deep Learning Model Architectures

SeedVision AI implements and benchmarks two distinct architectures:

### Model A: Custom SeedNet-Baseline CNN
- **Design Philosophy:** Edge-optimized, lightweight, designed for minimal latency on CPU or embedded industrial sorting microcontrollers.
- **Architecture:**
  - 4 Sequential Convolutional Stages (Conv2D $3\times3$, BatchNorm2D, LeakyReLU $\alpha=0.1$, MaxPool2D $2\times2$).
  - Channel Progression: $3 \rightarrow 32 \rightarrow 64 \rightarrow 128 \rightarrow 256$.
  - Adaptive Average Pooling ($2\times2$) to reduce feature spatial dimensions.
  - Regularized Dense Head: Linear(1024 $\rightarrow$ 256), BatchNorm, LeakyReLU, Dropout ($p=0.4$), Linear(256 $\rightarrow$ 4 classes).
- **Parameters & Latency:** 456,228 parameters (0.46M), 1.74 MB checkpoint size, **5.72 ms** inference latency per sample on standard CPU.

### Model B: MobileNetV2 Transfer Learning
- **Design Philosophy:** Representational capacity and transfer learning from large-scale natural image representations.
- **Architecture:**
  - Pretrained on ImageNet-1k, utilizing inverted residual bottleneck blocks with depthwise separable convolutions.
  - Linear classifier head replaced: Dropout ($p=0.3$) followed by Linear(1280 $\rightarrow$ 4 classes).
  - Two-stage training: Feature extractor frozen for initial epochs, followed by partial fine-tuning of top residual blocks with a reduced learning rate ($10^{-4}$).
- **Parameters & Latency:** 2,391,300 parameters (2.39M), 9.11 MB checkpoint size, **13.37 ms** CPU latency.

---

## 5. Image Preprocessing & Data Augmentation

### Evaluation Transform:
- Input Resize to $224 \times 224$ pixels using bilinear interpolation.
- Conversion to PyTorch float32 tensor ($[0, 1]$ range).
- Standardization using ImageNet normalization constants:
  $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$

### Training Augmentation Pipeline:
- **Random Horizontal & Vertical Flips ($p=0.5$):** Reflects physical rotational invariance of seeds in a tray.
- **Random Affine Transformations:** Rotation ($\pm 180^\circ$), translation ($\pm 10\%$), scale ($0.9 - 1.1$).
- **Color Jitter:** Minor brightness ($\pm 15\%$), contrast ($\pm 15\%$), and saturation ($\pm 10\%$) adjustments simulating ambient lighting variations.

---

## 6. Training, Validation & Leakage-Free Splitting

- **Loss Function:** Cross-Entropy Loss with optional label smoothing.
- **Optimizer:** AdamW with weight decay $10^{-4}$.
- **Learning Rate Schedule:** Cosine Annealing Learning Rate scheduler with warmup.
- **Early Stopping:** Monitored on validation loss with patience of 7 epochs.
- **Leakage-Resistant Group Splitting:**
  - Standard random splitting risks placing multiple photos of the same physical seed into both training and testing sets, artificially inflating test accuracy.
  - SeedVision AI groups all multi-view captures by `seed_id`. Stratified group splitting isolates entire physical seeds into either train ($70\%$), val ($15\%$), or test ($15\%$), guaranteeing strict statistical independence.

---

## 7. Scientific Evaluation Metrics

During an academic viva, explain that accuracy alone is insufficient for agricultural grading:

1. **Top-1 Accuracy:** Overall proportion of correct classifications across all classes.
2. **Per-Class Precision:** $\frac{\text{TP}}{\text{TP} + \text{FP}}$ — Critical for commercial grading: prevents defective seeds from being incorrectly certified as healthy.
3. **Per-Class Recall (Sensitivity):** $\frac{\text{TP}}{\text{TP} + \text{FN}}$ — Critical for food safety: guarantees diseased or shrivelled seeds are caught.
4. **Macro F1-Score:** Unweighted harmonic mean of precision and recall across all 4 classes, preventing majority-class bias.
5. **Confusion Matrix:** Identifies specific inter-class misclassifications (e.g., distinguishing mechanical fissures from deep wrinkled folds).
6. **Expected Calibration Error (ECE):**
   - Evaluates whether predicted probabilities match true empirical probabilities:
     $$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$$
   - SeedNet-Baseline achieved an $\text{ECE} = 0.0251$, demonstrating high probability calibration.

---

## 8. Explainability via Grad-CAM & Limitations

### 8.1 Grad-CAM Derivation
Gradient-weighted Class Activation Mapping computes the gradient of the target class score $y^c$ with respect to feature activation map $A^k$ of the final convolutional layer:

$$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$

The resulting activation map is upsampled to $224 \times 224$, color-mapped using the Jet colormap, and alpha-blended over the original input photo.

### 8.2 Practical Limitations of Grad-CAM:
- **Spatial Resolution Boundary:** The final convolutional feature map is $7 \times 7$ or $14 \times 14$ pixels. Fine hairline cracks under 0.5mm appear as diffuse localized activations rather than pixel-level segmentations.
- **Attribution vs. Causation:** Grad-CAM indicates where the convolutional filters responded strongly; it does not constitute a legal or statutory proof of defect cause.

---

## 9. Live Demonstration Protocol (Streamlit Application)

Follow this sequence during project demonstrations:

### Step 1: Launch Application
```bash
streamlit run app.py
```
Open browser to `http://localhost:8501`.

### Step 2: Tab 1 — System Overview & Architecture
- Walk through the problem statement, 4-stage processing pipeline, and target pulse defect taxonomy.
- Highlight the real-world vs. synthetic provenance disclaimer in the header.

### Step 3: Tab 2 — Live Single-Seed Inference & Grad-CAM
- Select a sample seed image from the test split (or upload a custom image).
- Select **SeedNet-Baseline** or **MobileNetV2**.
- Point out the predicted quality grade, confidence score, and inference latency.
- Demonstrate the **Grad-CAM visual overlay**: show how the heatmap focuses directly on the crack in `broken_cracked` or the dark fungal lesion in `discolored_spotted`.
- Note the **Uncertainty Advisory Flag**: show how confidence $<50\%$ automatically raises an advisory banner.

### Step 4: Tab 3 — Model Evaluation & Benchmarks
- Present the side-by-side benchmark table comparing SeedNet-Baseline and MobileNetV2.
- Explain the engineering trade-off: SeedNet-Baseline is 2.3× faster (5.72 ms vs 13.37 ms) and 5.2× more compact (0.46M vs 2.39M params).
- Display the confusion matrix and calibration reliability curves.
- Point out the **Multi-Gate Provenance Banner**: clearly showing that these runs are marked `Synthetic Pipeline Verification`.

### Step 5: Tab 4 — Real-Data Ingestion & Methodology
- Show the interactive protocol for ingesting genuine chickpea photos.
- Explain how `scripts/ingest_real_data.py` prevents cross-split seed leakage and filters ambiguous human labels.

---

## 10. Critical Limitations & Future Scope

### Fundamental Operational Boundaries:
- **Visual Surface Only:** Visual RGB inspection evaluates external seed coat morphology only. It **CANNOT** verify internal embryo viability (Tetrazolium assay), physiological germination rate, or internal moisture content.
- **Statutory Tolerance Distinction:** Project-defined defect definitions (e.g. 1.5mm crack length) are single-seed operational computer vision proxies. Official statutory standards (AGMARK India, USDA GIPSA) measure defect percentage by weight in bulk samples.

### Roadmap & Future Extensions:
1. **Multi-Seed Tray Segmentation:** Integration of YOLOv8/Faster-RCNN to isolate and grade dozens of seeds from a single laboratory tray capture.
2. **Test-Time Augmentation (TTA):** 4-fold rotation averaging to stabilize predictions on borderline orientations.
3. **Multi-Spectral NIR Integration:** Combining 700nm–1050nm reflectance sensors to estimate internal moisture alongside external morphology.
4. **Edge Deployment:** Exporting PyTorch models to ONNX Runtime and INT8 TensorRT for execution on Raspberry Pi 5 or NVIDIA Jetson sorting chutes.
