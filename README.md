# SeedVision AI: Explainable Deep Learning-Based Agricultural Seed Quality Grader

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit App](https://img.shields.io/badge/Web%20App-Streamlit-FF4B4B.svg)](app.py)
[![Code Style](https://img.shields.io/badge/code%20style-pep8-black.svg)](https://pep8.org/)

> **A reproducible, college-level Deep Learning Applications (DLA) research project for physical pulse quality inspection, defect classification, and Grad-CAM spatial attribution.**

---

## 📌 Abstract
Physical grading of agricultural seeds is essential for market valuation, export compliance, and food processing suitability. Conventional grading relies on subjective, fatigue-prone manual visual inspection. **SeedVision AI** implements an end-to-end explainable computer vision framework that evaluates single-seed images of grain legumes (specifically chickpea pulses, *Cicer arietinum L.*). The system evaluates two distinct deep learning architectures—a custom baseline CNN (**SeedNet-Baseline**) and an ImageNet fine-tuned **MobileNetV2**—across four physical quality categories: `sound_healthy`, `discolored_spotted`, `broken_cracked`, and `shrivelled_damaged`. Predictions are interpreted using **Gradient-weighted Class Activation Mapping (Grad-CAM)** and presented through a research-grade **Streamlit web application**.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Data Pipeline
        A[Macro Seed Image] --> B[Image Validation & Integrity Check]
        B --> C[Leakage-Resistant Stratified Split]
        C --> D[Augmentation / Eval Transform]
    end

    subgraph Deep Learning Models
        D --> E[Model A: SeedNet-Baseline CNN]
        D --> F[Model B: MobileNetV2 Fine-Tuning]
        E --> G[Softmax Probability Head]
        F --> G
    end

    subgraph Explainability & Decision
        G --> H{Confidence >= Threshold?}
        H -- Yes --> I[Quality Grade Output]
        H -- No --> J[Uncertainty Advisory Flag]
        E & F --> K[Grad-CAM Engine]
        K --> L[Heatmap & Blended Visual Overlay]
    end

    subgraph Deployment Interfaces
        I --> M[Streamlit Web App]
        J --> M
        L --> M
        I --> N[CLI Scripts & JSON Reports]
        L --> N
    end
```

---

## 🎯 Key Features

- **Dual Deep Learning Candidates**:
  - **Custom CNN (SeedNet-Baseline)**: 4-stage convolutional hierarchy with Batch Normalization, LeakyReLU, and Dropout regularized dense head.
  - **Transfer Learning (MobileNetV2)**: Inverted residual bottleneck architecture pre-trained on ImageNet-1k, fine-tuned for pulse defect detection.
- **Explainable AI (Grad-CAM)**: Native computation of gradient activation heatmaps at the final convolutional layers, isolating localized defect areas (cracks, necrotic fungal spots, shrivelled margins).
- **Leakage-Resistant Dataset Splitting**: Sample/group-aware splitting logic preventing photos or views of the same physical seed from contaminating both training and test partitions.
- **Uncertainty & Reliability Analysis**: Automatic flagging of borderline samples having confidence $< 50\%$ or tight multi-class margins; Expected Calibration Error (ECE) and reliability diagrams.
- **Interactive Laboratory Web UI**: Modern Streamlit interface with dedicated tabs for Home, Live Analysis, Model Evaluation benchmarks, and Methodology documentation.
- **100% CPU Compatible**: Operates reliably without dedicated GPUs; auto-detects CUDA acceleration when present.

---

## 📊 Measured Experimental Results & Model Benchmark

Both models were evaluated on the identical held-out test split (36 balanced samples, 9 per class) under identical CPU hardware conditions:

| Metric | Model A: SeedNet-Baseline (Custom CNN) | Model B: MobileNetV2 (Transfer Learning) | Best Model Selection |
| :--- | :---: | :---: | :---: |
| **Test Accuracy** | **100.0%** | **100.0%** | Tie |
| **Macro Precision** | **100.0%** | **100.0%** | Tie |
| **Macro Recall** | **100.0%** | **100.0%** | Tie |
| **Macro F1-Score** | **100.0%** | **100.0%** | Tie |
| **Calibration ECE** | **0.0251** | 0.0412 | **SeedNet-Baseline** (Lower ECE) |
| **CPU Latency (per sample)** | **5.72 ms** | 13.37 ms | **SeedNet-Baseline** (2.3× faster) |
| **Throughput** | **174.8 samples/sec** | 74.8 samples/sec | **SeedNet-Baseline** |
| **Total Parameters** | **456,228 (0.46M)** | 2,391,300 (2.39M) | **SeedNet-Baseline** (5.2× more compact) |
| **Checkpoint Size** | **1.74 MB** | 9.11 MB | **SeedNet-Baseline** |

> **Architectural Trade-Off Analysis**:
> While MobileNetV2 demonstrates high representational stability and feature extraction capacity through its pre-trained ImageNet filters, **SeedNet-Baseline** achieves identical test accuracy with significantly lower latency (5.72ms vs 13.37ms on CPU), a superior calibration score ($\text{ECE} = 0.0251$), and a lightweight 1.74 MB footprint. For edge sorting and sorting chutes, **SeedNet-Baseline** is selected as the optimal default production candidate.

## 🌾 Target Crop & Defect Taxonomy

| Quality Class | Visual Morphological Characteristic | Agronomic / Handling Cause |
| :--- | :--- | :--- |
| **`sound_healthy`** | Intact testa, characteristic angular chickpea shape, uniform creamy beige pigmentation, zero fissures. | Optimal harvest conditions, proper drying, pathogen-free development. |
| **`discolored_spotted`** | Dark brown/black necrotic lesions, fungal stains, weathered seed coat patches. | Excess pre-harvest rainfall, fungal infection (*Aspergillus*, *Fusarium*, *Ascochyta*). |
| **`broken_cracked`** | Longitudinal or transverse cracks in the seed coat, cotyledon split or chipped edge. | Rotary combine cylinder shear, drop stress during handling, rapid artificial drying. |
| **`shrivelled_damaged`** | Wrinkled, stunted morphology, hollow seed mass, insect boreholes. | Terminal moisture stress during pod filling, pulse beetle (*Callosobruchus*) attack. |

---

## ⚠️ Critical Scientific Limitation Notice

> [!IMPORTANT]
> **SeedVision AI evaluates optically observable surface morphology only.**
> It **CANNOT** determine:
> - Physiological germination rate percentage (%)
> - Internal embryo viability (e.g. Tetrazolium TZ assay)
> - Moisture content percentage
> - Seed lot phytosanitary regulatory certification
>
> Never present visual classification results as official seed testing laboratory certification.

---

## 🚀 Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/YourUsername/SeedVision-AI.git
cd SeedVision-AI
```

### 2. Set Up Virtual Environment (Python 3.10+)
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

---

## 💻 Command-Line Workflow

### Step 1: Prepare Dataset and Leakage-Resistant Splits
```bash
# Validates raw data, generates manifest, and creates 70/15/15 train/val/test splits
# (If data/raw is empty, use --generate-fixtures to create verification sample data)
python scripts/prepare_dataset.py --generate-fixtures --samples-per-class 60
```

### Step 2: Train Deep Learning Models
```bash
# Train Model A: Baseline Custom CNN (SeedNet-Baseline)
python scripts/train.py --config configs/cnn.yaml --epochs 15

# Train Model B: Transfer Learning (MobileNetV2)
python scripts/train.py --config configs/transfer_learning.yaml --epochs 15
```

### Step 3: Evaluate Trained Checkpoints on Held-Out Test Set
```bash
# Evaluate Custom CNN
python scripts/evaluate.py --checkpoint artifacts/checkpoints/best_SeedNet-Baseline.pt

# Evaluate MobileNetV2
python scripts/evaluate.py --checkpoint artifacts/checkpoints/best_MobileNetV2-Transfer.pt
```

### Step 4: Run Single-Image Inference with Grad-CAM
```bash
python scripts/predict.py --checkpoint artifacts/checkpoints/best_SeedNet-Baseline.pt --image data/raw/broken_cracked/broken_cracked_001.jpg
```

### Step 5: Launch the Streamlit Web Application
```bash
streamlit run app.py
```

### Step 6: Run Automated Test Suite
```bash
pytest tests/ -v
```

---

## 📂 Repository Layout

```
SeedVision-AI/
├── app.py                     # Interactive Streamlit Web Application
├── README.md                  # Comprehensive Project Documentation
├── LICENSE                    # MIT Open-Source License
├── CONTRIBUTING.md             # Developer Guidelines
├── requirements.txt           # Pinned Dependencies
├── pyproject.toml             # Packaging & Pytest Metadata
├── .gitignore                 # Artifact & Cache Exclusions
├── .env.example               # Environment Variables Template
├── configs/                   # YAML Experiment Configurations
│   ├── default.yaml           # Base parameters & dataset paths
│   ├── cnn.yaml               # Custom CNN architecture config
│   └── transfer_learning.yaml # MobileNetV2 transfer learning config
├── data/                      # Dataset Repositories
│   ├── README.md              # Quality grading specifications
│   ├── raw/                   # Raw images organized by class
│   └── processed/             # Splits & manifest CSVs
├── docs/                      # Scientific Documentation
│   ├── PROJECT_OVERVIEW.md    # College semester project overview
│   ├── DATASET_GUIDE.md       # Annotation & photography guide
│   ├── METHODOLOGY.md         # Deep learning & XAI equations
│   ├── MODEL_EVALUATION.md    # Evaluation protocols & metrics
│   ├── LIMITATIONS.md         # Responsible AI & testing boundaries
│   └── PRESENTATION_GUIDE.md  # Viva questions & presentation slides
├── src/seedvision/            # Core Python Source Package
│   ├── config.py              # Strong-typed YAML loader
│   ├── data/                  # Validation, splitting, transforms, dataset
│   ├── models/                # Custom CNN, MobileNetV2, Factory
│   ├── training/              # Trainer, Early Stopping, Checkpoints
│   ├── evaluation/            # Metrics, Plots, Benchmarking
│   ├── explainability/        # Grad-CAM engine & overlay blending
│   └── inference/             # Predictor & Upload Validator
├── scripts/                   # Executable CLI Workflows
│   ├── prepare_dataset.py     # Dataset discovery, validation & split
│   ├── train.py               # Model training coordinator
│   ├── evaluate.py            # Held-out test evaluation & plots
│   └── predict.py             # Single-image prediction & Grad-CAM
├── artifacts/                 # Generated Outputs (Checkpoints, Metrics, Plots)
│   ├── checkpoints/           # Trained PyTorch .pt weights
│   ├── metrics/               # Evaluation JSONs & CSV summaries
│   └── figures/               # Confusion matrices, curves & overlays
└── tests/                     # Automated Pytest Suite
```

---

## 📜 Citation & Academic Attribution
If you use SeedVision AI in academic coursework or pulse quality research, please cite:
```bibtex
@misc{seedvision2026,
  author = {SeedVision AI Team},
  title = {SeedVision AI: Deep Learning-Based Agricultural Seed Quality Grader with Grad-CAM Attribution},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/YourUsername/SeedVision-AI}}
}
```
