# SeedVision AI: UI/UX Design System & Experience Specification

## 1. Visual Direction & Design Philosophy

The visual interface for **SeedVision AI** embodies a **modern agricultural biotechnology laboratory**:
- **Clean & Scientific**: Distraction-free, airy layouts prioritizing high-resolution visual evidence, clear typography, and scientific data tables.
- **Organic Precision**: Deep botanical forest greens (`#1b5e20`, `#2e7d32`) paired with crisp neutral laboratory slate backgrounds (`#f8fafc`, `#ffffff`) and amber advisory accents.
- **Transparency-First**: Explainability heatmaps and uncertainty notices are given prominent visual hierarchy alongside top-1 predictions rather than hidden in submenus.
- **Zero AI Slop**: No generic neon gradients, no meaningless animated spinners, and no deceptive visual indicators.

---

## 2. Design Tokens & Color Palette

### 2.1 Core Palette (WCAG 2.1 AA Compliant)

| Token Name | Hex Code | Contrast Ratio vs Background | Semantic Purpose |
| :--- | :---: | :---: | :--- |
| `--color-brand-primary` | `#2e7d32` | 4.8:1 (Pass AA) | Primary buttons, active tabs, positive trends |
| `--color-brand-dark` | `#1b5e20` | 7.6:1 (Pass AAA) | Hero card gradients, high-emphasis headers |
| `--color-brand-light` | `#e8f5e9` | N/A | Badge backgrounds, highlighted table rows |
| `--color-accent-amber` | `#f59e0b` | 4.6:1 (Pass AA) | Uncertainty advisories, calibration markers |
| `--color-surface-bg` | `#f8fafc` | N/A | App body canvas |
| `--color-card-bg` | `#ffffff` | N/A | Container cards, model inspect widgets |
| `--color-text-primary` | `#0f172a` | 14.1:1 (Pass AAA) | Headings, primary metrics, class names |
| `--color-text-secondary`| `#475569` | 6.2:1 (Pass AAA) | Captions, axis labels, methodology copy |
| `--color-border-subtle` | `#e2e8f0` | N/A | Dividers, card borders, table grid lines |

### 2.2 Quality Grading Class Badges

```
[ sound_healthy ]       -> Background: #dcfce7 | Text: #166534 | Border: #bbf7d0
[ discolored_spotted ]  -> Background: #fef9c3 | Text: #854d0e | Border: #fef08a
[ broken_cracked ]      -> Background: #ffedd5 | Text: #9a3412 | Border: #fed7aa
[ shrivelled_damaged ]  -> Background: #fee2e2 | Text: #991b1b | Border: #fecaca]
```

### 2.3 Typography Scale
- **Display / Hero H1**: `2.2rem` (35px), SemiBold 700, letter-spacing `-0.5px`.
- **Section Headers H2**: `1.6rem` (26px), SemiBold 600, margin-bottom `1rem`.
- **Card Titles H3**: `1.15rem` (18px), Medium 600, color `#1b5e20`.
- **Body Regular**: `0.95rem` (15px), Regular 400, line-height `1.55`.
- **Captions / Footnotes**: `0.85rem` (13px), Regular 400, color `#64748b`.
- **Monospace (File paths & latencies)**: SFMono-Regular, Consolas, "Liberation Mono", Menlo.

---

## 3. Information Architecture & Navigation Wireframe

```
+---------------------------------------------------------------------------------------+
|  🌱 SeedVision AI  |  Crop: Chickpea (Cicer arietinum L.)  |  Device: CPU (Verified)  |
+---------------------------------------------------------------------------------------+
|  [Sidebar Navigation]     |  [Main Working Area]                                      |
|                           |                                                           |
|  (o) Home & Overview      |  HERO BANNER:                                             |
|  ( ) Seed Quality Analysis|  Title, Purpose, Inspection Workflow Diagram              |
|  ( ) Model Benchmark      |                                                           |
|  ( ) Methodology & Limits |  GRID OF QUALITY CLASSES:                                 |
|                           |  [Healthy] [Spotted] [Cracked] [Shrivelled]               |
|  ------------------------ |                                                           |
|  Model Checkpoints Ready: |  SCIENTIFIC DISCLAIMER:                                   |
|  * best_SeedNet-Baseline  |  "Notice: Optical RGB grading evaluates visual            |
|  * best_MobileNetV2-Trans |   morphology only, not internal germination viability."   |
+---------------------------------------------------------------------------------------+
```

---

## 4. Key Component Layouts

### 4.1 Seed Analysis Page (Inference & Grad-CAM)
When an image is submitted (or demo sample selected), the dashboard dynamically organizes into a structured 4-level display:
1. **Model Selector Bar**: Switch between `SeedNet-Baseline` and `MobileNetV2-Transfer`.
2. **Executive Grade KPI Cards**:
   - Card 1: **Predicted Quality Badge** (`BROKEN_CRACKED`).
   - Card 2: **Confidence Score** (`99.2%`).
   - Card 3: **Inference Latency** (`5.7 ms`).
   - Card 4: **Architecture Name** (`custom_cnn`).
3. **Probability Distribution Bar Chart**: Clean horizontal bar chart showing Softmax percentages across all 4 classes.
4. **Grad-CAM Visual Triplet**:
   - Column 1: **Original Seed Image** (Cropped RGB view).
   - Column 2: **Grad-CAM Activation Heatmap** (Jet colormap normalized [0, 1]).
   - Column 3: **Blended Decision Overlay** (45% alpha overlay highlighting fracture or lesion location).

### 4.2 Model Evaluation Page
- **Comparison Table**: Comprehensive markdown and DataFrame view summarizing Accuracy, Macro F1, Calibration ECE, and CPU latency.
- **Dual Confusion Matrices**: Raw integer counts beside row-normalized true positive recall rates.
- **Reliability Diagrams**: Calibration curve plotting predicted confidence against observed empirical accuracy.

---

## 5. Accessibility (WCAG 2.1 AA Compliance) & Mobile Responsiveness
- **Contrast**: All text meets or exceeds the WCAG AA minimum contrast ratio of 4.5:1 against light card backgrounds.
- **Color Independence**: Quality grades never rely on color alone; each status includes explicit text labels and textual badges.
- **Responsive Layout**: Streamlit columns collapse into a single-column stacked layout on mobile viewport widths ($< 768px$), preserving image legibility and button hit areas (min 44px height).
