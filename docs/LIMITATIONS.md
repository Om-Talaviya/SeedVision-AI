# Scientific Limitations & Responsible AI Statement

## 1. Physical vs Physiological Seed Properties
The fundamental scientific boundary of computer vision in agricultural seed inspection is the distinction between **visually observable physical morphology** and **internal physiological viability**:

| Feature Category | Detectable by SeedVision AI (Computer Vision) | Requires Laboratory Seed Testing Assays |
| :--- | :---: | :---: |
| **External Testa Integrity** | **YES** (Cracks, fractures, chips) | No |
| **Surface Pigmentation** | **YES** (Necrotic spots, staining) | No |
| **Gross Morphology** | **YES** (Shrivelling, wrinkling, size) | No |
| **Insect Boreholes** | **YES** (Visible puncture apertures) | No |
| **Germination Rate (%)** | **NO** | Standard Germination Blotter Test (7-14 days) |
| **Embryo Vitality** | **NO** | Tetrazolium Chloride (TZ) Staining Assay |
| **Moisture Content (%)** | **NO** | Air-Oven Drying Method / NIR Moisture Meter |
| **Pathogen Viability** | **NO** | Agar Plate Culture / Seed Wash PCR |
| **Regulatory Lot Certification** | **NO** | Official Seed Certification Agency (ISTA / AOSCA) |

> [!CAUTION]
> **NEVER present SeedVision AI predictions as official seed testing, germination certification, or seed safety guarantees.** Visual inspection cannot determine whether an embryo is biochemically alive or whether fungal spores present on a blemish are active.

## 2. Explainable AI (Grad-CAM) Limitations
- **Associative, Not Causal**: Grad-CAM heatmaps illustrate spatial regions whose feature activations correlate with the output logit. They do not prove causal reasoning by the neural network.
- **Resolution Constraints**: Grad-CAM heatmaps are computed at the final convolutional feature map (e.g., $4 \times 4$ or $8 \times 8$) and upsampled bilinearly, causing smooth diffusion around fine crack boundaries.
- **Target Layer Dependence**: Different layers in the CNN hierarchy capture different semantic levels; early layers capture edges while late layers capture class-specific composites.

## 3. Generalization & Distribution Shift
- **Species Restriction**: Models are trained specifically for Chickpea (*Cicer arietinum*). Submitting seeds of other species (e.g. green gram, soybean, wheat) will result in inaccurate out-of-distribution predictions.
- **Lighting and Background Sensitivity**: Drastic variations in illumination (colored light, harsh directional shadows) or textured backgrounds can degrade accuracy.
