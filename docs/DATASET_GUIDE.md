# SeedVision AI Dataset Guide & Quality Standards

## 1. Target Crop: Chickpea (*Cicer arietinum L.*)
Chickpea is one of the world's primary legume pulse crops, cultivated across the Mediterranean, South Asia, the Middle East, and the Americas. It is rich in dietary protein and essential amino acids. In global pulse marketing, visual seed grade directly dictates price and export eligibility.

## 2. Morphological Quality Classes & Grading Rules

| Class Name | Visual Quality Description | Primary Physical Defect Causes | Handling in SeedVision AI |
| :--- | :--- | :--- | :--- |
| **`sound_healthy`** | Intact testa (seed coat), characteristic angular chickpea keel/beak, smooth texture, uniform cream/beige pigmentation, zero lesions. | Optimal harvest conditions, proper drying, unblemished growth. | Target benchmark class. |
| **`discolored_spotted`** | Dark brown/black necrotic surface spots, patchy fungal blemishes, or weathering stains across the testa. | Pre-harvest moisture exposure, fungal colonization (*Aspergillus*, *Fusarium*, *Ascochyta* blight). | Classified as damaged; requires localized spatial attribution. |
| **`broken_cracked`** | Longitudinal or transverse cracks in the testa, cotyledon split, or missing seed coat fragments. | Rotary combine cylinder shear, high-drop handling, excessive drying speed. | Classified as defective; reduces seed storage life. |
| **`shrivelled_damaged`** | Wrinkled, hollowed, or stunted seed morphology; visible insect boreholes. | Terminal drought during pod-filling stage, or pulse beetle (*Callosobruchus maculatus*) attack. | Classified as damaged; unviable for planting. |

## 3. Data Collection & Photography Protocol
To replicate laboratory grading accuracy:
- **Background**: Neutral, matte laboratory inspection tray (off-white or light neutral gray) having RGB values within $[230, 245]$.
- **Illumination**: Diffuse dual LED lighting at 45-degree angle to eliminate harsh specular reflections and deep drop shadows.
- **Framing**: Single seed centered within the field of view, occupying approximately 60% to 75% of the frame.
- **Resolution**: Minimum $128 \times 128$ pixels; supported up to $4096 \times 4096$ pixels.

## 4. Leakage-Resistant Dataset Splitting
Data leakage is a severe failure mode in agricultural computer vision where multiple photographs or augmentations of the same physical seed appear in both training and test sets, artificially inflating test scores.

SeedVision AI implements **Group-Based Partitioning**:
- Each physical seed entity possesses an invariant `seed_id`.
- The split algorithm (`create_leakage_resistant_splits`) ensures all images, rotations, or views associated with `seed_id` are isolated strictly within one split partition.
- Split distribution: **70% Training**, **15% Validation**, **15% Held-out Testing**.
