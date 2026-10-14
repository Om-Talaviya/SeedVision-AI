# SeedVision AI Dataset Guide & Quality Standards

## 1. Target Crop: Chickpea (*Cicer arietinum L.*)
Chickpea is one of the world's primary legume pulse crops, cultivated across the Mediterranean, South Asia, the Middle East, and the Americas. It is rich in dietary protein and essential amino acids. In global pulse marketing, visual seed grade directly dictates price and export eligibility.

---

## 2. Morphological Quality Classes & Grading Rules

| Class Name | Visual Quality Description | Primary Physical Defect Causes | Operational CV Proxy | Official Standard Analogue |
| :--- | :--- | :--- | :--- | :--- |
| **`sound_healthy`** | Intact testa (seed coat), angular chickpea keel/beak, smooth texture, uniform cream/beige pigmentation. | Optimal harvest conditions, proper drying, unblemished growth. | Zero cracks; dark spots $\le 1.0\text{ mm}$ | Sound wholesome grain |
| **`discolored_spotted`** | Dark brown/black necrotic surface spots, patchy fungal blemishes, or weathering stains. | Pre-harvest rain, fungal colonization (*Aspergillus*, *Fusarium*, *Ascochyta*). | Necrotic spots $\ge 5\%$ surface area | "Slightly Damaged" (AGMARK/USDA) |
| **`broken_cracked`** | Fissures in the testa, cotyledon split, or missing seed coat fragments. | Rotary combine cylinder shear, high-drop handling, rapid drying. | Crack $> 1.5\text{ mm}$ exposing cotyledon | "Broken grains" (cotyledon split) |
| **`shrivelled_damaged`** | Wrinkled, hollowed, or stunted morphology; insect boreholes. | Drought at pod-filling, or pulse beetle (*Callosobruchus*) attack. | Volume $< 75\%$ norm or circular borehole | "Shrivelled / Weevilled" |

> [!NOTE]
> **STATUTORY VS OPERATIONAL PROXY DISTINCTION:**
> Official commercial pulse standards (AGMARK India, USDA GIPSA) regulate grain lots by **mass percentage (weight/weight)** in bulk samples (e.g. max 2% broken seeds by weight). Single-seed microscopic thresholds (e.g., $1.5\text{ mm}$ crack length, $5\%$ necrotic area) are **project-defined operational proxies** for RGB macro-computer-vision classification, NOT statutory standards. Full protocol documented in [REAL_DATA_ACQUISITION_WORKFLOW.md](file:///d:/AI/Projects/SeedVision%20AI/docs/REAL_DATA_ACQUISITION_WORKFLOW.md).

---

## 3. Public Pulse Dataset Investigation & Status

In accordance with rigorous ML auditing standards, an exhaustive search across public academic repositories (Mendeley Data, Zenodo, GitHub, Kaggle) was performed:

| Repository / Dataset | Stated Contents | Defect Suitability | Provenance & License | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **PulseLibrary** (Mendeley Data DOI: `10.17632/kw7x5xyprh.1`) | 3,123 real pulse images (chickpea, peas, lentils, mung bean, black gram) | Labels by **pulse species**, NOT by physical defect | Univ. of Asia Pacific & FIU (Sep 2026), CC BY 4.0 | Real chickpea images present; requires defect annotation before training. |
| **TRCS_8_SET** (GitHub: `ibrahimkilic/TRCS_8_SET`, MDPI) | 7,200 real chickpea photos across 8 certified varieties | Labels by **botanical cultivar**, NOT by defect | Selcuk University (2022-2023), Open Access | High quality; excellent for botanical baseline, but unsegmented by defect. |
| **SOYPR** (GitHub: `gabriel-bertolini/SOYPR`) | Legume defect images (intact, cercospora/spotted, mechanical, bug) | Matches defect taxonomy, but is **soybean**, not chickpea | State Univ. of Maringá, CC BY 4.0 | Cross-species domain gap; unsuitable for pure chickpea benchmark. |
| **Fusarium Wilt** (Kaggle) | Field photos of diseased plants/leaves | Plant foliage, NOT seed grains | Kaggle Open Dataset | Unsuitable for grain seed grading. |

> [!NOTE]
> **Conclusion:** No public dataset currently provides pre-annotated chickpea images split into the 4 physical defect classes. Researchers must either annotate real chickpea samples extracted from **PulseLibrary** or capture physical seeds in an agronomic laboratory.

---

## 4. Ingesting Genuine Seed Photographs

SeedVision AI provides a dedicated, automated CLI pipeline in `scripts/ingest_real_data.py` to ingest, validate, deduplicate, and split real seed photos without mixing them into synthetic test fixtures.

### Step 1: Initialize Real-Data Scaffold
```bash
python scripts/ingest_real_data.py --init-structure
```
This automatically scaffolds:
- `data/real_raw/sound_healthy/`
- `data/real_raw/discolored_spotted/`
- `data/real_raw/broken_cracked/`
- `data/real_raw/shrivelled_damaged/`
- `data/real_raw/annotations_template.csv`

### Step 2: Populate Images
Place collected chickpea photos into their corresponding class subdirectories.
- Recommended file naming: `{class}_{batch_id}_{seed_id}_{view}.jpg` (e.g. `broken_cracked_B1_S042_v1.jpg`).
- If multi-angle shots of the same seed are captured, preserve the identical `seed_id` stem so they are grouped together.

### Step 3: Run Ingestion, Deduplication, and Leakage-Resistant Splitting
```bash
python scripts/ingest_real_data.py --source-dir data/real_raw --splits-dir data/processed/splits_real --provenance-source "Trial Farm Batch 2026" --license "CC BY 4.0"
```
*(Note: `--target-dir` is optional; if omitted, `scripts/ingest_real_data.py` operates in-place without duplicating raw files or causing recursive copying loops. For external camera card staging, specify `--source-dir <card_path> --target-dir data/real_raw`.)*
The ingestion engine:
1. Validates image integrity (detects zero-byte files, corrupted headers, and sub-64px images).
2. Computes MD5 and average perceptual hashes (aHash) to detect exact and near-duplicates.
3. Groups samples by `seed_id` to guarantee zero cross-partition leakage.
4. Generates `data/processed/splits_real/` (`train_manifest.csv`, `val_manifest.csv`, `test_manifest.csv`).
5. Generates `data/provenance_real.json` recording source, license, and class balance.

---

## 5. Retraining & Real Evaluation Commands

Once genuine data splits exist under `data/processed/splits_real/`, execute model training and evaluation:

```bash
# 1. Train Custom CNN on Real Data
python scripts/train.py --config configs/cnn.yaml --splits-dir data/processed/splits_real --name-suffix Real

# 2. Train MobileNetV2 on Real Data
python scripts/train.py --config configs/transfer_learning.yaml --splits-dir data/processed/splits_real --name-suffix Real

# 3. Evaluate Checkpoints on Real Held-Out Test Set
python scripts/evaluate.py --checkpoint artifacts/checkpoints/best_SeedNet-Baseline-Real.pt --splits-dir data/processed/splits_real
python scripts/evaluate.py --checkpoint artifacts/checkpoints/best_MobileNetV2-Transfer-Real.pt --splits-dir data/processed/splits_real
```

---

## 6. Biological Boundary & Regulatory Disclaimer

> [!IMPORTANT]
> **SURFACE OPTICAL BOUNDARY:**  
> SeedVision AI evaluates **external physical appearance only**.
> Standard RGB computer vision **cannot evaluate**:
> - Physiological seed germination capacity (ISTA standard germination paper test).
> - Internal embryo viability (Biochemical Tetrazolium / TZ staining).
> - Seed moisture content (Oven dry method).
> - Internal seed-borne endophytic fungal pathogens.
> - Official agricultural seed lot certification.
