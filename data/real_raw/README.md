# Genuine Agricultural Seed Photography Ingestion

This directory is designated exclusively for **genuine, real-world photographs of chickpea seeds** (*Cicer arietinum L.*).

## Classification Schema
Place single-seed macro photographs into the corresponding quality class directory:
1. `sound_healthy/`: Intact testa, normal angular/spherical morphology, uniform color, zero cracks or necrotic spots.
2. `discolored_spotted/`: Fungal lesions, necrotic stains, moisture damage, or dark weathered patches.
3. `broken_cracked/`: Mechanical cracks, cotyledon split, or chipped edge caused by threshing or handling stress.
4. `shrivelled_damaged/`: Immature, wrinkled, hollowed, or insect-bored seeds (e.g. pulse beetle *Callosobruchus*).

## Recommended Photography Setup
- **Lighting**: Diffuse 5000K–6500K LED ring light to eliminate harsh shadows and specular hotspots.
- **Background**: Neutral matte 18% gray card or non-reflective white laboratory tray.
- **Resolution**: Single-seed macro capture with seed occupying at least 60% of frame (minimum 128x128 px).
- **Naming Convention**: `{class_name}_{batch_id}_{seed_id}_{view_idx}.jpg` (e.g., `broken_cracked_B1_S042_v1.jpg`).

## Ingestion Commands
To audit duplicates and generate leakage-free splits in-place:
```bash
python scripts/ingest_real_data.py --source-dir data/real_raw --splits-dir data/processed/splits_real --provenance-source "University Trial Farm"
```

If staging from an external raw camera dump into `data/real_raw`:
```bash
python scripts/ingest_real_data.py --source-dir /path/to/camera_dump --target-dir data/real_raw --splits-dir data/processed/splits_real --provenance-source "University Trial Farm"
```
