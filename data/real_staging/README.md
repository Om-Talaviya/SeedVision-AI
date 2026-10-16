# SeedVision AI — Real-Image Staging & Intake Scaffolding

This directory serves as the **isolated, reversible staging area** for genuine agricultural seed photographs.
No images placed here are automatically imported into approved training data.

## Photography Submission Requirements
1. **Single-Seed Framing**: Single seed occupying $\ge 60\%$ of the visual frame.
2. **Resolution**: Minimum 128x128 pixels (native macro capture recommended $\ge 512\times 512$ px).
3. **Lighting**: Diffuse 5000K–6500K LED illumination (ring light or diffuser box) to prevent harsh specular highlights.
4. **Background**: Neutral non-reflective matte gray (18% gray card) or white laboratory tray.
5. **Multi-View Grouping**: When capturing multiple views of the same seed, assign an identical `specimen_id` (e.g. `SPECIMEN-2026-001`) with distinct `view_id` (`v1`, `v2`, `v3`).

## Mandatory Metadata Fields in `staging_manifest.csv`
- `filename`: Local image file name inside `incoming/`
- `specimen_id`: Unique identifier for the individual physical seed
- `view_id`: Camera angle / rotation view index (`v1`, `v2`, `v3`)
- `candidate_label`: Target class (`sound_healthy`, `discolored_spotted`, `broken_cracked`, `shrivelled_damaged`)
- `review_status`: `pending`, `approved`, or `rejected` (must default to `pending`)
- `confidence`: `high`, `medium`, or `low`
- `source_provenance`: Collection institute, field plot, or genebank origin
- `variety`: Botanical cultivar / variety name
- `reviewer`: Human expert / technician identifier
- `diagnostic_notes`: Documented morphological evidence supporting the label

## Validation Command
```powershell
python scripts/stage_real_intake.py --staging-dir data/real_staging --manifest data/real_staging/staging_manifest.csv --dry-run
```
