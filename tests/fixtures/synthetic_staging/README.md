# Synthetic Staging Test Fixtures

This directory contains synthetic demonstration images relocated from data/real_staging/incoming/.

## Fixture Items
- chickpea_desi_JG11_SPECIMEN001_v1.jpg
- chickpea_desi_JG11_SPECIMEN001_v2.jpg
- synthetic_staging_manifest.csv

## Policy & Safeguards
- These images are **synthetic test fixtures** intended solely for unit and integration testing of multi-view staging, perceptual hashing, and pipeline validation.
- All fabricated claims (such as genuine field provenance, cultivar identity, expert reviewers, and quality labels) have been stripped.
- Under system safeguards, these fixtures are explicitly marked is_synthetic = True and **cannot be approved** as genuine agricultural samples in data/real_staging.
