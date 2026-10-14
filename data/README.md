# SeedVision AI Dataset Guide

## Target Crop & Agricultural Seed
**Target Crop:** Chickpea (*Cicer arietinum L.*) & Grain Legume Seeds.

Chickpea is a critical global pulse crop with well-defined physical grading guidelines issued by agricultural bodies (e.g., USDA Grain Inspection, Packers and Stockyards Administration (GIPSA), and AGMARK India). Visual grading evaluates external seed coat integrity, fungal discoloration, insect damage, and morphological deformation.

## Quality Grading Classes
The dataset is structured into 4 scientifically justifiable, visually observable classes:
1. **`sound_healthy`**: Intact seed coat, typical angular/spherical shape, uniform natural pigmentation (cream/brown depending on desi/kabuli variety), no visible cracks or punctures.
2. **`discolored_spotted`**: Seed coat exhibits necrotic staining, fungal discoloration, dark patches, or weathering due to excess moisture during harvest.
3. **`broken_cracked`**: Mechanical cleavage, cracked seed coat, split cotyledons resulting from mechanical threshing or handling stress.
4. **`shrivelled_damaged`**: Immature or insect-bored seeds, withered/wrinkled morphology, hollowed seed mass, or stunted development.

## Scientific Limitation Notice
Visual image classification evaluates **external physical appearance only**. It **does NOT** test or establish:
- Physiological seed germination rate (%)
- Internal embryo viability (e.g., Tetrazolium test results)
- Moisture content percentage
- Genetic purity or pathogen-free certification
- Official regulatory seed lot certification

Never present an inference result as official laboratory certification.
