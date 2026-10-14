# SeedVision AI: Model Evaluation Framework

## 1. Evaluation Methodology
Both deep learning models are evaluated on the identical held-out test partition (`data/processed/splits/test_manifest.csv`) to guarantee scientific fairness. The test dataset is never used during parameter optimization, early stopping, or learning rate scheduling.

## 2. Quantitative Evaluation Metrics
The evaluation suite measures performance across multiple operational dimensions:

1. **Overall Accuracy**: Fraction of correctly categorized seed images across all classes.
2. **Macro Precision, Recall, and F1-Score**: Unweighted arithmetic mean across all 4 classes to penalize poor performance on minority or difficult classes.
3. **Per-Class Metrics**: Precision, Recall, F1, and sample support calculated individually for:
   - `sound_healthy`
   - `discolored_spotted`
   - `broken_cracked`
   - `shrivelled_damaged`
4. **Confusion Matrix**: Full count matrix and row-normalized recall matrix revealing directional classification confusions (e.g. cracked vs healthy).
5. **Expected Calibration Error (ECE)**: Discrepancy between model confidence and observed empirical accuracy across confidence bins:
   $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
6. **Inference Latency & Throughput**: Average single-sample latency in milliseconds on CPU and throughput in samples per second.
7. **Model Footprint**: Total parameters, trainable parameters, and memory footprint in Megabytes (MB).

## 3. Execution Commands
To evaluate saved checkpoints on the held-out test split:
```bash
# Evaluate Model A (Custom CNN)
python scripts/evaluate.py --checkpoint artifacts/checkpoints/best_SeedNet-Baseline.pt

# Evaluate Model B (MobileNetV2 Transfer)
python scripts/evaluate.py --checkpoint artifacts/checkpoints/best_MobileNetV2-Transfer.pt
```

Evaluation outputs are archived in `artifacts/metrics/` and visual diagnostic figures in `artifacts/figures/`.
