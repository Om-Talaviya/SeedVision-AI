"""Evaluation script for trained SeedVision AI model checkpoints."""

import argparse
from pathlib import Path
import sys
import pandas as pd
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from seedvision.config import load_config
from seedvision.data.dataset import create_dataloaders
from seedvision.evaluation.compare import compare_models
from seedvision.evaluation.metrics import ModelEvaluator
from seedvision.evaluation.plots import (
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_per_class_metrics,
)
from seedvision.training.checkpoints import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained SeedVision AI model checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to .pt checkpoint file (e.g. artifacts/checkpoints/best_SeedNet-Baseline.pt).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to base configuration file.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["test", "val"],
        default="test",
        help="Dataset split to evaluate against (default: test).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Target device for evaluation (default: cpu).",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        print(f"[ERROR] Checkpoint not found at: {ckpt_path.resolve()}")
        sys.exit(1)

    cfg = load_config(args.config)
    device = torch.device(args.device)

    print(f"=== SeedVision AI Model Evaluation ===")
    print(f"Checkpoint: {ckpt_path.name}")
    print(f"Device:     {device}")
    print(f"Split:      {args.split}")

    # Load model and checkpoint metadata
    model, payload = load_checkpoint(ckpt_path, device=device)
    classes = payload.get("classes", cfg.dataset.classes)
    model_name = payload.get("model_name", ckpt_path.stem.replace("best_", ""))

    # Load requested split manifest
    splits_dir = Path(cfg.dataset.splits_dir)
    manifest_file = splits_dir / f"{args.split}_manifest.csv"

    if not manifest_file.is_file():
        print(f"[ERROR] Split manifest not found at: {manifest_file.resolve()}")
        sys.exit(1)

    split_df = pd.read_csv(manifest_file)
    print(f"Evaluating {len(split_df)} samples from {manifest_file.name}...")

    # DataLoaders
    _, _, test_loader = create_dataloaders(
        config=cfg,
        train_df=split_df.iloc[:2],  # dummy
        val_df=split_df.iloc[:2],  # dummy
        test_df=split_df,
    )

    evaluator = ModelEvaluator(
        model=model,
        classes=classes,
        device=device,
        model_name=model_name,
    )

    results = evaluator.evaluate(test_loader)

    # Save metrics JSON & CSV
    metrics_dir = Path(cfg.paths.metrics_dir)
    saved_json = evaluator.save_results(results, metrics_dir)

    # Generate visual figures
    figures_dir = Path(cfg.paths.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    cm_path = figures_dir / f"confusion_matrix_{model_name}.png"
    plot_confusion_matrix(
        cm=results["confusion_matrix"],
        classes=classes,
        output_path=cm_path,
        title=f"Confusion Matrix - {model_name}",
        normalize=False,
    )

    cm_norm_path = figures_dir / f"confusion_matrix_normalized_{model_name}.png"
    plot_confusion_matrix(
        cm=results["confusion_matrix_normalized"],
        classes=classes,
        output_path=cm_norm_path,
        title=f"Normalized Confusion Matrix - {model_name}",
        normalize=True,
    )

    per_class_path = figures_dir / f"per_class_metrics_{model_name}.png"
    plot_per_class_metrics(
        per_class_metrics=results["per_class_metrics"],
        output_path=per_class_path,
        title=f"Per-Class Metrics - {model_name}",
    )

    calib_path = figures_dir / f"calibration_{model_name}.png"
    plot_calibration_curve(
        calibration_bins=results["calibration_bins"],
        ece=results["expected_calibration_error"],
        output_path=calib_path,
        title=f"Reliability Diagram - {model_name}",
    )

    # Update cross-model comparison table and benchmark plot
    compare_df, summary_dict, comp_plot = compare_models(
        metrics_dir=metrics_dir,
        figures_dir=figures_dir,
    )

    print("\n" + "=" * 50)
    print(f"EVALUATION RESULTS: {model_name}")
    print("=" * 50)
    print(f"Overall Accuracy:       {results['accuracy']*100:.2f}%")
    print(f"Macro Precision:        {results['macro_precision']*100:.2f}%")
    print(f"Macro Recall:           {results['macro_recall']*100:.2f}%")
    print(f"Macro F1-Score:         {results['macro_f1']*100:.2f}%")
    print(f"Expected Calib Error:   {results['expected_calibration_error']:.4f}")
    print(f"Inference Latency:      {results['latency_ms_per_sample']:.2f} ms/sample")
    print(f"Throughput:             {results['throughput_samples_per_sec']:.1f} samples/sec")

    print("\n--- Per-Class Performance ---")
    for cls_name, vals in results["per_class_metrics"].items():
        print(
            f"  {cls_name:<20} | Precision: {vals['precision']*100:5.1f}% | "
            f"Recall: {vals['recall']*100:5.1f}% | F1: {vals['f1_score']*100:5.1f}% (N={vals['support']})"
        )

    print(f"\n[Artifacts Created]")
    print(f"  - Metrics JSON:       {saved_json.resolve()}")
    print(f"  - Confusion Matrix:   {cm_path.resolve()}")
    print(f"  - Per-Class Chart:    {per_class_path.resolve()}")
    print(f"  - Calibration Plot:   {calib_path.resolve()}")
    if comp_plot:
        print(f"  - Benchmark Summary:  {comp_plot.resolve()}")


if __name__ == "__main__":
    main()
