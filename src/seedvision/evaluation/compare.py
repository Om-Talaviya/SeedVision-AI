"""Cross-architecture comparison and benchmarking for SeedVision AI."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import logging
import pandas as pd

from seedvision.evaluation.plots import plot_model_comparison

logger = logging.getLogger(__name__)


def compare_models(
    metrics_dir: Union[str, Path] = "artifacts/metrics",
    figures_dir: Union[str, Path] = "artifacts/figures",
) -> Tuple[pd.DataFrame, Dict[str, Any], Optional[Path]]:
    """Scan metrics directory, parse evaluation JSONs, and produce unified comparison.

    Returns:
        Tuple of (comparison_dataframe, summary_dict, plot_path)
    """
    m_dir = Path(metrics_dir)
    f_dir = Path(figures_dir)
    f_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(m_dir.glob("eval_*.json"))
    if not json_files:
        logger.warning(f"No evaluation JSON files found in {metrics_dir}")
        empty_df = pd.DataFrame(
            columns=[
                "Model",
                "Accuracy (%)",
                "Macro F1 (%)",
                "Calibration ECE",
                "Latency (ms/sample)",
                "Throughput (sample/s)",
                "Total Params",
                "Size (MB)",
            ]
        )
        return empty_df, {}, None

    records = []
    raw_models_data = []

    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            raw_models_data.append(data)

            params = data.get("parameters", {})
            total_params = params.get("total_parameters", 0)
            size_mb = params.get("estimated_size_mb", 0.0)

            records.append(
                {
                    "Model": data.get("model_name", jf.stem.replace("eval_", "")),
                    "Accuracy (%)": round(data.get("accuracy", 0.0) * 100, 2),
                    "Macro F1 (%)": round(data.get("macro_f1", 0.0) * 100, 2),
                    "Calibration ECE": round(data.get("expected_calibration_error", 0.0), 4),
                    "Latency (ms/sample)": round(data.get("latency_ms_per_sample", 0.0), 2),
                    "Throughput (sample/s)": round(data.get("throughput_samples_per_sec", 0.0), 1),
                    "Total Params": f"{total_params / 1e6:.2f}M" if total_params >= 1e6 else f"{total_params:,}",
                    "Size (MB)": f"{size_mb:.2f} MB",
                    "Evaluation Type": data.get("evaluation_type", "synthetic_pipeline_verification"),
                }
            )
        except Exception as e:
            logger.error(f"Error loading {jf}: {e}")

    df = pd.DataFrame(records)

    # Save summary files
    csv_path = m_dir / "model_comparison_summary.csv"
    df.to_csv(csv_path, index=False)

    has_real = any(data.get("is_real_world_validated") is True for data in raw_models_data)
    all_real = (
        all(data.get("is_real_world_validated") is True for data in raw_models_data)
        if raw_models_data
        else False
    )

    summary_dict = {
        "num_models_evaluated": len(df),
        "evaluation_type": "real_world_test_evaluation" if all_real else ("mixed" if has_real else "synthetic_pipeline_verification"),
        "dataset_provenance": "manually_annotated_genuine_photographs" if all_real else ("mixed" if has_real else "synthetic_procedural_fixture"),
        "is_real_world_validated": all_real,
        "scientific_notice": (
            "Metrics reflect empirical evaluation on genuine agricultural pulse test photographs."
            if all_real
            else (
                "Metrics contain evaluations on real-world pulses as well as synthetic calibration fixtures."
                if has_real
                else "Metrics reflect software pipeline convergence on synthetic procedural fixtures. Real-world agronomic performance is pending certified pulse dataset acquisition."
            )
        ),
        "models": records,
        "best_accuracy_model": df.sort_values(by="Accuracy (%)", ascending=False).iloc[0]["Model"]
        if not df.empty
        else None,
        "fastest_model": df.sort_values(by="Latency (ms/sample)", ascending=True).iloc[0]["Model"]
        if not df.empty
        else None,
    }

    summary_json_path = m_dir / "model_comparison_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)

    # Generate comparison plot if at least 1 model is available
    plot_path = None
    if raw_models_data:
        plot_path = f_dir / "model_comparison_benchmark.png"
        plot_model_comparison(raw_models_data, plot_path)

    return df, summary_dict, plot_path
