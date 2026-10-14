"""CLI prediction and Grad-CAM visualization for a single seed image."""

import argparse
from pathlib import Path
import sys
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from seedvision.inference.predictor import SeedPredictor


def main():
    parser = argparse.ArgumentParser(description="Predict seed quality grade for an image using SeedVision AI.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pt).",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to seed image file to grade.",
    )
    parser.add_argument(
        "--explain-class",
        type=str,
        default=None,
        help="Optional specific class to explain via Grad-CAM (defaults to predicted top-1 class).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/figures",
        help="Directory to save generated Grad-CAM overlay image.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run inference on (default: cpu).",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_file():
        print(f"[ERROR] Image file does not exist: {image_path.resolve()}")
        sys.exit(1)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        print(f"[ERROR] Checkpoint file does not exist: {ckpt_path.resolve()}")
        sys.exit(1)

    predictor = SeedPredictor(checkpoint_path=ckpt_path, device=args.device)

    result = predictor.predict(
        image=image_path,
        explain_class=args.explain_class,
        generate_cam=True,
    )

    print("\n" + "=" * 55)
    print(f"SEEDVISION AI: INFERENCE RESULT")
    print("=" * 55)
    print(f"Sample:               {image_path.name}")
    print(f"Model:                {result.model_name} ({result.architecture})")
    print(f"Predicted Quality:    {result.predicted_class.upper()}")
    print(f"Confidence Score:     {result.confidence_score * 100:.2f}%")
    print(f"Inference Latency:    {result.inference_latency_ms:.1f} ms")

    if result.is_uncertain:
        print(f"\n[!] UNCERTAINTY ADVISORY: {result.uncertainty_reason}")

    print("\n--- Probability Distribution (Top-K) ---")
    for pred in result.top_k:
        marker = "-> " if pred.class_name == result.predicted_class else "   "
        print(f"{marker}Rank {pred.rank}: {pred.class_name:<20} {pred.percentage_str:>6}")

    # Save Grad-CAM visualization
    if result.overlay_rgb is not None:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cam_filename = f"gradcam_{image_path.stem}_{result.predicted_class}.png"
        cam_save_path = out_dir / cam_filename
        Image.fromarray(result.overlay_rgb).save(cam_save_path)
        print(f"\n[Grad-CAM Visualization Saved]")
        print(f"  Target Class: {result.gradcam_target_class}")
        print(f"  Saved Image:  {cam_save_path.resolve()}")
    elif result.gradcam_error:
        print(f"\n[Grad-CAM Notice]: {result.gradcam_error}")

    print("\n" + "-" * 55)
    print(result.scientific_notice)
    print("-" * 55)


if __name__ == "__main__":
    main()
