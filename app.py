"""SeedVision AI: Deep Learning-Based Agricultural Seed Quality Grader.

Streamlit Web Application for interactive inference, explainability, and evaluation.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from seedvision.config import load_config
from seedvision.evaluation.compare import compare_models
from seedvision.inference.predictor import (
    SCIENTIFIC_LIMITATION_NOTICE,
    PredictionResult,
    SeedPredictor,
)
from seedvision.inference.validation import InferenceInputValidator

# Page Configuration
st.set_page_config(
    page_title="SeedVision AI | Seed Quality Grader",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Modern Laboratory & Research Aesthetic
st.markdown(
    """
    <style>
    /* Main Theme Variables */
    :root {
        --primary-green: #2e7d32;
        --dark-green: #1b5e20;
        --light-green: #e8f5e9;
        --accent-gold: #f9a825;
        --card-bg: #ffffff;
        --text-primary: #1e293b;
        --text-muted: #64748b;
        --border-color: #e2e8f0;
    }

    /* Overall Container */
    .stApp {
        background-color: #f8fafc;
        color: var(--text-primary);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid var(--border-color);
    }

    /* Hero Header Card */
    .hero-card {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%);
        color: white;
        padding: 2.2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(27, 94, 32, 0.12);
    }
    .hero-card h1 {
        color: white !important;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.5px;
    }
    .hero-card p {
        color: #e8f5e9;
        font-size: 1.05rem;
        margin: 0;
        max-width: 800px;
        line-height: 1.5;
    }

    /* Metric & Feature Cards */
    .feature-card {
        background-color: white;
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 1.4rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .feature-card h3 {
        color: var(--dark-green);
        margin-top: 0;
        font-size: 1.15rem;
        font-weight: 600;
    }

    /* Quality Class Badges */
    .badge {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .badge-healthy { background-color: #dcfce7; color: #166534; border: 1px solid #bbf7d0; }
    .badge-spotted { background-color: #fef9c3; color: #854d0e; border: 1px solid #fef08a; }
    .badge-cracked { background-color: #ffedd5; color: #9a3412; border: 1px solid #fed7aa; }
    .badge-shrivelled { background-color: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }

    /* Advisory Box */
    .advisory-box {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 1rem 1.2rem;
        border-radius: 6px;
        margin: 1.2rem 0;
        color: #92400e;
        font-size: 0.92rem;
    }

    /* Scientific Disclaimer Box */
    .scientific-disclaimer {
        background-color: #f1f5f9;
        border-left: 4px solid #64748b;
        padding: 0.9rem 1.1rem;
        border-radius: 6px;
        margin-top: 1.5rem;
        color: #475569;
        font-size: 0.85rem;
        line-height: 1.45;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_cached_predictor(checkpoint_path_str: str) -> SeedPredictor:
    """Cached loader for model predictor to avoid reloading weights repeatedly."""
    return SeedPredictor(checkpoint_path=checkpoint_path_str, device="cpu")


def get_available_checkpoints() -> list:
    """Scan artifacts/checkpoints for trained .pt models."""
    ckpt_dir = Path("artifacts/checkpoints")
    if not ckpt_dir.exists():
        return []
    return sorted(list(ckpt_dir.glob("best_*.pt")))


def main():
    cfg = load_config("configs/default.yaml")

    # Sidebar Navigation
    st.sidebar.image(
        "https://images.unsplash.com/photo-1574943320219-553eb213f72d?auto=format&fit=crop&w=400&q=80",
        caption="SeedVision AI Laboratory",
        use_container_width=True,
    )
    st.sidebar.markdown("## **SeedVision AI**")
    st.sidebar.markdown(f"**Target Pulse:** `{cfg.dataset.seed_type}`")

    page = st.sidebar.radio(
        "Navigation",
        [
            "Home & Overview",
            "Seed Quality Analysis",
            "Model Evaluation & Benchmark",
            "Methodology & Limitations",
        ],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Availability")
    checkpoints = get_available_checkpoints()
    if checkpoints:
        st.sidebar.success(f"{len(checkpoints)} model checkpoint(s) ready")
        for cp in checkpoints:
            is_real = "real" in cp.name.lower()
            tag = "🌱 Real-Data" if is_real else "🧪 Synthetic"
            st.sidebar.markdown(f"- `{cp.name}` <small style='color: {'#166534' if is_real else '#0369a1'};'>({tag})</small>", unsafe_allow_html=True)
    else:
        st.sidebar.warning("No checkpoints found. Train models via CLI or scripts.")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style="background-color: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 6px; padding: 0.75rem; margin-top: 0.5rem; font-size: 0.8rem; color: #166534;">
            <strong>🧪 SOFTWARE VERIFICATION MODE:</strong><br>
            Loaded with 240 synthetic procedural fixtures. Real-world agronomic benchmarks pending certified field collection.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.caption("SeedVision AI v1.0.0 | Academic & DLA Research Platform")

    # -------------------------------------------------------------
    # PAGE 1: HOME & OVERVIEW
    # -------------------------------------------------------------
    if page == "Home & Overview":
        st.markdown(
            """
            <div class="hero-card">
                <h1>SeedVision AI</h1>
                <p>An Explainable Deep Learning System for Agricultural Seed Quality Inspection, Defect Detection, and Grad-CAM Visual Attribution.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.info(
            "🔬 **Software Verification Mode:** Current models are trained and evaluated on 240 procedural synthetic fixtures "
            "to rigorously test tensor transformations, gradient hooks, and latency profiling. Real-world agricultural benchmarks "
            "remain pending certified field pulse dataset collection."
        )

        col1, col2 = st.columns([1.8, 1.2])

        with col1:
            st.markdown("### Agricultural Problem & Objective")
            st.write(
                """
                Physical seed purity and external morphological integrity directly impact grain market value, 
                grading standards (e.g. AGMARK and USDA GIPSA), and processing suitability. Traditional quality grading 
                relies on labor-intensive, subjective manual visual inspection by grain inspectors.
                
                **SeedVision AI** demonstrates how deep convolutional neural networks can automate the visual 
                classification of physical seed defects while providing **transparent, interpretable decision attribution** 
                through Gradient-weighted Class Activation Mapping (Grad-CAM).
                """
            )

            st.markdown("### Inspection Workflow")
            st.markdown(
                """
                1. **Digital Acquisition**: Single-seed macro photography on neutral background.
                2. **Validation & Normalization**: Dimension check, RGB standardization, and ImageNet normalization.
                3. **Deep Learning Inference**: Dual-path evaluation comparing Custom CNN baseline vs Fine-Tuned MobileNetV2.
                4. **Calibrated Quality Grading**: Probability distribution with automated uncertainty flagging.
                5. **Grad-CAM Attribution**: Spatial gradient heatmaps highlighting specific defect regions (cracks, spots, wrinkles).
                """
            )

        with col2:
            st.markdown("### Supported Quality Classes")
            st.markdown(
                """
                <div class="feature-card">
                    <span class="badge badge-healthy">sound_healthy</span>
                    <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #475569;">
                        Intact seed coat, characteristic angular-spherical chickpea morphology, uniform pigmentation, zero visible lesions.
                    </p>
                </div>
                <div class="feature-card">
                    <span class="badge badge-spotted">discolored_spotted</span>
                    <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #475569;">
                        Surface fungal spots, necrotic lesions, moisture discoloration, or dark weathered seed coat patches.
                    </p>
                </div>
                <div class="feature-card">
                    <span class="badge badge-cracked">broken_cracked</span>
                    <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #475569;">
                        Mechanical fracture lines, split cotyledons, or severed outer testa caused by mechanical threshing.
                    </p>
                </div>
                <div class="feature-card">
                    <span class="badge badge-shrivelled">shrivelled_damaged</span>
                    <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #475569;">
                        Immature or insect-punctured grains, wrinkled non-convex morphology, or hollowed seed mass.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <div class="scientific-disclaimer">
                <strong>CRITICAL SCIENTIFIC LIMITATION NOTICE:</strong><br>
                {SCIENTIFIC_LIMITATION_NOTICE}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------
    # PAGE 2: SEED QUALITY ANALYSIS
    # -------------------------------------------------------------
    elif page == "Seed Quality Analysis":
        st.markdown("## 🔍 Seed Quality Analysis & Grad-CAM Interpretability")
        st.write(
            "Upload an image of a seed sample to perform automated visual quality grading and generate class activation heatmaps."
        )

        checkpoints = get_available_checkpoints()
        if not checkpoints:
            st.error(
                "No trained model checkpoints were found in `artifacts/checkpoints/`.\n\n"
                "To train the models on your machine, run the following commands in PowerShell/terminal:\n"
                "```bash\n"
                "python scripts/prepare_dataset.py\n"
                "python scripts/train.py --config configs/cnn.yaml\n"
                "python scripts/train.py --config configs/transfer_learning.yaml\n"
                "```"
            )
            return

        # Model Selection Bar
        col_m1, col_m2 = st.columns([2, 1])
        with col_m1:
            selected_ckpt = st.selectbox(
                "Select Trained Model Checkpoint",
                checkpoints,
                format_func=lambda x: f"{x.name} ({'Baseline CNN' if 'SeedNet' in x.name else 'MobileNetV2 Transfer'})",
            )
        with col_m2:
            st.info(f"Active Checkpoint: `{selected_ckpt.name}`")

        # Load Predictor
        try:
            predictor = get_cached_predictor(str(selected_ckpt))
        except Exception as e:
            st.error(f"Failed to load checkpoint: {e}")
            return

        # Upload & Sample Section
        col_up1, col_up2 = st.columns([1.5, 1])
        with col_up1:
            uploaded_file = st.file_uploader(
                "Upload Seed Image (.jpg, .jpeg, .png, .webp, .bmp)",
                type=["jpg", "jpeg", "png", "webp", "bmp"],
            )

        with col_up2:
            st.markdown("##### Or Select from Existing Test Samples:")
            test_samples = list(Path("data/raw").glob("*/*.jpg"))
            sample_choice = None
            if test_samples:
                sample_map = {f"{p.parent.name} / {p.name}": p for p in test_samples[:16]}
                selected_sample_key = st.selectbox("Quick Demo Samples", ["-- Select a demo image --"] + list(sample_map.keys()))
                if selected_sample_key != "-- Select a demo image --":
                    sample_choice = sample_map[selected_sample_key]

        # Resolve image to analyze
        target_image = None
        if uploaded_file is not None:
            validator = InferenceInputValidator()
            val_result = validator.validate_upload(uploaded_file)
            if not val_result.is_valid:
                st.error(f"Image Validation Error: {val_result.error_message}")
                return
            target_image = val_result.image
            source_desc = f"Uploaded File: `{uploaded_file.name}` ({val_result.width}x{val_result.height}px)"
        elif sample_choice is not None:
            target_image = Image.open(sample_choice).convert("RGB")
            source_desc = f"Demo Sample: `{sample_choice.parent.name} / {sample_choice.name}`"

        if target_image is not None:
            st.markdown("---")
            st.markdown(f"#### Analyzing: {source_desc}")

            # Explainability Class Target Selector
            col_ctrl1, col_ctrl2 = st.columns([2, 1])
            with col_ctrl1:
                target_explain_class = st.selectbox(
                    "Grad-CAM Attribution Target Class",
                    ["[Auto: Explain Top Predicted Class]"] + predictor.classes,
                    help="Select which class hypothesis the Grad-CAM activation heatmap should explain.",
                )
                explain_param = None if target_explain_class.startswith("[Auto") else target_explain_class

            with col_ctrl2:
                run_pred = st.button("Run Inspection & Explainability", type="primary", use_container_width=True)

            # Perform prediction
            with st.spinner("Processing deep learning forward & backward hooks..."):
                res: PredictionResult = predictor.predict(
                    image=target_image,
                    explain_class=explain_param,
                    generate_cam=True,
                )

            # Display Results
            st.markdown("### Inspection Results")

            # Result Metrics Row
            badge_class = f"badge-{res.predicted_class.split('_')[0]}"
            if "healthy" in res.predicted_class:
                badge_class = "badge-healthy"
            elif "spotted" in res.predicted_class:
                badge_class = "badge-spotted"
            elif "cracked" in res.predicted_class:
                badge_class = "badge-cracked"
            else:
                badge_class = "badge-shrivelled"

            r_col1, r_col2, r_col3, r_col4 = st.columns(4)
            with r_col1:
                st.markdown(
                    f"**Predicted Grade:**<br><span class='badge {badge_class}' style='font-size:1.1rem;'>{res.predicted_class.upper()}</span>",
                    unsafe_allow_html=True,
                )
            with r_col2:
                st.metric("Model Confidence", f"{res.confidence_score * 100:.1f}%")
            with r_col3:
                st.metric("Inference Latency", f"{res.inference_latency_ms:.1f} ms")
            with r_col4:
                st.metric("Model Architecture", res.architecture)

            # Uncertainty Advisory if flagged
            if res.is_uncertain:
                st.markdown(
                    f"""
                    <div class="advisory-box">
                        <strong>⚠️ UNCERTAINTY ADVISORY:</strong><br>
                        {res.uncertainty_reason}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Probability Distribution Bar Chart
            st.markdown("#### Class Probability Distribution (Softmax)")
            prob_df = pd.DataFrame(
                [{"Class": p.class_name, "Probability (%)": p.probability * 100} for p in res.top_k]
            ).sort_values(by="Probability (%)", ascending=True)
            st.bar_chart(prob_df.set_index("Class"), color="#2e7d32")

            # Grad-CAM Visualization Gallery
            st.markdown("#### Grad-CAM Explainability Triplet")
            st.caption(
                f"Attribution computed for class hypothesis: **`{res.gradcam_target_class}`** on target convolutional layer."
            )

            cam_col1, cam_col2, cam_col3 = st.columns(3)
            with cam_col1:
                st.image(target_image, caption="1. Original Seed Image", use_container_width=True)
            with cam_col2:
                if res.colored_heatmap_rgb is not None:
                    st.image(res.colored_heatmap_rgb, caption="2. Grad-CAM Activation Heatmap", use_container_width=True)
                else:
                    st.warning(res.gradcam_error or "Heatmap unavailable.")
            with cam_col3:
                if res.overlay_rgb is not None:
                    st.image(res.overlay_rgb, caption="3. Blended Decision Overlay", use_container_width=True)
                else:
                    st.warning(res.gradcam_error or "Overlay unavailable.")

            st.info(
                "💡 **Grad-CAM Interpretation Note:** Bright yellow and red regions indicate image areas that positively contributed to "
                f"the network scoring this seed as '{res.gradcam_target_class}'. Grad-CAM highlights associative feature correlations; "
                "it does not verify biochemical seed viability."
            )

    # -------------------------------------------------------------
    # PAGE 3: MODEL EVALUATION & BENCHMARK
    # -------------------------------------------------------------
    elif page == "Model Evaluation & Benchmark":
        st.markdown("## 📊 Scientific Model Evaluation & Benchmark")
        st.write(
            "Empirical performance metrics, confusion matrices, and calibration evaluation on held-out test datasets."
        )

        st.markdown(
            """
            <div class="advisory-box">
                <strong>🔬 SCIENTIFIC VALIDATION STATUS — SYNTHETIC CALIBRATION FIXTURES:</strong><br>
                The 100.0% test accuracy shown below represents model convergence and linear separability on 
                <strong>procedural synthetic calibration fixtures</strong> (36 test samples). It demonstrates that the backpropagation 
                and feature extraction pipelines function correctly. This metric must <strong>NOT</strong> be interpreted as 
                empirical real-world grading accuracy on physical farm crops. Real-data benchmarks are pending certified pulse image collection.
            </div>
            """,
            unsafe_allow_html=True,
        )

        metrics_dir = Path("artifacts/metrics")
        figures_dir = Path("artifacts/figures")

        comp_df, summary_dict, comp_plot = compare_models(metrics_dir, figures_dir)

        if comp_df.empty:
            st.warning(
                "No evaluation benchmark records found. Run model evaluation using the CLI:\n\n"
                "```bash\n"
                "python scripts/evaluate.py --checkpoint artifacts/checkpoints/best_SeedNet-Baseline.pt\n"
                "python scripts/evaluate.py --checkpoint artifacts/checkpoints/best_MobileNetV2-Transfer.pt\n"
                "```"
            )
            return

        st.markdown("### Cross-Architecture Benchmark Summary")
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        if comp_plot and comp_plot.exists():
            st.image(str(comp_plot), caption="Accuracy, Macro F1, and Latency Comparison", use_container_width=True)

        st.markdown("---")
        st.markdown("### Detailed Inspection by Architecture")

        eval_files = list(metrics_dir.glob("eval_*.json"))
        if eval_files:
            chosen_eval_file = st.selectbox(
                "Select Architecture Evaluation Artifact",
                eval_files,
                format_func=lambda x: x.stem.replace("eval_", ""),
            )

            import json

            with open(chosen_eval_file, "r", encoding="utf-8") as f:
                eval_data = json.load(f)

            m_name = eval_data.get("model_name", "Model")

            # Metrics row
            e_col1, e_col2, e_col3, e_col4 = st.columns(4)
            with e_col1:
                st.metric("Test Accuracy", f"{eval_data.get('accuracy', 0)*100:.2f}%")
            with e_col2:
                st.metric("Macro F1-Score", f"{eval_data.get('macro_f1', 0)*100:.2f}%")
            with e_col3:
                st.metric("Calibration ECE", f"{eval_data.get('expected_calibration_error', 0):.4f}")
            with e_col4:
                st.metric("CPU Latency", f"{eval_data.get('latency_ms_per_sample', 0):.1f} ms")

            # Display Confusion Matrix & Charts
            fig_col1, fig_col2 = st.columns(2)
            with fig_col1:
                cm_img = figures_dir / f"confusion_matrix_{m_name}.png"
                if cm_img.exists():
                    st.image(str(cm_img), caption="Confusion Matrix (Counts)", use_container_width=True)

            with fig_col2:
                cm_norm_img = figures_dir / f"confusion_matrix_normalized_{m_name}.png"
                if cm_norm_img.exists():
                    st.image(str(cm_norm_img), caption="Normalized Confusion Matrix (Recall Rates)", use_container_width=True)

            fig_col3, fig_col4 = st.columns(2)
            with fig_col3:
                per_class_img = figures_dir / f"per_class_metrics_{m_name}.png"
                if per_class_img.exists():
                    st.image(str(per_class_img), caption="Per-Class Precision, Recall & F1-Score", use_container_width=True)

            with fig_col4:
                calib_img = figures_dir / f"calibration_{m_name}.png"
                if calib_img.exists():
                    st.image(str(calib_img), caption="Confidence Calibration Reliability Curve", use_container_width=True)

    # -------------------------------------------------------------
    # PAGE 4: METHODOLOGY & LIMITATIONS
    # -------------------------------------------------------------
    elif page == "Methodology & Limitations":
        st.markdown("## 📖 Scientific Methodology & Research Principles")

        tabs = st.tabs(["Agricultural Taxonomy", "Deep Learning Architectures", "Grad-CAM Derivation", "Limitations & Ethics", "Dataset Provenance & Ingestion"])

        with tabs[0]:
            st.markdown("### Agricultural Seed Defect Taxonomy")
            st.write(
                """
                Grain legumes and pulses such as chickpea (*Cicer arietinum L.*) undergo strict physical classification:
                - **Healthy**: Uniform testa thickness, intact hilum, spherical-to-angular shape.
                - **Discolored / Stained**: Pre-harvest rainfall or Aspergillus/Fusarium colonization inducing dark necrotic pigmentation.
                - **Broken / Cracked**: Mechanical impact during rotary combine harvesting and post-harvest de-hulling causing seed coat fissures.
                - **Shrivelled / Damaged**: Immature drought stress or bruchid weevil (*Callosobruchus maculatus*) bore holes.
                """
            )

        with tabs[1]:
            st.markdown("### Model Architectures & Optimization")
            st.markdown(
                r"""
                - **Baseline CNN (SeedNet)**:
                  - 4-stage convolutional hierarchy (32 -> 64 -> 128 -> 256 channels).
                  - LeakyReLU activation ($\alpha = 0.1$), Batch Normalization, and Max Pooling.
                  - Fully convolutional feature extraction with Adaptive Average Pooling and Dropout regularized linear head.
                - **Transfer Learning (MobileNetV2)**:
                  - Inverted residual bottlenecks with depthwise separable convolutions.
                  - ImageNet pre-trained representations fine-tuned with a lower learning rate ($3 \times 10^{-4}$).
                  - Optimized for edge CPU deployment and mobile grading terminals.
                - **Optimization Strategy**:
                  - Cross-entropy loss with label smoothing ($\epsilon = 0.05$).
                  - Cosine annealing learning rate scheduler with warm restarts.
                  - Gradient clipping ($\|g\| \le 1.0$) and Early Stopping ($patience = 6$).
                """
            )

        with tabs[2]:
            st.markdown("### Grad-CAM Mathematical Formulation")
            st.markdown(
                r"""
                Grad-CAM computes the gradient of the unnormalized class score $y^c$ with respect to feature activation maps $A^k$:
                $$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^u \sum_{j=1}^v \frac{\partial y^c}{\partial A_{ij}^k}$$
                
                The class-discriminative localization map $L_{\text{Grad-CAM}}^c$ is computed as a rectified linear combination:
                $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$
                
                Applying the $\text{ReLU}$ operation ensures that only features with a positive influence on the target class contribute to the visualization.
                """
            )

        with tabs[3]:
            st.markdown("### Scientific Limitations & Intended Use")
            st.markdown(
                f"""
                <div class="scientific-disclaimer">
                    <h4>Responsible AI & Agricultural Governance</h4>
                    <p>{SCIENTIFIC_LIMITATION_NOTICE}</p>
                    <ul>
                        <li><strong>Surface Appearance vs Internal Chemistry:</strong> Optical RGB sensors can only inspect external reflection; internal embryo vitality requires biochemical assays (e.g. Tetrazolium staining).</li>
                        <li><strong>Non-Certification:</strong> Predictions must be corroborated by certified seed testing laboratories before regulatory seed lot marketing.</li>
                        <li><strong>Out-of-Distribution Warning:</strong> The models are calibrated exclusively for the specified crop and cannot grade unfamiliar pulse species.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with tabs[4]:
            st.markdown("### Dataset Provenance & Real-Data Ingestion Protocol")
            st.markdown(
                """
                #### Current Provenance Status
                - **Current Images (data/raw):** 240 procedural synthetic fixtures (60/class).
                - **Purpose:** Rigorous verification of software pipelines, PyTorch tensor transformations, Grad-CAM hooks, and latency benchmarking.
                - **Real-World Agronomic Validation:** Currently pending empirical pulse collection from agricultural trial stations.
                
                #### Protocol for Ingesting Real Agricultural Photos
                Researchers and laboratory technicians can plug in genuine seed photography datasets using the automated ingestion CLI:
                1. Place physical seed images into `data/real_raw/<class_name>/` (`sound_healthy`, `discolored_spotted`, `broken_cracked`, `shrivelled_damaged`).
                2. Run the ingestion command:
                   ```bash
                   python scripts/prepare_dataset.py --data-dir data/real_raw
                   ```
                3. Retrain and benchmark models:
                   ```bash
                   python scripts/train.py --config configs/cnn.yaml
                   python scripts/train.py --config configs/transfer_learning.yaml
                   python scripts/evaluate.py --checkpoint artifacts/checkpoints/best_SeedNet-Baseline.pt
                   ```
                For detailed cryptographic hash analysis and split audits, consult `docs/DATASET_AUDIT.md`.
                """
            )


if __name__ == "__main__":
    main()
