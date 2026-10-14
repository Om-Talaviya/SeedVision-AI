# SeedVision AI: Execution Flows & Sequence Workflows

This document illustrates the operational execution flows of SeedVision AI across data ingestion, model training, evaluation, real-time inference, and user web interactions.

---

## 1. Dataset Preparation & Leakage-Resistant Splitting Flow

```mermaid
flowchart TD
    Start([Execute scripts/prepare_dataset.py]) --> Discover[Discover Class Directories in data/raw/]
    Discover --> ValidateImages[ImageValidator: Decode & Hash Image]
    ValidateImages --> CheckCorrupt{Is Image Valid?}
    CheckCorrupt -- No --> LogCorrupt[Record in Corrupted Files List]
    CheckCorrupt -- Yes --> CheckDuplicate{SHA-256 in Seen Hashes?}
    CheckDuplicate -- Yes --> LogDuplicate[Record Duplicate File Pair]
    CheckDuplicate -- No --> AddManifest[Append to dataset_manifest.csv]

    AddManifest --> SplitGroups[Group by seed_id per Class]
    SplitGroups --> Partition[Partition 70% Train, 15% Val, 15% Test]
    Partition --> SaveCSV[Save train_manifest, val_manifest, test_manifest]
    SaveCSV --> End([Ready for Training & Evaluation])
```

---

## 2. Model Training & Checkpoint Lifecycle Flow

```mermaid
flowchart TD
    StartTrain([Execute scripts/train.py]) --> LoadConfig[Load Cascaded YAML Config]
    LoadConfig --> BuildLoaders[Instantiate PyTorch SeedDataset & DataLoaders]
    BuildLoaders --> InitModel[ModelFactory: Instantiate Custom CNN or MobileNetV2]
    InitModel --> EpochLoop[Start Epoch Loop 1..N]

    EpochLoop --> Forward[Forward Pass: Compute Predictions]
    Forward --> Loss[Compute CrossEntropyLoss with Label Smoothing]
    Loss --> Backward[Backward Pass & Gradient Clipping max_norm=1.0]
    Backward --> OptimizerStep[Optimizer Step & Cosine Annealing Scheduler]
    OptimizerStep --> ValLoop[Run Validation Loop on val_loader]

    ValLoop --> CheckBest{val_acc > best_val_acc or val_loss lower?}
    CheckBest -- Yes --> SaveBest[Save best_model.pt and .meta.json]
    CheckBest -- No --> IncPatience[Increment epochs_without_improvement]

    IncPatience --> CheckStop{epochs_without_improvement >= patience?}
    CheckStop -- Yes --> EarlyStop[Trigger Early Stopping & Exit Loop]
    CheckStop -- No --> EpochLoop
    SaveBest --> EpochLoop

    EarlyStop --> SavePlots[Plot Dual-Axis Training Curves to artifacts/figures/]
    SavePlots --> DoneTrain([Training Complete])
```

---

## 3. Real-Time Inference & Grad-CAM Saliency Flow

```mermaid
sequenceDiagram
    autonumber
    actor User as User / Client
    participant UI as Streamlit / CLI Predictor
    participant Val as InferenceInputValidator
    participant Model as Deep Learning Model
    participant GC as GradCAM Engine
    participant Output as Response Handler

    User->>UI: Submit seed image (Upload / Demo)
    UI->>Val: validate_upload(bytes)
    Val-->>UI: Image decoded, dimensions verified [RGB]
    UI->>Model: Forward pass through normalized tensor [1, 3, 128, 128]
    Model-->>UI: Output Logits
    UI->>UI: Compute Softmax probabilities & Top-K ranking
    UI->>UI: Check uncertainty threshold (< 50% or tight margin)
    
    UI->>GC: Attach forward & backward hooks to final conv layer
    GC->>Model: Forward pass & backward pass with respect to target class
    GC-->>UI: Extract activation maps & pooled gradients
    GC->>UI: Compute ReLU weighted heatmap & normalize to [0, 1]
    GC->>GC: remove_hooks() & release GPU/CPU memory
    UI->>UI: Blend heatmap with original image (alpha = 0.45)
    UI->>Output: Assemble PredictionResult with visual triplet & notice
    Output-->>User: Display Quality Grade Badge, Score, Latency, and Grad-CAM
```

---

## 4. Multi-Model Evaluation & Cross-Benchmark Flow

```mermaid
flowchart LR
    A[Checkpoint: SeedNet-Baseline.pt] --> C[ModelEvaluator on test_manifest.csv]
    B[Checkpoint: MobileNetV2-Transfer.pt] --> C
    C --> D[Compute Confusion Matrix, Precision, Recall, Macro F1]
    C --> E[Compute Expected Calibration Error ECE]
    C --> F[Measure Latency ms and Throughput fps]
    D & E & F --> G[Save eval_ModelName.json]
    G --> H[seedvision.evaluation.compare.compare_models]
    H --> I[model_comparison_summary.csv & JSON]
    H --> J[model_comparison_benchmark.png Chart]
```
