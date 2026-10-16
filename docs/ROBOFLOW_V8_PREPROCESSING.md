# Roboflow v8 review-first preprocessing

The Roboflow v8 export is an object-detection dataset, separate from SeedVision's
four-class image-classification pipeline. Its nine labels have not been verified
against SeedVision's taxonomy. No defect label is automatically assigned or
approved by this workflow.

The original export must remain unchanged. Run the CLI in report-only mode first:

```powershell
python scripts/preprocess_roboflow.py --dataset "C:\Users\OM\Downloads\chickpea seeds.v8i.yolov8"
```

The report checks the declared class names, image and label pairing, image
readability, integer class IDs, normalized finite coordinates, YOLO bounding-box
geometry, polygon shape and area, and repeated source images across splits. A
source group is inferred by removing Roboflow's `.rf.<hash>` suffix and encoded
image extension. Cross-split groups are flagged; the original split assignment
is never silently presented as leakage-safe. A structurally `valid` report does
not verify class meaning: `label_mapping_status` remains
`unverified_manual_review_required` until a human review is completed.

After reviewing that report, create a separate staging directory for crops:

```powershell
python scripts/preprocess_roboflow.py `
  --dataset "C:\Users\OM\Downloads\chickpea seeds.v8i.yolov8" `
  --stage-crops `
  --output "D:\AI\Projects\SeedVision AI\data\roboflow_v8_review"
```

Staging writes one crop per valid annotation under `crops/`, a
`manual_review.csv`, and `validation_report.json`. The output must be empty and
outside the source export. The CSV retains the source image, inferred source
group, original split, Roboflow class, geometry, and annotation line. All rows
start with `review_status=pending`; `reviewed_label`, `reviewer`, and
`review_notes` are blank. Polygon crops preserve the polygon silhouette with
transparent pixels outside it. Invalid annotations are omitted from crops and
remain in the validation report. Reviewers must resolve ambiguous or unverified
classes manually; pending or rejected rows must not enter classifier training.

Build an interactive visual gallery before recording human decisions in the CSV:

```powershell
python scripts/build_roboflow_review_gallery.py `
  --manifest "D:\AI\Projects\SeedVision AI\data\roboflow_v8_review\manual_review.csv" `
  --output "D:\AI\Projects\SeedVision AI\data\roboflow_v8_review\review_gallery.html" `
  --limit 2000
```

Open the resulting HTML file in a browser. It features client-side pagination, batch
navigation (200 crops per batch), dynamic source-class and review-status filters,
and displays crop width, height, aspect ratio, and low-resolution flags (<28px).
The page is strictly read-only and records no unsolicited mutations.

Generate crop dimensions and perform auditable, reversible foreign-material triage:

```powershell
python scripts/triage_roboflow_crops.py `
  --manifest "D:\AI\Projects\SeedVision AI\data\roboflow_v8_review\manual_review.csv" `
  --generate-metadata `
  --action triage `
  --backup-path "D:\AI\Projects\SeedVision AI\data\roboflow_v8_review\manual_review.backup.csv"
```

This operation:
- Confirms non-pulse foreign matter (`Material extrano inorganico` and `Material extrano organico`) and marks it as `rejected` with reason `foreign_material`.
- Never automatically maps any Roboflow class to SeedVision's four target classification labels.
- Preserves a manifest backup before any modification.
- Is strictly idempotent: re-runs detect existing state and report 0 changed rows.
- Is fully reversible via `--action revert`.
- Flags crops under 28px as `low_resolution` in `crop_metadata.csv` without auto-rejecting pulse defects.

Capture the enriched review state, including resolution statistics and triage counts:

```powershell
python scripts/summarize_roboflow_review.py `
  --manifest "D:\AI\Projects\SeedVision AI\data\roboflow_v8_review\manual_review.csv" `
  --output "D:\AI\Projects\SeedVision AI\data\roboflow_v8_review\review_report.json"
```

### Recording and Applying Human Review Decisions

Human review decisions can be staged directly in `review_gallery.html` using the on-card decision forms and exported via **Export Decisions CSV**. Alternatively, human decisions can be validated and applied directly to `manual_review.csv` with automated pre-modification backups:

```powershell
# Apply a batch of human decisions from a reviewed CSV:
python scripts/apply_manual_review.py `
  --manifest "D:\AI\Projects\SeedVision AI\data\roboflow_v8_review\manual_review.csv" `
  --decisions "human_review_decisions.csv"

# Or record a single verified crop decision:
python scripts/apply_manual_review.py `
  --manifest "D:\AI\Projects\SeedVision AI\data\roboflow_v8_review\manual_review.csv" `
  --crop-path "crops/ff52bace81fd91d5.png" `
  --status "approved" `
  --label "sound_healthy" `
  --reviewer "om" `
  --notes "verified intact testa with uniform coloration"

# Or revert a decision back to pending:
python scripts/apply_manual_review.py `
  --manifest "D:\AI\Projects\SeedVision AI\data\roboflow_v8_review\manual_review.csv" `
  --crop-path "crops/ff52bace81fd91d5.png" `
  --revert
```

This tool does not alter models, training code, `data/raw/`, or `data/real_raw/`,
and it does not train a model. Optical crop labels do not establish germination,
embryo viability, or official seed-lot certification.


