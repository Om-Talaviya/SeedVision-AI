"""Build an interactive, client-side visual review gallery from a Roboflow crop manifest."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from urllib.parse import quote

# Import helpers for dimensions, triage, and annotations
from seedvision.data.roboflow_review import (
    extract_source_annotations,
    get_crop_dimensions,
    get_crop_triage_flags,
    sync_source_images,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create an interactive HTML gallery with client-side pagination, filters, and batch navigation."
    )
    parser.add_argument("--manifest", required=True, type=Path, help="Path to manual_review.csv")
    parser.add_argument("--output", required=True, type=Path, help="HTML output path")
    parser.add_argument("--metadata", type=Path, help="Optional crop_metadata.csv or crop_metadata.json")
    parser.add_argument("--source-dataset", type=Path, help="Optional path to original Roboflow YOLO dataset")
    parser.add_argument("--sync-source-images", action="store_true", default=True, help="Copy/sync source images to review directory")
    parser.add_argument("--no-sync-source-images", dest="sync_source_images", action="store_false", help="Do not copy source images")
    parser.add_argument("--offset", type=int, default=0, help="Initial row offset")
    parser.add_argument("--limit", type=int, default=2000, help="Maximum crops to include (0 for all crops)")
    parser.add_argument("--source-class", help="Optional exact initial Roboflow source class filter")
    parser.add_argument("--review-status", default="all", help="Initial review status filter (default: all)")
    parser.add_argument("--batch-size", type=int, default=200, help="Number of crops per batch for batch navigation")
    parser.add_argument("--page-size", type=int, default=50, help="Default client-side page size")
    args = parser.parse_args()

    if args.offset < 0 or args.limit < 0:
        parser.error("--offset must be non-negative and --limit must be non-negative (0 = unlimited)")

    manifest = args.manifest.expanduser().absolute()
    output = args.output.expanduser().absolute()
    if not manifest.is_file():
        parser.error(f"Manifest does not exist: {manifest}")
    if output == manifest:
        parser.error("Output must not overwrite the review manifest")

    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    required = {"crop_path", "source_class_name", "source_group_id", "review_status", "reviewed_label"}
    if not rows or not required.issubset(rows[0]):
        parser.error("Manifest does not have the required manual-review columns")

    # Load metadata if present
    metadata_map: dict[str, dict[str, Any]] = {}
    metadata_file = args.metadata
    if metadata_file is None:
        candidate = manifest.parent / "crop_metadata.csv"
        if candidate.is_file():
            metadata_file = candidate

    if metadata_file and metadata_file.is_file():
        try:
            if metadata_file.suffix.lower() == ".json":
                meta_list = json.loads(metadata_file.read_text(encoding="utf-8"))
            else:
                with metadata_file.open(newline="", encoding="utf-8") as m_f:
                    meta_list = list(csv.DictReader(m_f))
            for m in meta_list:
                metadata_map[m["crop_path"]] = m
        except Exception as exc:
            print(f"Warning: Failed reading metadata from {metadata_file}: {exc}")

    # Initial filtering based on CLI arguments
    filtered = [
        row for row in rows
        if (args.source_class is None or row["source_class_name"] == args.source_class)
        and (args.review_status == "all" or row["review_status"] == args.review_status)
    ]

    selected_rows = filtered[args.offset :] if args.limit == 0 else filtered[args.offset : args.offset + args.limit]

    # Discover or load source dataset / annotations
    source_dataset = args.source_dataset
    if source_dataset is None:
        val_report = manifest.parent / "validation_report.json"
        if val_report.is_file():
            try:
                v_data = json.loads(val_report.read_text(encoding="utf-8"))
                if v_data.get("dataset") and Path(v_data["dataset"]).is_dir():
                    source_dataset = Path(v_data["dataset"])
            except Exception:
                pass

    source_annotations_file = manifest.parent / "source_annotations.json"
    source_annotations: dict[str, list[dict[str, Any]]] = {}

    if source_dataset and source_dataset.is_dir():
        if args.sync_source_images:
            try:
                sync_source_images(source_dataset, manifest.parent)
            except Exception as e:
                print(f"Warning: could not sync source images: {e}")
        try:
            source_annotations = extract_source_annotations(source_dataset)
            source_annotations_file.write_text(json.dumps(source_annotations), encoding="utf-8")
        except Exception as e:
            print(f"Warning: could not extract source annotations: {e}")
    elif source_annotations_file.is_file():
        try:
            source_annotations = json.loads(source_annotations_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Build crop records with dimensions, linkage, and triage flags
    crop_records = []
    root_dir = manifest.parent

    for index, row in enumerate(selected_rows, start=args.offset + 1):
        rel_path = row["crop_path"]
        meta = metadata_map.get(rel_path, {})

        width = int(meta.get("width", 0)) if meta.get("width") else 0
        height = int(meta.get("height", 0)) if meta.get("height") else 0
        aspect_ratio = float(meta.get("aspect_ratio", 0.0)) if meta.get("aspect_ratio") else 0.0

        if width == 0 or height == 0:
            full_crop_path = root_dir / rel_path
            if full_crop_path.is_file():
                try:
                    width, height = get_crop_dimensions(full_crop_path)
                    aspect_ratio = round(width / height, 3) if height > 0 else 0.0
                except Exception:
                    width, height, aspect_ratio = 0, 0, 0.0

        is_low_res = width < 28 or height < 28
        crop_url = quote(rel_path.replace("\\", "/"))

        # Source image and annotation linkage
        source_image = row.get("source_image", "")
        annotation_line = int(row["annotation_line"]) if row.get("annotation_line", "").isdigit() else 1
        geometry = row.get("geometry", "bbox")
        source_group_id = row.get("source_group_id", "")

        ann_list = source_annotations.get(source_image, [])
        crop_ann = None
        for a in ann_list:
            if a.get("line") == annotation_line:
                crop_ann = a
                break

        bbox = crop_ann["bbox"] if crop_ann else []
        polygon = crop_ann.get("polygon", []) if crop_ann else []
        triage_flags = get_crop_triage_flags(width, height, aspect_ratio, bbox=tuple(bbox) if len(bbox) == 4 else None)
        is_upscaled = (width < 240 or height < 240) and width > 0 and height > 0

        crop_records.append({
            "index": index,
            "crop_path": rel_path,
            "crop_url": crop_url,
            "source_class_name": row["source_class_name"],
            "source_group_id": source_group_id,
            "source_image": source_image,
            "annotation_line": annotation_line,
            "geometry": geometry,
            "source_split": row.get("source_split", ""),
            "review_status": row["review_status"],
            "reviewed_label": row["reviewed_label"],
            "reviewer": row.get("reviewer", ""),
            "review_notes": row.get("review_notes", ""),
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio,
            "is_low_res": is_low_res,
            "is_upscaled": is_upscaled,
            "triage_flags": triage_flags,
            "bbox": bbox,
            "polygon": polygon,
        })

    # Available classes and statuses for filter controls
    all_classes = sorted({r["source_class_name"] for r in rows})
    all_statuses = sorted({r["review_status"] for r in rows})

    # Server-rendered initial page (for testing & zero-JS environments)
    initial_render_count = min(args.page_size, len(crop_records))
    initial_cards = []
    for crop in crop_records[:initial_render_count]:
        cw, ch = crop["width"], crop["height"]
        low_res_badge = '<span class="badge badge-warning badge-lowres" title="Flagged advisory: seed resolution is below 28px">Low Res (&lt;28px)</span>' if crop["is_low_res"] else ""
        upscaled_badge = f'<span class="badge badge-upscaled" title="Displayed at 240px container height from native {cw}x{ch}px. Visual preview only; no detail synthesized.">Upscaled ({cw}&times;{ch} &rarr; 240px)</span>' if crop["is_upscaled"] else '<span class="badge badge-native">Native Res</span>'
        ar_badge = '<span class="badge badge-warning" title="Extreme aspect ratio (<0.7 or >1.5): seed may be clipped or distorted">Clipped AR</span>' if "extreme_aspect_ratio" in crop["triage_flags"] else ""
        edge_badge = '<span class="badge badge-warning" title="Annotation touches image boundary: seed may be truncated">Border Crop</span>' if "edge_crop" in crop["triage_flags"] else ""
        status_class = f"badge-{crop['review_status']}"

        initial_cards.append(
            f'<article class="crop-card" data-index="{crop["index"]}" data-path="{html.escape(crop["crop_path"])}" '
            f'data-class="{html.escape(crop["source_class_name"])}" '
            f'data-status="{html.escape(crop["review_status"])}" data-low-res="{str(crop["is_low_res"]).lower()}">'
            f'<div class="img-wrap" data-img="{crop["crop_url"]}" data-info="#{crop["index"]} &bull; {html.escape(crop["source_class_name"])} &bull; {crop["width"]}&times;{crop["height"]} px"><img src="{crop["crop_url"]}" alt="crop {crop["index"]}" loading="lazy"></div>'
            f'<div class="card-body">'
            f'<div class="card-header-row"><span class="card-idx">#{crop["index"]}</span>'
            f'<span class="badge {status_class}">{html.escape(crop["review_status"])}</span>'
            f'{upscaled_badge}{low_res_badge}{ar_badge}{edge_badge}</div>'
            f'<p class="meta-row"><b>Source class:</b> <span class="tag-class">{html.escape(crop["source_class_name"])}</span></p>'
            f'<p class="meta-row"><b>Dimensions:</b> <span class="tag-dim">{crop["width"]} &times; {crop["height"]} px</span> (AR: {crop["aspect_ratio"]})</p>'
            f'<p class="meta-row"><b>Source group:</b> <code>{html.escape(crop["source_group_id"])}</code></p>'
            f'<p class="meta-row"><b>Source image:</b> <code>{html.escape(crop["source_image"])}</code> (Line {crop["annotation_line"]})</p>'
            f'<p class="meta-row"><b>Reviewed label:</b> {html.escape(crop["reviewed_label"] or "(blank)")}</p>'
            + (f'<p class="meta-row note-row"><b>Reason/Notes:</b> {html.escape(crop["review_notes"])}</p>' if crop["review_notes"] else "")
            + f'<p class="crop-path"><code>{html.escape(crop["crop_path"])}</code></p>'
            f'<button class="btn btn-source-view" type="button" data-crop-idx="{crop["index"]}" data-path="{html.escape(crop["crop_path"])}" data-img="{html.escape(crop["source_image"])}" data-line="{crop["annotation_line"]}" data-class="{html.escape(crop["source_class_name"])}" data-geom="{crop["geometry"]}" data-bbox=\'{json.dumps(crop["bbox"])}\' data-poly=\'{json.dumps(crop["polygon"])}\' data-group="{html.escape(crop["source_group_id"])}" data-dims="{crop["width"]}&times;{crop["height"]}">Open Original Image</button>'
            f'<div class="decision-box">'
            f'<div class="decision-box-title">Human Verification Decision</div>'
            f'<div class="decision-field"><label>Status:</label>'
            f'<select class="input-status" data-path="{html.escape(crop["crop_path"])}">'
            f'<option value="">-- Decision --</option><option value="approved">Approve</option><option value="rejected">Reject</option><option value="pending">Keep Pending</option></select></div>'
            f'<div class="decision-field label-field" style="display:none;"><label>Target Label:</label>'
            f'<select class="input-label" data-path="{html.escape(crop["crop_path"])}">'
            f'<option value="">-- Select Target Label --</option>'
            f'<option value="sound_healthy">sound_healthy</option>'
            f'<option value="discolored_spotted">discolored_spotted</option>'
            f'<option value="broken_cracked">broken_cracked</option>'
            f'<option value="shrivelled_damaged">shrivelled_damaged</option></select></div>'
            f'<div class="decision-field"><label>Reviewer:</label><input type="text" class="input-reviewer" placeholder="Reviewer initials" data-path="{html.escape(crop["crop_path"])}"></div>'
            f'<div class="decision-field"><label>Diagnostic Notes:</label><input type="text" class="input-notes" placeholder="Required diagnostic justification" data-path="{html.escape(crop["crop_path"])}"></div>'
            f'</div>'
            f'</div></article>'
        )

    title = f"Roboflow Crop Review Gallery ({len(selected_rows)} of {len(rows)} crops)"
    json_data = json.dumps(crop_records)
    source_annotations_json = json.dumps(source_annotations)
    classes_json = json.dumps(all_classes)
    statuses_json = json.dumps(all_statuses)

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
  :root {{
    --bg-main: #0b0f19;
    --bg-card: #151d2e;
    --bg-card-hover: #1b263b;
    --bg-toolbar: #111827;
    --border-color: #223046;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --accent: #38bdf8;
    --accent-hover: #0ea5e9;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    background-color: var(--bg-main);
    color: var(--text-primary);
    line-height: 1.5;
    padding: 20px;
  }}
  main {{
    max-width: 1720px;
    margin: 0 auto;
  }}
  header {{
    margin-bottom: 24px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--border-color);
  }}
  h1 {{
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 12px;
  }}
  .disclaimer-banner {{
    background: rgba(245, 158, 11, 0.12);
    border-left: 4px solid var(--warning);
    padding: 12px 16px;
    border-radius: 6px;
    font-size: 0.9rem;
    color: #fde68a;
    margin-top: 12px;
  }}
  .controls-panel {{
    background: var(--bg-toolbar);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 24px;
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    align-items: center;
    justify-content: space-between;
  }}
  .filter-group {{
    display: flex;
    flex-wrap: wrap;
    gap: 14px;
    align-items: center;
  }}
  .filter-item {{
    display: flex;
    flex-direction: column;
    gap: 4px;
  }}
  .filter-item label {{
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-secondary);
    font-weight: 600;
  }}
  select, input[type="text"] {{
    background: #1e293b;
    border: 1px solid var(--border-color);
    color: var(--text-primary);
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 0.85rem;
    outline: none;
    min-width: 150px;
  }}
  select:focus, input[type="text"]:focus {{
    border-color: var(--accent);
  }}
  .btn {{
    background: #1e293b;
    border: 1px solid var(--border-color);
    color: var(--text-primary);
    padding: 8px 14px;
    border-radius: 6px;
    font-size: 0.85rem;
    cursor: pointer;
    font-weight: 500;
    transition: all 0.15s ease;
  }}
  .btn:hover:not(:disabled) {{
    background: #2b3952;
    border-color: var(--accent);
  }}
  .btn:disabled {{
    opacity: 0.4;
    cursor: not-allowed;
  }}
  .btn-accent {{
    background: var(--accent);
    color: #0b0f19;
    border-color: var(--accent);
    font-weight: 600;
  }}
  .btn-accent:hover:not(:disabled) {{
    background: var(--accent-hover);
    color: #0b0f19;
  }}
  .pagination-nav {{
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }}
  .batch-nav {{
    display: flex;
    align-items: center;
    gap: 8px;
    padding-left: 12px;
    border-left: 1px solid var(--border-color);
  }}
  .counter-display {{
    font-size: 0.88rem;
    color: var(--text-secondary);
  }}
  .counter-display strong {{
    color: var(--accent);
  }}
  #card-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
    gap: 16px;
  }}
  .crop-card {{
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    overflow: hidden;
    transition: transform 0.15s ease, border-color 0.15s ease;
    display: flex;
    flex-direction: column;
  }}
  .crop-card:hover {{
    transform: translateY(-2px);
    border-color: #3b5072;
    background: var(--bg-card-hover);
  }}
  .img-wrap {{
    background: #0f1523;
    background-image: 
      linear-gradient(45deg, #182234 25%, transparent 25%), 
      linear-gradient(-45deg, #182234 25%, transparent 25%), 
      linear-gradient(45deg, transparent 75%, #182234 75%), 
      linear-gradient(-45deg, transparent 75%, #182234 75%);
    background-size: 16px 16px;
    background-position: 0 0, 0 8px, 8px -8px, -8px 0px;
    height: 240px;
    width: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    border-bottom: 1px solid var(--border-color);
    padding: 12px;
    position: relative;
    overflow: hidden;
    cursor: zoom-in;
  }}
  .img-wrap img {{
    width: 100%;
    height: 100%;
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
    image-rendering: pixelated;
    display: block;
    transition: transform 0.15s ease;
  }}
  .img-wrap:hover img {{
    transform: scale(1.04);
  }}
  .lightbox-modal {{
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    z-index: 9999;
    display: flex;
    align-items: center;
    justify-content: center;
  }}
  .lightbox-overlay {{
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.85);
    backdrop-filter: blur(4px);
  }}
  .lightbox-content {{
    position: relative;
    z-index: 10000;
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 24px;
    max-width: 90vw;
    max-height: 90vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 14px;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.6);
  }}
  .lightbox-close {{
    position: absolute;
    top: 10px;
    right: 14px;
    background: transparent;
    border: none;
    color: var(--text-secondary);
    font-size: 1.8rem;
    cursor: pointer;
    line-height: 1;
  }}
  .lightbox-close:hover {{
    color: var(--text-primary);
  }}
  .lightbox-img-wrap {{
    background: #0f1523;
    background-image: 
      linear-gradient(45deg, #182234 25%, transparent 25%), 
      linear-gradient(-45deg, #182234 25%, transparent 25%), 
      linear-gradient(45deg, transparent 75%, #182234 75%), 
      linear-gradient(-45deg, transparent 75%, #182234 75%);
    background-size: 16px 16px;
    background-position: 0 0, 0 8px, 8px -8px, -8px 0px;
    padding: 16px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    max-width: 80vw;
    max-height: 65vh;
  }}
  .lightbox-img-wrap img {{
    max-width: 100%;
    max-height: 60vh;
    object-fit: contain;
    image-rendering: pixelated;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
  }}
  .lightbox-info {{
    font-size: 0.9rem;
    color: var(--text-secondary);
    text-align: center;
  }}
  .card-body {{
    padding: 12px;
    display: flex;
    flex-direction: column;
    gap: 6px;
    flex-grow: 1;
  }}
  .card-header-row {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 4px;
  }}
  .card-idx {{
    font-size: 0.8rem;
    font-weight: 700;
    color: var(--text-secondary);
  }}
  .badge {{
    font-size: 0.72rem;
    font-weight: 600;
    padding: 2px 7px;
    border-radius: 4px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }}
  .badge-pending {{ background: rgba(148, 163, 184, 0.2); color: #cbd5e1; }}
  .badge-rejected {{ background: rgba(239, 68, 68, 0.2); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.4); }}
  .badge-approved {{ background: rgba(16, 185, 129, 0.2); color: #86efac; border: 1px solid rgba(16, 185, 129, 0.4); }}
  .badge-warning {{ background: rgba(245, 158, 11, 0.2); color: #fcd34d; border: 1px solid rgba(245, 158, 11, 0.4); }}
  .badge-upscaled {{ background: rgba(168, 85, 247, 0.2); color: #d8b4fe; border: 1px solid rgba(168, 85, 247, 0.4); }}
  .badge-native {{ background: rgba(16, 185, 129, 0.2); color: #86efac; border: 1px solid rgba(16, 185, 129, 0.4); }}
  .badge-info {{ background: rgba(56, 189, 248, 0.2); color: #7dd3fc; border: 1px solid rgba(56, 189, 248, 0.4); }}
  .btn-source-view {{
    width: 100%;
    margin-top: 8px;
    background: #1e293b;
    border: 1px solid var(--accent);
    color: var(--accent);
    padding: 6px 10px;
    font-size: 0.8rem;
    font-weight: 600;
    border-radius: 6px;
    cursor: pointer;
    transition: all 0.15s ease;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
  }}
  .btn-source-view:hover {{
    background: #0284c7;
    color: #f8fafc;
    border-color: #0284c7;
  }}
  /* Source Image Inspection Modal */
  .source-modal {{
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    z-index: 10000;
    display: flex;
    align-items: center;
    justify-content: center;
  }}
  .source-overlay {{
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.85);
    backdrop-filter: blur(4px);
  }}
  .source-content {{
    position: relative;
    z-index: 10001;
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    width: 94vw;
    height: 92vh;
    max-width: 1280px;
    max-height: 950px;
    display: flex;
    flex-direction: column;
    box-shadow: 0 25px 60px rgba(0, 0, 0, 0.7);
    overflow: hidden;
  }}
  .source-header {{
    padding: 14px 20px;
    background: var(--bg-toolbar);
    border-bottom: 1px solid var(--border-color);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    flex-wrap: wrap;
  }}
  .source-title-group h2 {{
    font-size: 1.1rem;
    color: var(--text-primary);
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 10px;
  }}
  .source-badges {{
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    align-items: center;
  }}
  .source-header-actions {{
    display: flex;
    align-items: center;
    gap: 12px;
  }}
  .zoom-controls {{
    display: flex;
    align-items: center;
    gap: 4px;
    background: #0f172a;
    padding: 4px;
    border-radius: 6px;
    border: 1px solid var(--border-color);
  }}
  .zoom-controls .btn {{
    padding: 4px 10px;
    font-size: 0.8rem;
  }}
  .source-close {{
    background: transparent;
    border: none;
    color: var(--text-secondary);
    font-size: 1.8rem;
    cursor: pointer;
    line-height: 1;
    padding: 0 4px;
  }}
  .source-close:hover {{ color: var(--text-primary); }}
  .source-viewport {{
    flex: 1;
    overflow: hidden;
    position: relative;
    background: #090d16;
    background-image: 
      linear-gradient(45deg, #101726 25%, transparent 25%), 
      linear-gradient(-45deg, #101726 25%, transparent 25%), 
      linear-gradient(45deg, transparent 75%, #101726 75%), 
      linear-gradient(-45deg, transparent 75%, #101726 75%);
    background-size: 20px 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: grab;
    user-select: none;
  }}
  .source-viewport:active {{ cursor: grabbing; }}
  .source-canvas-wrap {{
    position: relative;
    display: inline-block;
    transform-origin: center center;
    transition: transform 0.05s linear;
  }}
  #source-svg {{
    display: block;
    width: 640px;
    height: 640px;
    max-width: none;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    border: 1px solid var(--border-color);
    background: #000;
  }}
  .nearby-seed-box {{
    fill: rgba(245, 158, 11, 0.05);
    stroke: rgba(245, 158, 11, 0.45);
    stroke-width: 1.5;
    stroke-dasharray: 4 2;
    transition: fill 0.1s ease, stroke 0.1s ease;
  }}
  .nearby-seed-box:hover {{
    fill: rgba(245, 158, 11, 0.25);
    stroke: rgba(245, 158, 11, 0.8);
    stroke-width: 2;
  }}
  .nearby-seed-poly {{
    fill: rgba(245, 158, 11, 0.05);
    stroke: rgba(245, 158, 11, 0.45);
    stroke-width: 1.5;
    stroke-dasharray: 4 2;
  }}
  .selected-seed-box {{
    fill: rgba(56, 189, 248, 0.22);
    stroke: #38bdf8;
    stroke-width: 3.5;
  }}
  .selected-seed-poly {{
    fill: rgba(56, 189, 248, 0.22);
    stroke: #38bdf8;
    stroke-width: 3.5;
  }}
  .source-footer {{
    padding: 10px 20px;
    background: var(--bg-toolbar);
    border-top: 1px solid var(--border-color);
    font-size: 0.82rem;
    color: var(--text-secondary);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
  }}
  .source-hint {{
    font-size: 0.78rem;
    color: #64748b;
  }}
  .source-error {{
    background: rgba(239, 68, 68, 0.15);
    border: 1px solid var(--danger);
    color: #fca5a5;
    padding: 24px;
    border-radius: 8px;
    max-width: 550px;
    text-align: center;
    line-height: 1.6;
  }}
  .meta-row {{
    font-size: 0.8rem;
    color: var(--text-secondary);
    display: flex;
    justify-content: space-between;
  }}
  .meta-row b {{ color: #cbd5e1; }}
  .tag-class {{
    color: #e2e8f0;
    font-weight: 600;
  }}
  .tag-dim {{
    color: var(--accent);
    font-family: monospace;
    font-size: 0.78rem;
  }}
  .note-row {{
    color: #f87171;
    font-style: italic;
  }}
  .crop-path {{
    font-size: 0.72rem;
    color: #64748b;
    word-break: break-all;
    margin-top: auto;
    padding-top: 6px;
    border-top: 1px dashed rgba(255, 255, 255, 0.08);
  }}
  .decision-box {{
    margin-top: 10px;
    padding: 10px;
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 6px;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }}
  .decision-box-title {{
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--accent);
  }}
  .decision-field {{
    display: flex;
    flex-direction: column;
    gap: 2px;
  }}
  .decision-field label {{
    font-size: 0.7rem;
    color: var(--text-secondary);
  }}
  .decision-field select, .decision-field input {{
    font-size: 0.75rem;
    padding: 4px 8px;
    min-width: auto;
    width: 100%;
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 4px;
    color: #f8fafc;
  }}
  .decision-field select:focus, .decision-field input:focus {{
    border-color: var(--accent);
  }}
  .review-action-group {{
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
    background: rgba(56, 189, 248, 0.06);
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 6px;
    padding: 8px 12px;
  }}
  .empty-state {{
    grid-column: 1 / -1;
    text-align: center;
    padding: 60px 20px;
    background: var(--bg-card);
    border-radius: 8px;
    border: 1px dashed var(--border-color);
  }}
  .empty-state h3 {{
    font-size: 1.2rem;
    margin-bottom: 8px;
    color: var(--text-primary);
  }}
  .empty-state p {{ color: var(--text-secondary); font-size: 0.9rem; }}
</style>
</head>
<body>
<main>
  <header>
    <h1>SeedVision AI &bull; Roboflow Crop Review Gallery</h1>
    <div class="counter-display">
      Loaded <strong id="total-crops-count">{len(crop_records)}</strong> crops from manifest <code>{html.escape(manifest.name)}</code>
    </div>
    <div class="disclaimer-banner">
      <strong>Auditable Review Protocol:</strong> Programmatic screening candidate flags are strictly advisory.
      No Roboflow class is automatically mapped to SeedVision classification labels.
      Every approved row requires human confirmation with diagnostic notes before entering <code>manual_review.csv</code>.
    </div>
  </header>

  <section class="controls-panel">
    <div class="filter-group">
      <div class="filter-item">
        <label for="source-class-filter">Source Class</label>
        <select id="source-class-filter">
          <option value="all">All Classes ({len(rows)})</option>
          {''.join(f'<option value="{html.escape(c)}">{html.escape(c)}</option>' for c in all_classes)}
        </select>
      </div>

      <div class="filter-item">
        <label for="review-status-filter">Review Status</label>
        <select id="review-status-filter">
          <option value="all">All Statuses</option>
          {''.join(f'<option value="{html.escape(s)}">{html.escape(s)}</option>' for s in all_statuses)}
        </select>
      </div>

      <div class="filter-item">
        <label for="resolution-filter">Resolution</label>
        <select id="resolution-filter">
          <option value="all">All Resolutions</option>
          <option value="low_res">Flagged Low-Res (&lt;28px)</option>
          <option value="standard">Standard Resolution (&ge;28px)</option>
        </select>
      </div>

      <div class="filter-item">
        <label for="quality-filter">Quality & Triage</label>
        <select id="quality-filter">
          <option value="all">All Crops</option>
          <option value="standard">Standard Resolution (&ge;28px)</option>
          <option value="small_crop">Flagged: Small Crop (&lt;28px)</option>
          <option value="upscaled">Upscaled Preview (&lt;240px)</option>
          <option value="native">Native Resolution (&ge;240px)</option>
          <option value="extreme_ar">Flagged: Clipped / Extreme AR</option>
          <option value="edge_crop">Flagged: Border / Edge Crop</option>
        </select>
      </div>

      <div class="filter-item">
        <label for="search-input">Search Group / Path</label>
        <input type="text" id="search-input" placeholder="e.g. img_4151...">
      </div>
    </div>

    <div class="filter-group">
      <div class="filter-item">
        <label for="page-size-select">Page Size</label>
        <select id="page-size-select">
          <option value="25">25 per page</option>
          <option value="50" selected>50 per page</option>
          <option value="100">100 per page</option>
          <option value="200">200 per page</option>
          <option value="500">500 per page</option>
        </select>
      </div>

      <div class="filter-item">
        <label>Pagination</label>
        <div class="pagination-nav">
          <button class="btn" id="btn-first" title="First Page">&laquo;</button>
          <button class="btn" id="btn-prev" title="Previous Page">&lsaquo;</button>
          <span class="counter-display" id="page-indicator">Page 1 of 1</span>
          <button class="btn" id="btn-next" title="Next Page">&rsaquo;</button>
          <button class="btn" id="btn-last" title="Last Page">&raquo;</button>
        </div>
      </div>

      <div class="filter-item batch-nav">
        <label>Batch Navigation (200/batch)</label>
        <div class="pagination-nav">
          <button class="btn" id="btn-batch-prev" title="Previous Batch">&laquo; Batch</button>
          <span class="counter-display" id="batch-indicator">Batch 1</span>
          <button class="btn" id="btn-batch-next" title="Next Batch">Batch &raquo;</button>
        </div>
      </div>

      <div class="review-action-group">
        <input type="text" id="global-reviewer" placeholder="Reviewer initials" style="width: 120px;" title="Auto-fills reviewer field">
        <button class="btn btn-accent" id="btn-export-decisions" title="Export staged human decisions to CSV">Export Decisions CSV</button>
        <span id="staged-count-badge" class="badge badge-approved" style="display:none;">0 Staged</span>
      </div>
    </div>
  </section>

  <div style="margin-bottom: 14px; font-size: 0.9rem; color: var(--text-secondary);" id="active-summary">
    Showing crops <strong>1 - {initial_render_count}</strong> of <strong>{len(crop_records)}</strong>
  </div>

  <section id="card-grid">
    {''.join(initial_cards)}
  </section>
</main>

<div id="lightbox-modal" class="lightbox-modal" style="display:none;" role="dialog" aria-modal="true">
  <div class="lightbox-overlay" id="lightbox-overlay"></div>
  <div class="lightbox-content">
    <button class="lightbox-close" id="lightbox-close" aria-label="Close">&times;</button>
    <div class="lightbox-img-wrap"><img id="lightbox-img" src="" alt="Zoomed crop"></div>
    <div class="lightbox-info" id="lightbox-info"></div>
  </div>
</div>

<div id="source-image-modal" class="source-modal" style="display:none;" role="dialog" aria-modal="true">
  <div class="source-overlay" id="source-overlay"></div>
  <div class="source-content">
    <div class="source-header">
      <div class="source-title-group">
        <h2 id="source-modal-title">Original Source Image Inspection</h2>
        <div class="source-badges" id="source-badges">
          <span class="badge badge-info" id="source-badge-res">Native: 640 &times; 640 px</span>
          <span class="badge badge-info" id="source-badge-group">Group: -</span>
          <span class="badge badge-approved" id="source-badge-target">Selected: -</span>
        </div>
      </div>
      <div class="source-header-actions">
        <div class="zoom-controls">
          <button class="btn" id="btn-zoom-in" title="Zoom In">+</button>
          <button class="btn" id="btn-zoom-out" title="Zoom Out">&minus;</button>
          <button class="btn" id="btn-zoom-reset" title="Reset (100%)">100%</button>
          <button class="btn btn-accent" id="btn-zoom-fit" title="Center & Magnify Selected Seed">Focus Seed</button>
        </div>
        <button class="source-close" id="source-close" aria-label="Close">&times;</button>
      </div>
    </div>
    <div class="source-viewport" id="source-viewport">
      <div class="source-canvas-wrap" id="source-canvas-wrap">
        <svg id="source-svg" viewBox="0 0 640 640" preserveAspectRatio="xMidYMid meet"></svg>
        <div id="source-error-msg" class="source-error" style="display:none;"></div>
      </div>
    </div>
    <div class="source-footer" id="source-footer">
      <span id="source-seed-count">Loading annotations...</span>
      <span class="source-hint">Drag image to pan &bull; Scroll wheel to zoom &bull; Click seed to inspect &bull; Press Esc to close</span>
    </div>
  </div>
</div>

<script id="crop-data" type="application/json">{json_data}</script>
<script id="source-annotations" type="application/json">{source_annotations_json}</script>

<script>
  (function() {{
    const rawData = JSON.parse(document.getElementById('crop-data').textContent);
    let currentPage = 1;
    let pageSize = parseInt(document.getElementById('page-size-select').value, 10) || 50;
    const batchSize = {args.batch_size};
    const stagedDecisions = {{}};

    const sourceAnnotations = JSON.parse(document.getElementById('source-annotations').textContent || '{{}}');
    const classFilter = document.getElementById('source-class-filter');
    const statusFilter = document.getElementById('review-status-filter');
    const resolutionFilter = document.getElementById('resolution-filter');
    const qualityFilter = document.getElementById('quality-filter');
    const searchInput = document.getElementById('search-input');
    const pageSizeSelect = document.getElementById('page-size-select');
    const grid = document.getElementById('card-grid');
    const activeSummary = document.getElementById('active-summary');
    const pageIndicator = document.getElementById('page-indicator');
    const batchIndicator = document.getElementById('batch-indicator');
    const globalReviewer = document.getElementById('global-reviewer');
    const btnExportDecisions = document.getElementById('btn-export-decisions');
    const stagedCountBadge = document.getElementById('staged-count-badge');

    const lightboxModal = document.getElementById('lightbox-modal');
    const lightboxOverlay = document.getElementById('lightbox-overlay');
    const lightboxClose = document.getElementById('lightbox-close');
    const lightboxImg = document.getElementById('lightbox-img');
    const lightboxInfo = document.getElementById('lightbox-info');

    const sourceModal = document.getElementById('source-image-modal');
    const sourceOverlay = document.getElementById('source-overlay');
    const sourceClose = document.getElementById('source-close');
    const sourceModalTitle = document.getElementById('source-modal-title');
    const sourceBadgeRes = document.getElementById('source-badge-res');
    const sourceBadgeGroup = document.getElementById('source-badge-group');
    const sourceBadgeTarget = document.getElementById('source-badge-target');
    const sourceViewport = document.getElementById('source-viewport');
    const sourceCanvasWrap = document.getElementById('source-canvas-wrap');
    const sourceSvg = document.getElementById('source-svg');
    const sourceErrorMsg = document.getElementById('source-error-msg');
    const sourceSeedCount = document.getElementById('source-seed-count');
    const btnZoomIn = document.getElementById('btn-zoom-in');
    const btnZoomOut = document.getElementById('btn-zoom-out');
    const btnZoomReset = document.getElementById('btn-zoom-reset');
    const btnZoomFit = document.getElementById('btn-zoom-fit');

    let zoomScale = 1.0;
    let panX = 0;
    let panY = 0;
    let isPanning = false;
    let startX = 0;
    let startY = 0;
    let currentSelectedBbox = null;

    function openLightbox(imgSrc, infoHtml) {{
      lightboxImg.src = imgSrc;
      lightboxInfo.innerHTML = infoHtml;
      lightboxModal.style.display = 'flex';
    }}

    function closeLightbox() {{
      lightboxModal.style.display = 'none';
      lightboxImg.src = '';
    }}

    if (lightboxClose) lightboxClose.addEventListener('click', closeLightbox);
    if (lightboxOverlay) lightboxOverlay.addEventListener('click', closeLightbox);

    function openSourceModal(cropIdx, cropPath, sourceImg, lineNum, className, geom, bbox, poly, groupId, dims) {{
      sourceModal.style.display = 'flex';
      sourceModalTitle.innerHTML = `Original Source Image &bull; <code>${{escapeHtml(sourceImg)}}</code>`;
      sourceBadgeRes.textContent = `Native: 640 \u00d7 640 px`;
      sourceBadgeGroup.textContent = `Group: ${{groupId}}`;
      sourceBadgeTarget.textContent = `Selected: #${{cropIdx}} ${{className}} (Line ${{lineNum}}, ${{geom}})`;

      currentSelectedBbox = bbox && bbox.length === 4 ? bbox : null;
      zoomScale = 1.0;
      panX = 0;
      panY = 0;
      updateTransform();

      const testImg = new Image();
      testImg.onload = function() {{
        sourceSvg.style.display = 'block';
        sourceErrorMsg.style.display = 'none';
        renderSourceSvg(sourceImg, lineNum, cropIdx, className, geom, bbox, poly);
      }};
      testImg.onerror = function() {{
        sourceSvg.style.display = 'none';
        sourceErrorMsg.style.display = 'block';
        sourceErrorMsg.innerHTML = `
          <h3>Source Image Not Found</h3>
          <p style="margin: 8px 0;">Could not load relative image: <code>source_images/${{escapeHtml(sourceImg)}}</code></p>
          <p style="font-size: 0.85rem; color: #cbd5e1;">Original export path may not be synced. Re-run gallery generation with:</p>
          <pre style="margin-top: 8px; padding: 8px; background: #0b0f19; border-radius: 4px; font-size: 0.8rem; overflow-x: auto;">python scripts/build_roboflow_review_gallery.py --source-dataset "&lt;path_to_yolov8&gt;" --sync-source-images</pre>
        `;
        sourceSeedCount.textContent = 'Source image unavailable';
      }};
      testImg.src = `source_images/${{sourceImg}}`;
    }}

    function renderSourceSvg(sourceImg, lineNum, cropIdx, className, geom, bbox, poly) {{
      const anns = sourceAnnotations[sourceImg] || [];
      sourceSeedCount.textContent = `${{anns.length}} seeds annotated in this image (Selected: Line #${{lineNum}})`;

      let svgContent = `
        <defs>
          <filter id="seedGlow" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>
        <image href="source_images/${{sourceImg}}" x="0" y="0" width="640" height="640" preserveAspectRatio="none" />
      `;

      // Draw nearby seeds in subtle dashed outline
      anns.forEach(a => {{
        if (a.line === lineNum) return;
        if (a.geometry === 'polygon' && a.polygon && a.polygon.length >= 3) {{
          const pts = a.polygon.map(pt => `${{pt[0] * 640}},${{pt[1] * 640}}`).join(' ');
          svgContent += `<polygon points="${{pts}}" class="nearby-seed-poly"><title>Line #${{a.line}}: ${{escapeHtml(a.class_name)}}</title></polygon>`;
        }} else if (a.bbox && a.bbox.length === 4) {{
          const bx = a.bbox[0] * 640;
          const by = a.bbox[1] * 640;
          const bw = (a.bbox[2] - a.bbox[0]) * 640;
          const bh = (a.bbox[3] - a.bbox[1]) * 640;
          svgContent += `<rect x="${{bx}}" y="${{by}}" width="${{bw}}" height="${{bh}}" class="nearby-seed-box"><title>Line #${{a.line}}: ${{escapeHtml(a.class_name)}}</title></rect>`;
        }}
      }});

      // Draw selected seed in glowing highlight
      if (geom === 'polygon' && poly && poly.length >= 3) {{
        const pts = poly.map(pt => `${{pt[0] * 640}},${{pt[1] * 640}}`).join(' ');
        svgContent += `<polygon points="${{pts}}" class="selected-seed-poly" filter="url(#seedGlow)" />`;
      }}
      if (bbox && bbox.length === 4) {{
        const sx = bbox[0] * 640;
        const sy = bbox[1] * 640;
        const sw = (bbox[2] - bbox[0]) * 640;
        const sh = (bbox[3] - bbox[1]) * 640;
        svgContent += `<rect x="${{sx}}" y="${{sy}}" width="${{sw}}" height="${{sh}}" class="selected-seed-box" filter="url(#seedGlow)" />`;
        const tagX = Math.max(10, Math.min(470, sx));
        const tagY = Math.max(20, sy - 8);
        svgContent += `
          <g transform="translate(${{tagX}}, ${{tagY}})">
            <rect rx="4" width="165" height="20" fill="#0284c7" />
            <text x="8" y="14" fill="#ffffff" font-size="11" font-weight="bold">#${{cropIdx}} ${{escapeHtml(className)}} (Selected)</text>
          </g>
        `;
      }}

      sourceSvg.innerHTML = svgContent;
    }}

    function closeSourceModal() {{
      sourceModal.style.display = 'none';
      sourceSvg.innerHTML = '';
    }}

    if (sourceClose) sourceClose.addEventListener('click', closeSourceModal);
    if (sourceOverlay) sourceOverlay.addEventListener('click', closeSourceModal);

    function updateTransform() {{
      sourceCanvasWrap.style.transform = `translate(${{panX}}px, ${{panY}}px) scale(${{zoomScale}})`;
    }}

    btnZoomIn.addEventListener('click', () => {{
      zoomScale = Math.min(5.0, zoomScale * 1.25);
      updateTransform();
    }});

    btnZoomOut.addEventListener('click', () => {{
      zoomScale = Math.max(0.5, zoomScale / 1.25);
      updateTransform();
    }});

    btnZoomReset.addEventListener('click', () => {{
      zoomScale = 1.0;
      panX = 0;
      panY = 0;
      updateTransform();
    }});

    btnZoomFit.addEventListener('click', () => {{
      if (currentSelectedBbox) {{
        const cx = ((currentSelectedBbox[0] + currentSelectedBbox[2]) / 2) * 640;
        const cy = ((currentSelectedBbox[1] + currentSelectedBbox[3]) / 2) * 640;
        zoomScale = 3.0;
        panX = (320 - cx) * zoomScale;
        panY = (320 - cy) * zoomScale;
        updateTransform();
      }} else {{
        zoomScale = 1.0;
        panX = 0;
        panY = 0;
        updateTransform();
      }}
    }});

    sourceViewport.addEventListener('mousedown', (e) => {{
      if (e.target.closest('.zoom-controls') || e.target.closest('.source-close')) return;
      isPanning = true;
      startX = e.clientX - panX;
      startY = e.clientY - panY;
    }});

    window.addEventListener('mousemove', (e) => {{
      if (!isPanning) return;
      panX = e.clientX - startX;
      panY = e.clientY - startY;
      updateTransform();
    }});

    window.addEventListener('mouseup', () => {{
      isPanning = false;
    }});

    sourceViewport.addEventListener('wheel', (e) => {{
      e.preventDefault();
      if (e.deltaY < 0) {{
        zoomScale = Math.min(5.0, zoomScale * 1.15);
      }} else {{
        zoomScale = Math.max(0.5, zoomScale / 1.15);
      }}
      updateTransform();
    }}, {{ passive: false }});

    grid.addEventListener('click', (e) => {{
      const sourceBtn = e.target.closest('.btn-source-view');
      if (sourceBtn) {{
        e.stopPropagation();
        const cropIdx = sourceBtn.getAttribute('data-crop-idx');
        const cropPath = sourceBtn.getAttribute('data-path');
        const sourceImg = sourceBtn.getAttribute('data-img');
        const lineNum = parseInt(sourceBtn.getAttribute('data-line'), 10) || 1;
        const className = sourceBtn.getAttribute('data-class');
        const geom = sourceBtn.getAttribute('data-geom');
        let bbox = [];
        let poly = [];
        try {{ bbox = JSON.parse(sourceBtn.getAttribute('data-bbox') || '[]'); }} catch (err) {{}}
        try {{ poly = JSON.parse(sourceBtn.getAttribute('data-poly') || '[]'); }} catch (err) {{}}
        const groupId = sourceBtn.getAttribute('data-group');
        const dims = sourceBtn.getAttribute('data-dims');
        openSourceModal(cropIdx, cropPath, sourceImg, lineNum, className, geom, bbox, poly, groupId, dims);
        return;
      }}
      const wrap = e.target.closest('.img-wrap');
      if (wrap) {{
        const imgSrc = wrap.getAttribute('data-img') || (wrap.querySelector('img') ? wrap.querySelector('img').src : '');
        const info = wrap.getAttribute('data-info') || '';
        if (imgSrc) {{
          openLightbox(imgSrc, info);
        }}
      }}
    }});

    const btnFirst = document.getElementById('btn-first');
    const btnPrev = document.getElementById('btn-prev');
    const btnNext = document.getElementById('btn-next');
    const btnLast = document.getElementById('btn-last');
    const btnBatchPrev = document.getElementById('btn-batch-prev');
    const btnBatchNext = document.getElementById('btn-batch-next');

    const initialClass = "{args.source_class or ''}";
    const initialStatus = "{args.review_status}";
    if (initialClass && classFilter) classFilter.value = initialClass;
    if (initialStatus && statusFilter && initialStatus !== "all") statusFilter.value = initialStatus;

    function getFilteredData() {{
      const c = classFilter.value;
      const s = statusFilter.value;
      const r = resolutionFilter.value;
      const qf = qualityFilter ? qualityFilter.value : 'all';
      const q = (searchInput.value || '').trim().toLowerCase();

      return rawData.filter(item => {{
        if (c !== 'all' && item.source_class_name !== c) return false;
        if (s !== 'all' && item.review_status !== s) return false;
        if (r === 'low_res' && !item.is_low_res) return false;
        if (r === 'standard' && item.is_low_res) return false;
        if (qf === 'small_crop' && !item.is_low_res) return false;
        if (qf === 'standard' && item.is_low_res) return false;
        if (qf === 'upscaled' && !item.is_upscaled) return false;
        if (qf === 'native' && item.is_upscaled) return false;
        if (qf === 'extreme_ar' && !(item.triage_flags || []).includes('extreme_aspect_ratio')) return false;
        if (qf === 'edge_crop' && !(item.triage_flags || []).includes('edge_crop')) return false;
        if (q) {{
          const target = (item.source_group_id + ' ' + item.crop_path + ' ' + (item.source_image || '') + ' ' + (item.review_notes || '')).toLowerCase();
          if (!target.includes(q)) return false;
        }}
        return true;
      }});
    }}

    function render() {{
      const filtered = getFilteredData();
      const totalItems = filtered.length;
      const totalPages = Math.max(1, Math.ceil(totalItems / pageSize));

      if (currentPage > totalPages) currentPage = totalPages;
      if (currentPage < 1) currentPage = 1;

      const startIndex = (currentPage - 1) * pageSize;
      const endIndex = Math.min(startIndex + pageSize, totalItems);
      const pageItems = filtered.slice(startIndex, endIndex);

      pageIndicator.textContent = `Page ${{currentPage}} of ${{totalPages}}`;
      const currentBatch = Math.floor(startIndex / batchSize) + 1;
      const totalBatches = Math.max(1, Math.ceil(totalItems / batchSize));
      batchIndicator.textContent = `Batch ${{currentBatch}} / ${{totalBatches}}`;

      if (totalItems === 0) {{
        activeSummary.innerHTML = 'Showing <strong>0</strong> crops (0 matches)';
      }} else {{
        activeSummary.innerHTML = `Showing crops <strong>${{startIndex + 1}} - ${{endIndex}}</strong> of <strong>${{totalItems}}</strong> matching filters`;
      }}

      btnFirst.disabled = currentPage === 1;
      btnPrev.disabled = currentPage === 1;
      btnNext.disabled = currentPage === totalPages || totalPages === 0;
      btnLast.disabled = currentPage === totalPages || totalPages === 0;
      btnBatchPrev.disabled = startIndex < batchSize;
      btnBatchNext.disabled = endIndex >= totalItems;

      if (pageItems.length === 0) {{
        grid.innerHTML = '<div class="empty-state"><h3>No Crops Match Active Filters</h3><p>Try resetting source class, review status, or quality filters.</p></div>';
        return;
      }}

      const htmlCards = pageItems.map(crop => {{
        const lowResBadge = crop.is_low_res ? '<span class="badge badge-warning badge-lowres" title="Flagged advisory: seed resolution is below 28px">Low Res (&lt;28px)</span>' : '';
        const upscaledBadge = crop.is_upscaled ? `<span class="badge badge-upscaled" title="Displayed at 240px container height from native ${{crop.width}}x${{crop.height}}px. Visual preview only; no detail synthesized.">Upscaled (${{crop.width}}&times;${{crop.height}} &rarr; 240px)</span>` : '<span class="badge badge-native">Native Res</span>';
        const arBadge = (crop.triage_flags || []).includes('extreme_aspect_ratio') ? '<span class="badge badge-warning" title="Extreme aspect ratio (<0.7 or >1.5): seed may be clipped">Clipped AR</span>' : '';
        const edgeBadge = (crop.triage_flags || []).includes('edge_crop') ? '<span class="badge badge-warning" title="Annotation touches image boundary: seed may be truncated">Border Crop</span>' : '';
        const statusClass = `badge-${{crop.review_status}}`;
        const noteRow = crop.review_notes ? `<p class="meta-row note-row"><b>Reason/Notes:</b> ${{escapeHtml(crop.review_notes)}}</p>` : '';
        const dec = stagedDecisions[crop.crop_path] || {{}};
        const curStatus = dec.review_status || '';
        const curLabel = dec.reviewed_label || '';
        const curReviewer = dec.reviewer || (globalReviewer ? globalReviewer.value : '') || '';
        const curNotes = dec.review_notes || '';
        const labelRowStyle = curStatus === 'approved' ? '' : 'style="display:none;"';

        return `
          <article class="crop-card" data-index="${{crop.index}}" data-path="${{escapeHtml(crop.crop_path)}}">
            <div class="img-wrap" data-img="${{crop.crop_url}}" data-info="#${{crop.index}} &bull; ${{escapeHtml(crop.source_class_name)}} &bull; ${{crop.width}}&times;${{crop.height}} px"><img src="${{crop.crop_url}}" alt="crop ${{crop.index}}" loading="lazy"></div>
            <div class="card-body">
              <div class="card-header-row">
                <span class="card-idx">#${{crop.index}}</span>
                <span class="badge ${{statusClass}}">${{escapeHtml(crop.review_status)}}</span>
                ${{upscaledBadge}}${{lowResBadge}}${{arBadge}}${{edgeBadge}}
              </div>
              <p class="meta-row"><b>Source class:</b> <span class="tag-class">${{escapeHtml(crop.source_class_name)}}</span></p>
              <p class="meta-row"><b>Dimensions:</b> <span class="tag-dim">${{crop.width}} &times; ${{crop.height}} px</span> (AR: ${{crop.aspect_ratio}})</p>
              <p class="meta-row"><b>Source group:</b> <code>${{escapeHtml(crop.source_group_id)}}</code></p>
              <p class="meta-row"><b>Source image:</b> <code>${{escapeHtml(crop.source_image)}}</code> (Line ${{crop.annotation_line}})</p>
              <p class="meta-row"><b>Reviewed label:</b> ${{escapeHtml(crop.reviewed_label || '(blank)')}}</p>
              ${{noteRow}}
              <p class="crop-path"><code>${{escapeHtml(crop.crop_path)}}</code></p>

              <button class="btn btn-source-view" type="button" data-crop-idx="${{crop.index}}" data-path="${{escapeHtml(crop.crop_path)}}" data-img="${{escapeHtml(crop.source_image)}}" data-line="${{crop.annotation_line}}" data-class="${{escapeHtml(crop.source_class_name)}}" data-geom="${{crop.geometry}}" data-bbox='${{JSON.stringify(crop.bbox || [])}}' data-poly='${{JSON.stringify(crop.polygon || [])}}' data-group="${{escapeHtml(crop.source_group_id)}}" data-dims="${{crop.width}}&times;${{crop.height}}">Open Original Image</button>

              <div class="decision-box">
                <div class="decision-box-title">Human Verification Decision</div>
                <div class="decision-field">
                  <label>Status:</label>
                  <select class="input-status" data-path="${{escapeHtml(crop.crop_path)}}">
                    <option value="" ${{curStatus===''?'selected':''}}>-- Decision --</option>
                    <option value="approved" ${{curStatus==='approved'?'selected':''}}>Approve</option>
                    <option value="rejected" ${{curStatus==='rejected'?'selected':''}}>Reject</option>
                    <option value="pending" ${{curStatus==='pending'?'selected':''}}>Keep Pending</option>
                  </select>
                </div>
                <div class="decision-field label-field" ${{labelRowStyle}}>
                  <label>Target Label:</label>
                  <select class="input-label" data-path="${{escapeHtml(crop.crop_path)}}">
                    <option value="" ${{curLabel===''?'selected':''}}>-- Select Target Label --</option>
                    <option value="sound_healthy" ${{curLabel==='sound_healthy'?'selected':''}}>sound_healthy</option>
                    <option value="discolored_spotted" ${{curLabel==='discolored_spotted'?'selected':''}}>discolored_spotted</option>
                    <option value="broken_cracked" ${{curLabel==='broken_cracked'?'selected':''}}>broken_cracked</option>
                    <option value="shrivelled_damaged" ${{curLabel==='shrivelled_damaged'?'selected':''}}>shrivelled_damaged</option>
                  </select>
                </div>
                <div class="decision-field">
                  <label>Reviewer:</label>
                  <input type="text" class="input-reviewer" placeholder="Reviewer initials" data-path="${{escapeHtml(crop.crop_path)}}" value="${{escapeHtml(curReviewer)}}">
                </div>
                <div class="decision-field">
                  <label>Diagnostic Notes:</label>
                  <input type="text" class="input-notes" placeholder="Required diagnostic justification" data-path="${{escapeHtml(crop.crop_path)}}" value="${{escapeHtml(curNotes)}}">
                </div>
              </div>
            </div>
          </article>
        `;
      }}).join('');

      grid.innerHTML = htmlCards;
      attachDecisionEvents();
    }}

    function attachDecisionEvents() {{
      grid.querySelectorAll('.input-status').forEach(select => {{
        select.addEventListener('change', (e) => {{
          const p = e.target.getAttribute('data-path');
          const status = e.target.value;
          const card = e.target.closest('.crop-card');
          const labelField = card.querySelector('.label-field');
          const labelSelect = card.querySelector('.input-label');
          const reviewerInput = card.querySelector('.input-reviewer');
          const notesInput = card.querySelector('.input-notes');

          if (status === 'approved') {{
            labelField.style.display = 'flex';
          }} else {{
            labelField.style.display = 'none';
            labelSelect.value = '';
          }}

          if (!reviewerInput.value && globalReviewer.value) {{
            reviewerInput.value = globalReviewer.value;
          }}

          saveDecision(p, status, labelSelect.value, reviewerInput.value, notesInput.value);
        }});
      }});

      grid.querySelectorAll('.input-label').forEach(select => {{
        select.addEventListener('change', (e) => {{
          const p = e.target.getAttribute('data-path');
          const card = e.target.closest('.crop-card');
          const status = card.querySelector('.input-status').value;
          const reviewer = card.querySelector('.input-reviewer').value;
          const notes = card.querySelector('.input-notes').value;
          saveDecision(p, status, e.target.value, reviewer, notes);
        }});
      }});

      grid.querySelectorAll('.input-reviewer').forEach(input => {{
        input.addEventListener('input', (e) => {{
          const p = e.target.getAttribute('data-path');
          const card = e.target.closest('.crop-card');
          const status = card.querySelector('.input-status').value;
          const label = card.querySelector('.input-label').value;
          const notes = card.querySelector('.input-notes').value;
          saveDecision(p, status, label, e.target.value, notes);
        }});
      }});

      grid.querySelectorAll('.input-notes').forEach(input => {{
        input.addEventListener('input', (e) => {{
          const p = e.target.getAttribute('data-path');
          const card = e.target.closest('.crop-card');
          const status = card.querySelector('.input-status').value;
          const label = card.querySelector('.input-label').value;
          const reviewer = card.querySelector('.input-reviewer').value;
          saveDecision(p, status, label, reviewer, e.target.value);
        }});
      }});
    }}

    function saveDecision(path, status, label, reviewer, notes) {{
      if (!status) {{
        delete stagedDecisions[path];
      }} else {{
        stagedDecisions[path] = {{
          crop_path: path,
          review_status: status,
          reviewed_label: status === 'approved' ? label : '',
          reviewer: reviewer || '',
          review_notes: notes || ''
        }};
      }}
      const count = Object.keys(stagedDecisions).length;
      if (count > 0) {{
        stagedCountBadge.style.display = 'inline-block';
        stagedCountBadge.textContent = `${{count}} Staged`;
      }} else {{
        stagedCountBadge.style.display = 'none';
      }}
    }}

    btnExportDecisions.addEventListener('click', () => {{
      const keys = Object.keys(stagedDecisions);
      if (keys.length === 0) {{
        alert('No decisions staged yet. Choose Status (Approve/Reject) on crops to stage decisions.');
        return;
      }}

      // Validate
      const invalid = [];
      const rows = keys.map(k => {{
        const d = stagedDecisions[k];
        if (d.review_status === 'approved' && !d.reviewed_label) {{
          invalid.push(`${{d.crop_path}}: Approved crops must have a target label.`);
        }}
        if (!d.reviewer || !d.reviewer.trim()) {{
          invalid.push(`${{d.crop_path}}: Reviewer identifier is required.`);
        }}
        if (!d.review_notes || !d.review_notes.trim()) {{
          invalid.push(`${{d.crop_path}}: Diagnostic notes are required.`);
        }}
        return [
          `"${{d.crop_path}}"`,
          `"${{d.review_status}}"`,
          `"${{d.reviewed_label}}"`,
          `"${{d.reviewer.replace(/"/g, '""')}}"`,
          `"${{d.review_notes.replace(/"/g, '""')}}"`
        ].join(',');
      }});

      if (invalid.length > 0) {{
        alert(`Validation errors in staged decisions:\\n\\n` + invalid.slice(0, 5).join('\\n') + (invalid.length > 5 ? `\\n...and ${{invalid.length - 5}} more.` : ''));
        return;
      }}

      const csvContent = 'crop_path,review_status,reviewed_label,reviewer,review_notes\\n' + rows.join('\\n');
      const blob = new Blob([csvContent], {{ type: 'text/csv;charset=utf-8;' }});
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.setAttribute('href', url);
      link.setAttribute('download', 'human_review_decisions.csv');
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);

      alert(`Exported ${{keys.length}} human decision(s) to human_review_decisions.csv.\\n\\nApply them reversibly using:\\npython scripts/apply_manual_review.py --manifest data/roboflow_v8_review/manual_review.csv --decisions human_review_decisions.csv`);
    }});

    function escapeHtml(str) {{
      if (!str) return '';
      return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }}

    // Event listeners
    classFilter.addEventListener('change', () => {{ currentPage = 1; render(); }});
    statusFilter.addEventListener('change', () => {{ currentPage = 1; render(); }});
    resolutionFilter.addEventListener('change', () => {{ currentPage = 1; render(); }});
    if (qualityFilter) qualityFilter.addEventListener('change', () => {{ currentPage = 1; render(); }});
    searchInput.addEventListener('input', () => {{ currentPage = 1; render(); }});
    pageSizeSelect.addEventListener('change', (e) => {{
      pageSize = parseInt(e.target.value, 10);
      currentPage = 1;
      render();
    }});

    btnFirst.addEventListener('click', () => {{ currentPage = 1; render(); }});
    btnPrev.addEventListener('click', () => {{ if (currentPage > 1) {{ currentPage--; render(); }} }});
    btnNext.addEventListener('click', () => {{ currentPage++; render(); }});
    btnLast.addEventListener('click', () => {{
      const filtered = getFilteredData();
      currentPage = Math.ceil(filtered.length / pageSize);
      render();
    }});

    btnBatchPrev.addEventListener('click', () => {{
      const newIndex = Math.max(0, (currentPage - 1) * pageSize - batchSize);
      currentPage = Math.floor(newIndex / pageSize) + 1;
      render();
    }});

    btnBatchNext.addEventListener('click', () => {{
      const newIndex = (currentPage - 1) * pageSize + batchSize;
      currentPage = Math.floor(newIndex / pageSize) + 1;
      render();
    }});

    window.addEventListener('keydown', (e) => {{
      if (e.key === 'Escape') {{
        if (sourceModal && sourceModal.style.display === 'flex') {{
          closeSourceModal();
          return;
        }}
        if (lightboxModal && lightboxModal.style.display === 'flex') {{
          closeLightbox();
          return;
        }}
      }}
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
      if (e.key === 'ArrowRight') {{ btnNext.click(); }}
      else if (e.key === 'ArrowLeft') {{ btnPrev.click(); }}
    }});

    render();
  }})();
</script>
</body>
</html>"""

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    missing = sum(not (manifest.parent / row["crop_path"]).is_file() for row in selected_rows)
    print(f"Wrote {output} with {len(crop_records)} crops (manifest total: {len(rows)}); missing crops={missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
