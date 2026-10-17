# SeedVision AI — Static Showcase Site

A dependency-free, fully static showcase for the **SeedVision AI** pulse-seed
quality-grading project. It presents the pipeline, the model comparison, the
Grad-CAM explainability layer, the advisory review workflow, an illustrative
in-browser demo, and the project's scientific limitations — with **no build
step, no CDN, and no network calls**.

```
docs/site/
├── index.html   # single-page site
├── styles.css   # design tokens + responsive layout
├── app.js       # nav, scroll-reveal, lightbox, simulated demo
└── README.md    # this file
```

Screenshots are consumed from the sibling directory as `../assets/*.png`, so
the site works whether opened from disk or served from the repository root.

---

## 1 · Preview locally

No server is required — open the file directly:

```
docs/site/index.html
```

If you prefer a local HTTP server (recommended, since it matches how GitHub
Pages serves the files):

```powershell
# from the repository root
python -m http.server 8000 --directory docs
# then browse to http://localhost:8000/site/
```

```bash
# macOS / Linux equivalent
python3 -m http.server 8000 --directory docs
```

> The site references images through relative paths only. Because the assets
> live at `docs/assets/`, opening the page as a `file://` URL works as long as
> the `docs` directory is kept intact.

---

## 2 · Publish with GitHub Pages

Two options; pick one. The site lives under `/docs/site/`, so the published
URL is `https://om-talaviya.github.io/SeedVision-AI/site/`.

### Option A — Serve from the `/docs` folder (recommended)

1. Push the repository to GitHub.
2. **Settings → Pages**.
3. Under **Build and deployment → Source**, choose **Deploy from a branch**.
4. Set **Branch** to `main` and **Folder** to `/docs`.
5. Save. After the first build, the site is at:
   `https://om-talaviya.github.io/SeedVision-AI/site/`

The repository's own Markdown lives beside the site under `/docs`, and the
*Setup & links* section uses absolute
`https://github.com/Om-Talaviya/SeedVision-AI/blob/main/…` URLs, so those links
resolve from both the local file and the published page.

### Option B — Serve the site folder with a GitHub Actions workflow

Use this if you want the site at the repository root URL
(`https://om-talaviya.github.io/SeedVision-AI/`) instead of under `/site/`.

```yaml
# .github/workflows/pages.yml
name: Deploy showcase site

on:
  push:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: true

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - uses: actions/checkout@v4
      - name: Stage site with assets
        run: |
          mkdir -p public/assets
          cp -r docs/site/. public/
          cp docs/assets/*.png public/assets/
      - uses: actions/configure-pages@v5
      - uses: actions/upload-pages-artifact@v3
        with:
          path: public
      - id: deployment
        uses: actions/deploy-pages@v4
```

Then set **Settings → Pages → Source** to **GitHub Actions**.

> With Option B the site sits at the published root, so the `../assets/…`
> paths still resolve because the assets are copied into `public/assets/`.
> The *Setup & links* documentation links already use absolute
> `https://github.com/Om-Talaviya/SeedVision-AI/blob/main/…` URLs, so they stay
> clickable under either option.

No `.nojekyll` file is needed — no files or directories start with an
underscore.

---

## 3 · What the demo does (and does not do)

The *Illustrative demo* section runs **entirely in the browser**. There is no
Python runtime, no model weights, and no network request.

- **Input** — either one of four bundled synthetic colour swatches drawn on a
  `<canvas>`, or an image you upload. Uploaded files never leave the machine.
- **Analysis** — a deterministic pixel-statistics heuristic computes mean
  brightness, luminance spread (σ), a Laplacian-variance focus metric, dark-pixel
  fraction, resolution and aspect ratio.
- **Output** — a quality-screening verdict, four KPIs, a findings list, and a
  **simulated** grade placeholder.

The word *simulated* is shown on the panel and in the findings. The simulated
grade is a generic threshold heuristic; it is **not** the trained CNN, and it
must not be read as an accuracy claim, a benchmark, or a validation result.
The real grader's numbers exist only as artifacts produced by the Python
runtime (`artifacts/metrics/`, `artifacts/checkpoints/`).

> **Honesty note.** All reportable metrics in this project come from a
> synthetic benchmark. `is_real_world_validated` is `False` throughout the
> evaluation artifacts, and the six real photographs in `data/real_staging/`
> remain unlabelled and unapproved. The site preserves that boundary and
> labels the demo as illustrative rather than validated.

---

## 4 · Accessibility & browser support

- Semantic landmarks, a skip link, and visible focus rings.
- Lightbox is a modal dialog with `Escape` dismissal, focus trapping, and focus
  restoration.
- Screenshots are lazy-loaded with intrinsic `width`/`height` to avoid layout
  shift; every image carries descriptive `alt` text.
- Motion is suppressed under `prefers-reduced-motion: reduce`; scroll-reveal
  degrades to an immediate, static presentation.
- No JavaScript? The page still renders fully — only the demo, nav toggle, and
  lightbox are enhanced by `app.js`.

Targets current evergreen Chrome, Edge, Firefox, and Safari.

---

## 5 · Editing the site

| Change | Where |
| :--- | :--- |
| Colours, spacing, type scale | `styles.css` → `:root` custom properties |
| Copy, section order, nav anchors | `index.html` |
| Adding a screenshot | Add the PNG to `docs/assets/`, then add a `.shot` button (with `data-full` / `data-title`) to `#galleryGrid` |
| Demo heuristics and thresholds | `app.js` → `WORK`, `MIN_INPUT`, `BRIGHT_*`, `CONTRAST_LOW`, `SHARP_*`, `suggestGrade()` |

Keep the forest-green + cream + white palette and the honesty framing when
editing: never present synthetic-benchmark results as real-world performance,
and never imply that optical classification verifies germination or internal
viability.
