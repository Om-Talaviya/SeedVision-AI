# Contributing to SeedVision AI

Thank you for your interest in contributing to **SeedVision AI**! We welcome contributions to improve deep learning architectures, dataset validation workflows, explainability techniques, and user interface ergonomics.

## Code of Conduct
We are committed to providing a friendly, safe, and welcoming environment for all contributors regardless of background or level of experience.

## Development Workflow
1. **Fork and Clone** the repository.
2. **Set up a Virtual Environment**:
   ```bash
   python -m venv venv
   # Linux/macOS:
   source venv/bin/activate
   # Windows:
   .\venv\Scripts\Activate.ps1
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```
4. **Run Unit Tests**:
   ```bash
   pytest tests/
   ```

## Contribution Guidelines
- **Modularity & Types**: Write maintainable, modular, typed Python code.
- **Scientific Rigor**: Never fabricate metrics, evaluation results, or dataset splits. Ensure all split operations prevent sample-level leakage.
- **Explainability**: Ensure Grad-CAM hooks are properly cleaned up after inference to prevent memory leaks.
- **Testing**: Add unit tests in `tests/` for any new model architecture, preprocessing pipeline, or metric calculation.
- **Commit Messages**: Write clear, descriptive commit messages following standard conventional commits (e.g., `feat: add test-time augmentation`, `fix: resolve gradcam tensor dimension mismatch`).
