"""Tests for Streamlit UI theme configuration, contrast compliance, and styling accessibility.

Verifies:
- .streamlit/config.toml presence, validity, and theme tokens
- WCAG 2.1 AA contrast ratio compliance (>= 4.5:1 for normal text, >= 7:1 for AAA)
- Quality class badge contrast ratios
- CSS scoping rules preventing white-on-white or unstyled dark-mode artifacts
- App module integrity and page functionality
"""

import math
from pathlib import Path
import re
import tomllib
import pytest


def srgb_to_linear(c_srgb: float) -> float:
    """Convert sRGB channel (0-1) to linear value for relative luminance calculation."""
    if c_srgb <= 0.04045:
        return c_srgb / 12.92
    return ((c_srgb + 0.055) / 1.055) ** 2.4


def hex_to_relative_luminance(hex_code: str) -> float:
    """Calculate WCAG 2.1 relative luminance from hex color string."""
    hex_clean = hex_code.lstrip("#")
    if len(hex_clean) == 3:
        hex_clean = "".join([c * 2 for c in hex_clean])
    r = int(hex_clean[0:2], 16) / 255.0
    g = int(hex_clean[2:4], 16) / 255.0
    b = int(hex_clean[4:6], 16) / 255.0

    r_lin = srgb_to_linear(r)
    g_lin = srgb_to_linear(g)
    b_lin = srgb_to_linear(b)

    return 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin


def calculate_contrast_ratio(hex1: str, hex2: str) -> float:
    """Calculate contrast ratio between two hex colors according to WCAG 2.1."""
    lum1 = hex_to_relative_luminance(hex1)
    lum2 = hex_to_relative_luminance(hex2)
    lighter = max(lum1, lum2)
    darker = min(lum1, lum2)
    return (lighter + 0.05) / (darker + 0.05)


def test_streamlit_config_exists_and_valid():
    """Verify .streamlit/config.toml exists and contains valid TOML syntax."""
    config_path = Path("D:/AI/Projects/SeedVision AI/.streamlit/config.toml")
    assert config_path.is_file(), ".streamlit/config.toml must exist"

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    assert "theme" in config, "config.toml must define a [theme] section"
    theme = config["theme"]
    assert "primaryColor" in theme
    assert "backgroundColor" in theme
    assert "secondaryBackgroundColor" in theme
    assert "textColor" in theme


def test_theme_wcag_contrast_ratios():
    """Verify that .streamlit/config.toml colors meet WCAG 2.1 AA (>= 4.5:1) and AAA standards."""
    config_path = Path("D:/AI/Projects/SeedVision AI/.streamlit/config.toml")
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    theme = config["theme"]
    text_color = theme["textColor"]
    bg_color = theme["backgroundColor"]
    sec_bg_color = theme["secondaryBackgroundColor"]
    primary_color = theme["primaryColor"]

    # Text against Main Canvas Background
    ratio_main = calculate_contrast_ratio(text_color, bg_color)
    assert ratio_main >= 7.0, f"textColor vs backgroundColor contrast ratio ({ratio_main:.2f}) must meet WCAG AAA (>= 7.0)"

    # Text against Secondary (Sidebar/Cards) Background
    ratio_sec = calculate_contrast_ratio(text_color, sec_bg_color)
    assert ratio_sec >= 7.0, f"textColor vs secondaryBackgroundColor contrast ratio ({ratio_sec:.2f}) must meet WCAG AAA (>= 7.0)"

    # Primary brand green on white (e.g. buttons/badges)
    ratio_primary = calculate_contrast_ratio(primary_color, "#ffffff")
    assert ratio_primary >= 4.5, f"primaryColor vs white contrast ratio ({ratio_primary:.2f}) must meet WCAG AA (>= 4.5)"


def test_badge_colors_wcag_compliance():
    """Verify that quality class badge text meets WCAG AA contrast against badge background."""
    badges = {
        "healthy": {"bg": "#dcfce7", "text": "#166534"},
        "spotted": {"bg": "#fef9c3", "text": "#854d0e"},
        "cracked": {"bg": "#ffedd5", "text": "#9a3412"},
        "shrivelled": {"bg": "#fee2e2", "text": "#991b1b"},
    }

    for name, colors in badges.items():
        ratio = calculate_contrast_ratio(colors["text"], colors["bg"])
        assert ratio >= 4.5, f"Badge '{name}' contrast ratio ({ratio:.2f}) must meet WCAG AA (>= 4.5)"


def test_app_css_no_unscoped_white_sidebar():
    """Ensure app.py does not contain hardcoded unscoped white background on sidebar."""
    app_path = Path("D:/AI/Projects/SeedVision AI/app.py")
    content = app_path.read_text(encoding="utf-8")

    # Anti-pattern: section[data-testid="stSidebar"] { background-color: #ffffff; } without text color
    assert 'section[data-testid="stSidebar"] {\n        background-color: #ffffff;' not in content, (
        "Found hardcoded unscoped background-color: #ffffff on sidebar, causing white-on-white text in dark mode."
    )


def test_app_css_has_accessible_sidebar_rules():
    """Verify that app.py defines high-contrast rules for sidebar headings, radio buttons, and active item."""
    app_path = Path("D:/AI/Projects/SeedVision AI/app.py")
    content = app_path.read_text(encoding="utf-8")

    assert 'section[data-testid="stSidebar"] h1' in content
    assert 'section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label' in content
    assert 'section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked)' in content
    assert 'prefers-color-scheme: dark' in content


def test_app_import_and_page_structure():
    """Verify that app.py imports cleanly and retains all page options."""
    import app
    assert hasattr(app, "main")

    app_path = Path("D:/AI/Projects/SeedVision AI/app.py")
    content = app_path.read_text(encoding="utf-8")

    # Verify all 5 pages exist in sidebar navigation
    expected_pages = [
        "Home & Overview",
        "Seed Quality Analysis",
        "Real Data Staging Review",
        "Model Evaluation & Benchmark",
        "Methodology & Limitations",
    ]
    for p in expected_pages:
        assert f'"{p}"' in content


def test_app_header_and_main_padding_optimization():
    """Verify that app.py defines compact top header padding, hides deploy button, and optimizes hero card margins."""
    app_path = Path("D:/AI/Projects/SeedVision AI/app.py")
    content = app_path.read_text(encoding="utf-8")

    # Header height reduction
    assert 'header[data-testid="stHeader"]' in content
    assert 'height: 2.25rem' in content

    # Deploy button hidden without breaking controls
    assert '[data-testid="stDeployButton"]' in content
    assert 'display: none !important;' in content

    # Block container padding-top optimized (not default 6rem)
    assert '.block-container' in content
    assert 'padding-top: 1.5rem !important;' in content

    # Hero card top margin eliminated and bottom margin clean
    assert '.hero-card {' in content
    assert 'margin-top: 0 !important;' in content

    # Mobile/responsive rules present
    assert '@media (max-width: 768px)' in content
