"""Unit tests for everalgo.parser._utils helpers (no LLM, no network)."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from everalgo.parser._utils import (
    check_aspect_ratio,
    clean_html_content,
    clean_html_for_llm,
    split_image_with_overlap,
)


def _png_bytes(width: int, height: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color="white").save(buf, format="PNG")
    return buf.getvalue()


# ---- check_aspect_ratio ----


def test_check_aspect_ratio_normal_image_not_tall() -> None:
    needs_split, ratio = check_aspect_ratio(_png_bytes(100, 200))
    assert needs_split is False
    assert ratio == pytest.approx(2.0)  # pyright: ignore[reportUnknownMemberType]


def test_check_aspect_ratio_tall_image_needs_split() -> None:
    needs_split, ratio = check_aspect_ratio(_png_bytes(100, 2000))
    assert needs_split is True
    assert ratio == pytest.approx(20.0)  # pyright: ignore[reportUnknownMemberType]


def test_check_aspect_ratio_custom_max_ratio() -> None:
    # 200 / 100 = 2.0; with max_ratio=1.5 it should need split
    needs_split, ratio = check_aspect_ratio(_png_bytes(100, 200), max_ratio=1.5)
    assert needs_split is True
    assert ratio == pytest.approx(2.0)  # pyright: ignore[reportUnknownMemberType]


def test_check_aspect_ratio_square_image() -> None:
    needs_split, ratio = check_aspect_ratio(_png_bytes(100, 100))
    assert needs_split is False
    assert ratio == pytest.approx(1.0)  # pyright: ignore[reportUnknownMemberType]


# ---- split_image_with_overlap ----


def test_split_image_with_overlap_short_image_returns_input() -> None:
    img = _png_bytes(100, 200)
    parts = split_image_with_overlap(img, max_ratio=10.0)
    assert parts == [img]


def test_split_image_with_overlap_tall_image_produces_multiple_parts() -> None:
    # height=2500, width=100 → ratio=25; ceil(25/10)=3 parts expected
    img = _png_bytes(100, 2500)
    parts = split_image_with_overlap(img, max_ratio=10.0)
    assert len(parts) >= 2
    for part in parts:
        loaded = Image.open(io.BytesIO(part))
        assert loaded.width == 100


def test_split_image_with_overlap_parts_are_valid_png() -> None:
    img = _png_bytes(50, 600)
    parts = split_image_with_overlap(img, max_ratio=5.0)
    for part in parts:
        # PIL.Image.open must not raise
        loaded = Image.open(io.BytesIO(part))
        assert loaded.format in ("PNG", "JPEG", None)


def test_split_image_at_exactly_max_ratio_boundary() -> None:
    # ratio == max_ratio exactly → should NOT split (not strictly greater)
    img = _png_bytes(100, 1000)
    parts = split_image_with_overlap(img, max_ratio=10.0)
    assert parts == [img]


# ---- clean_html_content ----


def test_clean_html_content_empty_string() -> None:
    assert clean_html_content("") == ""


def test_clean_html_content_whitespace_only() -> None:
    assert clean_html_content("   \n  ") == ""


def test_clean_html_content_strips_tags() -> None:
    html = "<html><body><h1>Title</h1><p>Body text</p></body></html>"
    out = clean_html_content(html)
    assert "Title" in out
    assert "Body text" in out
    # Tags themselves must be stripped
    assert "<h1>" not in out
    assert "<p>" not in out


def test_clean_html_content_removes_script_and_style() -> None:
    html = "<html><body><script>bad_js()</script><style>.x{}</style><p>keep</p></body></html>"
    out = clean_html_content(html)
    assert "bad_js" not in out
    assert ".x{}" not in out
    assert "keep" in out


def test_clean_html_content_collapses_triple_newlines() -> None:
    html = "<p>a</p>\n\n\n\n<p>b</p>"
    out = clean_html_content(html)
    assert "\n\n\n" not in out


# ---- clean_html_for_llm ----


def test_clean_html_for_llm_empty_input() -> None:
    assert clean_html_for_llm("") == ""


def test_clean_html_for_llm_keeps_structural_tags() -> None:
    html = "<html><body><h1>Title</h1><p>Hello</p></body></html>"
    out = clean_html_for_llm(html)
    assert "<h1>" in out
    assert "<p>" in out
    assert "Title" in out
    assert "Hello" in out


def test_clean_html_for_llm_removes_noise_tags() -> None:
    html = "<html><body><nav>Skip</nav><p>Content</p><footer>Footer</footer><style>.x{}</style></body></html>"
    out = clean_html_for_llm(html)
    assert "<nav>" not in out
    assert "<footer>" not in out
    assert "<style>" not in out
    assert "Content" in out


def test_clean_html_for_llm_strips_bloat_attributes() -> None:
    html = '<html><body><p class="big" style="color:red" id="foo">Text</p></body></html>'
    out = clean_html_for_llm(html)
    assert 'class="big"' not in out
    assert 'style="color:red"' not in out
    assert 'id="foo"' not in out
    assert "Text" in out


def test_clean_html_for_llm_removes_html_comments() -> None:
    html = "<html><body><!-- a comment --><p>Visible</p></body></html>"
    out = clean_html_for_llm(html)
    assert "a comment" not in out
    assert "Visible" in out
