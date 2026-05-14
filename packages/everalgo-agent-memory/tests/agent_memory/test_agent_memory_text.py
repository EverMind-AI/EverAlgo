"""Tests for everalgo.agent_memory._text — token-aware truncation + JSON default.

Port of opensource ``TestTruncateText`` (case extractor) + ``TestTruncateText`` (skill extractor) + edge
cases. Two modes covered:

- ``suffix is None`` → head + tail with ``"[... trimmed N tokens ...]"`` marker (case-extractor pattern)
- ``suffix is not None`` → head-only + suffix append (skill-extractor pattern)
"""

from __future__ import annotations

from datetime import datetime

from everalgo._tokenize import count_tokens
from everalgo.agent_memory._text import json_default, truncate_text


def _long_text(token_count: int) -> str:
    return " ".join(f"word_{i}" for i in range(token_count))


# ── truncate_text — head+tail mode ──────────────────────────────────────────────────────────────────


class TestTruncateTextHeadTailMode:
    def test_empty_returns_empty(self) -> None:
        assert truncate_text("", max_tokens=10) == ""

    def test_short_text_unchanged(self) -> None:
        text = "hello world"
        assert truncate_text(text, max_tokens=100) == text

    def test_long_text_carries_trim_marker(self) -> None:
        text = _long_text(500)
        result = truncate_text(text, max_tokens=50)
        assert "[... trimmed" in result
        assert "tokens ...]" in result

    def test_truncated_has_head_and_tail(self) -> None:
        text = _long_text(500)
        result = truncate_text(text, max_tokens=100, head_ratio=0.7)
        assert result.startswith("word_0")
        # Tail should still appear after the marker
        parts = result.split("[... trimmed")
        assert len(parts) == 2
        assert "tokens ...]" in parts[1]
        # The tail half should contain later words (sanity)
        assert "word_" in parts[1]

    def test_head_ratio_respected(self) -> None:
        """head_ratio=0.7 keeps ~70% of budget as head; smaller ratio yields shorter head."""
        text = _long_text(500)
        head_70 = truncate_text(text, max_tokens=100, head_ratio=0.7).split("\n[... trimmed")[0]
        head_50 = truncate_text(text, max_tokens=100, head_ratio=0.5).split("\n[... trimmed")[0]
        assert count_tokens(head_70) > count_tokens(head_50)

    def test_head_ratio_one_yields_head_only_with_ellipsis(self) -> None:
        text = _long_text(500)
        result = truncate_text(text, max_tokens=50, head_ratio=1.0)
        assert result.endswith("...")
        assert "[... trimmed" not in result

    def test_exact_limit_unchanged(self) -> None:
        text = _long_text(100)
        n = count_tokens(text)
        assert truncate_text(text, max_tokens=n) == text

    def test_result_is_shorter_than_original(self) -> None:
        text = _long_text(1000)
        result = truncate_text(text, max_tokens=50)
        assert len(result) < len(text)


# ── truncate_text — head-only suffix mode ───────────────────────────────────────────────────────────


class TestTruncateTextSuffixMode:
    def test_short_text_unchanged_even_with_suffix(self) -> None:
        text = "short"
        assert truncate_text(text, max_tokens=100, suffix="... [omitted]") == text

    def test_long_text_truncated_with_suffix(self) -> None:
        text = _long_text(500)
        result = truncate_text(text, max_tokens=50, suffix="... [omitted]")
        assert result.endswith("... [omitted]")
        assert "[... trimmed" not in result  # head-only mode, no head+tail marker
        assert result.startswith("word_0")

    def test_empty_suffix_string_still_truncates(self) -> None:
        """Empty string suffix still selects suffix mode (suffix is not None)."""
        text = _long_text(500)
        result = truncate_text(text, max_tokens=20, suffix="")
        # Truncated head, no marker
        assert "[... trimmed" not in result
        assert result.startswith("word_0")
        # Should be shorter than original
        assert len(result) < len(text)

    def test_custom_suffix(self) -> None:
        text = _long_text(500)
        result = truncate_text(text, max_tokens=30, suffix="<<TRUNCATED>>")
        assert result.endswith("<<TRUNCATED>>")


# ── json_default ────────────────────────────────────────────────────────────────────────────────────


class TestJsonDefault:
    def test_datetime_to_isoformat(self) -> None:
        assert json_default(datetime(2025, 3, 1, 12, 0, 0)) == "2025-03-01T12:00:00"

    def test_set_falls_back_to_str(self) -> None:
        result = json_default({1, 2, 3})
        assert isinstance(result, str)

    def test_bytes_falls_back_to_str(self) -> None:
        result = json_default(b"hello")
        assert isinstance(result, str)
        assert "hello" in result
