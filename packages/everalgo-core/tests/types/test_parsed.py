"""Tests for ``everalgo.types.parsed.ParsedContent``."""

from __future__ import annotations

from everalgo.types import Modality, ParsedContent


def test_parsed_content_defaults() -> None:
    pc = ParsedContent()
    assert pc.text == ""
    assert pc.modality is Modality.UNKNOWN
    assert pc.mime == ""
    assert pc.metadata == {}


def test_parsed_content_round_trip() -> None:
    pc = ParsedContent(
        text="hello",
        modality=Modality.PDF,
        mime="application/pdf",
        metadata={"page_count": 3, "model": "gemini-2.5-flash"},
    )
    rebuilt = ParsedContent.model_validate_json(pc.model_dump_json())
    assert rebuilt == pc


def test_parsed_content_extra_fields_ignored() -> None:
    pc = ParsedContent.model_validate(
        {
            "text": "x",
            "modality": "image",
            "mime": "image/png",
            "metadata": {},
            "legacy_field": "ignored",
        }
    )
    assert pc.text == "x"
    assert pc.modality is Modality.IMAGE
    assert not hasattr(pc, "legacy_field")
