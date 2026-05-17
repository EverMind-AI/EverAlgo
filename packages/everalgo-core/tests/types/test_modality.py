"""Tests for ``everalgo.types.modality``."""

from __future__ import annotations

import pytest

from everalgo.types import (
    EXTENSION_TO_MODALITY,
    MIME_TO_EXTENSION,
    MIME_TO_MODALITY,
    SUPPORTED_EXTENSIONS,
    SUPPORTED_MIMES,
    Modality,
    get_extension_from_mime,
    get_modality,
    get_modality_from_mime,
)


def test_modality_values_are_lowercase_strings() -> None:
    for member in Modality:
        assert member.value.islower()
        assert member.value == member.value.strip()


def test_modality_is_str_subclass() -> None:
    """str-enum so ``Modality.PDF == 'pdf'`` works for JSON / config lookup."""
    assert Modality.PDF == "pdf"  # type: ignore[comparison-overlap]


def test_modality_covers_eight_classified_plus_unknown() -> None:
    expected = {"image", "pdf", "audio", "document", "html", "email", "url", "direct", "unknown"}
    assert {m.value for m in Modality} == expected


@pytest.mark.parametrize(
    ("extension", "expected"),
    [
        ("png", Modality.IMAGE),
        ("jpg", Modality.IMAGE),
        ("PDF", Modality.PDF),  # case-insensitive
        (".pdf", Modality.PDF),  # leading dot tolerated
        ("mp3", Modality.AUDIO),
        ("docx", Modality.DOCUMENT),
        ("xlsx", Modality.DOCUMENT),
        ("pages", Modality.DOCUMENT),
        ("html", Modality.HTML),
        ("eml", Modality.EMAIL),
        ("txt", Modality.DIRECT),
        ("md", Modality.DIRECT),
        ("unknown_ext", Modality.UNKNOWN),
        ("", Modality.UNKNOWN),
    ],
)
def test_get_modality_classifies_known_and_unknown(extension: str, expected: Modality) -> None:
    assert get_modality(extension) is expected


def test_extension_to_modality_is_lowercase_keys() -> None:
    for key in EXTENSION_TO_MODALITY:
        assert key.islower()
        assert "." not in key


def test_supported_extensions_matches_mapping_keys() -> None:
    assert frozenset(EXTENSION_TO_MODALITY.keys()) == SUPPORTED_EXTENSIONS


# ---- MIME-based dispatch ----


@pytest.mark.parametrize(
    ("mime", "expected"),
    [
        ("image/png", Modality.IMAGE),
        ("image/jpeg", Modality.IMAGE),
        ("IMAGE/PNG", Modality.IMAGE),  # case-insensitive
        ("image/png; charset=utf-8", Modality.IMAGE),  # params stripped
        ("application/pdf", Modality.PDF),
        ("audio/mpeg", Modality.AUDIO),
        ("audio/wav", Modality.AUDIO),
        ("audio/x-flac", Modality.AUDIO),
        ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", Modality.DOCUMENT),
        ("application/msword", Modality.DOCUMENT),
        ("application/rtf", Modality.DOCUMENT),
        ("text/html", Modality.HTML),
        ("application/xhtml+xml", Modality.HTML),
        ("text/html; charset=utf-8", Modality.HTML),
        ("message/rfc822", Modality.EMAIL),
        ("text/plain", Modality.DIRECT),
        ("text/markdown", Modality.DIRECT),
        ("text/csv", Modality.DIRECT),
        ("application/x-totally-made-up", Modality.UNKNOWN),
        ("", Modality.UNKNOWN),
    ],
)
def test_get_modality_from_mime_classifies(mime: str, expected: Modality) -> None:
    assert get_modality_from_mime(mime) is expected


@pytest.mark.parametrize(
    ("mime", "expected_ext"),
    [
        ("application/pdf", "pdf"),
        ("image/png", "png"),
        ("image/jpeg", "jpg"),
        ("image/jpeg; charset=binary", "jpg"),  # params stripped
        ("audio/mpeg", "mp3"),
        ("audio/wav", "wav"),
        ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", "docx"),
        ("text/html", "html"),
        ("message/rfc822", "eml"),
        ("text/plain", "txt"),
        ("application/x-totally-made-up", ""),
        ("", ""),
    ],
)
def test_get_extension_from_mime_canonical(mime: str, expected_ext: str) -> None:
    assert get_extension_from_mime(mime) == expected_ext


def test_mime_to_modality_keys_are_lowercase() -> None:
    for key in MIME_TO_MODALITY:
        assert key == key.lower()


def test_mime_to_extension_is_a_subset_of_supported_extensions() -> None:
    """Every canonical extension we produce must be a supported parser extension."""
    for ext in MIME_TO_EXTENSION.values():
        assert ext in SUPPORTED_EXTENSIONS, f"canonical ext {ext!r} not in SUPPORTED_EXTENSIONS"


def test_supported_mimes_matches_mapping_keys() -> None:
    assert frozenset(MIME_TO_MODALITY.keys()) == SUPPORTED_MIMES


def test_mime_and_extension_dispatch_agree_for_canonical_pairs() -> None:
    """If a MIME maps to extension X, then get_modality(X) == get_modality_from_mime(MIME)."""
    for mime, ext in MIME_TO_EXTENSION.items():
        ext_modality = get_modality(ext)
        mime_modality = get_modality_from_mime(mime)
        assert ext_modality is mime_modality, (
            f"dispatch disagreement for mime={mime!r} ext={ext!r}: ext→{ext_modality.value} mime→{mime_modality.value}"
        )
