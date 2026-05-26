"""Tests for the multimodal content-part extension of ``ChatMessage``."""

from __future__ import annotations

import base64
import json
from typing import Any, cast

import pytest
from pydantic import ValidationError

from everalgo.llm.types import (
    ChatMessage,
    ImageUrlInner,
    ImageUrlPart,
    TextPart,
    image_url_part_from_bytes,
)


def test_text_part_serializes_with_type_field() -> None:
    part = TextPart(text="hello")
    dumped = part.model_dump()
    assert dumped == {"type": "text", "text": "hello"}


def test_image_url_part_default_detail_omitted_or_none() -> None:
    """ImageUrlInner.detail defaults to None and round-trips."""
    part = ImageUrlPart(image_url=ImageUrlInner(url="data:image/png;base64,AAA"))
    dumped = part.model_dump()
    assert dumped["type"] == "image_url"
    assert dumped["image_url"]["url"] == "data:image/png;base64,AAA"
    assert dumped["image_url"]["detail"] is None


def test_image_url_part_detail_high_round_trip() -> None:
    part = ImageUrlPart(image_url=ImageUrlInner(url="http://x", detail="high"))
    rebuilt = ImageUrlPart.model_validate_json(part.model_dump_json())
    assert rebuilt == part


def test_image_url_part_from_bytes_produces_data_uri() -> None:
    raw = b"%PDF-1.4 test"
    part = image_url_part_from_bytes(raw, "application/pdf")
    expected_b64 = base64.b64encode(raw).decode("ascii")
    assert part.image_url.url == f"data:application/pdf;base64,{expected_b64}"


def test_chat_message_accepts_string_content_unchanged() -> None:
    """Existing API: content as plain str still works."""
    msg = ChatMessage(role="user", content="hi")
    assert msg.content == "hi"
    assert json.loads(msg.model_dump_json()) == {"role": "user", "content": "hi"}


def test_chat_message_accepts_multimodal_content_list() -> None:
    """New API: content as list of TextPart + ImageUrlPart."""
    msg = ChatMessage(
        role="user",
        content=[
            TextPart(text="describe this"),
            image_url_part_from_bytes(b"PNGDATA", "image/png"),
        ],
    )
    dumped = msg.model_dump()
    assert dumped["role"] == "user"
    assert isinstance(dumped["content"], list)
    content = cast("list[dict[str, Any]]", dumped["content"])
    assert len(content) == 2
    assert content[0] == {"type": "text", "text": "describe this"}
    assert content[1]["type"] == "image_url"
    assert cast("str", content[1]["image_url"]["url"]).startswith("data:image/png;base64,")


def test_chat_message_multimodal_json_round_trip() -> None:
    msg = ChatMessage(
        role="user",
        content=[
            TextPart(text="extract"),
            ImageUrlPart(image_url=ImageUrlInner(url="https://example.com/x.png")),
        ],
    )
    rebuilt = ChatMessage.model_validate_json(msg.model_dump_json())
    assert rebuilt == msg


def test_chat_message_unknown_part_type_raises() -> None:
    """Discriminator rejects unknown ``type`` values in content parts."""
    with pytest.raises(ValidationError):
        ChatMessage.model_validate(
            {
                "role": "user",
                "content": [{"type": "unknown_kind", "text": "x"}],
            }
        )
