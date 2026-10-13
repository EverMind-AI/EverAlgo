"""The tall-image merge call site sends the model a user message with no slot left in it.

Asserting on the template alone would not catch the failure this call is exposed to: a call site that went back to
``str.replace`` would substitute inside the template's ``{{parts}}`` and ship the OCR text wrapped in stray braces —
no exception, no missing field. So the test drives ``aparse`` against a ``FakeLLMClient`` with a tall image and
compares the merge user message the client actually received with the expected rendering, spelled out literally.
"""

from __future__ import annotations

import io
import json
from typing import Any

from PIL import Image

from everalgo.parser import image
from everalgo.parser.prompts.en.image import PROMPT_FOR_MERGE
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import RawFile


def _tall_png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (100, 3000), color="white").save(buf, format="PNG")
    return buf.getvalue()


async def test_tall_image_merge_messages_carry_rules_and_tagged_parts() -> None:
    # The slice count follows the overlap maths, so the OCR texts are numbered as they are asked for
    # rather than scripted; the merge call is whichever one comes last.
    ocr_texts: list[str] = []

    def _handler(*_args: Any, **_kwargs: Any) -> str:
        ocr_texts.append(f"slice {len(ocr_texts) + 1} {{{{parts}}}} text")
        return json.dumps({"text": ocr_texts[-1]})

    client = FakeLLMClient(handler=_handler)
    await image.aparse(RawFile(content=_tall_png(), extension="png"), llm=client)

    # The merge call consumed a response too, so it contributed the last entry; drop it.
    slices = ocr_texts[:-1]
    assert len(slices) == 3, "the image was not tall enough to reach the merge call"
    system, user = client.calls[-1].messages
    assert (system.role, system.content) == ("system", PROMPT_FOR_MERGE)
    assert user.role == "user"
    assert user.content == (
        '<ocr_parts count="3">\n'
        '<part number="1">\nslice 1 {{parts}} text\n</part>\n'
        '<part number="2">\nslice 2 {{parts}} text\n</part>\n'
        '<part number="3">\nslice 3 {{parts}} text\n</part>\n'
        "</ocr_parts>"
    )
