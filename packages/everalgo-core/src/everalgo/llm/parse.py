"""Robust LLM JSON-object parsing.

Used wherever an LLM returns a string that should decode to a JSON object. Handles the common
LLM output quirks (markdown fence wrapping, prose preamble around the JSON, etc.) on top of
OpenAI JSON mode's happy path.
"""

from __future__ import annotations

import json
import re
from typing import Any, cast

__all__ = ["parse_llm_json_object"]


def parse_llm_json_object(raw: str) -> dict[str, Any]:
    """Parse ``raw`` as a JSON object via three strategies in order.

    Tries each strategy until one succeeds:

    1. `` ```json ... ``` `` fenced block
    2. Direct ``json.loads`` on the full string (the JSON-mode happy path)
    3. Outermost ``{ ... }`` substring (rescues prose-wrapped JSON)

    Args:
        raw: Raw string returned by the LLM.

    Returns:
        The parsed dict.

    Raises:
        ValueError: If all three strategies fail or the result is not a JSON object.
    """
    fence_match = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL)
    if fence_match:
        try:
            parsed = json.loads(fence_match.group(1))
            if isinstance(parsed, dict):
                return cast("dict[str, Any]", parsed)
        except (json.JSONDecodeError, ValueError):
            pass

    try:
        parsed = json.loads(raw.strip())
        if isinstance(parsed, dict):
            return cast("dict[str, Any]", parsed)
    except (json.JSONDecodeError, ValueError):
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(raw[start : end + 1])
            if isinstance(parsed, dict):
                return cast("dict[str, Any]", parsed)
        except (json.JSONDecodeError, ValueError):
            pass

    raise ValueError("Failed to parse LLM response as a JSON object")
