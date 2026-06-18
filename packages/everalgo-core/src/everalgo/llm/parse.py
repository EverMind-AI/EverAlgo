"""Robust LLM JSON-object parsing and answer extraction helpers.

Provides two utilities used on the LLM output path:

- ``parse_llm_json_object`` — multi-strategy JSON object extraction (fence, direct, braces).
- ``extract_final_answer`` — marker-based final-answer substring slicing.
"""

from __future__ import annotations

import json
import re
from typing import Any, cast

__all__ = ["extract_final_answer", "parse_llm_json_object"]


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


def extract_final_answer(raw: str, *, marker: str = "Final answer:") -> str:
    """Extract the substring after ``marker`` from LLM text output.

    Uses ``rsplit`` to take the portion after the **last** occurrence of ``marker``,
    which handles cases where the marker text appears in reasoning prose before the
    actual answer section.

    Args:
        raw: Raw LLM text output (may contain prose / chain-of-thought before the marker).
        marker: Sentinel string preceding the final answer. Default ``"Final answer:"``
            follows the answer-prompt convention with a leading ``Final answer:`` marker.

    Returns:
        The substring after the last occurrence of ``marker``, stripped of surrounding
        whitespace. If ``marker`` is not found, the entire ``raw`` text is returned stripped.
    """
    if marker not in raw:
        return raw.strip()
    return raw.rsplit(marker, 1)[1].strip()
