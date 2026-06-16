"""Robust JSON extraction from LLM text responses.

Handles two formats observed in the wild:

1. Fenced code blocks (preferred): triple-backtick with an optional
   ``json`` language tag.
2. Bare JSON inside surrounding prose — fall back to the outermost
   ``{...}`` span found by greedy regex.

Returns ``None`` rather than raising on parse failure, since callers
typically log and keep prior state.

NOT exposed in the package ``__all__`` — internal to the knowledge extractor.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

__all__ = ["parse_llm_json"]

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_OUTERMOST_BRACE_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_llm_json(response: str) -> dict[str, Any] | None:
    """Try to extract a JSON object from an LLM text response.

    Returns ``None`` on any parse failure; logs at WARNING.
    """
    if not response:
        return None

    fence_match = _FENCE_RE.search(response)
    if fence_match is not None:
        try:
            parsed = json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(parsed, dict):
                return parsed

    bare_match = _OUTERMOST_BRACE_RE.search(response)
    if bare_match is None:
        logger.warning("LLM response contains no JSON object")
        return None
    try:
        parsed = json.loads(bare_match.group())
    except json.JSONDecodeError:
        logger.warning("failed to parse JSON from LLM response")
        return None
    return parsed if isinstance(parsed, dict) else None
