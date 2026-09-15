"""Robust JSON extraction from LLM text responses.

Handles two formats observed in the wild:

1. Fenced code blocks (preferred): triple-backtick with an optional
   ``json`` language tag.
2. Bare JSON inside surrounding prose — fall back to the outermost
   ``{...}`` span found by greedy regex.

Either candidate gets a second attempt with invalid escapes repaired.

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

# Matches the escape PAIR. A lone-backslash scan would eat the first half of a correct ``\\``
# and "repair" the second half into a broken ``\p``.
_ESCAPE_RE = re.compile(r"\\(.)", re.DOTALL)
_VALID_ESCAPES = frozenset('"\\/bfnrtu')


def _repair_invalid_escapes(candidate: str) -> str:
    r"""Double every backslash that does not begin a legal JSON escape.

    Models copy LaTeX out of the source verbatim, so ``$\kappa$`` arrives as ``\k`` and
    ``json.loads`` rejects an otherwise complete response, losing the whole document. Lossy
    where a command collides with a legal escape (``\theta`` / ``\frac`` / ``\bar`` are tab /
    form-feed / backspace): those must be left alone, or genuine escapes would break. The real
    fix is upstream — ask the model for ``response_format={"type": "json_object"}``.
    """
    return _ESCAPE_RE.sub(
        lambda m: m.group(0) if m.group(1) in _VALID_ESCAPES else "\\" + m.group(0),
        candidate,
    )


def _loads_dict(candidate: str) -> dict[str, Any] | None:
    """``json.loads``, retried once with escapes repaired; ``None`` if neither attempt works."""
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        try:
            parsed = json.loads(_repair_invalid_escapes(candidate))
        except json.JSONDecodeError:
            return None
        logger.warning("LLM JSON recovered by repairing invalid backslash escapes")
    return parsed if isinstance(parsed, dict) else None


def parse_llm_json(response: str) -> dict[str, Any] | None:
    """Try to extract a JSON object from an LLM text response.

    Returns ``None`` on any parse failure; logs at WARNING.
    """
    if not response:
        return None

    fence_match = _FENCE_RE.search(response)
    if fence_match is not None:
        parsed = _loads_dict(fence_match.group(1))
        if parsed is not None:
            return parsed

    bare_match = _OUTERMOST_BRACE_RE.search(response)
    if bare_match is None:
        logger.warning("LLM response contains no JSON object")
        return None
    parsed = _loads_dict(bare_match.group())
    if parsed is None:
        logger.warning("failed to parse JSON from LLM response")
    return parsed
