"""LLM rerank tools — dual interface. Stubs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from evercore.llm.protocols import LLMClient

__all__ = ["arerank", "rerank"]


async def arerank(
    items: Sequence[Any],
    *,
    prompt: str,
    top_k: int,
    llm: LLMClient | None = None,
) -> list[Any]:
    """LLM rerank async. Stub — TBD."""
    raise NotImplementedError("stub")


def rerank(
    items: Sequence[Any],
    *,
    prompt: str,
    top_k: int,
    llm: LLMClient | None = None,
) -> list[Any]:
    """LLM rerank sync (bridges async). Stub — TBD."""
    raise NotImplementedError("stub")
