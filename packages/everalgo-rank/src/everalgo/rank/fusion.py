"""Fusion algorithm tools — pure compute. Stubs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["cosine_to_lr_score", "lr", "rrf", "score_propagation"]


def rrf(*sources: Sequence[Any], k: int = 60) -> list[Any]:
    """Reciprocal Rank Fusion. Stub — TBD."""
    raise NotImplementedError("stub")


def lr(*sources: Sequence[Any], weights: Sequence[float]) -> list[Any]:
    """Logistic Regression fusion. Stub — TBD."""
    raise NotImplementedError("stub")


def cosine_to_lr_score(sim: float, alpha: float) -> float:
    """Cosine similarity → LR score conversion. Stub — TBD."""
    raise NotImplementedError("stub")


def score_propagation(
    parents: Sequence[Any],
    children: Sequence[Any],
    *,
    alpha: float,
) -> list[Any]:
    """Hierarchical score propagation. Stub — TBD."""
    raise NotImplementedError("stub")
