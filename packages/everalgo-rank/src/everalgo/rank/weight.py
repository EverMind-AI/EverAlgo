"""Weighted score tools — pure compute. Stubs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["multi_field_weighting", "weighted_score"]


def weighted_score(items: Sequence[Any], *, fields: dict[str, float]) -> list[Any]:
    """Single field weighted score. Stub — TBD."""
    raise NotImplementedError("stub")


def multi_field_weighting(items: Sequence[Any], *, weights: dict[str, float]) -> list[Any]:
    """Multi-field weighted score. Stub — TBD."""
    raise NotImplementedError("stub")
