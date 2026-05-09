"""Weighted score tools — pure compute. Stubs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

__all__ = ["weighted_score", "multi_field_weighting"]


def weighted_score(items: Sequence[Any], *, fields: dict[str, float]) -> list[Any]:
    """Single field weighted score. Stub — TBD."""
    raise NotImplementedError("stub")


def multi_field_weighting(items: Sequence[Any], *, weights: dict[str, float]) -> list[Any]:
    """Multi-field weighted score. Stub — TBD."""
    raise NotImplementedError("stub")
