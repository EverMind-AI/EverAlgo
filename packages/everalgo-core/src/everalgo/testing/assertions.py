"""Structural assertions for memory types."""

from __future__ import annotations

from typing import Any

from everalgo.types import Episode


def assert_episode_shape(value: dict[str, Any] | Episode) -> Episode:
    """Assert ``value`` satisfies ``Episode`` minimal business invariants.

    Combines pydantic type-level validation with 4 business invariants that
    pydantic alone does not catch (LLM may emit empty strings, zero
    timestamps, wrong ``parent_type``, etc.).

    Layered checks (in order):

    1. **Type level** — ``Episode.model_validate(value)`` parses dict (or
       passes Episode through). Type errors raise ``ValidationError``
       unmodified so the caller sees the original pydantic message.
    2. **Business invariants** — 4 checks, each raising ``AssertionError``
       with the failing invariant name:

       a. ``episode`` is a non-empty string.
       b. ``timestamp > 0`` (Unix epoch ms; ``0`` or negative = bug).
       c. ``parent_type == "memcell"`` (EPISODE path only consumes MemCell).
       d. ``parent_id`` is a non-empty string (data lineage anchor).

    Args:
        value: ``dict`` (parsed via ``Episode.model_validate``) or already-
            parsed ``Episode``.

    Returns:
        The validated ``Episode`` instance, so callers can chain further
        assertions (e.g. ``ep = assert_episode_shape(d); assert "x" in
        ep.episode``).

    Raises:
        AssertionError: If any business invariant fails. The message names
            the failed invariant.
        pydantic.ValidationError: If type-level validation fails. Re-raised
            unmodified so the caller sees the original pydantic message.
    """
    episode = value if isinstance(value, Episode) else Episode.model_validate(value)
    assert episode.episode, "Episode.episode is empty"
    assert episode.timestamp > 0, f"Episode.timestamp must be positive (Unix epoch ms), got {episode.timestamp}"
    assert episode.parent_type == "memcell", (
        f"Episode.parent_type must be 'memcell' (EPISODE path), got {episode.parent_type!r}"
    )
    assert episode.parent_id, "Episode.parent_id is empty"
    return episode
