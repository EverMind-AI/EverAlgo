"""Structural assertions for memory types."""

from __future__ import annotations

from typing import Any

from everalgo.types import AtomicFact, Episode, Foresight, Profile


def assert_episode_shape(value: dict[str, Any] | Episode) -> Episode:
    """Assert ``value`` satisfies ``Episode`` minimal business invariants and return the validated instance.

    Raises:
        AssertionError: If ``episode`` is empty or ``timestamp <= 0``.
        pydantic.ValidationError: If type-level validation fails.
    """
    episode = value if isinstance(value, Episode) else Episode.model_validate(value)
    assert episode.episode, "Episode.episode is empty"
    assert episode.timestamp > 0, f"Episode.timestamp must be positive (Unix epoch ms), got {episode.timestamp}"
    return episode


def assert_foresight_shape(value: dict[str, Any] | Foresight) -> Foresight:
    """Assert ``value`` satisfies :class:`Foresight` minimal business invariants and return the validated instance.

    Raises:
        AssertionError: If ``foresight`` is empty or ``timestamp <= 0``.
        pydantic.ValidationError: If type-level validation fails.
    """
    foresight = value if isinstance(value, Foresight) else Foresight.model_validate(value)
    assert foresight.foresight, "Foresight.foresight is empty"
    assert foresight.timestamp > 0, f"Foresight.timestamp must be positive (Unix epoch ms), got {foresight.timestamp}"
    return foresight


def assert_atomic_fact_shape(value: dict[str, Any] | AtomicFact) -> AtomicFact:
    """Assert ``value`` satisfies :class:`AtomicFact` minimal business invariants and return the validated instance.

    Raises:
        AssertionError: If ``content`` is empty or ``timestamp <= 0``.
        pydantic.ValidationError: If type-level validation fails.
    """
    fact = value if isinstance(value, AtomicFact) else AtomicFact.model_validate(value)
    assert fact.content, "AtomicFact.content is empty"
    assert fact.timestamp > 0, f"AtomicFact.timestamp must be positive (Unix epoch ms), got {fact.timestamp}"
    return fact


def assert_profile_shape(value: dict[str, Any] | Profile) -> Profile:
    """Assert ``value`` satisfies :class:`Profile` minimal business invariants and return the validated instance.

    Raises:
        AssertionError: If ``summary`` or ``owner_id`` is empty, or ``timestamp <= 0``.
        pydantic.ValidationError: If type-level validation fails.
    """
    profile = value if isinstance(value, Profile) else Profile.model_validate(value)
    assert profile.summary, "Profile.summary is empty"
    assert profile.timestamp > 0, f"Profile.timestamp must be positive (Unix epoch ms), got {profile.timestamp}"
    assert profile.owner_id, "Profile.owner_id is empty"
    return profile
