"""Structural assertions for memory types."""

from __future__ import annotations

from typing import Any

from everalgo.types import AtomicFact, Episode, Foresight, Profile


def assert_episode_shape(value: dict[str, Any] | Episode) -> Episode:
    """Assert ``value`` satisfies ``Episode`` minimal business invariants.

    Combines pydantic type-level validation with 4 business invariants that pydantic alone does not catch
    (LLM may emit empty strings, zero timestamps, wrong ``parent_type``, etc.).

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

    Parameters
    ----------
    value : dict[str, Any] or Episode
        ``dict`` (parsed via ``Episode.model_validate``) or already-parsed ``Episode``.

    Returns
    -------
    Episode
        The validated ``Episode`` instance, so callers can chain further assertions (e.g.
        ``ep = assert_episode_shape(d); assert "x" in ep.episode``).

    Raises
    ------
    AssertionError
        If any business invariant fails. The message names the failed invariant.
    pydantic.ValidationError
        If type-level validation fails. Re-raised unmodified so the caller sees the original pydantic message.
    """
    episode = value if isinstance(value, Episode) else Episode.model_validate(value)
    assert episode.episode, "Episode.episode is empty"
    assert episode.timestamp > 0, f"Episode.timestamp must be positive (Unix epoch ms), got {episode.timestamp}"
    assert episode.parent_type == "memcell", (
        f"Episode.parent_type must be 'memcell' (EPISODE path), got {episode.parent_type!r}"
    )
    assert episode.parent_id, "Episode.parent_id is empty"
    return episode


def assert_foresight_shape(value: dict[str, Any] | Foresight) -> Foresight:
    """Assert ``value`` satisfies :class:`Foresight` minimal business invariants.

    Mirrors :func:`assert_episode_shape`: pydantic type-level validation + 4 business invariants that pydantic
    alone does not catch (LLM may emit empty strings, zero timestamps, wrong ``parent_type``, etc.).

    Layered checks (in order):

    1. **Type level** — ``Foresight.model_validate(value)`` parses dict (or passes a Foresight through).
       Type errors raise :class:`ValidationError` unmodified.
    2. **Business invariants** — 4 checks, each raising :class:`AssertionError` with the failing invariant
       name:

       a. ``foresight`` is a non-empty string.
       b. ``timestamp > 0`` (Unix epoch ms; ``0`` or negative = bug).
       c. ``parent_type == "memcell"`` (boundary output is the only source).
       d. ``parent_id`` is a non-empty string (data lineage anchor).

    Parameters
    ----------
    value : dict[str, Any] or Foresight
        ``dict`` (parsed via ``Foresight.model_validate``) or already-parsed ``Foresight``.

    Returns
    -------
    Foresight
        The validated :class:`Foresight` instance for caller chaining.

    Raises
    ------
    AssertionError
        If any business invariant fails. The message names the failed invariant.
    pydantic.ValidationError
        If type-level validation fails. Re-raised unmodified.
    """
    foresight = value if isinstance(value, Foresight) else Foresight.model_validate(value)
    assert foresight.foresight, "Foresight.foresight is empty"
    assert foresight.timestamp > 0, f"Foresight.timestamp must be positive (Unix epoch ms), got {foresight.timestamp}"
    assert foresight.parent_type == "memcell", f"Foresight.parent_type must be 'memcell', got {foresight.parent_type!r}"
    assert foresight.parent_id, "Foresight.parent_id is empty"
    return foresight


def assert_atomic_fact_shape(value: dict[str, Any] | AtomicFact) -> AtomicFact:
    """Assert ``value`` satisfies :class:`AtomicFact` minimal business invariants.

    Mirrors :func:`assert_episode_shape` / :func:`assert_foresight_shape`: pydantic type-level validation + 4
    business invariants:

    a. ``fact`` is a non-empty string.
    b. ``timestamp > 0`` (Unix epoch ms; ``0`` or negative = bug).
    c. ``parent_type == "memcell"`` (boundary output is the only source).
    d. ``parent_id`` is a non-empty string (data lineage anchor).

    Parameters
    ----------
    value : dict[str, Any] or AtomicFact
        ``dict`` (parsed via ``AtomicFact.model_validate``) or already-parsed ``AtomicFact``.

    Returns
    -------
    AtomicFact
        The validated :class:`AtomicFact` instance for caller chaining.

    Raises
    ------
    AssertionError
        If any business invariant fails. The message names the failed invariant.
    pydantic.ValidationError
        If type-level validation fails. Re-raised unmodified.
    """
    fact = value if isinstance(value, AtomicFact) else AtomicFact.model_validate(value)
    assert fact.fact, "AtomicFact.fact is empty"
    assert fact.timestamp > 0, f"AtomicFact.timestamp must be positive (Unix epoch ms), got {fact.timestamp}"
    assert fact.parent_type == "memcell", f"AtomicFact.parent_type must be 'memcell', got {fact.parent_type!r}"
    assert fact.parent_id, "AtomicFact.parent_id is empty"
    return fact


def assert_profile_shape(value: dict[str, Any] | Profile) -> Profile:
    """Assert ``value`` satisfies :class:`Profile` minimal business invariants.

    Unlike the per-MemCell types (Episode / Foresight / AtomicFact), Profile is a **user-level aggregate** —
    no ``parent_id`` / ``parent_type`` checks. The 3 business invariants:

    a. ``summary`` is a non-empty string.
    b. ``timestamp > 0`` (Unix epoch ms; ``0`` or negative = bug).
    c. ``owner_id`` is a non-empty string (every profile must belong to a user).

    Parameters
    ----------
    value : dict[str, Any] or Profile
        ``dict`` (parsed via ``Profile.model_validate``) or already-parsed ``Profile``.

    Returns
    -------
    Profile
        The validated :class:`Profile` instance for caller chaining.

    Raises
    ------
    AssertionError
        If any business invariant fails. The message names the failed invariant.
    pydantic.ValidationError
        If type-level validation fails. Re-raised unmodified.
    """
    profile = value if isinstance(value, Profile) else Profile.model_validate(value)
    assert profile.summary, "Profile.summary is empty"
    assert profile.timestamp > 0, f"Profile.timestamp must be positive (Unix epoch ms), got {profile.timestamp}"
    assert profile.owner_id, "Profile.owner_id is empty"
    return profile
