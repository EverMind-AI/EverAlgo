"""Tests for everalgo.types.memories.Episode."""

from typing import Any

import pytest
from pydantic import ValidationError

from everalgo.types import Episode


def _kwargs(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "owner_id": "u1",
        "episode": "Alice asked about Q3 plan.",
        "summary": "Alice asked about the Q3 plan.",
        "timestamp": 1700000000000,
    }
    base.update(overrides)
    return base


def test_episode_minimum_required_fields() -> None:
    ep = Episode(**_kwargs())
    assert ep.owner_id == "u1"
    assert ep.episode == "Alice asked about Q3 plan."
    assert ep.timestamp == 1700000000000


def test_episode_owner_id_required() -> None:
    with pytest.raises(ValidationError):
        Episode(  # type: ignore[call-arg]
            episode="text",
            timestamp=1,
        )


def test_episode_episode_field_required() -> None:
    with pytest.raises(ValidationError):
        Episode(  # type: ignore[call-arg]
            owner_id="u1",
            timestamp=1,
        )


def test_episode_summary_field_required() -> None:
    """Required rather than defaulted so a construction site that forgets it fails the type-checkers.

    A default would let the field silently regress to the blank preview that motivated declaring it.
    """
    with pytest.raises(ValidationError):
        Episode(  # type: ignore[call-arg]
            owner_id="u1",
            episode="text",
            timestamp=1,
        )


def test_episode_summary_is_declared_not_an_extra() -> None:
    """It used to arrive through ``extra="allow"``; being declared is what makes it part of the contract."""
    assert "summary" in Episode.model_fields


def test_episode_extra_fields_kept_accessible() -> None:
    """Episode uses ``extra='allow'`` so LLM-emitted secondary fields stay reachable."""
    ep = Episode.model_validate(
        _kwargs(
            subject="Alice",
            summary="short",
            keywords=["q3", "plan"],
            location="meeting room",
        )
    )
    assert ep.subject == "Alice"  # type: ignore[attr-defined]
    assert ep.summary == "short"
    assert ep.keywords == ["q3", "plan"]  # type: ignore[attr-defined]
    assert ep.location == "meeting room"  # type: ignore[attr-defined]


def test_episode_json_round_trip_preserves_extras() -> None:
    ep = Episode.model_validate(_kwargs(summary="s", keywords=["a"]))
    serialised = ep.model_dump_json()
    rebuilt = Episode.model_validate_json(serialised)
    assert rebuilt == ep
    assert rebuilt.summary == "s"
    assert rebuilt.keywords == ["a"]  # type: ignore[attr-defined]
