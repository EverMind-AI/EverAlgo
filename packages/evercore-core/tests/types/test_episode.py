"""Tests for evercore.types.memories.Episode."""

from typing import Any

import pytest
from pydantic import ValidationError

from evercore.types import Episode


def _kwargs(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "ep1",
        "owner_id": "u1",
        "episode": "Alice asked about Q3 plan.",
        "timestamp": 1700000000000,
        "parent_id": "m1",
    }
    base.update(overrides)
    return base


def test_episode_minimum_required_fields() -> None:
    ep = Episode(**_kwargs())
    assert ep.id == "ep1"
    assert ep.owner_id == "u1"
    assert ep.episode == "Alice asked about Q3 plan."
    assert ep.timestamp == 1700000000000
    assert ep.parent_id == "m1"


def test_episode_parent_type_default_is_memcell() -> None:
    ep = Episode(**_kwargs())
    assert ep.parent_type == "memcell"


def test_episode_parent_type_overridable() -> None:
    ep = Episode(**_kwargs(parent_type="episode"))
    assert ep.parent_type == "episode"


def test_episode_owner_id_required() -> None:
    with pytest.raises(ValidationError):
        Episode(  # type: ignore[call-arg]
            id="ep1",
            episode="text",
            timestamp=1,
            parent_id="m1",
        )


def test_episode_episode_field_required() -> None:
    with pytest.raises(ValidationError):
        Episode(  # type: ignore[call-arg]
            id="ep1",
            owner_id="u1",
            timestamp=1,
            parent_id="m1",
        )


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
    assert ep.summary == "short"  # type: ignore[attr-defined]
    assert ep.keywords == ["q3", "plan"]  # type: ignore[attr-defined]
    assert ep.location == "meeting room"  # type: ignore[attr-defined]


def test_episode_json_round_trip_preserves_extras() -> None:
    ep = Episode.model_validate(_kwargs(summary="s", keywords=["a"]))
    serialised = ep.model_dump_json()
    rebuilt = Episode.model_validate_json(serialised)
    assert rebuilt == ep
    assert rebuilt.summary == "s"  # type: ignore[attr-defined]
    assert rebuilt.keywords == ["a"]  # type: ignore[attr-defined]
