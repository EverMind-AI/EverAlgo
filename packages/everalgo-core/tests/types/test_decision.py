"""Tests for everalgo.types.memories.Decision and Principle."""

from typing import Any

import pytest
from pydantic import ValidationError

from everalgo.types import Decision, Principle

_ENGINEERING_FIELDS = ("session_id", "parent_id")


def _decision_kwargs(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "owner_id": "u1",
        "title": "Agent runtime language",
        "decision": "Use Python for the core Agent runtime.",
        "reason": "Faster iteration on agent capability.",
        "timestamp": 1700000000000,
    }
    base.update(overrides)
    return base


def _principle_kwargs(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "owner_id": "u1",
        "title": "Iteration over premature optimisation",
        "statement": "Agent architecture prioritises iteration speed.",
        "timestamp": 1700000000000,
    }
    base.update(overrides)
    return base


def test_decision_minimum_required_fields() -> None:
    dc = Decision(**_decision_kwargs())
    assert dc.owner_id == "u1"
    assert dc.title == "Agent runtime language"
    assert dc.decision == "Use Python for the core Agent runtime."
    assert dc.reason == "Faster iteration on agent capability."
    assert dc.impact is None
    assert dc.tags == []
    assert dc.timestamp == 1700000000000


def test_decision_owner_id_none_is_generic_path() -> None:
    dc = Decision(**_decision_kwargs(owner_id=None))
    assert dc.owner_id is None


def test_decision_missing_owner_id_raises() -> None:
    with pytest.raises(ValidationError):
        Decision(  # type: ignore[call-arg]
            title="t",
            decision="d",
            reason="r",
            timestamp=1,
        )


def test_decision_missing_required_text_fields_raise() -> None:
    with pytest.raises(ValidationError):
        Decision(  # type: ignore[call-arg]
            owner_id="u1",
            decision="d",
            reason="r",
            timestamp=1,
        )
    with pytest.raises(ValidationError):
        Decision(  # type: ignore[call-arg]
            owner_id="u1",
            title="t",
            reason="r",
            timestamp=1,
        )
    with pytest.raises(ValidationError):
        Decision(  # type: ignore[call-arg]
            owner_id="u1",
            title="t",
            decision="d",
            timestamp=1,
        )


def test_decision_has_no_everos_engineering_fields() -> None:
    for name in _ENGINEERING_FIELDS:
        assert name not in Decision.model_fields


def test_decision_tags_default_is_not_shared() -> None:
    a = Decision(**_decision_kwargs())
    b = Decision(**_decision_kwargs())
    a.tags.append("architecture")
    assert b.tags == []


def test_decision_extra_fields_kept_accessible() -> None:
    dc = Decision.model_validate(_decision_kwargs(confidence=0.9, topic="runtime"))
    assert dc.confidence == 0.9  # type: ignore[attr-defined]
    assert dc.topic == "runtime"  # type: ignore[attr-defined]


def test_decision_json_round_trip_preserves_extras() -> None:
    dc = Decision.model_validate(
        _decision_kwargs(impact="Keep device runtime in Rust.", tags=["runtime"], keywords=["py"])
    )
    rebuilt = Decision.model_validate_json(dc.model_dump_json())
    assert rebuilt == dc
    assert rebuilt.impact == "Keep device runtime in Rust."
    assert rebuilt.tags == ["runtime"]
    assert rebuilt.keywords == ["py"]  # type: ignore[attr-defined]


def test_principle_minimum_required_fields() -> None:
    pr = Principle(**_principle_kwargs())
    assert pr.owner_id == "u1"
    assert pr.title == "Iteration over premature optimisation"
    assert pr.statement == "Agent architecture prioritises iteration speed."
    assert pr.source_entry_ids == []
    assert pr.timestamp == 1700000000000


def test_principle_owner_id_required() -> None:
    with pytest.raises(ValidationError):
        Principle(  # type: ignore[call-arg]
            title="t",
            statement="s",
            timestamp=1,
        )


def test_principle_owner_id_none_rejected() -> None:
    with pytest.raises(ValidationError):
        Principle(**_principle_kwargs(owner_id=None))


def test_principle_has_no_everos_engineering_fields() -> None:
    for name in _ENGINEERING_FIELDS:
        assert name not in Principle.model_fields


def test_principle_source_entry_ids_default_is_not_shared() -> None:
    a = Principle(**_principle_kwargs())
    b = Principle(**_principle_kwargs())
    a.source_entry_ids.append("dc_20260824_0001")
    assert b.source_entry_ids == []


def test_principle_json_round_trip() -> None:
    pr = Principle.model_validate(_principle_kwargs(source_entry_ids=["dc_001"], confidence=0.8))
    rebuilt = Principle.model_validate_json(pr.model_dump_json())
    assert rebuilt == pr
    assert rebuilt.source_entry_ids == ["dc_001"]
    assert rebuilt.confidence == 0.8  # type: ignore[attr-defined]
