"""Tests for everalgo.testing.assertions — assert_X_shape suite."""

from typing import Any

import pytest
from pydantic import ValidationError

from everalgo.testing.assertions import (
    assert_atomic_fact_shape,
    assert_episode_shape,
    assert_foresight_shape,
    assert_profile_shape,
)
from everalgo.types import AtomicFact, Episode, Foresight, Profile

# ---- happy path ----------------------------------------------------------


def _valid_episode_dict() -> dict[str, Any]:
    """Return a fresh copy of a minimal valid Episode dict."""
    return {
        "id": "ep_001",
        "owner_id": "u1",
        "episode": "Alice scheduled the meeting",
        "timestamp": 1700000000000,
        "parent_id": "mc_001",
    }


def test_dict_input_parsed_and_validated() -> None:
    """Valid dict returns the parsed Episode instance."""
    episode = assert_episode_shape(_valid_episode_dict())
    assert isinstance(episode, Episode)
    assert episode.episode == "Alice scheduled the meeting"
    assert episode.parent_type == "memcell"  # default applied by pydantic


def test_episode_input_passed_through_same_instance() -> None:
    """Passing an Episode instance returns the same object (is, not eq)."""
    original = Episode(**_valid_episode_dict())
    returned = assert_episode_shape(original)
    assert returned is original


def test_chained_assertion_uses_returned_episode() -> None:
    """Caller can chain further assertions on the return value."""
    episode = assert_episode_shape(_valid_episode_dict())
    assert "Alice" in episode.episode


# ---- pydantic ValidationError re-raised unmodified ------------------------


def test_missing_required_field_raises_validation_error() -> None:
    """Type-level errors surface as pydantic ValidationError, not AssertionError."""
    bad = _valid_episode_dict()
    del bad["parent_id"]
    with pytest.raises(ValidationError):
        assert_episode_shape(bad)


# ---- 4 business invariants ------------------------------------------------


def test_empty_episode_string_raises_assertion_error() -> None:
    """Episode.episode must be non-empty."""
    bad = _valid_episode_dict()
    bad["episode"] = ""
    with pytest.raises(AssertionError, match=r"Episode\.episode is empty"):
        assert_episode_shape(bad)


def test_zero_timestamp_raises_assertion_error() -> None:
    """Episode.timestamp must be positive (> 0)."""
    bad = _valid_episode_dict()
    bad["timestamp"] = 0
    with pytest.raises(AssertionError, match="must be positive"):
        assert_episode_shape(bad)


def test_negative_timestamp_raises_assertion_error() -> None:
    """Episode.timestamp must be positive (> 0), not negative."""
    bad = _valid_episode_dict()
    bad["timestamp"] = -1
    with pytest.raises(AssertionError, match="must be positive"):
        assert_episode_shape(bad)


def test_wrong_parent_type_raises_assertion_error() -> None:
    """Episode.parent_type must be 'memcell' (EPISODE path only)."""
    bad = _valid_episode_dict()
    bad["parent_type"] = "raw_message"
    with pytest.raises(AssertionError, match="must be 'memcell'"):
        assert_episode_shape(bad)


def test_empty_parent_id_raises_assertion_error() -> None:
    """Episode.parent_id must be non-empty."""
    bad = _valid_episode_dict()
    bad["parent_id"] = ""
    with pytest.raises(AssertionError, match=r"Episode\.parent_id is empty"):
        assert_episode_shape(bad)


# ==========================================================================
# assert_foresight_shape — mirrors the episode test suite
# ==========================================================================


def _valid_foresight_dict() -> dict[str, Any]:
    """Return a fresh copy of a minimal valid Foresight dict."""
    return {
        "id": "fs_001",
        "owner_id": "u1",
        "foresight": "Alice will send the draft by Friday",
        "evidence": "I'll send Alice the draft by Friday",
        "timestamp": 1700000000000,
        "parent_id": "mc_001",
    }


def test_foresight_dict_input_parsed_and_validated() -> None:
    """Valid dict returns the parsed Foresight instance."""
    foresight = assert_foresight_shape(_valid_foresight_dict())
    assert isinstance(foresight, Foresight)
    assert foresight.foresight == "Alice will send the draft by Friday"
    assert foresight.parent_type == "memcell"  # default applied by pydantic


def test_foresight_input_passed_through_same_instance() -> None:
    """Passing a Foresight instance returns the same object (is, not eq)."""
    original = Foresight(**_valid_foresight_dict())
    returned = assert_foresight_shape(original)
    assert returned is original


def test_foresight_missing_required_field_raises_validation_error() -> None:
    """Type-level errors surface as pydantic ValidationError, not AssertionError."""
    bad = _valid_foresight_dict()
    del bad["parent_id"]
    with pytest.raises(ValidationError):
        assert_foresight_shape(bad)


def test_empty_foresight_string_raises_assertion_error() -> None:
    """Foresight.foresight must be non-empty."""
    bad = _valid_foresight_dict()
    bad["foresight"] = ""
    with pytest.raises(AssertionError, match=r"Foresight\.foresight is empty"):
        assert_foresight_shape(bad)


def test_foresight_zero_timestamp_raises_assertion_error() -> None:
    """Foresight.timestamp must be positive (> 0)."""
    bad = _valid_foresight_dict()
    bad["timestamp"] = 0
    with pytest.raises(AssertionError, match="must be positive"):
        assert_foresight_shape(bad)


def test_foresight_wrong_parent_type_raises_assertion_error() -> None:
    """Foresight.parent_type must be 'memcell'."""
    bad = _valid_foresight_dict()
    bad["parent_type"] = "raw_message"
    with pytest.raises(AssertionError, match="must be 'memcell'"):
        assert_foresight_shape(bad)


def test_foresight_empty_parent_id_raises_assertion_error() -> None:
    """Foresight.parent_id must be non-empty."""
    bad = _valid_foresight_dict()
    bad["parent_id"] = ""
    with pytest.raises(AssertionError, match=r"Foresight\.parent_id is empty"):
        assert_foresight_shape(bad)


# ==========================================================================
# assert_atomic_fact_shape — mirrors the episode test suite
# ==========================================================================


def _valid_atomic_fact_dict() -> dict[str, Any]:
    """Return a fresh copy of a minimal valid AtomicFact dict."""
    return {
        "id": "af_001",
        "owner_id": "u1",
        "fact": "Alice scheduled a 3pm meeting with Bob on 2024-03-14",
        "timestamp": 1700000000000,
        "parent_id": "mc_001",
    }


def test_atomic_fact_dict_input_parsed_and_validated() -> None:
    """Valid dict returns the parsed AtomicFact instance."""
    fact = assert_atomic_fact_shape(_valid_atomic_fact_dict())
    assert isinstance(fact, AtomicFact)
    assert "Alice" in fact.fact
    assert fact.parent_type == "memcell"  # default applied by pydantic


def test_atomic_fact_input_passed_through_same_instance() -> None:
    """Passing an AtomicFact instance returns the same object (is, not eq)."""
    original = AtomicFact(**_valid_atomic_fact_dict())
    returned = assert_atomic_fact_shape(original)
    assert returned is original


def test_atomic_fact_missing_required_field_raises_validation_error() -> None:
    """Type-level errors surface as pydantic ValidationError, not AssertionError."""
    bad = _valid_atomic_fact_dict()
    del bad["parent_id"]
    with pytest.raises(ValidationError):
        assert_atomic_fact_shape(bad)


def test_empty_atomic_fact_string_raises_assertion_error() -> None:
    """AtomicFact.fact must be non-empty."""
    bad = _valid_atomic_fact_dict()
    bad["fact"] = ""
    with pytest.raises(AssertionError, match=r"AtomicFact\.fact is empty"):
        assert_atomic_fact_shape(bad)


def test_atomic_fact_zero_timestamp_raises_assertion_error() -> None:
    """AtomicFact.timestamp must be positive (> 0)."""
    bad = _valid_atomic_fact_dict()
    bad["timestamp"] = 0
    with pytest.raises(AssertionError, match="must be positive"):
        assert_atomic_fact_shape(bad)


def test_atomic_fact_wrong_parent_type_raises_assertion_error() -> None:
    """AtomicFact.parent_type must be 'memcell'."""
    bad = _valid_atomic_fact_dict()
    bad["parent_type"] = "raw_message"
    with pytest.raises(AssertionError, match="must be 'memcell'"):
        assert_atomic_fact_shape(bad)


def test_atomic_fact_empty_parent_id_raises_assertion_error() -> None:
    """AtomicFact.parent_id must be non-empty."""
    bad = _valid_atomic_fact_dict()
    bad["parent_id"] = ""
    with pytest.raises(AssertionError, match=r"AtomicFact\.parent_id is empty"):
        assert_atomic_fact_shape(bad)


# ==========================================================================
# assert_profile_shape — user-level aggregate (no parent_id / parent_type)
# ==========================================================================


def _valid_profile_dict() -> dict[str, Any]:
    """Return a fresh copy of a minimal valid Profile dict."""
    return {
        "id": "pf_001",
        "owner_id": "u1",
        "summary": "Alice is a Python developer who prefers ruff for linting.",
        "timestamp": 1700000000000,
    }


def test_profile_dict_input_parsed_and_validated() -> None:
    """Valid dict returns the parsed Profile instance."""
    profile = assert_profile_shape(_valid_profile_dict())
    assert isinstance(profile, Profile)
    assert "Python" in profile.summary


def test_profile_input_passed_through_same_instance() -> None:
    """Passing a Profile instance returns the same object (is, not eq)."""
    original = Profile(**_valid_profile_dict())
    returned = assert_profile_shape(original)
    assert returned is original


def test_profile_missing_required_field_raises_validation_error() -> None:
    """Type-level errors surface as pydantic ValidationError, not AssertionError."""
    bad = _valid_profile_dict()
    del bad["owner_id"]
    with pytest.raises(ValidationError):
        assert_profile_shape(bad)


def test_empty_profile_summary_raises_assertion_error() -> None:
    """Profile.summary must be non-empty."""
    bad = _valid_profile_dict()
    bad["summary"] = ""
    with pytest.raises(AssertionError, match=r"Profile\.summary is empty"):
        assert_profile_shape(bad)


def test_profile_zero_timestamp_raises_assertion_error() -> None:
    """Profile.timestamp must be positive (> 0)."""
    bad = _valid_profile_dict()
    bad["timestamp"] = 0
    with pytest.raises(AssertionError, match="must be positive"):
        assert_profile_shape(bad)


def test_profile_empty_owner_id_raises_assertion_error() -> None:
    """Profile.owner_id must be non-empty."""
    bad = _valid_profile_dict()
    bad["owner_id"] = ""
    with pytest.raises(AssertionError, match=r"Profile\.owner_id is empty"):
        assert_profile_shape(bad)
