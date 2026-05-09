"""Tests for evercore.testing.assertions — assert_episode_shape."""

from typing import Any

import pytest
from pydantic import ValidationError

from evercore.testing.assertions import assert_episode_shape
from evercore.types import Episode

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
    with pytest.raises(AssertionError, match="Episode.episode is empty"):
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
    with pytest.raises(AssertionError, match="Episode.parent_id is empty"):
        assert_episode_shape(bad)
