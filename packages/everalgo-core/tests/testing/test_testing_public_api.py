"""Tests for everalgo.testing package-level public API.

Verifies the documented public symbols (per AGENTS.md §7 step 6 and §9 plus spec §3) are exported at the
top-level package.
"""

import everalgo.testing


def test_public_symbols_exposed_at_top_level() -> None:
    """Public symbols accessible via attribute access on the package."""
    assert hasattr(everalgo.testing, "FakeLLMClient")
    assert hasattr(everalgo.testing, "CallRecord")
    assert hasattr(everalgo.testing, "assert_episode_shape")
    assert hasattr(everalgo.testing, "assert_foresight_shape")
    assert hasattr(everalgo.testing, "assert_atomic_fact_shape")
    assert hasattr(everalgo.testing, "assert_profile_shape")


def test_dunder_all_lists_exact_public_surface() -> None:
    """__all__ enumerates the public surface — grows by one assert per memory type added."""
    assert sorted(everalgo.testing.__all__) == sorted(
        [
            "CallRecord",
            "FakeLLMClient",
            "assert_atomic_fact_shape",
            "assert_episode_shape",
            "assert_foresight_shape",
            "assert_profile_shape",
        ]
    )


def test_top_level_import_works() -> None:
    """Star-friendly import from the package root."""
    from everalgo.testing import (
        CallRecord,
        FakeLLMClient,
        assert_atomic_fact_shape,
        assert_episode_shape,
        assert_foresight_shape,
        assert_profile_shape,
    )

    client = FakeLLMClient(responses=["x"])
    assert client.call_count == 0
    record = CallRecord(messages=[])
    assert record.messages == []
    assert callable(assert_episode_shape)
    assert callable(assert_foresight_shape)
    assert callable(assert_atomic_fact_shape)
    assert callable(assert_profile_shape)
