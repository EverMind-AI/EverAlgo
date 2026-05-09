"""Tests for evercore.testing package-level public API.

Verifies the 3 documented public symbols (per AGENTS.md §7 step 6 and §9
plus spec §3) are exported at the top-level package.
"""

import evercore.testing


def test_public_symbols_exposed_at_top_level() -> None:
    """3 public symbols accessible via attribute access on the package."""
    assert hasattr(evercore.testing, "FakeLLMClient")
    assert hasattr(evercore.testing, "CallRecord")
    assert hasattr(evercore.testing, "assert_episode_shape")


def test_dunder_all_lists_exactly_3_symbols() -> None:
    """__all__ enumerates the public surface — exactly 3 entries."""
    assert sorted(evercore.testing.__all__) == sorted(
        [
            "CallRecord",
            "FakeLLMClient",
            "assert_episode_shape",
        ]
    )


def test_top_level_import_works() -> None:
    """Star-friendly import from the package root."""
    from evercore.testing import (
        CallRecord,
        FakeLLMClient,
        assert_episode_shape,
    )

    # smoke-instantiate to verify they are importable, not just present
    client = FakeLLMClient(responses=["x"])
    assert client.call_count == 0
    record = CallRecord(messages=[])
    assert record.messages == []
    assert callable(assert_episode_shape)
