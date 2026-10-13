"""Tests for the ``everalgo.testing`` public API."""

import everalgo.testing
from everalgo.llm import StrictOutputModel


def test_public_symbols_exposed_at_top_level() -> None:
    assert hasattr(everalgo.testing, "FakeLLMClient")
    assert hasattr(everalgo.testing, "CallRecord")
    assert hasattr(everalgo.testing, "assert_episode_shape")
    assert hasattr(everalgo.testing, "assert_foresight_shape")
    assert hasattr(everalgo.testing, "assert_atomic_fact_shape")
    assert hasattr(everalgo.testing, "assert_profile_shape")
    assert hasattr(everalgo.testing, "LLMClientTestHarness")
    assert hasattr(everalgo.testing, "EffectiveRequest")
    assert hasattr(everalgo.testing, "UpstreamStub")
    assert hasattr(everalgo.testing, "NOT_SENT")
    assert hasattr(everalgo.testing, "REQUIRED_CHECKS")
    assert hasattr(everalgo.testing, "OPTIONAL_CHECKS")


def test_dunder_all_lists_exact_public_surface() -> None:
    assert sorted(everalgo.testing.__all__) == sorted(
        [
            "CallRecord",
            "EffectiveRequest",
            "FakeLLMClient",
            "LLMClientTestHarness",
            "NOT_SENT",
            "OPTIONAL_CHECKS",
            "REQUIRED_CHECKS",
            "UpstreamStub",
            "assert_atomic_fact_shape",
            "assert_episode_shape",
            "assert_foresight_shape",
            "assert_profile_shape",
        ]
    )


def test_top_level_import_works() -> None:
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
    record = CallRecord(messages=[], response_format=StrictOutputModel)
    assert record.messages == []
    assert callable(assert_episode_shape)
    assert callable(assert_foresight_shape)
    assert callable(assert_atomic_fact_shape)
    assert callable(assert_profile_shape)
