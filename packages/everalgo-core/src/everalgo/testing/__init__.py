"""Testing helpers for EverAlgo: shape assertions, ``FakeLLMClient`` and the ``LLMClient`` conformance suite."""

from __future__ import annotations

__all__ = [
    "NOT_SENT",
    "OPTIONAL_CHECKS",
    "REQUIRED_CHECKS",
    "CallRecord",
    "EffectiveRequest",
    "FakeLLMClient",
    "LLMClientTestHarness",
    "UpstreamStub",
    "assert_atomic_fact_shape",
    "assert_episode_shape",
    "assert_foresight_shape",
    "assert_profile_shape",
]


import logging

from everalgo.testing.assertions import (
    assert_atomic_fact_shape,
    assert_episode_shape,
    assert_foresight_shape,
    assert_profile_shape,
)
from everalgo.testing.conformance import (
    NOT_SENT,
    OPTIONAL_CHECKS,
    REQUIRED_CHECKS,
    EffectiveRequest,
    LLMClientTestHarness,
    UpstreamStub,
)
from everalgo.testing.fake_llm import CallRecord, FakeLLMClient

# Library logging setup (ADR-013): NullHandler on each subpackage logger.
logging.getLogger(__name__).addHandler(logging.NullHandler())
