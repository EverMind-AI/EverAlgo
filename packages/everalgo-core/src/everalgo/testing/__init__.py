"""Public testing helpers for EverAlgo — assertions + fake_llm + judge.

Mirrors ``numpy.testing`` / ``torch.testing``: testing helpers live inside
``everalgo-core`` rather than as a separate distribution.

Public symbols (per AGENTS.md §7 step 6 + §9 + spec §3):

- ``FakeLLMClient`` — in-memory ``LLMClient`` Protocol implementation
- ``CallRecord``   — recorded chat() invocation type (for assertions)
- ``assert_episode_shape`` — Episode structural assertion helper
- ``assert_foresight_shape`` — Foresight structural assertion helper
- ``assert_atomic_fact_shape`` — AtomicFact structural assertion helper
- ``assert_profile_shape`` — Profile structural assertion helper
- ``JudgeResult``  — outcome of LLM-as-judge N-run majority vote
- ``allm_judge``   — async LLM-as-judge with parallel N-run majority vote
"""

import logging

from everalgo.testing.assertions import (
    assert_atomic_fact_shape,
    assert_episode_shape,
    assert_foresight_shape,
    assert_profile_shape,
)
from everalgo.testing.fake_llm import CallRecord, FakeLLMClient
from everalgo.testing.judge import JudgeResult, allm_judge

__all__ = [
    "CallRecord",
    "FakeLLMClient",
    "JudgeResult",
    "allm_judge",
    "assert_atomic_fact_shape",
    "assert_episode_shape",
    "assert_foresight_shape",
    "assert_profile_shape",
]

# Library logging setup (ADR-013): NullHandler on each subpackage logger.
logging.getLogger(__name__).addHandler(logging.NullHandler())
