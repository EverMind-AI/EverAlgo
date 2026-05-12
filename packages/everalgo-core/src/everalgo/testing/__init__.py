"""Public testing helpers for EverAlgo — assertions + fake_llm.

Mirrors ``numpy.testing`` / ``torch.testing`` (see ADR 005,
``docs/decisions/005-testing-as-public-subpackage.md``): testing helpers
live inside ``everalgo-core`` rather than as a separate distribution.

Public symbols (per AGENTS.md §7 step 6 + §9 + spec §3):

- ``FakeLLMClient`` — in-memory ``LLMClient`` Protocol implementation
- ``CallRecord``   — recorded chat() invocation type (for assertions)
- ``assert_episode_shape`` — Episode structural assertion helper
"""

import logging

from everalgo.testing.assertions import assert_episode_shape
from everalgo.testing.fake_llm import CallRecord, FakeLLMClient

__all__ = [
    "CallRecord",
    "FakeLLMClient",
    "assert_episode_shape",
]

# Library logging setup (ADR-013): NullHandler on each subpackage logger.
logging.getLogger(__name__).addHandler(logging.NullHandler())
