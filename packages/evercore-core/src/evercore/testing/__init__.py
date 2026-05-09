"""Public testing helpers for EverCore — assertions + fake_llm.

Mirrors ``numpy.testing`` / ``torch.testing`` (see ADR 005,
``docs/decisions/005-testing-as-public-subpackage.md``): testing helpers
live inside ``evercore-core`` rather than as a separate distribution.

Public symbols (per AGENTS.md §7 step 6 + §9 + spec §3):

- ``FakeLLMClient`` — in-memory ``LLMClient`` Protocol implementation
- ``CallRecord``   — recorded chat() invocation type (for assertions)
- ``assert_episode_shape`` — Episode structural assertion helper
"""

from evercore.testing.assertions import assert_episode_shape
from evercore.testing.fake_llm import CallRecord, FakeLLMClient

__all__ = [
    "CallRecord",
    "FakeLLMClient",
    "assert_episode_shape",
]
