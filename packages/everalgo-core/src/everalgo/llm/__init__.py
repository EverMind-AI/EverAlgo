"""LLM facade — chat-style abstraction over OpenAI-compatible providers.

Public surface (7 symbols, alphabetical-by-category):

- protocol:  LLMClient
- data:      ChatMessage, ChatResponse, Usage, LLMConfig
- error:     LLMError
- factory:   build_client
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager

from everalgo.llm.config import LLMConfig
from everalgo.llm.errors import LLMError, LLMNotConfiguredError
from everalgo.llm.factory import build_client
from everalgo.llm.protocols import LLMClient
from everalgo.llm.types import ChatMessage, ChatResponse, Usage

__all__ = [
    # sub-project 2 (LLM Stack — 7)
    "LLMClient",
    "ChatMessage",
    "ChatResponse",
    "Usage",
    "LLMConfig",
    "LLMError",
    "build_client",
    # sub-project 2.5 (3-layer injection — 5)
    "LLMNotConfiguredError",
    "configure",
    "current",
    "resolve",
    "use",
]


# Sub-project 2.5: 3-layer LLM injection (configure / use / current / resolve)

_default: LLMClient | None = None
"""Set-once global default LLM. Mutated only by configure().

Module-private (underscore prefix) — tests may monkey-patch via
``everalgo.llm._default = ...`` for isolation, but this is not part of
the documented public API.
"""

_active: contextvars.ContextVar[LLMClient | None] = contextvars.ContextVar(
    "everalgo_llm_active",
    default=None,
)
"""Scoped (per-asyncio-Task / per-thread) LLM override. Mutated only by use().

ContextVar — async-safe (asyncio Task auto-propagation) + thread-safe
(per-thread isolation). Mirrors DSPy ``thread_local_overrides`` ContextVar
in ``dspy/dsp/utils/settings.py:48``.
"""


def configure(llm: LLMClient) -> None:
    """Set the process-wide default LLM client (set-once semantics).

    Once configured, the default persists for the process lifetime. There is
    no reset mechanism — for testing isolation, pass ``llm=`` per-call to the
    operator (operators accept ``llm: LLMClient | None = None``); for multi-
    client switching, use the ``use(client)`` scoped contextmanager.

    Args:
        llm: An ``LLMClient`` instance. Required (no default value, ``None``
            not accepted by static type checking).
    """
    global _default
    _default = llm


@contextmanager
def use(client: LLMClient) -> Iterator[None]:
    """Temporarily override the active LLM within a sync ``with`` block.

    Sync ``@contextmanager`` (NOT ``@asynccontextmanager``) is the correct
    form here: the underlying ``ContextVar.set / reset`` operations are sync,
    and Python's asyncio.Task auto-propagates ContextVar state across
    ``await`` boundaries. Hence ``with use(client):`` works correctly inside
    ``async def`` functions, FastAPI endpoints, and Jupyter cells alike — no
    ``async with`` needed.

    Mirrors DSPy ``dspy.settings.context(lm=...)`` (sync ``@contextmanager``
    in ``dspy/dsp/utils/settings.py:216``) and pydantic-ai ``agent.override
    (model=...)`` (sync ``@contextmanager`` per-Agent ContextVar).

    Nested ``use()`` calls naturally stack (the inner block's reset token
    restores the outer block's value, not the global default).

    Args:
        client: The ``LLMClient`` to bind for the duration of the ``with``
            block.

    Yields:
        Control to the ``with`` block body. ``current()`` and ``resolve()``
        within the block return ``client``.
    """
    token = _active.set(client)
    try:
        yield
    finally:
        _active.reset(token)


def current() -> LLMClient | None:
    """Read-only query: the currently active LLM (scoped > default).

    Resolution order (no per-call layer here — that's ``resolve()``'s job):
    1. ``_active.get()`` — scoped contextvar (set by ``use(...)``)
    2. ``_default`` — process-wide default (set by ``configure(...)``)

    Returns ``None`` if neither has been set. This is a legitimate return
    value (NOT an error) — callers needing fail-fast on missing config use
    ``resolve()`` instead.

    Returns:
        The active ``LLMClient`` or ``None``.
    """
    return _active.get() or _default


def resolve(per_call: LLMClient | None = None) -> LLMClient:
    """3-layer fallback resolution: per_call > scoped > default.

    Single-line helper used inside operator implementations to avoid
    repeating the resolution boilerplate. Mirrors DSPy's ``Settings.lm``
    auto-fallback (``dspy/dsp/utils/settings.py:78``).

    Resolution order:
    1. ``per_call`` argument (function-call layer, highest priority)
    2. ``_active.get()`` (scoped contextvar)
    3. ``_default`` (global)

    Args:
        per_call: Per-call override passed by the operator's caller.
            Typical signature: ``async def aextract(memcell, *, llm=None)``,
            then internally ``client = everalgo.llm.resolve(llm)``.

    Returns:
        The resolved ``LLMClient``.

    Raises:
        LLMNotConfiguredError: If all 3 layers are ``None`` (developer forgot
            to inject). Message names the 3 fix paths (configure / use /
            per-call).
    """
    if per_call is not None:
        return per_call
    client = current()
    if client is None:
        raise LLMNotConfiguredError(
            "No LLM configured. Pass `llm=client` per-call, "
            "wrap in `everalgo.llm.use(client)`, "
            "or call `everalgo.configure(llm=client)` at startup."
        )
    return client
