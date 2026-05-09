"""Tests for evercore.llm 3-layer injection (configure / use / current / resolve).

Each test gets isolated _default + _active state via the autouse fixture
(directly mutating module-private variables — see spec §6.4 for the
rationale: BOSS rejected exposing reset_default() public API; tests use
monkeypatch-style isolation instead).
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

import evercore.llm
from evercore.llm.protocols import LLMClient
from evercore.testing.fake_llm import FakeLLMClient


@pytest.fixture(autouse=True)
def reset_evercore_llm_state() -> Iterator[None]:
    """Reset _default + _active before each test; restore after.

    _default is a plain module variable — save/restore directly.
    _active is a ContextVar — use set/reset token semantics.
    """
    saved_default = evercore.llm._default
    token = evercore.llm._active.set(None)
    try:
        evercore.llm._default = None
        yield
    finally:
        evercore.llm._default = saved_default
        evercore.llm._active.reset(token)


# ---- configure() (Task 2) -------------------------------------------------


def _make_fake() -> LLMClient:
    """Helper: return a FakeLLMClient with no scripted responses (we won't call .chat())."""
    return FakeLLMClient(responses=[])


def test_configure_sets_module_default() -> None:
    """configure(c) sets the module-private _default to c."""
    client = _make_fake()
    evercore.llm.configure(llm=client)
    assert evercore.llm._default is client


def test_configure_overwrites_previous_default() -> None:
    """Repeated configure() overwrites the prior default (last-write-wins)."""
    c1 = _make_fake()
    c2 = _make_fake()
    evercore.llm.configure(llm=c1)
    evercore.llm.configure(llm=c2)
    assert evercore.llm._default is c2


# ---- use() (Task 3) -------------------------------------------------------


def test_use_sets_active_inside_block() -> None:
    """Inside `with use(c):` the _active ContextVar holds c."""
    client = _make_fake()
    with evercore.llm.use(client):
        assert evercore.llm._active.get() is client


def test_use_resets_after_block_exits() -> None:
    """After `with use(c):` exits, _active is restored to None (the prior value)."""
    client = _make_fake()
    with evercore.llm.use(client):
        pass
    assert evercore.llm._active.get() is None


def test_use_can_nest() -> None:
    """Nested use() stacks: inner block sees inner client; outer restored after inner exits."""
    c1 = _make_fake()
    c2 = _make_fake()
    with evercore.llm.use(c1):
        assert evercore.llm._active.get() is c1
        with evercore.llm.use(c2):
            assert evercore.llm._active.get() is c2
        assert evercore.llm._active.get() is c1


async def test_use_works_inside_async_def() -> None:
    """ContextVar auto-propagates inside asyncio Task — sync `with` in async def works."""
    client = _make_fake()
    with evercore.llm.use(client):
        assert evercore.llm._active.get() is client
        # await something to force a yield point — _active must still hold client
        import asyncio

        await asyncio.sleep(0)
        assert evercore.llm._active.get() is client


# ---- current() (Task 4) ---------------------------------------------------


def test_current_returns_none_when_nothing_set() -> None:
    """With no configure() and no use(), current() returns None."""
    assert evercore.llm.current() is None


def test_current_returns_default_when_only_configured() -> None:
    """configure(c) sets default; current() returns c."""
    client = _make_fake()
    evercore.llm.configure(llm=client)
    assert evercore.llm.current() is client


def test_current_returns_scoped_over_default() -> None:
    """When both layers set, scoped (use) wins over default (configure)."""
    c_default = _make_fake()
    c_scoped = _make_fake()
    evercore.llm.configure(llm=c_default)
    with evercore.llm.use(c_scoped):
        assert evercore.llm.current() is c_scoped
    # After exiting use(), default is back
    assert evercore.llm.current() is c_default


# ---- resolve() (Task 5) ---------------------------------------------------


def test_resolve_per_call_takes_priority() -> None:
    """per_call argument wins over scoped + default."""
    c_default = _make_fake()
    c_scoped = _make_fake()
    c_per_call = _make_fake()
    evercore.llm.configure(llm=c_default)
    with evercore.llm.use(c_scoped):
        assert evercore.llm.resolve(c_per_call) is c_per_call


def test_resolve_falls_back_to_scoped() -> None:
    """When per_call=None, scoped wins over default."""
    c_default = _make_fake()
    c_scoped = _make_fake()
    evercore.llm.configure(llm=c_default)
    with evercore.llm.use(c_scoped):
        assert evercore.llm.resolve(None) is c_scoped


def test_resolve_falls_back_to_default() -> None:
    """When per_call=None and no scoped, default wins."""
    c_default = _make_fake()
    evercore.llm.configure(llm=c_default)
    assert evercore.llm.resolve(None) is c_default


def test_resolve_raises_when_all_layers_none() -> None:
    """All 3 layers None → LLMNotConfiguredError."""
    from evercore.llm.errors import LLMNotConfiguredError

    with pytest.raises(LLMNotConfiguredError, match="No LLM configured"):
        evercore.llm.resolve(None)


def test_resolve_error_message_lists_three_fix_paths() -> None:
    """Error message names all 3 fix paths: configure / use / per-call."""
    from evercore.llm.errors import LLMNotConfiguredError

    with pytest.raises(LLMNotConfiguredError) as excinfo:
        evercore.llm.resolve(None)
    msg = str(excinfo.value)
    assert "configure" in msg
    assert "use" in msg
    # per-call hint phrased as `llm=client`
    assert "llm=client" in msg
