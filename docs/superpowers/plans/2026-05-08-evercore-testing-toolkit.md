# EverAlgo Testing Toolkit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `everalgo-core` 内交付 `everalgo.testing` 子包，含 `FakeLLMClient`（含 `CallRecord`）和 `assert_episode_shape` 三个公开符号，让算法同学单测时能不打真 LLM、对 EPISODE 输出做结构断言。

**Architecture:** 两个独立 helper 模块（`fake_llm.py` + `assertions.py`），无内部状态共享，无依赖耦合。`FakeLLMClient` 双模二选一（scripted list XOR callable handler），handler 兼容 sync / async；`assert_episode_shape` 在 pydantic 类型层之上加 4 项业务不变量，返回 `Episode` 实例支持链式断言。

**Tech Stack:** Python 3.12 / pydantic v2 / pytest（asyncio_mode=auto）/ ruff / mypy strict / 仅 stdlib `inspect` 增量，零新依赖。

---

## 关键约束（必读，避免子项目 1+2 的重复返工）

1. **测试函数必须 `-> None` 注解**（mypy strict 模式不会让 `tests.*` override 兜底；显式标注最稳）。
2. **`tests/testing/` 不要 `__init__.py`**（沿子项目 1+2 的 `--import-mode=importlib` 决策）。
3. **复用 `tests/conftest.py` 现有占位文件**（不新增 fixture；spec §7.1 的 4 个明星项目证据已 cite）。
4. **`everalgo.llm.types.ChatMessage`、`ChatResponse`、`everalgo.llm.protocols.LLMClient` 已在子项目 2 落地**（`packages/everalgo-core/src/everalgo/llm/`），本 plan 只 import 不修改。
5. **`everalgo.types.Episode` 已在子项目 1 落地**（`packages/everalgo-core/src/everalgo/types/memories.py`），本 plan 只 import 不修改。
6. **零新增依赖** — 不动 `pyproject.toml`。
7. **commit 风格 `<emoji> <type>(<scope>): <description>`**，scope 用 `testing`，参考子项目 1+2 commit 历史。
8. **每个 commit 落 `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`**（不再写）。

---

## File Structure

```
packages/everalgo-core/
├── src/everalgo/testing/
│   ├── __init__.py        # Task 7: re-export 3 公开符号
│   ├── assertions.py      # Task 6: assert_episode_shape
│   └── fake_llm.py        # Task 1+2+3+4+5: CallRecord + FakeLLMClient
└── tests/testing/
    ├── test_assertions.py # Task 6
    └── test_fake_llm.py   # Task 1+2+3+4+5
```

设计依据：见 `docs/superpowers/specs/2026-05-08-everalgo-testing-toolkit-design.md` §2 (File Map) 和 §3 (公开 API)。

---

## Task 0: tests/testing/ 目录占位 + 现有 testing/__init__.py 不动

**Files:**
- Create: `packages/everalgo-core/tests/testing/` 目录（无 `__init__.py`）
- Verify: `packages/everalgo-core/src/everalgo/testing/__init__.py` 占位已存在（子项目 0 留下）

- [ ] **Step 1: 验证现有 `everalgo.testing` 包占位**

Run:
```bash
ls -la packages/everalgo-core/src/everalgo/testing/
```

Expected: 看到 `__init__.py`（内容是子项目 0 留下的占位 docstring + `__all__: list[str] = []`）。

如果不存在，停止并 escalate（这违反子项目 0 的预期布局）。

- [ ] **Step 2: 创建空的 `tests/testing/` 目录**

Run:
```bash
mkdir -p packages/everalgo-core/tests/testing/
```

**Do NOT create `__init__.py` in `tests/testing/`** — 沿用 `--import-mode=importlib` 约定，与现有 `tests/llm/`、`tests/types/` 一致。

Verify:
```bash
ls -la packages/everalgo-core/tests/testing/
test ! -f packages/everalgo-core/tests/testing/__init__.py && echo "OK: no __init__.py"
```

- [ ] **Step 3: 不 commit（空目录无文件）** — Task 1 第一次落 test 文件时一并 commit。

---

## Task 1: `CallRecord` pydantic 类（公开符号 1/3）

**Files:**
- Create: `packages/everalgo-core/src/everalgo/testing/fake_llm.py`
- Test: `packages/everalgo-core/tests/testing/test_fake_llm.py`

- [ ] **Step 1: Write the failing tests**

Write to `packages/everalgo-core/tests/testing/test_fake_llm.py`:

```python
"""Tests for everalgo.testing.fake_llm — CallRecord."""

from everalgo.llm.types import ChatMessage
from everalgo.testing.fake_llm import CallRecord


def test_call_record_minimum_fields() -> None:
    """messages is the only required field."""
    record = CallRecord(messages=[ChatMessage(role="user", content="hi")])
    assert record.messages == [ChatMessage(role="user", content="hi")]
    assert record.model is None
    assert record.temperature is None
    assert record.max_tokens is None
    assert record.response_format is None
    assert record.extra == {}


def test_call_record_all_fields_populated() -> None:
    """All optional fields can be set explicitly."""
    record = CallRecord(
        messages=[ChatMessage(role="user", content="hi")],
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=128,
        response_format={"type": "json_object"},
        extra={"seed": 42},
    )
    assert record.model == "gpt-4o-mini"
    assert record.temperature == 0.7
    assert record.max_tokens == 128
    assert record.response_format == {"type": "json_object"}
    assert record.extra == {"seed": 42}


def test_call_record_extra_field_default_factory_is_independent() -> None:
    """Each instance must own its own extra dict (default_factory, not mutable default)."""
    a = CallRecord(messages=[ChatMessage(role="user", content="a")])
    b = CallRecord(messages=[ChatMessage(role="user", content="b")])
    a.extra["key"] = "value"
    assert b.extra == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
uv run pytest packages/everalgo-core/tests/testing/test_fake_llm.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'everalgo.testing.fake_llm'`.

- [ ] **Step 3: Write minimal implementation**

Write to `packages/everalgo-core/src/everalgo/testing/fake_llm.py`:

```python
"""In-memory LLMClient double for unit tests.

See ``docs/superpowers/specs/2026-05-08-everalgo-testing-toolkit-design.md``
for the design rationale (4 industry references in §7).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from everalgo.llm.types import ChatMessage


class CallRecord(BaseModel):
    """Single recorded ``FakeLLMClient.chat`` invocation.

    Exposed publicly so tests can assert on captured arguments via
    ``client.calls[0].messages == [...]`` with full IDE type-checking
    (mirroring ``unittest.mock.call`` being a public symbol).
    """

    messages: list[ChatMessage]
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    response_format: Mapping[str, Any] | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
uv run pytest packages/everalgo-core/tests/testing/test_fake_llm.py -v
```

Expected: 3 PASS.

- [ ] **Step 5: Run quality gates**

Run:
```bash
uv run ruff check packages/everalgo-core/src/everalgo/testing/ packages/everalgo-core/tests/testing/
uv run ruff format --check packages/everalgo-core/src/everalgo/testing/ packages/everalgo-core/tests/testing/
uv run mypy packages/everalgo-core/src/everalgo/testing/
```

Expected: 全部 clean。

- [ ] **Step 6: Commit**

```bash
git add packages/everalgo-core/src/everalgo/testing/fake_llm.py \
        packages/everalgo-core/tests/testing/test_fake_llm.py
git commit -m "$(cat <<'EOF'
✨ feat(testing): add CallRecord type for fake LLM client

CallRecord captures one chat() invocation (messages + kwargs) so tests
can assert on recorded calls with full pydantic type-checking. Mirrors
unittest.mock.call being a public symbol — see spec §3.1 for rationale.
EOF
)"
```

---

## Task 2: `FakeLLMClient` 构造器 + 互斥校验（不实现 `chat`）

**Files:**
- Modify: `packages/everalgo-core/src/everalgo/testing/fake_llm.py` (append `FakeLLMClient` class)
- Modify: `packages/everalgo-core/tests/testing/test_fake_llm.py` (append constructor tests)

- [ ] **Step 1: Append failing tests**

Append to `packages/everalgo-core/tests/testing/test_fake_llm.py`:

```python


# ---- FakeLLMClient construction (Task 2) ----------------------------------

import pytest

from everalgo.llm.types import ChatResponse
from everalgo.testing.fake_llm import FakeLLMClient


def test_fake_llm_client_constructed_with_responses_only() -> None:
    """Scripted list mode is one valid construction path."""
    client = FakeLLMClient(responses=["hello"])
    assert client.call_count == 0


def test_fake_llm_client_constructed_with_handler_only() -> None:
    """Callable handler mode is the other valid construction path."""

    def handler(
        messages: list[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        return ChatResponse(content="ok", model="fake")

    client = FakeLLMClient(handler=handler)
    assert client.call_count == 0


def test_fake_llm_client_both_responses_and_handler_raises() -> None:
    """Mutual exclusion: passing both is a ValueError."""

    def handler(
        messages: list[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        return ChatResponse(content="ok", model="fake")

    with pytest.raises(ValueError, match="exactly one of"):
        FakeLLMClient(responses=["hi"], handler=handler)


def test_fake_llm_client_neither_responses_nor_handler_raises() -> None:
    """Mutual exclusion: passing neither is a ValueError."""
    with pytest.raises(ValueError, match="exactly one of"):
        FakeLLMClient()


def test_fake_llm_client_responses_invalid_element_type_raises() -> None:
    """Each responses element must be str or ChatResponse."""
    with pytest.raises(TypeError, match="must contain str or ChatResponse"):
        FakeLLMClient(responses=[123])  # type: ignore[list-item]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
uv run pytest packages/everalgo-core/tests/testing/test_fake_llm.py -v
```

Expected: 5 new tests FAIL with `ImportError: cannot import name 'FakeLLMClient'`. Existing 3 CallRecord tests still PASS.

- [ ] **Step 3: Append minimal implementation**

Append to `packages/everalgo-core/src/everalgo/testing/fake_llm.py`:

```python


# ---- FakeLLMClient (Task 2: constructor only) ----------------------------

from collections.abc import Awaitable, Callable

from everalgo.llm.types import ChatResponse

_HandlerReturn = ChatResponse | Awaitable[ChatResponse]
_Handler = Callable[..., _HandlerReturn]


class FakeLLMClient:
    """In-memory ``LLMClient`` Protocol implementation for unit tests.

    Two construction modes (mutually exclusive):

    1. **Scripted list** — ``responses=[...]`` popped in call order;
       exhaustion raises ``RuntimeError``.
    2. **Callable handler** — ``handler=callable`` invoked per call;
       sync or async return both accepted.

    See ``docs/superpowers/specs/2026-05-08-everalgo-testing-toolkit-design.md``
    §3.1 for full design rationale.
    """

    def __init__(
        self,
        responses: list[str | ChatResponse] | None = None,
        *,
        handler: _Handler | None = None,
    ) -> None:
        if (responses is None) == (handler is None):
            raise ValueError(
                "Provide exactly one of `responses` or `handler`"
            )
        self._responses: list[ChatResponse] | None = (
            [_coerce_response(r) for r in responses]
            if responses is not None
            else None
        )
        self._initial_response_count: int = (
            len(self._responses) if self._responses is not None else 0
        )
        self._handler: _Handler | None = handler
        self._calls: list[CallRecord] = []

    @property
    def call_count(self) -> int:
        """Number of times ``chat`` has been invoked."""
        return len(self._calls)


def _coerce_response(value: str | ChatResponse) -> ChatResponse:
    """Wrap raw str into a default ChatResponse; pass ChatResponse through."""
    if isinstance(value, ChatResponse):
        return value
    if isinstance(value, str):
        return ChatResponse(
            content=value,
            model="fake",
            usage=None,
            finish_reason="stop",
            raw=None,
        )
    raise TypeError(
        f"FakeLLMClient `responses` must contain str or ChatResponse, "
        f"got {type(value).__name__}"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
uv run pytest packages/everalgo-core/tests/testing/test_fake_llm.py -v
```

Expected: 8 PASS (3 CallRecord + 5 new constructor).

- [ ] **Step 5: Run quality gates**

Run:
```bash
uv run ruff check packages/everalgo-core/src/everalgo/testing/ packages/everalgo-core/tests/testing/
uv run ruff format --check packages/everalgo-core/src/everalgo/testing/ packages/everalgo-core/tests/testing/
uv run mypy packages/everalgo-core/src/everalgo/testing/
```

Expected: 全部 clean.

- [ ] **Step 6: Commit**

```bash
git add packages/everalgo-core/src/everalgo/testing/fake_llm.py \
        packages/everalgo-core/tests/testing/test_fake_llm.py
git commit -m "$(cat <<'EOF'
✨ feat(testing): add FakeLLMClient constructor with mutual exclusion

Introduce FakeLLMClient with two construction modes (responses XOR
handler), str→ChatResponse auto-wrap helper, and call_count property.
chat() to follow in next commit. See spec §3.1.
EOF
)"
```

---

## Task 3: `FakeLLMClient.chat` — scripted-list 模式

**Files:**
- Modify: `packages/everalgo-core/src/everalgo/testing/fake_llm.py` (add `chat` method, list-mode branch only)
- Modify: `packages/everalgo-core/tests/testing/test_fake_llm.py` (append scripted-list behavior tests)

- [ ] **Step 1: Append failing tests**

Append to `packages/everalgo-core/tests/testing/test_fake_llm.py`:

```python


# ---- FakeLLMClient.chat scripted-list mode (Task 3) -----------------------


async def test_chat_str_element_wrapped_to_default_chat_response() -> None:
    """str element gets auto-wrapped: model='fake', finish_reason='stop'."""
    client = FakeLLMClient(responses=["hello"])
    response = await client.chat(
        messages=[ChatMessage(role="user", content="hi")]
    )
    assert response.content == "hello"
    assert response.model == "fake"
    assert response.usage is None
    assert response.finish_reason == "stop"
    assert response.raw is None


async def test_chat_chat_response_element_passed_through_unchanged() -> None:
    """ChatResponse instance returned as-is, preserving usage/finish_reason."""
    from everalgo.llm.types import Usage

    canned = ChatResponse(
        content="canned",
        model="custom-model",
        usage=Usage(prompt_tokens=10, completion_tokens=5),
        finish_reason="length",
    )
    client = FakeLLMClient(responses=[canned])
    response = await client.chat(
        messages=[ChatMessage(role="user", content="hi")]
    )
    assert response is canned


async def test_chat_responses_popped_in_call_order() -> None:
    """Multiple responses returned in FIFO order."""
    client = FakeLLMClient(responses=["first", "second", "third"])
    msgs = [ChatMessage(role="user", content="hi")]
    r1 = await client.chat(messages=msgs)
    r2 = await client.chat(messages=msgs)
    r3 = await client.chat(messages=msgs)
    assert (r1.content, r2.content, r3.content) == ("first", "second", "third")


async def test_chat_exhausted_script_raises_runtime_error() -> None:
    """N+1th call raises RuntimeError with `(used N of N)` message."""
    client = FakeLLMClient(responses=["only"])
    msgs = [ChatMessage(role="user", content="hi")]
    await client.chat(messages=msgs)  # exhaust the script
    with pytest.raises(RuntimeError, match=r"script exhausted.*used 1 of 1"):
        await client.chat(messages=msgs)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
uv run pytest packages/everalgo-core/tests/testing/test_fake_llm.py -v
```

Expected: 4 new tests FAIL with `AttributeError: 'FakeLLMClient' object has no attribute 'chat'` (or similar). Previous 8 still PASS.

- [ ] **Step 3: Add `chat` to FakeLLMClient (list-mode branch only)**

Modify `packages/everalgo-core/src/everalgo/testing/fake_llm.py` — locate the `FakeLLMClient` class (defined in Task 2) and append the `chat` method **before** the closing of the class:

```python
    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: Mapping[str, Any] | None = None,
        **extra: Any,
    ) -> ChatResponse:
        """Match ``LLMClient.chat`` Protocol; record + dispatch by mode."""
        self._calls.append(
            CallRecord(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                extra=dict(extra),
            )
        )
        if self._handler is not None:
            raise NotImplementedError(
                "handler mode lands in Task 4"
            )  # pragma: no cover
        # scripted-list mode
        assert self._responses is not None  # narrowed by __init__ invariant
        if not self._responses:
            raise RuntimeError(
                f"FakeLLMClient script exhausted "
                f"(used {self._initial_response_count} of "
                f"{self._initial_response_count} responses)"
            )
        return self._responses.pop(0)
```

Note: `# pragma: no cover` on the `NotImplementedError` line lets coverage skip the line in Task 3; Task 4 will replace it.

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
uv run pytest packages/everalgo-core/tests/testing/test_fake_llm.py -v
```

Expected: 12 PASS (8 prior + 4 new scripted-list).

- [ ] **Step 5: Run quality gates**

Run:
```bash
uv run ruff check packages/everalgo-core/src/everalgo/testing/ packages/everalgo-core/tests/testing/
uv run ruff format --check packages/everalgo-core/src/everalgo/testing/ packages/everalgo-core/tests/testing/
uv run mypy packages/everalgo-core/src/everalgo/testing/
```

Expected: 全部 clean.

- [ ] **Step 6: Commit**

```bash
git add packages/everalgo-core/src/everalgo/testing/fake_llm.py \
        packages/everalgo-core/tests/testing/test_fake_llm.py
git commit -m "$(cat <<'EOF'
✨ feat(testing): add FakeLLMClient.chat scripted-list mode

Implements list-mode dispatch: pop in FIFO order, str auto-wrap,
ChatResponse pass-through, exhaustion raises RuntimeError with
`(used N of N)` message. Handler mode lands in next commit.
EOF
)"
```

---

## Task 4: `FakeLLMClient.chat` — callable handler 模式

**Files:**
- Modify: `packages/everalgo-core/src/everalgo/testing/fake_llm.py` (replace handler-branch placeholder + add `_invoke_handler` helper)
- Modify: `packages/everalgo-core/tests/testing/test_fake_llm.py` (append handler-mode tests)

- [ ] **Step 1: Append failing tests**

Append to `packages/everalgo-core/tests/testing/test_fake_llm.py`:

```python


# ---- FakeLLMClient.chat callable handler mode (Task 4) --------------------


async def test_chat_sync_handler_invoked_correctly() -> None:
    """Sync handler returning ChatResponse is dispatched normally."""

    def handler(
        messages: list[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        return ChatResponse(content="sync-out", model="fake-sync")

    client = FakeLLMClient(handler=handler)
    response = await client.chat(
        messages=[ChatMessage(role="user", content="hi")]
    )
    assert response.content == "sync-out"
    assert response.model == "fake-sync"


async def test_chat_async_handler_awaited_correctly() -> None:
    """Async handler is awaited; result is the resolved ChatResponse."""

    async def handler(
        messages: list[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        return ChatResponse(content="async-out", model="fake-async")

    client = FakeLLMClient(handler=handler)
    response = await client.chat(
        messages=[ChatMessage(role="user", content="hi")]
    )
    assert response.content == "async-out"
    assert response.model == "fake-async"


async def test_chat_handler_receives_messages_and_kwargs() -> None:
    """Handler sees messages + model + temperature + max_tokens + extras."""
    captured: dict[str, Any] = {}

    def handler(
        messages: list[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        captured["messages"] = messages
        captured["kwargs"] = kwargs
        return ChatResponse(content="x", model="fake")

    client = FakeLLMClient(handler=handler)
    msgs = [ChatMessage(role="user", content="hi")]
    await client.chat(
        messages=msgs,
        model="gpt-4o-mini",
        temperature=0.5,
        max_tokens=64,
        response_format={"type": "json_object"},
        seed=42,
    )
    assert captured["messages"] == msgs
    assert captured["kwargs"]["model"] == "gpt-4o-mini"
    assert captured["kwargs"]["temperature"] == 0.5
    assert captured["kwargs"]["max_tokens"] == 64
    assert captured["kwargs"]["response_format"] == {"type": "json_object"}
    assert captured["kwargs"]["seed"] == 42


async def test_chat_handler_wrong_return_type_raises() -> None:
    """Handler returning non-ChatResponse, non-Awaitable raises TypeError."""

    def handler(messages: list[ChatMessage], **kwargs: Any) -> Any:
        return {"content": "wrong"}  # not ChatResponse

    client = FakeLLMClient(handler=handler)
    with pytest.raises(TypeError, match=r"must return ChatResponse.*got dict"):
        await client.chat(
            messages=[ChatMessage(role="user", content="hi")]
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
uv run pytest packages/everalgo-core/tests/testing/test_fake_llm.py -v
```

Expected: 4 new tests FAIL with `NotImplementedError: handler mode lands in Task 4`. Previous 12 still PASS.

- [ ] **Step 3: Replace handler placeholder + add `_invoke_handler` helper**

In `packages/everalgo-core/src/everalgo/testing/fake_llm.py`:

3a. **Add `inspect` import** at the top of the file (alphabetically before `from collections.abc`):

```python
import inspect
```

3b. **Replace the placeholder** in `FakeLLMClient.chat`:

Find:
```python
        if self._handler is not None:
            raise NotImplementedError(
                "handler mode lands in Task 4"
            )  # pragma: no cover
```

Replace with:
```python
        if self._handler is not None:
            return await _invoke_handler(
                self._handler,
                messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                **extra,
            )
```

3c. **Append `_invoke_handler`** to the end of the file (after `_coerce_response`):

```python


async def _invoke_handler(
    handler: _Handler,
    messages: list[ChatMessage],
    **kwargs: Any,
) -> ChatResponse:
    """Call handler; await if it returned a coroutine; type-check result."""
    result = handler(messages, **kwargs)
    if inspect.isawaitable(result):
        result = await result
    if not isinstance(result, ChatResponse):
        raise TypeError(
            f"FakeLLMClient handler must return ChatResponse or "
            f"Awaitable[ChatResponse], got {type(result).__name__}"
        )
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
uv run pytest packages/everalgo-core/tests/testing/test_fake_llm.py -v
```

Expected: 16 PASS (12 prior + 4 new handler).

- [ ] **Step 5: Run quality gates**

Run:
```bash
uv run ruff check packages/everalgo-core/src/everalgo/testing/ packages/everalgo-core/tests/testing/
uv run ruff format --check packages/everalgo-core/src/everalgo/testing/ packages/everalgo-core/tests/testing/
uv run mypy packages/everalgo-core/src/everalgo/testing/
```

Expected: 全部 clean.

- [ ] **Step 6: Commit**

```bash
git add packages/everalgo-core/src/everalgo/testing/fake_llm.py \
        packages/everalgo-core/tests/testing/test_fake_llm.py
git commit -m "$(cat <<'EOF'
✨ feat(testing): add FakeLLMClient.chat callable handler mode

Implements handler-mode dispatch via _invoke_handler helper:
sync handlers run inline, async handlers are awaited
(inspect.isawaitable), wrong return types raise TypeError.
Replaces the Task 3 NotImplementedError placeholder.
EOF
)"
```

---

## Task 5: `FakeLLMClient.calls` 录制 + Protocol conformance

**Files:**
- Modify: `packages/everalgo-core/src/everalgo/testing/fake_llm.py` (add `calls` property)
- Modify: `packages/everalgo-core/tests/testing/test_fake_llm.py` (append recording + Protocol tests)

- [ ] **Step 1: Append failing tests**

Append to `packages/everalgo-core/tests/testing/test_fake_llm.py`:

```python


# ---- FakeLLMClient call recording + Protocol (Task 5) ---------------------

from everalgo.llm.protocols import LLMClient
from everalgo.testing.fake_llm import CallRecord


async def test_call_count_increments_per_invocation() -> None:
    """call_count tracks every chat() invocation, regardless of mode."""
    client = FakeLLMClient(responses=["a", "b"])
    assert client.call_count == 0
    msgs = [ChatMessage(role="user", content="hi")]
    await client.chat(messages=msgs)
    assert client.call_count == 1
    await client.chat(messages=msgs)
    assert client.call_count == 2


async def test_calls_property_records_messages_and_kwargs() -> None:
    """calls captures messages + each kwarg + extras into CallRecord."""
    client = FakeLLMClient(responses=["a"])
    msgs = [ChatMessage(role="user", content="hi")]
    await client.chat(
        messages=msgs,
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=128,
        response_format={"type": "json_object"},
        seed=99,
    )
    assert len(client.calls) == 1
    record = client.calls[0]
    assert isinstance(record, CallRecord)
    assert record.messages == msgs
    assert record.model == "gpt-4o-mini"
    assert record.temperature == 0.7
    assert record.max_tokens == 128
    assert record.response_format == {"type": "json_object"}
    assert record.extra == {"seed": 99}


async def test_calls_property_returns_defensive_copy() -> None:
    """Mutating the returned list must not affect internal state."""
    client = FakeLLMClient(responses=["a"])
    msgs = [ChatMessage(role="user", content="hi")]
    await client.chat(messages=msgs)
    snapshot = client.calls
    snapshot.clear()
    assert client.call_count == 1
    assert len(client.calls) == 1


def test_fake_llm_client_satisfies_LLMClient_protocol() -> None:
    """isinstance check works thanks to @runtime_checkable on LLMClient."""
    client = FakeLLMClient(responses=["a"])
    assert isinstance(client, LLMClient)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
uv run pytest packages/everalgo-core/tests/testing/test_fake_llm.py -v
```

Expected: 4 new tests FAIL — `calls` property does not exist yet (`AttributeError`). Previous 16 still PASS.

- [ ] **Step 3: Add `calls` property to `FakeLLMClient`**

Add the property right below the existing `call_count` property in `FakeLLMClient`:

```python
    @property
    def calls(self) -> list[CallRecord]:
        """All recorded ``chat`` invocations, in call order.

        Returns a defensive copy so callers can mutate the result without
        affecting internal state.
        """
        return list(self._calls)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
uv run pytest packages/everalgo-core/tests/testing/test_fake_llm.py -v
```

Expected: 20 PASS.

- [ ] **Step 5: Run quality gates**

Run:
```bash
uv run ruff check packages/everalgo-core/src/everalgo/testing/ packages/everalgo-core/tests/testing/
uv run ruff format --check packages/everalgo-core/src/everalgo/testing/ packages/everalgo-core/tests/testing/
uv run mypy packages/everalgo-core/src/everalgo/testing/
```

Expected: 全部 clean.

- [ ] **Step 6: Commit**

```bash
git add packages/everalgo-core/src/everalgo/testing/fake_llm.py \
        packages/everalgo-core/tests/testing/test_fake_llm.py
git commit -m "$(cat <<'EOF'
✨ feat(testing): expose FakeLLMClient.calls + Protocol conformance

Adds the .calls property (defensive copy of recorded CallRecords) and
a regression test asserting FakeLLMClient passes isinstance(_, LLMClient)
under @runtime_checkable.
EOF
)"
```

---

## Task 6: `assert_episode_shape`（公开符号 3/3）

**Files:**
- Create: `packages/everalgo-core/src/everalgo/testing/assertions.py`
- Test: `packages/everalgo-core/tests/testing/test_assertions.py`

- [ ] **Step 1: Write the failing tests**

Write to `packages/everalgo-core/tests/testing/test_assertions.py`:

```python
"""Tests for everalgo.testing.assertions — assert_episode_shape."""

from typing import Any

import pytest
from pydantic import ValidationError

from everalgo.testing.assertions import assert_episode_shape
from everalgo.types import Episode

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
    bad = _valid_episode_dict()
    bad["episode"] = ""
    with pytest.raises(AssertionError, match="Episode.episode is empty"):
        assert_episode_shape(bad)


def test_zero_timestamp_raises_assertion_error() -> None:
    bad = _valid_episode_dict()
    bad["timestamp"] = 0
    with pytest.raises(AssertionError, match="must be positive"):
        assert_episode_shape(bad)


def test_negative_timestamp_raises_assertion_error() -> None:
    bad = _valid_episode_dict()
    bad["timestamp"] = -1
    with pytest.raises(AssertionError, match="must be positive"):
        assert_episode_shape(bad)


def test_wrong_parent_type_raises_assertion_error() -> None:
    bad = _valid_episode_dict()
    bad["parent_type"] = "raw_message"
    with pytest.raises(AssertionError, match="must be 'memcell'"):
        assert_episode_shape(bad)


def test_empty_parent_id_raises_assertion_error() -> None:
    bad = _valid_episode_dict()
    bad["parent_id"] = ""
    with pytest.raises(AssertionError, match="Episode.parent_id is empty"):
        assert_episode_shape(bad)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
uv run pytest packages/everalgo-core/tests/testing/test_assertions.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'everalgo.testing.assertions'`.

- [ ] **Step 3: Write minimal implementation**

Write to `packages/everalgo-core/src/everalgo/testing/assertions.py`:

```python
"""Structural assertions for memory types.

See ``docs/superpowers/specs/2026-05-08-everalgo-testing-toolkit-design.md``
§3.2 for the design rationale.
"""

from __future__ import annotations

from typing import Any

from everalgo.types import Episode


def assert_episode_shape(value: dict[str, Any] | Episode) -> Episode:
    """Assert ``value`` satisfies ``Episode`` minimal business invariants.

    Combines pydantic type-level validation with 4 business invariants that
    pydantic alone does not catch (LLM may emit empty strings, zero
    timestamps, wrong ``parent_type``, etc.).

    Layered checks (in order):

    1. **Type level** — ``Episode.model_validate(value)`` parses dict (or
       passes Episode through). Type errors raise ``ValidationError``
       unmodified so the caller sees the original pydantic message.
    2. **Business invariants** — 4 checks, each raising ``AssertionError``
       with the failing invariant name:

       a. ``episode`` is a non-empty string.
       b. ``timestamp > 0`` (Unix epoch ms; ``0`` or negative = bug).
       c. ``parent_type == "memcell"`` (EPISODE path only consumes MemCell).
       d. ``parent_id`` is a non-empty string (data lineage anchor).

    Args:
        value: ``dict`` (parsed via ``Episode.model_validate``) or already-
            parsed ``Episode``.

    Returns:
        The validated ``Episode`` instance, so callers can chain further
        assertions (e.g. ``ep = assert_episode_shape(d); assert "x" in
        ep.episode``).

    Raises:
        AssertionError: If any business invariant fails. The message names
            the failed invariant.
        pydantic.ValidationError: If type-level validation fails. Re-raised
            unmodified so the caller sees the original pydantic message.
    """
    episode = (
        value if isinstance(value, Episode) else Episode.model_validate(value)
    )
    assert episode.episode, "Episode.episode is empty"
    assert episode.timestamp > 0, (
        f"Episode.timestamp must be positive (Unix epoch ms), "
        f"got {episode.timestamp}"
    )
    assert episode.parent_type == "memcell", (
        f"Episode.parent_type must be 'memcell' (EPISODE path), "
        f"got {episode.parent_type!r}"
    )
    assert episode.parent_id, "Episode.parent_id is empty"
    return episode
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
uv run pytest packages/everalgo-core/tests/testing/test_assertions.py -v
```

Expected: 9 PASS.

- [ ] **Step 5: Run quality gates**

Run:
```bash
uv run ruff check packages/everalgo-core/src/everalgo/testing/ packages/everalgo-core/tests/testing/
uv run ruff format --check packages/everalgo-core/src/everalgo/testing/ packages/everalgo-core/tests/testing/
uv run mypy packages/everalgo-core/src/everalgo/testing/
```

Expected: 全部 clean.

- [ ] **Step 6: Commit**

```bash
git add packages/everalgo-core/src/everalgo/testing/assertions.py \
        packages/everalgo-core/tests/testing/test_assertions.py
git commit -m "$(cat <<'EOF'
✨ feat(testing): add assert_episode_shape with 4 business invariants

assert_episode_shape combines pydantic type validation (re-raise on
ValidationError) with 4 business invariants that LLM-emitted JSON may
violate: episode non-empty, timestamp>0, parent_type=='memcell',
parent_id non-empty. Returns the validated Episode for chained
assertions. See spec §3.2.
EOF
)"
```

---

## Task 7: `everalgo.testing` 包导出 + 子项目验收

**Files:**
- Modify: `packages/everalgo-core/src/everalgo/testing/__init__.py` (replace placeholder)
- Test: `packages/everalgo-core/tests/testing/test_public_api.py` (new)

- [ ] **Step 1: Write the failing tests**

Write to `packages/everalgo-core/tests/testing/test_public_api.py`:

```python
"""Tests for everalgo.testing package-level public API.

Verifies the 3 documented public symbols (per AGENTS.md §7 step 6 and §9
plus spec §3) are exported at the top-level package.
"""

import everalgo.testing


def test_public_symbols_exposed_at_top_level() -> None:
    """3 public symbols accessible via attribute access on the package."""
    assert hasattr(everalgo.testing, "FakeLLMClient")
    assert hasattr(everalgo.testing, "CallRecord")
    assert hasattr(everalgo.testing, "assert_episode_shape")


def test_dunder_all_lists_exactly_3_symbols() -> None:
    """__all__ enumerates the public surface — exactly 3 entries."""
    assert sorted(everalgo.testing.__all__) == sorted([
        "CallRecord",
        "FakeLLMClient",
        "assert_episode_shape",
    ])


def test_top_level_import_works() -> None:
    """Star-friendly import from the package root."""
    from everalgo.testing import (
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
uv run pytest packages/everalgo-core/tests/testing/test_public_api.py -v
```

Expected: 3 FAIL — `everalgo.testing` currently exports `__all__: list[str] = []` (子项目 0 placeholder).

- [ ] **Step 3: Replace `__init__.py`**

Replace the contents of `packages/everalgo-core/src/everalgo/testing/__init__.py` with:

```python
"""Public testing helpers for EverAlgo — assertions + fake_llm.

Mirrors ``numpy.testing`` / ``torch.testing`` (see ADR 005,
``docs/decisions/005-testing-as-public-subpackage.md``): testing helpers
live inside ``everalgo-core`` rather than as a separate distribution.

Public symbols (per AGENTS.md §7 step 6 + §9 + spec §3):

- ``FakeLLMClient`` — in-memory ``LLMClient`` Protocol implementation
- ``CallRecord``   — recorded chat() invocation type (for assertions)
- ``assert_episode_shape`` — Episode structural assertion helper
"""

from everalgo.testing.assertions import assert_episode_shape
from everalgo.testing.fake_llm import CallRecord, FakeLLMClient

__all__ = [
    "CallRecord",
    "FakeLLMClient",
    "assert_episode_shape",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
uv run pytest packages/everalgo-core/tests/testing/ -v
```

Expected: 23 PASS total (3 CallRecord + 5 FakeLLMClient construct + 4 scripted-list + 4 handler + 4 recording/Protocol + 9 assert + 3 public_api = 32 ... wait let me re-count).

Actual test counts:
- `test_fake_llm.py`: 3 (CallRecord) + 5 (constructor) + 4 (scripted-list) + 4 (handler) + 4 (recording+Protocol) = 20
- `test_assertions.py`: 9
- `test_public_api.py`: 3

**Total expected: 32 PASS.**

- [ ] **Step 5: Full subproject acceptance — run everything**

Run all 4 quality gates as one batch:

```bash
uv run ruff check packages/everalgo-core/
uv run ruff format --check packages/everalgo-core/
uv run mypy packages/everalgo-core/
uv run pytest packages/everalgo-core/ -v
```

Expected:
- ruff check: clean
- ruff format check: clean
- mypy: 0 errors in src/everalgo/testing/
- pytest: **whole everalgo-core suite passes** (子项目 1+2 累计 ~50+ tests + 子项目 3 新增 32 = ~82+ tests)

If any of these fail, **stop and fix** — do not commit.

- [ ] **Step 6: Verify AGENTS.md contract end-to-end**

Run a quick sanity import to verify §7 step 6 + §9 contracts hold:

```bash
uv run python -c "from everalgo.testing import FakeLLMClient, CallRecord, assert_episode_shape; print('AGENTS.md contract OK:', FakeLLMClient, CallRecord, assert_episode_shape)"
```

Expected output: prints all 3 symbols' class/function repr without ImportError.

- [ ] **Step 7: Commit**

```bash
git add packages/everalgo-core/src/everalgo/testing/__init__.py \
        packages/everalgo-core/tests/testing/test_public_api.py
git commit -m "$(cat <<'EOF'
✨ feat(testing): re-export 3 public symbols at everalgo.testing top level

Replace the sub-project 0 placeholder __init__.py with the final
re-export block exposing FakeLLMClient, CallRecord, and
assert_episode_shape. Fulfils AGENTS.md §7 step 6 + §9 contract.

Sub-project 3 (Testing Toolkit) complete:
- 3 public symbols, ~32 tests, 0 new dependencies
- Industry references: pydantic-ai TestModel + DSPy DummyLM +
  LangChain core fake_chat_models + LangChain langchain_tests +
  instructor (per spec §7)
EOF
)"
```

---

## Self-Review Checklist (作者自审，已通过)

### 1. Spec coverage

| Spec 章节 | 实现 task |
|---|---|
| §2 File Map（3 src + 2 test 文件） | Task 0 (目录) + Task 1-5 (fake_llm) + Task 6 (assertions) + Task 7 (`__init__.py` + `test_public_api.py`) |
| §3 公开 API §3.1 (FakeLLMClient + CallRecord) | Task 1 (CallRecord) + Task 2 (constructor) + Task 3 (scripted-list) + Task 4 (handler) + Task 5 (recording + Protocol) |
| §3 公开 API §3.2 (assert_episode_shape) | Task 6 |
| §4 错误处理矩阵 6 行 | Task 2 (ValueError + TypeError on invalid element) + Task 3 (RuntimeError on exhaustion) + Task 4 (TypeError on wrong handler return) + Task 6 (AssertionError × 4 + ValidationError re-raise) |
| §5 测试矩阵 5 测试组 | Task 2 (TestConstructorValidation) + Task 3 (TestScriptedList 4/5) + Task 4 (TestCallableHandler) + Task 5 (TestRecording + TestProtocolConformance) + Task 6 (TestAssertEpisodeShape) |
| §6 验收标准 7 项 | Task 7 Step 5+6 全覆盖 |

无 spec 章节未覆盖。

### 2. Placeholder scan

`grep -nE "TODO|TBD|FIXME|XXX|implement later|fill in" docs/superpowers/plans/2026-05-08-everalgo-testing-toolkit.md`
预期：0 hits。

### 3. Type consistency

- `CallRecord` 字段顺序一致：Task 1 定义 → Task 5 测试 → Task 7 export
- `FakeLLMClient.__init__` 签名一致：Task 2 → Task 3 (chat 内部使用) → Task 4 (handler 调用) → Task 5 (calls 测试) → Task 7 (smoke)
- `_HandlerReturn` / `_Handler` 类型别名 Task 2 引入，Task 4 使用，无冲突
- `assert_episode_shape` 签名 `dict[str, Any] | Episode → Episode` 在 Task 6 spec + 实现 + 测试 三处一致

### 4. Lessons learned 应用

- ✅ 测试函数 `-> None` 注解 — 全部 task 的测试代码已显式标注
- ✅ `tests/testing/__init__.py` 不创建 — Task 0 Step 2 显式说明 + Verify
- ✅ 每个 task 的 commit 含 `Co-Authored-By` — 通过模板说明（关键约束 §8）
- ✅ ruff + mypy + pytest 三 gate 每 task 必过 — 每 task 的 Step 5
- ✅ 0 新依赖 — Task 0 Step 1 验证现有占位 + 不动 pyproject.toml
- ✅ 公开符号清单与 AGENTS.md / spec 严格一致 — Task 7 Step 1 测试断言 `__all__` 长度

### 5. 任务大小

- Task 0：约 5 分钟（目录占位）
- Task 1-2-3-4-5：每个 30-45 分钟（FakeLLMClient 增量积累）
- Task 6：约 35 分钟（assert_episode_shape 一次落地）
- Task 7：约 25 分钟（包导出 + smoke + 子项目验收）

总计：约 4-5 小时（含 review 循环）。
