# EverCore LLM Stack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 落地 EverCore 子项目 2 (LLM Stack) — `evercore.llm.*` chat-style 抽象层最小集（7 个对外符号）+ openai_compat 单家 provider，让 EpisodeExtractor 可以 `await client.chat(messages)` 调 LLM 并拿到结构化 `ChatResponse`。

**Architecture:** 严格按 [`docs/superpowers/specs/2026-05-08-evercore-llm-stack-design.md`](../specs/2026-05-08-evercore-llm-stack-design.md) 落地。`LLMClient` 是 `@runtime_checkable Protocol`，`async def chat()` 单 method。`ChatMessage / ChatResponse / Usage` 是 pydantic v2 `BaseModel`。`LLMConfig.api_key` 用 `SecretStr` 屏蔽日志。`LLMError` 单基类 + SDK 错误 `__cause__` 链。`build_client` 函数式工厂，lazy import OpenAICompatClient 保 cold-start 友好。`OpenAICompatClient` 薄 wrap `openai.AsyncOpenAI`，把 ChatMessage/dict 互转 + finish_reason 折叠到 3 值 Literal + Usage 包成结构化对象。

**Tech Stack:** Python 3.12, pydantic ≥ 2.7 (BaseModel + SecretStr + Field), openai ≥ 1.0 (AsyncOpenAI), pytest + respx (HTTP mock), uv workspace, ruff / mypy / pyright (子项目 1 已配置 strict 模式)。

**Code language convention:** 严格英文（identifier / docstring / inline comment / commit message body）。本 plan 自然语言注释中文允许（步骤标题、Run / Expected）。

**Lessons from sub-project 1（已固化在 ruff / mypy 配置中）:**
- `tests/llm/__init__.py` 不创建（namespace package + `--import-mode=importlib`）
- 测试函数加 `-> None`（mypy strict）
- 故意触发 ValidationError / 不匹配类型用 `# type: ignore[arg-type]` 或 `[call-arg]`
- 测试 docstring 不强制（ruff per-file-ignores D103 已配）
- 字段 / type 命名严格 PEP 8（`PascalCase` 类，`snake_case` 字段）

---

## File Structure

新增源文件 7 个：

| 文件 | 职责 |
|---|---|
| `packages/evercore-core/src/evercore/llm/__init__.py` | re-export 7 symbols + `__all__` |
| `packages/evercore-core/src/evercore/llm/protocols.py` | `LLMClient` Protocol |
| `packages/evercore-core/src/evercore/llm/types.py` | `ChatMessage` / `ChatResponse` / `Usage` |
| `packages/evercore-core/src/evercore/llm/config.py` | `LLMConfig` (含 `SecretStr api_key`) |
| `packages/evercore-core/src/evercore/llm/errors.py` | `LLMError` |
| `packages/evercore-core/src/evercore/llm/factory.py` | `build_client` (lazy import OpenAICompatClient) |
| `packages/evercore-core/src/evercore/llm/providers/openai_compat.py` | `OpenAICompatClient` |

修改源文件 1 个：

| 文件 | 改动 |
|---|---|
| `packages/evercore-core/pyproject.toml` | 在 `dependencies` 加 `openai>=1.0` |

新增测试文件 6 个：

| 文件 | 职责 |
|---|---|
| `tests/llm/test_types.py` | ChatMessage / ChatResponse / Usage round-trip + `extra="ignore"` |
| `tests/llm/test_config.py` | LLMConfig defaults + SecretStr 屏蔽 + `get_secret_value` 解包 |
| `tests/llm/test_errors.py` | `LLMError(...) from sdk_err` 链式抛出 + `__cause__` |
| `tests/llm/test_protocols.py` | `@runtime_checkable` 结构匹配 |
| `tests/llm/test_factory.py` | `build_client` 返回 LLMClient instance + lazy import 验证 |
| `tests/llm/providers/openai_compat_test.py`（命名说明：单文件直接 `test_openai_compat.py`，**不**用 `<feature>_test.py` 后缀；plan 文件名遵循 `test_*.py` 规范，与子项目 1 一致） | respx-mocked HTTP round-trip + 错误透传 |

预创建 `tests/llm/providers/` 目录（不放 `__init__.py`）。

---

## Task 0: pyproject.toml 加 openai 依赖 + tests/llm 目录骨架

**Files:**
- Modify: `packages/evercore-core/pyproject.toml`

- [ ] **Step 1: 修改 pyproject.toml 添加 openai 依赖**

把 `packages/evercore-core/pyproject.toml` 的 `dependencies` 字段从

```toml
dependencies = [
  "pydantic>=2.7",
]
```

改为

```toml
dependencies = [
  "openai>=1.0",
  "pydantic>=2.7",
]
```

> 字母序排列；`openai` 在 `pydantic` 前。

- [ ] **Step 2: 同步 workspace 依赖**

Run: `uv sync --all-packages`

Expected: 输出含 `+ openai==1.x.x`（任何 1.0+ 小版本），含其传递依赖（如 `httpx` / `tiktoken` 之类），无 error。

- [ ] **Step 3: 验证 openai 可 import**

Run: `uv run python -c "import openai; print(openai.__version__)"`

Expected: 输出 `1.x.x`（≥ 1.0），无 error。

- [ ] **Step 4: Commit**

```bash
git add packages/evercore-core/pyproject.toml uv.lock
git commit -m "🎉 chore(core): add openai dependency for LLM Stack subproject"
```

---

## Task 1: ChatMessage type

**Files:**
- Create: `packages/evercore-core/src/evercore/llm/types.py`
- Create: `packages/evercore-core/tests/llm/test_types.py`

- [ ] **Step 1: Write the failing tests**

Create `packages/evercore-core/tests/llm/test_types.py`:

```python
"""Tests for evercore.llm.types — ChatMessage / Usage / ChatResponse."""

import json

import pytest
from pydantic import ValidationError

from evercore.llm.types import ChatMessage


def test_chat_message_minimum_required_fields() -> None:
    msg = ChatMessage(role="user", content="hello")
    assert msg.role == "user"
    assert msg.content == "hello"


def test_chat_message_role_accepts_three_values() -> None:
    """Minimal set: system / user / assistant — tool / function are out of EPISODE scope."""
    for role in ("system", "user", "assistant"):
        msg = ChatMessage(role=role, content="x")  # type: ignore[arg-type]
        assert msg.role == role


def test_chat_message_invalid_role_raises() -> None:
    with pytest.raises(ValidationError):
        ChatMessage(role="tool", content="x")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        ChatMessage(role="developer", content="x")  # type: ignore[arg-type]


def test_chat_message_missing_required_field_raises() -> None:
    with pytest.raises(ValidationError):
        ChatMessage(role="user")  # type: ignore[call-arg]


def test_chat_message_extra_fields_silently_ignored() -> None:
    """OpenAI payload may carry name / tool_call_id — drop them."""
    msg = ChatMessage.model_validate(
        {
            "role": "assistant",
            "content": "hi",
            "name": "ignored",
            "tool_call_id": "ignored",
        }
    )
    assert not hasattr(msg, "name")
    assert not hasattr(msg, "tool_call_id")


def test_chat_message_json_round_trip() -> None:
    msg = ChatMessage(role="user", content="hi")
    serialised = msg.model_dump_json()
    assert json.loads(serialised) == {"role": "user", "content": "hi"}
    assert ChatMessage.model_validate_json(serialised) == msg
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/evercore-core/tests/llm/test_types.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'evercore.llm.types'`.

- [ ] **Step 3: Write minimal implementation**

Create `packages/evercore-core/src/evercore/llm/types.py`:

```python
"""LLM wire types — chat-style messages, response, token usage.

These are the on-the-wire data contracts a caller sees when invoking
``LLMClient.chat``. They mirror the OpenAI Chat Completions API closely so
the openai_compat provider can pass through values with minimal translation.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    """Single chat-style message turn.

    The ``role`` set is intentionally narrow (3 values). ``tool`` and
    multimodal ``content`` blocks are out of EPISODE scope; adding them later
    is a SemVer minor bump (extending a Literal is a backward-compatible
    structural widening).
    """

    role: Literal["system", "user", "assistant"]
    content: str

    model_config = ConfigDict(extra="ignore")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/evercore-core/tests/llm/test_types.py -v`

Expected: 6 PASS。

- [ ] **Step 5: Run quality gates**

```
uv run ruff check packages/evercore-core/
uv run ruff format --check packages/evercore-core/
uv run mypy packages/evercore-core/
uv run pyright packages/evercore-core/
```

Expected: all green (no new errors compared to sub-project 1 baseline).

- [ ] **Step 6: Commit**

```bash
git add packages/evercore-core/src/evercore/llm/types.py packages/evercore-core/tests/llm/test_types.py
git commit -m "✨ feat(llm): add ChatMessage type (Literal role, extra=ignore)"
```

---

## Task 2: Usage type

**Files:**
- Modify: `packages/evercore-core/src/evercore/llm/types.py` (extend with Usage class)
- Modify: `packages/evercore-core/tests/llm/test_types.py` (extend with Usage tests)

- [ ] **Step 1: Append failing tests to test_types.py**

```python
from evercore.llm.types import Usage


def test_usage_default_fields_are_none() -> None:
    """Both fields default to None to distinguish 'missing' from 'zero tokens'."""
    u = Usage()
    assert u.prompt_tokens is None
    assert u.completion_tokens is None


def test_usage_explicit_values_round_trip() -> None:
    u = Usage(prompt_tokens=12, completion_tokens=4)
    rebuilt = Usage.model_validate_json(u.model_dump_json())
    assert rebuilt == u
    assert rebuilt.prompt_tokens == 12
    assert rebuilt.completion_tokens == 4


def test_usage_partial_values_allowed() -> None:
    """Provider may report only one side of usage; the other stays None."""
    u = Usage(prompt_tokens=42)
    assert u.prompt_tokens == 42
    assert u.completion_tokens is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/evercore-core/tests/llm/test_types.py -v`

Expected: 3 new FAIL — `ImportError: cannot import name 'Usage'`. The 6 ChatMessage tests still PASS.

- [ ] **Step 3: Append minimal implementation to types.py**

```python
class Usage(BaseModel):
    """Token usage from a single LLM call.

    Both fields are ``int | None`` because some self-hosted / OpenAI-compatible
    backends do not return ``usage`` in the response. ``None`` semantically
    distinguishes "missing data" from "zero tokens used".
    """

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/evercore-core/tests/llm/test_types.py -v`

Expected: 9 PASS (6 ChatMessage + 3 Usage)。

- [ ] **Step 5: Commit**

```bash
git add packages/evercore-core/src/evercore/llm/types.py packages/evercore-core/tests/llm/test_types.py
git commit -m "✨ feat(llm): add Usage type (Optional int fields for missing-vs-zero)"
```

---

## Task 3: ChatResponse type

**Files:**
- Modify: `packages/evercore-core/src/evercore/llm/types.py` (extend with ChatResponse)
- Modify: `packages/evercore-core/tests/llm/test_types.py` (extend with ChatResponse tests)

- [ ] **Step 1: Append failing tests**

```python
from evercore.llm.types import ChatResponse


def test_chat_response_minimum_required_fields() -> None:
    resp = ChatResponse(content="hello", model="gpt-4o-mini")
    assert resp.content == "hello"
    assert resp.model == "gpt-4o-mini"
    assert resp.usage is None
    assert resp.finish_reason is None
    assert resp.raw is None


def test_chat_response_with_usage_and_finish_reason() -> None:
    resp = ChatResponse(
        content="ok",
        model="gpt-4o-mini",
        usage=Usage(prompt_tokens=5, completion_tokens=3),
        finish_reason="stop",
    )
    assert resp.usage is not None
    assert resp.usage.prompt_tokens == 5
    assert resp.finish_reason == "stop"


def test_chat_response_finish_reason_three_values() -> None:
    """finish_reason is a Literal of stop / length / content_filter."""
    for reason in ("stop", "length", "content_filter"):
        resp = ChatResponse(content="x", model="m", finish_reason=reason)  # type: ignore[arg-type]
        assert resp.finish_reason == reason


def test_chat_response_invalid_finish_reason_raises() -> None:
    with pytest.raises(ValidationError):
        ChatResponse(content="x", model="m", finish_reason="tool_calls")  # type: ignore[arg-type]


def test_chat_response_missing_content_raises() -> None:
    with pytest.raises(ValidationError):
        ChatResponse(model="m")  # type: ignore[call-arg]


def test_chat_response_raw_optional_dict() -> None:
    resp = ChatResponse(content="x", model="m", raw={"id": "chatcmpl-xyz"})
    assert resp.raw == {"id": "chatcmpl-xyz"}


def test_chat_response_json_round_trip_with_nested_usage() -> None:
    resp = ChatResponse(
        content="x",
        model="m",
        usage=Usage(prompt_tokens=10),
        finish_reason="length",
    )
    rebuilt = ChatResponse.model_validate_json(resp.model_dump_json())
    assert rebuilt == resp
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/evercore-core/tests/llm/test_types.py -v`

Expected: 7 new FAIL — `ImportError: cannot import name 'ChatResponse'`. The 9 existing tests still PASS.

- [ ] **Step 3: Append minimal implementation to types.py**

```python
class ChatResponse(BaseModel):
    """Structured response from a single LLM chat call."""

    content: str
    model: str
    usage: Usage | None = None
    finish_reason: Literal["stop", "length", "content_filter"] | None = None
    raw: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Provider-specific original response payload. Populated only when "
            "the provider implementation explicitly opts in (e.g. debug mode). "
            "Production callers should rely on the structured "
            "``content`` / ``usage`` / ``finish_reason`` fields and not "
            "depend on ``raw`` being non-None."
        ),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/evercore-core/tests/llm/test_types.py -v`

Expected: 16 PASS。

- [ ] **Step 5: Commit**

```bash
git add packages/evercore-core/src/evercore/llm/types.py packages/evercore-core/tests/llm/test_types.py
git commit -m "✨ feat(llm): add ChatResponse type (3-value finish_reason Literal)"
```

---

## Task 4: LLMConfig with SecretStr api_key

**Files:**
- Create: `packages/evercore-core/src/evercore/llm/config.py`
- Create: `packages/evercore-core/tests/llm/test_config.py`

- [ ] **Step 1: Write the failing tests**

Create `packages/evercore-core/tests/llm/test_config.py`:

```python
"""Tests for evercore.llm.config.LLMConfig."""

import pytest
from pydantic import SecretStr, ValidationError

from evercore.llm.config import LLMConfig


def test_llm_config_minimum_required_fields() -> None:
    cfg = LLMConfig(
        model="gpt-4o-mini",
        api_key="sk-real-secret",  # type: ignore[arg-type]
        base_url="https://api.openai.com/v1",
    )
    assert cfg.model == "gpt-4o-mini"
    assert cfg.base_url == "https://api.openai.com/v1"


def test_llm_config_default_field_values() -> None:
    cfg = LLMConfig(
        model="m",
        api_key="k",  # type: ignore[arg-type]
        base_url="u",
    )
    assert cfg.temperature == 0.0
    assert cfg.max_tokens is None
    assert cfg.timeout == 60.0
    assert cfg.extra == {}


def test_llm_config_api_key_is_secret_str() -> None:
    cfg = LLMConfig(model="m", api_key="sk-secret", base_url="u")  # type: ignore[arg-type]
    assert isinstance(cfg.api_key, SecretStr)


def test_llm_config_repr_masks_api_key() -> None:
    cfg = LLMConfig(model="m", api_key="sk-real-secret", base_url="u")  # type: ignore[arg-type]
    assert "sk-real-secret" not in repr(cfg)


def test_llm_config_model_dump_masks_api_key() -> None:
    cfg = LLMConfig(model="m", api_key="sk-real-secret", base_url="u")  # type: ignore[arg-type]
    dumped = cfg.model_dump()
    assert dumped["api_key"] != "sk-real-secret"


def test_llm_config_model_dump_json_masks_api_key() -> None:
    cfg = LLMConfig(model="m", api_key="sk-real-secret", base_url="u")  # type: ignore[arg-type]
    assert "sk-real-secret" not in cfg.model_dump_json()


def test_llm_config_get_secret_value_returns_raw() -> None:
    cfg = LLMConfig(model="m", api_key="sk-real-secret", base_url="u")  # type: ignore[arg-type]
    assert cfg.api_key.get_secret_value() == "sk-real-secret"


def test_llm_config_missing_required_field_raises() -> None:
    with pytest.raises(ValidationError):
        LLMConfig(model="m", api_key="k")  # type: ignore[call-arg, arg-type]


def test_llm_config_extra_dict_field_default_is_empty() -> None:
    cfg = LLMConfig(model="m", api_key="k", base_url="u")  # type: ignore[arg-type]
    cfg2 = LLMConfig(model="m", api_key="k", base_url="u")  # type: ignore[arg-type]
    cfg.extra["seed"] = 42
    assert cfg2.extra == {}, "extra default_factory must produce a fresh dict per instance"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/evercore-core/tests/llm/test_config.py -v`

Expected: 9 FAIL — `ModuleNotFoundError: No module named 'evercore.llm.config'`.

- [ ] **Step 3: Write minimal implementation**

Create `packages/evercore-core/src/evercore/llm/config.py`:

```python
"""LLM client configuration."""

from typing import Any

from pydantic import BaseModel, Field, SecretStr


class LLMConfig(BaseModel):
    """OpenAI-compatible LLM client configuration.

    The set of fields mirrors the openai-python SDK's ``AsyncOpenAI``
    constructor (``api_key`` / ``base_url`` / ``timeout``) plus per-call
    sampling defaults (``temperature`` / ``max_tokens``) and an ``extra``
    bucket for provider-specific knobs.

    ``api_key`` is wrapped in ``pydantic.SecretStr`` so that ``repr(config)``,
    ``config.model_dump()``, ``config.model_dump_json()`` and
    ``config.model_json_schema()`` all mask its value. Provider code must
    explicitly call ``config.api_key.get_secret_value()`` to obtain the raw
    string before passing it to the SDK — that explicit call is itself a
    safety checkpoint reminding the reader they are touching a credential.
    """

    model: str
    api_key: SecretStr
    base_url: str
    temperature: float = 0.0
    max_tokens: int | None = None
    timeout: float = 60.0
    extra: dict[str, Any] = Field(default_factory=dict)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/evercore-core/tests/llm/test_config.py -v`

Expected: 9 PASS。

- [ ] **Step 5: Commit**

```bash
git add packages/evercore-core/src/evercore/llm/config.py packages/evercore-core/tests/llm/test_config.py
git commit -m "✨ feat(llm): add LLMConfig with SecretStr api_key"
```

---

## Task 5: LLMError single base class

**Files:**
- Create: `packages/evercore-core/src/evercore/llm/errors.py`
- Create: `packages/evercore-core/tests/llm/test_errors.py`

- [ ] **Step 1: Write the failing tests**

Create `packages/evercore-core/tests/llm/test_errors.py`:

```python
"""Tests for evercore.llm.errors.LLMError."""

import pytest

from evercore.llm.errors import LLMError


def test_llm_error_is_exception_subclass() -> None:
    assert issubclass(LLMError, Exception)


def test_llm_error_with_message() -> None:
    err = LLMError("rate limit exceeded")
    assert str(err) == "rate limit exceeded"


def test_llm_error_chains_cause_via_pep_3134() -> None:
    """The provider layer should attach SDK-native exception via ``raise X from y``.

    Callers can then inspect ``e.__cause__`` to reach the original SDK
    exception class.
    """
    sdk_native = ValueError("upstream failure")

    with pytest.raises(LLMError) as caught:
        try:
            raise sdk_native
        except ValueError as exc:
            raise LLMError("wrapped failure") from exc

    assert caught.value.__cause__ is sdk_native
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/evercore-core/tests/llm/test_errors.py -v`

Expected: 3 FAIL — `ModuleNotFoundError: No module named 'evercore.llm.errors'`.

- [ ] **Step 3: Write minimal implementation**

Create `packages/evercore-core/src/evercore/llm/errors.py`:

```python
"""LLM-layer error types — minimal single-base set."""


class LLMError(Exception):
    """Raised on any LLM call failure.

    The provider implementation maps native SDK exceptions (e.g.
    ``openai.RateLimitError`` / ``openai.APIConnectionError``) to ``LLMError``
    using ``raise LLMError(...) from sdk_err``. Callers wanting fine-grained
    handling can either:

    - Inspect ``e.__cause__`` (PEP 3134) to reach the SDK-native exception, or
    - ``except`` the SDK type directly (the SDK exception is still in the
      cause chain).

    Future minor bumps may introduce subclasses (``LLMRateLimitError``,
    ``LLMTimeoutError``, etc.) without breaking ``except LLMError`` callers.
    """
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/evercore-core/tests/llm/test_errors.py -v`

Expected: 3 PASS。

- [ ] **Step 5: Commit**

```bash
git add packages/evercore-core/src/evercore/llm/errors.py packages/evercore-core/tests/llm/test_errors.py
git commit -m "✨ feat(llm): add LLMError single-base class"
```

---

## Task 6: LLMClient Protocol

**Files:**
- Create: `packages/evercore-core/src/evercore/llm/protocols.py`
- Create: `packages/evercore-core/tests/llm/test_protocols.py`

- [ ] **Step 1: Write the failing tests**

Create `packages/evercore-core/tests/llm/test_protocols.py`:

```python
"""Tests for evercore.llm.protocols.LLMClient — structural conformance."""

from typing import Any

from evercore.llm.protocols import LLMClient
from evercore.llm.types import ChatMessage, ChatResponse


class _ConformingClient:
    """Minimal class that structurally satisfies LLMClient."""

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: Any = None,
        **extra: Any,
    ) -> ChatResponse:
        return ChatResponse(content="", model=model or "fake")


class _NonConformingClient:
    """Missing the chat method — should not pass isinstance(... , LLMClient)."""

    def something_else(self) -> None:
        return None


def test_conforming_client_is_instance_of_protocol() -> None:
    """@runtime_checkable Protocol uses structural subtyping at runtime."""
    instance = _ConformingClient()
    assert isinstance(instance, LLMClient)


def test_non_conforming_client_is_not_instance_of_protocol() -> None:
    instance = _NonConformingClient()
    assert not isinstance(instance, LLMClient)


def test_protocol_is_runtime_checkable() -> None:
    """isinstance(_, LLMClient) must be supported via @runtime_checkable."""
    # If the decorator is missing, isinstance raises TypeError.
    isinstance(object(), LLMClient)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/evercore-core/tests/llm/test_protocols.py -v`

Expected: 3 FAIL — `ModuleNotFoundError: No module named 'evercore.llm.protocols'`.

- [ ] **Step 3: Write minimal implementation**

Create `packages/evercore-core/src/evercore/llm/protocols.py`:

```python
"""LLM client Protocol — the structural contract every provider satisfies."""

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from evercore.llm.types import ChatMessage, ChatResponse


@runtime_checkable
class LLMClient(Protocol):
    """Async LLM client structural contract.

    Implementations need not subclass this Protocol; structural conformance
    suffices (PEP 544). The ``@runtime_checkable`` decorator is for sanity
    checks (e.g. inside ``build_client``); production callers rely on static
    typing rather than ``isinstance``.
    """

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
        """Send a chat-style request and await the assistant reply.

        Args:
            messages: Ordered conversation, ending with the latest user turn.
            model: Override the per-config default model for this call.
            temperature / max_tokens: Override per-config defaults; ``None``
                falls back to the value baked into the config.
            response_format: OpenAI-compatible ``response_format`` field
                (e.g. ``{"type": "json_object"}`` for JSON mode).
            **extra: Provider-specific knobs forwarded as kwargs.

        Returns:
            ``ChatResponse`` with structured ``content`` / ``usage`` /
            ``finish_reason`` plus optional ``raw`` for debug.

        Raises:
            LLMError: Any provider-side failure, with the original SDK
                exception attached as ``__cause__`` (PEP 3134).
        """
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/evercore-core/tests/llm/test_protocols.py -v`

Expected: 3 PASS。

- [ ] **Step 5: Commit**

```bash
git add packages/evercore-core/src/evercore/llm/protocols.py packages/evercore-core/tests/llm/test_protocols.py
git commit -m "✨ feat(llm): add LLMClient @runtime_checkable Protocol"
```

---

## Task 7: OpenAICompatClient (provider implementation)

**Files:**
- Create: `packages/evercore-core/src/evercore/llm/providers/__init__.py`
- Create: `packages/evercore-core/src/evercore/llm/providers/openai_compat.py`
- Create: `packages/evercore-core/tests/llm/providers/test_openai_compat.py`

- [ ] **Step 1: Write the failing tests**

Create `packages/evercore-core/tests/llm/providers/test_openai_compat.py`:

```python
"""Tests for evercore.llm.providers.openai_compat.OpenAICompatClient."""

import json

import httpx
import pytest
import respx

from evercore.llm.config import LLMConfig
from evercore.llm.errors import LLMError
from evercore.llm.providers.openai_compat import OpenAICompatClient
from evercore.llm.types import ChatMessage, ChatResponse


def _build_config(**overrides: object) -> LLMConfig:
    base = dict(
        model="gpt-4o-mini",
        api_key="sk-test",
        base_url="https://api.openai.com/v1",
    )
    base.update(overrides)
    return LLMConfig.model_validate(base)


@pytest.fixture
def chat_completion_payload() -> dict[str, object]:
    return {
        "id": "chatcmpl-xyz",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "hi there"},
            }
        ],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
    }


async def test_openai_compat_client_chat_returns_structured_response(
    chat_completion_payload: dict[str, object],
) -> None:
    cfg = _build_config()
    client = OpenAICompatClient(cfg)

    with respx.mock(base_url=cfg.base_url, assert_all_called=True) as router:
        route = router.post("/chat/completions").mock(
            return_value=httpx.Response(200, json=chat_completion_payload)
        )
        resp = await client.chat([ChatMessage(role="user", content="hello")])

    assert isinstance(resp, ChatResponse)
    assert resp.content == "hi there"
    assert resp.model == "gpt-4o-mini"
    assert resp.usage is not None
    assert resp.usage.prompt_tokens == 7
    assert resp.usage.completion_tokens == 3
    assert resp.finish_reason == "stop"
    assert route.called


async def test_openai_compat_client_uses_config_defaults(
    chat_completion_payload: dict[str, object],
) -> None:
    cfg = _build_config(temperature=0.5, max_tokens=128)
    client = OpenAICompatClient(cfg)

    with respx.mock(base_url=cfg.base_url) as router:
        route = router.post("/chat/completions").mock(
            return_value=httpx.Response(200, json=chat_completion_payload)
        )
        await client.chat([ChatMessage(role="user", content="hi")])

    sent_body = json.loads(route.calls[0].request.content)
    assert sent_body["model"] == "gpt-4o-mini"
    assert sent_body["temperature"] == 0.5
    assert sent_body["max_tokens"] == 128


async def test_openai_compat_client_per_call_overrides_take_precedence(
    chat_completion_payload: dict[str, object],
) -> None:
    cfg = _build_config(temperature=0.5, max_tokens=128)
    client = OpenAICompatClient(cfg)

    with respx.mock(base_url=cfg.base_url) as router:
        route = router.post("/chat/completions").mock(
            return_value=httpx.Response(200, json=chat_completion_payload)
        )
        await client.chat(
            [ChatMessage(role="user", content="hi")],
            model="gpt-4o",
            temperature=0.0,
            max_tokens=64,
        )

    sent_body = json.loads(route.calls[0].request.content)
    assert sent_body["model"] == "gpt-4o"
    assert sent_body["temperature"] == 0.0
    assert sent_body["max_tokens"] == 64


async def test_openai_compat_client_wraps_sdk_error_as_llm_error() -> None:
    cfg = _build_config()
    client = OpenAICompatClient(cfg)

    with respx.mock(base_url=cfg.base_url) as router:
        router.post("/chat/completions").mock(
            return_value=httpx.Response(429, json={"error": {"message": "rate limit"}})
        )
        with pytest.raises(LLMError) as caught:
            await client.chat([ChatMessage(role="user", content="hi")])

    assert caught.value.__cause__ is not None
    # __cause__ must be a subclass of openai.OpenAIError
    import openai

    assert isinstance(caught.value.__cause__, openai.OpenAIError)


async def test_openai_compat_client_normalises_unknown_finish_reason(
    chat_completion_payload: dict[str, object],
) -> None:
    """If the provider emits a finish_reason outside the 3-value Literal, normalise to None."""
    payload = dict(chat_completion_payload)
    choices = list(payload["choices"])  # type: ignore[arg-type]
    choices[0] = {**choices[0], "finish_reason": "tool_calls"}  # type: ignore[index]
    payload["choices"] = choices  # type: ignore[assignment]

    cfg = _build_config()
    client = OpenAICompatClient(cfg)

    with respx.mock(base_url=cfg.base_url) as router:
        router.post("/chat/completions").mock(
            return_value=httpx.Response(200, json=payload)
        )
        resp = await client.chat([ChatMessage(role="user", content="hi")])

    assert resp.finish_reason is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/evercore-core/tests/llm/providers/test_openai_compat.py -v`

Expected: 5 FAIL — `ModuleNotFoundError: No module named 'evercore.llm.providers.openai_compat'`.

- [ ] **Step 3: Create empty providers package marker**

Create `packages/evercore-core/src/evercore/llm/providers/__init__.py` (empty file).

- [ ] **Step 4: Write minimal implementation**

Create `packages/evercore-core/src/evercore/llm/providers/openai_compat.py`:

```python
"""OpenAI-compatible provider — wraps openai.AsyncOpenAI."""

from collections.abc import Mapping
from typing import Any

import openai

from evercore.llm.config import LLMConfig
from evercore.llm.errors import LLMError
from evercore.llm.types import ChatMessage, ChatResponse, Usage


class OpenAICompatClient:
    """Thin async wrapper over ``openai.AsyncOpenAI``.

    Single-purpose: convert between EverCore's ``ChatMessage`` / ``ChatResponse``
    types and the openai SDK's native dict / object shapes. No retry layer,
    no rate-limit logic, no multi-key rotation — those are caller / deployment
    concerns (matching opensource ``OpenAIProvider`` simplicity, not
    Letta-grade orchestration).
    """

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._client = openai.AsyncOpenAI(
            api_key=config.api_key.get_secret_value(),
            base_url=config.base_url,
            timeout=config.timeout,
        )

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
        """Implementation of LLMClient.chat — see protocols.py for contract."""
        request_kwargs: dict[str, Any] = {
            "model": model or self._config.model,
            "messages": [m.model_dump() for m in messages],
            "temperature": (
                temperature if temperature is not None else self._config.temperature
            ),
        }
        max_tokens_val = (
            max_tokens if max_tokens is not None else self._config.max_tokens
        )
        if max_tokens_val is not None:
            request_kwargs["max_tokens"] = max_tokens_val
        if response_format is not None:
            request_kwargs["response_format"] = dict(response_format)
        request_kwargs.update(self._config.extra)
        request_kwargs.update(extra)

        try:
            completion = await self._client.chat.completions.create(**request_kwargs)
        except openai.OpenAIError as exc:
            raise LLMError(str(exc)) from exc

        choice = completion.choices[0]
        usage: Usage | None = None
        if completion.usage is not None:
            usage = Usage(
                prompt_tokens=completion.usage.prompt_tokens,
                completion_tokens=completion.usage.completion_tokens,
            )

        finish_reason = _normalise_finish_reason(choice.finish_reason)

        return ChatResponse(
            content=choice.message.content or "",
            model=completion.model,
            usage=usage,
            finish_reason=finish_reason,
            raw=None,
        )


def _normalise_finish_reason(value: str | None) -> str | None:
    """Collapse provider finish reasons to EverCore's 3-value Literal subset.

    EPISODE path treats ``tool_calls`` / ``function_call`` as out-of-scope
    (no tools wired); when a provider unexpectedly emits one the response is
    classified as ``None``. Logging the unknown value is left to providers.
    """
    if value in ("stop", "length", "content_filter"):
        return value
    return None
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest packages/evercore-core/tests/llm/providers/test_openai_compat.py -v`

Expected: 5 PASS。

- [ ] **Step 6: Run all quality gates**

```
uv run ruff check packages/evercore-core/
uv run ruff format --check packages/evercore-core/
uv run mypy packages/evercore-core/
uv run pyright packages/evercore-core/
```

Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add packages/evercore-core/src/evercore/llm/providers/__init__.py packages/evercore-core/src/evercore/llm/providers/openai_compat.py packages/evercore-core/tests/llm/providers/test_openai_compat.py
git commit -m "✨ feat(llm): add OpenAICompatClient (thin wrap over openai.AsyncOpenAI)"
```

---

## Task 8: build_client factory + lazy import test

**Files:**
- Create: `packages/evercore-core/src/evercore/llm/factory.py`
- Create: `packages/evercore-core/tests/llm/test_factory.py`

- [ ] **Step 1: Write the failing tests**

Create `packages/evercore-core/tests/llm/test_factory.py`:

```python
"""Tests for evercore.llm.factory.build_client."""

import importlib
import sys

from evercore.llm.config import LLMConfig
from evercore.llm.factory import build_client
from evercore.llm.protocols import LLMClient


def _config() -> LLMConfig:
    return LLMConfig.model_validate(
        {"model": "m", "api_key": "k", "base_url": "https://api.openai.com/v1"}
    )


def test_build_client_returns_llm_client_instance() -> None:
    client = build_client(_config())
    assert isinstance(client, LLMClient)


def test_build_client_returns_openai_compat_client() -> None:
    """Without a provider field on LLMConfig the only target is openai_compat."""
    from evercore.llm.providers.openai_compat import OpenAICompatClient

    client = build_client(_config())
    assert isinstance(client, OpenAICompatClient)


def test_factory_module_does_not_import_provider_eagerly() -> None:
    """``import evercore.llm.factory`` must not pull openai_compat into sys.modules.

    The lazy import inside ``build_client`` is load-bearing for cold-start
    cost; this test is a regression guard against a maintainer "fixing" it
    by hoisting the import to the top of the module.
    """
    # Force a clean reload of evercore.llm.factory.
    sys.modules.pop("evercore.llm.factory", None)
    sys.modules.pop("evercore.llm.providers.openai_compat", None)

    importlib.import_module("evercore.llm.factory")

    assert "evercore.llm.factory" in sys.modules
    assert "evercore.llm.providers.openai_compat" not in sys.modules, (
        "evercore.llm.factory must not import openai_compat at import time"
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/evercore-core/tests/llm/test_factory.py -v`

Expected: 3 FAIL — `ModuleNotFoundError: No module named 'evercore.llm.factory'`.

- [ ] **Step 3: Write minimal implementation**

Create `packages/evercore-core/src/evercore/llm/factory.py`:

```python
"""Factory for building an LLM client from configuration."""

from evercore.llm.config import LLMConfig
from evercore.llm.protocols import LLMClient


def build_client(config: LLMConfig) -> LLMClient:
    """Build an OpenAI-compatible LLM client from ``config``.

    Implementation note: ``OpenAICompatClient`` is imported lazily inside the
    function body so that ``evercore.llm.factory`` itself does not pull the
    ``openai`` SDK at import time. This keeps ``import evercore.llm`` cheap
    for callers that only need the Protocol / Config / Error types and never
    call ``build_client``. Maintainers — please do **not** "optimise" this
    into a top-level import; the laziness is load-bearing.
    """
    from evercore.llm.providers.openai_compat import OpenAICompatClient

    return OpenAICompatClient(config)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/evercore-core/tests/llm/test_factory.py -v`

Expected: 3 PASS。

- [ ] **Step 5: Commit**

```bash
git add packages/evercore-core/src/evercore/llm/factory.py packages/evercore-core/tests/llm/test_factory.py
git commit -m "✨ feat(llm): add build_client factory with lazy provider import"
```

---

## Task 9: evercore.llm.__init__.py — re-export 7 public symbols

**Files:**
- Modify: `packages/evercore-core/src/evercore/llm/__init__.py` (replace placeholder)
- Create: `packages/evercore-core/tests/llm/test_public_api.py`

- [ ] **Step 1: Write the failing test**

Create `packages/evercore-core/tests/llm/test_public_api.py`:

```python
"""Tests for the evercore.llm public API surface."""


def test_top_level_exports_are_seven_named_symbols() -> None:
    from evercore.llm import __all__

    assert sorted(__all__) == sorted(
        [
            "LLMClient",
            "ChatMessage",
            "ChatResponse",
            "Usage",
            "LLMConfig",
            "LLMError",
            "build_client",
        ]
    )


def test_top_level_imports_resolve() -> None:
    from evercore.llm import (
        ChatMessage,
        ChatResponse,
        LLMClient,
        LLMConfig,
        LLMError,
        Usage,
        build_client,
    )

    assert ChatMessage.__name__ == "ChatMessage"
    assert ChatResponse.__name__ == "ChatResponse"
    assert LLMClient.__name__ == "LLMClient"
    assert LLMConfig.__name__ == "LLMConfig"
    assert LLMError.__name__ == "LLMError"
    assert Usage.__name__ == "Usage"
    assert build_client.__name__ == "build_client"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/evercore-core/tests/llm/test_public_api.py -v`

Expected: 2 FAIL — `ImportError: cannot import name 'LLMClient' from 'evercore.llm'` (or similar), because `__init__.py` is still the placeholder `__all__: list[str] = []`.

- [ ] **Step 3: Replace `__init__.py`**

Replace ALL content of `packages/evercore-core/src/evercore/llm/__init__.py` with:

```python
"""LLM facade — chat-style abstraction over OpenAI-compatible providers.

Public surface (7 symbols, alphabetical-by-category):

- protocol:  LLMClient
- data:      ChatMessage, ChatResponse, Usage, LLMConfig
- error:     LLMError
- factory:   build_client
"""

from evercore.llm.config import LLMConfig
from evercore.llm.errors import LLMError
from evercore.llm.factory import build_client
from evercore.llm.protocols import LLMClient
from evercore.llm.types import ChatMessage, ChatResponse, Usage

__all__ = [
    "LLMClient",
    "ChatMessage",
    "ChatResponse",
    "Usage",
    "LLMConfig",
    "LLMError",
    "build_client",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/evercore-core/tests/llm/test_public_api.py -v`

Expected: 2 PASS。

- [ ] **Step 5: Verify smoke-import command from spec**

Run: `uv run python -c "from evercore.llm import LLMClient, LLMConfig, LLMError, ChatMessage, ChatResponse, Usage, build_client; print('OK')"`

Expected: `OK`。

- [ ] **Step 6: Commit**

```bash
git add packages/evercore-core/src/evercore/llm/__init__.py packages/evercore-core/tests/llm/test_public_api.py
git commit -m "✨ feat(llm): re-export 7 public symbols at evercore.llm top level"
```

---

## Task 10: Full subproject acceptance

**Files:** 无新增，仅运行验收命令。

- [ ] **Step 1: 全 workspace 同步**

Run: `uv sync --all-packages`

Expected: 无 error；含 openai 与其传递依赖。

- [ ] **Step 2: 顶层 import smoke**

Run: `uv run python -c "from evercore.llm import LLMClient, LLMConfig, LLMError, ChatMessage, ChatResponse, Usage, build_client; print('OK')"`

Expected: `OK`。

- [ ] **Step 3: 跑全 evercore-core 测试**

Run: `uv run pytest packages/evercore-core/tests/ -v`

Expected: 全绿，总数 ≈ 41 (sub-project 1) + 38 (sub-project 2: 16 types + 9 config + 3 errors + 3 protocols + 5 openai_compat + 3 factory + 2 public_api - 3 reused base counts) ≈ ~75 tests pass。具体数随 fixture 行为可能 ±2，**0 FAIL / 0 ERROR 是硬约束**。

- [ ] **Step 4: ruff check**

Run: `uv run ruff check packages/evercore-core/`

Expected: `All checks passed!`。

- [ ] **Step 5: ruff format check**

Run: `uv run ruff format --check packages/evercore-core/`

Expected: 无 diff。若有 diff，跑 `uv run ruff format packages/evercore-core/` 然后新增 `🎨 style` commit。

- [ ] **Step 6: mypy strict**

Run: `uv run mypy packages/evercore-core/`

Expected: `Success: no issues found in N source files`。若有 strict-mode 报错，按子项目 1 的 `# type: ignore` 模式针对性处理 + commit `🔧 fix(types)`。

- [ ] **Step 7: pyright**

Run: `uv run pyright packages/evercore-core/`

Expected: `0 errors`。

- [ ] **Step 8: 工作区整洁**

Run: `git status -sb`

Expected: clean (除了 BOSS 自有的 `?? docs/reference/` 等 pre-existing untracked)。`__pycache__` / `.ruff_cache` / `.mypy_cache` 不应 untracked（root .gitignore 应已忽略）。

- [ ] **Step 9: SecretStr 真实路径验收（手工 smoke）**

Run:

```bash
uv run python -c "
from evercore.llm import LLMConfig
cfg = LLMConfig(model='gpt-4o-mini', api_key='sk-real-secret', base_url='https://api.openai.com/v1')
print('repr:', repr(cfg))
print('dump_json:', cfg.model_dump_json())
print('unwrap:', cfg.api_key.get_secret_value())
"
```

Expected:
- `repr` 输出含 `api_key=SecretStr('**********')` 而**不**含 `sk-real-secret`
- `dump_json` 同样不含 `sk-real-secret`
- `unwrap` 输出 `sk-real-secret`

- [ ] **Step 10: 子项目 4 接口 smoke（前置验证）**

Run:

```bash
uv run python -c "
import asyncio
from evercore.llm import build_client, LLMConfig, ChatMessage

async def main():
    cfg = LLMConfig(
        model='gpt-4o-mini',
        api_key='sk-no-network-call-this-is-just-import-smoke',
        base_url='https://api.openai.com/v1',
    )
    client = build_client(cfg)
    print('client type:', type(client).__name__)

asyncio.run(main())
"
```

Expected: 输出 `client type: OpenAICompatClient`，**不调任何网络** —— `build_client(cfg)` 仅构造，未发起 chat 请求。

> 这一步证明子项目 4 EpisodeExtractor 可以拿到 client 并 `await client.chat(...)` 调用 —— LLM Stack 端到端可用。

- [ ] **Step 11: 不产生新 commit（验收 only）**

Task 10 自身仅运行验收命令，不产生新 commit。如果 Step 5 / 6 / 7 触发任何 fix，每个 fix 必须独立 commit（不与 Task 9 amend）。

- [ ] **Step 12: 查看最终 commit 链路**

Run: `git log --oneline -15`

Expected: 最近 10 个 commit 是 Task 0-9 + 可能的 0-2 个 fix commit。

---

## Self-Review Checklist (作者自审，已通过)

| 检查项 | 结果 |
|---|---|
| Spec coverage — 7 个对外符号 | ✅ Task 1-9 各覆盖 |
| Spec coverage — SecretStr | ✅ Task 4 |
| Spec coverage — Usage Optional 字段 | ✅ Task 2 |
| Spec coverage — ChatResponse.raw docstring | ✅ Task 3 |
| Spec coverage — build_client lazy import 注释 + 测试 | ✅ Task 8 |
| Spec coverage — `@runtime_checkable` LLMClient | ✅ Task 6 |
| Spec coverage — OpenAICompatClient + finish_reason 折叠 | ✅ Task 7 |
| Spec coverage — pyproject.toml openai 依赖 | ✅ Task 0 |
| Spec coverage — 4 个 quality gates 验收 | ✅ Task 10 |
| Placeholder scan | ✅ 无 TBD/TODO；所有代码块完整 |
| Type consistency | ✅ `LLMClient` / `ChatMessage` / `ChatResponse` / `Usage` / `LLMConfig` / `LLMError` / `build_client` 贯穿全 plan |
| 代码 100% 英文 | ✅ 所有 docstring / comment / commit message body 全英文 |
| TDD 严格 | ✅ 每个有 implementation 的 Task 都是 test → fail → impl → pass → commit |
| 测试覆盖 SecretStr 屏蔽 | ✅ Task 4 含 7 条针对 SecretStr 的断言（repr / dump / dump_json / get_secret_value / type） |
| 测试覆盖 lazy import | ✅ Task 8 用 sys.modules pop + reimport 验证 openai_compat 不在导入图里 |
| 测试覆盖 SDK 错误透传 | ✅ Task 7 `test_openai_compat_client_wraps_sdk_error_as_llm_error` |

---

*Plan 完成。下一步 SDD 执行（fresh subagent per task + 每 task spec/quality 双 review）。*
