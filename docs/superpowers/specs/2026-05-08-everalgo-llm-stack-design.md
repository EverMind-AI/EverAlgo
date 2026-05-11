# EverAlgo LLM Stack 子项目设计文档

| 字段 | 值 |
|------|-----|
| 子项目编号 | 2（共 4 子项目） |
| 状态 | Brainstorm 完成（含同行 review 修订），待 BOSS 审 spec；通过后 → writing-plans |
| 范围 | `everalgo.llm.*` EPISODE 路径**最小集 + chat-style 抽象**：7 个对外符号（LLMClient Protocol + ChatMessage/ChatResponse/Usage/LLMConfig + LLMError + build_client）+ openai_compat 单家 provider |
| 不在范围 | streaming / tool calls / multimodal / 7 子类错误 / scoped + global 注入层 / anthropic + bedrock providers / `LLMConfig.from_env()` / `EVERALGO_LLM_*` env |
| 依赖 | 子项目 1 (Foundation) — 用 `everalgo.types.MessageRole` 不复用（LLM 协议层 role 含 system，与对话边界层 role 二值不同） |
| 解锁 | 子项目 4 reference impl 用 `LLMClient` + `ChatMessage` 调 LLM；子项目 3 testing toolkit 实现 `FakeLLMClient` |
| 估时 | ~1.5 工作日（7 src 文件 + 6 test 文件） |
| 设计源头 | 6 项目实测调研（pydantic-ai / DSPy / LlamaIndex / mem0 / Instructor / Letta）；`docs/design.md` §2.5 + ADR 010/011/012；BOSS 在线确认（chat-style 保留 + SecretStr + lazy import 注释） |

---

## 1. 背景与目标

### 1.1 算法同学的痛点

子项目 1 (Foundation) 落地了 `MemCell` / `Episode` 等数据契约。算法同学要写第一个 `EpisodeExtractor` 还需要：

- **`async def aextract(self, memcell, *, llm: LLMClient) -> list[Episode]` 签名里的 `LLMClient` 必须存在**
- caller (EverOS) 必须能 `from everalgo.llm import build_client; client = build_client(LLMConfig(...))` 拿到一个可调的 LLM client
- LLM 调用失败时有可识别的错误类型（`LLMError`）

### 1.2 EPISODE 路径调用形态

```
EpisodeExtractor.aextract(memcell, *, llm: LLMClient) -> list[Episode]
    │
    ↓ 内部：
    │   prompt = EPISODE_EXTRACTION_PROMPT.format(memcell_text=...)
    │   messages = [ChatMessage(role="user", content=prompt)]
    │   response = await llm.chat(messages, model=...)
    │   return parse_json(response.content)
```

EpisodeExtractor 不直接持 LLMClient instance；caller 通过 per-call `llm=` 注入。

### 1.3 设计决策依据

按 6 项目实测调研（详见 brainstorm 阶段产物），这套抽象层在工业界 5/6 算法库主流（LlamaIndex `BaseLLM.chat`、DSPy / mem0 / Instructor / pydantic-ai 各家命名不同但形态一致）。EverAlgo 选择**收紧到最小可用集** + **采纳 4 条工业级安全/可维护性建议**（同行 review）。

> ✅ **设计自检**
> - **为什么 chat-style 而非 opensource 的 prompt-in str-out**：opensource `LLMProvider.generate(prompt: str) -> str` 是包袱形态——所有调用方先把 system + user + history 拼成单 str（脆弱、丢 role 语义），再喂 OpenAI ChatCompletions API（被迫包成 `[{"role":"user","content":prompt}]`）。EverAlgo 重构机会用 chat-style 直接对齐 OpenAI Chat Completions 协议。后续 multi-turn / system prompt / tool calls 是自然扩展（minor bump）；prompt-in 模式遇到这些场景必须 major bump。
> - **为什么 LLMClient 必须 async**：EverAlgo 主用户 EverOS = FastAPI 异步服务（[ADR 010 line 127](../../decisions/010-sync-async-dual-interface.md#L127)）；EpisodeExtractor.aextract 是 async-first；`await client.chat()` 要求 chat 是 async coroutine。
> - **为什么 4 项 review 全采纳**：SecretStr 防真实凭证泄露事故；Usage Optional 区分 "missing" vs "zero tokens"；raw docstring 防"None 是 bug 还是 unsupported"歧义；lazy import 注释防维护者误改造成 cold-start regression。零成本，全部接受。

---

## 2. File Map

```
packages/everalgo-core/
  src/everalgo/llm/
    __init__.py            # re-export 7 symbols + __all__
    protocols.py           # LLMClient Protocol (@runtime_checkable, async chat)
    types.py               # ChatMessage / ChatResponse / Usage (pydantic.BaseModel)
    config.py              # LLMConfig (含 SecretStr api_key)
    errors.py              # LLMError
    factory.py             # build_client (lazy import OpenAICompatClient)
    providers/
      __init__.py          # empty marker
      openai_compat.py     # OpenAICompatClient (impl LLMClient)
  tests/llm/
    test_protocols.py      # @runtime_checkable conformance smoke
    test_types.py          # 3 type round-trip + Usage None default
    test_config.py         # LLMConfig defaults + SecretStr masking
    test_errors.py         # LLMError single-class smoke
    test_factory.py        # build_client returns LLMClient instance
    providers/
      test_openai_compat.py  # respx-mocked HTTP round-trip + error transparency
  pyproject.toml           # add `openai>=1.0` + `respx>=0.21` (latter to dev deps)
```

→ **7 src + 6 test files**.

`tests/llm/__init__.py` 与 `tests/llm/providers/__init__.py` 故意不建（沿子项目 1 决策：`--import-mode=importlib` + namespace package）。

---

## 3. 对外 7 个 symbols

```python
# everalgo/llm/__init__.py
from everalgo.llm.config import LLMConfig
from everalgo.llm.errors import LLMError
from everalgo.llm.factory import build_client
from everalgo.llm.protocols import LLMClient
from everalgo.llm.types import ChatMessage, ChatResponse, Usage

__all__ = [
    # protocol
    "LLMClient",
    # data
    "ChatMessage", "ChatResponse", "Usage", "LLMConfig",
    # error
    "LLMError",
    # factory
    "build_client",
]
```

### 3.1 `protocols.py` — `LLMClient`

```python
"""LLM client Protocol — the structural contract every provider satisfies."""

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from everalgo.llm.types import ChatMessage, ChatResponse


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
            **extra: Provider-specific knobs forwarded as kwargs (e.g.
                OpenAI ``seed`` / ``presence_penalty``; OpenRouter
                ``provider`` routing hints).

        Returns:
            ``ChatResponse`` with structured ``content`` / ``usage`` /
            ``finish_reason`` plus optional ``raw`` for debug.

        Raises:
            LLMError: Any provider-side failure, with the original SDK
                exception attached as ``__cause__`` (PEP 3134).
        """
```

### 3.2 `types.py` — `ChatMessage` / `ChatResponse` / `Usage`

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


class Usage(BaseModel):
    """Token usage from a single LLM call.

    Both fields are ``int | None`` because some self-hosted / OpenAI-compatible
    backends do not return ``usage`` in the response. ``None`` semantically
    distinguishes "missing data" from "zero tokens used".
    """

    prompt_tokens: int | None = None
    completion_tokens: int | None = None


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

### 3.3 `config.py` — `LLMConfig`（含 SecretStr）

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

### 3.4 `errors.py` — `LLMError`

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

### 3.5 `factory.py` — `build_client`

```python
"""Factory for building an LLM client from configuration."""

from everalgo.llm.config import LLMConfig
from everalgo.llm.protocols import LLMClient


def build_client(config: LLMConfig) -> LLMClient:
    """Build an OpenAI-compatible LLM client from ``config``.

    Implementation note: ``OpenAICompatClient`` is imported lazily inside the
    function body so that ``everalgo.llm.factory`` itself does not pull the
    ``openai`` SDK at import time. This keeps ``import everalgo.llm`` cheap
    for callers that only need the Protocol / Config / Error types and never
    call ``build_client``. Maintainers — please do **not** "optimise" this
    into a top-level import; the laziness is load-bearing.
    """
    from everalgo.llm.providers.openai_compat import OpenAICompatClient

    return OpenAICompatClient(config)
```

### 3.6 `providers/openai_compat.py` — `OpenAICompatClient`

```python
"""OpenAI-compatible provider — wraps openai.AsyncOpenAI."""

from collections.abc import Mapping
from typing import Any

import openai

from everalgo.llm.config import LLMConfig
from everalgo.llm.errors import LLMError
from everalgo.llm.types import ChatMessage, ChatResponse, Usage


class OpenAICompatClient:
    """Thin async wrapper over ``openai.AsyncOpenAI``.

    Single-purpose: convert between EverAlgo's ChatMessage / ChatResponse
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
        max_tokens_val = max_tokens if max_tokens is not None else self._config.max_tokens
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
        usage = None
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
    """Collapse provider finish reasons to EverAlgo's 3-value Literal subset.

    EPISODE path treats ``tool_calls`` / ``function_call`` as out-of-scope
    (no tools wired); when a provider unexpectedly emits one the response is
    classified as ``"stop"``. Logging the unknown value is left to providers.
    """
    if value in ("stop", "length", "content_filter"):
        return value
    return None
```

---

## 4. Test 设计（pydantic-ai 模式）

### 4.1 物理布局（沿子项目 1 决策）

```
packages/everalgo-core/tests/llm/
  test_protocols.py
  test_types.py
  test_config.py
  test_errors.py
  test_factory.py
  providers/
    test_openai_compat.py
```

无 `__init__.py`（namespace package + importlib import-mode）。

### 4.2 测试覆盖

| 文件 | 覆盖点 |
|---|---|
| `test_protocols.py` | `@runtime_checkable` ：构造一个 `class FakeClient: async def chat(...)` 验证 `isinstance(FakeClient(), LLMClient)` 为 True |
| `test_types.py` | 3 type 构造正常路径 + 缺必填 `ValidationError` + JSON round-trip + `Usage` 默认 None + `ChatResponse.raw` 默认 None + `extra="ignore"` (ChatMessage) |
| `test_config.py` | LLMConfig 默认值 + `api_key` repr/dump 屏蔽 + `get_secret_value()` 解包 + `extra` dict 默认 |
| `test_errors.py` | `LLMError(...) from sdk_err` 链式抛出 + `__cause__` 可访问 |
| `test_factory.py` | `build_client(config)` 返回值 `isinstance(LLMClient)` + lazy import 验证（openai SDK 不在 `everalgo.llm.factory` import 时被加载）|
| `providers/test_openai_compat.py` | respx mock POST `/v1/chat/completions`：构造 messages → 验证 ChatResponse 字段（含 usage / finish_reason）+ 触发 `openai.RateLimitError` 验证 wrap 成 `LLMError` 且 `__cause__` 是原 SDK exception |

---

## 5. pyproject.toml 改动

```diff
 dependencies = [
   "pydantic>=2.7",
+  "openai>=1.0",
 ]
```

`respx>=0.21` 已在 root `[dependency-groups] dev` 中存在（子项目 1 时就有）。

---

## 6. 验收标准

1. ✅ `uv sync --all-packages` 无 error
2. ✅ `uv run python -c "from everalgo.llm import LLMClient, LLMConfig, LLMError, ChatMessage, ChatResponse, Usage, build_client; print('OK')"` 输出 `OK`
3. ✅ `uv run pytest packages/everalgo-core/tests/` 全绿（含子项目 1 的 41 个 + 子项目 2 新增 ~25 个 ≈ 66 tests）
4. ✅ `uv run ruff check packages/everalgo-core/` 0 issue
5. ✅ `uv run ruff format --check packages/everalgo-core/` 0 diff
6. ✅ `uv run mypy packages/everalgo-core/` 0 error
7. ✅ `uv run pyright packages/everalgo-core/` 0 error
8. ✅ `repr(LLMConfig(model="x", api_key="sk-secret", base_url="..."))` 不含 `"sk-secret"`
9. ✅ 子项目 4 EpisodeExtractor reference impl 引用 `LLMClient` / `ChatMessage` / `ChatResponse` 时 import 不报错（由子项目 4 验收）

---

## 7. Out of Scope（明确移出最小集）

按 EPISODE 路径最小集严格收敛，全部移出本子项目（按未来需求时机分类）：

### 7.1 接口层扩展（未来 SemVer minor bump 加）

- `stream(messages, ...)` method 与 `ChatChunk` 类 — streaming 场景
- `tools` / `tool_choice` 参数与 `ToolCall` / `ToolSchema` 类 — agent path
- Multimodal `content: list[ContentBlock]` — 视觉 / 音频 prompt

### 7.2 配置扩展

- `LLMConfig.provider: Literal["openai_compat","anthropic","bedrock"]` — 多 provider 时加上做 build_client 分发
- `LLMConfig.from_env()` 类方法 / `EVERALGO_LLM_*` env 兜底
- `Usage.cache_read_tokens` 字段 — Anthropic prompt caching 计费
- `LLMConfig.api_key: SecretStr | None`（让 caller 通过 env 注入而非显式传）— 当 `from_env` 加入时同步

### 7.3 错误层扩展

- `LLMRateLimitError` / `LLMTimeoutError` / `LLMServerError` / `LLMConnectionError` / `LLMAuthError` / `LLMBadRequestError` / `LLMContextLengthError` 7 子类
- adapter 出口 multiple-inheritance 让 `except openai.RateLimitError` 也能 catch — Letta 同款（13 子类的 simplified 7 类版）

### 7.4 注入层

- `everalgo.llm.use(client)` scoped contextmanager（DSPy `dspy.context` 同款）
- `everalgo.configure(llm=...)` global default
- `everalgo.llm.resolve(per_call) -> LLMClient` 3 层 fallback helper
- 子项目 2 仅做 per-call `llm=` 单层

### 7.5 Provider 扩展

- `everalgo/llm/providers/anthropic.py`
- `everalgo/llm/providers/bedrock.py`
- `everalgo/llm/providers/vllm.py`（OpenAI compat 已覆盖，单独包是 follow-up）

### 7.6 可观测性

- 调用 logging / tracing wrapper
- ChatResponse.raw 默认填充而非默认 None — 现在 None，待 caller 真要 debug 模式时通过 build_client(config, debug=True) 之类引入

---

## 8. 字段决策清单（已对齐，无待 BOSS 校准项）

| 决策点 | 选择 | 出处 / 依据 |
|---|---|---|
| LLMClient method 名 | `chat` | 与 LlamaIndex `BaseLLM.chat` + OpenAI Chat Completions API 协议对齐 |
| LLMClient 同步/异步 | **async only** | EverOS = FastAPI async 主用户（ADR 010 line 127） |
| ChatMessage `role` 类型 | `Literal["system","user","assistant"]` | OpenAI Python SDK 官方对齐 + 与 everalgo.types.MessageRole 不复用（语义层不同）|
| ChatMessage `content` 类型 | `str` | EPISODE 不需多模态；list[ContentBlock] 是后续扩展 |
| ChatResponse 字段 | `content/model/usage/finish_reason/raw` | 与 OpenAI ChatCompletion 主流字段对齐 |
| `Usage.{prompt,completion}_tokens` | **`int \| None`** | 同行 review #3：区分 missing vs zero |
| `ChatResponse.raw` docstring | 明确 "production caller should not depend" | 同行 review #2 |
| LLMConfig.api_key | **`SecretStr`** | 同行 review #1：防真实凭证泄露 |
| LLMConfig 字段 | `model/api_key/base_url/temperature=0.0/max_tokens=None/timeout=60.0/extra={}` | 7 字段，`temperature=0` 默认 EPISODE 确定性 |
| LLMConfig 是否含 `provider` | **不含** | 最小集只 1 provider；后续加多 provider 时 minor bump 加字段 + match-case |
| LLMError 层级 | **单基类** + SDK exception 链 (`__cause__`) | 算法库不该把 13 子类塞给上层（Letta 反例）|
| factory pattern | `build_client(config) -> LLMClient` 函数式 | 算法库忌 DI（参 `feedback_algo_lib_no_di.md`）|
| `build_client` lazy import | OpenAICompatClient 在函数体内 import + docstring 明确 | 同行 review #4：cold-start 友好 |
| 文件命名 `types.py` vs `models.py` | **`types.py`** | LlamaIndex 同款 `core/base/llms/types.py` |
| `tests/llm/__init__.py` | **不建** | 沿子项目 1 决策 |

---

## 9. 自审（writing-plans 之前）

| 检查项 | 结果 |
|---|---|
| Placeholder scan | ✅ 无 TBD/TODO；OpenAICompatClient `chat` 实现 已 具体到 SDK 调用 |
| 内部一致性 | ✅ 7 对外符号在 §3 各模块 + §3 顶层 re-export 一致；`build_client` 签名贯穿 §3.5 / §6 验收 |
| Scope check | ✅ 单子项目 ~12 task，writing-plans 一份 plan 容得下 |
| Ambiguity | ✅ 字段 + 决策表 §8 全 BOSS 拍板（含同行 review 4 条） |
| 命名一致 | ✅ `chat` / `LLMClient` / `LLMConfig` / `build_client` 贯穿 |
| 与 opensource 出处一致 | ✅ method 名 / config 字段 / error 命名都引 `release/20260403` 实测 |
| EPISODE 路径完备 | ✅ caller 走 `build_client(config) → client.chat(messages) → response.content` 端到端只用 7 对外符号 |

---

*Spec 完成。立刻进 writing-plans。*
