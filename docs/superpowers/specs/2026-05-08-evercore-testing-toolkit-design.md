# EverCore Testing Toolkit 子项目设计文档

> 本文档是 **子项目 3 / 4** 的设计 spec。
> 上游：子项目 1 (Foundation, 已完成) + 子项目 2 (LLM Stack, 已完成)。
> 下游：子项目 4 (Reference impl + CI) 将依赖本子项目交付的 `FakeLLMClient` 和 `assert_episode_shape` 写端到端单测。
>
> 落盘日期：2026-05-08
> 设计依据：所有公开符号选型均 cite 至少 2 个同定位明星项目代码（pydantic-ai / DSPy / LangChain core / LangChain langchain_tests / instructor），见 §7。

**Goal**：为算法同学交付**单元测试场景**下「不打真 LLM、不依赖真后端」的最小工具集 —— 一个 `LLMClient` Protocol 的内存版实现 + 一个 `Episode` 结构断言 helper。

**Architecture**：两个独立 helper 模块（`fake_llm.py` + `assertions.py`），无内部状态共享，无依赖耦合。归属 `evercore-core` 内的 `evercore.testing` 子包（与 `numpy.testing` / `torch.testing` 同模式，见 ADR 005）。

**Scope**：3 个公开符号（`FakeLLMClient`、`CallRecord`、`assert_episode_shape`），两个测试文件。**不**ship pytest fixture，**不**ship factory builder（依据：见 §7）。

---

## 1. 背景与目标

### 1.1 算法同学的痛点（沿子项目 1+2 同语境）

子项目 4 (reference impl) 会做`boundary.detect → episode.aextract` 端到端测试 —— 在一次端到端测试里，**两次 LLM 调用要返回不同结构的 JSON**：

- 第 1 次（boundary detect）：返回 `{"split_at": 5, "boundary_reason": "topic_shift"}` 之类
- 第 2 次（episode extract）：返回 `{"id": "ep_001", "owner_id": "u1", "episode": "...", ...}` 之类

如果只提供 scripted list（按调用顺序 pop），算法同学就要按顺序排列两条响应；如果提供 callable handler（按 prompt 内容路由），算法同学就能写「检测到 boundary prompt 就返 boundary JSON、检测到 extract prompt 就返 episode JSON」的条件分支。两种模式各有 80%/20% 场景，需求互补。

`assert_episode_shape` 解决另一个问题：LLM 返回的 JSON 经 `Episode.model_validate(json)` 之后通过了 pydantic 类型层校验，但**业务上还可能空洞**（`episode=""`、`timestamp=0`、`parent_type` 写错了）。pydantic 不会拦这些，单测的人工 `assert ep.episode != ""` 字眼分散且容易漏。

### 1.2 与 LangChain `langchain_tests` 的定位区分

LangChain 有一个独立 dist `langchain_tests`（contains base test classes for third-party providers self-conformance）—— 那是「provider 接入合规套件」（`integration_tests/` + `unit_tests/` 子目录），不是「业务用户写自己单测时使用的工具集」。

`evercore.testing` 对标的是 LangChain core 的 `fake_chat_models.py`（in-memory fake LLM 类）+ pydantic-ai 的 `models/test.py`（in-memory `TestModel`），**不是** `langchain_tests`。这是子项目 3 与子项目 4 端到端 CI 套件的边界。

### 1.3 设计决策依据

按 BOSS feedback「测试环节决策必须有明星项目代码引用」（见 memory `feedback_test_decisions_need_star_project_evidence.md`），本 spec 的所有「是否 ship X」决策均在 §7 表格中给出对应 4-5 个明星项目的代码引用作支撑。

---

## 2. File Map

```
packages/evercore-core/src/evercore/testing/
├── __init__.py        # re-export 3 个公开符号
├── assertions.py      # assert_episode_shape
└── fake_llm.py        # FakeLLMClient + CallRecord

packages/evercore-core/tests/testing/
├── test_assertions.py
└── test_fake_llm.py
```

无新增 dependency（`fake_llm.py` 仅依赖 stdlib `inspect` + `evercore.llm.types` + `evercore.llm.protocols`；`assertions.py` 仅依赖 `evercore.types`）。

---

## 3. 对外 3 个 symbols

```python
# evercore/testing/__init__.py
from evercore.testing.assertions import assert_episode_shape
from evercore.testing.fake_llm import CallRecord, FakeLLMClient

__all__ = [
    "CallRecord",
    "FakeLLMClient",
    "assert_episode_shape",
]
```

> ✅ **设计自检：CallRecord 公开**
> - **为什么这样设计**：`FakeLLMClient.calls[0].messages` 在测试里做断言时类型必须可见，否则用户只能拿到 `Any`
> - **规范依据**：unittest.mock 的 `call` 对象同样是公开符号（`mock.call_args_list` 元素类型）
> - **备选方案**：把 `CallRecord` 设为内部，`calls` 返 `list[Any]` —— 否决，类型隐藏让测试代码失去 IDE 补全

---

### 3.1 `fake_llm.py` — `FakeLLMClient` + `CallRecord`

```python
"""In-memory LLMClient double for unit tests."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from evercore.llm.types import ChatMessage, ChatResponse


class CallRecord(BaseModel):
    """Single recorded ``FakeLLMClient.chat`` invocation.

    Exposed publicly so tests can assert on captured arguments via
    ``client.calls[0].messages == [...]`` with full IDE type-checking.
    """

    messages: list[ChatMessage]
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    response_format: Mapping[str, Any] | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)


_HandlerReturn = ChatResponse | Awaitable[ChatResponse]
_Handler = Callable[..., _HandlerReturn]


class FakeLLMClient:
    """In-memory ``LLMClient`` Protocol implementation for unit tests.

    Two construction modes (mutually exclusive — choose at construction):

    1. **Scripted list** (``responses=...``): pre-canned responses, popped in
       call order. Exhaustion raises ``RuntimeError`` (fail-fast — see §4).
       Each element may be either:

       - ``str`` — auto-wrapped as
         ``ChatResponse(content=<str>, model="fake", usage=None,
         finish_reason="stop", raw=None)``
       - ``ChatResponse`` — returned as-is (lets the test precisely control
         ``usage`` / ``finish_reason``).

    2. **Callable handler** (``handler=...``): receives ``(messages, **kwargs)``
       (mirroring the ``chat`` signature), returns ``ChatResponse`` or
       ``Awaitable[ChatResponse]``. Useful for prompt-conditional branching
       (e.g. boundary call vs episode call returning different JSON in a
       single end-to-end test).

    The two modes are mutually exclusive: passing both or neither raises
    ``ValueError`` at construction time (avoids hidden precedence semantics).
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

    @property
    def calls(self) -> list[CallRecord]:
        """All recorded ``chat`` invocations, in call order."""
        return list(self._calls)  # defensive copy

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
            return await _invoke_handler(
                self._handler,
                messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                **extra,
            )
        # scripted-list mode
        assert self._responses is not None  # narrowed by __init__ invariant
        if not self._responses:
            raise RuntimeError(
                f"FakeLLMClient script exhausted "
                f"(used {self._initial_response_count} of "
                f"{self._initial_response_count} responses)"
            )
        return self._responses.pop(0)


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

**关键决策**：

> ✅ **设计自检：双模 + 互斥（`responses` XOR `handler`）**
> - **为什么这样设计**：scripted list 是 80% 单测最短路径；callable handler 解决子项目 4 端到端测试（boundary→episode 两次 LLM 调用返回不同 JSON）的 prompt-conditional 分支需求。互斥校验避免「两个都传，谁优先」的隐式语义
> - **规范依据**：LangChain `FakeMessagesListChatModel`（list 模式，`fake_chat_models.py:21-52`）+ `GenericFakeChatModel`（Iterator 钩子模式，`fake_chat_models.py:227-264`）拆成两个类。我们合一是因为 EverCore Protocol 只有一个 `chat` 方法、没必要拆类
> - **备选方案**：仅 list（YAGNI 极限） / 仅 handler（最简单测试也要写 lambda） —— 都被否

> ✅ **设计自检：耗尽 raise，不循环不兜底**
> - **为什么这样设计**：fail-fast 暴露 off-by-one bug；不学 LangChain 循环（line 41-45，掩盖 N+1 调用）；不学 DSPy `"No more responses"` 兜底字符串（line 130，错误现场离调用点远）
> - **规范依据**：LangChain `FakeMessagesListChatModel.responses[i]` 循环 vs DSPy `next(self.answers, default)` 兜底，两种模式都被本设计否决；本设计选择更严格的 raise 模式
> - **备选方案**：循环 / 兜底 —— 否

> ✅ **设计自检：`str` 元素自动包装为 `ChatResponse`**
> - **为什么这样设计**：80% 测试只关心 `content` 文本，不关心 `usage` / `finish_reason`；让用户写 `FakeLLMClient(["hello"])` 而不是 `FakeLLMClient([ChatResponse(content="hello", model="fake", finish_reason="stop")])`
> - **规范依据**：DSPy `DummyLM` 接受 `list[dict]` 自动展开成 LM 内部格式（`dummies.py:71-85`，list mode），同思路
> - **备选方案**：只接受 `ChatResponse` —— 否，每条都要写 6 个字段过于冗余

> ✅ **设计自检：handler 同步/异步双兼容（`inspect.isawaitable` 判断）**
> - **为什么这样设计**：算法同学常用 `lambda messages, **kw: ChatResponse(...)`（同步）；端到端测试可能用 `async def handler(...)`（异步）。两种场景都允许，避免强制 async-only 让简单测试啰嗦
> - **规范依据**：LangChain `GenericFakeChatModel` 通过 base class 同时支持 `_generate` 和 `_agenerate`（同思路）；asyncio 文档推荐 `inspect.isawaitable` 判定 awaitable
> - **备选方案**：仅 sync handler（端到端 async 测试不能用） / 仅 async handler（简单测试啰嗦） —— 都被否

---

### 3.2 `assertions.py` — `assert_episode_shape`

```python
"""Structural assertions for memory types."""

from __future__ import annotations

from typing import Any

from evercore.types import Episode


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

       a. ``episode`` is a non-empty string (LLM may emit ``""`` when prompt
          fails)
       b. ``timestamp > 0`` (Unix epoch ms; ``0`` or negative = bug)
       c. ``parent_type == "memcell"`` (EPISODE path only consumes MemCell)
       d. ``parent_id`` is a non-empty string (data lineage anchor required)

    Args:
        value: ``dict`` (parsed via ``Episode.model_validate``) or already-
            parsed ``Episode``.

    Returns:
        The validated ``Episode`` instance, so callers can chain further
        assertions::

            ep = assert_episode_shape(json_dict)
            assert "Alice" in ep.episode

    Raises:
        AssertionError: If any business invariant fails. The message names
            the failed invariant (e.g. ``"Episode.episode is empty"``).
        pydantic.ValidationError: If type-level validation fails. Re-raised
            unmodified.

    Examples:
        >>> ep = assert_episode_shape({
        ...     "id": "ep_001",
        ...     "owner_id": "u1",
        ...     "episode": "Alice scheduled the meeting",
        ...     "timestamp": 1700000000000,
        ...     "parent_id": "mc_001",
        ... })
        >>> ep.episode
        'Alice scheduled the meeting'
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

**关键决策**：

> ✅ **设计自检：返回 `Episode` 而不是 `None`**
> - **为什么这样设计**：链式断言友好（`ep = assert_episode_shape(d); assert "Alice" in ep.episode`）；如果输入是 dict，调用者拿到解析后实例避免重复 `model_validate`
> - **规范依据**：pytest 圈常见模式（`response = assert_response_ok(resp); assert response.json()["x"] == y`）；pydantic 自身的 `model_validate` 也返实例
> - **备选方案**：返 `None`（标准 `assert` 风格） —— 否，链式不友好

> ✅ **设计自检：`ValidationError` 直接 re-raise，不包装成 AssertionError**
> - **为什么这样设计**：pydantic 的错误消息已经是「字段名 + 期望类型 + 实际值」的标准格式，比再写一遍更清晰；包装成 AssertionError 反而丢失字段路径信息（嵌套对象错误尤其明显）
> - **规范依据**：pytest 默认让原始异常透出（`pytest.raises` 也是匹配原始类型）
> - **备选方案**：catch ValidationError 包装成 AssertionError —— 否，丢信息

> ✅ **设计自检：4 项业务不变量，不加 `id` 格式校验**
> - **为什么这样设计**：子项目 1 的 `Episode.id` 故意定义成自由 `str`（让上游 EverOS 决定是 UUIDv4 / hash / 业务 key）；强制 UUID 格式会让「id 由调用方注入」的场景假阳报错
> - **规范依据**：见 `docs/superpowers/specs/2026-05-07-evercore-foundation-design.md` 关于 `id: str` 的注释
> - **备选方案**：加 `assert isinstance(id, str) and len(id) >= 8`（最弱形态校验） —— 否，仍可能假阳

---

## 4. 错误处理

| 失败模式 | 抛出类型 | 消息形态 |
|---|---|---|
| 构造时 `responses` + `handler` 都传或都没传 | `ValueError` | `Provide exactly one of \`responses\` or \`handler\`` |
| `responses` 元素既不是 `str` 也不是 `ChatResponse` | `TypeError` | `FakeLLMClient \`responses\` must contain str or ChatResponse, got <type>` |
| Scripted list 耗尽 | `RuntimeError` | `FakeLLMClient script exhausted (used N of N responses)` |
| handler 返回值既非 `ChatResponse` 也非 `Awaitable[ChatResponse]` | `TypeError` | `FakeLLMClient handler must return ChatResponse or Awaitable[ChatResponse], got <type>` |
| `assert_episode_shape` 业务不变量违反 | `AssertionError` | `Episode.episode is empty` 等具体消息 |
| `assert_episode_shape` 类型层校验失败 | `pydantic.ValidationError` | pydantic 原始消息（不包装） |

**为什么不定义 `FakeLLMError` 自定义异常基类**：testing 工具的失败应该让用户**第一时间分辨是 stdlib 哪种 bug**（用错 API 用 `ValueError`、类型不对用 `TypeError`、状态错乱用 `RuntimeError`），自定义异常反而让用户多查一层。LangChain `FakeListChatModelError`（`fake_chat_models.py` line 55）也仅在内部 `_call` 错误传递时使用，不是 testing API 失败时使用。

---

## 5. 测试矩阵

> 单测文档遵循 BOSS 既定格式（per memory `feedback_unittest_doc_format.md`）：覆盖场景按被测文件分组、测试函数名描述行为而非实现。

### 5.1 `test_fake_llm.py`

按被测类分组：

**`TestScriptedList`**（responses 模式）
- `test_str_element_wrapped_to_default_chat_response` —— `str` 自动包装、`model="fake"` / `finish_reason="stop"` / `usage=None`
- `test_chat_response_element_passed_through_unchanged` —— `ChatResponse` 实例原样返回，`usage` 不被篡改
- `test_responses_popped_in_call_order` —— 多条响应严格按调用顺序返回
- `test_exhausted_script_raises_runtime_error` —— 耗尽后下一次 `chat()` raise `RuntimeError`，消息含 `(used N of N)`
- `test_invalid_element_type_raises_type_error` —— `responses=[123]` 在构造时 raise `TypeError`

**`TestCallableHandler`**（handler 模式）
- `test_sync_handler_invoked_correctly` —— `lambda messages, **kw: ChatResponse(...)` 同步返回
- `test_async_handler_awaited_correctly` —— `async def handler(...)` 被 await
- `test_handler_receives_messages_and_kwargs` —— handler 看到 `messages` + `model` + `temperature` + `**extra`
- `test_handler_wrong_return_type_raises_type_error` —— handler 返 `dict` raise `TypeError` 含「got dict」

**`TestConstructorValidation`**
- `test_both_responses_and_handler_raises_value_error`
- `test_neither_responses_nor_handler_raises_value_error`

**`TestRecording`**
- `test_call_count_increments_per_invocation`
- `test_calls_property_records_messages_and_kwargs`
- `test_calls_property_returns_defensive_copy` —— 用户 mutate 返回的 list 不影响内部状态

**`TestProtocolConformance`**
- `test_isinstance_of_LLMClient` —— `isinstance(client, LLMClient)` 为 True（依赖 `@runtime_checkable`）

### 5.2 `test_assertions.py`

按被测函数分组：

**`TestAssertEpisodeShape`**
- `test_dict_input_parsed_and_validated` —— 合法 dict 输入返回 `Episode` 实例
- `test_episode_input_passed_through` —— `Episode` 实例输入直接返回（`is` 同一实例）
- `test_chained_assertion_uses_returned_episode` —— 演示链式断言成功
- `test_missing_required_field_raises_validation_error` —— 缺 `parent_id` 字段 raise pydantic `ValidationError`（不包装）
- `test_empty_episode_string_raises_assertion_error` —— `episode=""` raise AssertionError 含 `is empty`
- `test_zero_timestamp_raises_assertion_error` —— `timestamp=0` raise，消息含 `must be positive`
- `test_negative_timestamp_raises_assertion_error` —— `timestamp=-1` raise
- `test_wrong_parent_type_raises_assertion_error` —— `parent_type="raw_message"` raise，消息含 `must be 'memcell'`
- `test_empty_parent_id_raises_assertion_error` —— `parent_id=""` raise

### 5.3 物理布局

```
packages/evercore-core/tests/testing/
├── test_fake_llm.py
└── test_assertions.py
```

无 `__init__.py`（沿子项目 1+2 已确立的 `--import-mode=importlib` 决策）。

无新增 `conftest.py` fixture（沿 §7 行业证据）。`packages/evercore-core/tests/conftest.py` 现有占位继续保留（注释已说明本子项目交付不引入 fixture）；如果 BOSS 想清理「placeholder」字眼，作为本子项目尾声小修。

---

## 6. 验收标准

- 3 个公开符号在 `evercore.testing` 正确导出，`from evercore.testing import FakeLLMClient, CallRecord, assert_episode_shape` 可用
- `FakeLLMClient` 通过 `isinstance(client, LLMClient)` 检测（runtime_checkable Protocol）
- 全部测试 PASS：`uv run pytest packages/evercore-core/tests/testing/ -v`
- mypy 严格模式通过：`uv run mypy packages/evercore-core/src/evercore/testing/`
- ruff 全 clean：`uv run ruff check packages/evercore-core/`
- 不引入新 dependency（仅 stdlib + 已有的 evercore.llm / evercore.types）
- AGENTS.md §7 step 6 + §9 提到的 2 个公开符号 100% 兑现

---

## 7. Out of Scope（明确移出最小集，附依据）

### 7.1 不 ship pytest fixture

**依据**（同定位项目 4 票全否）：

| 项目 | 文件 | 是否 ship pytest fixture | 引用 |
|---|---|---|---|
| pydantic-ai | `pydantic_ai_slim/pydantic_ai/models/test.py` | 无（无 `pytest` import） | 整个文件无 `@pytest.fixture` |
| DSPy | `dspy/utils/dummies.py` + `pyproject.toml` | 无（pyproject 无 `[project.entry-points.pytest11]`） | DummyLM 仅是普通 class |
| LangChain core | `libs/core/langchain_core/language_models/fake_chat_models.py` | 无（不导入 pytest） | 6 个 fake class，无 fixture |
| instructor | `pyproject.toml` | 无（无 `pytest11` entry_point） | testing 子包不存在 |

唯一例外是 LangChain `langchain_tests` 独立 dist 内有 `conftest.py`，但定位是「provider conformance test suite」，与本子项目目标不符（详见 §1.2）。

### 7.2 不 ship `make_episode` / `make_chat_response` factory

**依据**（5 票全否）：上述 4 项目 + LangChain `langchain_tests` 都没有为 domain object 提供 factory builder。统一用 pydantic 模型直接构造（`Episode(id="...", ...)` / `ChatResponse(content="...", ...)`），子项目 1 的 `extra="allow"` / `extra="ignore"` 已经处理了「字段宽容」需求，factory 没有补充价值。

### 7.3 流式 / Tool calls / multimodal

`LLMClient.chat` Protocol 本身不带流式；tool calls 与 multimodal 都在 EPISODE 路径之外（见子项目 2 spec §7.1）。本子项目同样不实现，未来 Protocol 扩展时再 SemVer minor bump 增加 `FakeLLMClient.stream` / `FakeLLMClient.tool_calls` 等。

### 7.4 其他 memory type 的 assert 函数

`assert_profile_shape` / `assert_atomic_fact_shape` / `assert_foresight_shape` / `assert_agent_case_shape` / `assert_agent_skill_shape` / `assert_knowledge_shape` 都不在本子项目交付。子项目 4 reference impl 完成 EPISODE 端到端后，未来按 SemVer minor bump 逐一增量引入（届时新增的 helper 必须与 `assert_episode_shape` 的契约一致：`dict | T` 输入、链式返回、5 项以内业务不变量）。

### 7.5 Episode `id` 格式校验

子项目 1 spec 的 `Episode.id: str` 是自由字符串（让上游 EverOS 决定生成策略）。`assert_episode_shape` 不强制 UUID/prefix 格式，避免假阳。

### 7.6 测试录制 / fixture 序列化

`FakeLLMClient` 不支持「录一次真 API 响应、序列化为 fixture、replay」（pytest-recording / VCR.py 模式）。这是 integration test 工具的职责（参考 §1.2，对应 LangChain `langchain_tests` 而不是 `evercore.testing`）。子项目 4 真要打真 API 时可单独引入 respx 或 pytest-recording。

---

## 8. 行业参考（明星项目代码引用汇总）

### 8.1 LLM 客户端 fake 实现的 3 种典型形态

| 项目 | 类名 | 入参形态 | 耗尽行为 | callable 钩子？ | 原生 async？ |
|---|---|---|---|---|---|
| DSPy `DummyLM` | `dspy/utils/dummies.py:71-85` | `list[dict]` 或 `dict[str, dict]` | 兜底 `"No more responses"` (line 130) | 否（list 内部转 iter） | 否（`aforward` 同步包装） |
| pydantic-ai `TestModel` | `pydantic_ai_slim/pydantic_ai/models/test.py:47-112` | seed-driven schema 自动生成 | N/A（无脚本） | 否 | 是（`async def request`） |
| LangChain `FakeMessagesListChatModel` | `langchain_core/language_models/fake_chat_models.py:21-52` | `responses: list[BaseMessage]` | 循环（`i = 0` line 41-45） | 否 | 是（base class） |
| LangChain `GenericFakeChatModel` | `langchain_core/language_models/fake_chat_models.py:227-264` | `messages: Iterator[AIMessage \| str]` | 用户 iterator 抛 `StopIteration` | 是 | 是 |

**EverCore 取舍**：scripted list（≈DSPy + LangChain `FakeMessagesListChatModel`）+ callable handler（≈LangChain `GenericFakeChatModel`）混合，但选择更严格的 raise 耗尽行为（不学循环不学兜底）。

### 8.2 testing 公开 surface 规模对比

| 项目 | testing 模块文件 | 公开符号数 | 含 fixture？ | 含 factory？ |
|---|---|---|---|---|
| pydantic-ai | `models/test.py` | 2（`TestModel`, `TestStreamedResponse`） | ❌ | ❌ |
| DSPy | `utils/dummies.py` | 1（`DummyLM`） | ❌ | ❌ |
| LangChain core | `fake_chat_models.py` | 6（5 fake + 1 error） | ❌ | ❌ |
| instructor | （无 testing 模块） | 0 | ❌ | ❌ |
| **EverCore** | `testing/{fake_llm,assertions}.py` | **3**（`FakeLLMClient`, `CallRecord`, `assert_episode_shape`） | ❌ | ❌ |

EverCore 的 3 个公开符号规模与同定位项目 (1-6 个) 一致。`CallRecord` 之于 `FakeLLMClient` 类似 unittest.mock 的 `call` 之于 `Mock`（recorded call 类型必须公开）。

---

## 9. 字段决策清单（已对齐，无待 BOSS 校准项）

| # | 决策点 | 取值 | 依据 |
|---|---|---|---|
| 1 | 双模 vs 单模 | 双模（list XOR handler） | §3.1 自检 + LangChain 双类前例 |
| 2 | 耗尽行为 | raise `RuntimeError` | §3.1 自检 + 否决 LangChain 循环 / DSPy 兜底 |
| 3 | str 自动包装 | 包装为 `ChatResponse(content=str, model="fake", finish_reason="stop")` | §3.1 自检 + DSPy list mode 类比 |
| 4 | handler async 兼容 | `inspect.isawaitable` 判定 | §3.1 自检 + LangChain `_generate`/`_agenerate` 双兼容 |
| 5 | assertion 返回值 | 返 `Episode` 实例 | §3.2 自检 + 链式断言可读性 |
| 6 | assertion 输入类型 | `dict \| Episode` | BOSS Q3 = B2 |
| 7 | 业务不变量数量 | 4 项（episode / timestamp / parent_type / parent_id） | BOSS Q3 = B + 子项目 1 spec 字段语义 |
| 8 | id 格式校验 | 不做 | 子项目 1 spec 的 `id: str` 自由形态约定 |
| 9 | 公开符号清单 | 3 个（FakeLLMClient + CallRecord + assert_episode_shape） | AGENTS.md §7 step 6 + §9 + CallRecord 类型可见性 |
| 10 | pytest fixture | 不 ship | §7.1 4/4 同定位项目证据 |
| 11 | factory builder | 不 ship | §7.2 5/5 同定位项目证据 |
| 12 | ValidationError 处理 | re-raise（不包装） | §3.2 自检 + pytest 圈惯例 |
| 13 | 自定义 FakeLLMError 基类 | 不定义 | §4 末段 |

---

## 10. 自审（writing-plans 之前）

✅ **Spec coverage**：所有 BOSS 4 个澄清问题（C / A / B+B2 / 仅 2 公开符号→实际 3 个含 CallRecord）的取值都对应到 §3-§5 具体段落
✅ **Placeholder scan**：grep 无 `TODO` / `TBD` / `FIXME`
✅ **Internal consistency**：§2 file map 的 3 个文件名 = §3 公开符号清单覆盖的实现 = §5 测试矩阵的两个 `test_*.py`
✅ **Scope check**：单 implementation plan 可实现（约 8-10 个 TDD 任务）
✅ **Ambiguity check**：handler 返回值的 sync/async 判定明确（`inspect.isawaitable`）；耗尽消息格式精确（`(used N of N)`）；assertion 业务不变量清单 4 项不可增减
✅ **行业依据 cite 完整**：每个「不 ship X」决策都给出 4-5 个明星项目的代码引用支撑

下一步：进入 `superpowers:writing-plans` skill 撰写 implementation plan。
