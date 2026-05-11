# EverAlgo LLM 3 层注入扩展子项目设计文档

> 本文档是 **子项目 2.5 / 4** 的设计 spec（在子项目 2 与子项目 3 之间增量补齐 design.md §2.5 既定的 3 层注入机制）。
> 上游：子项目 2 (LLM Stack, 已完成) + 子项目 3 (Testing Toolkit, 已完成)。
> 下游：子项目 4 (Reference impl + CI) 将依赖本子项目交付的 `everalgo.llm.resolve` 让算子内部按 design.md line 826 形态写「`client = everalgo.llm.resolve(llm)`」。
>
> 落盘日期：2026-05-08
> 设计依据：6 个明星项目代码引用（DSPy / LlamaIndex / instructor / LangChain core / openai-python / pydantic-ai），见 §8。

**Goal**：补齐 design.md §2.5 line 826 / 855-863 既定的 3 层 LLM 注入机制（per-call > scoped > default），以及配套的 `LLMNotConfiguredError` misuse 错误类，让算子内部用单行 `everalgo.llm.resolve(llm)` 解析 LLM client，避免每个用 LLM 的算子重复写 boilerplate。

**Architecture**：在子项目 2 已落地的 `everalgo.llm` 子包内增量补 4 个 module-level 函数（`resolve` / `current` / `use` / `configure`）+ 1 个错误类（`LLMNotConfiguredError`）+ 2 个 module-level 私有 state（`_default` / `_active` ContextVar）。**不**新建源文件 —— 4 个函数 + 2 个 state 全部放在 `__init__.py`（design.md line 786 既定 facade 结构）；`LLMNotConfiguredError` 加进 `errors.py`。**不**做任何 config loader（env / TOML 等归调用方）。**不**支持 reset（`configure` set-once）。**不**做 dual sync/async contextmanager（sync `@contextmanager` 已 async-safe）。

**Scope**：5 个新公开符号（`configure` / `use` / `current` / `resolve` / `LLMNotConfiguredError`），与子项目 2 的 7 个公开符号合计 12 个；约 80 行新增代码 + 约 15 个新测试。

---

## 1. 背景与目标

### 1.1 子项目 2 收敛太窄的发现

子项目 2 (LLM Stack) 设计 spec §3 line 277 把对外 surface 收敛到 7 个符号（`LLMClient` Protocol + `ChatMessage` / `ChatResponse` / `Usage` / `LLMConfig` / `LLMError` / `build_client`），其中只有 `build_client(config)` 是 LLM 实例的获取入口（per-call 注入主路径）。但 design.md line 826 / 833 / 841 明确写出 EverAlgo 算子内部的标准形态：

```python
class EpisodeExtractor:
    async def aextract(
        self, memcell, *, llm: LLMClient | None = None,
    ) -> list[Episode]:
        client = everalgo.llm.resolve(llm)    # 1 行：3 层 fallback + 兜底抛异常
        return await client.chat(messages, ...)
```

`everalgo.llm.resolve(llm)` 对外是 1 行，但内部要做 3 层 fallback：

| 层 | 用法 | 优先级 | 典型场景 |
|----|------|--------|----------|
| **per-call** | `aextract(..., llm=client)` 直接传参 | 最高 | EverOS 单调用按 scene 注入（主路径）|
| **scoped** | `with everalgo.llm.use(client):` contextmanager | 次高 | EverOS pipeline 段批量同 client（减少重复传参）|
| **default** | 启动期 `everalgo.configure(llm=default)` | 兜底 | dev / 测试 / Jupyter / 简单脚本（一次走天下）|

子项目 2 仅落地了 per-call 这一层，scoped 和 default 都未落地。子项目 4 (reference impl) 写 EpisodeExtractor 时若按 design.md line 833 的 1 行写法会立刻失败。

### 1.2 BOSS 决策：子项目 2.5 派生

2026-05-08 brainstorm 期间 BOSS 在子项目 4 Q2 明确说「**3 层 fallback 是 LLM stack 的核心能力，应该归子项目 2，而不是塞到子项目 4 reference impl 里**」。子项目 4 暂停，派生子项目 2.5 单独 brainstorm + spec + plan + SDD。子项目 4 等本子项目完成后重启。

### 1.3 设计哲学：算法库只接受配置

EverAlgo 是算法库（vs 端到端框架），其公开 API 接受**已构造好的配置对象**（`LLMConfig` / `LLMClient`），但**不负责从外部源（env / TOML / yaml / k8s ConfigMap / secret manager / pydantic-settings ...）加载或解析配置**。design.md line 815-822 提到的 `EVERALGO_LLM_*` env vars 是**调用方惯例 hint**，不是 EverAlgo 自带 feature。本子项目仅落地接受配置的 5 个符号，不实现任何 loader。

详见 memory `feedback_everalgo_accepts_config_not_loads.md`（2026-05-08 BOSS 拍板）。

---

## 2. File Map

```
packages/everalgo-core/src/everalgo/llm/
├── __init__.py        # MODIFY: 在现有 re-export 基础上加：
│                      #   - 2 个 module-level 私有 state: _default + _active (ContextVar)
│                      #   - 4 个 module-level 函数: configure / use / current / resolve
│                      #   - 5 个新符号加入 __all__
├── config.py          # 不动
├── errors.py          # MODIFY: 新增 LLMNotConfiguredError(RuntimeError)
├── factory.py         # 不动
├── protocols.py       # 不动
├── types.py           # 不动
└── providers/         # 不动

packages/everalgo-core/tests/llm/
├── test_injection.py  # NEW: 注入机制测试（约 15 个新测试）
└── test_errors.py     # MODIFY: 加 LLMNotConfiguredError 继承链测试
```

**仅 2 个源文件改动 + 1 个新测试文件**（`__init__.py` 大改 + `errors.py` 加 1 个类 + 新建 `test_injection.py`）。

`__init__.py` 行数预估：现有约 30 行 → 修正后约 80 行。同时承担 re-export hub + module-level 函数 + module state，与 design.md line 786 明示的 facade 角色一致。

测试文件名按 memory `feedback_test_module_name_unique.md`：必须工作区唯一 → 用 `test_injection.py`（无重名冲突；现有 `test_factory.py` / `test_config.py` 等都在 `tests/llm/` 下）。

---

## 3. 对外 5 个新符号（与子项目 2 合计 12 个）

```python
# everalgo/llm/__init__.py（修正后完整版）

"""everalgo.llm 子包 facade — 重新导出公开 surface + 注入机制 module-level 函数。

按 design.md line 786 既定结构（facade pattern），__init__.py 同时承担：
1. Re-export 子项目 2 的 7 个 + 子项目 2.5 的 5 个 = 12 个公开符号
2. 持有 module-level 私有 state（_default + _active ContextVar）
3. 暴露 4 个 module-level 函数（configure / use / current / resolve）
"""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Iterator

# 子项目 2 既有 re-export（保持不变）
from everalgo.llm.config import LLMConfig
from everalgo.llm.errors import LLMError, LLMNotConfiguredError
from everalgo.llm.factory import build_client
from everalgo.llm.protocols import LLMClient
from everalgo.llm.types import ChatMessage, ChatResponse, Usage


# 子项目 2.5 新增 module-level 私有 state
_default: LLMClient | None = None
"""Set-once global default LLM. Mutated only by configure().

Module-private (underscore prefix) — tests may monkey-patch via
``everalgo.llm._default = ...`` for isolation, but this is not part of
the documented public API.
"""

_active: contextvars.ContextVar[LLMClient | None] = contextvars.ContextVar(
    "everalgo_llm_active", default=None,
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
            not accepted).

    Raises:
        TypeError: If ``llm`` is None or not LLMClient-shaped (mypy / pydantic
            type enforcement, not a runtime check).
    """
    global _default
    _default = llm


@contextlib.contextmanager
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


__all__ = [
    # 子项目 2（7 个，保持不变）
    "ChatMessage",
    "ChatResponse",
    "LLMClient",
    "LLMConfig",
    "LLMError",
    "Usage",
    "build_client",
    # 子项目 2.5（5 个，新增）
    "LLMNotConfiguredError",
    "configure",
    "current",
    "resolve",
    "use",
]
```

**`errors.py` 新增**：

```python
# everalgo/llm/errors.py（仅追加 1 个类）

class LLMNotConfiguredError(RuntimeError):
    """Raised when no LLM is configured at any of the 3 injection layers.

    **Inherits ``RuntimeError`` (NOT ``LLMError``) intentionally** — this is
    a developer misuse error (forgot to inject), not a runtime SDK call
    failure. Mirrors pydantic-ai ``UserError(RuntimeError)`` (``pydantic_ai_
    slim/pydantic_ai/exceptions.py:144``); see §6 of the design spec for the
    Exception Family Boundaries decision rationale.
    """
```

---

## 4. 错误处理

| 失败模式 | 抛出类型 | 来源 |
|---|---|---|
| 调用算子时 per-call/scoped/default 全 None | `LLMNotConfiguredError(RuntimeError)` | `resolve()` 兜底 |
| 算子内 SDK call 失败（auth/rate/connection/timeout/...） | `LLMError`（family base，未来 7 子类） | 子项目 2 已落地 `OpenAICompatClient.chat` 内 `except openai.OpenAIError → raise LLMError`（参考 `providers/openai_compat.py`）|
| 调用方误传 `configure(llm=None)` | `TypeError`（mypy 严格模式静态拦截） | `configure(llm: LLMClient)` 必填，`None` 不在签名允许范围 |
| 调用方误传 `use(None)` | `TypeError`（mypy 严格模式静态拦截） | `use(client: LLMClient)` 必填 |

> ✅ **设计自检：configure 不接受 None**
> - **为什么这样设计**：`configure(llm=None)` 看起来像 no-op，实际清空 default 后下次任何算子调用立刻 `LLMNotConfiguredError` —— 这是反模式（BOSS 2026-05-08 拍板）
> - **规范依据**：DSPy `settings.configure` 不允许 None override（`dspy/dsp/utils/settings.py:165`）；pydantic-ai 无 module default 概念，自然无 reset
> - **备选方案**：允许 `configure(llm=None)` reset 全局 / 提供单独 `reset_default()` 函数 —— 都被否（BOSS Q2 拍板，per-call 注入已覆盖测试隔离场景，set-once 语义最简）

---

## 5. Exception Family 设计意图（按 BOSS 要求专门写清）

### 5.1 两个 Family 的语义边界

```text
LLMError(Exception)                    # SDK call errors (rate/auth/connection/timeout/...)
├── (子项目 2 已建 base，未来扩展 7 子类)
└── ...

LLMNotConfiguredError(RuntimeError)    # Misuse: 调用前忘了配置（fail-fast）
                                       # 故意不归 LLMError family
```

| Family | 边界 | 何时抛 | 调用方应对 |
|---|---|---|---|
| **`LLMError(Exception)`** | LLM 调用过程中的错误（SDK call 失败） | 算子内部调 `client.chat(...)` 时 SDK raise（auth fail / rate limit / connection error / timeout / ...） | 生产代码可以 retry / fallback / circuit break；属可恢复运行时错误 |
| **`LLMNotConfiguredError(RuntimeError)`** | 调用前的配置 misuse | `resolve()` 时 per-call/scoped/default 三层全 None | 应在 dev/test 阶段就暴露；不该被 retry |

### 5.2 调用方 catch idiom

| 场景 | 推荐 catch |
|---|---|
| 生产代码 retry SDK 抖动 | `except LLMError:` |
| 测试断言「忘配 LLM 会 fail-fast」 | `except LLMNotConfiguredError:` |
| 极少全栈兜底 | `except (LLMError, LLMNotConfiguredError):` |

### 5.3 为什么不归 LLMError 一锅端

1. **语义分家**：misuse error（开发期 should fail fast）vs SDK runtime error（运行时偶发可重试）—— 两类语义不同，不该混 family
2. **fail-fast 暗示**：`RuntimeError` 在 stdlib 圈意味着「程序状态/使用错误」，调用方不会习惯地 retry / catch
3. **明星项目实证**：pydantic-ai `UserError(RuntimeError)`（`pydantic_ai_slim/pydantic_ai/exceptions.py:144`）—— 现代 LLM 库选 stdlib 而不是自家 family base
4. **stdlib 范例**：`dict.fromkeys` 误用抛 `TypeError` 不抛 `DictError`；`pathlib.Path("x").read_text()` 在文件不存在时抛 `FileNotFoundError`（OSError 子类）而非 `PathError`
5. **Family 整洁度**：LLMError 留给真正的 SDK 错误（auth/rate/connection 等子类未来扩展），不被 misuse 污染语义

> ✅ **设计自检：LLMNotConfiguredError 继承 RuntimeError 而非 LLMError**
> - **为什么这样设计**：misuse error vs SDK runtime error 语义分家；继承 RuntimeError 暗示 fail-fast 不应 retry
> - **规范依据**：pydantic-ai `UserError(RuntimeError)`（`exceptions.py:144`）+ stdlib `dict.fromkeys → TypeError` 范式
> - **备选方案**：① `LLMNotConfiguredError(LLMError)` 同 instructor `ConfigurationError(InstructorError)` 风格（`instructor/core/exceptions.py:242`）—— 否决，污染 LLMError SDK family 语义；② `LLMNotConfiguredError(LLMError, RuntimeError)` 多重继承 —— 否决，MRO 复杂且无项目这么做

---

## 6. 测试矩阵（约 15 个新测试）

### 6.1 `test_injection.py`（约 14 个测试）

按被测函数分组（per memory `feedback_unittest_doc_format.md`）：

**`TestConfigure`**：
- `test_configure_sets_default_visible_via_current` —— `configure(c)` 后 `current() is c`
- `test_configure_overwrites_previous_default` —— 重复 configure 覆盖前值（set-once 单 source of truth；不报 reset 错误）
- `test_configure_default_is_isolated_from_scoped_layer` —— `configure(c1)` 后在 `with use(c2):` 内 `current() is c2`，退出后 `current() is c1`

**`TestUse`**：
- `test_use_sets_active_inside_block` —— `with use(c): assert current() is c`
- `test_use_resets_after_block_exits` —— 退出后 `current()` 返 default 或 None
- `test_use_can_nest` —— `with use(c1): with use(c2): assert current() is c2`，内层退出后外层 c1 还原
- `test_use_works_inside_async_def` —— `async def f(): with use(c): assert current() is c`（asyncio Task 内 ContextVar 正确传播）

**`TestCurrent`**：
- `test_current_returns_none_when_nothing_set` —— 模块加载后 default + scoped 都 None，`current()` 返 None
- `test_current_returns_scoped_over_default` —— `configure(c1) + with use(c2): current() is c2`

**`TestResolve`**：
- `test_resolve_per_call_takes_priority` —— `configure(c1) + with use(c2): resolve(c3) is c3`
- `test_resolve_falls_back_to_scoped` —— `configure(c1) + with use(c2): resolve(None) is c2`
- `test_resolve_falls_back_to_default` —— `configure(c1) + resolve(None) is c1`
- `test_resolve_raises_when_all_layers_none` —— 全 None 时 `pytest.raises(LLMNotConfiguredError, match="No LLM configured")`
- `test_resolve_error_message_lists_three_fix_paths` —— 错误消息含 `configure` / `use` / per-call 三个修复指引关键词

### 6.2 `test_errors.py` 加 1 组（约 1 个测试）

**`TestLLMNotConfiguredError`**：
- `test_inherits_runtime_error_not_llm_error` —— 双 isinstance：`isinstance(err, RuntimeError) is True`、`isinstance(err, LLMError) is False`

### 6.3 物理布局

```
packages/everalgo-core/tests/llm/
├── test_config.py         # 子项目 2（不动）
├── test_errors.py         # 子项目 2 + 新增 1 测试
├── test_factory.py        # 子项目 2（不动）
├── test_injection.py      # NEW: 子项目 2.5
├── test_protocols.py      # 子项目 2（不动）
├── test_public_api.py     # 子项目 2（应更新 __all__ 长度断言：7 → 12）
├── test_types.py          # 子项目 2（不动）
└── providers/test_openai_compat.py  # 子项目 2（不动）
```

> ⚠️ **`test_public_api.py` 必须同步更新**：子项目 2 的 `test_dunder_all_lists_exactly_7_symbols` 类测试断言 `__all__` 长度为 7，本子项目改 12 后需要同步改。

### 6.4 测试隔离策略

每个 `test_injection.py` 测试都 **必须** 通过 fixture 重置 `_default` + `_active`，否则测试间互相污染：

```python
import everalgo.llm

@pytest.fixture(autouse=True)
def reset_everalgo_llm_state() -> Iterator[None]:
    """通过 monkeypatch 直接操作内部 state，不走公开 API（_active 是 ContextVar，
    用 set/reset；_default 是普通变量，直接保存/恢复）。"""
    saved_default = everalgo.llm._default
    token = everalgo.llm._active.set(None)
    try:
        everalgo.llm._default = None
        yield
    finally:
        everalgo.llm._default = saved_default
        everalgo.llm._active.reset(token)
```

> ✅ **设计自检：测试 fixture 直接操作私有 state**
> - **为什么这样设计**：BOSS Q2 拍板「不暴露 reset_default()」，但测试需要互相隔离；通过 monkeypatch `_default` / `_active` 私有变量做 setUp/tearDown 是 Python 圈惯例（DSPy 测试同款）
> - **规范依据**：DSPy 测试 monkeypatch `dspy/dsp/utils/settings.py` 内部变量隔离测试；pytest 文档明示 `monkeypatch.setattr` 适用此场景
> - **备选方案**：① 暴露公开 `reset_default()` —— BOSS Q2 否决；② 全部测试用 per-call 不依赖 default —— 但有些测试就是要测 `configure` 的行为，必须设/恢复 default

---

## 7. 验收标准

- 5 个新公开符号在 `everalgo.llm` 正确导出（`from everalgo.llm import configure, use, current, resolve, LLMNotConfiguredError` 可用）
- `test_public_api.py` 断言 `__all__` 长度 12（7 原 + 5 新）
- `LLMNotConfiguredError` 继承 `RuntimeError`（非 `LLMError`），双 isinstance 测试通过
- `with use(client):` 在 `async def` 函数体内正常工作（asyncio Task ContextVar 传播）
- `resolve(per_call)` 三层 fallback 优先级正确（per-call > scoped > default）
- 全部测试 PASS：`uv run pytest packages/everalgo-core/ -v`（子项目 1+2+3 累计 ~114 + 子项目 2.5 新增 ~15 ≈ 129 tests）
- 4 quality gates 全 clean：ruff check / ruff format --check / mypy / pytest（mypy 从仓库根跑）
- 无新增依赖（`contextlib` + `contextvars` 都是 stdlib）
- `__init__.py` 行数控制在 ~80 行内

---

## 8. 行业参考（6 项目代码引用汇总）

按 BOSS feedback「测试环节决策必须有明星项目代码引用」（memory `feedback_test_decisions_need_star_project_evidence.md`），所有关键决策均 cite 至少 2 个明星项目。

### 8.1 3 层注入实现对比

| 项目 | Module default? | Scoped contextmanager? | Per-call arg? | Storage | sync vs async |
|---|---|---|---|---|---|
| **DSPy** | ✅ `settings.configure(lm=...)` | ✅ `dspy.settings.context(lm=...)` (`dspy/dsp/utils/settings.py:216`) | ❌ | module dict + ContextVar + threading.Lock | **sync `@contextmanager`** (line 216) |
| **LlamaIndex** | ✅ `Settings.llm = ...` (property setter) | ❌ | ✅ (`as_query_engine(llm=...)`) | module 单例属性（无 ContextVar） | N/A |
| **instructor** | ❌ | ❌ | ✅ (`Instructor(client=...)` constructor) | per-instance | 类拆分 `Instructor` / `AsyncInstructor` |
| **LangChain core** | ❌（仅 cache/verbose） | ❌ | ✅ (per-instance) | module global (`langchain_core/globals.py:13-15`) | N/A |
| **openai-python** | ✅ `openai.api_key=...` 模块变量 (`src/openai/__init__.py:83-104`) | ❌ | ✅ (`OpenAI(api_key=...)` + `with_options(...)`) | module variables + lazy `_load_client()` | sync only (`copy()`) |
| **pydantic-ai** | ❌（无 module default） | ✅ `agent.override(model=...)` (per-Agent ContextVar) | ✅ (`agent.iter(model=...)` per-run) | per-Agent ContextVar | **sync `@contextmanager`** |

**关键洞察**：6/6 项目里只有 2/6（DSPy + pydantic-ai）实现了 scoped contextmanager；这 2/2 全部用 sync `@contextmanager`，零项目用 async or dual。

### 8.2 misuse error vs SDK runtime error 区分

| 项目 | 区分 | misuse 类继承 | 备注 |
|---|---|---|---|
| **DSPy** (`dspy/utils/exceptions.py`) | ❌ 不区分 | N/A | 仅 2 个 exception，都继承 `Exception` |
| **LlamaIndex** | ❌ | N/A | 用 pydantic ValidationError |
| **instructor** (`instructor/core/exceptions.py`) | ✅ 区分（`ConfigurationError` line 242 + `ModeError` line 267） | **`InstructorError(Exception)` 自家 family base**（line 7） | 自家 family，不继承 RuntimeError |
| **openai-python** (`src/openai/_exceptions.py`) | ⚠️ 部分区分 | `InvalidWebhookSignatureError(ValueError)` 一个特例 | 大多数继承 OpenAIError |
| **pydantic-ai** (`pydantic_ai_slim/pydantic_ai/exceptions.py`) | ✅ 区分（`UserError` line 144 + `AgentRunError` line 150） | **`RuntimeError` stdlib**（line 144） | 现代 LLM 库（FastAPI 母团队）选 stdlib 风格 |

**EverAlgo 取舍**：选 pydantic-ai 模式（B 选项）— `LLMNotConfiguredError(RuntimeError)`，与 LLMError SDK family 分家。详见 §5。

### 8.3 use() 单形态 sync `@contextmanager` 在 FastAPI 场景的安全性

7 项 FastAPI 场景验证（基于 Python 标准库语义 + 工业实证）：

| # | 场景 | 是否安全 | 原因 |
|---|---|---|---|
| 1 | `async def endpoint` 内 `with use(client): await ...` | ✅ | 同 Task，set/reset 配对，await 不切 Task |
| 2 | 两个并发请求各自 `with use(c1/c2)` | ✅ | Task_A、Task_B 各自 copy_context；ContextVar 仅在自己 Task 见 |
| 3 | `with use(client):` 内 `await loop.run_in_executor(...)` | ✅ | Python 3.7+ 自动 copy_context 到线程池 |
| 4 | `with use(client):` 内 `asyncio.create_task(child)` | ✅ | create_task copy 当前 Context；子 Task 即使 use() 退出也见 client |
| 5 | use() 内 fire-and-forget `asyncio.create_task` 不 await | ✅ | 子 Task 已捕获 |
| 6 | use() 退出后 `BackgroundTasks` 跑 | ✅（符合预期） | bg task 在 response 之后跑，use() reset 已生效；bg 看到 default |
| 7 | `Depends(...)` / middleware 与 endpoint 同 Task | ✅ | FastAPI 全链路在同 Task 内 await，ContextVar 传播 |

**工业实证**：DSPy 在 FastAPI 后端跑 2+ 年；pydantic-ai 是 FastAPI 母团队（pydantic）自家产品，`agent.override` 同款实现 → 作者亲自验证。

---

## 9. 字段决策清单（已对齐，无待 BOSS 校准项）

| # | 决策点 | 取值 | 依据 |
|---|---|---|---|
| 1 | `use()` 单形态 vs dual sync+async | **单形态 sync `@contextmanager`** | §8.1 6/6 调研 + §8.3 FastAPI 7 场景验证 |
| 2 | `configure(llm)` 是否接受 None / 是否支持 reset | **必填、不接受 None、无 reset_default()** | BOSS Q2 拍板（2026-05-08）+ DSPy `settings.configure` 同款 set-once |
| 3 | `LLMNotConfiguredError` 继承链 | **`RuntimeError`（不是 LLMError）** | §5.3 + pydantic-ai `UserError(RuntimeError)` |
| 4 | env / TOML / yaml loader | **不实现** | BOSS Q4 拍板（2026-05-08）+ memory `feedback_everalgo_accepts_config_not_loads.md` |
| 5 | 文件结构（注入相关代码放哪） | **`__init__.py`**（不新建 `injection.py`） | design.md line 786 既定 facade 结构 + BOSS challenge（2026-05-08） |
| 6 | `current()` 返回类型 | **`LLMClient \| None`**（None 是合法返回值） | design.md line 846 明示「`current()` 查 ContextVar (scoped) → 查全局 default」 |
| 7 | `resolve()` 错误消息 | **3 fix paths（configure / use / per-call）都列出** | UX 友好；error message 是用户第一时间看到的修复指引 |
| 8 | `_default` / `_active` 模块私有 | **下划线前缀，不在 `__all__`，但允许测试 monkeypatch** | DSPy 同模式 + pytest fixture 标准做法 |
| 9 | 测试 fixture 重置策略 | **`autouse=True` 重置 `_default` + `_active`** | 测试间互相污染防护；不暴露公开 reset API |
| 10 | 公开符号清单 | **5 个新（configure / use / current / resolve / LLMNotConfiguredError）** | design.md §2.5 + BOSS 4 个澄清拍板 |

---

## 10. Out of Scope（明确移出最小集，附依据）

### 10.1 不做 env / TOML / yaml / k8s ConfigMap config loader

EverAlgo 算法库定位（per memory `feedback_everalgo_accepts_config_not_loads.md`）。调用方有完全自由度：

```python
# 调用方惯例（不属 EverAlgo feature）
config = LLMConfig(
    provider="openai_compat",
    api_key=os.environ["EVERALGO_LLM_API_KEY"],   # 调用方自己读 env
    base_url=os.environ["EVERALGO_LLM_BASE_URL"],
    model=os.environ["EVERALGO_LLM_MODEL"],
)
client = build_client(config)
everalgo.configure(llm=client)
```

design.md line 815-822 提到 `EVERALGO_LLM_*` env vars 是**调用方惯例 hint**，不是 EverAlgo 自带 feature。

### 10.2 不做 reset_default() / configure(llm=None)

BOSS Q2 拍板（2026-05-08）：set-once 语义；测试用 per-call 注入；dev/Jupyter 重启进程即重置。详见 §4 设计自检。

### 10.3 不做 dual sync + async contextmanager（`use()` + `ause()`）

BOSS Q1 拍板（2026-05-08）+ §8.1 6/6 调研支撑（2/2 实现 scoped 都用 sync only）。`use()` 单形态 sync `@contextmanager` 在 async 协程内 `with use(...)` 正常工作（ContextVar async-safe）。

### 10.4 不做 thread-local（`threading.local`）替代 ContextVar

ContextVar 是 Python 3.7+ 标准，async-safe（asyncio Task auto-propagation）+ thread-safe（per-thread isolation）。`threading.local` 不 propagate 到 asyncio Task，已淘汰。

### 10.5 不做 scene 路由

业务编排归 EverOS（design.md §2.5 + ADR 012 line 168 明示）。EverAlgo 完全无 scene 概念。

### 10.6 不做 per-instance Agent override（pydantic-ai 风格）

EverAlgo 算子是 stateless（[ADR 011](docs/decisions/011-protocol-vs-abc.md) Protocol 而非 ABC），无 instance state 持有 client。pydantic-ai `agent.override(...)` 是 per-Agent-instance ContextVar，因为它是 stateful Agent 类。EverAlgo 选模块级 ContextVar 路径，per-call 已覆盖该需求。

### 10.7 不做 LLMError 7 子类细分

design.md line 789 暗示未来扩展 7 子类（auth / rate / connection / timeout / ...）。本子项目仅 LLMError base（已有，子项目 2 落地）+ 新增 LLMNotConfiguredError（独立 family）。7 子类待真实需求触发时分阶段落地。

---

## 11. 自审（writing-plans 之前）

✅ **Spec coverage**：BOSS 4 个澄清问题（Q1 sync `@contextmanager` / Q2 set-once / Q3 RuntimeError / Q4 不做 loader）的取值都对应到 §3-§9 具体段落
✅ **Placeholder scan**：grep 无 `TODO` / `TBD` / `FIXME`
✅ **Internal consistency**：§2 file map 的 `__init__.py` + `errors.py` = §3 公开符号清单覆盖的实现 = §6 测试矩阵的 `test_injection.py` + `test_errors.py`
✅ **Scope check**：单 implementation plan 可实现（约 6-8 个 TDD 任务，scope 比子项目 1+2+3 都小）
✅ **Ambiguity check**：`use()` 双语境（sync `@contextmanager` 在 sync `with` + async `with` 都用）已明确；`current()` 返 None 是合法值已说明；`LLMNotConfiguredError` 继承链已明示
✅ **行业依据 cite 完整**：每个关键决策都给 6 个明星项目的代码引用支撑（§8）
✅ **设计自检 4 处全在文中**：configure 不接受 None / LLMNotConfiguredError 继承 RuntimeError / 测试 fixture 直接操作私有 state / 文件结构按 design.md 既定 facade

下一步：进入 `superpowers:writing-plans` skill 撰写 implementation plan（约 6-8 个 TDD 任务）。
