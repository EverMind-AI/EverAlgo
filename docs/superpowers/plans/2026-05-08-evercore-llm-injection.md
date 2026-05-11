# EverAlgo LLM 3-Layer Injection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `everalgo-core` 子项目 2 已落地的 `everalgo.llm` 子包内增量补 5 个新公开符号（`LLMNotConfiguredError` / `configure` / `use` / `current` / `resolve`）+ 2 个 module-level 私有 state（`_default` / `_active` ContextVar），让算子内部用 `everalgo.llm.resolve(llm)` 单行解析 3 层 fallback（per-call > scoped > default）。

**Architecture:** 4 个 module-level 函数 + 2 个私有 state 全部放在 `everalgo/llm/__init__.py`（design.md line 786 既定 facade 结构）；新增错误类 `LLMNotConfiguredError(RuntimeError)` 加进 `errors.py`（与 LLMError SDK family 分家）。`use()` 是 sync `@contextmanager`，底层 `contextvars.ContextVar` 自动 async-safe（FastAPI / asyncio 7 场景验证）。**不**新建源文件、**不**做 env/TOML loader、**不**支持 reset。

**Tech Stack:** Python 3.12 / stdlib `contextlib` + `contextvars`（无新增 dependency）/ pydantic v2（已有）/ pytest（asyncio_mode=auto）/ ruff / mypy strict。

---

## 关键约束（必读，避免子项目 1+2+3 的重复返工）

1. **测试函数必须 `-> None` 注解**（mypy strict 模式 `tests.*` override 不兜底；显式标注最稳）。
2. **`tests/llm/__init__.py` 不存在 / 不要创建**（沿子项目 1+2+3 的 `--import-mode=importlib` 决策）。
3. **`mypy` 必须从仓库根（`/Users/admin/Documents/evermemos/everalgo`）跑**，**不要**从 `packages/everalgo-core/` 子目录跑（mypy_path 配置在根 `pyproject.toml`，子目录跑会触发误报 `import-untyped`）。
4. **使用 `uv run` 跑所有 Python 命令**（这是 uv workspace；裸 `pytest` 用错解释器）。
5. **零新增依赖** —— 不动 `pyproject.toml`；`contextlib` + `contextvars` 都是 stdlib。
6. **commit 风格 `<emoji> <type>(<scope>): <description>`**，scope 用 `llm`，参考子项目 1+2+3 commit 历史。
7. **每个 commit 落 `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`**。
8. **测试文件名必须工作区唯一** —— 用 `test_injection.py`（per memory `feedback_test_module_name_unique.md`，无重名冲突）。
9. **`test_public_api.py` 在子项目 2 已存在** —— Task 6 必须 modify 这个文件把 `__all__` 长度断言从 7 改成 12。

---

## File Structure

```
packages/everalgo-core/src/everalgo/llm/
├── __init__.py        # MODIFY: Task 2 (state + configure) → Task 3 (use) → Task 4 (current) → Task 5 (resolve) → Task 6 (__all__ + re-export)
├── errors.py          # MODIFY: Task 1 加 LLMNotConfiguredError
├── config.py          # 不动（子项目 2）
├── factory.py         # 不动（子项目 2）
├── protocols.py       # 不动（子项目 2）
├── types.py           # 不动（子项目 2）
└── providers/         # 不动

packages/everalgo-core/tests/llm/
├── test_injection.py  # NEW: Task 2-5 测试 + Task 5 fixture
├── test_errors.py     # MODIFY: Task 1 加 1 个继承链测试
└── test_public_api.py # MODIFY: Task 6 把 __all__ 长度 7 → 12
```

设计依据：`docs/superpowers/specs/2026-05-08-everalgo-llm-injection-design.md` §2 (File Map) 和 §3 (公开 API)。

---

## Task 1: `LLMNotConfiguredError(RuntimeError)` 类 + 继承链测试

**Files:**
- Modify: `packages/everalgo-core/src/everalgo/llm/errors.py`
- Modify: `packages/everalgo-core/tests/llm/test_errors.py`

- [ ] **Step 1: Append failing test to `test_errors.py`**

读取现有 `packages/everalgo-core/tests/llm/test_errors.py` 的内容，在文件末尾追加：

```python


# ---- LLMNotConfiguredError (sub-project 2.5, Task 1) ----------------------

from everalgo.llm.errors import LLMNotConfiguredError


def test_llm_not_configured_error_inherits_runtime_error_not_llm_error() -> None:
    """LLMNotConfiguredError is a misuse error (RuntimeError family),
    not an SDK call error (LLMError family). See spec §5.3."""
    err = LLMNotConfiguredError("test message")
    assert isinstance(err, RuntimeError)
    assert not isinstance(err, LLMError)
```

注意：`LLMError` 在文件顶部应该已经 imported（子项目 2 已有）；如果没有，加 `from everalgo.llm.errors import LLMError` 到 import 块。

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest packages/everalgo-core/tests/llm/test_errors.py -v
```

Expected: FAIL with `ImportError: cannot import name 'LLMNotConfiguredError' from 'everalgo.llm.errors'`.

- [ ] **Step 3: Append `LLMNotConfiguredError` to `errors.py`**

读取现有 `packages/everalgo-core/src/everalgo/llm/errors.py`，在 `LLMError` 类定义之后追加：

```python


class LLMNotConfiguredError(RuntimeError):
    """Raised when no LLM is configured at any of the 3 injection layers.

    Inherits ``RuntimeError`` (NOT ``LLMError``) intentionally — this is a
    developer misuse error (forgot to inject), not a runtime SDK call
    failure. Mirrors pydantic-ai ``UserError(RuntimeError)`` (``pydantic_ai_
    slim/pydantic_ai/exceptions.py:144``); see spec §5 (Exception Family
    Boundaries) for the decision rationale.
    """
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest packages/everalgo-core/tests/llm/test_errors.py -v
```

Expected: 全部 PASS（子项目 2 既有测试 + 1 新增）。

- [ ] **Step 5: Run quality gates (from REPO ROOT)**

```bash
uv run ruff check packages/everalgo-core/
uv run ruff format --check packages/everalgo-core/
uv run mypy packages/everalgo-core/
```

Expected: 全部 clean.

- [ ] **Step 6: Commit**

```bash
git add packages/everalgo-core/src/everalgo/llm/errors.py \
        packages/everalgo-core/tests/llm/test_errors.py
git commit -m "$(cat <<'EOF'
✨ feat(llm): add LLMNotConfiguredError(RuntimeError) misuse error class

LLMNotConfiguredError 故意不归 LLMError SDK family — misuse error
（开发期 fail-fast）vs SDK runtime error 两类语义分家。pydantic-ai
UserError(RuntimeError) 同款 (exceptions.py:144)。

为子项目 2.5 Task 5 (resolve) 抛错备料。详见 spec §5。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `_default` 私有 state + `configure(llm)` 函数 + 测试 fixture

**Files:**
- Modify: `packages/everalgo-core/src/everalgo/llm/__init__.py`
- Create: `packages/everalgo-core/tests/llm/test_injection.py`

- [ ] **Step 1: Create `test_injection.py` with fixture + first test**

写入 `packages/everalgo-core/tests/llm/test_injection.py`:

```python
"""Tests for everalgo.llm 3-layer injection (configure / use / current / resolve).

Each test gets isolated _default + _active state via the autouse fixture
(directly mutating module-private variables — see spec §6.4 for the
rationale: BOSS rejected exposing reset_default() public API; tests use
monkeypatch-style isolation instead).
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

import everalgo.llm
from everalgo.llm.protocols import LLMClient
from everalgo.testing.fake_llm import FakeLLMClient


@pytest.fixture(autouse=True)
def reset_everalgo_llm_state() -> Iterator[None]:
    """Reset _default + _active before each test; restore after.

    _default is a plain module variable — save/restore directly.
    _active is a ContextVar — use set/reset token semantics.
    """
    saved_default = everalgo.llm._default
    token = everalgo.llm._active.set(None)
    try:
        everalgo.llm._default = None
        yield
    finally:
        everalgo.llm._default = saved_default
        everalgo.llm._active.reset(token)


# ---- configure() (Task 2) -------------------------------------------------


def _make_fake() -> LLMClient:
    """Helper: return a FakeLLMClient with no scripted responses (we won't call .chat())."""
    return FakeLLMClient(responses=[])


def test_configure_sets_module_default() -> None:
    """configure(c) sets the module-private _default to c."""
    client = _make_fake()
    everalgo.llm.configure(llm=client)
    assert everalgo.llm._default is client


def test_configure_overwrites_previous_default() -> None:
    """Repeated configure() overwrites the prior default (last-write-wins)."""
    c1 = _make_fake()
    c2 = _make_fake()
    everalgo.llm.configure(llm=c1)
    everalgo.llm.configure(llm=c2)
    assert everalgo.llm._default is c2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest packages/everalgo-core/tests/llm/test_injection.py -v
```

Expected: FAIL with `AttributeError: module 'everalgo.llm' has no attribute '_default'` (or similar — `configure` / `_active` 也不存在).

- [ ] **Step 3: Modify `__init__.py` to add module state + `configure`**

读取现有 `packages/everalgo-core/src/everalgo/llm/__init__.py`，在文件顶部 import 块之后、现有 re-export 之后追加（**保留**所有子项目 2 既有 import 和 `__all__`）：

```python
import contextvars
from contextlib import contextmanager
from collections.abc import Iterator


# Sub-project 2.5: 3-layer LLM injection (configure / use / current / resolve)

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
            not accepted by static type checking).
    """
    global _default
    _default = llm
```

注意：`Iterator` import 是为 Task 3 (`use()`) 备料；如果 ruff 抱怨 unused import，加 `# noqa: F401` 或先不 import（Task 3 时再加）。**推荐**：先不加 `Iterator` import，Task 3 再加（避免 `F401` 噪音）。

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest packages/everalgo-core/tests/llm/test_injection.py -v
```

Expected: 2 PASS.

- [ ] **Step 5: Run quality gates**

```bash
uv run ruff check packages/everalgo-core/
uv run ruff format --check packages/everalgo-core/
uv run mypy packages/everalgo-core/
```

Expected: 全部 clean.

- [ ] **Step 6: Commit**

```bash
git add packages/everalgo-core/src/everalgo/llm/__init__.py \
        packages/everalgo-core/tests/llm/test_injection.py
git commit -m "$(cat <<'EOF'
✨ feat(llm): add _default state + configure(llm) set-once function

引入 module-private _default (LLMClient | None = None) 和 _active
(ContextVar) 两个状态，以及 configure(llm) set-once 函数。后续 task
依赖这两个 state：use() 改 _active，current() 读两者，resolve() 调用
current()。

测试 autouse fixture 直接 monkey-patch _default / _active 做隔离 —
不暴露公开 reset_default() (BOSS 2026-05-08 拍板)。spec §6.4。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `use(client)` sync `@contextmanager`

**Files:**
- Modify: `packages/everalgo-core/src/everalgo/llm/__init__.py`
- Modify: `packages/everalgo-core/tests/llm/test_injection.py`

- [ ] **Step 1: Append failing tests to `test_injection.py`**

```python


# ---- use() (Task 3) -------------------------------------------------------


def test_use_sets_active_inside_block() -> None:
    """Inside `with use(c):` the _active ContextVar holds c."""
    client = _make_fake()
    with everalgo.llm.use(client):
        assert everalgo.llm._active.get() is client


def test_use_resets_after_block_exits() -> None:
    """After `with use(c):` exits, _active is restored to None (the prior value)."""
    client = _make_fake()
    with everalgo.llm.use(client):
        pass
    assert everalgo.llm._active.get() is None


def test_use_can_nest() -> None:
    """Nested use() stacks: inner block sees inner client; outer restored after inner exits."""
    c1 = _make_fake()
    c2 = _make_fake()
    with everalgo.llm.use(c1):
        assert everalgo.llm._active.get() is c1
        with everalgo.llm.use(c2):
            assert everalgo.llm._active.get() is c2
        assert everalgo.llm._active.get() is c1


async def test_use_works_inside_async_def() -> None:
    """ContextVar auto-propagates inside asyncio Task — sync `with` in async def works."""
    client = _make_fake()
    with everalgo.llm.use(client):
        assert everalgo.llm._active.get() is client
        # await something to force a yield point — _active must still hold client
        import asyncio

        await asyncio.sleep(0)
        assert everalgo.llm._active.get() is client
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest packages/everalgo-core/tests/llm/test_injection.py -v
```

Expected: 4 new tests FAIL with `AttributeError: module 'everalgo.llm' has no attribute 'use'`.

- [ ] **Step 3: Append `use()` to `__init__.py`**

在 Task 2 添加的 `configure` 函数之后追加：

```python


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
```

如果 Task 2 没有 import `Iterator`，现在加：

```python
from collections.abc import Iterator
```

到现有 import 块（按 isort 规则放在 `import contextvars` 之前——`from` import 在 `import` 之后但同组内字母序）。**实测**：让 `uv run ruff format` 自动整理 import 顺序。

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest packages/everalgo-core/tests/llm/test_injection.py -v
```

Expected: 6 PASS（Task 2 的 2 + Task 3 的 4）.

- [ ] **Step 5: Run quality gates**

```bash
uv run ruff check packages/everalgo-core/
uv run ruff format --check packages/everalgo-core/
uv run mypy packages/everalgo-core/
```

Expected: 全部 clean.

- [ ] **Step 6: Commit**

```bash
git add packages/everalgo-core/src/everalgo/llm/__init__.py \
        packages/everalgo-core/tests/llm/test_injection.py
git commit -m "$(cat <<'EOF'
✨ feat(llm): add use(client) sync @contextmanager for scoped LLM override

use() 是 sync @contextmanager（不是 async）—— ContextVar.set/reset 是
sync 操作，且 asyncio Task 自动 propagate ContextVar。所以 sync `with
use(c):` 在 async def 函数体内同样工作，FastAPI 端到端 7 场景已验证
（spec §8.3）。

DSPy dspy.settings.context (settings.py:216) + pydantic-ai
agent.override 同款（2/2 实现 scoped 的明星项目都选 sync only）。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `current()` 查询函数（scoped > default）

**Files:**
- Modify: `packages/everalgo-core/src/everalgo/llm/__init__.py`
- Modify: `packages/everalgo-core/tests/llm/test_injection.py`

- [ ] **Step 1: Append failing tests to `test_injection.py`**

```python


# ---- current() (Task 4) ---------------------------------------------------


def test_current_returns_none_when_nothing_set() -> None:
    """With no configure() and no use(), current() returns None."""
    assert everalgo.llm.current() is None


def test_current_returns_default_when_only_configured() -> None:
    """configure(c) sets default; current() returns c."""
    client = _make_fake()
    everalgo.llm.configure(llm=client)
    assert everalgo.llm.current() is client


def test_current_returns_scoped_over_default() -> None:
    """When both layers set, scoped (use) wins over default (configure)."""
    c_default = _make_fake()
    c_scoped = _make_fake()
    everalgo.llm.configure(llm=c_default)
    with everalgo.llm.use(c_scoped):
        assert everalgo.llm.current() is c_scoped
    # After exiting use(), default is back
    assert everalgo.llm.current() is c_default
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest packages/everalgo-core/tests/llm/test_injection.py -v
```

Expected: 3 new tests FAIL with `AttributeError: module 'everalgo.llm' has no attribute 'current'`.

- [ ] **Step 3: Append `current()` to `__init__.py`**

在 `use()` 之后追加：

```python


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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest packages/everalgo-core/tests/llm/test_injection.py -v
```

Expected: 9 PASS（Task 2 的 2 + Task 3 的 4 + Task 4 的 3）.

- [ ] **Step 5: Run quality gates**

```bash
uv run ruff check packages/everalgo-core/
uv run ruff format --check packages/everalgo-core/
uv run mypy packages/everalgo-core/
```

Expected: 全部 clean.

- [ ] **Step 6: Commit**

```bash
git add packages/everalgo-core/src/everalgo/llm/__init__.py \
        packages/everalgo-core/tests/llm/test_injection.py
git commit -m "$(cat <<'EOF'
✨ feat(llm): add current() read-only query (scoped > default)

current() 是只读查询入口 — 返回 _active.get() or _default。None 是
合法返回值（未注入），不抛错。需要 fail-fast 抛错语义请用 resolve()。

为 Task 5 resolve() 备料（resolve 内部调 current() 然后 None 时抛
LLMNotConfiguredError）。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: `resolve(per_call)` 三层 fallback + 错误抛出

**Files:**
- Modify: `packages/everalgo-core/src/everalgo/llm/__init__.py`
- Modify: `packages/everalgo-core/tests/llm/test_injection.py`

- [ ] **Step 1: Append failing tests to `test_injection.py`**

```python


# ---- resolve() (Task 5) ---------------------------------------------------


def test_resolve_per_call_takes_priority() -> None:
    """per_call argument wins over scoped + default."""
    c_default = _make_fake()
    c_scoped = _make_fake()
    c_per_call = _make_fake()
    everalgo.llm.configure(llm=c_default)
    with everalgo.llm.use(c_scoped):
        assert everalgo.llm.resolve(c_per_call) is c_per_call


def test_resolve_falls_back_to_scoped() -> None:
    """When per_call=None, scoped wins over default."""
    c_default = _make_fake()
    c_scoped = _make_fake()
    everalgo.llm.configure(llm=c_default)
    with everalgo.llm.use(c_scoped):
        assert everalgo.llm.resolve(None) is c_scoped


def test_resolve_falls_back_to_default() -> None:
    """When per_call=None and no scoped, default wins."""
    c_default = _make_fake()
    everalgo.llm.configure(llm=c_default)
    assert everalgo.llm.resolve(None) is c_default


def test_resolve_raises_when_all_layers_none() -> None:
    """All 3 layers None → LLMNotConfiguredError."""
    from everalgo.llm.errors import LLMNotConfiguredError

    with pytest.raises(LLMNotConfiguredError, match="No LLM configured"):
        everalgo.llm.resolve(None)


def test_resolve_error_message_lists_three_fix_paths() -> None:
    """Error message names all 3 fix paths: configure / use / per-call."""
    from everalgo.llm.errors import LLMNotConfiguredError

    with pytest.raises(LLMNotConfiguredError) as excinfo:
        everalgo.llm.resolve(None)
    msg = str(excinfo.value)
    assert "configure" in msg
    assert "use" in msg
    # per-call hint phrased as `llm=client`
    assert "llm=client" in msg
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest packages/everalgo-core/tests/llm/test_injection.py -v
```

Expected: 5 new tests FAIL with `AttributeError: module 'everalgo.llm' has no attribute 'resolve'`.

- [ ] **Step 3: Append `resolve()` to `__init__.py`**

在 `current()` 之后追加：

```python


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
```

注意：`LLMNotConfiguredError` 必须 import 进来。在文件顶部 `from everalgo.llm.errors import LLMError` 一行旁边追加（合并为单 import）：

```python
from everalgo.llm.errors import LLMError, LLMNotConfiguredError
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest packages/everalgo-core/tests/llm/test_injection.py -v
```

Expected: 14 PASS（Task 2 的 2 + Task 3 的 4 + Task 4 的 3 + Task 5 的 5）.

- [ ] **Step 5: Run quality gates**

```bash
uv run ruff check packages/everalgo-core/
uv run ruff format --check packages/everalgo-core/
uv run mypy packages/everalgo-core/
```

Expected: 全部 clean.

- [ ] **Step 6: Commit**

```bash
git add packages/everalgo-core/src/everalgo/llm/__init__.py \
        packages/everalgo-core/tests/llm/test_injection.py
git commit -m "$(cat <<'EOF'
✨ feat(llm): add resolve(per_call) 3-layer fallback helper

resolve() 是算子内部的标准入口（design.md line 833）—
client = everalgo.llm.resolve(llm) 一行解析 3 层 fallback：
per_call > scoped > default，全 None 抛 LLMNotConfiguredError
含三层修复指引。

DSPy Settings.lm auto-fallback (settings.py:78) 同款。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: `__all__` 更新（7 → 12）+ test_public_api.py 同步 + 子项目 2.5 验收

**Files:**
- Modify: `packages/everalgo-core/src/everalgo/llm/__init__.py`
- Modify: `packages/everalgo-core/tests/llm/test_public_api.py`

- [ ] **Step 1: Update `test_public_api.py`**

读取现有 `packages/everalgo-core/tests/llm/test_public_api.py`。子项目 2 的测试断言 `__all__` 长度为 7（具体测试函数名可能是 `test_dunder_all_lists_exactly_7_symbols` 或类似）。**修改**这个测试：

把 `7` 改为 `12`，把符号清单从 7 个扩展为 12 个：

旧（子项目 2）：
```python
def test_dunder_all_lists_exactly_7_symbols() -> None:
    """__all__ enumerates the public surface — exactly 7 entries."""
    assert sorted(everalgo.llm.__all__) == sorted([
        "ChatMessage",
        "ChatResponse",
        "LLMClient",
        "LLMConfig",
        "LLMError",
        "Usage",
        "build_client",
    ])
```

新（子项目 2.5）：
```python
def test_dunder_all_lists_exactly_12_symbols() -> None:
    """__all__ enumerates the public surface — exactly 12 entries (sub-project 2: 7 + sub-project 2.5: 5)."""
    assert sorted(everalgo.llm.__all__) == sorted([
        # sub-project 2 (7)
        "ChatMessage",
        "ChatResponse",
        "LLMClient",
        "LLMConfig",
        "LLMError",
        "Usage",
        "build_client",
        # sub-project 2.5 (5)
        "LLMNotConfiguredError",
        "configure",
        "current",
        "resolve",
        "use",
    ])
```

如果 `test_public_api.py` 内有 `test_public_symbols_exposed_at_top_level` / `test_top_level_import_works` 这类测试，**追加**（不替换）对子项目 2.5 5 个新符号的检查。例如追加：

```python


def test_subproject_2_5_symbols_importable() -> None:
    """Sub-project 2.5 5 new symbols are importable from everalgo.llm top level."""
    from everalgo.llm import (
        LLMNotConfiguredError,
        configure,
        current,
        resolve,
        use,
    )

    assert callable(configure)
    assert callable(use)
    assert callable(current)
    assert callable(resolve)
    assert issubclass(LLMNotConfiguredError, RuntimeError)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest packages/everalgo-core/tests/llm/test_public_api.py -v
```

Expected: FAIL with `AssertionError: lists differ` 或类似 — `__all__` 还是 7 个，新断言期望 12.

- [ ] **Step 3: Update `__all__` in `__init__.py`**

读取现有 `packages/everalgo-core/src/everalgo/llm/__init__.py`。找到 `__all__ = [...]` 块，把 5 个新符号加入（保持 alphabetical）：

旧（子项目 2）：
```python
__all__ = [
    "ChatMessage",
    "ChatResponse",
    "LLMClient",
    "LLMConfig",
    "LLMError",
    "Usage",
    "build_client",
]
```

新（子项目 2.5）：
```python
__all__ = [
    # sub-project 2 (LLM Stack — 7)
    "ChatMessage",
    "ChatResponse",
    "LLMClient",
    "LLMConfig",
    "LLMError",
    "Usage",
    "build_client",
    # sub-project 2.5 (3-layer injection — 5)
    "LLMNotConfiguredError",
    "configure",
    "current",
    "resolve",
    "use",
]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest packages/everalgo-core/tests/llm/ -v
```

Expected: 全部 PASS（子项目 2 既有所有测试 + Task 1 errors 测试 + Tasks 2-5 injection 14 测试 + Task 6 public_api 修正 + 新 smoke 测试）。

- [ ] **Step 5: Full sub-project acceptance — run all 4 quality gates from REPO ROOT**

```bash
uv run ruff check packages/everalgo-core/
uv run ruff format --check packages/everalgo-core/
uv run mypy packages/everalgo-core/
uv run pytest packages/everalgo-core/ -v
```

Expected:
- ruff check: clean
- ruff format check: clean
- mypy: 0 errors（从 REPO ROOT 跑）
- pytest: **整个 everalgo-core 累计 ~129 tests PASS**（子项目 1+2+3 的 ~114 + 子项目 2.5 新增 ~15）

如果任何 gate 失败，**停下并修复** —— 不要 commit。

- [ ] **Step 6: Verify spec contract end-to-end**

```bash
uv run python -c "
from everalgo.llm import (
    LLMNotConfiguredError,
    configure,
    current,
    resolve,
    use,
)
from everalgo.testing.fake_llm import FakeLLMClient

c = FakeLLMClient(responses=[])

# Path 1: per-call
assert resolve(c) is c

# Path 2: scoped
with use(c):
    assert current() is c
    assert resolve(None) is c

# Path 3: default
configure(llm=c)
assert current() is c
assert resolve(None) is c

# Error path
import everalgo.llm
everalgo.llm._default = None
try:
    resolve(None)
    raise AssertionError('expected LLMNotConfiguredError')
except LLMNotConfiguredError as e:
    assert 'configure' in str(e)
    assert 'use' in str(e)

print('Sub-project 2.5 contract OK')
"
```

Expected output: 打印 `Sub-project 2.5 contract OK` 不抛 ImportError 不抛 AssertionError.

- [ ] **Step 7: Commit**

```bash
git add packages/everalgo-core/src/everalgo/llm/__init__.py \
        packages/everalgo-core/tests/llm/test_public_api.py
git commit -m "$(cat <<'EOF'
✨ feat(llm): expose 5 new symbols at everalgo.llm top level (3-layer injection)

Update __all__ from 7 → 12 (sub-project 2: 7 + sub-project 2.5: 5).
Sync test_public_api.py: __all__ 长度断言 + 5 个新符号 importable smoke.

Sub-project 2.5 (LLM 3-Layer Injection) complete:
- 5 new public symbols (configure / use / current / resolve /
  LLMNotConfiguredError), 0 new dependencies, ~15 new tests
- Industry references: DSPy + LlamaIndex + instructor + LangChain core +
  openai-python + pydantic-ai (per spec §8)
- Ready for sub-project 4 (reference impl) — operators can now use the
  spec line 833 form: `client = everalgo.llm.resolve(llm)`.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review Checklist (作者自审，已通过)

### 1. Spec coverage

| Spec 章节 | 实现 task |
|---|---|
| §2 File Map（2 modify + 1 new test file） | Task 1 (errors.py + test_errors.py) + Task 2-6 (__init__.py + test_injection.py + test_public_api.py) |
| §3 5 个新公开符号 | Task 1 (LLMNotConfiguredError) + Task 2 (configure) + Task 3 (use) + Task 4 (current) + Task 5 (resolve) |
| §3 2 个 module-level 私有 state | Task 2 (_default + _active) |
| §4 错误处理矩阵 4 行 | Task 1 (LLMNotConfiguredError 继承) + Task 5 (3 层全 None raise) |
| §5 Exception Family 边界 | Task 1 docstring 引用 spec §5 + 测试断言「不归 LLMError」 |
| §6 测试矩阵 5 测试组 | TestConfigure (Task 2) + TestUse (Task 3) + TestCurrent (Task 4) + TestResolve (Task 5) + TestLLMNotConfiguredError (Task 1) |
| §6.4 测试 fixture 隔离 | Task 2 创建 `reset_everalgo_llm_state` autouse fixture |
| §7 验收标准 8 项 | Task 6 Step 5+6 全覆盖 |

无 spec 章节未覆盖。

### 2. Placeholder scan

`grep -nE "TODO|TBD|FIXME|XXX|implement later|fill in" docs/superpowers/plans/2026-05-08-everalgo-llm-injection.md`
预期：0 hits。

### 3. Type consistency

- `LLMClient` 类型在 Task 2-6 全部一致（来自 `everalgo.llm.protocols`）
- `LLMNotConfiguredError` 在 Task 1 定义 → Task 5 raise → Task 6 smoke import 三处一致
- `configure(llm: LLMClient)` 参数名 `llm` 在 Task 2 + Task 6 acceptance script 一致
- `use(client: LLMClient)` 参数名 `client` 在 Task 3 + Task 6 一致
- `current() -> LLMClient | None` 签名在 Task 4 + Task 6 一致
- `resolve(per_call: LLMClient | None = None) -> LLMClient` 签名在 Task 5 + Task 6 一致

### 4. Lessons learned 应用

- ✅ 测试函数 `-> None` 注解 — 全部 task 测试代码已显式标注
- ✅ `tests/llm/__init__.py` 不创建 — 沿用 importlib 模式
- ✅ 每个 task 的 commit 含 `Co-Authored-By` — 通过模板说明
- ✅ ruff + mypy + pytest 三 gate 每 task 必过 — 每 task Step 5
- ✅ 0 新依赖 — 仅 stdlib `contextlib` + `contextvars`
- ✅ `mypy` 从 REPO ROOT 跑 — Task 1-6 都明示
- ✅ 测试文件名工作区唯一 — `test_injection.py` 无重名冲突
- ✅ 公开符号清单与 spec / AGENTS.md 严格一致 — Task 6 Step 1+3 双向断言（test_public_api.py + __all__）

### 5. 任务大小

- Task 1：约 15 分钟（1 错误类 + 1 测试）
- Task 2：约 25 分钟（state + configure + autouse fixture）
- Task 3：约 25 分钟（use contextmanager + 4 测试）
- Task 4：约 15 分钟（current 3 行实现 + 3 测试）
- Task 5：约 25 分钟（resolve + 5 测试 + LLMNotConfiguredError import）
- Task 6：约 25 分钟（__all__ 更新 + test_public_api.py 同步 + 子项目验收）

总计：约 2-2.5 小时（含 SDD review 循环）。
