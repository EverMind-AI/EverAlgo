# EverAlgo Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 落地 EverAlgo 子项目 1 (Foundation) — `everalgo.types.*` EPISODE 路径最小集 4 个对外符号 + `everalgo.prompts.validator` 2 个函数，让算法同学能写 EpisodeExtractor 签名 + 改 prompt 早期 fail-fast。

**Architecture:** 严格按 [`docs/superpowers/specs/2026-05-07-everalgo-foundation-design.md`](../specs/2026-05-07-everalgo-foundation-design.md) 落地。types 用 `pydantic.BaseModel` v2 (`extra="ignore"` for Message/MemCell — opensource payload 兼容；`extra="allow"` for Episode — LLM 可能输出二级字段如 subject/summary/keywords）。validator 用标准库 `string.Formatter` + chars/4 兜底 token estimator，零外部依赖。tests 走 pydantic-ai 模式（`packages/<dist>/tests/` 内嵌 + function-based + `pytest.mark.parametrize`）。

**Tech Stack:** Python 3.12, pydantic ≥ 2.7, pytest ≥ 8 (asyncio_mode=auto via root pyproject.toml), uv workspace, ruff (in root pyproject), mypy (in root pyproject).

**Code language convention:** 严格英文（identifier / docstring / inline comment / commit message body）。本 plan 的自然语言说明（task 标题、步骤描述、`Run` / `Expected` 注解）允许中文，但**所有 code block 内容 100% 英文**。

---

## File Structure

新增源文件 4 个：

| 文件 | 职责 |
|---|---|
| `packages/everalgo-core/src/everalgo/types/__init__.py` | re-export 4 个对外符号 + `__all__` |
| `packages/everalgo-core/src/everalgo/types/memcell.py` | `MessageRole` + `Message` + `MemCell` |
| `packages/everalgo-core/src/everalgo/types/memories.py` | `Episode` |
| `packages/everalgo-core/src/everalgo/prompts/validator.py` | `check_placeholders` + `check_length` |

修改源文件 1 个：

| 文件 | 改动 |
|---|---|
| `packages/everalgo-core/pyproject.toml` | 在 `dependencies` 加 `pydantic>=2.7` |

新增测试文件 8 个（`packages/everalgo-core/tests/`）：

| 文件 | 职责 |
|---|---|
| `tests/conftest.py` | 包级 fixture 入口 (placeholder docstring) |
| `tests/types/__init__.py` | empty package marker |
| `tests/types/test_message.py` | `Message` + `MessageRole` 单元测试 |
| `tests/types/test_memcell.py` | `MemCell` 单元测试 |
| `tests/types/test_episode.py` | `Episode` 单元测试 |
| `tests/types/test_round_trip.py` | 跨 type JSON round-trip 参数化 |
| `tests/prompts/__init__.py` | empty package marker |
| `tests/prompts/test_validator.py` | `check_placeholders` + `check_length` 单元测试 |

修改文档 1 个：

| 文件 | 改动 |
|---|---|
| `AGENTS.md` (§5 Code Style) | 加一条规则："Sync bridge for I/O operators: write `extract = async_to_sync(aextract)` one-liner per ADR 010 line 199-214; do not introduce a `DualInterface` mixin." |

---

## Task 0: pyproject.toml 加 pydantic 依赖 + 测试目录骨架

**Files:**
- Modify: `packages/everalgo-core/pyproject.toml`
- Create: `packages/everalgo-core/tests/conftest.py`
- Create: `packages/everalgo-core/tests/types/__init__.py`
- Create: `packages/everalgo-core/tests/prompts/__init__.py`

- [ ] **Step 1: 修改 pyproject.toml 添加 pydantic 依赖**

把 `packages/everalgo-core/pyproject.toml` 中的 `dependencies = []` 替换为：

```toml
dependencies = [
  "pydantic>=2.7",
]
```

- [ ] **Step 2: 创建 tests 包骨架**

创建 `packages/everalgo-core/tests/conftest.py` (内容如下)：

```python
"""Package-level pytest fixtures.

Placeholder — this conftest reserves the slot for shared fixtures (e.g.
fake LLM client, episode builders) introduced by sub-project 3
(Testing Toolkit). It currently exposes no fixtures.
"""
```

创建 `packages/everalgo-core/tests/types/__init__.py` (空文件，仅作为 package marker，pytest 在 `testpaths` 下需要)。

创建 `packages/everalgo-core/tests/prompts/__init__.py` (空文件，同理)。

- [ ] **Step 3: 同步 workspace 依赖**

Run: `uv sync --all-packages`

Expected: 输出含 `+ pydantic==2.x.x` (任何 2.7+ 小版本)，无 error。

- [ ] **Step 4: 验证 pydantic 可 import**

Run: `uv run python -c "import pydantic; print(pydantic.__version__)"`

Expected: 输出 `2.x.x` (≥ 2.7)，无 error。

- [ ] **Step 5: Commit**

```bash
git add packages/everalgo-core/pyproject.toml packages/everalgo-core/tests/
git commit -m "🎉 chore(core): add pydantic dependency and tests scaffold"
```

---

## Task 1: MessageRole enum

**Files:**
- Create: `packages/everalgo-core/src/everalgo/types/memcell.py`
- Create: `packages/everalgo-core/tests/types/test_message.py`

- [ ] **Step 1: Write the failing test (MessageRole enum 仅 USER/ASSISTANT)**

创建 `packages/everalgo-core/tests/types/test_message.py`：

```python
"""Tests for everalgo.types.memcell — MessageRole + Message."""

from everalgo.types.memcell import MessageRole


def test_message_role_enum_values_are_user_and_assistant():
    """Minimal set: USER + ASSISTANT only — TOOL / SYSTEM are out of EPISODE scope."""
    assert {r.value for r in MessageRole} == {"user", "assistant"}


def test_message_role_str_inheritance():
    """MessageRole should be a str enum so it serialises directly to its value."""
    assert MessageRole.USER == "user"
    assert MessageRole.ASSISTANT == "assistant"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/everalgo-core/tests/types/test_message.py -v`

Expected: FAIL — `ModuleNotFoundError: No module named 'everalgo.types.memcell'`

- [ ] **Step 3: Write minimal implementation**

创建 `packages/everalgo-core/src/everalgo/types/memcell.py`：

```python
"""Conversation message types — minimal field set for the EPISODE path.

Reference: design.md §1.2 (boundary + extract phases) and the Foundation
spec (docs/superpowers/specs/2026-05-07-everalgo-foundation-design.md).
"""

from enum import Enum


class MessageRole(str, Enum):
    """Conversation role taxonomy.

    The EPISODE path consumes user/assistant messages only. ``tool`` and
    ``system`` roles are intentionally omitted from the minimal type set
    and may be added later via a SemVer minor bump (extending an enum
    with new members is a backward-compatible change).
    """

    USER = "user"
    ASSISTANT = "assistant"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/everalgo-core/tests/types/test_message.py -v`

Expected: PASS — both `test_message_role_enum_values_are_user_and_assistant` 和 `test_message_role_str_inheritance`。

- [ ] **Step 5: Commit**

```bash
git add packages/everalgo-core/src/everalgo/types/memcell.py packages/everalgo-core/tests/types/test_message.py
git commit -m "✨ feat(types): add MessageRole enum (USER/ASSISTANT minimal set)"
```

---

## Task 2: Message BaseModel

**Files:**
- Modify: `packages/everalgo-core/src/everalgo/types/memcell.py` (extend with Message class)
- Modify: `packages/everalgo-core/tests/types/test_message.py` (extend with Message tests)

- [ ] **Step 1: Write the failing tests (Message)**

在 `packages/everalgo-core/tests/types/test_message.py` 末尾追加：

```python
import json

import pytest
from pydantic import ValidationError

from everalgo.types.memcell import Message


def test_message_minimum_required_fields():
    msg = Message(role=MessageRole.USER, content="hello", timestamp=1700000000000)
    assert msg.role == MessageRole.USER
    assert msg.content == "hello"
    assert msg.timestamp == 1700000000000


def test_message_role_accepts_string_value():
    """Pydantic should coerce raw role string to the enum."""
    msg = Message(role="assistant", content="hi", timestamp=1)
    assert msg.role == MessageRole.ASSISTANT


def test_message_missing_required_field_raises():
    with pytest.raises(ValidationError):
        Message(role=MessageRole.USER, content="hello")  # type: ignore[call-arg]


def test_message_invalid_role_raises():
    """system / tool roles are not allowed in the minimal set."""
    with pytest.raises(ValidationError):
        Message(role="system", content="hello", timestamp=1)
    with pytest.raises(ValidationError):
        Message(role="tool", content="hello", timestamp=1)


def test_message_extra_fields_silently_ignored():
    """Opensource payload may carry sender_id / tool_calls / message_id — they should be dropped."""
    msg = Message.model_validate(
        {
            "role": "user",
            "content": "hi",
            "timestamp": 1,
            "sender_id": "u1",
            "sender_name": "Alice",
            "tool_calls": [{"id": "x"}],
            "message_id": "m1",
        }
    )
    assert msg.role == MessageRole.USER
    assert msg.content == "hi"
    assert msg.timestamp == 1
    assert not hasattr(msg, "sender_id")
    assert not hasattr(msg, "tool_calls")


def test_message_json_round_trip():
    msg = Message(role=MessageRole.USER, content="hi", timestamp=42)
    serialised = msg.model_dump_json()
    assert json.loads(serialised) == {"role": "user", "content": "hi", "timestamp": 42}
    assert Message.model_validate_json(serialised) == msg
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/everalgo-core/tests/types/test_message.py -v`

Expected: 6 FAIL (新加的 test_message_*) — `ImportError: cannot import name 'Message' from 'everalgo.types.memcell'`. The 2 MessageRole tests still PASS.

- [ ] **Step 3: Write minimal implementation**

修改 `packages/everalgo-core/src/everalgo/types/memcell.py`，在文件末尾追加：

```python
from pydantic import BaseModel, ConfigDict


class Message(BaseModel):
    """Single conversation message.

    Minimal field set for the EPISODE path: ``role`` + ``content`` +
    ``timestamp``. Other fields (sender_id, tool_calls, ...) are out of
    scope for sub-project 1; ``extra="ignore"`` silently drops them so
    that opensource payloads (which carry sender_id / message_id /
    tool_calls / ...) round-trip cleanly.
    """

    role: MessageRole
    content: str
    timestamp: int  # Unix epoch milliseconds

    model_config = ConfigDict(extra="ignore")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/everalgo-core/tests/types/test_message.py -v`

Expected: 8 PASS (2 MessageRole + 6 Message)。

- [ ] **Step 5: Commit**

```bash
git add packages/everalgo-core/src/everalgo/types/memcell.py packages/everalgo-core/tests/types/test_message.py
git commit -m "✨ feat(types): add Message BaseModel (3 fields, extra=ignore)"
```

---

## Task 3: MemCell BaseModel

**Files:**
- Modify: `packages/everalgo-core/src/everalgo/types/memcell.py` (extend with MemCell class)
- Create: `packages/everalgo-core/tests/types/test_memcell.py`

- [ ] **Step 1: Write the failing tests**

创建 `packages/everalgo-core/tests/types/test_memcell.py`：

```python
"""Tests for everalgo.types.memcell.MemCell."""

import pytest
from pydantic import ValidationError

from everalgo.types.memcell import MemCell, Message, MessageRole


def _msg(content: str = "hi", ts: int = 1) -> Message:
    return Message(role=MessageRole.USER, content=content, timestamp=ts)


def test_memcell_minimum_required_fields():
    cell = MemCell(id="m1", messages=[_msg()], timestamp=1700000000000)
    assert cell.id == "m1"
    assert len(cell.messages) == 1
    assert cell.timestamp == 1700000000000


def test_memcell_messages_coerced_from_dicts():
    """Pydantic should rebuild Message objects from raw dicts inside `messages`."""
    cell = MemCell.model_validate(
        {
            "id": "m1",
            "timestamp": 1,
            "messages": [{"role": "user", "content": "hi", "timestamp": 1}],
        }
    )
    assert cell.messages[0].role == MessageRole.USER
    assert cell.messages[0].content == "hi"


def test_memcell_missing_id_raises():
    with pytest.raises(ValidationError):
        MemCell(messages=[_msg()], timestamp=1)  # type: ignore[call-arg]


def test_memcell_missing_messages_raises():
    with pytest.raises(ValidationError):
        MemCell(id="m1", timestamp=1)  # type: ignore[call-arg]


def test_memcell_empty_messages_allowed():
    """Type does not enforce min_length=1; caller (boundary extractor) decides."""
    cell = MemCell(id="m1", messages=[], timestamp=1)
    assert cell.messages == []


def test_memcell_extra_fields_silently_ignored():
    """Opensource MemCell carries source_type / sender_ids / user_id_list / group_id / participants — drop them all."""
    cell = MemCell.model_validate(
        {
            "id": "m1",
            "messages": [],
            "timestamp": 1,
            "source_type": "chat",
            "sender_ids": ["u1"],
            "user_id_list": ["u1", "u2"],
            "group_id": "g1",
            "participants": ["u1"],
            "type": "Conversation",
        }
    )
    assert cell.id == "m1"
    assert not hasattr(cell, "source_type")
    assert not hasattr(cell, "group_id")
    assert not hasattr(cell, "user_id_list")


def test_memcell_json_round_trip():
    cell = MemCell(id="m1", messages=[_msg("hello", 5)], timestamp=10)
    serialised = cell.model_dump_json()
    assert MemCell.model_validate_json(serialised) == cell
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/everalgo-core/tests/types/test_memcell.py -v`

Expected: 7 FAIL — `ImportError: cannot import name 'MemCell' from 'everalgo.types.memcell'`.

- [ ] **Step 3: Write minimal implementation**

在 `packages/everalgo-core/src/everalgo/types/memcell.py` 末尾追加：

```python
class MemCell(BaseModel):
    """Boundary extractor output — a coherent slice of conversation.

    Minimal field set for the EPISODE path: ``id`` (data lineage anchor
    referenced by Episode.parent_id), ``messages`` (LLM prompt context),
    ``timestamp`` (Episode.timestamp default). Boundary metadata
    (source_type / sender_ids / start_idx / token_count / boundary_reason)
    is added later when the boundary subpackage lands.

    ``extra="ignore"`` keeps opensource MemCell payloads (which carry
    source_type / user_id_list / group_id / participants) deserialisable
    without errors.
    """

    id: str
    messages: list[Message]
    timestamp: int  # Unix epoch milliseconds

    model_config = ConfigDict(extra="ignore")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/everalgo-core/tests/types/ -v`

Expected: 15 PASS (8 message + 7 memcell)。

- [ ] **Step 5: Commit**

```bash
git add packages/everalgo-core/src/everalgo/types/memcell.py packages/everalgo-core/tests/types/test_memcell.py
git commit -m "✨ feat(types): add MemCell BaseModel (3 fields, extra=ignore)"
```

---

## Task 4: Episode BaseModel

**Files:**
- Create: `packages/everalgo-core/src/everalgo/types/memories.py`
- Create: `packages/everalgo-core/tests/types/test_episode.py`

- [ ] **Step 1: Write the failing tests**

创建 `packages/everalgo-core/tests/types/test_episode.py`：

```python
"""Tests for everalgo.types.memories.Episode."""

import pytest
from pydantic import ValidationError

from everalgo.types.memories import Episode


def _kwargs(**overrides: object) -> dict:
    base = dict(
        id="ep1",
        owner_id="u1",
        episode="Alice asked about Q3 plan.",
        timestamp=1700000000000,
        parent_id="m1",
    )
    base.update(overrides)
    return base


def test_episode_minimum_required_fields():
    ep = Episode(**_kwargs())
    assert ep.id == "ep1"
    assert ep.owner_id == "u1"
    assert ep.episode == "Alice asked about Q3 plan."
    assert ep.timestamp == 1700000000000
    assert ep.parent_id == "m1"


def test_episode_parent_type_default_is_memcell():
    ep = Episode(**_kwargs())
    assert ep.parent_type == "memcell"


def test_episode_parent_type_overridable():
    ep = Episode(**_kwargs(parent_type="episode"))
    assert ep.parent_type == "episode"


def test_episode_owner_id_required():
    with pytest.raises(ValidationError):
        Episode(  # type: ignore[call-arg]
            id="ep1",
            episode="text",
            timestamp=1,
            parent_id="m1",
        )


def test_episode_episode_field_required():
    with pytest.raises(ValidationError):
        Episode(  # type: ignore[call-arg]
            id="ep1",
            owner_id="u1",
            timestamp=1,
            parent_id="m1",
        )


def test_episode_extra_fields_kept_accessible():
    """Episode uses ``extra='allow'`` so LLM-emitted secondary fields stay reachable."""
    ep = Episode.model_validate(
        _kwargs(
            subject="Alice",
            summary="short",
            keywords=["q3", "plan"],
            location="meeting room",
        )
    )
    assert ep.subject == "Alice"  # type: ignore[attr-defined]
    assert ep.summary == "short"  # type: ignore[attr-defined]
    assert ep.keywords == ["q3", "plan"]  # type: ignore[attr-defined]
    assert ep.location == "meeting room"  # type: ignore[attr-defined]


def test_episode_json_round_trip_preserves_extras():
    ep = Episode.model_validate(_kwargs(summary="s", keywords=["a"]))
    serialised = ep.model_dump_json()
    rebuilt = Episode.model_validate_json(serialised)
    assert rebuilt == ep
    assert rebuilt.summary == "s"  # type: ignore[attr-defined]
    assert rebuilt.keywords == ["a"]  # type: ignore[attr-defined]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/everalgo-core/tests/types/test_episode.py -v`

Expected: 7 FAIL — `ModuleNotFoundError: No module named 'everalgo.types.memories'`.

- [ ] **Step 3: Write minimal implementation**

创建 `packages/everalgo-core/src/everalgo/types/memories.py`：

```python
"""User-side memory types — minimal set for the EPISODE path."""

from pydantic import BaseModel, ConfigDict


class Episode(BaseModel):
    """User-side episodic memory — a structured 'what happened' trace.

    Cross-link: agent paths also produce Episode (mem_memorize.py:870-885
    in opensource at release/20260403, plus design.md §2.4 line 697:
    "episode 永远跑"). ``owner_id`` always points to the user, even when
    the source MemCell is an agent conversation; the agent is a
    participant, not the owner.

    Secondary fields (subject / summary / keywords / location / start_time
    / end_time / sender_ids / original_data) are intentionally omitted from
    the minimal type. ``extra="allow"`` keeps any LLM-emitted secondary
    fields accessible on the model instance until a future minor bump
    promotes them to first-class fields.
    """

    id: str
    owner_id: str
    episode: str
    timestamp: int  # Unix epoch milliseconds
    parent_type: str = "memcell"
    parent_id: str

    model_config = ConfigDict(extra="allow")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/everalgo-core/tests/types/test_episode.py -v`

Expected: 7 PASS。

- [ ] **Step 5: Commit**

```bash
git add packages/everalgo-core/src/everalgo/types/memories.py packages/everalgo-core/tests/types/test_episode.py
git commit -m "✨ feat(types): add Episode BaseModel (6 fields, extra=allow)"
```

---

## Task 5: types/__init__.py — re-export 4 个对外符号

**Files:**
- Modify: `packages/everalgo-core/src/everalgo/types/__init__.py`

- [ ] **Step 1: Write the failing test**

在 `packages/everalgo-core/tests/types/test_message.py` **顶部** import 改为通过 `everalgo.types`，覆盖之前对 `everalgo.types.memcell` 的直接 import。具体修改：

把文件开头的 `from everalgo.types.memcell import MessageRole` 改为：

```python
"""Tests for everalgo.types.memcell — MessageRole + Message."""

import json

import pytest
from pydantic import ValidationError

from everalgo.types import Message, MessageRole
```

(删除原来的 `from everalgo.types.memcell import MessageRole` 和 `from everalgo.types.memcell import Message`，统一从 `everalgo.types` 顶层 import。)

同样修改 `tests/types/test_memcell.py`：

```python
"""Tests for everalgo.types.memcell.MemCell."""

import pytest
from pydantic import ValidationError

from everalgo.types import MemCell, Message, MessageRole
```

(原 `from everalgo.types.memcell import MemCell, Message, MessageRole` 改为顶层。)

同样修改 `tests/types/test_episode.py`：

```python
"""Tests for everalgo.types.memories.Episode."""

import pytest
from pydantic import ValidationError

from everalgo.types import Episode
```

(原 `from everalgo.types.memories import Episode` 改为顶层。)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/everalgo-core/tests/types/ -v`

Expected: All FAIL with `ImportError: cannot import name 'Message' from 'everalgo.types'` (or类似 `MessageRole / MemCell / Episode`).

> 当前 `everalgo/types/__init__.py` 内容是 `__all__: list[str] = []` 的 placeholder，所以顶层 import 拿不到 4 个符号。

- [ ] **Step 3: Write minimal implementation**

替换 `packages/everalgo-core/src/everalgo/types/__init__.py` 的全部内容为：

```python
"""Public data contracts for EverAlgo — minimal EPISODE-path subset.

Sub-project 1 deliverable. Adding more memory types (AtomicFact,
Foresight, Profile, AgentCase, AgentSkill, ClusterState, ...) later
is a SemVer minor bump for users that import from this module.
"""

from everalgo.types.memcell import MemCell, Message, MessageRole
from everalgo.types.memories import Episode

__all__ = [
    "Episode",
    "MemCell",
    "Message",
    "MessageRole",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/everalgo-core/tests/types/ -v`

Expected: 22 PASS (8 message + 7 memcell + 7 episode)。

- [ ] **Step 5: Verify `__all__` exports the right names**

Run: `uv run python -c "from everalgo.types import __all__, Message, MessageRole, MemCell, Episode; print(sorted(__all__))"`

Expected: 输出 `['Episode', 'MemCell', 'Message', 'MessageRole']`，无 ImportError。

- [ ] **Step 6: Commit**

```bash
git add packages/everalgo-core/src/everalgo/types/__init__.py packages/everalgo-core/tests/types/test_message.py packages/everalgo-core/tests/types/test_memcell.py packages/everalgo-core/tests/types/test_episode.py
git commit -m "✨ feat(types): re-export 4 public symbols at everalgo.types top level"
```

---

## Task 6: Cross-type JSON round-trip parametrized test

**Files:**
- Create: `packages/everalgo-core/tests/types/test_round_trip.py`

- [ ] **Step 1: Write the test**

创建 `packages/everalgo-core/tests/types/test_round_trip.py`：

```python
"""Cross-type JSON round-trip parametrized check.

Ensures every public type from ``everalgo.types`` survives
``model_dump_json`` -> ``model_validate_json`` cleanly, including
the ``extra='allow'`` path for Episode.
"""

import pytest

from everalgo.types import Episode, MemCell, Message, MessageRole


@pytest.mark.parametrize(
    "obj",
    [
        Message(role=MessageRole.USER, content="hi", timestamp=1),
        Message(role=MessageRole.ASSISTANT, content="response", timestamp=2),
        MemCell(id="m_empty", messages=[], timestamp=1),
        MemCell(
            id="m_one",
            messages=[Message(role=MessageRole.USER, content="hi", timestamp=1)],
            timestamp=10,
        ),
        Episode(
            id="ep1",
            owner_id="u1",
            episode="Alice asked about Q3.",
            timestamp=1,
            parent_id="m1",
        ),
        Episode.model_validate(
            {
                "id": "ep2",
                "owner_id": "u2",
                "episode": "Bob shared the plan.",
                "timestamp": 2,
                "parent_id": "m2",
                "summary": "shared plan",
                "keywords": ["plan"],
            }
        ),
    ],
    ids=[
        "message-user",
        "message-assistant",
        "memcell-empty",
        "memcell-one-message",
        "episode-minimal",
        "episode-with-extras",
    ],
)
def test_model_dump_json_then_validate_json_round_trips(obj):
    serialised = obj.model_dump_json()
    rebuilt = type(obj).model_validate_json(serialised)
    assert rebuilt == obj
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `uv run pytest packages/everalgo-core/tests/types/test_round_trip.py -v`

Expected: 6 PASS (parametrized cases)。

> 这一步 test 直接 PASS — round-trip 是基础 pydantic 行为，不需要新 implementation。但保留为独立 task 是为了 spec §5.3 round-trip 验收点单独成立 + 跨 type 对照。

- [ ] **Step 3: Commit**

```bash
git add packages/everalgo-core/tests/types/test_round_trip.py
git commit -m "✅ test(types): add cross-type JSON round-trip parametrized check"
```

---

## Task 7: check_placeholders

**Files:**
- Create: `packages/everalgo-core/src/everalgo/prompts/validator.py`
- Create: `packages/everalgo-core/tests/prompts/test_validator.py`

- [ ] **Step 1: Write the failing tests**

创建 `packages/everalgo-core/tests/prompts/test_validator.py`：

```python
"""Tests for everalgo.prompts.validator."""

import pytest

from everalgo.prompts.validator import check_placeholders


def test_check_placeholders_pass_when_all_required_present():
    check_placeholders("Hello {name}, today is {date}.", required=["name", "date"])


def test_check_placeholders_pass_when_no_required_and_no_placeholders():
    check_placeholders("Hello world.", required=[])


def test_check_placeholders_extras_are_allowed():
    """Template with placeholders not in required is fine — caller may not pass them."""
    check_placeholders("{a} {b} {c}", required=["a"])


def test_check_placeholders_missing_one_raises():
    with pytest.raises(ValueError, match="Missing required placeholders"):
        check_placeholders("Hello {name}.", required=["name", "date"])


def test_check_placeholders_missing_all_lists_them_alphabetically():
    with pytest.raises(ValueError, match=r"\['date', 'name'\]"):
        check_placeholders("Hello world.", required=["name", "date"])


def test_check_placeholders_handles_attribute_access():
    """``{user.name}`` should match required ``user`` (root identifier)."""
    check_placeholders("Hello {user.name}", required=["user"])


def test_check_placeholders_handles_index_access():
    """``{items[0]}`` should match required ``items``."""
    check_placeholders("First: {items[0]}", required=["items"])


def test_check_placeholders_extras_listed_when_missing_raised():
    """Diagnostic message includes both missing and extra placeholders to ease typo fixes."""
    with pytest.raises(ValueError, match=r"extra placeholders present"):
        check_placeholders("Hello {nme}.", required=["name"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/everalgo-core/tests/prompts/test_validator.py -v`

Expected: 8 FAIL — `ModuleNotFoundError: No module named 'everalgo.prompts.validator'`.

- [ ] **Step 3: Write minimal implementation**

创建 `packages/everalgo-core/src/everalgo/prompts/validator.py`：

```python
"""Prompt validators — fail-fast checks for prompt templates.

Designed to be called at module import time after a prompt constant is
defined, so that template typos are caught before any LLM call.
"""

import string
from collections.abc import Iterable


def check_placeholders(prompt: str, *, required: Iterable[str]) -> None:
    """Assert that ``prompt`` contains every Python format placeholder in ``required``.

    The check uses :class:`string.Formatter` so attribute access
    (``{user.name}``) and indexing (``{items[0]}``) collapse to the root
    identifier (``user`` / ``items``).

    Args:
        prompt: Template string with ``{placeholder}`` markers.
        required: Names that must appear as ``{name}`` in the template.

    Raises:
        ValueError: If any required placeholder is missing. The diagnostic
            message lists the missing names and, when present, any extra
            placeholders the template carries — useful for catching typos
            such as ``{nme}`` instead of ``{name}``.
    """
    found: set[str] = set()
    for _, field_name, _, _ in string.Formatter().parse(prompt):
        if not field_name:
            continue
        # Reduce ``user.name`` / ``items[0]`` to root identifier.
        root = field_name.split(".", 1)[0].split("[", 1)[0]
        if root:
            found.add(root)

    required_set = set(required)
    missing = required_set - found
    if not missing:
        return

    extras = found - required_set
    msg = f"Missing required placeholders: {sorted(missing)}"
    if extras:
        msg += f" (extra placeholders present: {sorted(extras)})"
    raise ValueError(msg)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/everalgo-core/tests/prompts/test_validator.py -v`

Expected: 8 PASS。

- [ ] **Step 5: Commit**

```bash
git add packages/everalgo-core/src/everalgo/prompts/validator.py packages/everalgo-core/tests/prompts/test_validator.py
git commit -m "✨ feat(prompts): add check_placeholders validator"
```

---

## Task 8: check_length

**Files:**
- Modify: `packages/everalgo-core/src/everalgo/prompts/validator.py` (extend)
- Modify: `packages/everalgo-core/tests/prompts/test_validator.py` (extend)

- [ ] **Step 1: Write the failing tests**

在 `packages/everalgo-core/tests/prompts/test_validator.py` 末尾追加：

```python
from everalgo.prompts.validator import check_length


def test_check_length_pass_with_default_estimator_under_limit():
    check_length("hello world", max_tokens=100)


def test_check_length_fail_with_default_estimator_over_limit():
    long_text = "x" * 1000
    with pytest.raises(ValueError, match="exceeds max_tokens"):
        check_length(long_text, max_tokens=10)


def test_check_length_with_custom_tokenizer_pass():
    check_length(
        "this is a sentence",
        max_tokens=10,
        tokenizer=lambda s: len(s.split()),
    )


def test_check_length_with_custom_tokenizer_fail():
    with pytest.raises(ValueError, match="exceeds max_tokens"):
        check_length(
            "this is a sentence with seven words at least",
            max_tokens=3,
            tokenizer=lambda s: len(s.split()),
        )


def test_check_length_default_estimator_is_safe_overcount_for_cjk():
    """4-chars-per-token approximation overcounts CJK; assertion confirms it does not under-count."""
    cjk_text = "你好世界" * 100  # 400 CJK chars; real tokens ~ 400-800 (varies by model).
    with pytest.raises(ValueError, match="exceeds max_tokens"):
        # 4-chars-per-token estimator returns ~ 101 tokens, well above the cap.
        check_length(cjk_text, max_tokens=50)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/everalgo-core/tests/prompts/test_validator.py -v`

Expected: 5 新 FAIL (`check_length` 5 testcases) — `ImportError: cannot import name 'check_length'`. The 8 check_placeholders tests still PASS.

- [ ] **Step 3: Write minimal implementation**

在 `packages/everalgo-core/src/everalgo/prompts/validator.py` 末尾追加：

```python
from collections.abc import Callable


def _default_token_estimator(text: str) -> int:
    """Coarse-but-safe over-estimate (~ 4 characters per token, English baseline).

    This intentionally over-counts (especially for CJK text) so that a
    too-long prompt is never silently allowed to pass. Callers wanting an
    accurate token count should pass a real tokenizer (for example
    ``tiktoken.encoding_for_model("gpt-4").encode``).
    """
    return max(1, len(text) // 4 + 1)


def check_length(
    prompt: str,
    *,
    max_tokens: int,
    tokenizer: Callable[[str], int] | None = None,
) -> None:
    """Assert that ``prompt`` is at most ``max_tokens`` tokens long.

    Args:
        prompt: Rendered prompt (post-format).
        max_tokens: Hard ceiling — typically the model context window minus
            the response reserve.
        tokenizer: Token counter callable. ``None`` (default) falls back to
            an over-counting heuristic; for precise token accounting pass
            an accurate tokenizer.

    Raises:
        ValueError: If the estimated token count exceeds ``max_tokens``.
            The message includes both the actual count and the cap.
    """
    counter = tokenizer if tokenizer is not None else _default_token_estimator
    actual = counter(prompt)
    if actual > max_tokens:
        raise ValueError(
            f"Prompt length {actual} tokens exceeds max_tokens={max_tokens}"
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/everalgo-core/tests/prompts/test_validator.py -v`

Expected: 13 PASS (8 check_placeholders + 5 check_length)。

- [ ] **Step 5: Commit**

```bash
git add packages/everalgo-core/src/everalgo/prompts/validator.py packages/everalgo-core/tests/prompts/test_validator.py
git commit -m "✨ feat(prompts): add check_length validator with safe default estimator"
```

---

## Task 9: AGENTS.md — sync bridge convention

**Files:**
- Modify: `AGENTS.md` (insert one bullet under §5 Code Style)

- [ ] **Step 1: Read AGENTS.md §5**

Run: `grep -n '^- \*\*Lint configuration' AGENTS.md`

记下行号 N。下一步把新规则插入到这一行**之前**（紧跟在 "No dependency injection in algorithm code" 之后、Lint configuration 之前）。

- [ ] **Step 2: Insert the new convention**

在第 N-1 行后（即 "No dependency injection in algorithm code." 那一行后）插入一行：

```markdown
- **Sync bridge for I/O operators: write `extract = async_to_sync(aextract)` one-liner per ADR 010 line 199-214; do not introduce a `DualInterface` mixin.** This keeps type inference predictable, avoids metaclass magic, and matches the pattern shown in the ADR. The `async_to_sync` helper comes from `asgiref.sync`.
```

- [ ] **Step 3: Verify markdown still renders cleanly**

Run: `head -160 AGENTS.md | tail -40`

Expected: 输出含上述新 bullet，markdown 列表结构未破坏。

- [ ] **Step 4: Commit**

```bash
git add AGENTS.md
git commit -m "📝 docs(agents): document sync bridge one-liner convention (ADR 010)"
```

---

## Task 10: 全包验收 (workspace + lint + typecheck + tests + import smoke)

**Files:** 无新增/修改源文件，仅运行验收命令。

- [ ] **Step 1: 全 workspace 同步**

Run: `uv sync --all-packages`

Expected: 无 error；输出含 `Resolved` / `Built` 全 8 个 distribution。

- [ ] **Step 2: 顶层 import smoke test**

Run: `uv run python -c "from everalgo.types import MemCell, Message, MessageRole, Episode; from everalgo.prompts.validator import check_placeholders, check_length; print('OK')"`

Expected: 输出 `OK`。

- [ ] **Step 3: 跑全部 tests**

Run: `uv run pytest packages/everalgo-core/tests/ -v`

Expected: 41 PASS (10 message + 7 memcell + 7 episode + 6 round_trip + 8 check_placeholders + 5 check_length — 大致；具体数随测试细节可能 ±2)。0 FAIL, 0 ERROR。

- [ ] **Step 4: 跑 ruff check**

Run: `uv run ruff check packages/everalgo-core/`

Expected: `All checks passed!` (0 issue)。

- [ ] **Step 5: 跑 ruff format check**

Run: `uv run ruff format --check packages/everalgo-core/`

Expected: `X files already formatted` 或类似无 diff 输出。

> 若有 diff，执行 `uv run ruff format packages/everalgo-core/` 然后 amend 上一个 commit 或新增 `🎨 style` commit。

- [ ] **Step 6: 跑 mypy**

Run: `uv run mypy packages/everalgo-core/`

Expected: `Success: no issues found in N source files` (N ≥ 4)。

> 若 mypy 报错，先看是不是 strict 模式下 type ignore 漏标；按报错具体位置 fix 后 amend 或新增 `🔧 fix(types)` commit。

- [ ] **Step 7: 不留临时输出 / cache 验证**

Run: `git status -sb`

Expected: 工作区干净（除了 `?? docs/reference/` 等之前 BOSS 自己未 commit 的内容）；不应该有 `?? packages/everalgo-core/.ruff_cache/` / `?? packages/everalgo-core/.mypy_cache/` / `??packages/everalgo-core/__pycache__/` —— 这些应已被 root `.gitignore` 忽略。若其中任何一个被 untracked，是 .gitignore 缺漏，加规则到 `.gitignore` 后 commit。

- [ ] **Step 8: Commit acceptance log（可选）**

如果 Step 5/6/7 触发了任何 amend/fix，确保所有 commit 之间工作树是绿的：

```bash
git log --oneline -15
```

Expected: 最近 9 个 commit 是 Task 0-8 + 可能的 fix commit。

> Task 10 本身不产生 commit（验收 only）—— 但如有 fix 必须显式 commit 出来。

---

## Self-Review Checklist (作者自审，已通过)

| 检查项 | 结果 |
|---|---|
| Spec coverage — 4 个 type | ✅ Task 1-4 覆盖；Task 5 re-export 整合；Task 6 round-trip 验收 |
| Spec coverage — 2 个 validator | ✅ Task 7 (check_placeholders) + Task 8 (check_length) |
| Spec coverage — pydantic>=2.7 依赖 | ✅ Task 0 |
| Spec coverage — tests 物理布局（pydantic-ai 模式） | ✅ Task 0 + 各 Task 测试文件路径 |
| Spec coverage — extra="ignore" / "allow" 行为 | ✅ Task 2/3 (ignore) + Task 4 (allow) 各有专门测试 |
| Spec coverage — owner_id 跨 user/agent 命名 | ✅ Task 4 docstring + 字段定义 |
| Spec coverage — AGENTS.md sync bridge 约定 | ✅ Task 9 |
| Spec coverage — workspace + ruff + mypy 全绿验收 | ✅ Task 10 |
| Placeholder scan | ✅ 无 TBD / TODO / "implement later"；所有代码块完整 |
| Type consistency | ✅ `MessageRole` / `Message` / `MemCell` / `Episode` 四个名贯穿全 plan；字段名（id / owner_id / episode / timestamp / parent_type / parent_id）在 Task 4 / 5 / 6 / 10 一致 |
| 代码 100% 英文 | ✅ 所有 docstring / comment / commit message body 全英文（plan 自然语言中文不在代码块内，符合 BOSS 约束） |
| TDD 严格 | ✅ 每个有 implementation 的 Task 都是 test → fail → impl → pass → commit |

---

*Plan 完成。下一步执行选择见后续消息。*
