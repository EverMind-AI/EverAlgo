# EverAlgo Reference Implementation + CI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land minimal reference implementations of `ChatMemCellExtractor` (in `everalgo-boundary`) and `EpisodeExtractor` (in `everalgo-user-memory`) following the framework defined by ADR 010 + ADR 011 + design.md §1.2 / §2.3, plus upgrade `.gitlab-ci.yml` to strict mode (4 quality gates).

**Architecture:** Stateless callable classes (`ChatMemCellExtractor.adetect` / `EpisodeExtractor.aextract`) with one-line `extract = async_to_sync(aextract)` sync bridges. LLM injection via sub-project 2.5's `everalgo.llm.resolve()` + per-call `llm=` argument. Prompts as Python module constants in `prompts/{en,zh}/<op>.py` (per design.md §1.4) with per-call `prompt=` override + caller monkey-patch paths. Token counting via simplified `len(text) // 4` heuristic (zero new dependencies beyond `asgiref` for sync bridging).

**Tech Stack:** Python 3.12 / pydantic v2 / asgiref ≥3.0 (sync bridge) / pytest + pytest-asyncio (auto mode) / ruff / mypy strict / GitLab CI / **0** new runtime dependencies beyond `asgiref`.

---

## 关键约束（必读，避免子项目 1+2+3+2.5 的重复返工）

1. **测试函数必须 `-> None` 注解**（mypy strict 模式 `tests.*` override 不兜底）。
2. **`tests/__init__.py` / `tests/<subpkg>/__init__.py` / `tests/integration/__init__.py` 全部不要创建**（沿子项目 1+2+3+2.5 的 `--import-mode=importlib` 决策）。
3. **`mypy` 必须从仓库根（`/Users/admin/Documents/evermemos/everalgo`）跑**，**不要**从 `packages/<X>/` 子目录跑（会触发 `import-untyped` 误报）。
4. **使用 `uv run` 跑所有 Python 命令**（这是 uv workspace；裸 `pytest` 用错解释器）。
5. **commit 风格 `<emoji> <type>(<scope>): <description>`** 全英文（per memory `feedback_everalgo_commit_message_english.md`），scope 用 `boundary` / `user_memory` / `ci` 等。
6. **每个 commit 落 `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`**。
7. **测试文件名必须工作区唯一** —— 用 `test_boundary_public_api.py` / `test_user_memory_public_api.py` / `test_chat.py` / `test_episode.py` / `test_tokenize.py` / `test_boundary_to_episode_e2e.py`（per memory `feedback_test_module_name_unique.md`）。
8. **prompt 文件命名后缀 `_EN` / `_ZH`** —— 常量用 `CHAT_BOUNDARY_DETECT_PROMPT_EN` / `CHAT_BOUNDARY_DETECT_PROMPT_ZH`（per spec §4.1）。
9. **算子是 stateless class** —— 无 `__init__` / 无 instance state；用户 `EpisodeExtractor()` 直接实例化。
10. **prompt 内的 JSON braces 在 f-string 模板中要 escape** —— prompt 模板用 `.format()` 渲染，JSON 例子里的 `{` / `}` 要写成 `{{` / `}}`（否则 `.format` 会尝试解析为字段名报 KeyError）。

---

## File Structure

```
packages/everalgo-boundary/
├── pyproject.toml                                # MODIFY: T0 加 asgiref dep
├── src/everalgo/boundary/
│   ├── __init__.py                               # MODIFY: T4 re-export ChatMemCellExtractor
│   ├── chat.py                                   # NEW: T3 ChatMemCellExtractor
│   ├── _tokenize.py                              # NEW: T1 count_tokens
│   └── prompts/
│       ├── __init__.py                           # NEW: T2 (empty docstring)
│       ├── en/
│       │   ├── __init__.py                       # NEW: T2
│       │   └── chat.py                           # NEW: T2 CHAT_BOUNDARY_DETECT_PROMPT_EN
│       └── zh/
│           ├── __init__.py                       # NEW: T2
│           └── chat.py                           # NEW: T2 CHAT_BOUNDARY_DETECT_PROMPT_ZH
└── tests/boundary/
    ├── test_chat.py                              # NEW: T3 (3 tests)
    ├── test_tokenize.py                          # NEW: T1 (3 tests)
    └── test_boundary_public_api.py               # NEW: T4 (2 tests)

packages/everalgo-user-memory/
├── pyproject.toml                                # MODIFY: T5 去 everalgo-clustering + 加 asgiref
├── src/everalgo/user_memory/
│   ├── __init__.py                               # MODIFY: T7 re-export
│   ├── episode.py                                # NEW: T6 EpisodeExtractor
│   └── prompts/
│       ├── __init__.py                           # NEW: T5
│       ├── en/
│       │   ├── __init__.py                       # NEW: T5
│       │   └── episode.py                        # NEW: T5 EPISODE_EXTRACT_PROMPT_EN
│       └── zh/
│           ├── __init__.py                       # NEW: T5
│           └── episode.py                        # NEW: T5 EPISODE_EXTRACT_PROMPT_ZH
└── tests/user_memory/
    ├── test_episode.py                           # NEW: T6 (3 tests)
    └── test_user_memory_public_api.py            # NEW: T7 (2 tests)

tests/                                            # workspace-root, NO __init__.py
└── integration/                                  # NEW directory, NO __init__.py
    └── test_boundary_to_episode_e2e.py           # NEW: T8 (1 test)

.gitlab-ci.yml                                    # MODIFY: T9 4 jobs strict
```

设计依据：`docs/superpowers/specs/2026-05-08-everalgo-reference-impl-design.md` §2 (File Map) 和 §3 (公开 API)。

---

## Task 0: 项目基建 — pyproject 改 + 目录骨架

**Files:**
- Modify: `packages/everalgo-boundary/pyproject.toml`
- Modify: `packages/everalgo-user-memory/pyproject.toml`
- Create directories: `packages/everalgo-boundary/tests/boundary/`, `packages/everalgo-user-memory/tests/user_memory/`, `tests/integration/`

- [ ] **Step 1: 修改 `packages/everalgo-boundary/pyproject.toml`**

读取现有 `pyproject.toml`。找到 `dependencies` 块：

```toml
dependencies = [
  "everalgo-core>=0.1.0,<2.0.0",
]
```

替换为：

```toml
dependencies = [
  "asgiref>=3.0",
  "everalgo-core>=0.1.0,<2.0.0",
]
```

- [ ] **Step 2: 修改 `packages/everalgo-user-memory/pyproject.toml`**

找到 `dependencies` 块：

```toml
dependencies = [
  "everalgo-core>=0.1.0,<2.0.0",
  "everalgo-boundary>=0.1.0,<2.0.0",
  "everalgo-clustering>=0.1.0,<2.0.0",
]
```

替换为：

```toml
dependencies = [
  "asgiref>=3.0",
  "everalgo-boundary>=0.1.0,<2.0.0",
  "everalgo-core>=0.1.0,<2.0.0",
]
```

注意：① 删除 `everalgo-clustering` 依赖（最小集 EpisodeExtractor 不依赖 cluster，spec §7 决策 11）；② 添加 `asgiref>=3.0`；③ 字母顺序 `asgiref → everalgo-boundary → everalgo-core`。

- [ ] **Step 3: 同步 workspace 依赖**

```bash
uv sync --all-packages
```

Expected: `Resolved N packages` + 安装 `asgiref` 到 `.venv`，无错误。

- [ ] **Step 4: 创建测试目录骨架**

```bash
mkdir -p packages/everalgo-boundary/tests/boundary/
mkdir -p packages/everalgo-user-memory/tests/user_memory/
mkdir -p tests/integration/
```

**Do NOT create any `__init__.py`** — sub-project 1+2+3+2.5 沿用 `--import-mode=importlib`。

Verify:
```bash
ls -la packages/everalgo-boundary/tests/boundary/ \
       packages/everalgo-user-memory/tests/user_memory/ \
       tests/integration/
test ! -f packages/everalgo-boundary/tests/boundary/__init__.py && \
test ! -f packages/everalgo-user-memory/tests/user_memory/__init__.py && \
test ! -f tests/integration/__init__.py && \
echo "OK: no __init__.py in any test directory"
```

Expected: 3 directories exist, all empty (or `__pycache__/` only), no `__init__.py`.

- [ ] **Step 5: 验证 asgiref 可 import**

```bash
uv run python -c "from asgiref.sync import async_to_sync; print('OK:', async_to_sync)"
```

Expected: `OK: <function async_to_sync at 0x...>`.

- [ ] **Step 6: Commit**

```bash
git add packages/everalgo-boundary/pyproject.toml \
        packages/everalgo-user-memory/pyproject.toml \
        uv.lock
git commit -m "$(cat <<'EOF'
🎉 chore(deps): add asgiref + drop everalgo-clustering for sub-project 4

Add asgiref>=3.0 to everalgo-boundary and everalgo-user-memory for the
async-to-sync bridging pattern (ADR 010 line 220, sync extract = async_to_sync(
aextract) one-liner used by the new ChatMemCellExtractor and EpisodeExtractor).

Drop everalgo-clustering from everalgo-user-memory dependencies — the minimal
sub-project 4 EpisodeExtractor does not depend on cluster_id (design.md §2.3
line 540 + line 686). The dependency will be re-added when ProfileExtractor
lands in a future SemVer minor bump.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

(Empty test directories will be picked up by git when Tasks 1+ create files inside them.)

---

## Task 1: `everalgo.boundary._tokenize.count_tokens` + 3 tests

**Files:**
- Create: `packages/everalgo-boundary/src/everalgo/boundary/_tokenize.py`
- Create: `packages/everalgo-boundary/tests/boundary/test_tokenize.py`

- [ ] **Step 1: Write failing tests**

Write to `packages/everalgo-boundary/tests/boundary/test_tokenize.py`:

```python
"""Tests for everalgo.boundary._tokenize — count_tokens helper."""

from everalgo.boundary._tokenize import count_tokens


def test_count_tokens_empty_string_is_zero() -> None:
    """Empty input yields zero tokens."""
    assert count_tokens("") == 0


def test_count_tokens_short_text_proportional() -> None:
    """40 chars yields 10 tokens under the 4-char heuristic."""
    assert count_tokens("a" * 40) == 10


def test_count_tokens_returns_non_negative() -> None:
    """Any non-empty string yields a non-negative count."""
    samples = ["x", "hello world", "你好", "a" * 1000]
    for text in samples:
        assert count_tokens(text) >= 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest packages/everalgo-boundary/tests/boundary/test_tokenize.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'everalgo.boundary._tokenize'`.

- [ ] **Step 3: Write minimal implementation**

Write to `packages/everalgo-boundary/src/everalgo/boundary/_tokenize.py`:

```python
"""Token counting helper for boundary extractors.

Minimal reference implementation using a 4-character heuristic (roughly
matching English GPT tokens). For production use, replace with tiktoken or a
real tokenizer (planned for a future SemVer minor bump).

NOT exposed in __all__ — module-private utility for boundary algorithms.
"""

from __future__ import annotations

_CHARS_PER_TOKEN_HEURISTIC = 4


def count_tokens(text: str) -> int:
    """Estimate token count — minimal reference impl.

    Uses ``len(text) // CHARS_PER_TOKEN`` as a rough proxy for GPT-style
    tokenization. Accuracy is sufficient for "is this MemCell larger than
    the LLM context window?" decisions but NOT for billing / quota.

    Args:
        text: Input string. Empty string returns 0.

    Returns:
        Estimated token count (always >= 0).
    """
    return len(text) // _CHARS_PER_TOKEN_HEURISTIC
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest packages/everalgo-boundary/tests/boundary/test_tokenize.py -v
```

Expected: 3 PASS.

- [ ] **Step 5: Run quality gates (from REPO ROOT)**

```bash
uv run ruff check packages/everalgo-boundary/
uv run ruff format --check packages/everalgo-boundary/
uv run mypy packages/everalgo-boundary/
```

Expected: 全部 clean.

- [ ] **Step 6: Commit**

```bash
git add packages/everalgo-boundary/src/everalgo/boundary/_tokenize.py \
        packages/everalgo-boundary/tests/boundary/test_tokenize.py
git commit -m "$(cat <<'EOF'
✨ feat(boundary): add _tokenize.count_tokens minimal heuristic

count_tokens uses a 4-character heuristic (len(text) // 4) as a rough
proxy for GPT-style tokenization. Sufficient for boundary "is this
MemCell larger than the LLM context window?" decisions but NOT for
billing / quota — replace with tiktoken in a future SemVer minor
bump if production accuracy is needed.

Module-private (underscore prefix) per design.md §1.2 line 106 +
spec §3.2; not exposed in __all__.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `everalgo.boundary.prompts.{en,zh}.chat` — 4 prompt files

**Files:**
- Create: `packages/everalgo-boundary/src/everalgo/boundary/prompts/__init__.py`
- Create: `packages/everalgo-boundary/src/everalgo/boundary/prompts/en/__init__.py`
- Create: `packages/everalgo-boundary/src/everalgo/boundary/prompts/en/chat.py`
- Create: `packages/everalgo-boundary/src/everalgo/boundary/prompts/zh/__init__.py`
- Create: `packages/everalgo-boundary/src/everalgo/boundary/prompts/zh/chat.py`

- [ ] **Step 1: Create the prompts package + en + zh sub-packages**

Write `packages/everalgo-boundary/src/everalgo/boundary/prompts/__init__.py`:

```python
"""Boundary extractor prompts (en + zh).

Each prompt is a module-level Python string constant per design.md §1.4
(no external .md / .yaml / .toml prompt stores). Algorithm authors
customize via per-call ``prompt=`` argument or by monkey-patching the
constant at startup.
"""
```

Write `packages/everalgo-boundary/src/everalgo/boundary/prompts/en/__init__.py`:

```python
"""English boundary extractor prompts."""
```

Write `packages/everalgo-boundary/src/everalgo/boundary/prompts/zh/__init__.py`:

```python
"""Chinese boundary extractor prompts."""
```

- [ ] **Step 2: Write `prompts/en/chat.py`**

Write `packages/everalgo-boundary/src/everalgo/boundary/prompts/en/chat.py`:

```python
"""English prompt for ChatMemCellExtractor.adetect."""

CHAT_BOUNDARY_DETECT_PROMPT_EN = """You are a conversation boundary detector. Given a chat message stream, identify whether the topic shifts mid-stream and at which message index the shift occurs.

Messages:
{messages}

Token count of full stream: {token_count}

Instructions:
1. Read all messages and identify the dominant topic.
2. If a clear topic shift occurs, return the index of the FIRST message in the new topic. The index is 0-based and matches the message list.
3. If the entire stream stays on one coherent topic, return null.
4. If the stream is empty or has only one message, return null.

Output format (JSON only, no prose):
{{
  "split_at": <int | null>
}}
"""
```

**Note about JSON braces**: The `{{` and `}}` in the JSON example escape the f-string-style `.format()` braces. When `chat.py:adetect` calls `.format(messages=..., token_count=...)`, the `{messages}` and `{token_count}` placeholders are replaced, and `{{` / `}}` collapse to literal `{` / `}` in the rendered output (per Python `str.format` doc).

- [ ] **Step 3: Write `prompts/zh/chat.py`**

Write `packages/everalgo-boundary/src/everalgo/boundary/prompts/zh/chat.py`:

```python
"""Chinese prompt for ChatMemCellExtractor.adetect."""

CHAT_BOUNDARY_DETECT_PROMPT_ZH = """你是一个对话边界检测器。给定一段聊天消息流，请判断主题是否在中途切换，以及切换发生在哪条消息的位置。

消息流：
{messages}

整段消息的 token 总数：{token_count}

指令：
1. 阅读所有消息，识别主导话题。
2. 如果出现明显的话题切换，请返回新话题首条消息的索引（0-based，对应消息列表）。
3. 如果整段消息保持单一连贯话题，返回 null。
4. 如果消息流为空或仅有一条消息，返回 null。

输出格式（仅 JSON，不要前后缀）：
{{
  "split_at": <int | null>
}}
"""
```

- [ ] **Step 4: Verify imports work**

```bash
uv run python -c "
from everalgo.boundary.prompts.en.chat import CHAT_BOUNDARY_DETECT_PROMPT_EN
from everalgo.boundary.prompts.zh.chat import CHAT_BOUNDARY_DETECT_PROMPT_ZH
print('EN length:', len(CHAT_BOUNDARY_DETECT_PROMPT_EN))
print('ZH length:', len(CHAT_BOUNDARY_DETECT_PROMPT_ZH))
"
```

Expected: prints two non-zero lengths, no ImportError.

- [ ] **Step 5: Verify .format works (JSON brace escaping)**

```bash
uv run python -c "
from everalgo.boundary.prompts.en.chat import CHAT_BOUNDARY_DETECT_PROMPT_EN
rendered = CHAT_BOUNDARY_DETECT_PROMPT_EN.format(messages='[user] hi', token_count=2)
assert '{' in rendered and '\"split_at\":' in rendered, 'JSON braces did not collapse'
print('Format OK')
"
```

Expected: prints `Format OK`. If you see `KeyError` or `IndexError`, the `{{ }}` escape on JSON braces is missing.

- [ ] **Step 6: Run quality gates (from REPO ROOT)**

```bash
uv run ruff check packages/everalgo-boundary/
uv run ruff format --check packages/everalgo-boundary/
uv run mypy packages/everalgo-boundary/
```

Expected: 全部 clean.

- [ ] **Step 7: Commit**

```bash
git add packages/everalgo-boundary/src/everalgo/boundary/prompts/
git commit -m "$(cat <<'EOF'
✨ feat(boundary): add bilingual chat boundary prompts (en + zh)

Add CHAT_BOUNDARY_DETECT_PROMPT_EN and CHAT_BOUNDARY_DETECT_PROMPT_ZH
as Python module constants under prompts/{en,zh}/chat.py, per
design.md §1.4 (Python string modules, not .md / .yaml).

Each prompt asks the LLM to identify a single topic-shift split point
in a chat message stream and return JSON {"split_at": int | null}.
The {{ }} pairs around the JSON example escape str.format brace
parsing so the rendered output contains literal {/}.

Algorithm authors customize via per-call `prompt=` argument or
monkey-patch the constant at startup (design.md §1.4 line 397-410).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `everalgo.boundary.chat.ChatMemCellExtractor` + 3 tests

**Files:**
- Create: `packages/everalgo-boundary/src/everalgo/boundary/chat.py`
- Create: `packages/everalgo-boundary/tests/boundary/test_chat.py`

- [ ] **Step 1: Write failing tests**

Write to `packages/everalgo-boundary/tests/boundary/test_chat.py`:

```python
"""Tests for everalgo.boundary.chat — ChatMemCellExtractor."""

from __future__ import annotations

from typing import Any

from everalgo.boundary.chat import ChatMemCellExtractor
from everalgo.llm.types import ChatMessage as LLMChatMessage, ChatResponse
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import Message, MessageRole


def _user(content: str, ts: int = 1700000000000) -> Message:
    """Helper: build a user-role Message."""
    return Message(role=MessageRole.USER, content=content, timestamp=ts)


def _assistant(content: str, ts: int = 1700000001000) -> Message:
    """Helper: build an assistant-role Message."""
    return Message(role=MessageRole.ASSISTANT, content=content, timestamp=ts)


async def test_adetect_returns_single_memcell_when_llm_returns_no_split() -> None:
    """split_at=null in LLM response yields a single coherent MemCell."""
    fake = FakeLLMClient(
        responses=[ChatResponse(content='{"split_at": null}', model="fake")]
    )
    msgs = [_user("hello"), _assistant("hi there")]

    memcells = await ChatMemCellExtractor().adetect(msgs, llm=fake)

    assert len(memcells) == 1
    assert memcells[0].messages == msgs


async def test_adetect_returns_two_memcells_when_llm_returns_split_index() -> None:
    """split_at=2 yields two MemCells: messages[:2] and messages[2:]."""
    fake = FakeLLMClient(
        responses=[ChatResponse(content='{"split_at": 2}', model="fake")]
    )
    msgs = [
        _user("topic A part 1"),
        _assistant("topic A part 2"),
        _user("now topic B"),
        _assistant("topic B reply"),
    ]

    memcells = await ChatMemCellExtractor().adetect(msgs, llm=fake)

    assert len(memcells) == 2
    assert memcells[0].messages == msgs[:2]
    assert memcells[1].messages == msgs[2:]


async def test_adetect_per_call_prompt_overrides_default() -> None:
    """Per-call prompt= argument is the rendered prompt, not the default."""
    captured: dict[str, Any] = {}

    def handler(
        messages: list[LLMChatMessage], **kwargs: Any
    ) -> ChatResponse:
        captured["content"] = messages[0].content
        return ChatResponse(content='{"split_at": null}', model="fake")

    fake = FakeLLMClient(handler=handler)
    custom_prompt = "CUSTOM PROMPT messages={messages} tokens={token_count}"

    await ChatMemCellExtractor().adetect(
        [_user("hi")], llm=fake, prompt=custom_prompt
    )

    assert captured["content"].startswith("CUSTOM PROMPT")
    assert "[user] hi" in captured["content"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest packages/everalgo-boundary/tests/boundary/test_chat.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'everalgo.boundary.chat'`.

- [ ] **Step 3: Write `chat.py` implementation**

Write to `packages/everalgo-boundary/src/everalgo/boundary/chat.py`:

```python
"""Chat-style MemCell extractor — slice a message stream by topic boundaries."""

from __future__ import annotations

import json

from asgiref.sync import async_to_sync

import everalgo.llm
from everalgo.boundary._tokenize import count_tokens
from everalgo.boundary.prompts.en.chat import CHAT_BOUNDARY_DETECT_PROMPT_EN
from everalgo.llm.protocols import LLMClient
from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.types import MemCell, Message


class ChatMemCellExtractor:
    """Detect MemCell boundaries in a chat-style message stream.

    Stateless: no ``__init__``, no instance state. Thread/async safe —
    instances are interchangeable. Customize per call via ``llm=`` and
    ``prompt=`` arguments.

    Algorithm (minimal reference impl):
        1. Render LLM prompt with the message stream + token budget hint.
        2. Call LLM, parse JSON ``{"split_at": int | null}``.
        3. Build one or two MemCells from the split.

    For production-grade boundary detection (multi-split / token-aware
    force_split / boundary_reason classification), replace the prompt
    via ``prompt=`` argument or monkey-patch the module constant.
    """

    async def adetect(
        self,
        messages: list[Message],
        *,
        llm: LLMClient | None = None,
        prompt: str | None = None,
    ) -> list[MemCell]:
        """Async main implementation: ask LLM for boundary split point.

        Args:
            messages: Ordered chat messages (user/assistant turns).
            llm: Per-call LLM override. Falls back through scoped
                (``use(...)``) and default (``configure(...)``); raises
                ``LLMNotConfiguredError`` if all None.
            prompt: Per-call prompt override. Defaults to
                ``CHAT_BOUNDARY_DETECT_PROMPT_EN``.

        Returns:
            list[MemCell] — at least one cell. The minimal ref impl produces
            either 1 cell (no split) or 2 cells (one split point).
        """
        client = everalgo.llm.resolve(llm)
        rendered = (prompt or CHAT_BOUNDARY_DETECT_PROMPT_EN).format(
            messages=_format_messages_for_prompt(messages),
            token_count=count_tokens(_concat_messages(messages)),
        )
        response = await client.chat(
            messages=[LLMChatMessage(role="user", content=rendered)],
            response_format={"type": "json_object"},
        )
        return _build_memcells_from_llm_response(response.content, messages)

    detect = async_to_sync(adetect)
    """Sync bridge — only callable from non-event-loop contexts."""


# Module-level helper functions — stateless utilities (per AGENTS.md §5).


def _concat_messages(messages: list[Message]) -> str:
    """Concatenate messages into a single prompt-friendly string."""
    return "\n".join(f"[{m.role.value}] {m.content}" for m in messages)


def _format_messages_for_prompt(messages: list[Message]) -> str:
    """Format messages with index prefix (LLM uses index for split_at)."""
    return "\n".join(
        f"{i}. [{m.role.value}] {m.content}" for i, m in enumerate(messages)
    )


def _build_memcells_from_llm_response(
    raw: str, messages: list[Message]
) -> list[MemCell]:
    """Parse LLM JSON ``{"split_at": int | null}`` and build MemCell list."""
    parsed = json.loads(raw)
    split_at = parsed.get("split_at")
    if split_at is None:
        return [_make_memcell(messages, suffix="0")]
    return [
        _make_memcell(messages[:split_at], suffix="0"),
        _make_memcell(messages[split_at:], suffix="1"),
    ]


def _make_memcell(slice_msgs: list[Message], *, suffix: str) -> MemCell:
    """Build a MemCell with deterministic id derived from timestamp + suffix."""
    timestamp = slice_msgs[-1].timestamp if slice_msgs else 0
    return MemCell(
        id=f"mc_{timestamp}_{suffix}",
        messages=slice_msgs,
        timestamp=timestamp,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest packages/everalgo-boundary/tests/boundary/test_chat.py -v
```

Expected: 3 PASS (4 tokenize + 3 chat = 6 total once we run the whole boundary tests dir).

- [ ] **Step 5: Run quality gates (from REPO ROOT)**

```bash
uv run ruff check packages/everalgo-boundary/
uv run ruff format --check packages/everalgo-boundary/
uv run mypy packages/everalgo-boundary/
```

Expected: 全部 clean.

- [ ] **Step 6: Commit**

```bash
git add packages/everalgo-boundary/src/everalgo/boundary/chat.py \
        packages/everalgo-boundary/tests/boundary/test_chat.py
git commit -m "$(cat <<'EOF'
✨ feat(boundary): add ChatMemCellExtractor minimal reference impl

ChatMemCellExtractor follows the framework defined in ADR 010 line
205-212: stateless class + async `adetect` + one-line `detect =
async_to_sync(adetect)` sync bridge. LLM injection via
sub-project 2.5's everalgo.llm.resolve(). Prompt via per-call
override or default CHAT_BOUNDARY_DETECT_PROMPT_EN.

Algorithm: render prompt with messages + token count, call LLM, parse
JSON {"split_at": int | null}, build 1-2 MemCells. Module-level
helpers (_concat_messages / _format_messages_for_prompt /
_build_memcells_from_llm_response / _make_memcell) keep the class
body focused on the public contract.

For production multi-split / force_split / boundary_reason, override
the prompt or monkey-patch the constant.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `everalgo.boundary.__init__` re-export + `test_boundary_public_api.py`

**Files:**
- Modify: `packages/everalgo-boundary/src/everalgo/boundary/__init__.py`
- Create: `packages/everalgo-boundary/tests/boundary/test_boundary_public_api.py`

- [ ] **Step 1: Write failing tests**

Write to `packages/everalgo-boundary/tests/boundary/test_boundary_public_api.py`:

```python
"""Tests for everalgo.boundary package-level public API."""

import everalgo.boundary


def test_chat_memcell_extractor_exported_at_top_level() -> None:
    """ChatMemCellExtractor is accessible via attribute on the package."""
    assert hasattr(everalgo.boundary, "ChatMemCellExtractor")


def test_dunder_all_lists_exactly_one_symbol() -> None:
    """Sub-project 4 minimal exposes exactly 1 symbol from boundary."""
    assert everalgo.boundary.__all__ == ["ChatMemCellExtractor"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest packages/everalgo-boundary/tests/boundary/test_boundary_public_api.py -v
```

Expected: FAIL — current `__init__.py` is the empty placeholder, no `ChatMemCellExtractor` attribute.

- [ ] **Step 3: Replace `__init__.py`**

Replace contents of `packages/everalgo-boundary/src/everalgo/boundary/__init__.py` with:

```python
"""Boundary extractors — chat (sub-project 4 minimal).

workspace + agent extractors are out of sub-project 4 scope and will land
in a future SemVer minor bump (see spec §10).

Public surface (sub-project 4 minimal):
- ChatMemCellExtractor — slice chat messages into MemCells
"""

from everalgo.boundary.chat import ChatMemCellExtractor

__all__ = ["ChatMemCellExtractor"]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest packages/everalgo-boundary/tests/boundary/ -v
```

Expected: 8 PASS (3 tokenize + 3 chat + 2 public_api).

- [ ] **Step 5: Run quality gates (from REPO ROOT)**

```bash
uv run ruff check packages/everalgo-boundary/
uv run ruff format --check packages/everalgo-boundary/
uv run mypy packages/everalgo-boundary/
```

Expected: 全部 clean.

- [ ] **Step 6: Verify the user-facing import path**

```bash
uv run python -c "
from everalgo.boundary import ChatMemCellExtractor
print('OK:', ChatMemCellExtractor)
"
```

Expected: prints `OK: <class 'everalgo.boundary.chat.ChatMemCellExtractor'>`.

- [ ] **Step 7: Commit**

```bash
git add packages/everalgo-boundary/src/everalgo/boundary/__init__.py \
        packages/everalgo-boundary/tests/boundary/test_boundary_public_api.py
git commit -m "$(cat <<'EOF'
✨ feat(boundary): re-export ChatMemCellExtractor at everalgo.boundary top

Replace the placeholder __init__.py with the final sub-project 4
public surface: a single ChatMemCellExtractor symbol re-exported from
everalgo.boundary.chat. Two access paths now work (per design.md §1.2
line 75-78):
- `from everalgo.boundary import ChatMemCellExtractor` (EverOS contract)
- `from everalgo.boundary.chat import ChatMemCellExtractor` (algorithm
  authors iterating the boundary algorithm family)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: `everalgo.user_memory.prompts.{en,zh}.episode` — 4 prompt files

**Files:**
- Create: `packages/everalgo-user-memory/src/everalgo/user_memory/prompts/__init__.py`
- Create: `packages/everalgo-user-memory/src/everalgo/user_memory/prompts/en/__init__.py`
- Create: `packages/everalgo-user-memory/src/everalgo/user_memory/prompts/en/episode.py`
- Create: `packages/everalgo-user-memory/src/everalgo/user_memory/prompts/zh/__init__.py`
- Create: `packages/everalgo-user-memory/src/everalgo/user_memory/prompts/zh/episode.py`

- [ ] **Step 1: Create prompts package structure**

Write `packages/everalgo-user-memory/src/everalgo/user_memory/prompts/__init__.py`:

```python
"""User-memory extractor prompts (en + zh).

Each prompt is a module-level Python string constant per design.md §1.4.
Algorithm authors customize via per-call ``prompt=`` argument or by
monkey-patching the constant at startup.
"""
```

Write `packages/everalgo-user-memory/src/everalgo/user_memory/prompts/en/__init__.py`:

```python
"""English user-memory extractor prompts."""
```

Write `packages/everalgo-user-memory/src/everalgo/user_memory/prompts/zh/__init__.py`:

```python
"""Chinese user-memory extractor prompts."""
```

- [ ] **Step 2: Write `prompts/en/episode.py`**

Write `packages/everalgo-user-memory/src/everalgo/user_memory/prompts/en/episode.py`:

```python
"""English prompt for EpisodeExtractor.aextract."""

EPISODE_EXTRACT_PROMPT_EN = """You are an episodic memory generation expert. Given a conversation slice (MemCell), extract structured Episode memories.

Conversation:
{memcell_text}

Conversation timestamp (Unix epoch ms): {timestamp}

Instructions:
1. Identify each distinct episodic event — a complete "what happened" trace with participants, place, time, action, outcome.
2. Convert dialogue format into third-person narrative.
3. Preserve names, dates, locations, decisions, emotions.
4. Use the conversation timestamp as the episode time anchor.
5. Generate a unique id for each episode (e.g. "ep_<random>").
6. Use a stable owner_id from the conversation (default "u_default" if unclear).

Output format (JSON only, no prose):
{{
  "episodes": [
    {{
      "id": "<string>",
      "owner_id": "<string>",
      "episode": "<narrative text>",
      "timestamp": <int>
    }}
  ]
}}

Note: parent_type and parent_id will be auto-filled by the caller; do not emit them.
"""
```

- [ ] **Step 3: Write `prompts/zh/episode.py`**

Write `packages/everalgo-user-memory/src/everalgo/user_memory/prompts/zh/episode.py`:

```python
"""Chinese prompt for EpisodeExtractor.aextract."""

EPISODE_EXTRACT_PROMPT_ZH = """你是一名情景记忆生成专家。给定一段对话切片（MemCell），请提取结构化的 Episode 记忆。

对话内容：
{memcell_text}

对话时间戳（Unix epoch 毫秒）：{timestamp}

指令：
1. 识别每个独立的情景事件 —— 一段完整的「发生了什么」轨迹，包含参与者、地点、时间、行为、结果。
2. 将对话形式转换为第三人称叙述。
3. 保留人名、日期、地点、决策、情感。
4. 用对话时间戳作为情景的时间锚点。
5. 为每个 Episode 生成唯一 id（如 "ep_<random>"）。
6. 从对话中选取稳定的 owner_id（如不明则默认 "u_default"）。

输出格式（仅 JSON，不带前后缀）：
{{
  "episodes": [
    {{
      "id": "<string>",
      "owner_id": "<string>",
      "episode": "<叙述文本>",
      "timestamp": <int>
    }}
  ]
}}

注意：parent_type 和 parent_id 由调用方自动补齐，不要输出。
"""
```

- [ ] **Step 4: Verify imports work**

```bash
uv run python -c "
from everalgo.user_memory.prompts.en.episode import EPISODE_EXTRACT_PROMPT_EN
from everalgo.user_memory.prompts.zh.episode import EPISODE_EXTRACT_PROMPT_ZH
print('EN length:', len(EPISODE_EXTRACT_PROMPT_EN))
print('ZH length:', len(EPISODE_EXTRACT_PROMPT_ZH))
"
```

Expected: prints two non-zero lengths, no ImportError.

- [ ] **Step 5: Verify .format works**

```bash
uv run python -c "
from everalgo.user_memory.prompts.en.episode import EPISODE_EXTRACT_PROMPT_EN
rendered = EPISODE_EXTRACT_PROMPT_EN.format(memcell_text='[user] hi', timestamp=1700000000000)
assert '\"episodes\":' in rendered
print('Format OK')
"
```

Expected: prints `Format OK`.

- [ ] **Step 6: Run quality gates (from REPO ROOT)**

```bash
uv run ruff check packages/everalgo-user-memory/
uv run ruff format --check packages/everalgo-user-memory/
uv run mypy packages/everalgo-user-memory/
```

Expected: 全部 clean.

- [ ] **Step 7: Commit**

```bash
git add packages/everalgo-user-memory/src/everalgo/user_memory/prompts/
git commit -m "$(cat <<'EOF'
✨ feat(user_memory): add bilingual episode extract prompts (en + zh)

Add EPISODE_EXTRACT_PROMPT_EN and EPISODE_EXTRACT_PROMPT_ZH as Python
module constants under prompts/{en,zh}/episode.py. Each asks the LLM
to produce JSON {"episodes": [{"id", "owner_id", "episode",
"timestamp"}, ...]}. parent_type / parent_id are auto-filled by the
caller — the prompt explicitly tells the LLM not to emit them.

Following the same pattern as boundary/prompts (sub-project 4 Task 2):
{{ }} pairs escape str.format brace parsing, and per-call `prompt=`
argument overrides the default at runtime.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: `everalgo.user_memory.episode.EpisodeExtractor` + 3 tests

**Files:**
- Create: `packages/everalgo-user-memory/src/everalgo/user_memory/episode.py`
- Create: `packages/everalgo-user-memory/tests/user_memory/test_episode.py`

- [ ] **Step 1: Write failing tests**

Write to `packages/everalgo-user-memory/tests/user_memory/test_episode.py`:

```python
"""Tests for everalgo.user_memory.episode — EpisodeExtractor."""

from __future__ import annotations

from typing import Any

from everalgo.llm.types import ChatMessage as LLMChatMessage, ChatResponse
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import MemCell, Message, MessageRole
from everalgo.user_memory.episode import EpisodeExtractor


def _memcell() -> MemCell:
    """Helper: build a minimal MemCell."""
    return MemCell(
        id="mc_test_001",
        messages=[
            Message(
                role=MessageRole.USER,
                content="Schedule a meeting with Alice at 3pm",
                timestamp=1700000000000,
            ),
            Message(
                role=MessageRole.ASSISTANT,
                content="Done. Sent invite for 3pm.",
                timestamp=1700000001000,
            ),
        ],
        timestamp=1700000001000,
    )


async def test_aextract_returns_episode_list_from_llm_json() -> None:
    """Valid LLM JSON yields a list[Episode] with all fields populated."""
    llm_json = (
        '{"episodes": [{"id": "ep_001", '
        '"owner_id": "u_alice", '
        '"episode": "User scheduled a meeting with Alice at 3pm.", '
        '"timestamp": 1700000000000}]}'
    )
    fake = FakeLLMClient(
        responses=[ChatResponse(content=llm_json, model="fake")]
    )

    episodes = await EpisodeExtractor().aextract(_memcell(), llm=fake)

    assert len(episodes) == 1
    ep = episodes[0]
    assert ep.id == "ep_001"
    assert ep.owner_id == "u_alice"
    assert "Alice" in ep.episode
    assert ep.timestamp == 1700000000000


async def test_aextract_auto_fills_parent_id_from_memcell() -> None:
    """LLM-emitted JSON without parent_id gets parent_id from the source MemCell."""
    llm_json = (
        '{"episodes": [{"id": "ep_002", "owner_id": "u_x", '
        '"episode": "x", "timestamp": 1700000000000}]}'
    )
    fake = FakeLLMClient(
        responses=[ChatResponse(content=llm_json, model="fake")]
    )
    mc = _memcell()

    episodes = await EpisodeExtractor().aextract(mc, llm=fake)

    assert episodes[0].parent_id == mc.id
    assert episodes[0].parent_type == "memcell"


async def test_aextract_per_call_llm_overrides_default() -> None:
    """Per-call llm= argument is the one used by the extractor."""
    captured: dict[str, Any] = {}

    def handler(
        messages: list[LLMChatMessage], **kwargs: Any
    ) -> ChatResponse:
        captured["called"] = True
        return ChatResponse(
            content=(
                '{"episodes": [{"id": "ep_x", "owner_id": "u_x", '
                '"episode": "x", "timestamp": 1700000000000}]}'
            ),
            model="fake",
        )

    fake = FakeLLMClient(handler=handler)

    await EpisodeExtractor().aextract(_memcell(), llm=fake)

    assert captured["called"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest packages/everalgo-user-memory/tests/user_memory/test_episode.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'everalgo.user_memory.episode'`.

- [ ] **Step 3: Write `episode.py` implementation**

Write to `packages/everalgo-user-memory/src/everalgo/user_memory/episode.py`:

```python
"""Episode extractor — derive Episode memories from a single MemCell."""

from __future__ import annotations

import json

from asgiref.sync import async_to_sync

import everalgo.llm
from everalgo.llm.protocols import LLMClient
from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.types import Episode, MemCell
from everalgo.user_memory.prompts.en.episode import EPISODE_EXTRACT_PROMPT_EN


class EpisodeExtractor:
    """Extract Episode memories from a single MemCell.

    Stateless callable class. Per design.md line 687: "EpisodeExtractor —
    单 MemCell → list[Episode]" + line 697 "episode 永远跑（任何 MemCell
    类型）". This is the unconditional EPISODE-path operator.
    """

    async def aextract(
        self,
        memcell: MemCell,
        *,
        llm: LLMClient | None = None,
        prompt: str | None = None,
    ) -> list[Episode]:
        """Async main implementation: ask LLM to extract Episodes.

        Args:
            memcell: Source MemCell (boundary output).
            llm: Per-call LLM override (sub-project 2.5 fallback chain).
            prompt: Per-call prompt override; defaults to
                ``EPISODE_EXTRACT_PROMPT_EN``.

        Returns:
            list[Episode] — typically 1 Episode per MemCell, but the LLM
            may emit multiple if it detects sub-events.

        Raises:
            LLMNotConfiguredError, LLMError: same as boundary.
        """
        client = everalgo.llm.resolve(llm)
        rendered = (prompt or EPISODE_EXTRACT_PROMPT_EN).format(
            memcell_text=_render_memcell_text(memcell),
            timestamp=memcell.timestamp,
        )
        response = await client.chat(
            messages=[LLMChatMessage(role="user", content=rendered)],
            response_format={"type": "json_object"},
        )
        return _build_episodes_from_llm_response(response.content, memcell)

    extract = async_to_sync(aextract)
    """Sync bridge — only callable from non-event-loop contexts."""


# Module-level helper functions.


def _render_memcell_text(memcell: MemCell) -> str:
    """Render a MemCell as a prompt-friendly conversation transcript."""
    return "\n".join(
        f"[{m.role.value}] {m.content}" for m in memcell.messages
    )


def _build_episodes_from_llm_response(
    raw: str, memcell: MemCell
) -> list[Episode]:
    """Parse LLM JSON and build Episode list.

    parent_id and parent_type are auto-filled from the source memcell
    (LLM is instructed not to emit them; see prompts/en/episode.py).
    """
    parsed = json.loads(raw)
    episodes: list[Episode] = []
    for ep_dict in parsed.get("episodes", []):
        ep_dict.setdefault("parent_type", "memcell")
        ep_dict.setdefault("parent_id", memcell.id)
        episodes.append(Episode.model_validate(ep_dict))
    return episodes
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest packages/everalgo-user-memory/tests/user_memory/test_episode.py -v
```

Expected: 3 PASS.

- [ ] **Step 5: Run quality gates (from REPO ROOT)**

```bash
uv run ruff check packages/everalgo-user-memory/
uv run ruff format --check packages/everalgo-user-memory/
uv run mypy packages/everalgo-user-memory/
```

Expected: 全部 clean.

- [ ] **Step 6: Commit**

```bash
git add packages/everalgo-user-memory/src/everalgo/user_memory/episode.py \
        packages/everalgo-user-memory/tests/user_memory/test_episode.py
git commit -m "$(cat <<'EOF'
✨ feat(user_memory): add EpisodeExtractor minimal reference impl

EpisodeExtractor follows the framework defined in ADR 010 line 205-212
+ design.md line 829: stateless class + async `aextract` + one-line
`extract = async_to_sync(aextract)` sync bridge. LLM injection via
sub-project 2.5's everalgo.llm.resolve(). Prompt via per-call override
or default EPISODE_EXTRACT_PROMPT_EN.

Algorithm: render prompt with memcell text + timestamp, call LLM,
parse JSON {"episodes": [...]}, validate each via Episode.model_validate
(sub-project 1 type), auto-fill parent_id / parent_type from the source
MemCell (the prompt instructs the LLM not to emit them).

Module-level helpers (_render_memcell_text /
_build_episodes_from_llm_response) keep the class body focused on the
public contract.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: `everalgo.user_memory.__init__` re-export + `test_user_memory_public_api.py`

**Files:**
- Modify: `packages/everalgo-user-memory/src/everalgo/user_memory/__init__.py`
- Create: `packages/everalgo-user-memory/tests/user_memory/test_user_memory_public_api.py`

- [ ] **Step 1: Write failing tests**

Write to `packages/everalgo-user-memory/tests/user_memory/test_user_memory_public_api.py`:

```python
"""Tests for everalgo.user_memory package-level public API."""

import everalgo.boundary
import everalgo.user_memory


def test_chat_memcell_extractor_re_exported_from_boundary() -> None:
    """user_memory.ChatMemCellExtractor IS boundary.ChatMemCellExtractor (identity)."""
    assert (
        everalgo.user_memory.ChatMemCellExtractor
        is everalgo.boundary.ChatMemCellExtractor
    )


def test_episode_extractor_exported_at_top_level() -> None:
    """EpisodeExtractor is accessible via attribute on the package."""
    assert hasattr(everalgo.user_memory, "EpisodeExtractor")


def test_dunder_all_lists_exactly_two_symbols() -> None:
    """Sub-project 4 minimal exposes 2 symbols: re-export + EpisodeExtractor."""
    assert sorted(everalgo.user_memory.__all__) == sorted(
        ["ChatMemCellExtractor", "EpisodeExtractor"]
    )
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest packages/everalgo-user-memory/tests/user_memory/test_user_memory_public_api.py -v
```

Expected: FAIL — `__init__.py` is empty placeholder.

- [ ] **Step 3: Replace `__init__.py`**

Replace contents of `packages/everalgo-user-memory/src/everalgo/user_memory/__init__.py` with:

```python
"""User-side memory extractors — episode (sub-project 4 minimal).

foresight / atomic_fact / profile extractors are out of sub-project 4
scope; they will land in future SemVer minor bumps (see spec §10).

Public surface (sub-project 4 minimal):
- ChatMemCellExtractor — re-exported from boundary (design.md line 122)
- EpisodeExtractor — extract Episode from a single MemCell
"""

from everalgo.boundary.chat import ChatMemCellExtractor
from everalgo.user_memory.episode import EpisodeExtractor

__all__ = ["ChatMemCellExtractor", "EpisodeExtractor"]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest packages/everalgo-user-memory/tests/user_memory/ -v
```

Expected: 5 PASS (3 episode + 3 public_api... wait, 3+3 should be 6; the public_api file has 3 tests so total is 6. Re-count: 3 episode + 3 public_api = 6 PASS).

- [ ] **Step 5: Run quality gates (from REPO ROOT)**

```bash
uv run ruff check packages/everalgo-user-memory/
uv run ruff format --check packages/everalgo-user-memory/
uv run mypy packages/everalgo-user-memory/
```

Expected: 全部 clean.

- [ ] **Step 6: Verify the user-facing import paths**

```bash
uv run python -c "
from everalgo.user_memory import ChatMemCellExtractor, EpisodeExtractor
import everalgo.boundary
assert ChatMemCellExtractor is everalgo.boundary.ChatMemCellExtractor
print('OK: re-export identity holds; EpisodeExtractor =', EpisodeExtractor)
"
```

Expected: prints `OK: re-export identity holds; EpisodeExtractor = <class ...>`.

- [ ] **Step 7: Commit**

```bash
git add packages/everalgo-user-memory/src/everalgo/user_memory/__init__.py \
        packages/everalgo-user-memory/tests/user_memory/test_user_memory_public_api.py
git commit -m "$(cat <<'EOF'
✨ feat(user_memory): re-export ChatMemCellExtractor from boundary + EpisodeExtractor

Replace the placeholder __init__.py with the final sub-project 4
public surface for everalgo.user_memory:

- ChatMemCellExtractor — re-exported from everalgo.boundary.chat
  (design.md line 122 contract: "everalgo.user_memory.ChatMemCellExtractor"
  is the EverOS-facing path even though the implementation physically
  lives in boundary/)
- EpisodeExtractor — extract Episode from a single MemCell

The re-export uses identity equality: `everalgo.user_memory.
ChatMemCellExtractor is everalgo.boundary.ChatMemCellExtractor` is True.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: End-to-end integration test

**Files:**
- Create: `tests/integration/test_boundary_to_episode_e2e.py`

- [ ] **Step 1: Write the e2e test**

Write to `tests/integration/test_boundary_to_episode_e2e.py`:

```python
"""End-to-end pipeline test: messages → boundary → episode.

Verifies the full boundary→episode data flow with a FakeLLMClient handler
that returns distinct JSON per call (boundary call returns split decision,
episode call returns episode JSON). This is the sub-project 4 reference
implementation acceptance test.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

import everalgo.llm
from everalgo.boundary.chat import ChatMemCellExtractor
from everalgo.llm.types import ChatMessage as LLMChatMessage, ChatResponse
from everalgo.testing.assertions import assert_episode_shape
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import Message, MessageRole
from everalgo.user_memory.episode import EpisodeExtractor


@pytest.fixture(autouse=True)
def reset_everalgo_llm_state() -> Iterator[None]:
    """Reset everalgo.llm._default + _active per test (sub-project 2.5 fixture).

    Without this, test pollution between e2e and other test files could
    leak global state.
    """
    saved_default = everalgo.llm._default
    token = everalgo.llm._active.set(None)
    try:
        everalgo.llm._default = None
        yield
    finally:
        everalgo.llm._default = saved_default
        everalgo.llm._active.reset(token)


async def test_boundary_to_episode_pipeline_e2e() -> None:
    """boundary detects 1 MemCell, episode extracts 1 Episode from it.

    Uses FakeLLMClient handler mode to return distinct JSON per call:
    - Call 1 (boundary detect): {"split_at": null} (no split)
    - Call 2 (episode extract): episode JSON

    Verifies:
    1. Both extractors run without errors (full integration of types,
       LLM stack, prompts, sub-project 2.5 resolve, sub-project 3
       FakeLLMClient + assert_episode_shape).
    2. parent_id flows from MemCell.id to Episode.parent_id.
    3. assert_episode_shape passes (sub-project 3 helper).
    """
    call_count = 0

    def handler(
        messages: list[LLMChatMessage], **kwargs: Any
    ) -> ChatResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            content = '{"split_at": null}'
        else:
            content = (
                '{"episodes": [{"id": "ep_test_001", '
                '"owner_id": "u_test", '
                '"episode": "User scheduled a meeting with Alice at 3pm.", '
                '"timestamp": 1700000000000}]}'
            )
        return ChatResponse(content=content, model="fake")

    fake = FakeLLMClient(handler=handler)
    msgs = [
        Message(
            role=MessageRole.USER,
            content="Schedule a meeting with Alice at 3pm",
            timestamp=1700000000000,
        ),
        Message(
            role=MessageRole.ASSISTANT,
            content="Done. Sent invite for 3pm.",
            timestamp=1700000001000,
        ),
    ]

    memcells = await ChatMemCellExtractor().adetect(msgs, llm=fake)
    assert len(memcells) == 1
    mc = memcells[0]

    episodes = await EpisodeExtractor().aextract(mc, llm=fake)
    assert len(episodes) == 1

    ep = assert_episode_shape(episodes[0])
    assert ep.parent_id == mc.id
    assert ep.parent_type == "memcell"
    assert "Alice" in ep.episode
    assert call_count == 2
```

- [ ] **Step 2: Run the e2e test**

```bash
uv run pytest tests/integration/test_boundary_to_episode_e2e.py -v
```

Expected: 1 PASS.

- [ ] **Step 3: Run quality gates (from REPO ROOT)**

```bash
uv run ruff check tests/
uv run ruff format --check tests/
uv run mypy tests/
```

Expected: 全部 clean.

- [ ] **Step 4: Verify the test discovers from workspace root**

```bash
uv run pytest tests/ -v
```

Expected: collects + passes the 1 e2e test.

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_boundary_to_episode_e2e.py
git commit -m "$(cat <<'EOF'
✅ test(integration): add boundary→episode e2e test

End-to-end pipeline verification: messages → ChatMemCellExtractor.adetect
→ MemCell → EpisodeExtractor.aextract → Episode. Uses FakeLLMClient
handler mode to return distinct JSON per call (boundary returns
{"split_at": null}, episode returns episode JSON).

This is sub-project 4 acceptance: confirms the full integration of
sub-project 1 types + sub-project 2 LLM stack + sub-project 2.5
3-layer injection + sub-project 3 FakeLLMClient + assert_episode_shape
+ sub-project 4 reference extractors all wire together correctly,
with parent_id flowing from MemCell to Episode (data lineage).

Located at workspace-root tests/integration/ per AGENTS.md §9
(cross-package integration smoke tests).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: `.gitlab-ci.yml` strict mode upgrade

**Files:**
- Modify: `.gitlab-ci.yml`

- [ ] **Step 1: Read existing `.gitlab-ci.yml`**

```bash
cat .gitlab-ci.yml
```

The current file is the 57-line placeholder with 3 stages (lint / test / build), `pytest.allow_failure: true`, no `mypy` job.

- [ ] **Step 2: Replace with strict mode**

Replace contents of `.gitlab-ci.yml` with:

```yaml
# GitLab CI for the EverAlgo monorepo (sub-project 4).
#
# Strict mode: 4 quality gates (ruff check / ruff format --check / mypy /
# pytest) all enforced. The 3 lint jobs run in parallel within the lint
# stage; pytest follows in the test stage. Per-package matrix (only:
# changes:) is deferred to a future sub-project once enough distributions
# have releasable artifacts.

stages:
  - lint
  - test

ruff-check:
  stage: lint
  image: python:3.12-slim
  before_script:
    - pip install --quiet uv
    - uv sync --all-packages --group dev
  script:
    - uv run ruff check .
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
    - if: $CI_COMMIT_BRANCH == "main"

ruff-format:
  stage: lint
  image: python:3.12-slim
  before_script:
    - pip install --quiet uv
    - uv sync --all-packages --group dev
  script:
    - uv run ruff format --check .
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
    - if: $CI_COMMIT_BRANCH == "main"

mypy:
  stage: lint
  image: python:3.12-slim
  before_script:
    - pip install --quiet uv
    - uv sync --all-packages --group dev
  script:
    - uv run mypy .
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
    - if: $CI_COMMIT_BRANCH == "main"

pytest:
  stage: test
  image: python:3.12-slim
  before_script:
    - pip install --quiet uv
    - uv sync --all-packages --group dev
  script:
    - uv run pytest -v
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
    - if: $CI_COMMIT_BRANCH == "main"
```

Note removed:
- `build` stage and the placeholder build job (deferred until first PyPI release)
- `pytest.allow_failure: true` (now strict; failures block the pipeline)
- The legacy placeholder comment about path-based pipelines (replaced by the strict-mode comment header)

- [ ] **Step 3: Validate YAML syntax locally**

```bash
uv run python -c "import yaml; yaml.safe_load(open('.gitlab-ci.yml'))" && echo "YAML OK"
```

Expected: prints `YAML OK`. (PyYAML is in dev deps; if not installed, `uv sync --group dev` first.)

If PyYAML is missing, install ad-hoc:

```bash
uv run --with pyyaml python -c "import yaml; yaml.safe_load(open('.gitlab-ci.yml'))" && echo "YAML OK"
```

- [ ] **Step 4: Verify the locally-runnable equivalents pass**

The CI runs 4 commands. Run them locally to confirm the pipeline will pass on push:

```bash
uv run ruff check . \
  && uv run ruff format --check . \
  && uv run mypy . \
  && uv run pytest -v \
  && echo "ALL 4 CI EQUIVALENTS PASS LOCALLY"
```

Expected: prints `ALL 4 CI EQUIVALENTS PASS LOCALLY`. If anything fails, fix before commit.

- [ ] **Step 5: Commit**

```bash
git add .gitlab-ci.yml
git commit -m "$(cat <<'EOF'
🚀 ci: upgrade .gitlab-ci.yml to strict mode (4 quality gates)

Replace the placeholder pipeline (3 stages, pytest allow_failure=true,
no mypy) with strict mode:

- 3 parallel lint jobs (ruff check / ruff format --check / mypy)
- 1 test job (pytest, allow_failure removed)
- Drop the build stage (deferred until first PyPI release)

This is the sub-project 4 CI deliverable. Per-package matrix (only:
changes:) is deferred — single matrix runs all 8 packages each pipeline,
which is fine while the codebase is small. Future work: GitLab CI
parallel:matrix once each distribution has independent release cadence
(see spec §6.4).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Sub-project 4 final acceptance + workspace verification

**Files:** No new files. This task runs the full acceptance gate across the whole workspace.

- [ ] **Step 1: Run all 4 quality gates from REPO ROOT**

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy .
uv run pytest -v
```

Expected:
- `ruff check .`: clean
- `ruff format --check .`: clean
- `mypy .`: 0 errors
- `pytest -v`: **all tests pass** — the workspace-cumulative test count should be approximately:
  - Sub-project 1 (Foundation): ~28 tests in `packages/everalgo-core/tests/types/`
  - Sub-project 2 (LLM Stack): ~41 tests in `packages/everalgo-core/tests/llm/`
  - Sub-project 3 (Testing Toolkit): ~32 tests in `packages/everalgo-core/tests/testing/`
  - Sub-project 2.5 (LLM 3-layer injection): ~15 tests in `packages/everalgo-core/tests/llm/test_injection.py` + 1 test in `test_errors.py`
  - Sub-project 4 boundary: 8 tests (3 chat + 3 tokenize + 2 boundary public_api)
  - Sub-project 4 user_memory: 6 tests (3 episode + 3 user_memory public_api)
  - Sub-project 4 e2e: 1 test
  - **Total: ~131 tests** (exact count depends on prior task counts; expect ≥ 130 PASS)

If any gate fails, **stop and fix** — do not commit.

- [ ] **Step 2: Verify the user-facing import surface**

Run an end-to-end import smoke that mirrors how an EverOS or external user would consume EverAlgo:

```bash
uv run python <<'EOF'
# Imports an EverOS user would write (per design.md line 311-313 contract)
from everalgo.boundary import ChatMemCellExtractor
from everalgo.user_memory import EpisodeExtractor
from everalgo.user_memory import ChatMemCellExtractor as UMChat
import everalgo.boundary as boundary_mod

# Identity check — re-export must point to the same class
assert UMChat is ChatMemCellExtractor
assert UMChat is boundary_mod.ChatMemCellExtractor

# Smoke instantiate (stateless, no args)
chat = ChatMemCellExtractor()
ep = EpisodeExtractor()

# Verify the public API methods exist
assert callable(chat.adetect)
assert callable(chat.detect)    # sync bridge
assert callable(ep.aextract)
assert callable(ep.extract)

# Verify __all__ contracts
import everalgo.boundary
import everalgo.user_memory
assert everalgo.boundary.__all__ == ["ChatMemCellExtractor"]
assert sorted(everalgo.user_memory.__all__) == sorted(
    ["ChatMemCellExtractor", "EpisodeExtractor"]
)

print("Sub-project 4 contract OK")
EOF
```

Expected: prints `Sub-project 4 contract OK` without ImportError or AssertionError.

- [ ] **Step 3: Verify cumulative branch state**

```bash
git log --oneline | head -25
git diff origin/main --stat | tail -5
```

Expected: see ~10 sub-project 4 commits since the last sub-project 2.5 commit `7d5261f`. The diff stat reflects the cumulative work across sub-projects 1+2+2.5+3+4.

- [ ] **Step 4: No commit needed for this task**

This task is acceptance-only (no new artifacts). All sub-project 4 commits are already in place from Tasks 0-9.

If everything in Steps 1-3 passes, sub-project 4 is **complete**.

---

## Self-Review Checklist (作者自审，已通过)

### 1. Spec coverage

| Spec 章节 | 实现 task |
|---|---|
| §2 File Map (boundary + user_memory + tests/integration) | Task 0-9 全覆盖 |
| §3.1 ChatMemCellExtractor 完整实现 | Task 3 |
| §3.2 _tokenize.count_tokens | Task 1 |
| §3.3 EpisodeExtractor 完整实现 | Task 6 |
| §3.4 包级 __init__.py re-export | Task 4 (boundary) + Task 7 (user_memory) |
| §4 Prompt 模块组织 (en + zh × 2 算子) | Task 2 (boundary) + Task 5 (user_memory) |
| §5 测试矩阵（boundary 8 + user_memory 6 + e2e 1） | Task 1 (3 tokenize) + Task 3 (3 chat) + Task 4 (2 boundary public_api) + Task 6 (3 episode) + Task 7 (3 user_memory public_api — re-count: 3 not 2 per the test code in Task 7) + Task 8 (1 e2e) |
| §6 CI Pipeline 升级 | Task 9 |
| §7 关键设计决策 15 项 | 设计已落实到具体 task 实现 |

无 spec 章节未覆盖。

### 2. Placeholder scan

`grep -nE "TODO|TBD|FIXME|XXX|implement later|fill in" docs/superpowers/plans/2026-05-08-everalgo-reference-impl.md`
预期：0 hits。

### 3. Type consistency

- `MemCell` / `Message` / `MessageRole` / `Episode` 类型在 Task 3 + Task 6 + Task 8 一致（均来自 `everalgo.types`）
- `LLMClient` / `LLMChatMessage` / `ChatResponse` / `FakeLLMClient` / `assert_episode_shape` 类型在 Task 3 + Task 6 + Task 8 一致（均来自 sub-project 1+2+3）
- `ChatMemCellExtractor.adetect` 签名（messages, *, llm=None, prompt=None）在 Task 3 + Task 4 + Task 8 三处一致
- `EpisodeExtractor.aextract` 签名（memcell, *, llm=None, prompt=None）在 Task 6 + Task 7 + Task 8 三处一致
- prompt 常量名 `CHAT_BOUNDARY_DETECT_PROMPT_{EN,ZH}` / `EPISODE_EXTRACT_PROMPT_{EN,ZH}` 在 Task 2 + Task 3 + Task 5 + Task 6 一致

### 4. Lessons learned 应用

- ✅ 测试函数 `-> None` 注解 — 全部 task 测试代码已显式标注
- ✅ 测试目录无 `__init__.py` — Task 0 Step 4 显式 verify
- ✅ commit message 全英文 + Co-Authored-By — Task 0-9 全部使用英文模板
- ✅ ruff + mypy + pytest 三 gate 每 task 必过 — 每 task Step 5 / 6
- ✅ mypy 从 REPO ROOT 跑 — Task 1-9 都明示
- ✅ 测试文件名工作区唯一 — `test_chat.py` / `test_tokenize.py` / `test_boundary_public_api.py` / `test_episode.py` / `test_user_memory_public_api.py` / `test_boundary_to_episode_e2e.py` 都唯一（避开 sub-project 2 的 `tests/llm/test_public_api.py`）
- ✅ 算子是 stateless class — Task 3 / Task 6 实现严格遵循 ADR 010 line 205-212 模式
- ✅ Prompt 命名后缀 `_EN` / `_ZH` 消歧义
- ✅ JSON braces 在 prompt template `.format()` 中 escape `{{` / `}}` — Task 2 + Task 5 都明示并 Step 5 验证

### 5. 任务大小

- Task 0：约 15 分钟（pyproject 改 + 目录骨架）
- Task 1：约 20 分钟（count_tokens + 3 测试，最简单）
- Task 2：约 25 分钟（5 prompt 文件，含 JSON brace escape 校验）
- Task 3：约 50 分钟（ChatMemCellExtractor + 4 helpers + 3 测试）
- Task 4：约 20 分钟（__init__.py + 3 测试）
- Task 5：约 25 分钟（与 Task 2 类似但 user_memory 分支）
- Task 6：约 50 分钟（与 Task 3 类似）
- Task 7：约 25 分钟（__init__.py + 3 测试，含 identity check）
- Task 8：约 35 分钟（e2e test，含 autouse fixture + 双 LLM call handler）
- Task 9：约 25 分钟（CI yaml 升级 + 本地 4 gate 验证）
- Task 10：约 15 分钟（acceptance run，无 commit）

总计：约 5-6 小时（含 SDD review 循环）。

### 6. 修正的 self-review 错误

- Task 7 的测试数实际是 3（test_chat_memcell_extractor_re_exported_from_boundary + test_episode_extractor_exported_at_top_level + test_dunder_all_lists_exactly_two_symbols），不是 2。spec §5.2 写的是「~2 测试」，plan Task 7 实际写了 3 测试，更全面。Task 10 Step 1 的预期总数 ≥ 130 已经合理覆盖。
