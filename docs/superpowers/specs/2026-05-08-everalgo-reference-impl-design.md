# EverAlgo Reference Implementation + CI 子项目设计文档

> 本文档是 **子项目 4 / 4** 的设计 spec（5 个子项目最后一个）。
> 上游：子项目 1 (Foundation) + 子项目 2 (LLM Stack) + 子项目 2.5 (3-Layer Injection) + 子项目 3 (Testing Toolkit) 全部已完成。
> 下游：本子项目落地后，EverAlgo 进入 release-ready 状态。
>
> 落盘日期：2026-05-08
> 设计依据：design.md §1.2 / §2.3 / §2.5 + ADR 010 + ADR 011 + memsys_opensource release/20260403 reference 算法。

**Goal**：在 design.md / ADR 既定框架内，落地 `everalgo-boundary` + `everalgo-user-memory` 的最小集 reference 算法（boundary 检测 + episode 提取），覆盖端到端 boundary→episode 数据流，并升级 `.gitlab-ci.yml` 到 strict 模式。

**Architecture**：严格按 ADR 010 (sync/async dual-interface via `asgiref.async_to_sync` 一行桥接) + ADR 011 (Protocol structural subtyping) + design.md §1.2 物理布局。**框架不变**——本子项目仅在框架内填实现 + 选最小 scope。

**Scope**：2 个 extractor class（`ChatMemCellExtractor` / `EpisodeExtractor`）+ 1 个 module-level 共享 helper（`count_tokens`）+ 4 个 prompt 文件（en/zh × 2 算子）+ ~14 个新测试 + `.gitlab-ci.yml` strict 模式升级。约 600 行新增代码。

---

## 1. 背景与目标

### 1.1 Reference impl 的定位

子项目 1+2+2.5+3 已经提供了**算法库地基**：types / LLM stack / 3-layer injection / Testing toolkit。但**至今没有任何真实算法**落地：
- `everalgo-boundary` 是空骨架（仅 `pyproject.toml` + `README.md` + 空 `__init__.py`）
- `everalgo-user-memory` 是空骨架同样
- 5 个产品性子包（agent_memory / knowledge / rank / parser / clustering）也都是空骨架

子项目 4 的目标是**最小可工作的 reference impl** —— 让 EverAlgo 算法库**接上就能用**。具体：
- 算法同学拿到 EverAlgo 后，能跑 `await EpisodeExtractor().aextract(memcell, llm=client)` 真实从 LLM 提取 Episode
- 端到端数据流可演示：`messages → ChatMemCellExtractor.adetect → MemCell → EpisodeExtractor.aextract → Episode`
- prompt 是**真实可调优的精简版**（vs memsys_opensource 575+146 行的产品级 prompt）—— 算法同学可以通过 per-call `prompt=` 参数或 monkey-patch 模块常量替换为 production 版本

### 1.2 BOSS 框架优先决策（2026-05-08）

BOSS 在子项目 4 brainstorm 期间明确：「核心要先定好框架，然后往里面放东西，框架不对改起来成本很高，框架要职责清晰，易维护，易扩展」。

复盘后发现：**框架已经被 design.md / ADR 010 / ADR 011 钉死**，子项目 4 不需要重新设计框架，只需在框架内：
1. 选最小 scope（哪几个 extractor 落地，哪些推迟到 SemVer minor bump）
2. 把 prompt / tokenizer 等具体实现细节定下来
3. 升级 CI 到 strict 模式

详见 §7 关键设计决策。

### 1.3 Reference impl 的功能层级（BOSS 拍板 = B 选项）

3 档候选：

| 档位 | 形态 | 算法量 |
|---|---|---|
| A. Demo placeholder | 仿 opensource mdfirst `demo_episode.py`（39 行）：不调 LLM、直接拼字符串 | 极小 |
| **B. Minimal reference impl（拍板）** | 每 extractor ~50-80 行：真 LLM 调用 + 简化 prompt（基于 opensource 精简）+ JSON parse | 中等 |
| C. Full opensource port | 完整复现 opensource release/20260403 `EpisodeMemoryExtractor` (404 行) + `conv_memcell_extractor.py` (575 行) + 146 行 prompt | 大 |

**B 档已选**：算法骨架可工作，prompt 是真实可调优的精简版；EverAlgo 算法库定位下「算法同学接上就能调到 production」即够用，C 档（完整复现 opensource SOTA）不属算法库职责，应由 EverOS 编排层落地。

---

## 2. 物理布局（File Map）

```
packages/everalgo-boundary/
├── pyproject.toml          # MODIFY: 加 asgiref>=3.0 依赖
├── src/everalgo/boundary/
│   ├── __init__.py         # MODIFY: re-export ChatMemCellExtractor (公开 surface)
│   ├── chat.py             # NEW: ChatMemCellExtractor class（adetect/detect）
│   ├── _tokenize.py        # NEW: count_tokens(text) -> int (sync only)
│   └── prompts/
│       ├── __init__.py     # NEW
│       ├── en/
│       │   ├── __init__.py # NEW
│       │   └── chat.py     # NEW: CHAT_BOUNDARY_DETECT_PROMPT_EN 常量
│       └── zh/
│           ├── __init__.py # NEW
│           └── chat.py     # NEW: CHAT_BOUNDARY_DETECT_PROMPT_ZH 常量
└── tests/boundary/
    ├── test_chat.py            # NEW: ChatMemCellExtractor unit tests (~3)
    ├── test_tokenize.py        # NEW: count_tokens unit tests (~3)
    └── test_public_api.py      # NEW: boundary 公开 API smoke (~2)

packages/everalgo-user-memory/
├── pyproject.toml          # MODIFY: 暂去 everalgo-clustering 依赖（最小集不需要）
├── src/everalgo/user_memory/
│   ├── __init__.py         # MODIFY: re-export ChatMemCellExtractor (from boundary) + EpisodeExtractor
│   ├── episode.py          # NEW: EpisodeExtractor class（aextract/extract）
│   └── prompts/
│       ├── __init__.py     # NEW
│       ├── en/
│       │   ├── __init__.py # NEW
│       │   └── episode.py  # NEW: EPISODE_EXTRACT_PROMPT_EN 常量
│       └── zh/
│           ├── __init__.py # NEW
│           └── episode.py  # NEW: EPISODE_EXTRACT_PROMPT_ZH 常量
└── tests/user_memory/
    ├── test_episode.py         # NEW: EpisodeExtractor unit tests (~3)
    └── test_public_api.py      # NEW: user_memory 公开 API smoke (~2)

tests/                      # NEW: workspace 根级集成测试目录
└── integration/
    └── test_boundary_to_episode_e2e.py    # NEW: e2e 测试（~1 test）

.gitlab-ci.yml              # MODIFY: 4 jobs (ruff check / ruff format / mypy / pytest)
                            #         + allow_failure=false（移除 placeholder allow_failure）
                            #         + 单一矩阵（不分 package）
```

**注**：
- `tests/integration/` 是 workspace 根级新目录（与 packages/<X>/tests/ 隔离），符合 AGENTS.md §9「Cross-package integration smoke tests belong in the workspace-root tests/ directory」。**不要**创建 `tests/__init__.py` 或 `tests/integration/__init__.py`（沿用 importlib 模式）。
- 测试文件名工作区唯一（per memory `feedback_test_module_name_unique.md`）：`test_chat.py` / `test_episode.py` / `test_tokenize.py` / `test_boundary_to_episode_e2e.py` / `test_public_api.py`（**注意**：boundary 和 user_memory 都有 `test_public_api.py`，与子项目 2 的 `tests/llm/test_public_api.py` + 子项目 3 的 `tests/testing/test_testing_public_api.py` 都需要错开命名 —— 见 §5.3 物理布局）。

---

## 3. 关键 API 签名（最小集）

### 3.1 `everalgo.boundary.chat.ChatMemCellExtractor`

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
    ``prompt=`` arguments. Mirrors the design.md line 829 EpisodeExtractor
    pattern — class is a namespace, not stateful.

    Algorithm (minimal reference impl):
        1. Render LLM prompt with the message stream + token budget hint
        2. Call LLM, parse JSON response with `split_at` index
        3. Build one MemCell from messages[:split_at] (or whole stream
           if split_at is null/None — i.e. coherent topic)
        4. Return list[MemCell] (length 1 in the no-split case)

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
                (``use(...)``) and default (``configure(llm=...)``); raises
                ``LLMNotConfiguredError`` if all None (sub-project 2.5).
            prompt: Per-call prompt override. Defaults to
                ``CHAT_BOUNDARY_DETECT_PROMPT_EN``. To use Chinese, pass
                ``CHAT_BOUNDARY_DETECT_PROMPT_ZH`` from
                ``everalgo.boundary.prompts.zh.chat``.

        Returns:
            list[MemCell] — at least one cell. The minimal ref impl produces
            either 1 cell (no split) or 2 cells (one split point); future
            multi-split versions return more.

        Raises:
            LLMNotConfiguredError: If no LLM is resolvable (sub-project 2.5).
            LLMError: Provider-side errors (auth / rate / connection /
                timeout) — these propagate from the SDK call site.
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
    """Sync bridge — only callable from non-event-loop contexts (CLI scripts,
    plain unit tests). Inside Jupyter / FastAPI / any ``async def`` context,
    use ``await adetect(...)`` directly.
    """


# Module-level helper functions — stateless utilities (per AGENTS.md §5
# "Module-level functions + global config + monkeypatch in tests" — algorithm
# authors should be one keystroke away from running these).


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
    """Parse LLM JSON `{"split_at": int | null}` and build MemCell list."""
    parsed = json.loads(raw)
    split_at = parsed.get("split_at")
    if split_at is None:
        return [_make_memcell(messages, suffix="0")]
    return [
        _make_memcell(messages[:split_at], suffix="0"),
        _make_memcell(messages[split_at:], suffix="1"),
    ]


def _make_memcell(slice_msgs: list[Message], *, suffix: str) -> MemCell:
    """Build a MemCell with deterministic id (caller-supplied if needed)."""
    timestamp = slice_msgs[-1].timestamp if slice_msgs else 0
    return MemCell(
        id=f"mc_{timestamp}_{suffix}",
        messages=slice_msgs,
        timestamp=timestamp,
    )
```

> ✅ **设计自检：算子是 stateless class + module-level 私有 helpers**
> - **为什么这样设计**：ADR 010 line 205-212 是 EverAlgo 唯一推荐模式；ADR 011 line 117 选 Protocol structural（不强制继承）；class 提供 namespace + sync 桥接合并位置，private helpers (`_concat_messages` / `_format_messages_for_prompt` / `_build_memcells_from_llm_response` / `_make_memcell`) 单测友好（algorithm authors 可 monkeypatch 替换）
> - **规范依据**：sklearn `KMeans` / pytorch `nn.Linear` 同模式（class as namespace + 内部用 module-level helpers）；AGENTS.md §5「Module-level functions + global config + monkeypatch in tests」
> - **备选方案**：① module-level function `detect_chat(messages, ...)`（无 namespace 组织，未来多 extractor 命名冲突）；② class with `__init__` 持配置（与 stateless 哲学冲突）—— 都被 ADR 011 否决

### 3.2 `everalgo.boundary._tokenize.count_tokens`

```python
"""Token counting helper for boundary extractors.

Minimal reference impl using a 4-character heuristic (roughly matching
English GPT tokens). For production use, replace with tiktoken or a real
tokenizer. NOT exposed in __all__ — module-private utility.
"""

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

> ✅ **设计自检：count_tokens 用简化估算而非 tiktoken**
> - **为什么这样设计**：保持子项目 1-3 「零新增依赖」基线一致；最小集 reference impl 不引入 tiktoken（~5MB binary 模型 + 加载时间）；算法同学若要 production 精确度可 monkey-patch 替换实现
> - **规范依据**：design.md line 550 仅规定 `boundary._tokenize.count_tokens` 是共享底层，未指定具体实现
> - **备选方案**：① tiktoken 依赖（行业标准，但 +1 dependency + binary 加载开销）；② 完全去掉 token 计数（Q4=C 选项，但 boundary 算法本质需要 token 上限决策）—— 都被 BOSS Q4=A 否决

### 3.3 `everalgo.user_memory.episode.EpisodeExtractor`

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


# Module-level private helpers


def _render_memcell_text(memcell: MemCell) -> str:
    """Render a MemCell as a prompt-friendly conversation transcript."""
    return "\n".join(
        f"[{m.role.value}] {m.content}" for m in memcell.messages
    )


def _build_episodes_from_llm_response(
    raw: str, memcell: MemCell
) -> list[Episode]:
    """Parse LLM JSON `{"episodes": [{"id": ..., "owner_id": ..., ...}, ...]}`.

    Each episode dict is parsed via Episode.model_validate (sub-project 1
    type system). parent_id and parent_type fields are auto-filled from
    the source memcell.
    """
    parsed = json.loads(raw)
    episodes: list[Episode] = []
    for ep_dict in parsed.get("episodes", []):
        ep_dict.setdefault("parent_type", "memcell")
        ep_dict.setdefault("parent_id", memcell.id)
        episodes.append(Episode.model_validate(ep_dict))
    return episodes
```

### 3.4 包级 `__init__.py` re-export

```python
# everalgo/boundary/__init__.py
"""Boundary extractors — chat (sub-project 4 minimal).

workspace + agent extractors are out of sub-project 4 scope and will land
in a future SemVer minor bump (see spec §10).

Public surface (sub-project 4 minimal):
- ChatMemCellExtractor — slice chat messages into MemCells
"""
from everalgo.boundary.chat import ChatMemCellExtractor

__all__ = ["ChatMemCellExtractor"]


# everalgo/user_memory/__init__.py
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

> ✅ **设计自检：user_memory re-export ChatMemCellExtractor from boundary**
> - **为什么这样设计**：design.md line 122 + line 311-313 已规定双访问路径：`from everalgo.user_memory import ChatMemCellExtractor`（EverOS 契约）+ `from everalgo.boundary.chat import ChatMemCellExtractor`（算法同学按算法族 import）
> - **规范依据**：design.md §1.2 line 75-78 "EverOS 文档契约 ... 通过各产品性子包的 __init__.py re-export 实现，与物理路径解耦"
> - **备选方案**：仅暴露 `everalgo.boundary.chat` 单一路径 —— 否决，破坏 design.md 既定契约

---

## 4. Prompt 模块组织

### 4.1 命名约定

每个 prompt 文件 1 个 module-level 常量，命名为 `<OPERATION>_<CONTEXT>_PROMPT_<LANG>`：

| 文件 | 常量 |
|---|---|
| `boundary/prompts/en/chat.py` | `CHAT_BOUNDARY_DETECT_PROMPT_EN` |
| `boundary/prompts/zh/chat.py` | `CHAT_BOUNDARY_DETECT_PROMPT_ZH` |
| `user_memory/prompts/en/episode.py` | `EPISODE_EXTRACT_PROMPT_EN` |
| `user_memory/prompts/zh/episode.py` | `EPISODE_EXTRACT_PROMPT_ZH` |

> 备注：`_EN` / `_ZH` 后缀用于消歧义，避免算法同学 monkey-patch 时混淆。算子默认 import `_EN` 版本（design.md 默认英文）。

### 4.2 切换语言路径

**主路径（per-call）**：
```python
from everalgo.boundary.prompts.zh.chat import CHAT_BOUNDARY_DETECT_PROMPT_ZH

await ChatMemCellExtractor().adetect(
    messages, llm=client, prompt=CHAT_BOUNDARY_DETECT_PROMPT_ZH,
)
```

**全局切换（caller monkey-patch）** —— design.md §1.4 line 397-410 推荐：
```python
import everalgo.boundary.chat as chat_mod
import everalgo.boundary.prompts.zh.chat as zh
chat_mod.CHAT_BOUNDARY_DETECT_PROMPT_EN = zh.CHAT_BOUNDARY_DETECT_PROMPT_ZH
# 后续所有 ChatMemCellExtractor.adetect() 调用都用中文 prompt
```

### 4.3 Prompt 内容（最小集）

#### `CHAT_BOUNDARY_DETECT_PROMPT_EN`（约 30 行）

```text
You are a conversation boundary detector. Given a chat message stream,
identify whether the topic shifts mid-stream and at which message index
the shift occurs.

Messages:
{messages}

Token count of full stream: {token_count}

Instructions:
1. Read all messages and identify the dominant topic.
2. If a clear topic shift occurs, return the index of the FIRST message
   in the new topic. The index is 0-based and matches the message list.
3. If the entire stream stays on one coherent topic, return null.
4. If the stream is empty or has only one message, return null.

Output format (JSON only, no prose):
{
  "split_at": <int | null>
}
```

#### `EPISODE_EXTRACT_PROMPT_EN`（约 40 行，基于 opensource episode_mem_prompts.py 精简）

```text
You are an episodic memory generation expert. Given a conversation slice
(MemCell), extract structured Episode memories.

Conversation:
{memcell_text}

Conversation timestamp: {timestamp}

Instructions:
1. Identify each distinct episodic event (a complete "what happened" trace
   with participants, place, time, action, outcome).
2. Convert dialogue format into third-person narrative.
3. Preserve names, dates, locations, decisions, emotions.
4. Use the conversation timestamp as the episode time anchor.
5. Generate a unique id for each episode (e.g. "ep_<random>").
6. Use a stable owner_id from the conversation (default "u_default" if
   unclear).

Output format (JSON only, no prose):
{
  "episodes": [
    {
      "id": "<string>",
      "owner_id": "<string>",
      "episode": "<narrative text>",
      "timestamp": <int>
    }
  ]
}

Note: parent_type and parent_id will be auto-filled by the caller; do not
emit them.
```

中文版同义平移（约 30-40 行），保持 JSON 输出格式不变。

> ✅ **设计自检：Prompt 文件名后缀 `_EN` / `_ZH`**
> - **为什么这样设计**：避免算法同学 monkey-patch 时把 EN 替换成 ZH 后调试混乱（明确文件名告知语言）；与子项目 2 的 ChatMessage / ChatResponse 命名保持「英文为主、中文是 i18n 变体」一致
> - **规范依据**：memsys_opensource `release/20260403:src/memory_layer/prompts/en/episode_mem_prompts.py` + `zh/episode_mem_prompts.py` 同模式
> - **备选方案**：① 同名常量（`CHAT_BOUNDARY_DETECT_PROMPT` 在 en/zh 两个目录下）—— 否决，monkey-patch 时极易混淆；② 仅做英文（Q1 选项 B） —— BOSS Q1=A 否决

---

## 5. 测试矩阵

### 5.1 `everalgo-boundary/tests/boundary/`（共约 8 测试）

**`test_chat.py`**（3 测试）：
- `test_adetect_returns_single_memcell_when_llm_returns_no_split` —— FakeLLM 返 `{"split_at": null}` → list 长度 1
- `test_adetect_returns_two_memcells_when_llm_returns_split_index` —— FakeLLM 返 `{"split_at": 2}` → list 长度 2，前后内容正确
- `test_adetect_per_call_prompt_overrides_default` —— per-call `prompt="custom"` 时 FakeLLM handler mode 收到 custom 字符串

**`test_tokenize.py`**（3 测试）：
- `test_count_tokens_empty_string_is_zero`
- `test_count_tokens_short_text_proportional` —— `count_tokens("a" * 40)` 返回 10
- `test_count_tokens_returns_non_negative` —— 所有字符串都返非负值

**`test_public_api.py`**（2 测试）：
- `test_chat_memcell_extractor_exported` —— `everalgo.boundary.ChatMemCellExtractor` 可访问
- `test_dunder_all_lists_one_symbol` —— `__all__` 长度 1

### 5.2 `everalgo-user-memory/tests/user_memory/`（共约 5 测试）

**`test_episode.py`**（3 测试）：
- `test_aextract_returns_episode_list_from_llm_json` —— FakeLLM 返合法 JSON → list[Episode] 字段正确
- `test_aextract_auto_fills_parent_id_from_memcell` —— LLM 输出未填 parent_id 时算子自动补
- `test_aextract_per_call_llm_overrides_default` —— per-call `llm=fake` 优先于 default

**`test_public_api.py`**（2 测试）：
- `test_chat_memcell_extractor_re_exported_from_boundary` —— `everalgo.user_memory.ChatMemCellExtractor is everalgo.boundary.ChatMemCellExtractor`（identity check 验证 re-export）
- `test_episode_extractor_exported`

### 5.3 端到端集成测试 `tests/integration/test_boundary_to_episode_e2e.py`（1 测试）

```python
"""End-to-end pipeline test: messages → boundary → episode."""

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
def reset_everalgo_llm_state():
    """Reset everalgo.llm._default + _active per test (sub-project 2.5)."""
    saved = everalgo.llm._default
    token = everalgo.llm._active.set(None)
    try:
        everalgo.llm._default = None
        yield
    finally:
        everalgo.llm._default = saved
        everalgo.llm._active.reset(token)


async def test_boundary_to_episode_pipeline_e2e() -> None:
    """boundary detects 1 MemCell, episode extracts 1 Episode from it.

    Uses FakeLLMClient handler mode to return distinct JSON per call:
    - Call 1 (boundary detect): `{"split_at": null}` (no split, single MemCell)
    - Call 2 (episode extract): episode JSON

    Verifies:
    1. Both extractors run without errors
    2. parent_id flows from MemCell.id to Episode.parent_id
    3. assert_episode_shape passes (sub-project 3 helper)
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
```

### 5.4 物理布局 + 命名唯一性

| 测试文件 | 路径 |
|---|---|
| `test_chat.py` | `packages/everalgo-boundary/tests/boundary/test_chat.py` |
| `test_tokenize.py` | `packages/everalgo-boundary/tests/boundary/test_tokenize.py` |
| `test_public_api.py` | `packages/everalgo-boundary/tests/boundary/test_public_api.py` ⚠ 同名冲突 |
| `test_episode.py` | `packages/everalgo-user-memory/tests/user_memory/test_episode.py` |
| `test_public_api.py` | `packages/everalgo-user-memory/tests/user_memory/test_public_api.py` ⚠ 同名冲突 |
| `test_boundary_to_episode_e2e.py` | `tests/integration/test_boundary_to_episode_e2e.py` |

⚠ **`test_public_api.py` 同名冲突**：boundary + user_memory + 子项目 2 (`tests/llm/test_public_api.py`) + 子项目 3 (`tests/testing/test_testing_public_api.py` 已加 testing 前缀避冲突) 都有 `test_public_api.py`。子项目 3 经验已经把 testing 的改成 `test_testing_public_api.py`，本子项目同样改：

| 改名后 | 路径 |
|---|---|
| `test_boundary_public_api.py` | `packages/everalgo-boundary/tests/boundary/test_boundary_public_api.py` |
| `test_user_memory_public_api.py` | `packages/everalgo-user-memory/tests/user_memory/test_user_memory_public_api.py` |

工作区累计 5 个 `test_*_public_api.py` 文件名都唯一：
- `test_public_api.py`（子项目 2 LLM）
- `test_testing_public_api.py`（子项目 3）
- `test_boundary_public_api.py`（子项目 4 boundary）
- `test_user_memory_public_api.py`（子项目 4 user_memory）

> ✅ **设计自检：test_*_public_api.py 加 sub-package 前缀**
> - **规范依据**：memory `feedback_test_module_name_unique.md`（mypy 按文件名推导 module，无 `__init__.py` 时跨目录同名 → `Duplicate module` 错误）

### 5.5 测试隔离

- `e2e` 测试用 autouse fixture 重置 `everalgo.llm._default` + `_active`（与子项目 2.5 模式一致）
- boundary / user_memory unit tests 用 per-call `llm=fake` 注入，不依赖 `everalgo.llm` 全局状态（无需 fixture）

---

## 6. CI Pipeline 升级

### 6.1 当前状态（占位）

`.gitlab-ci.yml` 57 行，3 stages（lint / test / build），pytest `allow_failure: true`，无 mypy job，build stage 是 placeholder。

### 6.2 升级目标

```yaml
# .gitlab-ci.yml (sub-project 4 final)

stages:
  - lint
  - test

# Lint stage — 3 parallel jobs
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

# Test stage — runs after lint passes
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
  # allow_failure: removed (was true; now strict)
```

### 6.3 删除项

- `build` stage —— 首次 PyPI release 再加（每个 `everalgo-*` distribution 独立 build）
- `pytest.allow_failure: true` —— strict 模式

### 6.4 不做项（推迟到子项目 5+）

- per-package matrix（`only: changes:` 只跑改动 package 的 test）—— LangChain `libs/` 风格，需要 GitLab CI `parallel: matrix` 写法，scope 较大
- `--cov` coverage 收集
- multi-Python-version 矩阵（3.12 + 3.13）—— 当前只支持 3.12

---

## 7. 关键设计决策

| # | 决策 | 取值 | 依据 |
|---|---|---|---|
| 1 | 算子形态 | stateless class（PascalCase）+ `aextract`/`adetect` async + `extract`/`detect = async_to_sync()` 一行桥接 | ADR 010 line 205-212 |
| 2 | 接口契约 | Protocol structural subtyping（不 ABC，不强制继承） | ADR 011 line 117 |
| 3 | LLM 注入 | per-call `llm: LLMClient \| None = None` + `everalgo.llm.resolve(llm)` | 子项目 2.5（已落地）+ design.md line 833 |
| 4 | Prompt 注入 | per-call `prompt: str \| None = None`（细粒度）+ caller monkey-patch 模块常量（粗粒度全局） | design.md §1.4 line 397-410 |
| 5 | Prompt 双语 | en + zh 都做（4 prompt 文件） | design.md §1.2 line 96 + AGENTS.md §7 step 6 + BOSS Q1=A |
| 6 | Prompt 命名后缀 | `_EN` / `_ZH` 后缀消歧义 | memsys_opensource `prompts/{en,zh}/episode_mem_prompts.py` 同模式 + monkey-patch 友好 |
| 7 | `count_tokens` 实现 | 简化估算（`len(text) // 4`），不引入 tiktoken | 子项目 1-3 「零新增依赖」基线一致 + BOSS Q4 拍板 |
| 8 | 错误层 | EverAlgo 不加 retry / fallback；LLM SDK 自带 2 次重试 | design.md §2.3 line 528-530 |
| 9 | `everalgo-boundary/pyproject.toml` 加 `asgiref>=3.0` | sync 桥接需要 | ADR 010 line 220 |
| 10 | `everalgo-user-memory/pyproject.toml` 暂去 `everalgo-clustering` 依赖 | 最小集 EpisodeExtractor 不依赖 cluster | design.md §2.3 line 540 + line 686 |
| 11 | user_memory re-export ChatMemCellExtractor from boundary | EverOS 双访问路径契约 | design.md line 122 + line 311-313 |
| 12 | 测试 e2e + per-extractor unit 双层 | 子项目 4 命名「Reference impl + CI」明示 e2e 必需 | BOSS Q2=A |
| 13 | 测试文件名加 sub-package 前缀避同名 | `test_boundary_public_api.py` + `test_user_memory_public_api.py` | memory `feedback_test_module_name_unique.md` |
| 14 | CI 4 jobs（ruff check / format / mypy / pytest）+ 单一矩阵 + strict | 真实有 lint/mypy/test 跑；per-package matrix 推迟 | BOSS Q3=B |
| 15 | 不做 `boundary/{workspace,agent}.py` / `_force_split.py` / `user_memory/{foresight,atomic_fact,profile}.py` | 子项目 4 最小集 | BOSS Q1（功能层级）=B 拍板 |

---

## 8. 行业参考

### 8.1 算法库 reference impl 形态（class with single async method）

| 项目 | 参考 | 形态 |
|---|---|---|
| **memsys_opensource** | `release/20260403:src/memory_layer/memory_extractor/episode_memory_extractor.py:46-100` | `class EpisodeMemoryExtractor(MemoryExtractor): __init__(self, llm_provider, episode_prompt, ...) async def extract(self, request) -> ...` |
| **DSPy** | `dspy.Predict` 等 module | class with `forward()` method |
| **LlamaIndex** | `BaseExtractor.aextract()` (`llama-index-core/.../extractors/interface.py`) | abstract method on Pydantic model class |
| **pydantic-ai** | `Agent.run()` / `Agent.iter()` | class with both sync + async methods |

EverAlgo 选 **stateless class + module-level helpers** 模式 —— 类似 sklearn `KMeans()` 但完全无 `fit()` / 状态持有；ADR 011 已选 Protocol structural（不强制继承），保留 BYOI（Bring Your Own Implementation）灵活性。

### 8.2 Prompt 双语模式

memsys_opensource `release/20260403:src/memory_layer/prompts/`：
- `en/episode_mem_prompts.py` 146 行（EpisodeExtractor 用）
- `zh/episode_mem_prompts.py` 同款中文
- 每文件 1 个 module-level 常量 `EPISODE_GENERATION_PROMPT`
- `memory_layer/prompts/__init__.py` 内 `get_prompt_by(name)` helper（按全局 `LANG` env 选 en/zh）

EverAlgo 简化版（no env-based selection，per-call + monkey-patch 双路径）。

### 8.3 GitLab CI Python 模板

| 项目 | 参考 | 关键模式 |
|---|---|---|
| **pytorch** | `.github/workflows/lint.yml` | 多 lint job 并行 stage（ruff / pyright / shellcheck） |
| **pydantic-ai** | `.github/workflows/ci.yml` | uv-based + multiple lint jobs + pytest matrix（3.12 + 3.13） |
| **LangChain core** | `libs/core/Makefile` + GitHub Actions | per-package job（`libs/core/...`），用 path filtering 触发 |

EverAlgo 选**单一矩阵**（不分 package）—— scope 控制，子项目 5+ 真有 path-based 矩阵需求时升级。

---

## 9. 字段决策清单（已对齐，无待 BOSS 校准项）

| # | 决策点 | 取值 | 来源 |
|---|---|---|---|
| 1 | 功能层级（demo / minimal / full） | minimal reference impl | BOSS Q1=B（2026-05-08） |
| 2 | LLM 注入机制 | 子项目 2.5 `everalgo.llm.resolve()` | 子项目 2.5 已落地 |
| 3 | Prompt 双语 | en + zh | BOSS Q1=A（重新定义后） |
| 4 | 测试范围 | unit + e2e 双层 | BOSS Q2=A |
| 5 | CI scope | 4 jobs + 单一矩阵 + strict | BOSS Q3=B |
| 6 | `_tokenize` | 落地 `count_tokens` 简化版 | BOSS Q4=A + 后续细化（不引入 tiktoken） |
| 7 | `boundary` 物理布局 | `chat.py` + `_tokenize.py` + `prompts/{en,zh}/<op>.py` | design.md §1.2 line 104-107 |
| 8 | `user_memory` 物理布局 | `episode.py` + `prompts/{en,zh}/<op>.py` | design.md §1.2 line 121-124 |
| 9 | 算子形态 | stateless class | ADR 010 line 205-212 |
| 10 | sync 桥接 | `extract = async_to_sync(aextract)` 一行 | ADR 010 line 212 |

---

## 10. Out of Scope（明确移出最小集）

1. `boundary/{workspace, agent}.py` —— design.md 列出，未来 SemVer minor bump 增量加
2. `boundary/_force_split.py` —— count_tokens 简化版不需要硬切分
3. `user_memory/{foresight, atomic_fact, profile}.py` —— 4 个独立路径，未来加
4. `agent_memory/*` —— 子项目 5+
5. `clustering/*` —— 独立子包 `everalgo-clustering`，子项目 6+
6. `rank/*` —— 独立子包 `everalgo-rank`，子项目 7+
7. `parser/*` 多模态 —— 独立子包，未来扩展
8. `knowledge/*` —— 独立子包
9. tiktoken 真实 tokenizer —— 简化估算够用
10. CI per-package matrix —— 单一矩阵足够
11. CI build stage —— 首次 PyPI release 再加
12. `--cov` coverage 收集
13. multi-Python-version 矩阵（3.13） —— 当前只支持 3.12

---

## 11. 自审（writing-plans 之前）

✅ **Spec coverage**：BOSS 4 个澄清问题（Q1 prompt 双语 / Q2 测试范围 / Q3 CI scope / Q4 tokenize）的取值都对应到 §3-§7 具体段落
✅ **Placeholder scan**：grep 无 `TODO` / `TBD` / `FIXME`
✅ **Internal consistency**：§2 file map 的文件 = §3 公开 API 实现 = §5 测试矩阵覆盖
✅ **Scope check**：单 implementation plan 可实现（约 12-15 个 TDD 任务，scope 比子项目 1+2+3+2.5 都大但仍可控）
✅ **Ambiguity check**：算子 `prompt=` per-call 注入 + 模块常量 monkey-patch 双路径已说明（§4.2）；`test_*_public_api.py` 命名冲突解法明示（§5.4）；`pyproject.toml` 改动列出（§7 决策 9-10）
✅ **行业依据 cite 完整**：每关键决策都给依据（§7 + §8）
✅ **设计自检 4 处全在文中**：算子形态 / count_tokens 实现 / Prompt 后缀 / re-export

下一步：进入 `superpowers:writing-plans` skill 撰写 implementation plan（约 12-15 个 TDD 任务）。
