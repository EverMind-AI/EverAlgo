# EverAlgo Foundation 子项目设计文档

| 字段 | 值 |
|------|-----|
| 子项目编号 | 1（共 4 子项目） |
| 状态 | Brainstorm 完成（已收紧到 EPISODE 最小集），**待 BOSS 审 spec**；通过后 → writing-plans |
| 范围 | (a) `everalgo.types.*` —— **EPISODE 路径最小集** 4 个对外符号 + (b) `everalgo.prompts.validator` 2 函数 |
| 不在范围 | DualInterface mixin / 非 Episode 路径的 6 个 memory type / clustering 全部 type / ADR 013 + design.md §2.4 patch / Message tool_calls / sender_id / source_type 等扩展字段。详见 §8 |
| 依赖 | 无（最底层） |
| 解锁 | 子项目 2/3/4（同样按 EPISODE 最小集 scope） |
| 估时 | ~0.5 工作日 |
| 设计源头 | [`docs/design.md`](../../design.md) §1.2 / §1.4 + ADR 010/011；[`memsys_opensource@release/20260403`](../../../../memsys_opensource/) 实测字段；BOSS 在线确认（`owner_id` 跨 user/agent + EPISODE 最小集 + 删 ADR 013） |

---

## 1. 背景与目标

### 1.1 算法同学的痛点

8 个 distribution 当前 `__init__.py` 全是 `__all__: list[str] = []` placeholder。算法同学要写第一个 `EpisodeExtractor` 必须先有：

- **签名能写出来**：`async def aextract(self, memcell: MemCell) -> list[Episode]` 中 `MemCell` / `Episode` 必须是真实可构造的 type
- **Prompt 改了能立刻发现错误**：占位符缺失不能等到 LLM 调用才暴露

本子项目交付这两条最小地基，仅覆盖 EPISODE 路径。

### 1.2 EPISODE 路径调用形态（参照 opensource 完整路径）

完整生产路径（实测 [`mem_memorize.py:805-885`](../../../../memsys_opensource/src/biz_layer/mem_memorize.py)）：

```
caller 喂 raw messages: list[Message]
    │
    ↓  ChatMemCellExtractor.adetect(messages, *, llm) → list[MemCell]    [boundary 切分]
    │  （LLM-decided 边界检测；opensource 还有 force-split + StatusResult
    │   控制位，最小 reference impl 不做）
    │
    ↓  for memcell in memcells:                                          [extract phase]
    │     EpisodeExtractor.aextract(memcell, *, llm) → list[Episode]
    │  （内部：拼 prompt + check_placeholders → call LLM → parse JSON）
    │
    返回 list[Episode(id, owner_id, episode, timestamp, parent_type, parent_id)]
```

这条端到端流程**只用到 4 个对外符号**：MessageRole / Message / MemCell / Episode。boundary 输入是 `list[Message]`，输出 `list[MemCell]`；extractor 输入 `MemCell`，输出 `list[Episode]`。

**子项目 4 reference impl 含 boundary + episode 完整端到端**（boundary 最简实现，无 force-split / 无 StatusResult / 单条强制切出 ≥1 MemCell；production-grade boundary 留 follow-up 子项目）。

### 1.3 字段来源原则（BOSS 拍板）

字段严格按 `memsys_opensource@release/20260403` 现状裁剪，**只保留 EPISODE 路径真实用到的字段**：

- **保留**：LLM prompt 输入必需（role/content/timestamp）+ extractor 输出主字段（episode）+ 数据血缘（id/parent_id/parent_type）+ 主体（owner_id）
- **删除**：tool_calls / tool_call_id / sender_id / source_type / 二级字段（subject/summary/keywords/...）—— 这些扩展由后续 SemVer minor bump 增量加（pydantic BaseModel 加非必填字段是兼容变更）
- **重命名**：`user_id` → `owner_id`（跨 user/agent 命名一致；agent path 也跑 EpisodeExtractor 的实测见 [`mem_memorize.py:870-885`](../../../../memsys_opensource/src/biz_layer/mem_memorize.py)）

> ✅ **设计自检**
> - **为什么 owner_id 而非 user_id**：实测 `mem_memorize.py:870-885` agent path 也调 EpisodeExtractor（`asyncio.gather(_timed_extract_episodes, _timed_extract_agent_case)`），design.md §2.4 line 697 显式声明"episode 永远跑（任何 MemCell 类型）"——Episode 跨 user/agent；统一 owner_id 命名解决跨用语义。
> - **为什么 Message 不留 tool_calls 字段**：caller 喂 EpisodeExtractor 前已过滤 tool message（opensource `MemCell.conversation_data` 过滤同款），过滤逻辑由 caller 写；Message 类型上根本不存在 tool_calls 字段语义最干净，避免 caller 不过滤直接传时 LLM prompt 含 tool_calls 干扰。
> - **为什么 type 用 `pydantic.BaseModel` 而非 dataclass**：本子项目 3 个 memory type 都是入参/落盘/跨进程传输；pydantic 提供字段校验 + JSON round-trip 内置。
> - **为什么 Episode 二级字段（subject/summary/keywords）全砍**：第一版 LLM prompt 只 require `{"episode": "..."}` 主字段；二级字段是 hybrid 检索 / BM25 索引才需要的（最小范例不做检索）；`extra="allow"` 让 LLM 多输出字段被静默接受，未来 minor bump 加。

---

## 2. Type 字段定义表（4 个对外符号，EPISODE 最小集）

### 2.1 `everalgo.types.memcell`

#### 2.1.1 `MessageRole`

```python
from enum import Enum

class MessageRole(str, Enum):
    """Conversation role taxonomy — minimal for EPISODE path.

    EPISODE path consumes user/assistant messages only. ``tool`` role
    messages are filtered by caller before reaching EpisodeExtractor
    (opensource ``MemCell.conversation_data`` 同款过滤). Adding ``TOOL`` /
    ``SYSTEM`` later is a SemVer minor bump compatible change.
    """
    USER = "user"
    ASSISTANT = "assistant"
```

#### 2.1.2 `Message`

| 字段 | 类型 | 默认 | 算法用途（一句话） |
|---|---|---|---|
| `role` | `MessageRole` | required | LLM prompt 拼接时区分 user/assistant 句首；boundary 切分识别交替模式（最小集不涉 boundary，留作未来扩展兼容） |
| `content` | `str` | required | LLM prompt 拼接源；Episode 抽取的实际素材 |
| `timestamp` | `int` (ms) | required | Episode.timestamp 推导（取 messages 末尾 / 代表时间） |

```python
from pydantic import BaseModel, ConfigDict

class Message(BaseModel):
    """Single conversation message — minimal fields for EPISODE path.

    字段集严格按 EpisodeExtractor 真实需要：role / content / timestamp 三件套。
    其余字段（sender_id / tool_calls / tool_call_id）按未来真实需求 SemVer minor
    bump 增量加（兼容变更）。
    """
    role: MessageRole
    content: str
    timestamp: int  # Unix epoch milliseconds

    model_config = ConfigDict(extra="ignore")
```

> `extra="ignore"` 让 EverOS payload 多余字段（`sender_id` / `sender_name` / `message_id` / `tool_calls` 等）被静默忽略，反序列化兼容 opensource 现状 dict。

#### 2.1.3 `MemCell`

| 字段 | 类型 | 默认 | 算法用途 |
|---|---|---|---|
| `id` | `str` | required | EpisodeExtractor 输出的 Episode.parent_id 锚点（数据血缘） |
| `messages` | `list[Message]` | required | LLM prompt 上下文素材 |
| `timestamp` | `int` (ms) | required | Episode.timestamp 默认值（caller 不显式传时取） |

```python
class MemCell(BaseModel):
    """Boundary extractor output — coherent slice of conversation.

    Minimal fields for EPISODE path. boundary metadata
    (source_type / sender_ids / start_idx / token_count / boundary_reason / ...)
    add later when boundary subpackage lands.
    """
    id: str
    messages: list[Message]
    timestamp: int

    model_config = ConfigDict(extra="ignore")
```

### 2.2 `everalgo.types.memories`

#### 2.2.1 `Episode`

| 字段 | 类型 | 默认 | 算法用途 |
|---|---|---|---|
| `id` | `str` | required | Episode 主键（uuid4） |
| `owner_id` | `str` | required | 归属主体（user_id / agent_id 跨 path 共用名）；caller 传入 |
| `episode` | `str` | required | LLM 抽取的事件叙述全文（**主字段**） |
| `timestamp` | `int` (ms) | required | 事件代表时间；从 MemCell 继承或 caller 传 |
| `parent_type` | `str` | `"memcell"` | 数据血缘类型（design.md §2.4 ProfileExtractor 反查依据） |
| `parent_id` | `str` | required | 上游 MemCell.id 反查 |

```python
class Episode(BaseModel):
    """User-side episodic memory — minimal fields for first reference impl.

    Cross-link: agent path also produces Episode (mem_memorize.py:870-885 实测 +
    design.md §2.4 line 697 "episode 永远跑"). owner_id 在 agent path 下仍指 user
    （agent 是参与者，非 owner）.

    Secondary fields (subject / summary / keywords / location / start_time /
    end_time / sender_ids / original_data) intentionally omitted — caller's LLM
    extractor may emit them, ``extra="allow"`` lets them through transparently
    until a future minor bump promotes them to first-class fields.
    """
    id: str
    owner_id: str
    episode: str
    timestamp: int
    parent_type: str = "memcell"
    parent_id: str

    model_config = ConfigDict(extra="allow")
```

> `extra="allow"` 比 `"ignore"` 更松——LLM 抽出的 secondary 字段（如 `subject`）会被保留在 model 实例上（`.subject` 可访问），方便 future-proof EverOS 持久化层用未来字段。

### 2.3 `everalgo.types.__init__` re-export 清单

```python
"""Public data contracts for EverAlgo — minimal EPISODE-path subset."""

from everalgo.types.memcell import Message, MessageRole, MemCell
from everalgo.types.memories import Episode

__all__ = [
    "Message", "MessageRole", "MemCell",
    "Episode",
]
```

→ 4 个对外符号。算法同学可 `from everalgo.types import MemCell` 或 `from everalgo.types.memcell import MemCell` 两条路径。

---

## 3. Prompts validator

### 3.1 `everalgo.prompts.validator.check_placeholders`

```python
import string
from typing import Iterable


def check_placeholders(prompt: str, *, required: Iterable[str]) -> None:
    """Assert that ``prompt`` contains every Python format placeholder in ``required``.

    Used at module-import time to fail fast if a prompt template was edited and a
    placeholder removed.

    Args:
        prompt: Template string with ``{placeholder}`` markers.
        required: Names that must appear as ``{name}`` in the template.

    Raises:
        ValueError: If any required placeholder is missing. Diagnostic message lists
            both missing names and any extra placeholders the template contains.

    Example:
        >>> check_placeholders(EPISODE_EXTRACTION_PROMPT, required=["messages", "now"])
    """
```

实现要点（写到 plan 里）：
- 用 `string.Formatter().parse(prompt)` 提取所有 `{name}` 占位符
- 计算 `set(required) - set(found)` 缺失项 → raise
- 同时计算 `set(found) - set(required)` 多余项（提示但不 raise，多余 kwargs 不算错）
- raise ValueError 时带具体名字列表

### 3.2 `everalgo.prompts.validator.check_length`

```python
from typing import Callable


def check_length(
    prompt: str,
    *,
    max_tokens: int,
    tokenizer: Callable[[str], int] | None = None,
) -> None:
    """Assert that ``prompt`` is at most ``max_tokens`` tokens long.

    Args:
        prompt: Rendered prompt (post-format).
        max_tokens: Hard ceiling.
        tokenizer: Token counter. ``None`` falls back to ``len(prompt) // 4 + 1``
            (OpenAI 经验法则；over-estimates 偏严不会假阴性放行超长 prompt).

    Raises:
        ValueError: If estimated count exceeds ``max_tokens``.
    """
```

实现要点：
- 默认 fallback：`max(1, len(prompt) // 4 + 1)`
- 算法同学若要精确计数，传 `tokenizer=` 注入

> ✅ **设计自检**
> - **为什么不强制传 tokenizer**：子项目 1 没有 boundary 包，无 `count_tokens` 可注入；强制传等于阻塞 validator 早期使用。粗估 fallback 偏严不会假阴性。
> - **为什么不删 check_length**：长度超限是 LLM ContextLength 错误（[ADR 012 §D4](../../decisions/012-llm-stack-architecture.md)），但**预检比 LLM 调用失败便宜**（省 1 次 round-trip + 给清晰错误位置）；EPISODE prompt 拼对话历史时易超长，预检有价值。

---

## 4. 测试结构规范（pydantic-ai 模式）

### 4.1 物理布局

```
packages/everalgo-core/tests/
  conftest.py              # 包级 fixture 入口（暂为空骨架，子项目 3 填）
  types/
    test_message.py
    test_memcell.py
    test_episode.py
    test_round_trip.py     # 跨 type 序列化 round-trip 参数化
  prompts/
    test_validator.py
```

仓库根 `tests/` 子项目 1 不动（跨包集成测试到子项目 4 才出现）。

### 4.2 规范要点（6 家明星项目调研结论）

| 项 | 选择 | 出处 |
|---|---|---|
| 命名 | `test_<module>.py` | pydantic-ai / sklearn / dspy / numpy / pytorch / transformers 6/6 一致 |
| 单测内部结构 | function + `@pytest.mark.parametrize` 为主 | pydantic-ai / sklearn / dspy 三家主流 |
| conftest | 仓库根一份 + 每包 `packages/<dist>/tests/conftest.py` 一份 | pydantic-ai / dspy 模式 |
| 跨包 fixture | 通过 `everalgo.testing.*` 公开 import | tests 不安装到 venv，跨包 import 唯一可靠路径 |
| integration vs unit 分目录 | **不分**，按领域拆 | 6/6 一致 |

### 4.3 单测内容（每 type 至少覆盖）

| 测试 | 覆盖什么 |
|---|---|
| 构造正常路径（minimum required + 全字段 + extra） | 字段类型校验 / 默认值生效 / extra="allow" 字段访问 |
| 构造缺必填字段 → ValidationError | pydantic 强制约束 |
| 序列化 round-trip（`.model_dump_json()` → `.model_validate_json()`） | JSON 兼容性 |
| extra 字段处理 | Message/MemCell `extra="ignore"`：opensource payload 含多余字段（sender_id/tool_calls/...）被静默忽略；Episode `extra="allow"`：LLM 多输出 subject/summary 等字段保留可访问 |
| owner_id 必填测试（仅 Episode） | 主体语义不可缺 |

`check_placeholders` 测试：
- pass：所有占位符齐全
- fail：缺 1 个 / 缺多个 / 完全没有占位符
- multiple required 但 prompt 里也有多余占位符 → pass（多余不 raise）

`check_length` 测试：
- pass：default tokenizer 在 max_tokens 内
- fail：default tokenizer 超 max_tokens
- pass with custom tokenizer：注入精确 tokenizer 改变结果

---

## 5. File Map

```
packages/everalgo-core/
  src/everalgo/
    types/
      __init__.py     # re-export 4 个符号 + __all__
      memcell.py      # Message, MessageRole, MemCell
      memories.py     # Episode
    prompts/
      __init__.py     # 已存在 placeholder, 不改
      validator.py    # check_placeholders, check_length
  tests/
    conftest.py       # 空骨架（子项目 3 填）
    types/
      test_message.py
      test_memcell.py
      test_episode.py
      test_round_trip.py
    prompts/
      test_validator.py
  pyproject.toml      # 加 pydantic>=2.7
```

### 5.1 `everalgo-core/pyproject.toml` 改动

```diff
 dependencies = [
+  "pydantic>=2.7",
 ]
```

> **`pydantic>=2.7`**：BaseModel + Field + ConfigDict 都用 v2 API，2.7 是 ConfigDict 稳定版本。**numpy 不在最小集**（ClusterState 不引入），后续 clustering 子项目落地时再加。

---

## 6. 验收标准

子项目 1 完成的判定：

1. ✅ `uv sync --all-packages` 全 8 dist 安装成功
2. ✅ `uv run python -c "from everalgo.types import MemCell, Message, MessageRole, Episode; from everalgo.prompts.validator import check_placeholders, check_length; print('OK')"` 输出 `OK`
3. ✅ `uv run pytest packages/everalgo-core/tests/` 全绿
4. ✅ `uv run ruff check packages/everalgo-core/` 0 issue
5. ✅ `uv run ruff format --check packages/everalgo-core/` 0 diff
6. ✅ `uv run mypy packages/everalgo-core/` 0 error
7. ✅ 子项目 4 EpisodeExtractor reference impl 引用本子项目类型时 import 不报错（这条由子项目 4 验收）

---

## 7. Out of Scope

按 EPISODE 路径最小集原则，下面这些**全部**移出本子项目：

### 7.1 类型扩展字段（未来 SemVer minor bump 增量加）

| Type | 移出字段 | 加回时机 |
|---|---|---|
| `MessageRole` | `TOOL` 值 | agent path tool message 真实需要时（按 boundary 子项目 / agent_memory 子项目落地） |
| `Message` | `sender_id` / `tool_calls` / `tool_call_id` / `ToolCall` 子类 | agent_memory 子项目 / group 场景需要 per-sender 时 |
| `MemCell` | `source_type` / `SourceType` enum / `sender_ids` | boundary 子项目（多 source 路由分发）+ agent_memory（per-sender 拆分） |
| `Episode` | `subject` / `summary` / `keywords` / `location` / `start_time` / `end_time` / `sender_ids` / `original_data` | hybrid 检索 / BM25 索引需要二级字段时；`extra="allow"` 让 LLM 提前输出无害 |

### 7.2 完全移出的 type（其他子包/未来子项目落地）

- **`AtomicFact / Foresight / Profile`** —— 其他 user_memory extractor 输出，第 2 阶段做（依赖各自的 prompt + extractor 设计）
- **`AgentCase / AgentSkill`** —— agent_memory 子项目
- **`ClusterState / ClusterConfig / Candidate / ClusterId`** —— clustering 子项目
- **`RawFile / ParsedContent / KnowledgeMemory / RankInput / RankOutput / Hit / WorkspaceData / AgentTrace`** —— 各自子包用到时落地（YAGNI）
- **`ChatMessage / ChatResponse / Usage / ChatChunk`** —— LLM stack 子项目（归 `everalgo.llm.types` 内部，与 `everalgo.types` 区分）

### 7.3 架构改动暂不做

- **ADR 013（ClusterState 上提到 `everalgo.types.cluster`）** —— 没 ClusterState type，物理位置改动不需要
- **`docs/design.md` §2.4 line 564 注释 patch** —— 同上
- **`everalgo.types.common.py` / `agent.py` / `cluster.py` 三个文件** —— 没对应 type，不创建空文件

### 7.4 其他

- **DualInterface mixin**（原 1b）：YAGNI 删除，算法同学手写 `extract = async_to_sync(aextract)` 一行解决（[ADR 010 line 199-214 主样例](../../decisions/010-sync-async-dual-interface.md#L199)）；AGENTS.md §5 已经写了规则
- **AGENTS.md 加"手写 sync 桥接"规则的更新**：本子项目顺手做（不算独立 deliverable，写到 plan 里）
- **boundary `_tokenize.count_tokens`**：子项目 1 没有 boundary 包，prompt validator 默认用粗估 fallback；精确 tokenizer 等 boundary 子项目落地

---

## 8. 字段决策清单（已对齐，无待 BOSS 校准项）

| Type | 字段 | BOSS 拍板决策 |
|---|---|---|
| `MessageRole` | 取值 | **USER / ASSISTANT 二值**（去 TOOL；EPISODE 路径过滤后 caller 不喂 tool message） |
| `Message` | 字段集 | **role / content / timestamp 三件套** |
| `Message` | 删 | sender_id / tool_calls / tool_call_id / message_id / sender_name / content multimodal union |
| `MemCell` | 字段集 | **id / messages / timestamp 三件套** |
| `MemCell` | 删 | user_id_list / group_id / participants / source_type / sender_ids / event_id rename id / `_conversation_data_cache` |
| `Episode` | 字段集 | **id / owner_id / episode / timestamp / parent_type / parent_id 六字段** |
| `Episode` | 删 | episode_id 冗余 / participants / group_id / metadata / extend / created_at / updated_at / 二级字段（subject/summary/...） |
| 全部 type | 主体命名 | **owner_id**（替代 user_id；跨 user/agent path 共用） |
| 全部 type | metadata + extend | 合并为 `model_config extra="ignore"` (Message/MemCell) 或 `extra="allow"` (Episode) |

---

## 9. 自审（writing-plans 之前）

执行 brainstorming skill 的 spec self-review checklist：

| 检查项 | 结果 |
|---|---|
| Placeholder scan | ✅ 无 TBD/TODO；validator 实现细节标"see plan"是合理（plan 阶段的事） |
| 内部一致性 | ✅ 4 个对外符号在 §2.3 re-export 与 §2.1-§2.2 各 type 表能一一对应；`__init__` 列表 vs File Map 一致；§6 验收命令 import 列表与 re-export 完全匹配 |
| Scope check | ✅ 单子项目 = 4 type + 2 validator，~0.5 工作日，writing-plans 一份 plan 容得下 |
| Ambiguity | ✅ §8 字段决策清单全部 BOSS 拍板，无歧义 |
| 命名一致 | ✅ owner_id 在 §1.3 / §2.2.1 / §6 / §8 全部一致；MessageRole 仅 USER/ASSISTANT 二值贯穿 |
| 与 opensource 出处一致 | ✅ 字段决策回引 `release/20260403` 真实代码（memory_models.py / memory_types.py / mem_memorize.py 等） |
| EPISODE 路径完备 | ✅ caller 构造 MemCell → EpisodeExtractor.aextract → 返 list[Episode] 端到端只用 4 个对外符号，无外部类型依赖 |

---

*Spec 完成。BOSS 审 spec 文件无异议后，立刻调用 writing-plans skill 出实施计划。*
