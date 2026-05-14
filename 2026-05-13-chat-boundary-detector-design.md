# Chat Boundary Detector — 设计草案

> 状态：落稿（Section 1 + Section 2 + 与 new release 的对齐说明）
> 日期：2026-05-14（对齐 `evermemos-opensource` new release）
> 适用分支：`feat/user-memory-v2`
> 文件位置：仓库根目录（暂存于 brainstorming 区，正式归档前不入 `docs/decisions/`）

---

## 设计动机

把 opensource `ConvMemCellExtractor.extract_memcell()`（new release 575 行，杂揉 force-split / LLM 边界 / flush / MemCell 业务字段填充 / metrics / DI）拆解为**无状态算子** `adetect`，让切分决策、tail 管理、flush 策略、MemCell 构造各自归位。

核心痛点（来自 opensource 现状审查）：

- **tail 隐式丢弃**：opensource 算子只返 `(memcells, StatusResult)`，没切完的 messages 由上游靠自己维护的 history buffer「隐式」承接 —— 契约靠暗示不靠类型
- **flush 钉在算子内**：opensource 算子接 `request.flush: bool` 入参，混淆机制与策略
- **业务字段强耦合**：opensource 把 `user_id_list / group_id` 通过 `request` 对象注入算子，跨数据类型复用困难
- **依赖 DI**：opensource tokenizer 走 `TokenizerFactory` 单例，LLM 走 `LLMProvider` bean —— 单元测试需要 boot 整套 DI 容器

EverAlgo 的解法是把这四点统一外推：**算子无状态、tail 显式返回、is_final 替代 flush、LLM/tokenizer 走 Protocol**。算法内部（force-split + batch detection + 5-retry + 索引校验）则**逐行对齐 new release**。

---

## Section 1 · 算子签名 + 类型契约

```python
from typing import NamedTuple, Protocol


class LLMClient(Protocol):
    """对 LLM 客户端的最小协议：能补全 + 能数 token。"""

    async def acomplete(self, prompt: str) -> str: ...

    def count_tokens(self, text: str) -> int: ...


class DetectionOutput(NamedTuple):
    cells: list[MemCell]       # 已切出的 MemCell list (0 / 1 / N)
    tail: list[Message]         # 未切走的剩余 messages


async def adetect(
    messages: list[Message],
    *,
    llm: LLMClient | None = None,
    prompt: str | None = None,
    is_final: bool = False,
    hard_token_limit: int = 65536,
    hard_msg_limit: int = 500,
) -> DetectionOutput:
    """Stateless conversation boundary detection.

    Splits messages into MemCells via force-split rules + LLM batch detection.
    The trailing segment that LLM cannot confidently close is returned as `tail`
    for the caller to persist; on the next call, prepend it to fresh messages.

    Args:
        messages: Conversation messages.
        llm: LLM client (acomplete + count_tokens). None resolves to the
             algorithm library default registry.
        prompt: Batch boundary detection prompt template. None uses default.
        is_final: True forces the trailing segment into the last MemCell
                  (no tail allowed). False allows tail to be non-empty.
        hard_token_limit: Force-split threshold by token count.
        hard_msg_limit: Force-split threshold by message count.

    Returns:
        DetectionOutput(cells, tail).
            is_final=False -> tail may be non-empty.
            is_final=True  -> tail is always [].

    Raises:
        RuntimeError: All LLM retries exhausted.
    """
```

### 关键设计决策（已与 BOSS 对齐）

| 决策点 | 取舍 | 理由 |
|---|---|---|
| 入参形态 | 单一 `messages: list[Message]` | new release batch prompt 已合并 history+new 再切；保留区分等于让调用方做无意义拆分 |
| 返回形态 | `DetectionOutput(NamedTuple)` | 既支持位置 unpacking，又支持 `.cells / .tail` 命名访问，未来加字段不破坏二元解构 |

> **关于 NamedTuple 的访问方式**（澄清常见疑问）
>
> `NamedTuple` 继承自 `tuple`，下面三种访问都成立、可混用：
>
> ```python
> # 位置 unpacking —— 和返回 tuple 完全一样
> cells, tail = await adetect(messages)
>
> # 命名访问
> output = await adetect(messages)
> output.cells   # list[MemCell]
> output.tail    # list[Message]
>
> # 索引访问
> output[0], output[1]
> ```
>
> 这是选 `NamedTuple` 而不是 `dataclass` 的核心理由——`dataclass` 不是 tuple 子类，`a, b = result` 会直接报错。
| tail | 显式返回（属于返回值） | 替代 opensource 的隐式丢弃契约；类型可检查 |
| flush 语义 | 用 `is_final: bool` 入参替代 | 「机制 vs 策略分离」—— 算子永不主动收口，is_final 由上游 state machine 决定 |
| LLM 注入 | Protocol（duck typing） | 不走 DI，单元测试只需一个 fake 类 |
| tokenizer | 挂在 `LLMClient.count_tokens` | 消除独立 tokenizer 依赖；和 LangChain `BaseLanguageModel.get_num_tokens` 同模式 |
| llm / prompt 默认 | 保留 `Optional`，由算法库 registry 解析 | 「接上就能用」的开箱体验，evermem 一贯目标 |
| LLM 失败 | 5 次重试后 `raise RuntimeError` | 维持 opensource 行为；算子单一职责，失败由上游决策（重试/降级/跳过） |
| force-split 顺序 | 先于 LLM 调用 | 避免超长 prompt 喂给 LLM 触发 context overflow；和 LangChain `RecursiveCharacterTextSplitter` 同模式 |

---

## 使用示例

### 场景 1 · 流式增量调用（最常见）

上游 service 持续接收消息；攒一批就调 `adetect`，cells 立即持久化，tail 留作下次的种子。

```python
from evermem.memory.extract.chat import adetect

class ConversationBuffer:
    """上游 service 持有的 pending buffer，把 tail 显式接住。"""

    def __init__(self) -> None:
        self._pending: list[Message] = []

    async def on_new_messages(self, new_msgs: list[Message]) -> list[MemCell]:
        output = await adetect(self._pending + new_msgs)
        self._pending = output.tail        # 没切完的留到下次
        return output.cells                 # 已切出的交给持久化层
```

要点：
- `self._pending` 是上游的状态，算子本身仍是无状态
- 下次调用 `adetect` 时 tail 重新喂回 —— 等价于 504 「history + new」的双段语义，但合并由调用方完成

### 场景 2 · flush 强制收口

session 结束 / 用户主动 flush / pending 超时等场景，要求 tail 不再保留。

```python
async def on_session_end(buffer: ConversationBuffer) -> list[MemCell]:
    output = await adetect(buffer._pending, is_final=True)
    buffer._pending = []                    # is_final=True 保证 output.tail == []
    assert output.tail == []                # 类型契约
    return output.cells
```

要点：
- `is_final=True` 的语义是「不允许留 tail」—— 算子会把残段强行打包成最后一个 MemCell
- 调用方不需要再判断 `should_wait`，类型契约保证 tail 为空

### 场景 3 · 自定义 LLM / prompt（替换默认）

离线评测、A/B 测试、跨模型对比时，调用方可显式注入 LLM 客户端和 prompt。

```python
from evermem.component.llm import OpenRouterClient
from my_research.prompts import EXPERIMENTAL_BOUNDARY_PROMPT

my_llm = OpenRouterClient(model="claude-sonnet-4-6", temperature=0.0)

output = await adetect(
    messages=eval_fixture,
    llm=my_llm,
    prompt=EXPERIMENTAL_BOUNDARY_PROMPT,
    is_final=True,                          # 评测是 batch，必须切完
    hard_token_limit=32768,                 # 试更紧的限制
)

print(f"切出 {len(output.cells)} 个 cells, tail={len(output.tail)}")
```

要点：
- `OpenRouterClient` 只需实现 `LLMClient` Protocol 两个方法，duck typing 自动适配
- 评测场景几乎总是 `is_final=True`（不留尾、行为可断言）
- prompt template 必须含 `{messages}` 占位符（由 batch prompt 渲染规范保证）

---

## Section 2 · 算法流程

算子内部分 5 个 phase：输入校验 → 默认值解析 → force-split 循环 → LLM batch detection（含重试）→ 按 boundaries 切片 + `is_final` 收口。

### 流程图

```text
┌──────────────────────────────────────────────────────────────┐
│                  adetect(messages, ...)                      │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
                     ╔══════════════════════╗
                     ║ Phase 1 · 输入校验   ║
                     ╚══════════╤═══════════╝
                                │
                       messages == []?
                  ┌─────────────┴─────────────┐
                  │ yes                       │ no
                  ▼                           ▼
            return ([], [])         ╔══════════════════════════╗
                                    ║ Phase 2 · 默认值解析     ║
                                    ║ llm    = llm or default  ║
                                    ║ prompt = prompt or DFLT  ║
                                    ╚══════════╤═══════════════╝
                                               │
                                               ▼
                      ╔══════════════════════════════════════════╗
                ┌────▶║ Phase 3 · Force-split 循环               ║
                │     ║                                          ║
                │     ║ tokens(msgs) >= hard_token_limit         ║
                │     ║   OR len(msgs) >= hard_msg_limit ?       ║
                │     ╚══════════╤═══════════════════════════════╝
                │                │
                │      ┌─────────┴─────────┐
                │      │ yes               │ no
                │      ▼                   │
                │  split_at =              │
                │    find_force_split_pt   │
                │  cells.append(force_cell │
                │    messages[:split_at])  │
                │  messages = messages[    │
                │             split_at:]   │
                └──────┘                   │
                                           ▼
                            ╔════════════════════════════════════╗
                  ┌────────▶║ Phase 4 · LLM batch detection      ║
                  │ retry   ║                                    ║
                  │ (n<5)   ║ resp = await llm.acomplete(...)    ║
                  │         ║ parse JSON + schema validate       ║
                  │         ╚══════════╤═════════════════════════╝
                  │                    │
                  │         ┌──────────┴──────────┐
                  │         │ parse fail          │ parse ok
                  │         ▼                     ▼
                  │   attempt < 5 ?        filter 1<=b<len
                  │   ┌────┴────┐          sort boundaries
                  │   │ yes     │ no              │
                  └───┘         ▼                 ▼
                           raise RuntimeError     │
                                                  │
                                                  ▼
                            ╔════════════════════════════════════╗
                            ║ Phase 5 · 按 boundaries 切片       ║
                            ║                                    ║
                            ║ prev = 0                           ║
                            ║ for b in boundaries:               ║
                            ║   cells.append(make_cell(          ║
                            ║     messages[prev:b]))             ║
                            ║   prev = b                         ║
                            ║ tail = messages[prev:]             ║
                            ╚══════════╤═════════════════════════╝
                                       │
                                       ▼
                              is_final AND tail ?
                       ┌───────────────┴──────────────┐
                       │ yes                          │ no
                       ▼                              │
              cells.append(make_cell(tail))           │
              tail = []                               │
                       └───────────────┬──────────────┘
                                       │
                                       ▼
                            ╔════════════════════════════════════╗
                            ║ return DetectionOutput(            ║
                            ║   cells=cells, tail=tail)          ║
                            ╚════════════════════════════════════╝
```

### Phase 1 · 输入校验

```python
if not messages:
    return DetectionOutput(cells=[], tail=[])
```

空入参直接返回，避免后续无谓的 LLM 调用。这也覆盖了「`is_final=True` 但没消息」的退化场景——返回空二元组、调用方自行 noop。

### Phase 2 · 默认值解析

```python
llm = llm or _default_llm_from_registry()
prompt = prompt or DEFAULT_BATCH_BOUNDARY_PROMPT
```

`Optional` 入参在此被 resolve。默认 LLM / prompt 的注册机制属于算法库实现细节（在 Section 4 展开过；本文档只做 Section 1+2，注册机制留给实施阶段补）。

### Phase 3 · Force-split 循环

```python
while True:
    total_tokens = llm.count_tokens(_render_text(messages))
    total_msgs = len(messages)
    exceeds_token = total_tokens >= hard_token_limit
    exceeds_msgs = total_msgs >= hard_msg_limit
    if not (exceeds_token or exceeds_msgs):
        break

    split_at = _find_force_split_point(
        messages, llm, hard_token_limit, hard_msg_limit
    )
    trigger = "token_limit" if exceeds_token else "msg_limit"
    cells.append(_force_cell(messages[:split_at], trigger=trigger))
    messages = messages[split_at:]
```

要点：
- **顺序关键**：force-split 必须在 LLM 调用**之前**。超长 prompt 喂给 LLM 会触发 context overflow / 质量下降 / 成本激增
- `_find_force_split_point` 沿用 opensource 的 binary-halving 策略（起步 `hard_msg_limit - 1`，超 token 时折半；`_count_tokens` 必须包含 sender_name 前缀，因为 LLM 看到的也是这个前缀）
- 循环可能切出多个 cell；每次切完 `messages` 缩短直到都在 limit 内

> ✅ **设计自检**
> - **为什么 force-split 在 LLM 之前**：避免给 LLM 喂超过 context window 的 prompt
> - **规范依据**：LangChain `RecursiveCharacterTextSplitter`、OpenAI cookbook 长文档切分同模式
> - **备选方案**：滑窗多次调用 LLM 让它自己处理超长（C.1 路径）—— cost 线性放大、prompt 设计复杂，不可取

### Phase 4 · LLM Batch Detection（含重试）

```python
messages_text = _format_messages_with_indices(messages)
full_prompt = prompt.format(messages=messages_text)

for attempt in range(5):
    resp = await llm.acomplete(full_prompt)
    result = _parse_batch_response(resp)        # {boundaries, should_wait}
    if result is not None:
        valid = sorted([b for b in result.boundaries if 1 <= b < len(messages)])
        return BatchBoundaryResult(boundaries=valid)

raise RuntimeError("All 5 retries exhausted for boundary detection")
```

prompt 渲染（与 new release 对齐）：

```text
[1] [2024-03-10 09:00:00+00:00] Alice: Can you help me debug the login issue?
[2] [2024-03-10 09:01:00+00:00] Bob: Sure, let me check the logs.
...
```

- 格式：`[N] [YYYY-MM-DD HH:MM:SS+TZ] sender_name: content`
- `sender_name` 必须由 caller 在上游 enrichment 阶段填好（new release `GroupAddRequest` 在 API 边界用 Pydantic validator 强制 sender_id 必填，是这一阶段的落点）
- new release 已不再给 LLM 喂 `_calculate_time_gap` 那套"5 minutes / 2 hours"的自然语言时差描述 —— LLM 自己看 ISO timestamp 算时差

要点：
- **单次 LLM 调用**——batch prompt 一次输出所有 boundaries（沿用 new release 设计）
- **重试触发条件**：仅 JSON 解析失败 / schema 不匹配；auth / rate-limit / 网络等基础设施错误由 LLM 客户端处理后向上抛
- **JSON 解析容错**：依次尝试 ```` ```json ... ``` ```` fence 内、整体直接 `json.loads`、最外层 `{...}` 提取（new release `_parse_batch_boundary_response` 三段式 fallback）
- **index 校验**：`1 <= b < len(messages)`（1-based、严格小于末尾——末尾留给 tail）；越界条目直接丢弃 + 排序
- **`should_wait` 字段**：new release 仍解析并通过 `StatusResult.should_wait` 透传给上游作为「最后一段信息不足、建议先攒着」的 advice 信号。EverAlgo 的 `adetect` 用 `is_final` 入参把"是否留 tail"作为 policy 单点表达，不再把 should_wait 暴露给调用方 —— 上游若想表达"信息不足、再等等"，直接不传 `is_final=True`、用返回的 `tail` 做下一次输入即可。

> ✅ **设计自检**
> - **为什么不暴露 LLM 输出的 should_wait**：opensource 把 should_wait 既作 LLM 提示又作 StatusResult 返回，policy 信号和 mechanism 输出混在一起。EverAlgo 把 policy 收敛到 `is_final` 一个入参 —— LLM 只负责 mechanism（找出哪里能切），should_wait 在算子返回值里就是冗余信息（caller 已经能通过 `len(tail) > 0` 判断"残段是否存在"）
> - **规范依据**：UNIX 哲学「mechanism not policy」
> - **备选方案**：把 should_wait 透传到 `DetectionOutput` —— 多此一举，caller 用 `tail` + `is_final` 已可覆盖所有控制流

### Phase 5 · 按 boundaries 切片 + `is_final` 收口

```python
prev = 0
for b in boundaries:
    cells.append(_make_cell(messages[prev:b], trigger="llm"))
    prev = b
tail = messages[prev:]

if is_final and tail:
    cells.append(_make_cell(tail, trigger="forced_final"))
    tail = []

return DetectionOutput(cells=cells, tail=tail)
```

要点：
- `boundaries` 已排序，依次切片；每个 cell 对应 LLM 判定的一个完整 episode
- **末段始终留给 tail**——LLM 永不主动收口最后一段（它无法判断「后面是否还有更多」）
- `is_final=True` 是调用方表达「不留尾」的唯一开关——算子据此把 tail 强行打包成最后一个 cell，否则按 tail 返回

> ✅ **设计自检**
> - **为什么 LLM 永不主动收口 tail**：LLM 拿到的只是「过去发生过的消息」，无法判断「后面是否还有更多」；把「这段算不算完整 episode」的决策权留给 LLM 是反语义的
> - **规范依据**：new release batch prompt 设计原则「default to merging」隐含同样语义；机制/策略分离原则
> - **备选方案**：算子按「LLM 信心度」自动收口 —— 信心度阈值是策略，不在算子职责范围

### Trigger 类型汇总

| trigger | 出现 phase | 触发条件 |
|---|---|---|
| `"token_limit"` | Phase 3 | 累计 token 超 `hard_token_limit` |
| `"msg_limit"` | Phase 3 | 累计 msg 数超 `hard_msg_limit` |
| `"llm"` | Phase 5 | LLM batch detection 决定的边界 |
| `"forced_final"` | Phase 5 | `is_final=True` 强制收口 |

> 注：trigger 信息不进入 `MemCell` 字段。new release 的 `MemCell` 是有明确 schema 的 dataclass（`user_id_list / original_data / event_id / group_id / sender_ids / type / participants / timestamp`），但 trigger 不在其中——它是切分**过程**的元数据，不是结果的属性。如果未来要追踪「这个 cell 是怎么切出来的」（指标观测、debug），可以通过算子内部 metric 标签承载——属于上游观测，不属于算子接口契约。

---

## Section 3 · 与 new release 的对齐边界

为方便代码审查与未来回填，把"哪里逐行对齐 / 哪里刻意分叉"一次说清。

### 完全对齐（实现细节复刻 opensource）

| 对齐项 | new release 出处 | EverAlgo 落点 |
|---|---|---|
| force-split 二分折半 | `ConvMemCellExtractor._find_force_split_point` | `boundary/chat.py::_find_force_split_point` |
| Token 计数包含 sender 前缀 | `ConvMemCellExtractor._count_tokens` | `boundary/chat.py::_count_tokens` |
| 消息渲染格式 `[N] [ISO+TZ] sender_name: content` | `ConvMemCellExtractor._format_messages_with_indices` | `boundary/chat.py::_format_messages_with_indices` |
| `participant_ids` 只取 `role==user` 的 `sender_id`，**不读 refer_list** | `ConvMemCellExtractor._extract_participant_ids` | `boundary/chat.py::_extract_participants` |
| batch prompt + boundaries 数组输出 | `CONV_BATCH_BOUNDARY_DETECTION_PROMPT` | `boundary/prompts/{en,zh}/chat.py` |
| 5 次重试、仅 JSON 解析失败重试 | `ConvMemCellExtractor._detect_boundaries` | `boundary/chat.py::_detect_boundaries` |
| index 校验 `1 <= b < len(messages)` | 同上 | 同上 |
| `MemCell` schema | `api_specs/memory_types.py::MemCell` | `core/types/memcell.py::MemCell`（`user_id_list / original_data / event_id / group_id / sender_ids / type`） |
| `MessageSenderRole` 三值 `USER / ASSISTANT / TOOL` | `api_specs/memory_models.py::MessageSenderRole` | `core/types/memcell.py::MessageRole` |

### 刻意分叉（EverAlgo 改进 opensource 的部分）

| 分叉项 | new release | EverAlgo | 理由 |
|---|---|---|---|
| 算子状态 | `ConvMemCellExtractor.__init__(llm_provider, prompt, limits)` 持有 DI 配置 | 无状态函数 `adetect(messages, *, llm, prompt, is_final, ...)` | 单元测试一行 fake 就能跑；instance 可互换 |
| 返回值 | `(list[MemCell], StatusResult(should_wait))` | `DetectionOutput(cells, tail)` | tail 隐式 → 显式，类型可检查 |
| flush 表达 | `request.flush: bool` 在 `MemCellExtractRequest` 里 | `is_final: bool` 作为 `adetect` kwarg | 算子无 request 对象；policy 单点表达 |
| should_wait | `StatusResult.should_wait` 返回给 caller 做参考 | 不暴露 | caller 已有 `len(tail) > 0` + `is_final` 覆盖所有控制流 |
| group_id / user_id_list 注入 | 通过 `MemCellExtractRequest` 注入算子 | caller 在 `MemCell` 构造时自己填 | 算法库不持有业务字段 |
| 默认 limits 来源 | `os.getenv("MEMCELL_HARD_TOKEN_LIMIT", "65536")` | 模块常量 + per-call kwarg | 算法库不读环境变量 |

> 三句话说清：**算法内部 = 复刻 new release；公共接口 = 保留 stateless function + 显式 tail + is_final；业务字段 = caller 填，不由算子注入。**

---

## 文档完结

Section 1 + Section 2 + Section 3 已落稿，覆盖了算子的**接口契约**、**算法流程**、**与 new release 的对齐边界**三个核心面。其余 sections（错误处理细节、默认 registry 设计、测试策略、文件落位）属于实施阶段补充，按需在 `local/plans/` 下的实施计划文档中展开。
