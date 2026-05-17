# API Index

Each EverAlgo distribution has its own README with a quick-start, public API surface, prompt customisation guide, and testing patterns.

---

## Per-distribution READMEs

| Distribution | README | What it provides |
|---|---|---|
| `everalgo-core` | [`packages/everalgo-core/README.md`](../../packages/everalgo-core/README.md) | `ChatMessage`, `MemCell`, `Episode`, `RankInput`/`RankOutput`, `LLMClient`, `LLMConfig`, `FakeLLMClient` |
| `everalgo-boundary` | [`packages/everalgo-boundary/README.md`](../../packages/everalgo-boundary/README.md) | `detect_boundaries`, `DetectionResult`, `WorkspaceMemCellExtractor`, `_tokenize` |
| `everalgo-clustering` | [`packages/everalgo-clustering/README.md`](../../packages/everalgo-clustering/README.md) | `Cluster`, `cluster_by_geometry`, `cluster_by_llm` |
| `everalgo-rank` | [`packages/everalgo-rank/README.md`](../../packages/everalgo-rank/README.md) | `rank.episodic`, `rank.profile`, `rank.case`, `rank.skill`, `rank.fusion`, `rank.weight`, `rank.rerank` |
| `everalgo-parser` | [`packages/everalgo-parser/README.md`](../../packages/everalgo-parser/README.md) | `aparse`, `ParsedContent`, image / audio / document / video / URL parsers |
| `everalgo-user-memory` | [`packages/everalgo-user-memory/README.md`](../../packages/everalgo-user-memory/README.md) | `BoundaryDetector`, `EpisodeExtractor`, `ForesightExtractor`, `AtomicFactExtractor`, `ProfileExtractor` |
| `everalgo-agent-memory` | [`packages/everalgo-agent-memory/README.md`](../../packages/everalgo-agent-memory/README.md) | `AgentBoundaryDetector`, `AgentCaseExtractor`, `AgentSkillExtractor` |
| `everalgo-knowledge` | [`packages/everalgo-knowledge/README.md`](../../packages/everalgo-knowledge/README.md) | `KnowledgeExtractor`, `KnowledgeMemory` |

---

## Key types at a glance

All shared data contracts are in `everalgo.types`:

```python
from everalgo.types import (
    ChatMessage,        # single conversation turn — kind="text"
    ToolCallRequest,    # assistant-emitted tool invocation — kind="tool_call"
    ToolCallResult,     # tool execution result — kind="tool_result"
    ConversationItem,   # ChatMessage | ToolCallRequest | ToolCallResult (discriminated union)
    MemCell,            # boundary-segmented conversation slice; items: list[ConversationItem]
    Episode,            # narrative memory — owner_id, episode, subject, timestamp
    Foresight,          # anticipated future event — owner_id, foresight, evidence, start_time, end_time, duration_days
    AtomicFact,         # single verifiable assertion — owner_id (str | None), fact, timestamp
    Profile,            # structured user profile — owner_id, summary, timestamp; extra fields via extra="allow"
    AgentCase,          # distilled agent trajectory — id, timestamp, task_intent, approach, quality_score, key_insight
    AgentSkill,         # aggregated skill — id, cluster_id (caller-stamped), name, description, content, confidence
    KnowledgeMemory,    # extracted knowledge from a file (EXPERIMENTAL)
    RankInput,          # multi-route recall candidates + cross-memory linkage
    RankOutput,         # ranked memory list
    ParsedContent,      # multimodal file parsed into structured content (EXPERIMENTAL)
)
```

The clustering value object lives in `everalgo.clustering`:

```python
from everalgo.clustering import Cluster
# Cluster: id (caller-supplied), centroid, count, last_ts, preview, members (caller entity ids)
```

LLM wire types live in `everalgo.llm.types`:

```python
from everalgo.llm.types import ChatMessage, ChatResponse, Usage
```

`everalgo.llm.types.ChatMessage` is the LLM prompt wire type (sent to the model). `everalgo.types.ChatMessage` is the domain type (conversation message). They are distinct classes.
