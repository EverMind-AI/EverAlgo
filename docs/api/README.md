# API Index

Each EverAlgo distribution has its own README with a quick-start, public API surface, prompt customisation guide, and testing patterns.

---

## Per-distribution READMEs

| Distribution | README | What it provides |
|---|---|---|
| `everalgo-core` | [`packages/everalgo-core/README.md`](../../packages/everalgo-core/README.md) | `ChatMessage`, `MemCell`, `Episode`, `RankInput`/`RankOutput`, `LLMClient`, `OpenAICompatClient`, `FakeLLMClient` |
| `everalgo-boundary` | [`packages/everalgo-boundary/README.md`](../../packages/everalgo-boundary/README.md) | `adetect_boundaries`, `detect_boundaries`, `DetectionResult`, `adetect_boundary_step`, `BoundaryDecision` |
| `everalgo-clustering` | [`packages/everalgo-clustering/README.md`](../../packages/everalgo-clustering/README.md) | `Cluster`, `cluster_by_geometry`, `acluster_by_llm`, `cluster_by_llm` |
| `everalgo-rank` | [`packages/everalgo-rank/README.md`](../../packages/everalgo-rank/README.md) | `hybrid`, `agentic`, `cluster`, `maxsim`, `category`, business rankers, `fusion`, `weight`, `rerank` |
| `everalgo-parser` | [`packages/everalgo-parser/README.md`](../../packages/everalgo-parser/README.md) | `aparse`, `parse`, `ParsedContent`, `Modality`, `RawFile`; image / audio / document / URL parsers (video deferred) |
| `everalgo-user-memory` | [`packages/everalgo-user-memory/README.md`](../../packages/everalgo-user-memory/README.md) | `BoundaryDetector`, `EpisodeExtractor`, `EpisodeReflector`, `EpisodeSummarizer`, `ForesightExtractor`, `AtomicFactExtractor`, `ProfileExtractor`, `OutputLanguage` |
| `everalgo-agent-memory` | [`packages/everalgo-agent-memory/README.md`](../../packages/everalgo-agent-memory/README.md) | `BoundaryDetector` (re-exported from `everalgo-boundary`), three extractors, extraction result / skip reason types, `OpOutcome` |
| `everalgo-knowledge` | [`packages/everalgo-knowledge/README.md`](../../packages/everalgo-knowledge/README.md) | `KnowledgeExtractor`, `aclassify_category`, `classify_category`; `KnowledgeMemory` / `CategorySpec` live in `everalgo.types` |

---

## Integration contracts

- [`从 Episode 提取 Profile`](profile-from-episode-texts.zh-CN.md) — Chinese upstream integration contract for `ProfileExtractor.aextract_from_episodes`. The [English version](profile-from-episode-texts.md) remains the repository reference.

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
    Episode,            # narrative memory — owner_id, episode, subject, summary, timestamp
    Foresight,          # anticipated future event — owner_id, foresight, evidence, start_time, end_time, duration_days
    AtomicFact,         # single verifiable assertion — owner_id (str | None), content, timestamp
    Profile,            # structured user profile — owner_id, summary, timestamp, explicit_info, implicit_traits
    ExplicitInfoItem,   # one Profile.explicit_info item — description, category, evidence, observed_at; extra keys kept
    ImplicitTraitItem,  # one Profile.implicit_traits item — description, trait, basis, observed_at; extra keys kept
    AgentCase,          # distilled agent trajectory — id, timestamp, task_intent, approach, quality_score, key_insight
    AgentSkill,         # aggregated skill — id, cluster_id (caller-stamped), name, description, content, confidence
    AgentProfilePatch,  # one section-level SOUL.md / AGENTS.md edit — file, action, section, old_text, new_text
    AgentProfileUpdate, # AgentProfileExtractor result — patches, soul_diff / agents_diff, new_*_md, signals
    AgentProfileSignal, # below-gate implicit signal — caller persists and feeds back as pending_signals
    KnowledgeMemory,    # extracted knowledge from a file
    RankInput,          # multi-route recall candidates + cross-memory linkage
    RankOutput,         # ranked memory list
    ParsedContent,      # multimodal file parsed into structured content
)
```

The clustering value object lives in `everalgo.clustering`:

```python
from everalgo.clustering import Cluster
# Cluster: id (caller-supplied), centroid, count, last_ts, preview, members (caller entity ids)
```

The LLM contract lives in `everalgo.llm`:

```python
from everalgo.llm import LLMClient, LLMMessage, LLMOutputError, StrictOutputModel, parse_reply
```

`LLMClient.achat(messages, *, response_format, temperature=None, max_tokens=None, **extra)` takes `LLMMessage`s and a `StrictOutputModel` subclass and returns the reply text; an operator validates it with `parse_reply` and raises `LLMOutputError` on a bad reply. `everalgo.llm.LLMMessage` is the request message sent to the model; `everalgo.types.ChatMessage` is the domain type (conversation message). They are distinct classes. The full contract is [concepts/llm-contract.md](../concepts/llm-contract.md).
