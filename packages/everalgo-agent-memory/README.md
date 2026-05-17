# everalgo-agent-memory

Agent-side memory products for EverAlgo — `AgentCaseExtractor` distils an agent trajectory `MemCell` into one `AgentCase`; `AgentSkillExtractor` maintains a cluster's reusable skill set from accumulated cases. `AgentBoundaryDetector` handles boundary detection over mixed `ConversationItem` trajectories (chat + tool calls).

See the umbrella project: [EverAlgo monorepo](../../README.md) and the architecture document at [`docs/concepts/architecture.md`](../../docs/concepts/architecture.md).

## Install

```bash
pip install everalgo-agent-memory
# Auto-pulls: everalgo-core, everalgo-boundary, everalgo-clustering
```

## What this distribution provides

| Symbol | Role |
|---|---|
| `AgentBoundaryDetector` | Boundary detection on agent trajectories (filter → detect → remap for mixed `ConversationItem` lists) |
| `AgentCaseExtractor` | Distils one agent-trajectory `MemCell` into `[] \| [AgentCase]` (11-step pipeline) |
| `AgentSkillExtractor` | Aggregates one new `AgentCase` into incremental skill operations for a cluster; returns add / update / retire entries |

## Quick start

```python
import asyncio
import json

from everalgo.agent_memory.case import AgentCaseExtractor
from everalgo.llm.types import ChatResponse
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import AgentCase, ChatMessage, MemCell, ToolCall, ToolCallFunction, ToolCallRequest, ToolCallResult

_CASE_JSON = json.dumps({
    "task_intent": "Search for Python async retry libraries",
    "approach": "1. Search. 2. Filter. 3. Summarise.",
    "quality_score": 0.82,
    "key_insight": "Use tenacity AsyncRetrying for native async back-off.",
})

async def main() -> None:
    fake = FakeLLMClient(responses=[ChatResponse(content=_CASE_JSON, model="fake")])
    mc = MemCell(
        items=[
            ChatMessage(id="u1", role="user", content="Best async retry libs?", timestamp=1_700_000_000_000, sender_id="user"),
            ToolCallRequest(tool_calls=[ToolCall(id="c1", function=ToolCallFunction(name="web.search", arguments='{}'))], timestamp=1_700_000_000_100, sender_id="assistant"),
            ToolCallResult(tool_call_id="c1", content="Found: tenacity.", timestamp=1_700_000_000_200),
            ToolCallRequest(tool_calls=[ToolCall(id="c2", function=ToolCallFunction(name="web.search", arguments='{}'))], timestamp=1_700_000_000_300, sender_id="assistant"),
            ToolCallResult(tool_call_id="c2", content="tenacity supports async.", timestamp=1_700_000_000_400),
            ChatMessage(id="a1", role="assistant", content="Use tenacity AsyncRetrying.", timestamp=1_700_000_000_500, sender_id="assistant"),
        ],
        timestamp=1_700_000_000_500,
    )

    cases: list[AgentCase] = await AgentCaseExtractor(llm=fake).aextract(mc)
    if cases:
        print(cases[0].task_intent)

asyncio.run(main())
```

See [`examples/04_agent_memory_case.py`](../../examples/04_agent_memory_case.py) for the full runnable example.

## API surface

```python
class AgentBoundaryDetector:
    def __init__(self, *, llm: LLMClient) -> None: ...
    async def adetect(
        self, items: list[ConversationItem], *, is_final: bool = False, prompt: str | None = None
    ) -> DetectionResult: ...

class AgentCaseExtractor:
    def __init__(self, *, llm: LLMClient) -> None: ...
    async def aextract(
        self, memcell: MemCell, *,
        prompt_filter: str | None = None,
        prompt_compress: str | None = None,
        prompt_tool_pre_compress: str | None = None,
    ) -> list[AgentCase]: ...  # length 0 (filtered) or 1

class AgentSkillExtractor:
    def __init__(self, *, llm: LLMClient) -> None: ...
    async def aextract(
        self,
        case: AgentCase,
        *,
        existing_relevant_skills: Sequence[AgentSkill],
        supporting_cases: Sequence[AgentCase],
        prompt_success: str | None = None,
        prompt_failure: str | None = None,
        prompt_maturity: str | None = None,
        skip_quality_threshold: float = 0.2,
        skip_maturity_scoring: bool = True,
        maturity_threshold: float = 0.6,
        retire_confidence: float = 0.1,
        failure_quality_threshold: float = 0.5,
        max_case_history: int = 9,
        max_description_tokens: int = 400,
        max_content_tokens: int = 5000,
        maturity_trivial_change_ratio: float = 0.2,
        maturity_reeval_change_ratio: float = 0.4,
    ) -> list[AgentSkill]: ...
```

All class methods have a sync bridge: `extractor.extract(...)` is `async_to_sync(aextract)`.

### AgentCaseExtractor pipeline

The extractor runs an 11-step pipeline: strip-before-first-user → structural pre-filter → heuristic trim → over-size bail → LLM filter (skipped when ≥2 tool rounds) → tool pre-compress → LLM compress → parse → validate → build `AgentCase`. Returns `[]` when the trajectory is filtered out; returns `[AgentCase]` on success.

### AgentSkillExtractor return contract

Pass the new `AgentCase` and the pre-filtered `existing_relevant_skills` (e.g. top-K cosine from the caller's store) plus the associated `supporting_cases` (cases referenced by existing skills). The caller decodes add / update / retire by checking whether `skill.id` is already in `existing_relevant_skills` and whether `skill.confidence < retire_confidence`.

Cases with `quality_score < skip_quality_threshold` (default 0.2) short-circuit to `[]` without calling the LLM.

`AgentSkill.cluster_id` is always `""` on extraction — the caller stamps the cluster identity after persisting.

### Customising prompts

```python
import everalgo.agent_memory.prompts.case_filter as _cf
_cf.AGENT_CASE_FILTER_PROMPT = my_custom_filter_prompt   # global override
```

Or per-call: pass `prompt_filter=` / `prompt_compress=` / `prompt_tool_pre_compress=` to `aextract`.

## Related distributions

- [`everalgo-boundary`](../everalgo-boundary/) — `detect_boundaries` primitive used by `AgentBoundaryDetector`
- [`everalgo-clustering`](../everalgo-clustering/) — `cluster_by_llm` for skill-cluster assignment
- [`everalgo-rank`](../everalgo-rank/) — `CaseRanker` / `SkillRanker` for read-time ranking
