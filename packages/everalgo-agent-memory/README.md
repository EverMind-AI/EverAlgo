# everalgo-agent-memory

Agent-side memory products for EverAlgo — `AgentCaseExtractor` distils an agent trajectory `MemCell` into one `AgentCase`; `AgentSkillExtractor` maintains a cluster's reusable skill set from accumulated cases; `AgentProfileExtractor` proposes precision-first section-level patches to the agent's injected config files (SOUL.md / AGENTS.md). `BoundaryDetector`, re-exported unchanged from `everalgo-boundary`, cuts mixed `ConversationItem` trajectories (chat + tool calls) into `MemCell`s.

See the umbrella project: [EverAlgo monorepo](../../README.md) and the architecture document at [`docs/concepts/architecture.md`](../../docs/concepts/architecture.md).

## Install

```bash
pip install everalgo-agent-memory
# Auto-pulls: everalgo-core, everalgo-boundary, everalgo-clustering
```

## What this distribution provides

| Symbol | Role |
|---|---|
| `AgentCaseExtractor` | Distils one agent-trajectory `MemCell` into `[] \| [AgentCase]` (11-step pipeline) |
| `AgentSkillExtractor` | Aggregates one new `AgentCase` into incremental skill operations for a cluster; returns add / update / retire entries |
| `BoundaryDetector` | `everalgo.boundary.BoundaryDetector`, re-exported: boundary detection on agent trajectories; only chat messages reach the model, tool items stay in the cells |
| `AgentProfileExtractor` | Screens one trajectory `MemCell` for durable agent-config signals (four-gate, default noop) and returns validated section-level SOUL.md / AGENTS.md patches + unified diffs |

## Quick start

```python
import asyncio
import json

from everalgo.agent_memory.case import AgentCaseExtractor
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import AgentCase, ChatMessage, MemCell, ToolCall, ToolCallFunction, ToolCallRequest, ToolCallResult

# Every reply is the JSON object its call's reply model describes; the fake returns the text as scripted.
_FILTER_JSON = json.dumps({"has_exploration": True, "has_user_correction": False, "reason": "Searched, then verified."})
_CASE_JSON = json.dumps({
    "task_intent": "Search for Python async retry libraries",
    "approach": "1. Search. 2. Filter. 3. Summarise.",
    "key_insight": "Use tenacity AsyncRetrying for native async back-off.",
    "quality_score": 0.82,
})

async def main() -> None:
    fake = FakeLLMClient(responses=[_FILTER_JSON, _CASE_JSON])
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

    # Two tool rounds: lower the default min_tool_call_rounds gate (3) so the trajectory reaches the filter.
    cases: list[AgentCase] = await AgentCaseExtractor(llm=fake, min_tool_call_rounds=2).aextract(mc)
    if cases:
        print(cases[0].task_intent)

asyncio.run(main())
```

See [`examples/04_agent_memory_case.py`](../../examples/04_agent_memory_case.py) for the full runnable example.

## API surface

```python
class BoundaryDetector:  # everalgo.boundary.BoundaryDetector, re-exported
    def __init__(self, *, llm: LLMClient) -> None: ...
    async def adetect(
        self, items: Sequence[ConversationItem], *, is_final: bool = False, prompt: str | None = None,
        hard_token_limit: int = 65_536, hard_msg_limit: int = 500,
    ) -> DetectionResult: ...

class AgentCaseExtractor:
    def __init__(self, *, llm: LLMClient, gate: AgentCaseGate | None = None) -> None: ...
    async def aextract(self, memcell: MemCell) -> list[AgentCase]: ...  # length 0 (filtered) or 1

class AgentSkillExtractor:
    def __init__(self, *, llm: LLMClient) -> None: ...
    async def aextract(
        self,
        case: AgentCase,
        *,
        existing_relevant_skills: Sequence[AgentSkill],
        supporting_cases: Sequence[AgentCase],
        skip_quality_threshold: float = 0.2,
        skip_maturity_scoring: bool = True,
        maturity_threshold: float = 0.6,
        retire_confidence: float = 0.1,
        failure_quality_threshold: float = 0.5,
        max_description_tokens: int = 400,
        max_content_tokens: int = 5000,
        maturity_trivial_change_ratio: float = 0.2,
        maturity_reeval_change_ratio: float = 0.4,
    ) -> list[AgentSkill]: ...

class AgentProfileExtractor:
    def __init__(
        self, *, llm: LLMClient,
        min_recurrence: int = 2,      # implicit signals must recur this many times across sessions (gate 3)
        max_file_tokens: int = 8000,  # anti-bloat budget for each patched file
    ) -> None: ...
    async def aextract(
        self,
        memcell: MemCell,
        *,
        soul_md: str,
        agents_md: str,
        pending_signals: Sequence[AgentProfileSignal] = (),
    ) -> AgentProfileUpdate: ...
```

All class methods have a sync bridge: `extractor.extract(...)` is `async_to_sync(aextract)`.

Every LLM call follows the `LLMClient` contract of `everalgo-core` (`docs/concepts/llm-contract.md`): the operator
sends a static system message of rules plus a user message holding only the call's data, passes a pydantic reply
model as `response_format`, and validates the returned text against it. A reply that does not validate, or that
fails a business check the operator cannot recover from (a pre-compressed chunk of the wrong length, a tool message
with no `tool_call_id`), raises `LLMOutputError`; operators never retry.

### Diagnosing an empty result

`aextract` reports every rejection the same way — an empty list plus a log line — so a caller looking at `[]` cannot tell a too-simple trajectory from an LLM-filtered one. `AgentCaseExtractor` and `AgentSkillExtractor` each expose an `aextract_with_reason` variant (plus its `extract_with_reason` sync bridge) that returns the same decisions as typed values:

```python
result = await case_extractor.aextract_with_reason(memcell)
result.cases    # list[AgentCase] — same value aextract returns
result.reason   # CaseSkipReason | None ; None means a case was produced
result.detail   # dict — observed value + threshold, e.g. {"rounds": 2, "min_rounds": 3}

result = await skill_extractor.aextract_with_reason(case, existing_relevant_skills=..., supporting_cases=...)
result.pre_reason  # SkillSkipReason | None — short-circuit that fired before any LLM call
result.pre_detail  # dict — context for pre_reason, e.g. {"quality": 0.15, "threshold": 0.2}
result.outcomes    # one OpOutcome per LLM-proposed operation; len == len(operations)
result.skills      # the operations that succeeded — same value aextract returns
result.dropped     # the operations that did not, each with .reason and .detail
```

Note where each rejection's context lands: a per-operation drop carries it on `OpOutcome.detail`, while the pre-LLM short-circuit carries it on `pre_detail` — it fires before any operation exists, so there is no `OpOutcome` to hang it off.

The reasons are algorithmic: each names the gate that rejected the input. Attribution — caller mistake vs. not-yet-satisfied vs. infrastructure fault — depends on deployment context the library does not have, so that mapping belongs to the caller. `detail` carries numbers as structured data rather than prose so callers can compose their own message without parsing strings.

The two extractors differ in shape because the operations differ: a MemCell yields at most one case, so a single `reason` suffices; a skill extraction integrates one case against N existing skills and may add, update, or leave alone each of them, so outcomes are per operation. Note that `OpOutcome.op_index` is the operation's position in the LLM's `operations` list — not the `index` inside an update operation, which addresses `existing_relevant_skills`.

Exceptions are not reasons: whatever the LLM client raises, and `LLMOutputError` for a malformed reply, still propagate. They mean the pipeline could not run, not that the input was judged unworthy. A structurally malformed skill operation (an action outside add / update / none, a non-integer `index`) therefore fails the whole reply rather than producing a per-operation reason.

### AgentCaseExtractor pipeline

The extractor runs an 11-step pipeline: strip-before-first-user → structural pre-filter → heuristic trim → over-size bail → LLM filter (skipped when ≥2 tool rounds) → tool pre-compress → LLM compress → parse → validate → build `AgentCase`. Returns `[]` when the trajectory is filtered out; returns `[AgentCase]` on success.

### AgentSkillExtractor return contract

Pass the new `AgentCase` and the pre-filtered `existing_relevant_skills` (e.g. top-K cosine from the caller's store) plus the associated `supporting_cases` (cases referenced by existing skills). The caller decodes add / update / retire by checking whether `skill.id` is already in `existing_relevant_skills` and whether `skill.confidence < retire_confidence`.

Cases with `quality_score < skip_quality_threshold` (default 0.2) short-circuit to `[]` without calling the LLM.

`AgentSkill.cluster_id` is always `""` on extraction — the caller stamps the cluster identity after persisting.

### AgentProfileExtractor contract

SOUL.md / AGENTS.md are injected into every future system prompt and are self-reinforcing, so the operator is precision-first: it defaults to noop and a candidate signal must pass all four gates (persistence, directedness, evidence strength, novelty) before any patch is proposed. Routing follows one line: SOUL = who the agent is and how it speaks; AGENTS = global rules the agent must obey when acting; everything else (user facts, future intents, one-off task parameters, task solutions) is rejected at the gate. A single LLM call emits each candidate's gate verdicts together with its proposed patch; the prompt sees the full chat view (user + assistant text turns, tool traffic excluded) for context, but config authority stays user-only — every candidate must quote the user verbatim and the quote is re-checked in code against user messages alone. The gates are re-enforced in code.

The returned `AgentProfileUpdate` carries section-level `patches` (never whole-file rewrites — human edits are preserved), `soul_diff` / `agents_diff` unified diffs, and `new_soul_md` / `new_agents_md` with all patches applied. Patches with `is_conflict=True` override an existing rule (the user changed their mind); they are applied like any other patch, and the flag lets the caller route them through a user-confirmation step or surface them in debugging. `signals` are implicit below-gate observations: persist them and pass them back as `pending_signals` on later runs so recurring corrections accumulate toward `min_recurrence`.

Every LLM proposal is re-validated in code: the evidence quote must appear verbatim in the user messages, a `modify` `old_text` must match the file exactly once, an `add` must not duplicate existing content, and a patch that would push a file past `max_file_tokens` is dropped.

### Customising prompts

Each prompt is a static system message of rules (`AGENT_*_SYSTEM_PROMPT`) and a user template of data that the
operator always renders. To replace the rules for the whole process, patch the system constant on the operator module
that re-exports it, at startup:

```python
import everalgo.agent_memory.case as _case
_case.AGENT_CASE_FILTER_SYSTEM_PROMPT = my_custom_filter_rules   # global override
```

There is no per-call prompt override.

## Related distributions

- [`everalgo-boundary`](../everalgo-boundary/) — defines `BoundaryDetector`
- [`everalgo-clustering`](../everalgo-clustering/) — `acluster_by_llm` for skill-cluster assignment
- [`everalgo-rank`](../everalgo-rank/) — `CaseRanker` / `SkillRanker` for read-time ranking


## Optional correctness admission

`AgentCaseExtractor(llm=..., gate=AgentCaseGate(...))` replaces the minimum tool-round, no-tool complexity, and value-LLM filters. With no gate the existing behavior is unchanged. The shared boundary detector, structural checks, resource limits, Case generation, clustering, and retrieval are unchanged.

The gate asks its `judge` one question per conversation. `CaseJudge` is a Protocol: the host implements `apredict(conversation, question) -> float` against whichever probability API it deploys, mapping the provider-neutral `CaseGateQuestion` (instructions plus `true` / `false` criteria) onto that provider's request format.

```python
from everalgo.agent_memory import AgentCaseExtractor, AgentCaseGate, CaseGateConfig, CaseGateQuestion

class MyJudge:
    async def apredict(self, conversation: str, question: CaseGateQuestion) -> float:
        ...  # call the host's probability API; return the probability that the question is true

async def extract_with_gate(memcell, llm):
    gate = AgentCaseGate(judge=MyJudge(), config=CaseGateConfig(success_threshold=0.70, bypass_token_threshold=50_000))
    return await AgentCaseExtractor(llm=llm, gate=gate).aextract_with_reason(memcell)
```

Sizing uses `o200k_base` on the OpenAI-wire JSON of the supplied conversation, including tool arguments and results, after dropping the system head and **before** any clipping or compression. It excludes the fixed judge question. Exactly 50,000 tokens still goes to the judge; strictly more bypasses it. This is a reproducible routing measure, not the selected provider tokenizer's billed input count. The input is the supplied closed MemCell, which may be a segment rather than an entire session. The question evaluates that segment's available evidence; the threshold has not been calibrated on production segments by the earlier whole-trajectory benchmark.

The default probability threshold `0.70` favors retaining more Cases and is an initial policy setting, not a validated precision guarantee for any judge model or input distribution. Calibrate it on labeled target inputs. The routing threshold is 50,000 tokens under `o200k_base`, while the provider bills its own tokenizer plus the judge question. Do not silently truncate evidence to fit a provider limit.

Short input calls the judge once and accepts probability `>=0.70`; rejection returns `SUCCESS_GATE_REJECTED` and does not call the generative extractor. Long input skips the judge and continues extraction. The gate raises `LLMOutputError` when the judge returns anything other than a finite probability in `[0, 1]`. Judge errors (including provider context-limit errors), cancellation, and deadline expiry propagate to the host unchanged; none is converted to probability zero, automatic admission, or a fallback LLM decision.

An admitted Case carries `case_gate` provenance with the policy, method (`decision_api` or `length_bypass`), token count, thresholds, and probability (`None` for a length bypass). Hosts must persist and return it unchanged when reconstructing a Case for Skill extraction. `quality_score` remains the generative extractor's diagnostic score. `should_extract_skill(quality_score, case_gate=...)` centralizes eligibility for hosts that cluster before extracting skills. Existing Cases without admission retain the original quality-based routing. Admitted Cases use the success-experience prompt, while preserving real pitfalls and never asserting that length-based admission or a probability proves success.

The host owns the judge implementation: provider request format, HTTP transport lifetime, endpoint, credentials, deadline, deployment settings, storage, and retries. The algorithm owns the correctness question text, token sizing, routing thresholds, probability validation, and Case/Skill policy. No source manifest, benchmark labels, or shared-boundary changes are required.
