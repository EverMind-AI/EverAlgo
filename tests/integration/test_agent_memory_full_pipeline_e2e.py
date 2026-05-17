"""End-to-end pipeline test: agent trajectory → boundary → AgentCase → clustering → AgentSkill.

Algorithm-correctness gate for the full agent-memory pipeline. Captures every rendered LLM prompt
across 5 stages and asserts:

1. AgentBoundaryDetector preserves tool calls in MemCell but filters them from LLM prompt
2. AgentCaseExtractor consumes the FULL trajectory (tool calls included in case prompts)
3. AgentSkillExtractor lifts the case into a cluster-level skill (success-branch add path)
4. Skill prompt sees the case's task_intent / approach / quality_score, including case_prior from
   the cluster's supporting_cases fan-in

Pipeline stages:
1. AgentBoundaryDetector.adetect(items)    → MemCell          (1 LLM call: boundary)
2. AgentCaseExtractor.aextract(mc)         → [AgentCase]      (2 LLM calls: case_filter + case_compress)
   [cluster_by_llm step — no LLM call; both invocations fast-path past LLM]
3. AgentSkillExtractor.aextract(case, ...) → [AgentSkill]     (2 LLM calls: skill_extract + skill_maturity)

Total: 5 LLM calls.  cluster_by_llm does no LLM call in this test: first invocation fast-paths
because state is empty; second fast-paths because top-1 sim ~0.9999 >= llm_skip_threshold=0.85.
skill_maturity fires because skip_maturity_scoring=False is passed explicitly —
the default is True (skip) and would reduce the call count to 4.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from everalgo.agent_memory import AgentBoundaryDetector, AgentCaseExtractor, AgentSkillExtractor
from everalgo.clustering import Cluster, cluster_by_llm
from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.llm.types import ChatResponse
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import (
    AgentCase,
    AgentSkill,
    ChatMessage,
    ToolCall,
    ToolCallFunction,
    ToolCallRequest,
    ToolCallResult,
)

# ---------------------------------------------------------------------------
# Stage ordering and fake responses
# ---------------------------------------------------------------------------

# 5 LLM calls in deterministic order:
#   boundary → case_filter → case_compress → skill_extract → skill_maturity
_STAGE_ORDER = ("boundary", "case_filter", "case_compress", "skill_extract", "skill_maturity")

_BOUNDARY_JSON = '{"reasoning": "single coherent task", "boundaries": [], "should_wait": false}'

# case_filter: only "worth_extracting" is consumed; extra fields are ignored.
_CASE_FILTER_JSON = '{"worth_extracting": true, "reason": "agent resolved a production issue via tool use"}'

# case_compress: quality_score >= 0.5 (failure_quality_threshold) to force success-branch in skill.
_CASE_COMPRESS_JSON = (
    '{"task_intent": "Diagnose high API latency in production by correlating recent deploys",'
    ' "approach": "Called list_recent_deploys to surface the suspect deploy at 20:30 UTC by bob",'
    ' "quality_score": 0.9,'
    ' "key_insight": "Latency regressions often correlate with a recent deploy — check deploy timing first"}'
)

# skill_extract: success-branch add path — single "add" operation.
# name + description (non-empty) + content meeting _is_skill_content_sufficient (>=5 lines, >=50 chars)
# confidence must be parseable as float (clamped to [0,1]).
_SKILL_EXTRACT_JSON = (
    '{"operations": [{"action": "add", "data": {'
    '"name": "Production Latency Triage via Deploy Correlation",'
    '"description": "Identify latency regressions by correlating incident timing with recent deploys.",'
    '"content": "## Steps\\n'
    "1. Capture the latency onset timestamp from monitoring alerts.\\n"
    "   - How: Compare p99 latency spike time with the deploy feed.\\n"
    "2. Query recent deploys in the affected service window.\\n"
    "   - How: Use deploy-listing tooling filtered to the incident time window.\\n"
    "3. Correlate deploy timing with the latency spike.\\n"
    "   - Decision: If a deploy lands within 30 min of the spike → prime suspect.\\n"
    "4. Confirm by examining deploy diff for high-risk changes.\\n"
    "   - How: Look for connection-pool, caching, or timeout-setting changes.\\n"
    '5. Escalate or roll back based on findings.",'
    '"confidence": 0.5}}'
    '], "update_note": "New case — no existing skills to overlap against."}'
)

# skill_maturity: 4 dimensions each 1-5; raw_total / 20 → maturity_score.
# Using completeness=4, executability=4, evidence=3, clarity=4 → raw=15 → 0.75
_SKILL_MATURITY_JSON = (
    '{"completeness": 4, "executability": 4, "evidence": 3, "clarity": 4,'
    ' "reason": "Good procedural steps with concrete tool use; evidence from one case so far."}'
)

_STAGE_RESPONSE: dict[str, str] = {
    "boundary": _BOUNDARY_JSON,
    "case_filter": _CASE_FILTER_JSON,
    "case_compress": _CASE_COMPRESS_JSON,
    "skill_extract": _SKILL_EXTRACT_JSON,
    "skill_maturity": _SKILL_MATURITY_JSON,
}


# ---------------------------------------------------------------------------
# Fake LLM client — captures every prompt keyed by stage name
# ---------------------------------------------------------------------------


def _make_capturing_client() -> tuple[FakeLLMClient, dict[str, str]]:
    """Return a fake client whose handler captures each call's prompt by pipeline-stage name."""
    captured: dict[str, str] = {}
    call_index = 0

    def handler(messages: list[LLMChatMessage], **_kwargs: Any) -> ChatResponse:
        nonlocal call_index
        stage = _STAGE_ORDER[call_index]
        captured[stage] = messages[0].content
        call_index += 1
        return ChatResponse(content=_STAGE_RESPONSE[stage], model="fake")

    return FakeLLMClient(handler=handler), captured


# ---------------------------------------------------------------------------
# Test trajectory — one realistic agent task with a single tool-call round
# ---------------------------------------------------------------------------


def _agent_trajectory() -> list[ChatMessage | ToolCallRequest | ToolCallResult]:
    """User asks → assistant calls list_recent_deploys → tool returns → assistant answers."""
    return [
        ChatMessage(
            kind="text",
            id="m1",
            role="user",
            content="What's causing high API latency in prod?",
            timestamp=1_700_000_000_000,
            sender_id="u_alice",
            sender_name="Alice",
        ),
        ToolCallRequest(
            tool_calls=[
                ToolCall(
                    id="call_abc123",
                    function=ToolCallFunction(
                        name="list_recent_deploys",
                        arguments='{"hours": 24, "service": "api-gateway"}',
                    ),
                ),
            ],
            timestamp=1_700_000_001_000,
            content="Let me check recent deploys.",
            sender_id="assistant",
            sender_name="Claude",
        ),
        ToolCallResult(
            tool_call_id="call_abc123",
            content='{"deploys": [{"sha": "abc123", "time": "2023-11-14T20:30:00Z", "author": "bob"}]}',
            timestamp=1_700_000_002_000,
        ),
        ChatMessage(
            kind="text",
            id="m2",
            role="assistant",
            content="The latency correlates with the deploy at 20:30 UTC by bob.",
            timestamp=1_700_000_003_000,
            sender_id="assistant",
            sender_name="Claude",
        ),
    ]


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------


async def test_agent_memory_full_pipeline_e2e() -> None:
    """Run boundary → AgentCase → AgentSkill and assert all 5 prompts are wired correctly."""
    fake, captured = _make_capturing_client()

    # --- 1. Boundary -------------------------------------------------------
    boundary_output = await AgentBoundaryDetector(llm=fake).adetect(
        _agent_trajectory(),  # type: ignore[arg-type]
        is_final=True,
    )
    assert boundary_output.tail == [], "expected no tail in is_final=True mode"
    assert len(boundary_output.cells) == 1, "expected exactly one MemCell from a single-topic trajectory"
    mc = boundary_output.cells[0]

    # MemCell preserves ALL 4 items (user_chat + tool_call + tool_result + assistant_chat)
    assert len(mc.items) == 4, "MemCell must preserve the full trajectory, including tool calls"

    boundary_prompt = captured["boundary"]
    assert "Alice" in boundary_prompt, "boundary prompt must include sender_name from ChatMessage"
    # Tool items must NOT reach the boundary LLM — filter→detect→remap contract.
    assert "list_recent_deploys" not in boundary_prompt, "boundary LLM must not see tool call names"
    assert "call_abc123" not in boundary_prompt, "boundary LLM must not see tool call ids"

    # --- 2. Case extraction ------------------------------------------------
    cases = await AgentCaseExtractor(llm=fake).aextract(mc)
    assert len(cases) == 1, "expected exactly one AgentCase for a successful trajectory"
    case = cases[0]
    assert isinstance(case, AgentCase), "aextract must return AgentCase instances"
    assert case.task_intent, "AgentCase.task_intent must be non-empty"
    assert case.quality_score >= 0.5, "quality_score must reach success-branch threshold for skill extraction"

    # case_filter + case_compress both see the full trajectory (tool calls included)
    case_filter_prompt = captured["case_filter"]
    assert "list_recent_deploys" in case_filter_prompt, "case_filter prompt must include tool call name"
    assert "call_abc123" in case_filter_prompt, "case_filter prompt must include tool call id"

    case_compress_prompt = captured["case_compress"]
    assert "list_recent_deploys" in case_compress_prompt, "case_compress prompt must include tool call name"
    assert "abc123" in case_compress_prompt, "case_compress prompt must include tool result content"

    # --- cluster_by_llm (no LLM call — both invocations fast-path) ------------
    # A prior case constructed directly (no extractor call) to simulate a cluster with history.
    case_prior = AgentCase(
        id="case_prior",
        timestamp=1_699_900_000_000,
        task_intent="Investigate slow database queries by inspecting the recent migration logs",
        approach="Used list_migrations to find a schema change introduced 2 days ago",
        quality_score=0.85,
        key_insight="Migration-induced latency often appears 1-2 days after rollout",
    )

    clusters: list[Cluster] = []
    vec_prior = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    vec_current = np.array([0.99, 0.01, 0.0], dtype=np.float32)  # cosine ~0.9999 > 0.85 skip threshold

    # First call: clusters is empty → fast-paths to new cluster without LLM.
    new_c_prior = Cluster(
        id="cluster_000",
        centroid=vec_prior,
        last_ts=case_prior.timestamp,
        preview=[case_prior.task_intent],
        members=[case_prior.id],
    )
    result_prior = await cluster_by_llm(new_c_prior, clusters, llm=fake)
    assert result_prior is None
    clusters.append(new_c_prior)

    # Second call: top-1 sim ~0.9999 >= llm_skip_threshold=0.85 → fast-paths without LLM.
    new_c_current = Cluster(centroid=vec_current, last_ts=case.timestamp, preview=[case.task_intent], members=[case.id])
    result_current = await cluster_by_llm(new_c_current, clusters, llm=fake)
    assert result_current is not None, "expected merged Cluster, got None"
    assert result_current.id == "cluster_000", f"expected id passthrough, got {result_current.id!r}"
    clusters[0] = result_current

    cid_current = "cluster_000"
    assert len(clusters) == 1, f"expected 1 cluster minted, got {len(clusters)}"
    assert clusters[0].members == [case_prior.id, case.id], (
        f"expected members=[case_prior.id, case.id], got {clusters[0].members!r}"
    )

    # --- 3. Skill extraction -----------------------------------------------
    # Pass skip_maturity_scoring=False to force the maturity LLM call (5th stage).
    # A seed skill whose source_case_ids references case_prior lets _format_existing_skills render
    # case_prior as a supporting_case — the only code path that brings supporting_cases into the prompt.
    # (When existing_relevant_skills is empty, _format_existing_skills short-circuits to a placeholder
    # string and supporting_cases is never consulted, so a non-empty existing list is required here.)
    seed_skill = AgentSkill(
        id="skill_seed",
        cluster_id=cid_current,
        name="Diagnose Latency via Inspect Logs",
        description="Prior skill seeded from case_prior for fan-in test.",
        content="## Steps\n1. Check logs.\n",
        confidence=0.5,
        source_case_ids=["case_prior"],
    )
    raw_skills = await AgentSkillExtractor(llm=fake).aextract(
        case,
        existing_relevant_skills=[seed_skill],
        supporting_cases=[case_prior],
        skip_maturity_scoring=False,
    )
    # Caller stamps cluster_id after extraction
    skills = [s.model_copy(update={"cluster_id": cid_current}) for s in raw_skills]
    assert len(skills) == 1, "add-path must emit exactly one new AgentSkill"
    skill = skills[0]
    assert isinstance(skill, AgentSkill), "aextract must return AgentSkill instances"
    assert skill.cluster_id == "cluster_000", "skill must carry the cluster-assigned cluster_id"
    assert skill.name, "AgentSkill.name must be non-empty per the skill_extract JSON"
    assert skill.confidence > 0.0, "AgentSkill.confidence must be positive"
    # maturity_score: raw_total=15 / 20 = 0.75 (from _SKILL_MATURITY_JSON)
    assert abs(skill.maturity_score - 0.75) < 1e-6, f"expected maturity_score=0.75, got {skill.maturity_score}"

    # Skill prompt must embed case context so the LLM can reason about the task
    skill_extract_prompt = captured["skill_extract"]
    assert case.task_intent in skill_extract_prompt, "skill_extract prompt must contain case.task_intent"
    # case_prior.task_intent appears as a supporting_case under seed_skill — proves supporting_cases fan-in.
    assert case_prior.task_intent in skill_extract_prompt, (
        "skill_extract prompt must contain case_prior.task_intent to confirm supporting_cases fan-in"
    )

    # --- 5 stages, 5 captures ----------------------------------------------
    assert set(captured.keys()) == set(_STAGE_ORDER), (
        "every pipeline stage must capture exactly one prompt; "
        f"expected {set(_STAGE_ORDER)}, got {set(captured.keys())}"
    )
