"""Admission boundaries, direct extraction, downstream routing, and the judge contract."""

from __future__ import annotations

import json
import math

import pytest

from everalgo.agent_memory import (
    AgentCaseExtractor,
    AgentCaseGate,
    AgentSkillExtractor,
    CaseGateConfig,
    CaseGateDecision,
    CaseGateQuestion,
    CaseSkipReason,
    should_extract_skill,
)
from everalgo.agent_memory.case import _dump_messages
from everalgo.agent_memory.prompts.case_gate import (
    CASE_GATE_FALSE_CRITERION,
    CASE_GATE_INSTRUCTIONS,
    CASE_GATE_TRUE_CRITERION,
)
from everalgo.agent_memory.prompts.skill_success import AGENT_SKILL_ADMITTED_EXTRACT_SYSTEM_PROMPT
from everalgo.llm import LLMOutputError
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import (
    ChatMessage,
    ConversationItem,
    MemCell,
    ToolCall,
    ToolCallFunction,
    ToolCallRequest,
    ToolCallResult,
)
from everalgo.utils import count_tokens


class FakeJudge:
    def __init__(self, probability: object = 0.70, error: Exception | None = None) -> None:
        self.probability = probability
        self.error = error
        self.calls: list[str] = []
        self.questions: list[CaseGateQuestion] = []

    async def apredict(self, conversation: str, question: CaseGateQuestion) -> float:
        self.calls.append(conversation)
        self.questions.append(question)
        if self.error is not None:
            raise self.error
        return self.probability  # type: ignore[return-value]  # malformed-reply tests feed non-floats


def cell(*, tool_output: str | None = None) -> MemCell:
    items: list[ConversationItem] = [
        ChatMessage(id="u", role="user", content="Fix the range bound.", timestamp=1000, sender_id="user")
    ]
    if tool_output is not None:
        items += [
            ToolCallRequest(
                tool_calls=[ToolCall(id="t", function=ToolCallFunction(name="test", arguments="{}"))],
                timestamp=1001,
                sender_id="agent",
            ),
            ToolCallResult(tool_call_id="t", content=tool_output, timestamp=1002),
        ]
    items.append(
        ChatMessage(id="a", role="assistant", content="Fixed and verified.", timestamp=1003, sender_id="agent")
    )
    return MemCell(items=items, timestamp=1003)


def compress_llm(quality: float = 0.1) -> FakeLLMClient:
    return FakeLLMClient(
        responses=[
            json.dumps(
                {
                    "task_intent": "Fix the bound",
                    "approach": "Check the range and run the regression.",
                    "key_insight": "The upper bound is exclusive.",
                    "quality_score": quality,
                }
            )
        ]
    )


@pytest.mark.parametrize(
    "offset,method,calls", [(-1, "length_bypass", 0), (0, "decision_api", 1), (1, "decision_api", 1)]
)
async def test_token_boundary_is_strictly_greater(offset: int, method: str, calls: int) -> None:
    conversation = _dump_messages(cell().items)
    judge = FakeJudge()
    gate = AgentCaseGate(judge=judge, config=CaseGateConfig(bypass_token_threshold=count_tokens(conversation) + offset))
    result = await gate.aevaluate(conversation)
    assert result.method == method
    assert result.accepted
    assert len(judge.calls) == calls
    assert (result.success_probability is None) == (method == "length_bypass")


@pytest.mark.parametrize(
    "tokens,method,calls",
    [(49_999, "decision_api", 1), (50_000, "decision_api", 1), (50_001, "length_bypass", 0)],
)
async def test_default_50k_boundary_uses_original_input(tokens: int, method: str, calls: int) -> None:
    conversation = "evidence " * (tokens - 2)
    assert count_tokens(conversation) == tokens
    judge = FakeJudge()
    result = await AgentCaseGate(judge=judge).aevaluate(conversation)
    assert result.bypass_token_threshold == 50_000
    assert result.input_tokens == tokens
    assert result.method == method
    assert result.accepted
    assert judge.calls == [conversation] * calls


@pytest.mark.parametrize(
    "probability,accepted", [(0.699, False), (0.70, True), (0.80, True), (0.849, True), (0.85, True), (0.9, True)]
)
async def test_probability_gate_replaces_old_round_and_value_filters(probability: float, *, accepted: bool) -> None:
    judge = FakeJudge(probability)
    llm = compress_llm()
    result = await AgentCaseExtractor(llm=llm, gate=AgentCaseGate(judge=judge)).aextract_with_reason(cell())
    assert len(judge.calls) == 1
    assert bool(result.cases) == accepted
    assert llm.call_count == int(accepted)  # compress only; no value LLM, even on a one-turn no-tool cell
    if accepted:
        assert result.cases[0].quality_score == 0.1
        assert result.cases[0].model_dump()["case_gate"] == result.detail["case_gate"]
    else:
        assert result.reason == CaseSkipReason.SUCCESS_GATE_REJECTED


async def test_sizing_and_judging_use_untrimmed_tool_evidence() -> None:
    source = cell(tool_output="unique evidence " * 4000)
    judge = FakeJudge()
    await AgentCaseExtractor(llm=compress_llm(), gate=AgentCaseGate(judge=judge)).aextract(source)
    assert json.loads(judge.calls[0])[2]["content"] == "unique evidence " * 4000


async def test_length_bypass_uses_original_size_before_tool_trimming() -> None:
    source = cell(tool_output="evidence " * 52_000)
    judge = FakeJudge(error=AssertionError("long inputs must not call the judge"))
    result = await AgentCaseExtractor(llm=compress_llm(), gate=AgentCaseGate(judge=judge)).aextract_with_reason(source)
    assert result.cases
    assert result.detail["case_gate"]["input_tokens"] > 50_000
    assert result.detail["case_gate"]["method"] == "length_bypass"
    assert not judge.calls


async def test_incomplete_input_is_still_held_without_a_judge_request() -> None:
    source = cell()
    source.items = source.items[:-1]
    judge = FakeJudge()
    llm = FakeLLMClient(responses=[])
    result = await AgentCaseExtractor(llm=llm, gate=AgentCaseGate(judge=judge)).aextract_with_reason(source)
    assert result.reason == CaseSkipReason.NO_ASSISTANT_MESSAGE
    assert not judge.calls
    assert llm.call_count == 0


async def test_judge_timeout_is_not_a_rejection_or_automatic_admission() -> None:
    llm = FakeLLMClient(responses=[])
    with pytest.raises(TimeoutError):
        await AgentCaseExtractor(llm=llm, gate=AgentCaseGate(judge=FakeJudge(error=TimeoutError()))).aextract(cell())
    assert llm.call_count == 0


@pytest.mark.parametrize("method", ["decision_api", "length_bypass"])
async def test_admitted_case_preserves_quality_and_routes_success_skill(method: str) -> None:
    cfg = CaseGateConfig(bypass_token_threshold=1 if method == "length_bypass" else 50_000)
    case = (
        await AgentCaseExtractor(llm=compress_llm(), gate=AgentCaseGate(judge=FakeJudge(), config=cfg)).aextract(cell())
    )[0]
    # Round-trip through persistence must preserve the decision for downstream workers.
    case = type(case).model_validate(case.model_dump(mode="json"))
    assert should_extract_skill(case.quality_score, case_gate=case.model_dump()["case_gate"])
    skill_llm = FakeLLMClient(responses=['{"operations":[{"action":"none"}],"update_note":"duplicate"}'])
    await AgentSkillExtractor(llm=skill_llm).aextract(case, existing_relevant_skills=[], supporting_cases=[])
    assert skill_llm.calls[0].messages[0].content == AGENT_SKILL_ADMITTED_EXTRACT_SYSTEM_PROMPT
    assert case.quality_score == 0.1


async def test_legacy_case_and_skill_admission_remain_unchanged() -> None:
    llm = FakeLLMClient(responses=[])
    assert await AgentCaseExtractor(llm=llm).aextract(cell()) == []
    assert not should_extract_skill(0.1)
    assert should_extract_skill(0.2)
    assert llm.call_count == 0


def test_invalid_or_rejected_provenance_cannot_admit_a_skill() -> None:
    rejected = CaseGateDecision(
        method="decision_api",
        input_tokens=100,
        bypass_token_threshold=50_000,
        success_threshold=0.85,
        success_probability=0.1,
    )
    with pytest.raises(ValueError, match="pass admission"):
        should_extract_skill(0.9, case_gate=rejected.model_dump())
    with pytest.raises(ValueError):
        CaseGateConfig(success_threshold=float("nan"))


async def test_judge_receives_the_bundled_question() -> None:
    judge = FakeJudge()
    await AgentCaseGate(judge=judge).aevaluate("evidence")
    assert judge.questions == [
        CaseGateQuestion(
            instructions=CASE_GATE_INSTRUCTIONS,
            true_criterion=CASE_GATE_TRUE_CRITERION,
            false_criterion=CASE_GATE_FALSE_CRITERION,
        )
    ]


@pytest.mark.parametrize("probability", [True, None, "0.9", -0.1, 1.1, math.nan, math.inf])
async def test_malformed_judge_probabilities_raise(probability: object) -> None:
    llm = FakeLLMClient(responses=[])
    with pytest.raises(LLMOutputError):
        await AgentCaseExtractor(llm=llm, gate=AgentCaseGate(judge=FakeJudge(probability))).aextract(cell())
    assert llm.call_count == 0


async def test_judge_errors_propagate_without_admission() -> None:
    llm = FakeLLMClient(responses=[])
    with pytest.raises(RuntimeError, match="context limit"):
        await AgentCaseExtractor(
            llm=llm, gate=AgentCaseGate(judge=FakeJudge(error=RuntimeError("context limit")))
        ).aextract(cell())
    assert llm.call_count == 0
