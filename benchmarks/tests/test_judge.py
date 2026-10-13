"""Tests for benchmarks.common.judge — LLM-as-judge with N-run majority vote."""

from __future__ import annotations

import json

import pytest

from benchmarks.common.judge import JudgeResult, _JudgeVerdict, allm_judge
from benchmarks.common.metrics import estimate_tokens
from everalgo.llm import LLMOutputError
from everalgo.testing import FakeLLMClient

_TEST_JUDGE_PROMPT = (
    "Question: {{question}}\nGold: {{gold_answer}}\nResponse: {{response}}\nOutput JSON with label CORRECT or WRONG."
)


def _verdict(label: str, reasoning: str = "") -> str:
    return json.dumps({"reasoning": reasoning, "label": label})


async def _judge_once(fake: FakeLLMClient, *, num_runs: int = 1, system: str | None = None) -> JudgeResult:
    return await allm_judge(
        question="Q",
        golden_answer="A",
        generated_answer="A",
        judge_prompt=_TEST_JUDGE_PROMPT,
        judge_system_prompt=system,
        llm=fake,
        num_runs=num_runs,
    )


async def test_judge_single_run_correct() -> None:
    reply = _verdict("CORRECT", "exact match")
    fake = FakeLLMClient(responses=[reply])
    result = await _judge_once(fake)
    assert isinstance(result, JudgeResult)
    assert result.is_correct is True
    assert result.runs == [True]
    assert result.reasoning == ["exact match"]
    sent = fake.calls[0].messages
    assert result.prompt_tokens == sum(estimate_tokens(str(m.content)) for m in sent) > 0
    assert result.completion_tokens == estimate_tokens(reply) > 0


async def test_judge_requests_the_verdict_model() -> None:
    fake = FakeLLMClient(responses=[_verdict("WRONG")])
    await _judge_once(fake)
    assert fake.calls[0].response_format is _JudgeVerdict


async def test_judge_majority_vote_3_runs_correct() -> None:
    fake = FakeLLMClient(responses=[_verdict("CORRECT"), _verdict("WRONG"), _verdict("CORRECT")])
    result = await _judge_once(fake, num_runs=3)
    assert result.is_correct is True
    assert len(result.runs) == 3
    assert sum(result.runs) == 2
    assert len(result.reasoning) == 3
    single = await _judge_once(FakeLLMClient(responses=[_verdict("CORRECT")]))
    assert result.prompt_tokens == 3 * single.prompt_tokens


async def test_judge_majority_vote_3_runs_wrong() -> None:
    fake = FakeLLMClient(responses=[_verdict("WRONG"), _verdict("WRONG"), _verdict("CORRECT")])
    result = await _judge_once(fake, num_runs=3)
    assert result.is_correct is False


@pytest.mark.parametrize(
    "reply",
    [
        pytest.param("The response is correct.", id="plain-prose"),
        pytest.param(json.dumps({"is_correct": True, "reasoning": "ok"}), id="legacy-is-correct-field"),
        pytest.param(json.dumps({"reasoning": "ok", "label": "MAYBE"}), id="unknown-label"),
        pytest.param(json.dumps({"reasoning": "ok", "label": "correct"}), id="lowercase-label"),
        pytest.param(json.dumps({"label": "CORRECT"}), id="missing-reasoning"),
        pytest.param(json.dumps({"reasoning": "ok", "label": "CORRECT", "score": 1}), id="extra-field"),
    ],
)
async def test_judge_rejects_a_reply_that_is_not_a_verdict(reply: str) -> None:
    """Without retry, an invalid reply propagates immediately as ``LLMOutputError``."""
    fake = FakeLLMClient(responses=[reply])
    with pytest.raises(LLMOutputError, match="judge reply is not a valid _JudgeVerdict"):
        await _judge_once(fake)


async def test_judge_system_prompt_prepended_when_provided() -> None:
    """When judge_system_prompt is given, the chat messages start with role=system."""
    fake = FakeLLMClient(handler=lambda messages, **_kwargs: _verdict("CORRECT"))
    await _judge_once(fake, system="You are a strict judge.")
    assert fake.calls[0].messages[0].role == "system"
    assert fake.calls[0].messages[0].content == "You are a strict judge."
    assert fake.calls[0].messages[1].role == "user"


async def test_judge_no_system_prompt_user_only() -> None:
    fake = FakeLLMClient(handler=lambda messages, **_kwargs: _verdict("CORRECT"))
    await _judge_once(fake)
    assert len(fake.calls[0].messages) == 1
    assert fake.calls[0].messages[0].role == "user"


async def test_judge_strict_majority_ties_break_to_wrong() -> None:
    """For num_runs=4, a 2/4 tie should NOT pass (strict majority requires > 2)."""
    fake = FakeLLMClient(
        responses=[_verdict("CORRECT"), _verdict("CORRECT"), _verdict("WRONG"), _verdict("WRONG")],
    )
    result = await _judge_once(fake, num_runs=4)
    assert result.is_correct is False
    assert result.runs == [True, True, False, False]


def test_judge_result_is_frozen() -> None:
    from pydantic import ValidationError

    r = JudgeResult(is_correct=True)
    with pytest.raises(ValidationError):
        r.is_correct = False  # type: ignore[misc]
