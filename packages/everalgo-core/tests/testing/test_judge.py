"""Tests for testing.judge — LLM-as-judge with N-run majority vote."""

from __future__ import annotations

import json

import pytest

from everalgo.llm.types import ChatResponse, Usage
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.testing.judge import JudgeResult, allm_judge

_TEST_JUDGE_PROMPT = (
    "Question: {question}\nGold: {gold_answer}\nResponse: {response}\nOutput JSON with label CORRECT or WRONG."
)


def _judge_chat_response(
    label: str, reasoning: str = "", prompt_tokens: int = 10, completion_tokens: int = 5
) -> ChatResponse:
    return ChatResponse(
        content=json.dumps({"label": label, "reasoning": reasoning}),
        model="fake",
        usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


async def test_judge_single_run_correct() -> None:
    fake = FakeLLMClient(responses=[_judge_chat_response("CORRECT", "exact match")])
    result = await allm_judge(
        question="Q",
        golden_answer="A",
        generated_answer="A",
        judge_prompt=_TEST_JUDGE_PROMPT,
        llm=fake,
        num_runs=1,
    )
    assert isinstance(result, JudgeResult)
    assert result.is_correct is True
    assert result.runs == [True]
    assert "exact match" in result.reasoning[0]
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 5


async def test_judge_majority_vote_3_runs_correct() -> None:
    fake = FakeLLMClient(
        responses=[
            _judge_chat_response("CORRECT"),
            _judge_chat_response("WRONG"),
            _judge_chat_response("CORRECT"),
        ]
    )
    result = await allm_judge(
        question="Q",
        golden_answer="A",
        generated_answer="A",
        judge_prompt=_TEST_JUDGE_PROMPT,
        llm=fake,
        num_runs=3,
    )
    assert result.is_correct is True  # 2/3 vote
    assert len(result.runs) == 3
    assert sum(result.runs) == 2
    assert len(result.reasoning) == 3
    assert result.prompt_tokens == 30
    assert result.completion_tokens == 15


async def test_judge_majority_vote_3_runs_wrong() -> None:
    fake = FakeLLMClient(
        responses=[
            _judge_chat_response("WRONG"),
            _judge_chat_response("WRONG"),
            _judge_chat_response("CORRECT"),
        ]
    )
    result = await allm_judge(
        question="Q",
        golden_answer="A",
        generated_answer="B",
        judge_prompt=_TEST_JUDGE_PROMPT,
        llm=fake,
        num_runs=3,
    )
    assert result.is_correct is False  # 1/3 vote — not strict majority


async def test_judge_raises_on_legacy_is_correct_field() -> None:
    """Legacy ``{is_correct: bool}`` schema is no longer accepted — fail-loud.

    evercore judge prompt only emits ``{"label": ...}``; the legacy fallback was
    dead code masking real judge-format drift. After retries are exhausted the
    parse failure surfaces as ``ValueError``.
    """
    fake = FakeLLMClient(
        responses=[
            ChatResponse(content=json.dumps({"is_correct": True, "reasoning": "ok"}), model="fake"),
        ]
    )
    with pytest.raises(ValueError, match="missing 'label'"):
        await allm_judge(
            question="Q",
            golden_answer="A",
            generated_answer="A",
            judge_prompt=_TEST_JUDGE_PROMPT,
            llm=fake,
            num_runs=1,
            max_retries=1,
        )


async def test_judge_raises_on_plain_prose_without_json() -> None:
    """No JSON in LLM output → fail-loud (keyword-scan fallback was banned)."""
    fake = FakeLLMClient(
        responses=[ChatResponse(content="The response is correct.", model="fake")],
    )
    with pytest.raises(ValueError, match="No JSON object found"):
        await allm_judge(
            question="Q",
            golden_answer="A",
            generated_answer="A",
            judge_prompt=_TEST_JUDGE_PROMPT,
            llm=fake,
            num_runs=1,
            max_retries=1,
        )


async def test_judge_raises_on_unknown_label() -> None:
    """Label outside ``{CORRECT, WRONG}`` is rejected, not silently mapped."""
    fake = FakeLLMClient(
        responses=[ChatResponse(content=json.dumps({"label": "MAYBE"}), model="fake")],
    )
    with pytest.raises(ValueError, match="Unknown judge label"):
        await allm_judge(
            question="Q",
            golden_answer="A",
            generated_answer="A",
            judge_prompt=_TEST_JUDGE_PROMPT,
            llm=fake,
            num_runs=1,
            max_retries=1,
        )


async def test_judge_retry_recovers_from_transient_parse_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bad parse followed by a valid response succeeds; retry budget covers it."""

    # Collapse exponential backoff so the test does not actually wait.
    async def _no_sleep(_seconds: float) -> None:
        return

    monkeypatch.setattr("everalgo.testing.judge.asyncio.sleep", _no_sleep)

    fake = FakeLLMClient(
        responses=[
            ChatResponse(content="not json at all", model="fake"),
            _judge_chat_response("CORRECT", "recovered"),
        ]
    )
    result = await allm_judge(
        question="Q",
        golden_answer="A",
        generated_answer="A",
        judge_prompt=_TEST_JUDGE_PROMPT,
        llm=fake,
        num_runs=1,
        max_retries=3,
    )
    assert result.is_correct is True
    assert "recovered" in result.reasoning[0]


async def test_judge_system_prompt_prepended_when_provided() -> None:
    """When judge_system_prompt is given, the chat messages start with role=system."""
    fake = FakeLLMClient(
        handler=lambda messages, **_kwargs: _judge_chat_response("CORRECT"),  # type: ignore[misc]
    )
    await allm_judge(
        question="Q",
        golden_answer="A",
        generated_answer="A",
        judge_prompt=_TEST_JUDGE_PROMPT,
        judge_system_prompt="You are a strict judge.",
        llm=fake,
        num_runs=1,
    )
    assert fake.calls[0].messages[0].role == "system"
    assert fake.calls[0].messages[0].content == "You are a strict judge."
    assert fake.calls[0].messages[1].role == "user"


async def test_judge_no_system_prompt_user_only() -> None:
    fake = FakeLLMClient(
        handler=lambda messages, **_kwargs: _judge_chat_response("CORRECT"),  # type: ignore[misc]
    )
    await allm_judge(
        question="Q",
        golden_answer="A",
        generated_answer="A",
        judge_prompt=_TEST_JUDGE_PROMPT,
        llm=fake,
        num_runs=1,
    )
    assert len(fake.calls[0].messages) == 1
    assert fake.calls[0].messages[0].role == "user"


async def test_judge_strict_majority_ties_break_to_wrong() -> None:
    """For num_runs=4, a 2/4 tie should NOT pass (strict majority requires > 2)."""
    fake = FakeLLMClient(
        responses=[
            _judge_chat_response("CORRECT"),
            _judge_chat_response("CORRECT"),
            _judge_chat_response("WRONG"),
            _judge_chat_response("WRONG"),
        ]
    )
    result = await allm_judge(
        question="Q",
        golden_answer="A",
        generated_answer="A",
        judge_prompt=_TEST_JUDGE_PROMPT,
        llm=fake,
        num_runs=4,
    )
    assert result.is_correct is False
    assert result.runs == [True, True, False, False]


def test_judge_result_is_frozen() -> None:
    from pydantic import ValidationError

    r = JudgeResult(is_correct=True)
    with pytest.raises(ValidationError):
        r.is_correct = False  # type: ignore[misc]
