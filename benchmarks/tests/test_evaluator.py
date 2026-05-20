"""Tests for LLM judge logic."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from benchmarks.common.evaluator import _parse_judge_output, judge_qa


def test_parse_judge_output_label_correct() -> None:
    is_correct, _ = _parse_judge_output('{"label": "CORRECT"}')
    assert is_correct is True


def test_parse_judge_output_label_wrong() -> None:
    is_correct, _ = _parse_judge_output('{"label": "WRONG"}')
    assert is_correct is False


def test_parse_judge_output_label_case_insensitive() -> None:
    assert _parse_judge_output('{"label": "correct"}')[0] is True
    assert _parse_judge_output('{"label": "Correct"}')[0] is True


def test_parse_judge_output_label_with_reasoning() -> None:
    is_correct, reasoning = _parse_judge_output('{"label": "CORRECT", "reasoning": "matches gold"}')
    assert is_correct is True
    assert reasoning == "matches gold"


# Legacy path: some judges still return is_correct boolean
def test_parse_judge_output_with_clean_json() -> None:
    is_correct, reasoning = _parse_judge_output('{"is_correct": true, "reasoning": "matches gold"}')
    assert is_correct is True
    assert reasoning == "matches gold"


def test_parse_judge_output_with_extra_text_around_json() -> None:
    raw = 'Looking at this: {"is_correct": false, "reasoning": "different"} additional notes'
    is_correct, _ = _parse_judge_output(raw)
    assert is_correct is False


def test_parse_judge_output_falls_back_to_keyword_scan() -> None:
    assert _parse_judge_output("The answer is correct.")[0] is True
    assert _parse_judge_output("The answer is incorrect.")[0] is False
    assert _parse_judge_output("Not correct at all.")[0] is False


def test_parse_judge_output_invalid_json_falls_back() -> None:
    # Malformed JSON should fall through to keyword scan
    is_correct, _ = _parse_judge_output("{is_correct: yes} which is correct.")
    assert is_correct is True


@pytest.mark.asyncio
async def test_judge_qa_majority_vote_pass() -> None:
    """2 pass + 1 fail → majority pass."""
    llm = MagicMock()
    bodies = ['{"is_correct": true}', '{"is_correct": false}', '{"is_correct": true}']
    call_idx = 0

    async def fake_chat(messages: list[dict[str, Any]], *, model: str | None = None, **kwargs: Any) -> MagicMock:
        nonlocal call_idx
        body = bodies[call_idx]
        call_idx += 1
        return MagicMock(content=body, prompt_tokens=10, completion_tokens=5)

    llm.chat = fake_chat
    result = await judge_qa(
        question="q",
        golden_answer="g",
        generated_answer="a",
        judge_prompt="{question}/{gold_answer}/{response}",
        llm=llm,
        num_runs=3,
    )
    assert result.is_correct is True
    assert result.runs == [True, False, True]
    # 3 runs x (10 prompt + 5 completion)
    assert result.prompt_tokens == 30
    assert result.completion_tokens == 15


@pytest.mark.asyncio
async def test_judge_qa_majority_vote_fail() -> None:
    """1 pass + 2 fail → majority fail."""
    llm = MagicMock()
    bodies = ['{"is_correct": false}', '{"is_correct": true}', '{"is_correct": false}']
    idx = 0

    async def fake_chat(messages: list[dict[str, Any]], *, model: str | None = None, **kwargs: Any) -> MagicMock:
        nonlocal idx
        body = bodies[idx]
        idx += 1
        return MagicMock(content=body, prompt_tokens=10, completion_tokens=5)

    llm.chat = fake_chat
    result = await judge_qa(
        question="q",
        golden_answer="g",
        generated_answer="a",
        judge_prompt="{question}/{gold_answer}/{response}",
        llm=llm,
        num_runs=3,
    )
    assert result.is_correct is False


@pytest.mark.asyncio
async def test_judge_qa_passes_judge_temperature_to_chat() -> None:
    """judge_temperature flows into every llm.chat() call."""
    llm = MagicMock()
    received_temps: list[float | None] = []

    async def fake_chat(
        messages: list[dict[str, Any]], *, model: str | None = None, temperature: float | None = None, **kwargs: Any
    ) -> MagicMock:
        received_temps.append(temperature)
        return MagicMock(content='{"label": "CORRECT"}', prompt_tokens=1, completion_tokens=1)

    llm.chat = fake_chat
    await judge_qa(
        question="q",
        golden_answer="g",
        generated_answer="a",
        judge_prompt="{question}/{gold_answer}/{response}",
        llm=llm,
        num_runs=2,
        judge_temperature=0.0,
    )
    assert received_temps == [0.0, 0.0]


@pytest.mark.asyncio
async def test_judge_qa_passes_judge_model_to_chat() -> None:
    """Optional judge_model parameter must be forwarded to llm.chat."""
    llm = MagicMock()
    received_models: list[str | None] = []

    async def fake_chat(messages: list[dict[str, Any]], *, model: str | None = None, **kwargs: Any) -> MagicMock:
        received_models.append(model)
        return MagicMock(content='{"is_correct": true}', prompt_tokens=1, completion_tokens=1)

    llm.chat = fake_chat
    await judge_qa(
        question="q",
        golden_answer="g",
        generated_answer="a",
        judge_prompt="{question}/{gold_answer}/{response}",
        llm=llm,
        num_runs=2,
        judge_model="openai/gpt-4o-mini",
    )
    assert received_models == ["openai/gpt-4o-mini"] * 2
