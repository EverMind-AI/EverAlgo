"""LLM-as-judge with N-run majority vote.

``judge_prompt`` is always caller-supplied (no default). Dataset-specific
templates live in the caller's prompts module (e.g. LoCoMo ``JUDGE_PROMPT`` in
``benchmarks/datasets/locomo/prompts.py``).

Retry is the caller's responsibility.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from benchmarks.common.metrics import estimate_chat_tokens
from everalgo.llm import LLMMessage, StrictOutputModel, parse_reply
from everalgo.utils import render_prompt

if TYPE_CHECKING:
    from everalgo.llm.protocols import LLMClient

__all__ = ["JudgeResult", "allm_judge"]

logger = logging.getLogger(__name__)


class JudgeResult(BaseModel):
    """Outcome of LLM-as-judge with N-run majority vote."""

    model_config = ConfigDict(frozen=True)

    is_correct: bool
    runs: list[bool] = Field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    reasoning: list[str] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0


class _JudgeVerdict(StrictOutputModel):
    """The judge's reply: the one-sentence reasoning the prompt asks for first, then the label."""

    reasoning: str
    label: Literal["CORRECT", "WRONG"]


async def allm_judge(
    *,
    question: str,
    golden_answer: str,
    generated_answer: str,
    judge_prompt: str,
    llm: LLMClient,
    num_runs: int = 3,
    judge_model: str | None = None,
    judge_temperature: float = 0.0,
    judge_system_prompt: str | None = None,
) -> JudgeResult:
    """Run judge LLM ``num_runs`` times in parallel; majority vote (> N/2 pass = pass).

    Each run is a single LLM call. On failure the exception propagates immediately —
    the caller owns the retry policy.

    Args:
        question: The question to evaluate.
        golden_answer: The ground-truth answer.
        generated_answer: The model-generated answer to judge.
        judge_prompt: Template with ``{{question}}``, ``{{gold_answer}}``, ``{{response}}`` placeholders.
        llm: LLM client used to call the judge.
        num_runs: Number of independent judge calls (default 3).
        judge_model: Optional model override; the contract has no ``model`` parameter, so it rides
            the ``**extra`` escape hatch and is omitted entirely when unset.
        judge_temperature: Temperature for judge calls (default 0.0 for determinism).
        judge_system_prompt: Optional system-role message prepended to the user prompt.

    Returns:
        JudgeResult with per-run outcomes, majority-vote result, and aggregated estimated token counts.

    Raises:
        LLMOutputError: If a run's reply is not a valid verdict.
    """
    rendered = render_prompt(
        judge_prompt,
        None,
        question=question,
        gold_answer=golden_answer,
        response=generated_answer,
    )
    messages: list[LLMMessage] = []
    if judge_system_prompt:
        messages.append(LLMMessage(role="system", content=judge_system_prompt))
    messages.append(LLMMessage(role="user", content=rendered))

    model_override: dict[str, Any] = {"model": judge_model} if judge_model is not None else {}

    async def _one_run() -> tuple[bool, str, int, int]:
        reply = await llm.achat(
            messages,
            response_format=_JudgeVerdict,
            temperature=judge_temperature,
            **model_override,
        )
        verdict = parse_reply(_JudgeVerdict, reply, operator="judge")
        prompt_tokens, completion_tokens = estimate_chat_tokens(messages, reply)
        return verdict.label == "CORRECT", verdict.reasoning, prompt_tokens, completion_tokens

    outcomes = await asyncio.gather(*(_one_run() for _ in range(num_runs)))
    runs = [c for c, _, _, _ in outcomes]
    reasoning_list = [r for _, r, _, _ in outcomes]
    prompt_tokens = sum(p for _, _, p, _ in outcomes)
    completion_tokens = sum(c for _, _, _, c in outcomes)

    logger.debug("judge: %d/%d voted correct", sum(runs), num_runs)

    return JudgeResult(
        is_correct=sum(runs) > num_runs / 2,
        runs=runs,
        reasoning=reasoning_list,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
