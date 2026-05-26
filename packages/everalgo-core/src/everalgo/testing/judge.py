"""LLM-as-judge with N-run majority vote.

``judge_prompt`` is always caller-supplied (no default). Dataset-specific
templates live in the caller's prompts module (e.g. LoCoMo ``JUDGE_PROMPT`` in
``benchmarks/datasets/locomo/prompts.py``).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from everalgo.llm.types import ChatMessage

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


def _extract_outermost_json_object(text: str) -> str:
    """Extract the outermost ``{...}`` block from ``text`` via greedy slice (``find("{")..rfind("}")+1``).

    More forgiving than a non-greedy ``{[^{}]*}`` regex (tolerates nested objects and surrounding
    prose). Raises ``ValueError`` when no ``{...}`` is found.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in judge LLM response: {text[:200]!r}")
    return text[start : end + 1]


def _parse_judge_output(content: str) -> tuple[bool, str]:
    """Extract pass/fail + reasoning from judge LLM output.

    Extract the outermost ``{...}`` from the LLM response, parse as JSON, require
    a ``label`` field equal to ``"CORRECT"`` or ``"WRONG"`` (case-insensitive).
    All deviations raise — caller is responsible for retry / fail-loud.

    Args:
        content: Raw text response from the judge LLM.

    Returns:
        Tuple of ``(is_correct, reasoning)`` extracted from the content.

    Raises:
        ValueError: When no ``{...}`` is found, or ``label`` is missing /
            not one of ``CORRECT`` / ``WRONG``.
        json.JSONDecodeError: When the extracted slice is not valid JSON.
    """
    data = json.loads(_extract_outermost_json_object(content))
    if "label" not in data:
        raise ValueError(f"Judge JSON missing 'label': {data!r}")
    label = str(data["label"]).strip().upper()
    if label not in ("CORRECT", "WRONG"):
        raise ValueError(f"Unknown judge label: {label!r}")
    reasoning = str(data.get("reasoning", ""))
    return label == "CORRECT", reasoning


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
    max_retries: int = 5,
) -> JudgeResult:
    """Run judge LLM ``num_runs`` times in parallel; majority vote (> N/2 pass = pass).

    Args:
        question: The question to evaluate.
        golden_answer: The ground-truth answer.
        generated_answer: The model-generated answer to judge.
        judge_prompt: Template with ``{question}``, ``{gold_answer}``, ``{response}`` placeholders.
            Caller-supplied; no default — dataset-specific templates live in the caller's prompts module.
        llm: LLM client used to call the judge.
        num_runs: Number of independent judge calls (default 3).
        judge_model: Optional model override forwarded to ``llm.chat``.
        judge_temperature: Temperature for judge calls (default 0.0 for determinism).
        judge_system_prompt: Optional system-role message prepended to the user prompt.
            When provided, the judge call uses ``[{system}, {user}]`` instead of ``[{user}]``.
        max_retries: Per-run retry budget for transient LLM errors or judge-output
            parse failures (default 5). Exponential backoff ``1.0 * 2**attempt``
            between attempts. When all ``max_retries`` attempts of a single run
            fail, the underlying exception propagates and aborts the judge call.

    Returns:
        JudgeResult with per-run outcomes, majority-vote result, and aggregated token counts.

    Raises:
        Exception: Whatever the last failing attempt raised — typically
            ``LLMError`` from the client or ``ValueError`` / ``json.JSONDecodeError``
            from ``_parse_judge_output``. Caller layer aborts the benchmark
            (fail-loud); silent fallback is forbidden.
    """
    rendered = judge_prompt.format(
        question=question,
        gold_answer=golden_answer,
        response=generated_answer,
    )
    messages: list[ChatMessage] = []
    if judge_system_prompt:
        messages.append(ChatMessage(role="system", content=judge_system_prompt))
    messages.append(ChatMessage(role="user", content=rendered))

    async def _one_run() -> tuple[bool, str, int, int]:
        # Per-run retry mirrors stage 3 / stage 4: any exception inside the
        # attempt (LLM transport, parse failure) triggers exponential-backoff
        # retry; the final attempt re-raises so the outer benchmark aborts.
        for attempt in range(max_retries):
            try:
                resp = await llm.chat(messages, model=judge_model, temperature=judge_temperature)
                is_correct, reasoning = _parse_judge_output(resp.content)
            except Exception:
                if attempt == max_retries - 1:
                    raise
                logger.warning(
                    "judge attempt %d/%d failed; retrying",
                    attempt + 1,
                    max_retries,
                    exc_info=True,
                )
                await asyncio.sleep(1.0 * (2**attempt))
                continue
            pt = (resp.usage.prompt_tokens or 0) if resp.usage is not None else 0
            ct = (resp.usage.completion_tokens or 0) if resp.usage is not None else 0
            return is_correct, reasoning, pt, ct
        raise RuntimeError(f"_one_run: exhausted {max_retries} retries without success or raise")

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
