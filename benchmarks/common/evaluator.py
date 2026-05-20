"""LLM judge — N-run majority vote.

For each QA triple, run the judge prompt N times and take majority vote.
Matches EverCore evaluator's methodology.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from benchmarks.common.services import LLMClient


@dataclass(frozen=True)
class EvalResult:
    """Outcome of judging one (question, golden, generated) triple."""

    is_correct: bool
    runs: list[bool]
    reasoning: list[str]
    prompt_tokens: int = 0
    completion_tokens: int = 0


_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse_judge_output(content: str) -> tuple[bool, str]:
    """Extract pass/fail from judge LLM output.

    LoCoMo's judge prompt asks for {"label": "CORRECT" | "WRONG"}. Falls back to
    keyword scan on raw text if no JSON is found.
    """
    m = _JSON_OBJECT_RE.search(content)
    if m:
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            data = None
        if data is not None:
            label = str(data.get("label", "")).strip().upper()
            if label in ("CORRECT", "WRONG"):
                reasoning = str(data.get("reasoning", ""))
                return label == "CORRECT", reasoning
            # Legacy fallback: some judges might use is_correct
            if "is_correct" in data:
                return bool(data["is_correct"]), str(data.get("reasoning", ""))
    # Heuristic last resort
    lower = content.lower()
    is_correct = "correct" in lower and "incorrect" not in lower and "not correct" not in lower and "wrong" not in lower
    return is_correct, content


async def judge_qa(
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
) -> EvalResult:
    """Run judge LLM ``num_runs`` times in parallel; majority vote (>N/2 pass = pass).

    Args:
        question: The question to evaluate.
        golden_answer: The ground-truth answer.
        generated_answer: The model-generated answer to judge.
        judge_prompt: Template with ``{question}``, ``{gold_answer}``, ``{response}`` placeholders.
        llm: LLM client used to call the judge.
        num_runs: Number of independent judge calls (default 3).
        judge_model: Optional model override forwarded to ``llm.chat``.
        judge_temperature: Temperature for judge calls (default 0.0 for determinism, matching EverCore).
        judge_system_prompt: Optional system-role message prepended to the user prompt.
            When provided, the judge call becomes ``[{system}, {user}]`` instead of
            ``[{user}]``, mirroring EverCore's ``locomo_grader`` two-message layout
            (``stage5_eval.py:61-66``).

    Returns:
        EvalResult with per-run outcomes and majority vote result.
    """
    prompt = judge_prompt.format(
        question=question,
        gold_answer=golden_answer,
        response=generated_answer,
    )
    messages: list[dict[str, str]] = []
    if judge_system_prompt:
        messages.append({"role": "system", "content": judge_system_prompt})
    messages.append({"role": "user", "content": prompt})

    async def _one_run() -> tuple[bool, str, int, int]:
        resp = await llm.chat(
            messages,
            model=judge_model,
            temperature=judge_temperature,
        )
        is_correct, reasoning = _parse_judge_output(resp.content)
        return is_correct, reasoning, resp.prompt_tokens, resp.completion_tokens

    outcomes = await asyncio.gather(*(_one_run() for _ in range(num_runs)))
    runs = [c for c, _, _, _ in outcomes]
    reasoning = [r for _, r, _, _ in outcomes]
    prompt_tokens = sum(p for _, _, p, _ in outcomes)
    completion_tokens = sum(c for _, _, _, c in outcomes)
    return EvalResult(
        is_correct=sum(runs) > num_runs / 2,
        runs=runs,
        reasoning=reasoning,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
