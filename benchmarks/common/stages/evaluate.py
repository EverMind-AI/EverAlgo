"""Stage 5 — LLM judge + per-category aggregation.

Reads ``answers.json`` from the previous stage, filters out adversarial
categories, runs N-parallel judge calls per QA, and writes ``eval_results.json``.

Scoring algorithm: **mean-of-runs** (mirror EverCore main ``stage5_eval.py:245-267``).
For each independent judge run, compute accuracy across all questions; report the
mean of the N run-level accuracies (and std deviation). This is the canonical
EverCore baseline 92.32% computation. Per-question ``is_correct`` (majority vote
from ``EvalResult.is_correct``) is still persisted in ``detailed_results`` for
inspection but does not drive the headline ``accuracy`` number.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from benchmarks.common.evaluator import judge_qa
from benchmarks.common.stages.types import StageStats

if TYPE_CHECKING:
    from benchmarks.common.stages.types import StageContext


async def _judge_one(
    answer: dict[str, Any],
    ctx: StageContext,
    judge_prompt: str,
    judge_system_prompt: str | None,
) -> dict[str, Any]:
    """Run judge for a single QA answer dict; returns enriched dict with judge outcome."""
    try:
        result = await judge_qa(
            question=answer["question"],
            golden_answer=answer["golden_answer"],
            generated_answer=answer["answer"],
            judge_prompt=judge_prompt,
            llm=ctx.services.llm,
            num_runs=ctx.config.judge_runs,
            judge_model=ctx.config.judge_model,
            judge_temperature=ctx.config.judge_temperature,
            judge_system_prompt=judge_system_prompt,
        )
    except Exception as exc:
        return {**answer, "is_correct": False, "judge_error": repr(exc)}
    else:
        return {
            **answer,
            "is_correct": result.is_correct,
            "judge_runs": result.runs,
            "_prompt_tokens": result.prompt_tokens,
            "_completion_tokens": result.completion_tokens,
        }


def _aggregate_per_category(
    detailed: list[dict[str, Any]],
    category_label_fn: Any,
    num_runs: int,
) -> dict[str, dict[str, Any]]:
    """Build per-category mean-of-runs accuracy keyed by category number string.

    For each category, compute the accuracy of each of the ``num_runs``
    independent judge runs, then report the **mean** across runs (mirror
    EverCore main ``stage5_eval.py:245-267``). ``correct`` is the mean-rounded
    integer for legible reporting; the precise mean is in ``accuracy``.
    """
    cat_items: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in detailed:
        cat_items[str(item.get("category", ""))].append(item)

    summary: dict[str, dict[str, Any]] = {}
    for cat, items in cat_items.items():
        total = len(items)
        run_accs: list[float] = []
        for i in range(num_runs):
            correct_i = sum(1 for it in items if (it.get("judge_runs") or [False] * num_runs)[i])
            run_accs.append(correct_i / total if total else 0.0)
        mean_acc = sum(run_accs) / len(run_accs) if run_accs else 0.0
        summary[cat] = {
            "label": category_label_fn(cat),
            "correct": round(mean_acc * total),
            "total": total,
            "accuracy": mean_acc,
            "run_accuracies": run_accs,
        }
    return summary


async def run_evaluate_stage(ctx: StageContext) -> StageStats:
    """Stage 5 — judge every answer and aggregate per-category accuracy.

    Reads ``answers.json``, filters adversarial categories, runs the LLM
    judge with majority vote, writes ``eval_results.json``.
    """
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    stats = StageStats(stage_name="evaluate")
    started = time.monotonic()

    answers: list[dict[str, Any]] = json.loads((ctx.input_dir / "answers.json").read_text())
    filter_cats = ctx.dataset.filter_categories()
    scored_inputs = [a for a in answers if str(a.get("category")) not in filter_cats]

    if ctx.smoke:
        smoke_limit = ctx.smoke_conv_limit * ctx.smoke_qa_limit
        scored_inputs = scored_inputs[:smoke_limit]

    sem = asyncio.Semaphore(ctx.config.max_concurrent_qa)
    judge_prompt = ctx.dataset.judge_prompt()
    # Datasets may optionally expose a system-role prompt; LoCoMo does, others may
    # not. ``getattr`` with a lambda default avoids forcing every dataset to add
    # the method while still mirroring EverCore when present.
    judge_system_prompt_fn = getattr(ctx.dataset, "judge_system_prompt", lambda: None)
    judge_system_prompt: str | None = judge_system_prompt_fn()

    async def _process(ans: dict[str, Any]) -> dict[str, Any]:
        async with sem:
            return await _judge_one(ans, ctx, judge_prompt, judge_system_prompt)

    from tqdm.asyncio import tqdm as async_tqdm  # type: ignore[import-untyped]

    detailed: list[dict[str, Any]] = await async_tqdm.gather(  # type: ignore[attr-defined]
        *(_process(a) for a in scored_inputs),
        desc="evaluate",
        unit="q",
        dynamic_ncols=True,
    )

    total = len(detailed)
    num_runs = ctx.config.judge_runs

    # Mean-of-runs accuracy (mirror EverCore main ``stage5_eval.py:245-267``):
    # compute per-run accuracy across all questions, then mean ± std across runs.
    run_accuracies: list[float] = []
    for i in range(num_runs):
        correct_i = sum(1 for d in detailed if (d.get("judge_runs") or [False] * num_runs)[i])
        run_accuracies.append(correct_i / total if total else 0.0)
    mean_accuracy = sum(run_accuracies) / len(run_accuracies) if run_accuracies else 0.0
    std_accuracy = statistics.stdev(run_accuracies) if len(run_accuracies) > 1 else 0.0
    correct_rounded = round(mean_accuracy * total)

    per_category_summary = _aggregate_per_category(detailed, ctx.dataset.category_label, num_runs)

    stats.prompt_tokens = sum(d.get("_prompt_tokens", 0) for d in detailed)
    stats.completion_tokens = sum(d.get("_completion_tokens", 0) for d in detailed)

    # Strip internal token-tracking keys before persisting
    clean_detailed = [{k: v for k, v in d.items() if not k.startswith("_")} for d in detailed]

    out: dict[str, Any] = {
        "total_questions": total,
        "correct": correct_rounded,
        "accuracy": mean_accuracy,
        "std_accuracy": std_accuracy,
        "run_accuracies": run_accuracies,
        "per_category": per_category_summary,
        "detailed_results": clean_detailed,
    }
    (ctx.output_dir / "eval_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))

    stats.success = correct_rounded
    stats.failed = total - correct_rounded
    stats.duration_seconds = time.monotonic() - started
    return stats
