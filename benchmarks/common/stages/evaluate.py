"""Stage 5 — LLM judge + per-category aggregation.

Reads ``answers.json``, drops adversarial categories, runs N parallel judge
calls per QA, and writes ``eval_results.json``. Headline accuracy is the
mean of per-run accuracies (mean-of-runs); per-question ``is_correct`` is
persisted in ``detailed_results`` for inspection.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np

from benchmarks.common.stages.types import StageStats
from everalgo.testing.judge import JudgeResult, allm_judge

if TYPE_CHECKING:
    from benchmarks.common.stages.types import StageContext


async def _judge_one(
    answer: dict[str, Any],
    ctx: StageContext,
    judge_prompt: str,
    judge_system_prompt: str | None,
) -> dict[str, Any]:
    """Run judge for a single QA. Exceptions propagate so the stage fails loudly."""
    result: JudgeResult = await allm_judge(
        question=answer["question"],
        golden_answer=answer["golden_answer"],
        generated_answer=answer["answer"],
        judge_prompt=judge_prompt,
        llm=ctx.services.llm,  # type: ignore[arg-type]
        num_runs=ctx.config.judge_runs,
        judge_model=ctx.config.judge_model,
        judge_temperature=ctx.config.judge_temperature,
        judge_system_prompt=judge_system_prompt,
        max_retries=ctx.config.llm_max_retries,
    )
    return {
        **answer,
        "is_correct": result.is_correct,
        "judge_runs": result.runs,
        "_prompt_tokens": result.prompt_tokens,
        "_completion_tokens": result.completion_tokens,
    }


# ─── Step 1: load + filter ────────────────────────────────────────────────


def _load_scored_answers(ctx: StageContext) -> list[dict[str, Any]]:
    """Read ``answers.json``, drop adversarial categories, apply smoke cap."""
    answers: list[dict[str, Any]] = json.loads((ctx.input_dir / "answers.json").read_text())
    filter_cats = ctx.dataset.filter_categories()
    scored = [a for a in answers if str(a.get("category")) not in filter_cats]
    if ctx.smoke:
        scored = scored[: ctx.smoke_conv_limit * ctx.smoke_qa_limit]
    return scored


# ─── Step 2: judge ────────────────────────────────────────────────────────


async def _judge_all(
    scored_inputs: list[dict[str, Any]],
    ctx: StageContext,
    judge_prompt: str,
    judge_system_prompt: str | None,
) -> list[dict[str, Any]]:
    """Fan out ``_judge_one`` over all QAs with bounded concurrency."""
    from benchmarks.common._progress import gather_with_progress

    sem = asyncio.Semaphore(ctx.config.max_concurrent_qa)

    async def _process(ans: dict[str, Any]) -> dict[str, Any]:
        async with sem:
            return await _judge_one(ans, ctx, judge_prompt, judge_system_prompt)

    return await gather_with_progress(
        *(_process(a) for a in scored_inputs),
        desc="evaluate",
        unit="q",
    )


# ─── Step 3: aggregate ────────────────────────────────────────────────────


def _per_run_accuracies(detailed: list[dict[str, Any]], num_runs: int) -> list[float]:
    """Accuracy of each independent judge run across all questions."""
    total = len(detailed)
    if not total:
        return [0.0] * num_runs
    return [sum(1 for d in detailed if (d.get("judge_runs") or [False] * num_runs)[i]) / total for i in range(num_runs)]


def _overall_metrics(
    detailed: list[dict[str, Any]],
    num_runs: int,
) -> tuple[int, float, float, list[float]]:
    """Return ``(correct, mean_accuracy, std_accuracy, run_accuracies)``."""
    run_accs = _per_run_accuracies(detailed, num_runs)
    mean_acc = sum(run_accs) / len(run_accs) if run_accs else 0.0
    std_acc = float(np.std(run_accs)) if run_accs else 0.0
    correct = round(mean_acc * len(detailed))
    return correct, mean_acc, std_acc, run_accs


def _aggregate_per_category(
    detailed: list[dict[str, Any]],
    category_label_fn: Any,
    num_runs: int,
) -> dict[str, dict[str, Any]]:
    """Build per-category mean-of-runs accuracy keyed by category number string."""
    cat_items: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in detailed:
        cat_items[str(item.get("category", ""))].append(item)

    summary: dict[str, dict[str, Any]] = {}
    for cat, items in cat_items.items():
        total = len(items)
        run_accs = _per_run_accuracies(items, num_runs)
        mean_acc = sum(run_accs) / len(run_accs) if run_accs else 0.0
        maj_correct = sum(1 for d in items if d.get("is_correct"))
        summary[cat] = {
            "label": category_label_fn(cat),
            "correct": round(mean_acc * total),
            "total": total,
            "accuracy": mean_acc,
            "majority_correct": maj_correct,
            "majority_accuracy": maj_correct / total if total else 0.0,
            "run_accuracies": run_accs,
        }
    return summary


# ─── Step 4: persist ──────────────────────────────────────────────────────


def _strip_internal_keys(detailed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop ``_``-prefixed token-tracking keys before persisting."""
    return [{k: v for k, v in d.items() if not k.startswith("_")} for d in detailed]


# ─── Entry point ──────────────────────────────────────────────────────────


async def run_evaluate_stage(ctx: StageContext) -> StageStats:
    """Stage 5 — judge every answer and aggregate per-category accuracy."""
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    stats = StageStats(stage_name="evaluate")
    started = time.monotonic()

    # Step 1: load + filter answers
    scored_inputs = _load_scored_answers(ctx)

    # Step 2: judge each QA (LoCoMo provides judge_system_prompt; others may not)
    judge_prompt = ctx.dataset.judge_prompt()
    # Datasets may optionally expose a system-role prompt; LoCoMo does, others may not.
    # ``getattr`` with a lambda default avoids forcing every dataset to add the method.
    judge_system_prompt_fn = getattr(ctx.dataset, "judge_system_prompt", lambda: None)
    judge_system_prompt: str | None = judge_system_prompt_fn()
    detailed = await _judge_all(scored_inputs, ctx, judge_prompt, judge_system_prompt)

    # Step 3: compute overall + per-category accuracy
    num_runs = ctx.config.judge_runs
    correct, mean_acc, std_acc, run_accs = _overall_metrics(detailed, num_runs)
    majority_correct = sum(1 for d in detailed if d.get("is_correct"))
    per_category = _aggregate_per_category(detailed, ctx.dataset.category_label, num_runs)

    # Step 4: persist eval_results.json + return stats
    stats.prompt_tokens = sum(d.get("_prompt_tokens", 0) for d in detailed)
    stats.completion_tokens = sum(d.get("_completion_tokens", 0) for d in detailed)
    total = len(detailed)
    out: dict[str, Any] = {
        "total_questions": total,
        "correct": correct,
        "accuracy": mean_acc,
        "majority_correct": majority_correct,
        "majority_accuracy": majority_correct / total if total else 0.0,
        "std_accuracy": std_acc,
        "run_accuracies": run_accs,
        "per_category": per_category,
        "detailed_results": _strip_internal_keys(detailed),
    }
    (ctx.output_dir / "eval_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))

    stats.success = correct
    stats.failed = total - correct
    stats.duration_seconds = time.monotonic() - started
    return stats
