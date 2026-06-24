"""Pipeline orchestrator — wires stages 1..7 in sequence."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmarks.common.metrics import avg_context_tokens, stage_summary
from benchmarks.common.profile import collect_config_dump, collect_package_versions
from benchmarks.common.report import generate_reports
from benchmarks.common.services import Services
from benchmarks.common.stages.answer import run_answer_stage
from benchmarks.common.stages.enrich import run_enrich_stage
from benchmarks.common.stages.evaluate import run_evaluate_stage
from benchmarks.common.stages.extract import run_extract_base_stage
from benchmarks.common.stages.index import run_index_stage
from benchmarks.common.stages.reflect import run_reflect_stage
from benchmarks.common.stages.search import run_search_stage
from benchmarks.common.stages.types import StageContext, StageStats
from benchmarks.datasets import discover_datasets

if TYPE_CHECKING:
    from benchmarks.common.config import BenchmarkConfig


@dataclass(frozen=True)
class PipelineRequest:
    """Inputs for a single benchmark run."""

    dataset_name: str
    run_name: str
    config: BenchmarkConfig
    stages: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7])
    smoke: bool = False
    smoke_conv_limit: int = 1
    smoke_msg_limit: int = 50
    conv_indices: list[int] | None = None
    data_path: Path | None = None
    output_dir: Path | None = None


_STAGE_RUNNERS = {
    1: ("stage1_extract_base", run_extract_base_stage),
    2: ("stage2_reflect", run_reflect_stage),
    3: ("stage3_enrich", run_enrich_stage),
    4: ("stage4_index", run_index_stage),
    5: ("stage5_search", run_search_stage),
    6: ("stage6_answer", run_answer_stage),
    7: ("stage7_evaluate", run_evaluate_stage),
}

# Approximate prices per million tokens (USD), as of 2025-Q2.
# Stage 1 (extract_base): gpt-4.1-mini via OpenRouter
# Stage 2 (reflect):      gpt-4.1-mini via OpenRouter
# Stage 3 (enrich):       gpt-4.1-mini via OpenRouter (atomic-fact extraction)
# Stage 4 (index):        Qwen3-Embedding-4B via DeepInfra — embedding only, no completion
# Stage 5 (search):       gpt-4.1-mini via OpenRouter (agentic sufficiency / query refinement)
# Stage 6 (answer):       gpt-4.1-mini via OpenRouter
# Stage 7 (evaluate):     gpt-4o-mini via OpenRouter
_PRICES_PER_MTOK: dict[str, dict[str, float]] = {
    "extract": {"prompt": 0.40, "completion": 1.60},
    "reflect": {"prompt": 0.40, "completion": 1.60},
    "enrich": {"prompt": 0.40, "completion": 1.60},
    "index": {"prompt": 0.10, "completion": 0.0},
    "search": {"prompt": 0.40, "completion": 1.60},
    "answer": {"prompt": 0.40, "completion": 1.60},
    "evaluate": {"prompt": 0.15, "completion": 0.60},
}


def _stage_cost(stage_name: str, prompt: int, completion: int) -> float:
    """Estimate USD cost for a single stage given token counts.

    Args:
        stage_name: Stage name key matching ``_PRICES_PER_MTOK``.
        prompt: Prompt token count.
        completion: Completion token count.

    Returns:
        Estimated cost in USD.
    """
    prices = _PRICES_PER_MTOK.get(stage_name, {"prompt": 0.0, "completion": 0.0})
    return (prompt * prices["prompt"] + completion * prices["completion"]) / 1_000_000


async def run_pipeline(req: PipelineRequest) -> dict[str, Any]:
    """Run the requested stages in sequence and return a summary dict.

    Args:
        req: Immutable pipeline request specifying dataset, stages, and config.

    Returns:
        Dict with ``stats`` (list of StageStats) and ``output_dir`` (str path).

    Raises:
        ValueError: If dataset name is unknown or data_path is not provided.
    """
    _validate_request(req)

    dataset = _load_dataset(req)
    output_root = _resolve_output_dir(req)
    services = Services.from_config(req.config)

    all_stats = await _run_stages(req, dataset, services, output_root)
    summary_data = stage_summary(all_stats)

    if 7 in req.stages:
        _maybe_generate_reports(req, output_root, summary_data, all_stats)

    _write_profile(output_root, summary_data)

    return {"stats": all_stats, "output_dir": str(output_root)}


def _validate_request(req: PipelineRequest) -> None:
    """Raise ValueError for obviously invalid requests before touching disk."""
    registry = discover_datasets()
    if req.dataset_name not in registry:
        raise ValueError(f"Unknown dataset: {req.dataset_name!r}. Available: {sorted(registry)}")
    if req.data_path is None:
        raise ValueError("data_path is required")


def _load_dataset(req: PipelineRequest) -> Any:
    """Instantiate the dataset from the registry factory.

    Passes ``session_filter`` from the config as a keyword argument when the
    factory accepts it (LoCoMo does; future datasets may not). Uses
    ``inspect.signature`` to avoid coupling the runner to dataset-specific APIs.
    """
    import inspect

    registry = discover_datasets()
    factory = registry[req.dataset_name]
    sig = inspect.signature(factory)
    if "session_filter" in sig.parameters:
        return factory(req.data_path, session_filter=req.config.session_filter)  # type: ignore[call-arg]
    return factory(req.data_path)


def _resolve_output_dir(req: PipelineRequest) -> Path:
    """Return and create the output root directory."""
    output_root = req.output_dir or (Path("benchmarks/results") / f"{req.dataset_name}-{req.run_name}")
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root


async def _run_stages(
    req: PipelineRequest,
    dataset: Any,
    services: Services,
    output_root: Path,
) -> list[StageStats]:
    """Execute each requested stage in sorted numeric order, chaining output dirs."""
    all_stats: list[StageStats] = []
    # When resuming from a non-first stage (e.g. ``--stages 2 3 4 5``), seed prev_output
    # with the prior stage's output subdir so the resumed stage finds the on-disk artifact
    # written by the previous run (each stage's run writes ``output_root/{stageN_xxx}/``).
    sorted_stages = sorted(req.stages)
    if sorted_stages and sorted_stages[0] > 1:
        prev_subdir, _ = _STAGE_RUNNERS[sorted_stages[0] - 1]
        prev_output: Path = output_root / prev_subdir
    else:
        prev_output = output_root  # stage 1 ignores input_dir; later stages chain

    for stage_num in sorted_stages:
        subdir_name, runner = _STAGE_RUNNERS[stage_num]
        stage_out = output_root / subdir_name
        print(f"\n{'=' * 60}\nStage {stage_num}: {subdir_name}\n{'=' * 60}", flush=True)
        ctx = StageContext(
            config=req.config,
            services=services,
            dataset=dataset,
            input_dir=prev_output,
            output_dir=stage_out,
            smoke=req.smoke,
            smoke_conv_limit=req.smoke_conv_limit,
            smoke_msg_limit=req.smoke_msg_limit,
            conv_indices=req.conv_indices,
        )
        stats = await runner(ctx)
        all_stats.append(stats)
        prev_output = stage_out
        if stats.stage_name == "evaluate":
            print(
                f"  ↳ done: {stats.success} correct, {stats.failed} wrong, {stats.duration_seconds:.1f}s",
                flush=True,
            )
        else:
            print(
                f"  ↳ done: {stats.success} ok, {stats.failed} failed, {stats.duration_seconds:.1f}s",
                flush=True,
            )

    return all_stats


def _maybe_generate_reports(
    req: PipelineRequest,
    output_root: Path,
    summary_data: dict[str, Any],
    all_stats: list[StageStats],
) -> None:
    """Generate text + JSON reports if stage 7 output files exist."""
    eval_path = output_root / "stage7_evaluate" / "eval_results.json"
    answer_path = output_root / "stage6_answer" / "answers.json"
    if not (eval_path.exists() and answer_path.exists()):
        return
    eval_results = json.loads(eval_path.read_text())
    answers = json.loads(answer_path.read_text())
    generate_reports(
        output_dir=output_root,
        eval_results=eval_results,
        stage_summary=summary_data,
        answers=answers,
        dataset_name=req.dataset_name,
        run_name=req.run_name,
        config_dump=collect_config_dump(req.config),
        package_versions=collect_package_versions(),
    )
    _print_summary(
        output_root=output_root,
        dataset_name=req.dataset_name,
        run_name=req.run_name,
        eval_results=eval_results,
        answers=answers,
        all_stats=all_stats,
    )


def _print_accuracy(
    dataset_name: str,
    run_name: str,
    eval_results: dict[str, Any],
    answers: list[dict[str, Any]],
) -> None:
    """Print the overall accuracy header block."""
    total = eval_results.get("total_questions", 0)
    avg_tokens = avg_context_tokens(answers)
    maj_correct = eval_results.get("majority_correct", eval_results.get("correct", 0))
    maj_acc = eval_results.get("majority_accuracy", eval_results.get("accuracy", 0.0))

    print("\n" + "=" * 60, flush=True)
    print(f"📊 Results: {dataset_name} / {run_name}", flush=True)
    print("=" * 60, flush=True)
    print(f"{'Overall:':<32}  {maj_acc * 100:>6.2f}%   ({maj_correct}/{total})", flush=True)
    print(f"{'Avg context tokens per question:':<32}  {avg_tokens:>6}    (Stage 6 retrieved context)", flush=True)


def _print_categories(per_cat: dict[str, Any]) -> None:
    """Print per-category accuracy breakdown."""
    if not per_cat:
        return
    print("\nPer-Category:", flush=True)
    for cat_num, v in sorted(per_cat.items(), key=lambda kv: kv[0]):
        label = v.get("label", f"unknown-{cat_num}")
        mc = v.get("majority_correct", v.get("correct", 0))
        t = v.get("total", 0)
        ma = v.get("majority_accuracy", v.get("accuracy", 0.0))
        cat_str = f"Category {cat_num} ({label})"
        print(f"  {cat_str:<30} {ma * 100:>6.2f}%   ({mc:>4}/{t:>4})", flush=True)


def _print_timings(all_stats: list[StageStats]) -> None:
    """Print per-stage timing breakdown."""
    print("\nStage timings:", flush=True)
    total_secs = sum(s.duration_seconds for s in all_stats)
    for s in all_stats:
        print(f"  {s.stage_name:<10}  {s.duration_seconds:>10.1f}s", flush=True)
    print(f"  {'total':<10}  {total_secs:>10.1f}s", flush=True)


def _print_token_usage(all_stats: list[StageStats]) -> None:
    """Print per-stage token usage and cost estimates."""
    print("\nStage token usage (estimated):", flush=True)
    total_cost = 0.0
    for s in all_stats:
        cost = _stage_cost(s.stage_name, s.prompt_tokens, s.completion_tokens)
        total_cost += cost
        if s.completion_tokens > 0:
            print(
                f"  {s.stage_name:<10} {s.prompt_tokens:>11,} prompt + "
                f"{s.completion_tokens:>9,} completion ≈ ${cost:.2f}",
                flush=True,
            )
        else:
            print(f"  {s.stage_name:<10} {s.prompt_tokens:>11,} prompt{'':>23}≈ ${cost:.2f}", flush=True)
    print(f"  {'total':<10} {'':>34}≈ ${total_cost:.2f}", flush=True)


def _print_summary(
    output_root: Path,
    dataset_name: str,
    run_name: str,
    eval_results: dict[str, Any],
    answers: list[dict[str, Any]],
    all_stats: list[StageStats],
) -> None:
    """Print a final results summary block to stdout."""
    _print_accuracy(dataset_name, run_name, eval_results, answers)
    _print_categories(eval_results.get("per_category", {}))
    _print_timings(all_stats)
    _print_token_usage(all_stats)
    print(f"\nReport: {output_root}/report.txt", flush=True)
    print("=" * 60, flush=True)


def _write_profile(output_root: Path, summary_data: dict[str, Any]) -> None:
    """Write per-stage summary to profile.json at the output root."""
    (output_root / "profile.json").write_text(json.dumps(summary_data, ensure_ascii=False, indent=2))
