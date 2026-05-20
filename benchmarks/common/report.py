"""Report rendering — produces human-readable text + structured JSON outputs."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from benchmarks.common.metrics import avg_context_tokens

if TYPE_CHECKING:
    from pathlib import Path


def generate_reports(
    *,
    output_dir: Path,
    eval_results: dict[str, Any],
    stage_summary: dict[str, dict[str, Any]],
    answers: list[dict[str, Any]],
    dataset_name: str,
    run_name: str,
    config_dump: dict[str, Any],
    package_versions: dict[str, str],
) -> None:
    """Write report.txt (markdown) and report.json (structured) to ``output_dir``.

    Args:
        output_dir: Directory where report.txt and report.json will be written.
        eval_results: Evaluation metrics (accuracy, per_category, total_questions, correct).
        stage_summary: Stage-level metrics (duration_seconds, prompt_tokens, completion_tokens).
        answers: List of answer records (each with formatted_context for token calculation).
        dataset_name: Name of the dataset (e.g., "locomo").
        run_name: Name of the run (e.g., "v1").
        config_dump: Configuration dict (e.g., llm_model, judge_runs).
        package_versions: Package versions dict (e.g., {"everalgo-core": "0.1.0"}).
    """
    from pathlib import Path as PathlibPath

    output_dir = PathlibPath(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    avg_tokens = avg_context_tokens(answers)
    overall = eval_results.get("accuracy", 0.0)
    per_cat = eval_results.get("per_category", {})

    # ---- JSON report ----
    report_json = {
        "dataset": dataset_name,
        "run_name": run_name,
        "overall_accuracy": overall,
        "total_questions": eval_results.get("total_questions", 0),
        "correct": eval_results.get("correct", 0),
        "per_category": per_cat,
        "avg_context_tokens": avg_tokens,
        "stage_summary": stage_summary,
        "config": config_dump,
        "package_versions": package_versions,
    }
    (output_dir / "report.json").write_text(json.dumps(report_json, ensure_ascii=False, indent=2))

    # ---- Text report ----
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append(f"EverAlgo Benchmark Report — {dataset_name} / {run_name}")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Total Questions: {eval_results.get('total_questions', 0)}")
    lines.append(f"Correct: {eval_results.get('correct', 0)}")
    lines.append(f"Overall Accuracy: {overall:.4f} ({overall * 100:.2f}%)")
    lines.append("")

    if per_cat:
        lines.append("Per-Category Accuracy:")
        sorted_cats = sorted(per_cat.items(), key=lambda kv: kv[0])
        for cat_num, v in sorted_cats:
            label = v.get("label", f"unknown-{cat_num}")
            acc = v.get("accuracy", 0.0)
            c = v.get("correct", 0)
            t = v.get("total", 0)
            lines.append(f"  Category {cat_num} ({label:<13}) {acc:.4f} ({c:>3}/{t:>3})")
        lines.append("")

    lines.append(f"Avg Context Tokens: {avg_tokens}")
    lines.append("")

    if stage_summary:
        lines.append("Stage Summary:")
        for stage, s in stage_summary.items():
            lines.append(
                f"  {stage:<10} duration={s.get('duration_seconds', 0):.1f}s  "
                f"prompt_tokens={s.get('prompt_tokens', 0)}  "
                f"completion_tokens={s.get('completion_tokens', 0)}"
            )
        lines.append("")

    if config_dump:
        lines.append("Config:")
        for k, v in config_dump.items():
            lines.append(f"  {k}: {v}")
        lines.append("")

    if package_versions:
        lines.append("Package Versions:")
        for k, v in package_versions.items():
            lines.append(f"  {k}: {v}")
        lines.append("")

    lines.append("=" * 60)
    (output_dir / "report.txt").write_text("\n".join(lines) + "\n")
