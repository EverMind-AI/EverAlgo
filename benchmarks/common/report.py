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
    """Write report.txt and report.json to ``output_dir``."""
    from pathlib import Path as PathlibPath

    output_dir = PathlibPath(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    avg_tokens = avg_context_tokens(answers)
    total = eval_results.get("total_questions", 0)
    mean_acc = eval_results.get("accuracy", 0.0)
    mean_correct = eval_results.get("correct", 0)
    maj_acc = eval_results.get("majority_accuracy", mean_acc)
    maj_correct = eval_results.get("majority_correct", mean_correct)
    per_cat = eval_results.get("per_category", {})

    report_json = {
        "dataset": dataset_name,
        "run_name": run_name,
        "total_questions": total,
        "majority_accuracy": maj_acc,
        "majority_correct": maj_correct,
        "mean_of_runs_accuracy": mean_acc,
        "mean_of_runs_correct": mean_correct,
        "per_category": per_cat,
        "avg_context_tokens": avg_tokens,
        "stage_summary": stage_summary,
        "config": config_dump,
        "package_versions": package_versions,
    }
    (output_dir / "report.json").write_text(json.dumps(report_json, ensure_ascii=False, indent=2))

    lines = _build_text_report(
        dataset_name=dataset_name,
        run_name=run_name,
        total=total,
        avg_tokens=avg_tokens,
        maj_acc=maj_acc,
        maj_correct=maj_correct,
        mean_acc=mean_acc,
        mean_correct=mean_correct,
        per_cat=per_cat,
        eval_results=eval_results,
        stage_summary=stage_summary,
        config_dump=config_dump,
        package_versions=package_versions,
    )
    (output_dir / "report.txt").write_text("\n".join(lines) + "\n")


def _build_text_report(
    *,
    dataset_name: str,
    run_name: str,
    total: int,
    avg_tokens: int,
    maj_acc: float,
    maj_correct: int,
    mean_acc: float,
    mean_correct: int,
    per_cat: dict[str, Any],
    eval_results: dict[str, Any],
    stage_summary: dict[str, dict[str, Any]],
    config_dump: dict[str, Any],
    package_versions: dict[str, str],
) -> list[str]:
    """Build the lines of the human-readable text report."""
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append(f"EverAlgo Benchmark Report — {dataset_name} / {run_name}")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Total Questions: {total}")
    lines.append(f"Avg Context Tokens: {avg_tokens}")
    lines.append("")

    _append_accuracy_sections(
        lines, total, maj_acc, maj_correct, mean_acc, mean_correct, per_cat, eval_results, config_dump
    )
    _append_stage_summary(lines, stage_summary)
    _append_config(lines, config_dump, package_versions)

    lines.append("=" * 60)
    return lines


def _append_accuracy_sections(
    lines: list[str],
    total: int,
    maj_acc: float,
    maj_correct: int,
    mean_acc: float,
    mean_correct: int,
    per_cat: dict[str, Any],
    eval_results: dict[str, Any],
    config_dump: dict[str, Any],
) -> None:
    sorted_cats = sorted(per_cat.items(), key=lambda kv: kv[0]) if per_cat else []

    lines.append("— Majority-Vote (≥2/3 judges agree) —")
    lines.append(f"Overall: {maj_acc * 100:.2f}% ({maj_correct}/{total})")
    for cat_num, v in sorted_cats:
        label = v.get("label", f"unknown-{cat_num}")
        mc = v.get("majority_correct", v.get("correct", 0))
        t = v.get("total", 0)
        ma = v.get("majority_accuracy", v.get("accuracy", 0.0))
        lines.append(f"  Category {cat_num} ({label:<13}) {ma * 100:>6.2f}%  ({mc:>3}/{t:>3})")
    lines.append("")

    lines.append("— Mean-of-Runs (average of per-judge-run accuracies) —")
    lines.append(f"Overall: {mean_acc * 100:.2f}% ({mean_correct}/{total})")
    for cat_num, v in sorted_cats:
        label = v.get("label", f"unknown-{cat_num}")
        acc = v.get("accuracy", 0.0)
        c = v.get("correct", 0)
        t = v.get("total", 0)
        lines.append(f"  Category {cat_num} ({label:<13}) {acc * 100:>6.2f}%  ({c:>3}/{t:>3})")
    lines.append("")

    lines.append(
        f"Judge runs: {config_dump.get('judge_runs', 'N/A')}, std: {eval_results.get('std_accuracy', 0.0):.4f}"
    )
    lines.append("")


def _append_stage_summary(lines: list[str], stage_summary: dict[str, dict[str, Any]]) -> None:
    if not stage_summary:
        return
    lines.append("Stage Summary:")
    for stage, s in stage_summary.items():
        lines.append(
            f"  {stage:<10} duration={s.get('duration_seconds', 0):.1f}s  "
            f"prompt_tokens={s.get('prompt_tokens', 0)}  "
            f"completion_tokens={s.get('completion_tokens', 0)}"
        )
    lines.append("")


def _append_config(lines: list[str], config_dump: dict[str, Any], package_versions: dict[str, str]) -> None:
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
