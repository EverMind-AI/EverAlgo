"""Tests for report generation."""

import json
from pathlib import Path
from typing import Any, cast

from benchmarks.common.report import generate_reports


def test_generate_reports_writes_both_formats(tmp_path: Path):
    eval_results: dict[str, Any] = cast(
        "dict[str, Any]",
        {
            "total_questions": 1540,
            "correct": 1391,
            "accuracy": 0.9032,
            "per_category": {
                "1": {"label": "single-hop", "correct": 271, "total": 282, "accuracy": 0.9634},
                "4": {"label": "multi-hop", "correct": 774, "total": 841, "accuracy": 0.9203},
            },
            "detailed_results": [],
        },
    )
    stage_summary_data = {
        "extract": {"duration_seconds": 100.0, "prompt_tokens": 1000, "completion_tokens": 200},
        "answer": {"duration_seconds": 5.0, "prompt_tokens": 500, "completion_tokens": 50},
    }
    answers = [{"formatted_context": "x" * 100}]
    generate_reports(
        output_dir=tmp_path,
        eval_results=eval_results,
        stage_summary=stage_summary_data,
        answers=answers,
        dataset_name="locomo",
        run_name="v1",
        config_dump={"extract_model": "openai/gpt-4.1-mini", "answer_model": "openai/gpt-4.1-mini", "judge_runs": 3},
        package_versions={"everalgo-core": "0.1.0", "everalgo-rank": "0.1.0"},
    )

    txt = (tmp_path / "report.txt").read_text()
    # Verify key sections present
    assert "locomo" in txt
    assert "v1" in txt
    assert "90.32%" in txt or "0.9032" in txt
    assert "single-hop" in txt
    assert "0.9634" in txt or "96.34%" in txt
    assert "extract" in txt  # stage name
    assert "everalgo-core" in txt  # package version

    rep = json.loads((tmp_path / "report.json").read_text())
    assert rep["dataset"] == "locomo"
    assert rep["run_name"] == "v1"
    assert rep["majority_accuracy"] == 0.9032
    assert rep["mean_of_runs_accuracy"] == 0.9032
    assert rep["total_questions"] == 1540
    assert rep["majority_correct"] == 1391
    assert rep["mean_of_runs_correct"] == 1391
    # per_category now keyed by category number
    assert rep["per_category"]["1"]["accuracy"] == 0.9634
    assert rep["per_category"]["1"]["label"] == "single-hop"
    assert rep["avg_context_tokens"] > 0
    assert rep["stage_summary"]["extract"]["duration_seconds"] == 100.0
    assert rep["config"]["extract_model"] == "openai/gpt-4.1-mini"
    assert rep["package_versions"]["everalgo-core"] == "0.1.0"


def test_generate_reports_creates_output_dir_if_missing(tmp_path: Path):
    target = tmp_path / "doesnt-exist-yet"
    generate_reports(
        output_dir=target,
        eval_results={"total_questions": 0, "correct": 0, "accuracy": 0.0, "per_category": {}},
        stage_summary={},
        answers=[],
        dataset_name="x",
        run_name="y",
        config_dump={},
        package_versions={},
    )
    assert (target / "report.txt").exists()
    assert (target / "report.json").exists()


def test_generate_reports_handles_empty_results(tmp_path: Path):
    """No questions, no stages, no packages — output should still be valid."""
    generate_reports(
        output_dir=tmp_path,
        eval_results={"total_questions": 0, "correct": 0, "accuracy": 0.0, "per_category": {}},
        stage_summary={},
        answers=[],
        dataset_name="empty",
        run_name="zero",
        config_dump={},
        package_versions={},
    )
    rep = json.loads((tmp_path / "report.json").read_text())
    assert rep["total_questions"] == 0
    assert rep["avg_context_tokens"] == 0
