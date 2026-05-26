"""Tests for Stage 5 evaluate stage."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch

from benchmarks.common.stages.evaluate import run_evaluate_stage


def test_run_evaluate_stage_is_async() -> None:
    assert inspect.iscoroutinefunction(run_evaluate_stage)


@pytest.mark.asyncio
async def test_run_evaluate_stage_writes_eval_results_json(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    from benchmarks.common.config import BenchmarkConfig
    from benchmarks.common.services import Services
    from benchmarks.common.stages.types import StageContext
    from benchmarks.datasets.locomo.loader import LocomoDataset

    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test")

    cfg = BenchmarkConfig()
    services = Services.from_config(cfg)
    # Mock judge LLM to always say correct.
    # allm_judge reads resp.usage.prompt_tokens — mock must expose .usage with the right shape.
    mock_usage = MagicMock(prompt_tokens=10, completion_tokens=5)
    monkeypatch.setattr(
        services.llm,
        "chat",
        AsyncMock(
            return_value=MagicMock(
                content='{"label": "CORRECT", "reasoning": "ok"}',
                usage=mock_usage,
            )
        ),
    )

    # Prepare stage 4 output (answers.json)
    stage4_dir: Path = tmp_path / "stage4_answer"
    stage4_dir.mkdir()
    answers: list[dict[str, Any]] = [
        {
            "question_id": "q1",
            "question": "?",
            "answer": "A1",
            "golden_answer": "G1",
            "category": "1",
            "conversation_id": "c0",
            "formatted_context": "...",
            "raw_response": "...",
        },
        {
            "question_id": "q2",
            "question": "?",
            "answer": "A2",
            "golden_answer": "G2",
            "category": "2",
            "conversation_id": "c0",
            "formatted_context": "...",
            "raw_response": "...",
        },
        # Category 5 should be filtered out
        {
            "question_id": "q3",
            "question": "?",
            "answer": "A3",
            "golden_answer": "G3",
            "category": "5",
            "conversation_id": "c0",
            "formatted_context": "...",
            "raw_response": "...",
        },
    ]
    (stage4_dir / "answers.json").write_text(json.dumps(answers))

    fixture = Path(__file__).parent / "fixtures" / "locomo_mini.json"
    ctx = StageContext(
        config=cfg,
        services=services,
        dataset=LocomoDataset(data_path=fixture),
        input_dir=stage4_dir,
        output_dir=tmp_path / "stage5_evaluate",
    )
    stats = await run_evaluate_stage(ctx)
    assert stats.stage_name == "evaluate"

    out: Path = tmp_path / "stage5_evaluate" / "eval_results.json"
    assert out.exists()
    data: dict[str, Any] = json.loads(out.read_text())
    # Adversarial cat 5 filtered out
    assert data["total_questions"] == 2
    # Both correct (mock always returns is_correct=true)
    assert data["correct"] == 2
    assert data["accuracy"] == 1.0
    # Per-category breakdown — now keyed by category number
    assert "1" in data["per_category"]
    assert "2" in data["per_category"]
    assert data["per_category"]["1"]["label"] == "single-hop"
    assert data["per_category"]["2"]["label"] == "temporal"
    assert data["per_category"]["1"]["total"] == 1
    assert data["per_category"]["2"]["total"] == 1
    # Detailed results include all evaluated items
    assert len(data["detailed_results"]) == 2
    assert all(item["is_correct"] for item in data["detailed_results"])


@pytest.mark.asyncio
async def test_run_evaluate_stage_raises_when_judge_keeps_failing(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Fail-loud: persistent judge LLM exceptions exhaust retries and abort the stage.

    Silent ``is_correct=False`` placeholders would poison the headline accuracy
    with judge outages; the new contract surfaces the failure instead.
    """
    from benchmarks.common.config import BenchmarkConfig
    from benchmarks.common.services import Services
    from benchmarks.common.stages.types import StageContext
    from benchmarks.datasets.locomo.loader import LocomoDataset

    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test")
    # Collapse the per-attempt backoff so the test does not wait 1+2+4+8 = 15 s.
    monkeypatch.setattr("everalgo.testing.judge.asyncio.sleep", AsyncMock())

    cfg = BenchmarkConfig()
    services = Services.from_config(cfg)
    services.llm.chat = AsyncMock(side_effect=RuntimeError("judge LLM down"))  # type: ignore[method-assign]

    stage4_dir: Path = tmp_path / "stage4_answer"
    stage4_dir.mkdir()
    (stage4_dir / "answers.json").write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "question": "?",
                    "answer": "A",
                    "golden_answer": "G",
                    "category": "1",
                    "conversation_id": "c0",
                    "formatted_context": "",
                    "raw_response": "",
                }
            ]
        )
    )

    fixture = Path(__file__).parent / "fixtures" / "locomo_mini.json"
    ctx = StageContext(
        config=cfg,
        services=services,
        dataset=LocomoDataset(data_path=fixture),
        input_dir=stage4_dir,
        output_dir=tmp_path / "stage5_evaluate",
    )
    with pytest.raises(RuntimeError, match="judge LLM down"):
        await run_evaluate_stage(ctx)
    assert not (tmp_path / "stage5_evaluate" / "eval_results.json").exists()
