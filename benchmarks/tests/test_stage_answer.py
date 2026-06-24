"""Tests for Stage 6 answer generation (entity-split data model)."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch

from benchmarks.common.stages.answer import _extract_final_answer, run_answer_stage


def test_run_answer_stage_is_async() -> None:
    assert inspect.iscoroutinefunction(run_answer_stage)


def test_extract_final_answer_finds_section() -> None:
    raw = """## STEP 1: ...
some reasoning

## STEP 7: FINAL ANSWER
Alice ate pizza on Tuesday.

extra text after"""
    assert _extract_final_answer(raw) == "Alice ate pizza on Tuesday.\n\nextra text after"


def test_extract_final_answer_step7_priority_over_final_answer_colon() -> None:
    raw = (
        "## STEP 3: CROSS-MEMORY LINKING & INFERENCE\n"
        "- Connections found: see FINAL ANSWER: section for summary\n\n"
        "## STEP 7: FINAL ANSWER\n"
        "The real answer is 42."
    )
    assert _extract_final_answer(raw) == "The real answer is 42."


def test_extract_final_answer_rsplit_takes_last_marker() -> None:
    raw = "I note that FINAL ANSWER: might be X.\n\n## STEP 7: FINAL ANSWER\nThe definitive answer is Y."
    assert _extract_final_answer(raw) == "The definitive answer is Y."


def test_extract_final_answer_strips_whitespace() -> None:
    raw = "FINAL ANSWER:\n   indented answer   \n\n"
    assert _extract_final_answer(raw) == "indented answer"


def test_extract_final_answer_falls_back_to_full_response() -> None:
    raw = "Just a plain answer with no marker."
    assert _extract_final_answer(raw) == "Just a plain answer with no marker."


def test_extract_final_answer_at_end_of_string() -> None:
    raw = "FINAL ANSWER: The answer is 42"
    assert _extract_final_answer(raw) == "The answer is 42"


@pytest.mark.asyncio
async def test_run_answer_stage_writes_answers_json(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """End-to-end: stage1 episodes + stage3 search_results → answers.json."""
    from benchmarks.common.config import BenchmarkConfig
    from benchmarks.common.services import Services
    from benchmarks.common.stages.types import StageContext
    from benchmarks.datasets.locomo.loader import LocomoDataset

    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test")

    cfg = BenchmarkConfig()
    services = Services.from_config(cfg)

    mock_usage = MagicMock(prompt_tokens=100, completion_tokens=20)
    monkeypatch.setattr(
        services.llm,
        "chat",
        AsyncMock(
            return_value=MagicMock(
                content="## FINAL ANSWER:\nAlice greeted Bob first.\n",
                usage=mock_usage,
            )
        ),
    )

    # Prepare stage1 output (entity-split: episodes file, not monolithic memcells)
    stage1_dir: Path = tmp_path / "stage3_enrich"
    stage1_dir.mkdir()
    episode: dict[str, Any] = {
        "id": "0",
        "owner_id": None,
        "memcell_ids": ["0"],
        "subject": "greeting",
        "episode": "Alice said hi to Bob first",
        "timestamp": 1683525360000,
        "embeddings": {"episode": [0.1, 0.2], "subject": [0.3, 0.4]},
    }
    (stage1_dir / "episodes_conv_0.json").write_text(json.dumps([episode]))

    # Prepare stage3 output (search_results)
    stage3_dir: Path = tmp_path / "stage5_search"
    stage3_dir.mkdir()
    search_results: dict[str, Any] = {
        "locomo_exp_user_0": [
            {
                "question_id": "locomo_exp_user_0_qa0",
                "query": "Who greeted whom first?",
                "members": ["0"],
                "original_qa": {
                    "question_id": "locomo_exp_user_0_qa0",
                    "conv_id": "locomo_exp_user_0",
                    "question": "Who greeted whom first?",
                    "golden_answer": "Alice greeted Bob first.",
                    "category": "1",
                },
                "retrieval_metadata": {},
            }
        ]
    }
    (stage3_dir / "search_results.json").write_text(json.dumps(search_results))

    fixture = Path(__file__).parent / "fixtures" / "locomo_mini.json"
    ctx = StageContext(
        config=cfg,
        services=services,
        dataset=LocomoDataset(data_path=fixture),
        input_dir=stage3_dir,
        output_dir=tmp_path / "stage6_answer",
    )
    stats = await run_answer_stage(ctx)
    assert stats.stage_name == "answer"
    assert stats.success >= 1

    out: Path = tmp_path / "stage6_answer" / "answers.json"
    assert out.exists()
    data: list[dict[str, Any]] = json.loads(out.read_text())
    assert isinstance(data, list)
    assert len(data) >= 1
    item: dict[str, Any] = data[0]
    for field in (
        "question_id",
        "question",
        "answer",
        "golden_answer",
        "category",
        "conversation_id",
        "formatted_context",
        "raw_response",
    ):
        assert field in item
    assert item["answer"] == "Alice greeted Bob first."
    assert "## FINAL ANSWER:" in item["raw_response"]
    assert item["prompt_tokens"] == 100
    assert item["completion_tokens"] == 20


@pytest.mark.asyncio
async def test_run_answer_stage_raises_when_llm_keeps_failing(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Fail-loud: persistent LLM exceptions exhaust retries and abort the stage."""
    from benchmarks.common.config import BenchmarkConfig
    from benchmarks.common.services import Services
    from benchmarks.common.stages.types import StageContext
    from benchmarks.datasets.locomo.loader import LocomoDataset

    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test")
    monkeypatch.setattr("benchmarks.common.stages.answer.asyncio.sleep", AsyncMock())

    cfg = BenchmarkConfig()
    services = Services.from_config(cfg)
    services.llm.chat = AsyncMock(side_effect=RuntimeError("LLM down"))  # type: ignore[method-assign]

    # Minimal stage 1 output (entity-split: episodes)
    stage1_dir: Path = tmp_path / "stage3_enrich"
    stage1_dir.mkdir()
    (stage1_dir / "episodes_conv_0.json").write_text(
        json.dumps(
            [
                {
                    "id": "0",
                    "owner_id": None,
                    "memcell_ids": ["0"],
                    "subject": "x",
                    "episode": "y",
                    "timestamp": 0,
                    "embeddings": {"episode": None, "subject": None},
                }
            ]
        )
    )

    stage3_dir: Path = tmp_path / "stage5_search"
    stage3_dir.mkdir()
    (stage3_dir / "search_results.json").write_text(
        json.dumps(
            {
                "locomo_exp_user_0": [
                    {
                        "question_id": "q1",
                        "query": "?",
                        "members": ["0"],
                        "original_qa": {
                            "question_id": "q1",
                            "conv_id": "locomo_exp_user_0",
                            "question": "?",
                            "golden_answer": "A",
                            "category": "1",
                        },
                        "retrieval_metadata": {},
                    }
                ]
            }
        )
    )

    fixture = Path(__file__).parent / "fixtures" / "locomo_mini.json"
    ctx = StageContext(
        config=cfg,
        services=services,
        dataset=LocomoDataset(data_path=fixture),
        input_dir=stage3_dir,
        output_dir=tmp_path / "stage6_answer",
    )
    with pytest.raises(RuntimeError, match="LLM down"):
        await run_answer_stage(ctx)
    assert not (tmp_path / "stage6_answer" / "answers.json").exists()
