"""Tests for Stage 4 answer generation."""

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
    # rsplit on last marker: trailing text after the answer is preserved (no
    # blank-line or ##-heading truncation in the 93-branch logic).
    raw = """## STEP 1: ...
some reasoning

## STEP 7: FINAL ANSWER
Alice ate pizza on Tuesday.

extra text after"""
    assert _extract_final_answer(raw) == "Alice ate pizza on Tuesday.\n\nextra text after"


def test_extract_final_answer_step7_priority_over_final_answer_colon() -> None:
    # When both "## STEP 7: FINAL ANSWER" and an earlier "FINAL ANSWER:" appear,
    # the STEP 7 branch fires first (highest priority), taking everything after
    # the last "## STEP 7: FINAL ANSWER" occurrence.
    raw = (
        "## STEP 3: CROSS-MEMORY LINKING & INFERENCE\n"
        "- Connections found: see FINAL ANSWER: section for summary\n\n"
        "## STEP 7: FINAL ANSWER\n"
        "The real answer is 42."
    )
    assert _extract_final_answer(raw) == "The real answer is 42."


def test_extract_final_answer_rsplit_takes_last_marker() -> None:
    # rsplit behaviour: when "FINAL ANSWER:" appears twice, we get text after the
    # LAST occurrence (the actual answer), not the first (reasoning chatter).
    raw = "I note that FINAL ANSWER: might be X.\n\n## STEP 7: FINAL ANSWER\nThe definitive answer is Y."
    # "## STEP 7: FINAL ANSWER" fires (highest priority), so result is after that.
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
    """End-to-end: stage1 memcells + stage3 search_results → answers.json."""
    from benchmarks.common.config import BenchmarkConfig
    from benchmarks.common.services import Services
    from benchmarks.common.stages.types import StageContext
    from benchmarks.datasets.locomo.loader import LocomoDataset

    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test")

    cfg = BenchmarkConfig()
    services = Services.from_config(cfg)

    # Mock LLM to return a structured response with FINAL ANSWER
    monkeypatch.setattr(
        services.llm,
        "chat",
        AsyncMock(
            return_value=MagicMock(
                content="## FINAL ANSWER:\nAlice greeted Bob first.\n",
                prompt_tokens=100,
                completion_tokens=20,
            )
        ),
    )

    # Prepare stage1 output (memcells — EverAlgo-native schema)
    stage1_dir: Path = tmp_path / "stage1_extract"
    stage1_dir.mkdir()
    memcell: dict[str, Any] = {
        "id": "0",
        "timestamp": 1683525360000,
        "items": [
            {
                "id": "m1",
                "role": "user",
                "content": "Alice said hi to Bob first",
                "timestamp": 1683525360000,
                "sender_id": "u_alice",
                "sender_name": "Alice",
            }
        ],
        "episode": {
            "subject": "greeting",
            "content": "Alice said hi to Bob first",
        },
        "atomic_facts": [{"fact": "Alice greeted Bob first"}],
    }
    (stage1_dir / "memcells_conv_0.json").write_text(json.dumps([memcell]))

    # Prepare stage3 output (search_results — new schema)
    stage3_dir: Path = tmp_path / "stage3_search"
    stage3_dir.mkdir()
    search_results: dict[str, Any] = {
        "locomo_exp_user_0": [
            {
                "question_id": "locomo_exp_user_0_qa0",
                "query": "Who greeted whom first?",
                "memcell_ids": ["0"],
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
        output_dir=tmp_path / "stage4_answer",
    )
    stats = await run_answer_stage(ctx)
    assert stats.stage_name == "answer"
    assert stats.success >= 1

    out: Path = tmp_path / "stage4_answer" / "answers.json"
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
async def test_run_answer_stage_handles_llm_error_per_question(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """When LLM raises, the stage writes error item but continues."""
    from benchmarks.common.config import BenchmarkConfig
    from benchmarks.common.services import Services
    from benchmarks.common.stages.types import StageContext
    from benchmarks.datasets.locomo.loader import LocomoDataset

    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test")

    cfg = BenchmarkConfig()
    services = Services.from_config(cfg)
    services.llm.chat = AsyncMock(side_effect=RuntimeError("LLM down"))  # type: ignore[method-assign]

    # Minimal stage 1/3 outputs (EverAlgo-native schema)
    stage1_dir: Path = tmp_path / "stage1_extract"
    stage1_dir.mkdir()
    (stage1_dir / "memcells_conv_0.json").write_text(
        json.dumps(
            [
                {
                    "id": "0",
                    "timestamp": 0,
                    "items": [],
                    "episode": {"subject": "x", "content": "y"},
                    "atomic_facts": [],
                }
            ]
        )
    )

    stage3_dir: Path = tmp_path / "stage3_search"
    stage3_dir.mkdir()
    (stage3_dir / "search_results.json").write_text(
        json.dumps(
            {
                "locomo_exp_user_0": [
                    {
                        "question_id": "q1",
                        "query": "?",
                        "memcell_ids": ["0"],
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
        output_dir=tmp_path / "stage4_answer",
    )
    stats = await run_answer_stage(ctx)
    assert stats.failed >= 1
    result_data: list[dict[str, Any]] = json.loads((tmp_path / "stage4_answer" / "answers.json").read_text())
    assert result_data[0].get("error") is True
