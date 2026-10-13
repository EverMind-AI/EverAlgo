"""Tests for Stage 6 answer generation (entity-split data model)."""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch

from benchmarks.common.metrics import estimate_tokens
from benchmarks.common.stages.answer import _AnswerReply, _extract_final_answer, run_answer_stage
from everalgo.llm import LLMOutputError


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

    raw = "## FINAL ANSWER:\nAlice greeted Bob first.\n"
    chat_mock = AsyncMock(return_value=json.dumps({"text": raw}))
    monkeypatch.setattr(services.llm, "achat", chat_mock)

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
    assert item["raw_response"] == raw
    assert chat_mock.call_args.kwargs["response_format"] is _AnswerReply
    assert item["prompt_tokens"] == estimate_tokens(chat_mock.call_args.args[0][0].content) > 0
    assert item["completion_tokens"] == estimate_tokens(raw) > 0
    assert stats.prompt_tokens == item["prompt_tokens"]

    # The answer prompt is rendered inside the stage, so this is the only place its call site is
    # observable: a revert to `.format` would leave a literal `{context}` in what the model saw.
    sent = chat_mock.call_args.args[0][0].content
    assert re.search(r"\{\{?[A-Za-z_][A-Za-z0-9_]*\}\}?", sent) is None, sent
    assert "Alice said hi to Bob first" in sent


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("achat", "error", "match"),
    [
        pytest.param(AsyncMock(side_effect=RuntimeError("LLM down")), RuntimeError, "LLM down", id="client-failure"),
        pytest.param(
            AsyncMock(return_value="FINAL ANSWER: not wrapped in JSON"),
            LLMOutputError,
            "answer reply is not a valid _AnswerReply",
            id="invalid-reply",
        ),
    ],
)
async def test_run_answer_stage_raises_when_llm_keeps_failing(
    tmp_path: Path, monkeypatch: MonkeyPatch, achat: AsyncMock, error: type[Exception], match: str
) -> None:
    """Fail-loud: a persistent client failure or invalid reply exhausts retries and aborts the stage."""
    from benchmarks.common.config import BenchmarkConfig
    from benchmarks.common.services import Services
    from benchmarks.common.stages.types import StageContext
    from benchmarks.datasets.locomo.loader import LocomoDataset

    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test")
    monkeypatch.setattr("benchmarks.common.stages.answer.asyncio.sleep", AsyncMock())

    cfg = BenchmarkConfig()
    services = Services.from_config(cfg)
    services.llm.achat = achat  # type: ignore[method-assign]

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
    with pytest.raises(error, match=match):
        await run_answer_stage(ctx)
    assert achat.await_count == cfg.llm_max_retries
    assert not (tmp_path / "stage6_answer" / "answers.json").exists()
