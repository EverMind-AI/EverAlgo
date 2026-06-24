"""Tests for Stage 3 Enrich — AtomicFact extraction from episodes.

Unit tests cover:
- ``run_enrich_stage`` callable/coroutine guard
- ``_extract_facts_for_episode`` — empty facts raises ValueError (fail-loud policy)
- ``_enrich_one_conversation`` — happy path writes expected JSON
- ``_enrich_one_conversation`` — per-conversation error isolation (writes .error.txt)
- ``run_enrich_stage`` — end-to-end with mocked LLM + embedding clients
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchmarks.common.config import BenchmarkConfig
from benchmarks.common.stages.enrich import (
    _enrich_one_conversation,
    _extract_facts_for_episode,
    run_enrich_stage,
)
from benchmarks.common.stages.serialization import write_json
from benchmarks.common.stages.types import StageContext

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_episode(
    ep_id: str = "0",
    episode_text: str = "Alice went to the store.",
    timestamp: int = 1_700_000_000_000,
    owner_id: str | None = None,
) -> dict[str, Any]:
    """Return a minimal episode dict matching the Extract Base output schema."""
    return {
        "id": ep_id,
        "owner_id": owner_id,
        "memcell_ids": [ep_id],
        "subject": "shopping",
        "episode": episode_text,
        "timestamp": timestamp,
        "embeddings": {"episode": [0.1, 0.2, 0.3], "subject": None},
    }


def _make_fake_atomic_fact(content: str = "Alice went to the store.", timestamp: int = 1_700_000_000_000) -> Any:
    """Return a minimal AtomicFact-like mock."""
    af = MagicMock()
    af.content = content
    af.timestamp = timestamp
    return af


def _make_embedding_client(vector: list[float] | None = None) -> MagicMock:
    """Return a mock EmbeddingClient whose embed() returns one vector per text."""
    vec = vector or [0.1, 0.2, 0.3]
    client = MagicMock()
    # Return a vector for each text in the input list
    client.embed = AsyncMock(side_effect=lambda texts: [[*vec] for _ in texts])
    return client


def _make_llm_mock() -> MagicMock:
    """Return a minimal LLM mock (unused directly; AtomicFactExtractor is patched)."""
    return MagicMock()


def _make_stage_context(tmp_path: Path, *, input_dir: Path | None = None, smoke: bool = False) -> StageContext:
    """Build a minimal StageContext for testing."""
    cfg = BenchmarkConfig()
    mock_services = MagicMock()
    mock_services.embedding = _make_embedding_client()
    return StageContext(
        config=cfg,
        services=mock_services,
        dataset=MagicMock(),
        input_dir=input_dir or tmp_path / "input",
        output_dir=tmp_path / "output",
        smoke=smoke,
    )


# ---------------------------------------------------------------------------
# Structural check
# ---------------------------------------------------------------------------


def test_run_enrich_stage_is_coroutine() -> None:
    """Structural check: ``run_enrich_stage`` is an async function."""
    import inspect

    assert inspect.iscoroutinefunction(run_enrich_stage)


# ---------------------------------------------------------------------------
# Unit: _extract_facts_for_episode — empty-facts raises ValueError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_facts_raises_on_zero_facts() -> None:
    """If AtomicFactExtractor returns [], ValueError must be raised (fail-loud)."""
    episode = _make_episode("0", episode_text="some episode text")
    llm = _make_llm_mock()
    embedding_client = _make_embedding_client()

    with (
        patch(
            "benchmarks.common.stages.enrich.AtomicFactExtractor",
            return_value=MagicMock(aextract_from_text=AsyncMock(return_value=[])),
        ),
        pytest.raises(ValueError, match="0 facts"),
    ):
        await _extract_facts_for_episode(episode, 0, llm, embedding_client, max_attempts=1)


@pytest.mark.asyncio
async def test_extract_facts_happy_path() -> None:
    """Happy path: 2 facts extracted, embedded, serialised correctly."""
    episode = _make_episode("ep5", episode_text="episode body", timestamp=1_700_000_001_000, owner_id="user_1")
    llm = _make_llm_mock()
    vec = [0.5, 0.6, 0.7]
    embedding_client = _make_embedding_client(vector=vec)

    fake_facts = [
        _make_fake_atomic_fact("Alice went shopping.", 1_700_000_001_000),
        _make_fake_atomic_fact("Alice bought apples.", 1_700_000_001_000),
    ]

    with patch(
        "benchmarks.common.stages.enrich.AtomicFactExtractor",
        return_value=MagicMock(aextract_from_text=AsyncMock(return_value=fake_facts)),
    ):
        result = await _extract_facts_for_episode(episode, 5, llm, embedding_client, max_attempts=1)

    assert len(result) == 2
    # IDs are local (relative to episode) — caller re-indexes globally
    assert result[0]["id"] == "0"
    assert result[1]["id"] == "1"
    for fact in result:
        assert fact["episode_id"] == "ep5"
        assert fact["owner_id"] == "user_1"
        assert fact["timestamp"] == 1_700_000_001_000
        assert isinstance(fact["embeddings"], list)
        assert len(fact["embeddings"]) > 0
        assert "content" in fact


@pytest.mark.asyncio
async def test_extract_facts_embeds_all_facts_in_one_batch() -> None:
    """All facts for one episode must be embedded in a single embed() call."""
    episode = _make_episode("0", episode_text="multi-fact episode")
    llm = _make_llm_mock()
    embedding_client = _make_embedding_client()
    n_facts = 4
    fake_facts = [_make_fake_atomic_fact(f"Fact {i}.") for i in range(n_facts)]

    with patch(
        "benchmarks.common.stages.enrich.AtomicFactExtractor",
        return_value=MagicMock(aextract_from_text=AsyncMock(return_value=fake_facts)),
    ):
        await _extract_facts_for_episode(episode, 0, llm, embedding_client, max_attempts=1)

    # embed() must be called exactly once with all 4 texts
    embedding_client.embed.assert_awaited_once()
    call_args = embedding_client.embed.call_args[0][0]
    assert len(call_args) == n_facts


# ---------------------------------------------------------------------------
# Unit: _enrich_one_conversation — happy path + error isolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enrich_one_conversation_writes_json(tmp_path: Path) -> None:
    """Happy path: writes ``atomic_facts_conv_<i>.json`` with globally re-indexed IDs."""
    episodes = [_make_episode("0"), _make_episode("1")]
    llm = _make_llm_mock()
    embedding_client = _make_embedding_client()
    semaphore = asyncio.Semaphore(20)

    # Each episode returns 2 facts
    fake_facts_per_ep = [
        _make_fake_atomic_fact("Fact A."),
        _make_fake_atomic_fact("Fact B."),
    ]

    with patch(
        "benchmarks.common.stages.enrich.AtomicFactExtractor",
        return_value=MagicMock(aextract_from_text=AsyncMock(return_value=fake_facts_per_ep)),
    ):
        ok = await _enrich_one_conversation(
            3, episodes, llm, embedding_client, output_dir=tmp_path, semaphore=semaphore, max_attempts=1
        )

    assert ok is True
    out_file = tmp_path / "atomic_facts_conv_3.json"
    assert out_file.exists()

    facts: list[dict[str, Any]] = json.loads(out_file.read_text())
    assert len(facts) == 4  # 2 episodes x 2 facts each

    # Global IDs must be 0, 1, 2, 3 in order
    ids = [f["id"] for f in facts]
    assert ids == ["0", "1", "2", "3"]

    # Episode grouping: first 2 facts belong to episode "0", next 2 to "1"
    assert facts[0]["episode_id"] == "0"
    assert facts[1]["episode_id"] == "0"
    assert facts[2]["episode_id"] == "1"
    assert facts[3]["episode_id"] == "1"


@pytest.mark.asyncio
async def test_enrich_one_conversation_isolates_errors(tmp_path: Path) -> None:
    """On exception, returns False and writes ``.error.txt`` — no crash, no partial file."""
    episodes = [_make_episode("0")]
    llm = _make_llm_mock()
    embedding_client = _make_embedding_client()
    semaphore = asyncio.Semaphore(20)

    with patch(
        "benchmarks.common.stages.enrich.AtomicFactExtractor",
        return_value=MagicMock(aextract_from_text=AsyncMock(side_effect=RuntimeError("LLM exploded"))),
    ):
        ok = await _enrich_one_conversation(
            7, episodes, llm, embedding_client, output_dir=tmp_path, semaphore=semaphore, max_attempts=1
        )

    assert ok is False
    err_file = tmp_path / "atomic_facts_conv_7.error.txt"
    assert err_file.exists()
    assert "LLM exploded" in err_file.read_text()


# ---------------------------------------------------------------------------
# Integration: run_enrich_stage — end-to-end with file I/O
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_enrich_stage_writes_atomic_facts(tmp_path: Path) -> None:
    """End-to-end: stage reads episode files, writes atomic_facts_conv_<i>.json."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    # Write two episode files (convs 0 and 1)
    for i in range(2):
        episodes = [
            _make_episode(str(j), episode_text=f"episode {j} text", timestamp=1_700_000_000_000 + j) for j in range(2)
        ]
        write_json(input_dir / f"episodes_conv_{i}.json", episodes)

    cfg = BenchmarkConfig()
    mock_services = MagicMock()
    mock_services.embedding = _make_embedding_client()
    ctx = StageContext(
        config=cfg,
        services=mock_services,
        dataset=MagicMock(),
        input_dir=input_dir,
        output_dir=output_dir,
        smoke=False,
    )

    fake_facts = [_make_fake_atomic_fact("Some fact.")]

    with (
        patch("benchmarks.common.stages.enrich.build_llm_client", return_value=_make_llm_mock()),
        patch(
            "benchmarks.common.stages.enrich.AtomicFactExtractor",
            return_value=MagicMock(aextract_from_text=AsyncMock(return_value=fake_facts)),
        ),
    ):
        stats = await run_enrich_stage(ctx)

    assert stats.stage_name == "enrich"
    assert stats.success == 2
    assert stats.failed == 0
    assert stats.duration_seconds > 0

    for i in range(2):
        out_file = output_dir / f"atomic_facts_conv_{i}.json"
        assert out_file.exists(), f"missing {out_file}"
        facts = json.loads(out_file.read_text())
        assert isinstance(facts, list)
        assert len(facts) >= 1
        for f in facts:
            assert "id" in f
            assert "episode_id" in f
            assert "content" in f
            assert "embeddings" in f
            assert isinstance(f["embeddings"], list)
            assert len(f["embeddings"]) > 0


@pytest.mark.asyncio
async def test_run_enrich_stage_smoke_limits_convs(tmp_path: Path) -> None:
    """Smoke mode: only the first ``smoke_conv_limit`` conversation files are processed."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    for i in range(3):
        episodes = [_make_episode("0")]
        write_json(input_dir / f"episodes_conv_{i}.json", episodes)

    cfg = BenchmarkConfig()
    mock_services = MagicMock()
    mock_services.embedding = _make_embedding_client()
    ctx = StageContext(
        config=cfg,
        services=mock_services,
        dataset=MagicMock(),
        input_dir=input_dir,
        output_dir=output_dir,
        smoke=True,
        smoke_conv_limit=1,
    )

    fake_facts = [_make_fake_atomic_fact("Some fact.")]

    with (
        patch("benchmarks.common.stages.enrich.build_llm_client", return_value=_make_llm_mock()),
        patch(
            "benchmarks.common.stages.enrich.AtomicFactExtractor",
            return_value=MagicMock(aextract_from_text=AsyncMock(return_value=fake_facts)),
        ),
    ):
        stats = await run_enrich_stage(ctx)

    assert stats.success == 1  # only 1 conversation processed
    assert stats.failed == 0
    assert (output_dir / "atomic_facts_conv_0.json").exists()
    assert not (output_dir / "atomic_facts_conv_1.json").exists()
    assert not (output_dir / "atomic_facts_conv_2.json").exists()


@pytest.mark.asyncio
async def test_run_enrich_stage_no_input_files(tmp_path: Path) -> None:
    """Stage must succeed gracefully with 0 episode files (no-op)."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    cfg = BenchmarkConfig()
    mock_services = MagicMock()
    mock_services.embedding = _make_embedding_client()
    ctx = StageContext(
        config=cfg,
        services=mock_services,
        dataset=MagicMock(),
        input_dir=input_dir,
        output_dir=output_dir,
    )

    with patch("benchmarks.common.stages.enrich.build_llm_client", return_value=_make_llm_mock()):
        stats = await run_enrich_stage(ctx)

    assert stats.success == 0
    assert stats.failed == 0


@pytest.mark.asyncio
async def test_run_enrich_stage_counts_failed_conversations(tmp_path: Path) -> None:
    """Failed conversation must increment ``failed`` counter, not ``success``."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    episodes = [_make_episode("0")]
    write_json(input_dir / "episodes_conv_0.json", episodes)

    cfg = BenchmarkConfig()
    mock_services = MagicMock()
    mock_services.embedding = _make_embedding_client()
    ctx = StageContext(
        config=cfg,
        services=mock_services,
        dataset=MagicMock(),
        input_dir=input_dir,
        output_dir=output_dir,
    )

    with (
        patch("benchmarks.common.stages.enrich.build_llm_client", return_value=_make_llm_mock()),
        patch(
            "benchmarks.common.stages.enrich.AtomicFactExtractor",
            return_value=MagicMock(aextract_from_text=AsyncMock(side_effect=RuntimeError("boom"))),
        ),
    ):
        stats = await run_enrich_stage(ctx)

    assert stats.success == 0
    assert stats.failed == 1
