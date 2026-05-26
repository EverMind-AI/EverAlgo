"""Tests for Stage 1 MemCell extraction.

Most tests are skipped without API credentials (integration). A structural
test verifies the stage entry point exists and imports. Unit tests cover the
clustering helpers with mocked embedding and operator calls.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from benchmarks.common.config import BenchmarkConfig
from benchmarks.common.stages.extract import (
    _cluster_one_memcell,
    _detect_all_boundaries,
    _run_clustering_pass,
    _serialize_cluster_file,
    run_extract_stage,
)
from benchmarks.common.stages.types import StageContext
from everalgo.boundary import DetectionResult
from everalgo.clustering.state import Cluster

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_embedding_client(vector: list[float] | None = None) -> MagicMock:  # pyright: ignore[reportUnusedFunction]
    """Return a mock EmbeddingClient whose embed() returns a single vector."""
    vec = vector or [0.1, 0.2, 0.3]
    client = MagicMock()
    client.embed = AsyncMock(return_value=[vec])
    return client


def _make_memcell(
    mc_id: str = "0",
    timestamp: int = 1_000_000,
    content: str = "hello world",
    content_embeddings: list[float] | None = None,
) -> dict[str, Any]:
    """Return a minimal memcell dict matching the stage 1 output schema."""
    return {
        "id": mc_id,
        "timestamp": timestamp,
        "items": [],
        "episode": {"subject": "test", "summary": "test", "content": content, "content_embeddings": content_embeddings},
        "atomic_facts": {
            "time": "",
            "timestamp": 0,
            "atomic_fact": [],
            "fact_embeddings": [],
        },
    }


def test_run_extract_stage_callable():
    """Structural check: function imports and is async."""
    import inspect

    assert inspect.iscoroutinefunction(run_extract_stage)


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="requires OPENROUTER_API_KEY for real LLM call",
)
@pytest.mark.asyncio
async def test_extract_writes_memcell_json_for_mini_fixture(tmp_path: Path) -> None:
    """End-to-end: run extract on 1-conv fixture; verify output shape."""
    import json

    from benchmarks.common.services import Services
    from benchmarks.datasets.locomo.loader import LocomoDataset

    fixture = Path(__file__).parent / "fixtures" / "locomo_mini.json"
    dataset = LocomoDataset(data_path=fixture)
    cfg = BenchmarkConfig()
    ctx = StageContext(
        config=cfg,
        services=Services.from_config(cfg),
        dataset=dataset,
        input_dir=tmp_path,
        output_dir=tmp_path / "stage1_extract",
        smoke=True,
    )
    stats = await run_extract_stage(ctx)
    assert stats.stage_name == "extract"
    assert stats.success >= 1
    assert stats.duration_seconds > 0

    out_file = tmp_path / "stage1_extract" / "memcells_conv_0.json"
    assert out_file.exists()
    data: list[Any] = json.loads(out_file.read_text())
    assert isinstance(data, list)
    assert len(data) >= 1
    mc: dict[str, Any] = data[0]
    # Required EverAlgo-native fields
    for required in (
        "id",
        "timestamp",
        "items",
        "episode",
        "atomic_facts",
    ):
        assert required in mc, f"missing field: {required}"
    # Verify removed multi-tenant fields are absent
    assert "event_id" not in mc
    assert "group_id" not in mc
    assert "participants" not in mc
    assert "sender_ids" not in mc
    assert "original_data" not in mc
    # id is session-local sequence string
    assert mc["id"] == "0"
    # episode is a nested dict with subject + content
    assert isinstance(mc["episode"], dict)
    assert "subject" in mc["episode"]
    assert "content" in mc["episode"]
    # atomic_facts is a dict {"time", "timestamp", "atomic_fact": list[str], "fact_embeddings": list[list[float]]}
    assert isinstance(mc["atomic_facts"], dict)
    assert "atomic_fact" in mc["atomic_facts"]
    assert isinstance(mc["atomic_facts"]["atomic_fact"], list)


# ---------------------------------------------------------------------------
# Clustering unit tests (no API credentials required)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cluster_one_memcell_mints_first_cluster() -> None:
    """First memcell in an empty list must always create cluster_0."""
    mc = _make_memcell("mc0", timestamp=1000, content="first episode", content_embeddings=[1.0, 0.0, 0.0])

    result = await _cluster_one_memcell(mc, [], threshold=0.70, time_window_days=7.0)

    assert len(result) == 1
    assert result[0].id == "cluster_0"
    assert result[0].members == ["mc0"]
    assert result[0].last_ts == 1000


@pytest.mark.asyncio
async def test_cluster_one_memcell_merges_into_existing() -> None:
    """A memcell similar to an existing cluster must merge, not mint a new one."""
    # Both vectors are identical → cosine similarity == 1.0, well above threshold.
    existing = Cluster(
        id="cluster_0",
        centroid=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        count=1,
        last_ts=500,
        members=["mc0"],
        preview=["old preview"],
    )
    mc = _make_memcell("mc1", timestamp=1000, content="similar episode", content_embeddings=[1.0, 0.0, 0.0])

    result = await _cluster_one_memcell(mc, [existing], threshold=0.70, time_window_days=7.0)

    # Must still have exactly one cluster (merged, not appended).
    assert len(result) == 1
    assert result[0].id == "cluster_0"
    assert "mc1" in result[0].members
    assert result[0].count == 2


@pytest.mark.asyncio
async def test_cluster_one_memcell_raises_on_missing_embeddings() -> None:
    """MemCell with no content_embeddings must raise ValueError (fail-loud)."""
    mc = _make_memcell("mc0", content="some episode", content_embeddings=None)

    with pytest.raises(ValueError, match="content_embeddings is missing"):
        await _cluster_one_memcell(mc, [], threshold=0.70, time_window_days=7.0)


@pytest.mark.asyncio
async def test_cluster_one_memcell_appends_new_cluster_when_dissimilar() -> None:
    """A dissimilar memcell must mint a second cluster, not overwrite the first."""
    # Orthogonal vectors → cosine == 0.0, below threshold.
    existing = Cluster(
        id="cluster_0",
        centroid=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        count=1,
        last_ts=0,
        members=["mc0"],
        preview=["old"],
    )
    mc = _make_memcell("mc1", timestamp=0, content="orthogonal topic", content_embeddings=[0.0, 1.0, 0.0])

    result = await _cluster_one_memcell(mc, [existing], threshold=0.70, time_window_days=7.0)

    assert len(result) == 2
    ids = {c.id for c in result}
    assert ids == {"cluster_0", "cluster_1"}


def test_serialize_cluster_file_shape() -> None:
    """Output dict must have 'clusters' list and 'memcell_to_cluster' map."""
    clusters = [
        Cluster(
            id="cluster_0",
            centroid=np.array([0.5, 0.5], dtype=np.float32),
            count=2,
            last_ts=2000,
            members=["a", "b"],
            preview=["preview text"],
        )
    ]
    out = _serialize_cluster_file(clusters)

    assert "clusters" in out
    assert "memcell_to_cluster" in out

    c = out["clusters"][0]
    assert c["id"] == "cluster_0"
    assert isinstance(c["centroid"], list)  # tolist() → plain Python floats
    assert all(isinstance(v, float) for v in c["centroid"])
    assert c["count"] == 2
    assert c["last_ts"] == 2000
    assert c["members"] == ["a", "b"]

    m2c = out["memcell_to_cluster"]
    assert m2c == {"a": "cluster_0", "b": "cluster_0"}


def test_serialize_cluster_file_is_json_serialisable() -> None:
    """Centroid stored as np.float32 must serialise without TypeError."""
    clusters = [
        Cluster(
            id="cluster_0",
            centroid=np.array([0.1, 0.2], dtype=np.float32),
            count=1,
            last_ts=0,
            members=["x"],
            preview=[],
        )
    ]
    out = _serialize_cluster_file(clusters)
    # Must not raise
    serialised = json.dumps(out)
    assert "cluster_0" in serialised


@pytest.mark.asyncio
async def test_run_clustering_pass_writes_json(tmp_path: Path) -> None:
    """Clustering pass must write clusters_conv_<i>.json with correct shape."""
    memcells = [_make_memcell("0", timestamp=1000, content="episode one", content_embeddings=[1.0, 0.0])]

    await _run_clustering_pass(3, memcells, tmp_path, threshold=0.70, time_window_days=7.0)

    out_file = tmp_path / "clusters_conv_3.json"
    assert out_file.exists()
    data = json.loads(out_file.read_text())
    assert "clusters" in data
    assert "memcell_to_cluster" in data
    assert len(data["clusters"]) == 1
    assert data["clusters"][0]["id"] == "cluster_0"
    assert data["memcell_to_cluster"] == {"0": "cluster_0"}


@pytest.mark.asyncio
async def test_clustering_skipped_when_disabled(tmp_path: Path) -> None:
    """When enable_clustering=False, no clusters_conv_*.json must be written."""
    from benchmarks.common.stages.extract import _process_conversation

    cfg = BenchmarkConfig(enable_clustering=False)

    # Minimal mock services with a callable embedding (must NOT be invoked).
    mock_embedding = MagicMock()
    mock_embedding.embed = AsyncMock(return_value=[[0.1, 0.2]])
    mock_services = MagicMock()
    mock_services.embedding = mock_embedding

    ctx = MagicMock(spec=StageContext)
    ctx.config = cfg
    ctx.services = mock_services

    fake_payload = [_make_memcell("0", content="something")]

    with (
        patch(
            "benchmarks.common.stages.extract._extract_one_conversation",
            new=AsyncMock(return_value=(fake_payload, 10, 5)),
        ),
    ):
        ok, _pt, _ct = await _process_conversation(
            0,
            MagicMock(),  # conv
            MagicMock(),  # llm
            asyncio.Semaphore(1),
            asyncio.Semaphore(20),
            tmp_path,
            smart_mask=True,
            max_attempts=1,
            ctx=ctx,
        )

    assert ok is True
    assert not (tmp_path / "clusters_conv_0.json").exists()
    mock_embedding.embed.assert_not_awaited()


# ---------------------------------------------------------------------------
# _detect_all_boundaries unit tests (incremental path)
# ---------------------------------------------------------------------------


def _make_chat_msg(idx: int) -> Any:
    """Return a minimal ChatMessage-compatible MagicMock for boundary tests."""
    from everalgo.types import ChatMessage

    return ChatMessage(
        id=f"m{idx}",
        role="user",
        content=f"hello {idx}",
        timestamp=1_700_000_000_000 + idx * 30_000,
        sender_id=f"u{idx}",
        sender_name=None,
    )


@pytest.mark.asyncio
async def test_detect_all_boundaries_empty_messages_returns_empty() -> None:
    """Zero messages must return an empty cell list without calling adetect_step."""
    mock_llm = AsyncMock()
    mock_step = AsyncMock()
    with patch("benchmarks.common.stages.extract.BoundaryDetector") as mock_cls:
        mock_cls.return_value.adetect_step = mock_step
        cells = await _detect_all_boundaries([], llm=mock_llm)
    assert cells == []
    mock_step.assert_not_awaited()


@pytest.mark.asyncio
async def test_detect_all_boundaries_buffers_first_two_messages() -> None:
    """Front-2 buffer: adetect_step must not be called for the first 2 messages.

    With only 2 messages the loop never reaches the algo; the residual history
    is flushed as a single final-tail MemCell.
    """
    mock_llm = AsyncMock()
    mock_step = AsyncMock()
    msgs = [_make_chat_msg(i) for i in range(2)]
    with patch("benchmarks.common.stages.extract.BoundaryDetector") as mock_cls:
        mock_cls.return_value.adetect_step = mock_step
        cells = await _detect_all_boundaries(msgs, llm=mock_llm)
    assert len(cells) == 1  # only the final-flush cell
    assert mock_step.await_count == 0  # front-2 buffer: algo never called


@pytest.mark.asyncio
async def test_detect_all_boundaries_should_wait_accumulates_without_emit() -> None:
    """When every adetect_step returns empty cells, only the final-flush cell is emitted.

    Each should_wait call returns ``DetectionResult(cells=[], tail=[*history, new])``.
    The outer loop replaces history with the returned tail.
    """
    mock_llm = AsyncMock()
    msgs = [_make_chat_msg(i) for i in range(6)]

    async def step_wait(history: list[Any], new: Any, **_: Any) -> DetectionResult:
        return DetectionResult(cells=[], tail=[*history, new])

    mock_step = AsyncMock(side_effect=step_wait)
    with patch("benchmarks.common.stages.extract.BoundaryDetector") as mock_cls:
        mock_cls.return_value.adetect_step = mock_step
        cells = await _detect_all_boundaries(msgs, llm=mock_llm)
    # 6 msgs: first 2 buffered, 4 algo calls all returning should_wait → 1 final flush
    assert len(cells) == 1
    assert mock_step.await_count == 4


@pytest.mark.asyncio
async def test_detect_all_boundaries_flushes_final_tail() -> None:
    """One closed cell from algo + remaining msgs form the final-tail flush."""
    from typing import cast

    from everalgo.types import ConversationItem, MemCell

    mock_llm = AsyncMock()
    msgs = [_make_chat_msg(i) for i in range(6)]
    closed_cell = MemCell(items=cast("list[ConversationItem]", msgs[:3]), timestamp=msgs[2].timestamp)

    # msg 2 → wait; msg 3 → cut (clean cut, tail=[msg3]); msgs 4-5 → wait.
    side_effects = [
        DetectionResult(cells=[], tail=[msgs[0], msgs[1], msgs[2]]),
        DetectionResult(cells=[closed_cell], tail=[msgs[3]]),
        DetectionResult(cells=[], tail=[msgs[3], msgs[4]]),
        DetectionResult(cells=[], tail=[msgs[3], msgs[4], msgs[5]]),
    ]
    mock_step = AsyncMock(side_effect=side_effects)

    with patch("benchmarks.common.stages.extract.BoundaryDetector") as mock_cls:
        mock_cls.return_value.adetect_step = mock_step
        cells = await _detect_all_boundaries(msgs, llm=mock_llm, smart_mask=False)

    # 1 closed cell + 1 final-tail flush = 2 total
    assert len(cells) == 2
    assert cells[0] is closed_cell


@pytest.mark.asyncio
async def test_detect_all_boundaries_threads_returned_tail_into_next_call() -> None:
    """Outer loop must feed ``DetectionResult.tail`` as the next call's history.

    Records the ``history`` argument each call sees; verifies the post-cut call
    receives the bridged tail (smart-mask path is owned by the algo, so the
    outer loop just trusts the returned tail).
    """
    from typing import cast

    from everalgo.types import ConversationItem, MemCell

    mock_llm = AsyncMock()
    msgs = [_make_chat_msg(i) for i in range(5)]
    # msgs 0,1 buffered. Call sequence (history shown is what algo SHOULD see):
    #   call0: history=[0,1],   new=2 → wait, tail=[0,1,2]
    #   call1: history=[0,1,2], new=3 → cut, tail=[2,3]   (algo decided to bridge)
    #   call2: history=[2,3],   new=4 → wait, tail=[2,3,4]
    closed_cell = MemCell(items=cast("list[ConversationItem]", msgs[:3]), timestamp=msgs[2].timestamp)
    side_effects = [
        DetectionResult(cells=[], tail=[msgs[0], msgs[1], msgs[2]]),
        DetectionResult(cells=[closed_cell], tail=[msgs[2], msgs[3]]),
        DetectionResult(cells=[], tail=[msgs[2], msgs[3], msgs[4]]),
    ]

    captured_histories: list[list[Any]] = []
    call_idx = {"i": 0}

    async def recording_step(history: list[Any], new: Any, **_: Any) -> DetectionResult:
        captured_histories.append(list(history))
        result = side_effects[call_idx["i"]]
        call_idx["i"] += 1
        return result

    with patch("benchmarks.common.stages.extract.BoundaryDetector") as mock_cls:
        mock_cls.return_value.adetect_step = recording_step
        cells = await _detect_all_boundaries(msgs, llm=mock_llm, smart_mask=True)

    # Validate histories passed to algo on each call.
    assert captured_histories[0] == [msgs[0], msgs[1]]
    assert captured_histories[1] == [msgs[0], msgs[1], msgs[2]]
    # After cut: call2 must see the returned tail [msgs[2], msgs[3]].
    assert captured_histories[2] == [msgs[2], msgs[3]]
    # 1 closed cell + 1 final-tail flush.
    assert len(cells) == 2
    assert cells[0] is closed_cell
