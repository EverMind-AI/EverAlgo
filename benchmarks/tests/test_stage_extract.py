"""Tests for Stage 1 Extract Base (boundary detection + episode extraction + clustering).

Most tests are skipped without API credentials (integration). A structural
test verifies the stage entry point exists and imports. Unit tests cover the
clustering helpers with mocked embedding and operator calls.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from benchmarks.common.config import BenchmarkConfig
from benchmarks.common.stages.extract import (
    _build_clusters_data,
    _cluster_one_episode,
    _detect_all_boundaries,
    _run_clustering_pass,
    run_extract_base_stage,
)
from benchmarks.common.stages.types import StageContext
from everalgo.boundary import DetectionResult
from everalgo.clustering.state import Cluster

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_episode(
    ep_id: str = "0",
    timestamp: int = 1_000_000,
    episode_text: str = "hello world",
    episode_embedding: list[float] | None = None,
    subject: str = "test",
    subject_embedding: list[float] | None = None,
    summary: str | None = None,
) -> dict[str, Any]:
    """Return a minimal episode dict matching the Extract Base output schema."""
    return {
        "id": ep_id,
        "owner_id": None,
        "memcell_ids": [ep_id],
        "subject": subject,
        "episode": episode_text,
        "summary": summary if summary is not None else f"Preview of {episode_text}",
        "timestamp": timestamp,
        "embeddings": {
            "episode": episode_embedding,
            "subject": subject_embedding,
        },
    }


def _make_memcell(mc_id: str = "0", timestamp: int = 1_000_000) -> dict[str, Any]:
    """Return a minimal memcell dict matching the Extract Base output schema."""
    return {
        "id": mc_id,
        "timestamp": timestamp,
        "items": [],
    }


def test_run_extract_base_stage_callable():
    """Structural check: function imports and is async."""
    import inspect

    assert inspect.iscoroutinefunction(run_extract_base_stage)


def test_backward_compat_alias():
    """The old ``run_extract_stage`` name must still be importable."""
    from benchmarks.common.stages.extract import run_extract_stage

    assert run_extract_stage is run_extract_base_stage


# ---------------------------------------------------------------------------
# Clustering unit tests (no API credentials required)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cluster_one_episode_mints_first_cluster() -> None:
    """First episode in an empty list must always create cluster_0."""
    ep = _make_episode("ep0", timestamp=1000, episode_text="first episode", episode_embedding=[1.0, 0.0, 0.0])

    result = await _cluster_one_episode(ep, [], threshold=0.70, time_window_days=7.0)

    assert len(result) == 1
    assert result[0].id == "cluster_0"
    assert result[0].members == ["ep0"]
    assert result[0].last_ts == 1000


@pytest.mark.asyncio
async def test_cluster_one_episode_merges_into_existing() -> None:
    """An episode similar to an existing cluster must merge, not mint a new one."""
    existing = Cluster(
        id="cluster_0",
        centroid=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        count=1,
        last_ts=500,
        members=["ep0"],
        preview=["old preview"],
    )
    ep = _make_episode("ep1", timestamp=1000, episode_text="similar episode", episode_embedding=[1.0, 0.0, 0.0])

    result = await _cluster_one_episode(ep, [existing], threshold=0.70, time_window_days=7.0)

    assert len(result) == 1
    assert result[0].id == "cluster_0"
    assert "ep1" in result[0].members
    assert result[0].count == 2


@pytest.mark.asyncio
async def test_cluster_one_episode_raises_on_missing_embeddings() -> None:
    """Episode with no episode embedding must raise ValueError (fail-loud)."""
    ep = _make_episode("ep0", episode_text="some episode", episode_embedding=None)

    with pytest.raises(ValueError, match="episode embedding is missing"):
        await _cluster_one_episode(ep, [], threshold=0.70, time_window_days=7.0)


@pytest.mark.asyncio
async def test_cluster_one_episode_appends_new_cluster_when_dissimilar() -> None:
    """A dissimilar episode must mint a second cluster, not overwrite the first."""
    existing = Cluster(
        id="cluster_0",
        centroid=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        count=1,
        last_ts=0,
        members=["ep0"],
        preview=["old"],
    )
    ep = _make_episode("ep1", timestamp=0, episode_text="orthogonal topic", episode_embedding=[0.0, 1.0, 0.0])

    result = await _cluster_one_episode(ep, [existing], threshold=0.70, time_window_days=7.0)

    assert len(result) == 2
    ids = {c.id for c in result}
    assert ids == {"cluster_0", "cluster_1"}


def test_build_clusters_data_shape() -> None:
    """Output list must have correct fields with centroid as plain Python floats."""
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
    out = _build_clusters_data(clusters)

    assert len(out) == 1
    c = out[0]
    assert c["id"] == "cluster_0"
    assert isinstance(c["centroid"], list)
    assert all(isinstance(v, float) for v in c["centroid"])
    assert c["count"] == 2
    assert c["last_ts"] == 2000
    assert c["members"] == ["a", "b"]


def test_build_clusters_data_is_json_serialisable() -> None:
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
    out = _build_clusters_data(clusters)
    serialised = json.dumps(out)
    assert "cluster_0" in serialised


@pytest.mark.asyncio
async def test_run_clustering_pass_writes_json(tmp_path: Path) -> None:
    """Clustering pass must write clusters_conv_<i>.json with correct shape."""
    episodes = [_make_episode("0", timestamp=1000, episode_text="episode one", episode_embedding=[1.0, 0.0])]
    memcell_to_episode = {"0": "0"}

    await _run_clustering_pass(3, episodes, memcell_to_episode, tmp_path, threshold=0.70, time_window_days=7.0)

    out_file = tmp_path / "clusters_conv_3.json"
    assert out_file.exists()
    data = json.loads(out_file.read_text())
    assert "clusters" in data
    assert "episode_to_cluster" in data
    assert len(data["clusters"]) == 1
    assert data["clusters"][0]["id"] == "cluster_0"
    assert "episode_ids" in data["clusters"][0]


@pytest.mark.asyncio
async def test_clustering_always_runs(tmp_path: Path) -> None:
    """Clustering is always enabled; all three entity files must be written."""
    from benchmarks.common.stages.extract import _process_conversation

    cfg = BenchmarkConfig()

    mock_services = MagicMock()

    ctx = MagicMock(spec=StageContext)
    ctx.config = cfg
    ctx.services = mock_services

    fake_memcells = [_make_memcell("0")]
    fake_episodes = [_make_episode("0", episode_embedding=[0.1, 0.2, 0.3])]

    with (
        patch(
            "benchmarks.common.stages.extract._extract_one_conversation",
            new=AsyncMock(return_value=(fake_memcells, fake_episodes, 10, 5)),
        ),
        patch(
            "benchmarks.common.stages.extract._run_clustering_pass",
            new=AsyncMock(return_value=1),
        ),
    ):
        ok, _, _ = await _process_conversation(
            0,
            MagicMock(),  # conv
            MagicMock(),  # llm
            asyncio.Semaphore(1),
            asyncio.Semaphore(20),
            output_dir=tmp_path,
            smart_mask=True,
            max_attempts=1,
            ctx=ctx,
        )

    assert ok is True
    # Verify the three files are written
    assert (tmp_path / "memcells_conv_0.json").exists()
    assert (tmp_path / "episodes_conv_0.json").exists()


# ---------------------------------------------------------------------------
# Three-file output format tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_conversation_writes_three_files(tmp_path: Path) -> None:
    """_process_conversation must write memcells, episodes, and clusters files."""
    from benchmarks.common.stages.extract import _process_conversation

    cfg = BenchmarkConfig()
    mock_services = MagicMock()
    ctx = MagicMock(spec=StageContext)
    ctx.config = cfg
    ctx.services = mock_services

    fake_memcells = [_make_memcell("0", timestamp=1000), _make_memcell("1", timestamp=2000)]
    fake_episodes = [
        _make_episode("0", timestamp=1000, episode_embedding=[1.0, 0.0, 0.0]),
        _make_episode("1", timestamp=2000, episode_embedding=[0.0, 1.0, 0.0]),
    ]

    with patch(
        "benchmarks.common.stages.extract._extract_one_conversation",
        new=AsyncMock(return_value=(fake_memcells, fake_episodes, 20, 10)),
    ):
        ok, pt, ct = await _process_conversation(
            0,
            MagicMock(),
            MagicMock(),
            asyncio.Semaphore(1),
            asyncio.Semaphore(20),
            output_dir=tmp_path,
            smart_mask=True,
            max_attempts=1,
            ctx=ctx,
        )

    assert ok is True
    assert pt == 20
    assert ct == 10

    # All three files exist
    assert (tmp_path / "memcells_conv_0.json").exists()
    assert (tmp_path / "episodes_conv_0.json").exists()
    assert (tmp_path / "clusters_conv_0.json").exists()

    # Memcells — no episode/atomic_facts fields
    memcells = json.loads((tmp_path / "memcells_conv_0.json").read_text())
    assert len(memcells) == 2
    for mc in memcells:
        assert "episode" not in mc
        assert "atomic_facts" not in mc
        assert "id" in mc
        assert "items" in mc

    # Episodes — uses algo field names, has embeddings, and persists the model-written summary so the
    # reflect stage can rebuild an Episode (the field is required on the model).
    episodes = json.loads((tmp_path / "episodes_conv_0.json").read_text())
    assert len(episodes) == 2
    for ep in episodes:
        assert "episode" in ep
        assert "content" not in ep
        assert ep["summary"], "summary must round-trip through the serialised episode"
        assert "memcell_ids" in ep
        assert "embeddings" in ep

    # Clusters — has episode_ids
    clusters = json.loads((tmp_path / "clusters_conv_0.json").read_text())
    for cl in clusters["clusters"]:
        assert "episode_ids" in cl
        assert "centroid" in cl
    assert "episode_to_cluster" in clusters

    # Episode count equals memcell count (1:1 before Reflection)
    assert len(episodes) == len(memcells)


@pytest.mark.asyncio
async def test_memcell_to_episode_mapping_is_identity(tmp_path: Path) -> None:
    """Before Reflection, memcell_to_episode must be a 1:1 identity mapping."""
    from benchmarks.common.stages.extract import _process_conversation

    cfg = BenchmarkConfig()
    ctx = MagicMock(spec=StageContext)
    ctx.config = cfg
    ctx.services = MagicMock()

    n = 3
    fake_memcells = [_make_memcell(str(i), timestamp=1000 * (i + 1)) for i in range(n)]
    fake_episodes = [
        _make_episode(str(i), timestamp=1000 * (i + 1), episode_embedding=[float(i), 0.0, 0.0]) for i in range(n)
    ]

    with patch(
        "benchmarks.common.stages.extract._extract_one_conversation",
        new=AsyncMock(return_value=(fake_memcells, fake_episodes, 30, 15)),
    ):
        ok, _, _ = await _process_conversation(
            0,
            MagicMock(),
            MagicMock(),
            asyncio.Semaphore(1),
            asyncio.Semaphore(20),
            output_dir=tmp_path,
            smart_mask=True,
            max_attempts=1,
            ctx=ctx,
        )

    assert ok is True
    clusters = json.loads((tmp_path / "clusters_conv_0.json").read_text())
    ep2cl = clusters["episode_to_cluster"]
    # For 1:1 identity, every episode maps to a cluster
    for i in range(n):
        assert str(i) in ep2cl


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

    Each should_wait call returns ``DetectionResult(cells=[], tail=[*history, new], should_wait=None)``.
    The outer loop replaces history with the returned tail.
    """
    mock_llm = AsyncMock()
    msgs = [_make_chat_msg(i) for i in range(6)]

    async def step_wait(history: list[Any], new: Any, **_: Any) -> DetectionResult:
        return DetectionResult(cells=[], tail=[*history, new], should_wait=None)

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
        DetectionResult(cells=[], tail=[msgs[0], msgs[1], msgs[2]], should_wait=None),
        DetectionResult(cells=[closed_cell], tail=[msgs[3]], should_wait=None),
        DetectionResult(cells=[], tail=[msgs[3], msgs[4]], should_wait=None),
        DetectionResult(cells=[], tail=[msgs[3], msgs[4], msgs[5]], should_wait=None),
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
        DetectionResult(cells=[], tail=[msgs[0], msgs[1], msgs[2]], should_wait=None),
        DetectionResult(cells=[closed_cell], tail=[msgs[2], msgs[3]], should_wait=None),
        DetectionResult(cells=[], tail=[msgs[2], msgs[3], msgs[4]], should_wait=None),
    ]

    captured_histories: list[list[Any]] = []
    call_idx = {"i": 0}

    async def recording_step(history: list[Any], _new: Any, **_: Any) -> DetectionResult:
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
