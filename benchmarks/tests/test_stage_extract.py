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
from everalgo.boundary import Cuts
from everalgo.clustering.state import Cluster
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import ChatMessage

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


def _make_chat_msg(idx: int) -> ChatMessage:
    """Return a minimal ChatMessage for boundary tests."""
    return ChatMessage(
        id=f"m{idx}",
        role="user",
        content=f"hello {idx}",
        timestamp=1_700_000_000_000 + idx * 30_000,
        sender_id=f"u{idx}",
        sender_name=None,
    )


def _scripted_step(*cuts: Cuts) -> tuple[AsyncMock, list[list[str]]]:
    """An ``adetect_step`` stand-in returning ``cuts`` in order and recording the ids of each ``items`` it gets."""
    seen: list[list[str]] = []
    script = iter(cuts)

    async def step(items: list[ChatMessage], **_: Any) -> Cuts:
        seen.append([m.id for m in items])
        return next(script)

    return AsyncMock(side_effect=step), seen


@pytest.mark.asyncio
async def test_detect_all_boundaries_empty_messages_returns_empty() -> None:
    """Zero messages must return an empty cell list without calling adetect_step."""
    mock_step, _ = _scripted_step()
    with patch("benchmarks.common.stages.extract.BoundaryDetector") as mock_cls:
        mock_cls.return_value.adetect_step = mock_step
        cells = await _detect_all_boundaries([], llm=AsyncMock())
    assert cells == []
    mock_step.assert_not_awaited()


@pytest.mark.asyncio
async def test_detect_all_boundaries_judges_every_message_without_a_warmup_of_its_own() -> None:
    """The warmup lives in ``adetect_step``: the stage hands it every message, history first."""
    msgs = [_make_chat_msg(i) for i in range(3)]
    mock_step, seen = _scripted_step(Cuts([], 0), Cuts([], 0), Cuts([], 0))
    with patch("benchmarks.common.stages.extract.BoundaryDetector") as mock_cls:
        mock_cls.return_value.adetect_step = mock_step
        cells = await _detect_all_boundaries(msgs, llm=AsyncMock(), smart_mask=False)
    assert seen == [["m0"], ["m0", "m1"], ["m0", "m1", "m2"]]
    assert all(call.kwargs == {"smart_mask": False} for call in mock_step.await_args_list)
    assert [cell.items for cell in cells] == [msgs]  # only the final-flush cell


@pytest.mark.asyncio
async def test_detect_all_boundaries_closes_cut_segments_and_flushes_the_rest() -> None:
    """A clean cut closes ``items[:3]``; the next history starts at ``tail_start``; the rest is flushed at the end."""
    msgs = [_make_chat_msg(i) for i in range(6)]
    mock_step, seen = _scripted_step(Cuts([], 0), Cuts([], 0), Cuts([], 0), Cuts([3], 3), Cuts([], 0), Cuts([], 0))
    with patch("benchmarks.common.stages.extract.BoundaryDetector") as mock_cls:
        mock_cls.return_value.adetect_step = mock_step
        cells = await _detect_all_boundaries(msgs, llm=AsyncMock())
    assert seen[4] == ["m3", "m4"]
    assert [cell.items for cell in cells] == [msgs[:3], msgs[3:]]
    assert [cell.timestamp for cell in cells] == [msgs[2].timestamp, msgs[5].timestamp]


@pytest.mark.asyncio
async def test_detect_all_boundaries_threads_a_bridged_tail_into_the_next_call() -> None:
    """A bridged cut (``tail_start = n - 1``) carries the closed segment's last message into the next history."""
    msgs = [_make_chat_msg(i) for i in range(5)]
    mock_step, seen = _scripted_step(Cuts([], 0), Cuts([], 0), Cuts([], 0), Cuts([3], 2), Cuts([], 0))
    with patch("benchmarks.common.stages.extract.BoundaryDetector") as mock_cls:
        mock_cls.return_value.adetect_step = mock_step
        cells = await _detect_all_boundaries(msgs, llm=AsyncMock(), smart_mask=True)
    assert seen == [["m0"], ["m0", "m1"], ["m0", "m1", "m2"], ["m0", "m1", "m2", "m3"], ["m2", "m3", "m4"]]
    assert [cell.items for cell in cells] == [msgs[:3], msgs[2:]]


@pytest.mark.asyncio
async def test_detect_all_boundaries_warms_up_once_with_the_real_detector() -> None:
    """With everalgo's own warmup of two messages, five messages and no cut cost three model calls."""
    reply = json.dumps({"reasoning": "same", "should_end": False})
    fake = FakeLLMClient(handler=lambda *_, **__: reply)
    msgs = [_make_chat_msg(i) for i in range(5)]

    cells = await _detect_all_boundaries(msgs, llm=fake)  # type: ignore[arg-type]

    assert fake.call_count == 3
    assert [cell.items for cell in cells] == [msgs]
