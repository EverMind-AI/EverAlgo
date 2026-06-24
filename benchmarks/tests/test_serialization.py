"""Tests for entity serialization/deserialization."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from benchmarks.common.stages.serialization import (
    load_atomic_facts,
    load_clusters,
    load_episodes,
    load_memcells,
    serialize_atomic_fact,
    serialize_clusters,
    serialize_episode,
    serialize_memcell,
    write_json,
)


def _make_mock_memcell(idx: int = 0, n_messages: int = 2, timestamp: int = 1000) -> SimpleNamespace:
    """Create a mock MemCell-like object for testing."""
    items = [
        SimpleNamespace(
            id=f"msg_{i}",
            role="user",
            content=f"msg {i}",
            timestamp=timestamp + i,
            sender_id="u",
            sender_name="User",
        )
        for i in range(n_messages)
    ]
    return SimpleNamespace(items=items, timestamp=timestamp)


class TestSerializeMemcell:
    def test_memcell_has_no_episode(self) -> None:
        mc = _make_mock_memcell(idx=0, n_messages=3)
        result = serialize_memcell(0, mc)
        assert "episode" not in result
        assert "atomic_facts" not in result
        assert result["id"] == "0"
        assert len(result["items"]) == 3

    def test_memcell_preserves_timestamp(self) -> None:
        mc = _make_mock_memcell(idx=0, n_messages=1, timestamp=1234567890)
        result = serialize_memcell(0, mc)
        assert result["timestamp"] == 1234567890

    def test_memcell_items_have_required_fields(self) -> None:
        mc = _make_mock_memcell(idx=0, n_messages=2)
        result = serialize_memcell(0, mc)
        for item in result["items"]:
            assert {"id", "role", "content", "timestamp", "sender_id", "sender_name"} <= item.keys()

    def test_memcell_id_is_string(self) -> None:
        mc = _make_mock_memcell(idx=7)
        result = serialize_memcell(7, mc)
        assert result["id"] == "7"
        assert isinstance(result["id"], str)


class TestSerializeEpisode:
    def test_episode_uses_algo_field_name(self) -> None:
        result = serialize_episode(
            ep_idx=0,
            subject="Test",
            episode_text="Some narrative",
            memcell_ids=["0"],
            timestamp=1234,
            owner_id=None,
            embeddings={"episode": [0.1, 0.2], "subject": [0.3, 0.4]},
        )
        assert result["episode"] == "Some narrative"
        assert "content" not in result
        assert result["memcell_ids"] == ["0"]

    def test_episode_has_multiple_memcell_ids(self) -> None:
        result = serialize_episode(
            ep_idx=5,
            subject="Merged",
            episode_text="Merged text",
            memcell_ids=["0", "1", "2"],
            timestamp=9999,
            owner_id=None,
            embeddings={"episode": [0.1], "subject": None},
        )
        assert result["memcell_ids"] == ["0", "1", "2"]

    def test_episode_id_is_stringified_int(self) -> None:
        result = serialize_episode(
            ep_idx=3,
            subject="S",
            episode_text="E",
            memcell_ids=["3"],
            timestamp=1,
            owner_id=None,
            embeddings={"episode": [0.1], "subject": None},
        )
        assert result["id"] == "3"

    def test_episode_embeddings_dict_preserved(self) -> None:
        emb: dict[str, list[float] | None] = {"episode": [0.1, 0.2], "subject": [0.3, 0.4]}
        result = serialize_episode(
            ep_idx=0,
            subject="S",
            episode_text="E",
            memcell_ids=["0"],
            timestamp=1,
            owner_id="user_1",
            embeddings=emb,
        )
        assert result["embeddings"] == emb
        assert result["owner_id"] == "user_1"

    def test_episode_subject_null_embeddings(self) -> None:
        result = serialize_episode(
            ep_idx=0,
            subject="S",
            episode_text="E",
            memcell_ids=["0"],
            timestamp=1,
            owner_id=None,
            embeddings={"episode": [0.1], "subject": None},
        )
        assert result["embeddings"]["subject"] is None


class TestSerializeAtomicFact:
    def test_atomic_fact_structure(self) -> None:
        result = serialize_atomic_fact(
            af_idx=0,
            content="Fact text",
            episode_id="0",
            timestamp=1234,
            owner_id=None,
            embeddings=[0.1, 0.2],
        )
        assert result["id"] == "0"
        assert result["episode_id"] == "0"
        assert result["content"] == "Fact text"
        assert result["embeddings"] == [0.1, 0.2]

    def test_atomic_fact_id_is_string(self) -> None:
        result = serialize_atomic_fact(
            af_idx=42,
            content="F",
            episode_id="5",
            timestamp=1,
            owner_id=None,
            embeddings=[],
        )
        assert result["id"] == "42"
        assert isinstance(result["id"], str)

    def test_atomic_fact_owner_id(self) -> None:
        result = serialize_atomic_fact(
            af_idx=0,
            content="F",
            episode_id="0",
            timestamp=1,
            owner_id="owner_123",
            embeddings=[0.5],
        )
        assert result["owner_id"] == "owner_123"

    def test_atomic_fact_per_item_not_grouped(self) -> None:
        """Each fact is its own dict, not nested inside a list-of-facts container."""
        result = serialize_atomic_fact(
            af_idx=7,
            content="Standalone fact",
            episode_id="3",
            timestamp=2,
            owner_id=None,
            embeddings=[0.9],
        )
        assert isinstance(result, dict)
        assert "content" in result
        assert "id" in result


class TestSerializeClusters:
    def test_cluster_has_episode_ids(self) -> None:
        clusters_data = [
            {
                "id": "cluster_0",
                "centroid": [0.1],
                "count": 2,
                "last_ts": 1000,
                "members": ["0", "1"],
                "preview": ["text"],
            },
        ]
        mc_to_ep = {"0": "0", "1": "1"}
        result = serialize_clusters(clusters_data, mc_to_ep)
        assert result["clusters"][0]["episode_ids"] == ["0", "1"]
        assert result["episode_to_cluster"]["0"] == "cluster_0"

    def test_cluster_missing_mc_in_map_skipped(self) -> None:
        """Members not in memcell_to_episode map are excluded from episode_ids."""
        clusters_data = [
            {"id": "cluster_0", "centroid": [], "count": 2, "last_ts": 0, "members": ["0", "1"], "preview": []},
        ]
        result = serialize_clusters(clusters_data, {"0": "ep_0"})  # "1" not in map
        assert result["clusters"][0]["episode_ids"] == ["ep_0"]

    def test_multiple_clusters(self) -> None:
        clusters_data = [
            {"id": "cluster_0", "centroid": [0.1], "count": 1, "last_ts": 1, "members": ["0"], "preview": []},
            {"id": "cluster_1", "centroid": [0.9], "count": 1, "last_ts": 2, "members": ["1"], "preview": []},
        ]
        mc_to_ep = {"0": "ep_0", "1": "ep_1"}
        result = serialize_clusters(clusters_data, mc_to_ep)
        assert len(result["clusters"]) == 2
        assert result["episode_to_cluster"]["ep_1"] == "cluster_1"


class TestWriteJson:
    def test_write_json_creates_file(self, tmp_path: Path) -> None:
        data = {"key": "value", "num": 42}
        path = tmp_path / "out.json"
        write_json(path, data)
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded == data

    def test_write_json_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "dir" / "file.json"
        write_json(path, [1, 2, 3])
        assert path.exists()

    def test_write_json_unicode(self, tmp_path: Path) -> None:
        path = tmp_path / "unicode.json"
        data = {"text": "Chinese: 你好"}
        write_json(path, data)
        content = path.read_text(encoding="utf-8")
        assert "你好" in content


class TestRoundtrip:
    def test_memcells_roundtrip(self, tmp_path: Path) -> None:
        mc = _make_mock_memcell(idx=0, n_messages=2)
        data = [serialize_memcell(0, mc)]
        path = tmp_path / "memcells.json"
        write_json(path, data)
        loaded = load_memcells(path)
        assert loaded[0]["id"] == "0"
        assert len(loaded[0]["items"]) == 2

    def test_episodes_roundtrip(self, tmp_path: Path) -> None:
        episodes = [
            serialize_episode(
                ep_idx=0,
                subject="S",
                episode_text="E",
                memcell_ids=["0"],
                timestamp=1,
                owner_id=None,
                embeddings={"episode": [0.1], "subject": [0.2]},
            )
        ]
        path = tmp_path / "episodes.json"
        path.write_text(json.dumps(episodes, ensure_ascii=False, indent=2))
        loaded = load_episodes(path)
        assert loaded[0]["episode"] == "E"

    def test_atomic_facts_roundtrip(self, tmp_path: Path) -> None:
        facts = [
            serialize_atomic_fact(af_idx=0, content="F1", episode_id="0", timestamp=1, owner_id=None, embeddings=[0.1]),
            serialize_atomic_fact(af_idx=1, content="F2", episode_id="0", timestamp=2, owner_id=None, embeddings=[0.2]),
        ]
        path = tmp_path / "atomic_facts.json"
        write_json(path, facts)
        loaded = load_atomic_facts(path)
        assert len(loaded) == 2
        assert loaded[1]["content"] == "F2"

    def test_clusters_roundtrip(self, tmp_path: Path) -> None:
        clusters_data = [
            {"id": "cluster_0", "centroid": [0.5], "count": 1, "last_ts": 100, "members": ["0"], "preview": ["p"]},
        ]
        data = serialize_clusters(clusters_data, {"0": "ep_0"})
        path = tmp_path / "clusters.json"
        write_json(path, data)
        loaded = load_clusters(path)
        assert loaded["clusters"][0]["id"] == "cluster_0"
        assert loaded["episode_to_cluster"]["ep_0"] == "cluster_0"
