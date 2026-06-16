"""Unit tests for the multi-batch consolidation helpers."""

from __future__ import annotations

import json
from typing import Any

from everalgo.knowledge._batch_merge import (
    MIN_TOPICS_FOR_MERGE,
    _apply_merge_groups_with_remap,
    _flatten_for_merge,
    _rebuild_hierarchy,
    amerge_doc_summary,
    amerge_topics,
)
from everalgo.testing.fake_llm import FakeLLMClient

# ── pure-compute helpers ─────────────────────────────────────────────


def test_flatten_for_merge_carries_parent_context() -> None:
    nested: list[dict[str, Any]] = [
        {
            "topic": "Section A",
            "summary": "sA",
            "block_refs": "0-2",
            "content_labels": ["x"],
            "children": [
                {"topic": "A.1", "summary": "sA1", "block_refs": "1"},
                {"topic": "A.2", "summary": "sA2", "block_refs": "2"},
            ],
        },
        {"topic": "Section B", "summary": "sB", "block_refs": "3"},
    ]
    flat, parents = _flatten_for_merge(nested)

    assert [e["topic"] for e in flat] == ["Section A", "A.1", "A.2", "Section B"]
    assert [e["parent_topic"] for e in flat] == [None, "Section A", "Section A", None]
    assert parents == {0: None, 1: 0, 2: 0, 3: None}
    # block_refs / content_labels are carried through unchanged
    assert flat[0]["content_labels"] == ["x"]
    assert flat[1]["block_refs"] == "1"


def test_apply_merge_groups_unions_refs_and_labels() -> None:
    flat = [
        {
            "index": 0,
            "topic": "Death",
            "summary": "s0",
            "block_refs": "1-3",
            "content_labels": ["v"],
            "parent_topic": "Plot",
        },
        {
            "index": 1,
            "topic": "Other",
            "summary": "s1",
            "block_refs": "4",
            "content_labels": [],
            "parent_topic": "Plot",
        },
        {
            "index": 2,
            "topic": "Murder",
            "summary": "s2",
            "block_refs": "12-14",
            "content_labels": ["v", "a"],
            "parent_topic": "Plot",
        },
    ]
    merges = [{"indices": [0, 2], "topic": "Death/Murder", "summary": "merged"}]

    survivors, remap = _apply_merge_groups_with_remap(flat, merges)

    assert remap == {0: 0, 2: 0}
    # Topic 1 untouched; topic 0 survives with merged fields; topic 2 absorbed.
    assert [s["index"] for s in survivors] == [0, 1]
    merged_entry = survivors[0]
    assert merged_entry["topic"] == "Death/Murder"
    assert merged_entry["summary"] == "merged"
    assert merged_entry["block_refs"] == "1-3,12-14"
    assert merged_entry["content_labels"] == ["v", "a"]  # order-preserving dedupe


def test_apply_merge_skips_groups_with_lt_two_indices() -> None:
    flat = [{"index": 0, "topic": "T", "summary": "", "block_refs": "", "content_labels": [], "parent_topic": None}]
    survivors, remap = _apply_merge_groups_with_remap(flat, [{"indices": [0]}])
    assert remap == {}
    assert survivors == flat


def test_rebuild_hierarchy_reparents_absorbed_children() -> None:
    # Original tree:
    #   0 Plot
    #     1 Death
    #       2 weapon-detail
    #     3 Murder
    #       4 motive-detail
    # LLM merges Death (1) and Murder (3) → surviving = 1.
    # weapon-detail (parent=1) stays under 1.
    # motive-detail (parent=3) gets reparented to 1.
    nested = [
        {
            "topic": "Plot",
            "summary": "p",
            "children": [
                {"topic": "Death", "summary": "d", "children": [{"topic": "weapon"}]},
                {"topic": "Murder", "summary": "m", "children": [{"topic": "motive"}]},
            ],
        }
    ]
    flat, parents = _flatten_for_merge(nested)
    merges = [{"indices": [1, 3], "topic": "DeathMerge", "summary": "merged"}]
    survivors, remap = _apply_merge_groups_with_remap(flat, merges)
    roots = _rebuild_hierarchy(survivors, parents, remap)

    assert len(roots) == 1
    plot = roots[0]
    assert plot["topic"] == "Plot"
    assert [c["topic"] for c in plot["children"]] == ["DeathMerge"]
    merged_node = plot["children"][0]
    assert {c["topic"] for c in merged_node["children"]} == {"weapon", "motive"}
    # internal fields stripped
    for node in [plot, merged_node, *merged_node["children"]]:
        assert "index" not in node
        assert "parent_topic" not in node


def test_min_topics_constant_matches_skip_rule() -> None:
    # Sanity: the constant we publish drives merge_topics' skip path.
    assert MIN_TOPICS_FOR_MERGE == 4


# ── LLM-driven entry points (with FakeLLMClient) ────────────────────


async def test_merge_doc_summary_single_batch_does_not_call_llm() -> None:
    client = FakeLLMClient(responses=[])
    out = await amerge_doc_summary(client, "T", [{"summary": "only one"}])
    assert out == "only one"
    assert client.call_count == 0


async def test_merge_doc_summary_returns_merged_field() -> None:
    client = FakeLLMClient(responses=['{"summary": "combined"}'])
    out = await amerge_doc_summary(
        client,
        "T",
        [{"summary": "s1", "subject": "u1", "keywords": []}, {"summary": "s2"}],
    )
    assert out == "combined"
    assert client.call_count == 1


async def test_merge_doc_summary_falls_back_when_unparsable() -> None:
    client = FakeLLMClient(responses=["not json"])
    out = await amerge_doc_summary(client, "T", [{"summary": "fallback"}, {"summary": "s2"}])
    assert out == "fallback"


async def test_merge_doc_summary_falls_back_when_field_missing() -> None:
    client = FakeLLMClient(responses=['{"language": "English"}'])
    out = await amerge_doc_summary(client, "T", [{"summary": "fallback"}, {"summary": "s2"}])
    assert out == "fallback"


async def test_merge_topics_short_circuits_for_small_input() -> None:
    client = FakeLLMClient(responses=[])
    topics = [
        {"topic": "A", "summary": "a", "block_refs": "0"},
        {"topic": "B", "summary": "b", "block_refs": "1"},
    ]
    out = await amerge_topics(client, "T", topics)
    assert out is topics
    assert client.call_count == 0


async def test_merge_topics_returns_original_on_empty_merges() -> None:
    topics = [{"topic": f"T{i}", "summary": f"s{i}", "block_refs": str(i)} for i in range(MIN_TOPICS_FOR_MERGE)]
    client = FakeLLMClient(responses=['{"merges": []}'])
    out = await amerge_topics(client, "T", topics)
    assert out is topics
    assert client.call_count == 1


async def test_merge_topics_applies_merges() -> None:
    topics = [
        {"topic": "Death", "summary": "d", "block_refs": "1"},
        {"topic": "Murder", "summary": "m", "block_refs": "5"},
        {"topic": "Setting", "summary": "s", "block_refs": "2"},
        {"topic": "Cast", "summary": "c", "block_refs": "3"},
    ]
    response = json.dumps({"merges": [{"indices": [0, 1], "topic": "Death/Murder", "summary": "merged"}]})
    client = FakeLLMClient(responses=[response])
    out = await amerge_topics(client, "T", topics)

    assert client.call_count == 1
    assert [t["topic"] for t in out] == ["Death/Murder", "Setting", "Cast"]
    assert out[0]["block_refs"] == "1,5"


async def test_merge_topics_falls_back_on_unparsable_response() -> None:
    topics = [{"topic": f"T{i}", "summary": "", "block_refs": str(i)} for i in range(MIN_TOPICS_FOR_MERGE)]
    client = FakeLLMClient(responses=["garbage"])
    out = await amerge_topics(client, "T", topics)
    assert out is topics
