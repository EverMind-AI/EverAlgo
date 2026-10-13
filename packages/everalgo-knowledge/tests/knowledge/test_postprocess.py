"""Unit tests for ``_postprocess``."""

from __future__ import annotations

import json
from typing import Any

import pytest

from everalgo.knowledge._postprocess import (
    POSTPROCESS_MIN_TOTAL_CHARS,
    aassign_uncovered_blocks,
    apostprocess_topics,
    asplit_unsplit_leaves,
    detect_uncovered_blocks,
    detect_unsplit_leaves,
)
from everalgo.knowledge._replies import TopicSplitReply, UncoveredAssignReply
from everalgo.llm import LLMOutputError
from everalgo.testing.fake_llm import FakeLLMClient

from ._scripted import assign_reply


def _child(name: str, refs: str) -> dict[str, Any]:
    return {"topic": name, "summary": f"{name} summary", "block_refs": refs, "children": []}


def _kept(name: str) -> dict[str, Any]:
    return {"original_topic": name, "should_split": False, "reason": "tight", "block_refs": None, "children": None}


# ── detect_unsplit_leaves ────────────────────────────────────────────


def test_detect_unsplit_leaves_triggers_on_two_plus_headings_and_size() -> None:
    # Need total_chars >= MIN_TOTAL (2000) and avg >= MIN_AVG (500); 1100*2 covers both.
    body = "x" * 1100
    atom_map = {0: "## H1", 1: body, 2: "## H2", 3: body}
    topics: list[dict[str, Any]] = [
        {"topic": "Big Leaf", "block_refs": "0-3"},
    ]
    out = detect_unsplit_leaves(topics, atom_map)
    assert len(out) == 1
    assert out[0]["topic_name"] == "Big Leaf"
    assert out[0]["heading_ids"] == [0, 2]
    assert out[0]["path"] == [0]


def test_detect_unsplit_leaves_skips_when_one_heading() -> None:
    atom_map = {0: "## H1", 1: "z" * POSTPROCESS_MIN_TOTAL_CHARS}
    topics: list[dict[str, Any]] = [{"topic": "Tiny", "block_refs": "0-1"}]
    assert detect_unsplit_leaves(topics, atom_map) == []


def test_detect_unsplit_leaves_skips_when_too_short() -> None:
    atom_map = {0: "## H1", 1: "short body", 2: "## H2", 3: "short body 2"}
    topics: list[dict[str, Any]] = [{"topic": "Short", "block_refs": "0-3"}]
    assert detect_unsplit_leaves(topics, atom_map) == []


def test_detect_unsplit_leaves_recurses_into_children() -> None:
    body = "a" * 1100
    atom_map = {0: "## H1", 1: body, 2: "## H2", 3: body}
    topics: list[dict[str, Any]] = [
        {
            "topic": "Parent",
            "block_refs": "",
            "children": [{"topic": "Leaf", "block_refs": "0-3"}],
        },
    ]
    out = detect_unsplit_leaves(topics, atom_map)
    assert len(out) == 1
    assert out[0]["path"] == [0, 0]


# ── detect_uncovered_blocks ──────────────────────────────────────────


def test_detect_uncovered_blocks_basic() -> None:
    atom_map = {0: "covered", 1: "alone", 2: "also covered"}
    topics: list[dict[str, Any]] = [{"topic": "T", "block_refs": "0,2"}]
    out = detect_uncovered_blocks(topics, atom_map)
    assert [u["block_id"] for u in out] == [1]


def test_detect_uncovered_blocks_skips_separators_and_blanks() -> None:
    atom_map = {0: "covered", 1: "---", 2: "", 3: "real orphan"}
    topics: list[dict[str, Any]] = [{"topic": "T", "block_refs": "0"}]
    out = detect_uncovered_blocks(topics, atom_map)
    assert [u["block_id"] for u in out] == [3]


# ── split_unsplit_leaves (LLM pass) ──────────────────────────────────


async def test_split_unsplit_leaves_applies_decision() -> None:
    atom_map = {0: "## A", 1: "body A", 2: "## B", 3: "body B"}
    topics: list[dict[str, Any]] = [{"topic": "Big", "block_refs": "0-3"}]
    unsplit = [
        {
            "topic_name": "Big",
            "path": [0],
            "block_ids": [0, 1, 2, 3],
            "heading_ids": [0, 2],
            "total_chars": 999,
        },
    ]
    split = {
        "original_topic": "Big",
        "should_split": True,
        "reason": "two sections",
        "block_refs": "",
        "children": [_child("Sec A", "0-1"), _child("Sec B", "2-3")],
    }
    client = FakeLLMClient(responses=[json.dumps({"results": [split]})])
    out = await asplit_unsplit_leaves(client, "Doc", topics, unsplit, atom_map)
    # Original identity preserved; children injected; block_refs swapped to parent intro only.
    assert out[0]["topic"] == "Big"
    assert [child["topic"] for child in out[0]["children"]] == ["Sec A", "Sec B"]
    assert out[0]["block_refs"] == ""
    assert client.calls[0].response_format is TopicSplitReply


async def test_split_unsplit_leaves_noop_on_should_split_false() -> None:
    atom_map = {0: "## A", 1: "body", 2: "## B", 3: "body"}
    topics: list[dict[str, Any]] = [{"topic": "Big", "block_refs": "0-3"}]
    unsplit = [
        {"topic_name": "Big", "path": [0], "block_ids": [0, 1, 2, 3], "heading_ids": [0, 2], "total_chars": 1},
    ]
    client = FakeLLMClient(responses=[json.dumps({"results": [_kept("Big")]})])
    out = await asplit_unsplit_leaves(client, "Doc", topics, unsplit, atom_map)
    assert "children" not in out[0]
    assert out[0]["block_refs"] == "0-3"


@pytest.mark.parametrize(
    "reply",
    [
        pytest.param("not json at all", id="not-json"),
        pytest.param(
            '{"results": [{"original_topic": "Big", "should_split": false, "reason": "tight"}]}',
            id="omitted-nullable-fields",
        ),
        pytest.param('{"results": [], "summary": "none"}', id="extra-field"),
    ],
)
async def test_split_unsplit_leaves_invalid_reply_raises(reply: str) -> None:
    topics: list[dict[str, Any]] = [{"topic": "Big", "block_refs": "0-1"}]
    unsplit = [
        {"topic_name": "Big", "path": [0], "block_ids": [0, 1], "heading_ids": [0, 1], "total_chars": 1},
    ]
    client = FakeLLMClient(responses=[reply])
    with pytest.raises(LLMOutputError, match="knowledge topic_split reply is not a valid TopicSplitReply"):
        await asplit_unsplit_leaves(client, "Doc", topics, unsplit, {0: "a", 1: "b"})


async def test_split_unsplit_leaves_ignores_a_result_for_an_unknown_topic() -> None:
    topics: list[dict[str, Any]] = [{"topic": "Big", "block_refs": "0-3"}]
    unsplit = [
        {"topic_name": "Big", "path": [0], "block_ids": [0, 1, 2, 3], "heading_ids": [0, 2], "total_chars": 1},
    ]
    stray = {**_kept("Other"), "should_split": True, "block_refs": "", "children": [_child("A", "0-1")]}
    client = FakeLLMClient(responses=[json.dumps({"results": [stray]})])
    out = await asplit_unsplit_leaves(client, "Doc", topics, unsplit, {0: "## A", 1: "a", 2: "## B", 3: "b"})
    assert out == [{"topic": "Big", "block_refs": "0-3"}]


# ── assign_uncovered_blocks (LLM pass) ───────────────────────────────


async def test_assign_uncovered_blocks_appends_to_existing_refs() -> None:
    atom_map = {0: "covered text", 1: "orphan text"}
    topics: list[dict[str, Any]] = [{"topic": "T", "block_refs": "0"}]
    uncovered = [{"block_id": 1, "text": "orphan text"}]
    client = FakeLLMClient(responses=[assign_reply((1, 0))])
    out = await aassign_uncovered_blocks(client, "Doc", topics, uncovered, atom_map)
    assert out[0]["block_refs"] == "0,1"
    assert client.calls[0].response_format is UncoveredAssignReply


async def test_assign_uncovered_blocks_handles_empty_existing_refs() -> None:
    atom_map = {0: "orphan"}
    topics: list[dict[str, Any]] = [{"topic": "T", "block_refs": "", "children": []}]
    uncovered = [{"block_id": 0, "text": "orphan"}]
    client = FakeLLMClient(responses=[assign_reply((0, 0))])
    out = await aassign_uncovered_blocks(client, "Doc", topics, uncovered, atom_map)
    assert out[0]["block_refs"] == "0"


async def test_assign_uncovered_blocks_skips_invalid_topic_index() -> None:
    atom_map = {0: "covered", 1: "orphan"}
    topics: list[dict[str, Any]] = [{"topic": "T", "block_refs": "0"}]
    uncovered = [{"block_id": 1, "text": "orphan"}]
    client = FakeLLMClient(responses=[assign_reply((1, 99), (1, -1))])
    out = await aassign_uncovered_blocks(client, "Doc", topics, uncovered, atom_map)
    assert out[0]["block_refs"] == "0"  # untouched


@pytest.mark.parametrize(
    "reply",
    [
        pytest.param("not json at all", id="not-json"),
        pytest.param('{"assignments": [{"block_id": 1, "topic_index": 0}]}', id="missing-field"),
        pytest.param('{"assignments": [{"block_id": "1", "topic_index": 0, "reason": "r"}]}', id="string-block-id"),
        pytest.param('{"assignments": [], "notes": "none"}', id="extra-field"),
    ],
)
async def test_assign_uncovered_blocks_invalid_reply_raises(reply: str) -> None:
    topics: list[dict[str, Any]] = [{"topic": "T", "block_refs": "0"}]
    client = FakeLLMClient(responses=[reply])
    with pytest.raises(LLMOutputError, match="knowledge uncovered_assign reply is not a valid UncoveredAssignReply"):
        await aassign_uncovered_blocks(
            client, "Doc", topics, [{"block_id": 1, "text": "orphan"}], {0: "a", 1: "orphan"}
        )


# ── postprocess_topics orchestrator ──────────────────────────────────


async def test_postprocess_topics_noop_when_nothing_to_fix() -> None:
    atom_map = {0: "covered"}
    topics: list[dict[str, Any]] = [{"topic": "T", "block_refs": "0"}]
    # No LLM responses needed because no detector fires.
    client = FakeLLMClient(responses=[])
    out = await apostprocess_topics(client, "Doc", topics, list(atom_map.items()))
    assert out == topics
    assert client.call_count == 0


async def test_postprocess_topics_invokes_assign_only_when_uncovered() -> None:
    atom_map = {0: "covered", 1: "orphan"}
    topics: list[dict[str, Any]] = [{"topic": "T", "block_refs": "0"}]
    client = FakeLLMClient(responses=[assign_reply((1, 0))])
    out = await apostprocess_topics(client, "Doc", topics, list(atom_map.items()))
    assert out[0]["block_refs"] == "0,1"
    assert client.call_count == 1
