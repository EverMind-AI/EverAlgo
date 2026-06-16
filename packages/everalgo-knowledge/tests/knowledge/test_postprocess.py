"""Unit tests for ``_llm_json`` and ``_postprocess``."""

from __future__ import annotations

from typing import Any

from everalgo.knowledge._llm_json import parse_llm_json
from everalgo.knowledge._postprocess import (
    POSTPROCESS_MIN_TOTAL_CHARS,
    aassign_uncovered_blocks,
    apostprocess_topics,
    asplit_unsplit_leaves,
    detect_uncovered_blocks,
    detect_unsplit_leaves,
)
from everalgo.testing.fake_llm import FakeLLMClient

# ── parse_llm_json ───────────────────────────────────────────────────


def test_parse_llm_json_fence_with_lang() -> None:
    text = 'Here you go:\n```json\n{"k": 1}\n```'
    assert parse_llm_json(text) == {"k": 1}


def test_parse_llm_json_fence_without_lang() -> None:
    text = '```\n{"k": 2}\n```'
    assert parse_llm_json(text) == {"k": 2}


def test_parse_llm_json_bare() -> None:
    text = 'prose before {"k": 3} prose after'
    assert parse_llm_json(text) == {"k": 3}


def test_parse_llm_json_empty_or_none_returns_none() -> None:
    assert parse_llm_json("") is None


def test_parse_llm_json_no_object_returns_none() -> None:
    assert parse_llm_json("no JSON at all") is None


def test_parse_llm_json_malformed_returns_none() -> None:
    assert parse_llm_json("{not valid json") is None


def test_parse_llm_json_non_dict_returns_none() -> None:
    # JSON array at top level is parseable but not a dict — we reject.
    assert parse_llm_json("[1, 2, 3]") is None


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
    llm_response = """
    {
      "results": [
        {
          "original_topic": "Big",
          "should_split": true,
          "block_refs": "",
          "children": [
            {"topic": "Sec A", "block_refs": "0-1"},
            {"topic": "Sec B", "block_refs": "2-3"}
          ]
        }
      ]
    }
    """
    client = FakeLLMClient(responses=[llm_response])
    out = await asplit_unsplit_leaves(client, "Doc", topics, unsplit, atom_map)
    # Original identity preserved; children injected; block_refs swapped to parent intro only.
    assert out[0]["topic"] == "Big"
    assert len(out[0]["children"]) == 2
    assert out[0]["block_refs"] == ""


async def test_split_unsplit_leaves_noop_on_should_split_false() -> None:
    atom_map = {0: "## A", 1: "body", 2: "## B", 3: "body"}
    topics: list[dict[str, Any]] = [{"topic": "Big", "block_refs": "0-3"}]
    unsplit = [
        {"topic_name": "Big", "path": [0], "block_ids": [0, 1, 2, 3], "heading_ids": [0, 2], "total_chars": 1},
    ]
    llm_response = '{"results": [{"original_topic": "Big", "should_split": false, "reason": "tight"}]}'
    client = FakeLLMClient(responses=[llm_response])
    out = await asplit_unsplit_leaves(client, "Doc", topics, unsplit, atom_map)
    assert "children" not in out[0]
    assert out[0]["block_refs"] == "0-3"


async def test_split_unsplit_leaves_noop_on_parse_failure() -> None:
    topics: list[dict[str, Any]] = [{"topic": "Big", "block_refs": "0-1"}]
    unsplit = [
        {"topic_name": "Big", "path": [0], "block_ids": [0, 1], "heading_ids": [0, 1], "total_chars": 1},
    ]
    client = FakeLLMClient(responses=["not json at all"])
    out = await asplit_unsplit_leaves(client, "Doc", topics, unsplit, {0: "a", 1: "b"})
    assert out == topics


# ── assign_uncovered_blocks (LLM pass) ───────────────────────────────


async def test_assign_uncovered_blocks_appends_to_existing_refs() -> None:
    atom_map = {0: "covered text", 1: "orphan text"}
    topics: list[dict[str, Any]] = [{"topic": "T", "block_refs": "0"}]
    uncovered = [{"block_id": 1, "text": "orphan text"}]
    llm_response = '{"assignments": [{"block_id": 1, "topic_index": 0, "reason": "ok"}]}'
    client = FakeLLMClient(responses=[llm_response])
    out = await aassign_uncovered_blocks(client, "Doc", topics, uncovered, atom_map)
    assert out[0]["block_refs"] == "0,1"


async def test_assign_uncovered_blocks_handles_empty_existing_refs() -> None:
    atom_map = {0: "orphan"}
    topics: list[dict[str, Any]] = [{"topic": "T", "block_refs": "", "children": []}]
    uncovered = [{"block_id": 0, "text": "orphan"}]
    llm_response = '{"assignments": [{"block_id": 0, "topic_index": 0}]}'
    client = FakeLLMClient(responses=[llm_response])
    out = await aassign_uncovered_blocks(client, "Doc", topics, uncovered, atom_map)
    assert out[0]["block_refs"] == "0"


async def test_assign_uncovered_blocks_skips_invalid_topic_index() -> None:
    atom_map = {0: "covered", 1: "orphan"}
    topics: list[dict[str, Any]] = [{"topic": "T", "block_refs": "0"}]
    uncovered = [{"block_id": 1, "text": "orphan"}]
    llm_response = '{"assignments": [{"block_id": 1, "topic_index": 99}]}'
    client = FakeLLMClient(responses=[llm_response])
    out = await aassign_uncovered_blocks(client, "Doc", topics, uncovered, atom_map)
    assert out[0]["block_refs"] == "0"  # untouched


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
    llm_response = '{"assignments": [{"block_id": 1, "topic_index": 0}]}'
    client = FakeLLMClient(responses=[llm_response])
    out = await apostprocess_topics(client, "Doc", topics, list(atom_map.items()))
    assert out[0]["block_refs"] == "0,1"
    assert client.call_count == 1
