"""Unit tests for ``everalgo.knowledge._topic_build``."""

from __future__ import annotations

from typing import Any

from everalgo.knowledge._topic_build import (
    TOPIC_MAX_DEPTH,
    TopicClip,
    build_topic_clips,
    collapse_trivial_children,
    collect_all_block_refs,
    deduplicate_parent_refs,
    parse_segment_ranges,
)

# ── parse_segment_ranges ─────────────────────────────────────────────


def test_parse_segment_ranges_empty() -> None:
    assert parse_segment_ranges("") == []
    assert parse_segment_ranges("   ") == []


def test_parse_segment_ranges_single_ids() -> None:
    assert parse_segment_ranges("5") == [5]
    assert parse_segment_ranges("1,2,3") == [1, 2, 3]


def test_parse_segment_ranges_range() -> None:
    assert parse_segment_ranges("1-3") == [1, 2, 3]


def test_parse_segment_ranges_mixed() -> None:
    assert parse_segment_ranges("1-3,5,8-10") == [1, 2, 3, 5, 8, 9, 10]


def test_parse_segment_ranges_tolerates_garbage() -> None:
    # Garbage segments are skipped silently (logged at WARNING).
    assert parse_segment_ranges("1, foo, 3-bar, 5") == [1, 5]


def test_parse_segment_ranges_tolerates_whitespace() -> None:
    assert parse_segment_ranges(" 1 , 3 - 5 ") == [1, 3, 4, 5]


# ── build_topic_clips ────────────────────────────────────────────────


def _atoms(*pairs: tuple[int, str]) -> list[tuple[int, str]]:
    return list(pairs)


def test_build_flat_topic_list() -> None:
    atoms = _atoms((0, "intro"), (1, "body"), (2, "end"))
    topics = [
        {"topic": "Intro", "summary": "s1", "block_refs": "0"},
        {"topic": "Body", "summary": "s2", "block_refs": "1-2"},
    ]
    clips = build_topic_clips(topics, atoms)
    assert len(clips) == 2
    assert clips[0].topic == "Intro"
    assert clips[0].block_refs == [0]
    assert clips[0].content == "intro"
    assert clips[1].block_refs == [1, 2]
    assert clips[1].content == "body\nend"


def test_build_nested_topic_tree_dedups_parent_refs() -> None:
    atoms = _atoms((0, "h"), (1, "a"), (2, "b"))
    topics = [
        {
            "topic": "Parent",
            "summary": "ps",
            "block_refs": "0-2",
            "children": [
                {"topic": "Child1", "summary": "s1", "block_refs": "1"},
                {"topic": "Child2", "summary": "s2", "block_refs": "2"},
            ],
        },
    ]
    clips = build_topic_clips(topics, atoms)
    # Two siblings, no overlap; parent is not heading-only (3 blocks).
    # collapse leaves the tree intact; dedup removes [1, 2] from parent.
    assert len(clips) == 1
    assert clips[0].topic == "Parent"
    assert clips[0].block_refs == [0]
    assert clips[0].content == "h"
    assert len(clips[0].children) == 2
    assert clips[0].children[0].block_refs == [1]
    assert clips[0].children[1].block_refs == [2]


def test_build_respects_topic_max_depth() -> None:
    # Build a chain deeper than TOPIC_MAX_DEPTH; deeper levels should be dropped.
    def _nest(depth: int) -> dict[str, object]:
        node: dict[str, object] = {"topic": f"L{depth}", "summary": "", "block_refs": ""}
        if depth > 0:
            node["children"] = [_nest(depth - 1)]
        return node

    too_deep = _nest(TOPIC_MAX_DEPTH + 2)
    clips = build_topic_clips([too_deep], [])

    def _depth(c: TopicClip) -> int:
        return 1 + max((_depth(ch) for ch in c.children), default=0)

    assert _depth(clips[0]) <= TOPIC_MAX_DEPTH + 1


# ── collapse_trivial_children ────────────────────────────────────────


def test_collapse_overlapping_children_absorbs_into_parent() -> None:
    # The trigger is "fewer unique refs than children" — fully-overlapping
    # siblings (both pointing at [1, 2]) yield 2 uniques < 2 children = false;
    # we need actual redundancy (here both children point at block 1 only).
    atom_map = {0: "h", 1: "x"}
    parent = TopicClip(
        topic="P",
        summary="s",
        block_refs=[0],
        children=[
            TopicClip(topic="C1", summary="", block_refs=[1]),
            TopicClip(topic="C2", summary="", block_refs=[1]),  # redundant
        ],
    )
    collapse_trivial_children([parent], atom_map)
    assert parent.children == []
    assert parent.block_refs == [0, 1]


def test_collapse_heading_only_parent_promotes_child() -> None:
    atom_map = {0: "h", 1: "body"}
    parent = TopicClip(
        topic="Heading",
        summary="hs",
        block_refs=[0],
        children=[TopicClip(topic="Child", summary="cs", block_refs=[1])],
    )
    collapse_trivial_children([parent], atom_map)
    assert parent.topic == "Child"
    assert parent.summary == "cs"
    assert parent.block_refs == [0, 1]
    assert parent.children == []


def test_collapse_recurses_bottom_up() -> None:
    # Inner over-split collapse fires first (C1 + C2 both at [2]).
    # Then outer heading-only single-child collapse fires (GP is [0], one child).
    atom_map = {0: "h", 1: "x", 2: "y"}
    grandparent = TopicClip(
        topic="GP",
        summary="",
        block_refs=[0],
        children=[
            TopicClip(
                topic="P",
                summary="ps",
                block_refs=[1],
                children=[
                    TopicClip(topic="C1", summary="", block_refs=[2]),
                    TopicClip(topic="C2", summary="", block_refs=[2]),  # redundant
                ],
            ),
        ],
    )
    collapse_trivial_children([grandparent], atom_map)
    # GP promotes inner P after P collapses its overlapping children.
    assert grandparent.children == []
    assert grandparent.topic == "P"
    assert grandparent.summary == "ps"
    assert grandparent.block_refs == [0, 1, 2]


# ── deduplicate_parent_refs ──────────────────────────────────────────


def test_deduplicate_removes_parent_refs_present_in_children() -> None:
    atom_map = {0: "h", 1: "a", 2: "b"}
    parent = TopicClip(
        topic="P",
        summary="",
        block_refs=[0, 1, 2],  # 1 and 2 also live in the child
        children=[TopicClip(topic="C", summary="", block_refs=[1, 2])],
    )
    deduplicate_parent_refs([parent], atom_map)
    assert parent.block_refs == [0]
    assert parent.content == "h"  # rebuilt from remaining ref
    assert parent.children[0].block_refs == [1, 2]


def test_deduplicate_recurses_into_grandchildren() -> None:
    atom_map = {0: "h", 1: "x", 2: "y"}
    gp = TopicClip(
        topic="GP",
        summary="",
        block_refs=[0, 1, 2],
        children=[
            TopicClip(
                topic="P",
                summary="",
                block_refs=[1, 2],
                children=[TopicClip(topic="C", summary="", block_refs=[2])],
            ),
        ],
    )
    deduplicate_parent_refs([gp], atom_map)
    assert gp.block_refs == [0]
    assert gp.children[0].block_refs == [1]
    assert gp.children[0].children[0].block_refs == [2]


# ── collect_all_block_refs ───────────────────────────────────────────


def test_collect_all_block_refs_nested() -> None:
    topics: list[dict[str, Any]] = [
        {
            "topic": "P",
            "block_refs": "0-1",
            "children": [
                {"topic": "C", "block_refs": "2,4"},
                {"topic": "C2", "block_refs": "5-6", "children": [{"topic": "GC", "block_refs": "7"}]},
            ],
        },
        {"topic": "Sibling", "block_refs": "8"},
    ]
    assert collect_all_block_refs(topics) == {0, 1, 2, 4, 5, 6, 7, 8}


def test_collect_all_block_refs_empty() -> None:
    assert collect_all_block_refs([]) == set()
