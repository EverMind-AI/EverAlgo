"""Unit tests for ``_doc_root`` and ``_flatten``."""

from __future__ import annotations

from everalgo.knowledge._doc_root import build_doc_root
from everalgo.knowledge._flatten import flatten
from everalgo.knowledge._topic_build import TopicClip

# ── build_doc_root ───────────────────────────────────────────────────


def test_build_doc_root_empty_forest() -> None:
    root = build_doc_root([], doc_title="Some Title", doc_summary="A short summary.")
    assert root.topic == "Some Title"
    assert root.summary == "A short summary."
    assert root.content == ""
    assert root.block_refs == []
    assert root.content_labels == []
    assert root.children == []


def test_build_doc_root_wraps_forest_verbatim() -> None:
    forest = [
        TopicClip(topic="A", summary="sa", block_refs=[0]),
        TopicClip(topic="B", summary="sb", block_refs=[1]),
    ]
    root = build_doc_root(
        forest,
        doc_title="Document",
        doc_summary="doc summary",
        doc_content_labels=["financial"],
    )
    assert root.topic == "Document"
    assert root.content_labels == ["financial"]
    assert root.children == forest


def test_build_doc_root_accepts_long_title_unrestricted() -> None:
    # Root is constructed programmatically — the <=20 chars LLM cap does not apply.
    long_title = "An exceptionally long document title with many descriptive words"
    root = build_doc_root([], doc_title=long_title, doc_summary="x")
    assert root.topic == long_title
    assert len(root.topic) > 20


# ── flatten ──────────────────────────────────────────────────────────


def test_flatten_single_root_only() -> None:
    root = TopicClip(topic="Doc", summary="doc summary")
    out = flatten(root, doc_id="d1")
    assert len(out) == 1
    km = out[0]
    assert km.doc_id == "d1"
    assert km.topic_index == 0
    assert km.depth == 0
    assert km.parent_index is None
    assert km.children_index == []
    assert km.topic_path == "Doc"
    assert km.topic == "Doc"
    assert km.summary == "doc summary"


def test_flatten_dfs_order_and_indices() -> None:
    # Tree:
    #   Doc (root)
    #   ├── A
    #   │   ├── A.1
    #   │   └── A.2
    #   └── B
    root = TopicClip(
        topic="Doc",
        summary="ds",
        children=[
            TopicClip(
                topic="A",
                summary="as",
                block_refs=[0],
                children=[
                    TopicClip(topic="A.1", summary="a1s", block_refs=[1]),
                    TopicClip(topic="A.2", summary="a2s", block_refs=[2]),
                ],
            ),
            TopicClip(topic="B", summary="bs", block_refs=[3]),
        ],
    )
    out = flatten(root, doc_id="d1")
    assert [km.topic for km in out] == ["Doc", "A", "A.1", "A.2", "B"]
    assert [km.topic_index for km in out] == [0, 1, 2, 3, 4]
    assert [km.depth for km in out] == [0, 1, 2, 2, 1]
    assert [km.parent_index for km in out] == [None, 0, 1, 1, 0]
    assert [km.children_index for km in out] == [[1, 4], [2, 3], [], [], []]
    assert [km.topic_path for km in out] == [
        "Doc",
        "Doc > A",
        "Doc > A > A.1",
        "Doc > A > A.2",
        "Doc > B",
    ]


def test_flatten_propagates_block_refs_and_content() -> None:
    root = TopicClip(
        topic="Doc",
        summary="ds",
        children=[TopicClip(topic="A", summary="as", content="payload", block_refs=[5, 6])],
    )
    out = flatten(root, doc_id="d1")
    assert out[1].content == "payload"
    assert out[1].block_refs == [5, 6]


def test_flatten_returns_copies_not_aliases() -> None:
    inner_refs = [1, 2]
    root = TopicClip(
        topic="Doc",
        summary="ds",
        children=[TopicClip(topic="A", summary="as", block_refs=inner_refs)],
    )
    out = flatten(root, doc_id="d1")
    out[1].block_refs.append(99)
    assert inner_refs == [1, 2]  # original is untouched


def test_flatten_default_category_id_is_empty() -> None:
    root = TopicClip(topic="Doc", summary="ds", children=[TopicClip(topic="A", summary="as")])
    out = flatten(root, doc_id="d1")
    assert all(km.category_id == "" for km in out)


def test_flatten_propagates_category_id_to_every_node() -> None:
    # Tree:
    #   Doc
    #   ├── A
    #   │   └── A.1
    #   └── B
    root = TopicClip(
        topic="Doc",
        summary="ds",
        children=[
            TopicClip(
                topic="A",
                summary="as",
                children=[TopicClip(topic="A.1", summary="a1s")],
            ),
            TopicClip(topic="B", summary="bs"),
        ],
    )
    out = flatten(root, doc_id="d1", category_id="how-to")
    assert [km.category_id for km in out] == ["how-to"] * 4
