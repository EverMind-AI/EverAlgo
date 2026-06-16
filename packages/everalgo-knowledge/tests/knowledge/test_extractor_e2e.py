"""End-to-end tests for ``KnowledgeExtractor.aextract``.

Uses ``FakeLLMClient`` with scripted JSON responses to drive the full
preprocess → topic-tree → postprocess → build → flatten pipeline without
touching a real LLM.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from everalgo.knowledge import KnowledgeExtractor
from everalgo.knowledge import extractor as extractor_module
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import CategorySpec, ParsedContent

if TYPE_CHECKING:
    import pytest


async def test_aextract_empty_text_returns_empty() -> None:
    parsed = ParsedContent(text="", mime="text/plain")
    client = FakeLLMClient(responses=[])  # not consumed
    out = await KnowledgeExtractor(llm=client).aextract(parsed, doc_id="d0", title="Empty")
    assert out == []
    assert client.call_count == 0


async def test_aextract_unparsable_llm_returns_empty() -> None:
    parsed = ParsedContent(text="paragraph one\n\nparagraph two", mime="text/plain")
    client = FakeLLMClient(responses=["not json at all"])
    out = await KnowledgeExtractor(llm=client).aextract(parsed, doc_id="d1", title="Doc")
    assert out == []
    assert client.call_count == 1


async def test_aextract_single_batch_simple_document() -> None:
    parsed = ParsedContent(
        text="first paragraph\n\nsecond paragraph\n\nthird paragraph",
        mime="text/markdown",
    )
    llm_response = """
    {
      "language": "English",
      "summary": "A sample doc with three paragraphs.",
      "subject": "Sample",
      "keywords": ["sample"],
      "content_labels": ["financial"],
      "topics": [
        {
          "topic": "Intro",
          "summary": "intro section",
          "block_refs": "0-2",
          "content_labels": [],
          "children": []
        }
      ]
    }
    """
    client = FakeLLMClient(responses=[llm_response])

    out = await KnowledgeExtractor(llm=client).aextract(parsed, doc_id="doc1", title="Sample Doc")

    # Synthetic root + one extracted topic.
    assert len(out) == 2
    assert client.call_count == 1  # no postprocess fired

    # Root assertions.
    root = out[0]
    assert root.doc_id == "doc1"
    assert root.topic_index == 0
    assert root.depth == 0
    assert root.parent_index is None
    assert root.topic == "Sample Doc"
    assert root.summary == "A sample doc with three paragraphs."
    assert root.children_index == [1]
    assert root.topic_path == "Sample Doc"
    assert root.content_labels == ["financial"]

    # Topic assertions.
    topic = out[1]
    assert topic.topic_index == 1
    assert topic.depth == 1
    assert topic.parent_index == 0
    assert topic.topic == "Intro"
    assert topic.block_refs == [0, 1, 2]
    assert topic.topic_path == "Sample Doc > Intro"


async def test_aextract_nested_topics_complete_pipeline() -> None:
    parsed = ParsedContent(text="alpha\n\nbeta\n\ngamma", mime="text/plain")
    llm_response = """
    {
      "summary": "Two-level doc.",
      "topics": [
        {
          "topic": "Section A",
          "summary": "alpha intro",
          "block_refs": "0",
          "children": [
            {"topic": "A.1", "summary": "beta", "block_refs": "1"},
            {"topic": "A.2", "summary": "gamma", "block_refs": "2"}
          ]
        }
      ]
    }
    """
    client = FakeLLMClient(responses=[llm_response])

    out = await KnowledgeExtractor(llm=client).aextract(parsed, doc_id="d2", title="Tech")

    # Root + Section A + A.1 + A.2.
    assert len(out) == 4
    assert [k.topic for k in out] == ["Tech", "Section A", "A.1", "A.2"]
    assert [k.depth for k in out] == [0, 1, 2, 2]
    assert [k.parent_index for k in out] == [None, 0, 1, 1]
    assert out[1].topic_path == "Tech > Section A"
    assert out[2].topic_path == "Tech > Section A > A.1"
    assert out[1].children_index == [2, 3]


async def test_aextract_default_title_fallback() -> None:
    parsed = ParsedContent(text="only one line", mime="text/plain")
    llm_response = '{"summary": "s", "topics": [{"topic": "T", "summary": "ts", "block_refs": "0"}]}'
    client = FakeLLMClient(responses=[llm_response])
    out = await KnowledgeExtractor(llm=client).aextract(parsed, doc_id="d3", title="")
    assert out[0].topic == "Untitled"


async def test_aextract_triggers_assign_when_blocks_uncovered() -> None:
    # Three atoms, but the extractor only claims block 0 — block 1 and 2 are uncovered
    # and should drive an assign-uncovered LLM pass.
    parsed = ParsedContent(text="alpha\n\nbeta\n\ngamma", mime="text/plain")
    extract_response = """
    {
      "summary": "only first claimed",
      "topics": [
        {"topic": "T", "summary": "ts", "block_refs": "0", "children": []}
      ]
    }
    """
    assign_response = """
    {
      "assignments": [
        {"block_id": 1, "topic_index": 0, "reason": "x"},
        {"block_id": 2, "topic_index": 0, "reason": "y"}
      ]
    }
    """
    client = FakeLLMClient(responses=[extract_response, assign_response])

    out = await KnowledgeExtractor(llm=client).aextract(parsed, doc_id="d4", title="Doc")

    assert client.call_count == 2
    # Root + one topic; topic should now own all three blocks via the assign pass.
    assert len(out) == 2
    assert out[1].block_refs == [0, 1, 2]


def _multi_batch_fixture(monkeypatch: pytest.MonkeyPatch) -> tuple[ParsedContent, FakeLLMClient]:
    """Build a multi-batch scenario with a cross-batch topic duplicate (Death ≡ Murder)."""
    parsed = ParsedContent(text="alpha\n\nbeta\n\ngamma\n\ndelta", mime="text/plain")

    def _two_batches(atoms: list[tuple[int, str]]) -> list[list[tuple[int, str]]]:
        midpoint = len(atoms) // 2
        return [atoms[:midpoint], atoms[midpoint:]]

    monkeypatch.setattr(extractor_module, "split_and_batch_content", _two_batches)

    extract_batch_1 = (
        '{"summary": "First half summary.", "content_labels": ["financial"],'
        ' "topics": [{"topic": "Death", "summary": "Death of X", "block_refs": "0", "children": []},'
        ' {"topic": "Setup", "summary": "Setup scene", "block_refs": "1", "children": []}]}'
    )
    extract_batch_2 = (
        '{"summary": "Second half summary.", "content_labels": ["financial", "violence"],'
        ' "topics": [{"topic": "Murder", "summary": "Murder of X", "block_refs": "2", "children": []},'
        ' {"topic": "Aftermath", "summary": "Aftermath", "block_refs": "3", "children": []}]}'
    )
    content_merge = '{"summary": "Combined narrative."}'
    topic_merge = (
        '{"merges": [{"indices": [0, 2], "topic": "Death/Murder", "summary": "Both refer to the same death scene."}]}'
    )

    client = FakeLLMClient(responses=[extract_batch_1, extract_batch_2, content_merge, topic_merge])
    return parsed, client


async def test_aextract_multi_batch_runs_content_and_topic_merge(monkeypatch: pytest.MonkeyPatch) -> None:
    """Multi-batch documents go through the cross-batch merge LLM passes."""
    parsed, client = _multi_batch_fixture(monkeypatch)
    out = await KnowledgeExtractor(llm=client).aextract(parsed, doc_id="multi", title="Long Doc")

    assert client.call_count == 4

    root = out[0]
    assert root.topic == "Long Doc"
    assert root.summary == "Combined narrative."
    assert root.content_labels == ["financial", "violence"]

    topic_names = [k.topic for k in out[1:]]
    assert "Death/Murder" in topic_names
    assert "Setup" in topic_names
    assert "Aftermath" in topic_names

    merged_topic = next(k for k in out if k.topic == "Death/Murder")
    assert set(merged_topic.block_refs) == {0, 2}


# ── category classification (write path) ─────────────────────────────


async def test_aextract_without_categories_leaves_category_id_empty() -> None:
    """Backward compatibility: callers that omit ``categories=`` get the previous behavior."""
    parsed = ParsedContent(text="first\n\nsecond\n\nthird", mime="text/plain")
    llm_response = """
    {
      "summary": "A doc.",
      "topics": [{"topic": "Intro", "summary": "intro", "block_refs": "0-2", "children": []}]
    }
    """
    client = FakeLLMClient(responses=[llm_response])
    out = await KnowledgeExtractor(llm=client).aextract(parsed, doc_id="d", title="T")

    assert client.call_count == 1  # no classifier call
    assert len(out) >= 2
    assert all(km.category_id == "" for km in out)


async def test_aextract_with_categories_runs_classifier_and_denormalizes() -> None:
    """When ``categories`` is passed, classifier fires once and result is on every node."""
    parsed = ParsedContent(text="alpha\n\nbeta", mime="text/plain")
    extract_response = """
    {
      "summary": "A how-to about alpha and beta.",
      "topics": [
        {"topic": "Setup", "summary": "setup step", "block_refs": "0", "children": []},
        {"topic": "Run", "summary": "run step", "block_refs": "1", "children": []}
      ]
    }
    """
    classify_response = '{"category_id": "how-to"}'
    client = FakeLLMClient(responses=[extract_response, classify_response])

    categories = [
        CategorySpec(id="how-to", description="Step-by-step tutorials."),
        CategorySpec(id="reference", description="API references."),
    ]
    out = await KnowledgeExtractor(llm=client).aextract(
        parsed,
        doc_id="d",
        title="T",
        categories=categories,
    )

    assert client.call_count == 2  # extract + classify (no postprocess fired)
    assert len(out) == 3  # root + 2 topics
    assert all(km.category_id == "how-to" for km in out)


async def test_aextract_with_pre_supplied_category_id_skips_classifier() -> None:
    """Explicit ``category_id=`` short-circuits the classifier entirely."""
    parsed = ParsedContent(text="alpha\n\nbeta", mime="text/plain")
    extract_response = """
    {
      "summary": "summary",
      "topics": [{"topic": "Only", "summary": "only", "block_refs": "0-1", "children": []}]
    }
    """
    client = FakeLLMClient(responses=[extract_response])  # only one response — no classifier call
    categories = [CategorySpec(id="how-to", description="Tutorials.")]

    out = await KnowledgeExtractor(llm=client).aextract(
        parsed,
        doc_id="d",
        title="T",
        categories=categories,
        category_id="reference",
    )

    assert client.call_count == 1
    assert all(km.category_id == "reference" for km in out)


async def test_aextract_with_categories_but_classifier_misses_leaves_empty() -> None:
    """Classifier predicts out-of-set → falls back to ``""`` and all nodes carry it."""
    parsed = ParsedContent(text="alpha\n\nbeta", mime="text/plain")
    extract_response = """
    {
      "summary": "summary",
      "topics": [{"topic": "X", "summary": "x", "block_refs": "0-1", "children": []}]
    }
    """
    bad_classify = '{"category_id": "not-in-taxonomy"}'
    client = FakeLLMClient(responses=[extract_response, bad_classify])
    categories = [CategorySpec(id="how-to", description="Tutorials.")]

    out = await KnowledgeExtractor(llm=client).aextract(
        parsed,
        doc_id="d",
        title="T",
        categories=categories,
    )

    assert client.call_count == 2
    assert all(km.category_id == "" for km in out)
