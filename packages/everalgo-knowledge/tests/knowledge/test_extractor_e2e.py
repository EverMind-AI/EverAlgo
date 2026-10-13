"""End-to-end tests for ``KnowledgeExtractor.aextract``.

Uses ``FakeLLMClient`` with scripted JSON responses to drive the full
preprocess → topic-tree → postprocess → build → flatten pipeline without
touching a real LLM.
"""

from __future__ import annotations

import inspect
import json

import pytest

from everalgo.knowledge import KnowledgeExtractor
from everalgo.knowledge import extractor as extractor_module
from everalgo.knowledge._replies import (
    CategoryReply,
    ContentMergeReply,
    TopicMergeReply,
    TopicSplitReply,
    TopicTreeReply,
    UncoveredAssignReply,
)
from everalgo.llm import LLMOutputError
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import CategorySpec, ParsedContent

from ._scripted import assign_reply, content_merge_reply, topic, topic_tree_reply


async def test_aextract_empty_text_returns_empty() -> None:
    parsed = ParsedContent(text="", mime="text/plain")
    client = FakeLLMClient(responses=[])  # not consumed
    out = await KnowledgeExtractor(llm=client).aextract(parsed, doc_id="d0", title="Empty")
    assert out == []
    assert client.call_count == 0


async def test_aextract_unparsable_llm_reply_raises() -> None:
    parsed = ParsedContent(text="paragraph one\n\nparagraph two", mime="text/plain")
    client = FakeLLMClient(responses=["not json at all"])
    with pytest.raises(LLMOutputError, match="knowledge topic_tree_extract reply is not a valid TopicTreeReply"):
        await KnowledgeExtractor(llm=client).aextract(parsed, doc_id="d1", title="Doc")
    assert client.call_count == 1


async def test_aextract_reply_missing_a_field_raises() -> None:
    """A leaf without ``children`` was accepted by the lenient parser; the strict model requires it."""
    parsed = ParsedContent(text="only one line", mime="text/plain")
    reply = json.loads(topic_tree_reply("s", topic("T", "0")))
    del reply["topics"][0]["children"]
    client = FakeLLMClient(responses=[json.dumps(reply)])
    with pytest.raises(LLMOutputError, match=r"topics\.0\.children"):
        await KnowledgeExtractor(llm=client).aextract(parsed, doc_id="d", title="Doc")


async def test_aextract_reply_with_an_extra_field_raises() -> None:
    parsed = ParsedContent(text="only one line", mime="text/plain")
    reply = json.loads(topic_tree_reply("s", topic("T", "0")))
    reply["confidence"] = 0.9
    client = FakeLLMClient(responses=[json.dumps(reply)])
    with pytest.raises(LLMOutputError, match="confidence"):
        await KnowledgeExtractor(llm=client).aextract(parsed, doc_id="d", title="Doc")


async def test_aextract_client_exception_propagates_unchanged() -> None:
    class _HostError(Exception):
        pass

    parsed = ParsedContent(text="only one line", mime="text/plain")
    client = FakeLLMClient(responses=[_HostError("upstream down")])
    with pytest.raises(_HostError, match="upstream down"):
        await KnowledgeExtractor(llm=client).aextract(parsed, doc_id="d", title="Doc")


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
    llm_response = topic_tree_reply(
        "Two-level doc.",
        topic("Section A", "0", topic("A.1", "1"), topic("A.2", "2")),
    )
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
    llm_response = topic_tree_reply("s", topic("T", "0"))
    client = FakeLLMClient(responses=[llm_response])
    out = await KnowledgeExtractor(llm=client).aextract(parsed, doc_id="d3", title="")
    assert out[0].topic == "Untitled"


async def test_aextract_triggers_assign_when_blocks_uncovered() -> None:
    # Three atoms, but the extractor only claims block 0 — block 1 and 2 are uncovered
    # and should drive an assign-uncovered LLM pass.
    parsed = ParsedContent(text="alpha\n\nbeta\n\ngamma", mime="text/plain")
    extract_response = topic_tree_reply("only first claimed", topic("T", "0"))
    client = FakeLLMClient(responses=[extract_response, assign_reply((1, 0), (2, 0))])

    out = await KnowledgeExtractor(llm=client).aextract(parsed, doc_id="d4", title="Doc")

    assert [call.response_format for call in client.calls] == [TopicTreeReply, UncoveredAssignReply]
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

    extract_batch_1 = topic_tree_reply(
        "First half summary.", topic("Death", "0"), topic("Setup", "1"), labels=("financial",)
    )
    extract_batch_2 = topic_tree_reply(
        "Second half summary.", topic("Murder", "2"), topic("Aftermath", "3"), labels=("financial", "violence")
    )
    content_merge = content_merge_reply("Combined narrative.")
    topic_merge = (
        '{"merges": [{"indices": [0, 2], "topic": "Death/Murder", "summary": "Both refer to the same death scene."}]}'
    )

    client = FakeLLMClient(responses=[extract_batch_1, extract_batch_2, content_merge, topic_merge])
    return parsed, client


async def test_aextract_multi_batch_runs_content_and_topic_merge(monkeypatch: pytest.MonkeyPatch) -> None:
    """Multi-batch documents go through the cross-batch merge LLM passes."""
    parsed, client = _multi_batch_fixture(monkeypatch)
    out = await KnowledgeExtractor(llm=client).aextract(parsed, doc_id="multi", title="Long Doc")

    assert [call.response_format for call in client.calls] == [
        TopicTreeReply,
        TopicTreeReply,
        ContentMergeReply,
        TopicMergeReply,
    ]

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
    llm_response = topic_tree_reply("A doc.", topic("Intro", "0-2"))
    client = FakeLLMClient(responses=[llm_response])
    out = await KnowledgeExtractor(llm=client).aextract(parsed, doc_id="d", title="T")

    assert client.call_count == 1  # no classifier call
    assert len(out) >= 2
    assert all(km.category_id == "" for km in out)


async def test_aextract_with_categories_runs_classifier_and_denormalizes() -> None:
    """When ``categories`` is passed, classifier fires once and result is on every node."""
    parsed = ParsedContent(text="alpha\n\nbeta", mime="text/plain")
    extract_response = topic_tree_reply("A how-to about alpha and beta.", topic("Setup", "0"), topic("Run", "1"))
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

    assert [call.response_format for call in client.calls] == [TopicTreeReply, CategoryReply]  # no postprocess
    assert len(out) == 3  # root + 2 topics
    assert all(km.category_id == "how-to" for km in out)


async def test_aextract_with_pre_supplied_category_id_skips_classifier() -> None:
    """Explicit ``category_id=`` short-circuits the classifier entirely."""
    parsed = ParsedContent(text="alpha\n\nbeta", mime="text/plain")
    extract_response = topic_tree_reply("summary", topic("Only", "0-1"))
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
    extract_response = topic_tree_reply("summary", topic("X", "0-1"))
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


# ── custom_instructions: one value, every stage ──────────────────────


def _every_stage_fixture(monkeypatch: pytest.MonkeyPatch) -> tuple[ParsedContent, list[CategorySpec], list[str]]:
    """Two batches, four topics to merge, a taxonomy, a leaf with two sub-headings and an orphan block: all six stages."""
    parsed = ParsedContent(
        text="## H1\n\n" + "x" * 1100 + "\n\n## H2\n\n" + "y" * 1100 + "\n\ngamma\n\ndelta", mime="text/plain"
    )
    monkeypatch.setattr(extractor_module, "split_and_batch_content", lambda atoms: [atoms[:4], atoms[4:]])
    replies = [
        topic_tree_reply("first", topic("Big", "0-3")),
        topic_tree_reply("second", topic("Gamma", "4"), topic("Gamma bis", "4"), topic("Gamma ter", "4")),
        content_merge_reply("merged"),
        json.dumps({"merges": []}),
        '{"category_id": "how-to"}',
        json.dumps(
            {
                "results": [
                    {
                        "original_topic": "Big",
                        "should_split": False,
                        "reason": "r",
                        "block_refs": None,
                        "children": None,
                    }
                ]
            }
        ),
        assign_reply((5, 0)),
    ]
    return parsed, [CategorySpec(id="how-to", description="Tutorials.")], replies


async def _every_stage_systems(
    monkeypatch: pytest.MonkeyPatch, custom_instructions: str | None = None
) -> FakeLLMClient:
    parsed, categories, replies = _every_stage_fixture(monkeypatch)
    client = FakeLLMClient(responses=replies)
    await KnowledgeExtractor(llm=client).aextract(
        parsed, doc_id="d", title="Doc", categories=categories, custom_instructions=custom_instructions
    )
    return client


async def test_aextract_custom_instructions_close_the_system_message_of_every_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = "Name topics in the document's own terms.\nKeep {{title}} literal."
    default = await _every_stage_systems(monkeypatch)

    with_custom = await _every_stage_systems(monkeypatch, custom_instructions=text)

    assert [call.response_format for call in with_custom.calls] == [
        TopicTreeReply,
        TopicTreeReply,
        ContentMergeReply,
        TopicMergeReply,
        CategoryReply,
        TopicSplitReply,
        UncoveredAssignReply,
    ]
    block = f"\n\n<custom_instructions>\n{text}\n</custom_instructions>"
    assert [call.messages[0].content for call in with_custom.calls] == [
        f"{call.messages[0].content}{block}" for call in default.calls
    ]
    assert [call.messages[1] for call in with_custom.calls] == [call.messages[1] for call in default.calls]


@pytest.mark.parametrize("blank", ["", "  \n\t "])
async def test_aextract_blank_custom_instructions_are_the_same_as_none(
    monkeypatch: pytest.MonkeyPatch, blank: str
) -> None:
    default = await _every_stage_systems(monkeypatch)

    blank_run = await _every_stage_systems(monkeypatch, custom_instructions=blank)

    assert [call.messages for call in blank_run.calls] == [call.messages for call in default.calls]
    assert all("custom_instructions" not in str(call.messages[0].content) for call in default.calls)


def test_aextract_takes_custom_instructions_and_no_prompt() -> None:
    parameters = inspect.signature(KnowledgeExtractor.aextract).parameters

    assert parameters["custom_instructions"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["custom_instructions"].default is None
    assert "prompt" not in parameters


async def test_aextract_rejects_the_removed_prompt_override() -> None:
    client = FakeLLMClient(responses=[])
    parsed = ParsedContent(text="alpha", mime="text/plain")

    with pytest.raises(TypeError, match="prompt"):
        await KnowledgeExtractor(llm=client).aextract(parsed, prompt="Rules.")  # type: ignore[call-arg]  # the removed keyword
    assert client.call_count == 0
