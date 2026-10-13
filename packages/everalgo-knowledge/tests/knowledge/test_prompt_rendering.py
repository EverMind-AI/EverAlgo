"""Every knowledge call site sends the model a prompt with no slot left in it.

Asserting on the template alone would not catch the failure this package is exposed to: the
prompts and their call sites were migrated together, and a call site that went back to
``str.format`` would turn the template's ``{{title}}`` into a literal ``{title}`` and ship it —
no exception, no missing field, just the word ``{title}`` where the document title belonged.
So each test drives the real function against a ``FakeLLMClient`` and reads the prompt the client
actually received.

The check rejects both brace shapes for that reason: ``{{title}}`` means the renderer never
substituted it, ``{title}`` means something other than the renderer did.
"""

from __future__ import annotations

import re
from typing import Any

from everalgo.knowledge import aclassify_category
from everalgo.knowledge._batch_merge import amerge_doc_summary, amerge_topics
from everalgo.knowledge._postprocess import aassign_uncovered_blocks, asplit_unsplit_leaves
from everalgo.knowledge.extractor import _aextract_batch
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import CategorySpec

from ._scripted import assign_reply, content_merge_reply, topic_tree_reply

_SLOT_LEFT_BEHIND = re.compile(r"\{\{[A-Za-z_][A-Za-z0-9_]*\}\}|(?<!\{)\{[A-Za-z_][A-Za-z0-9_]*\}(?!\})")


def _assert_no_slot_reached_the_model(client: FakeLLMClient) -> None:
    assert client.call_count > 0, "the call site issued no LLM call, so nothing was checked"
    for record in client.calls:
        for message in record.messages:
            leftover = _SLOT_LEFT_BEHIND.findall(str(message.content))
            assert not leftover, f"unsubstituted slots {leftover} in: {message.content}"


async def test_classify_category_sends_a_fully_rendered_prompt() -> None:
    client = FakeLLMClient(responses=['{"category_id": "how-to"}'])
    taxonomy = [CategorySpec(id="how-to", description="Step-by-step tutorials.")]

    await aclassify_category(client, "Doc", "A short summary.", taxonomy)

    _assert_no_slot_reached_the_model(client)


async def test_extract_batch_sends_a_fully_rendered_prompt() -> None:
    client = FakeLLMClient(responses=[topic_tree_reply("s")])

    await _aextract_batch(client, "Doc", [(1, "first para"), (2, "second para")], None)

    _assert_no_slot_reached_the_model(client)


async def test_split_unsplit_leaves_sends_a_fully_rendered_prompt() -> None:
    client = FakeLLMClient(responses=['{"results": []}'])
    topics: list[dict[str, Any]] = [{"topic": "Big", "block_refs": "0-3"}]
    unsplit = [{"topic_name": "Big", "path": [0], "block_ids": [0, 1], "heading_ids": [0], "total_chars": 99}]

    await asplit_unsplit_leaves(client, "Doc", topics, unsplit, {0: "## A", 1: "body A"})

    _assert_no_slot_reached_the_model(client)


async def test_assign_uncovered_blocks_sends_a_fully_rendered_prompt() -> None:
    client = FakeLLMClient(responses=[assign_reply()])
    topics: list[dict[str, Any]] = [{"topic": "T", "block_refs": "0"}]

    await aassign_uncovered_blocks(client, "Doc", topics, [{"block_id": 1, "text": "orphan"}], {0: "a", 1: "orphan"})

    _assert_no_slot_reached_the_model(client)


async def test_merge_doc_summary_sends_a_fully_rendered_prompt() -> None:
    client = FakeLLMClient(responses=[content_merge_reply("merged")])
    batches: list[dict[str, Any]] = [{"summary": "s1"}, {"summary": "s2"}]

    await amerge_doc_summary(client, "Doc", batches)

    _assert_no_slot_reached_the_model(client)


async def test_merge_topics_sends_a_fully_rendered_prompt() -> None:
    client = FakeLLMClient(responses=['{"merges": []}'])
    topics: list[dict[str, Any]] = [{"topic": f"T{i}", "summary": f"s{i}", "block_refs": str(i)} for i in range(10)]

    await amerge_topics(client, "Doc", topics)

    _assert_no_slot_reached_the_model(client)
