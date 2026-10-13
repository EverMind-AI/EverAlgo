"""Unit tests for ``aclassify_category`` (FakeLLMClient-driven)."""

from __future__ import annotations

import pytest

from everalgo.knowledge import aclassify_category
from everalgo.knowledge._replies import CategoryReply
from everalgo.knowledge.prompts.en.category_classify import CATEGORY_CLASSIFY_SYSTEM_PROMPT
from everalgo.llm import LLMOutputError
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import CategorySpec


def _make_taxonomy() -> list[CategorySpec]:
    return [
        CategorySpec(id="how-to", description="Step-by-step tutorials and walkthroughs."),
        CategorySpec(id="reference", description="API references, lookup tables, specifications."),
        CategorySpec(id="news", description="Time-bound announcements and release notes."),
    ]


async def test_aclassify_category_in_set_returned_verbatim() -> None:
    client = FakeLLMClient(responses=['{"category_id": "how-to"}'])
    out = await aclassify_category(client, "Title", "A short summary.", _make_taxonomy())
    assert out == "how-to"
    assert client.call_count == 1
    assert client.calls[0].response_format is CategoryReply


async def test_aclassify_category_out_of_set_falls_back_to_empty() -> None:
    client = FakeLLMClient(responses=['{"category_id": "tutorial"}'])  # not in taxonomy
    out = await aclassify_category(client, "Title", "Summary", _make_taxonomy())
    assert out == ""


@pytest.mark.parametrize(
    "reply",
    [
        pytest.param("not json at all, just prose", id="not-json"),
        pytest.param("{}", id="missing-field"),
        pytest.param('{"category_id": "how-to", "confidence": 0.9}', id="extra-field"),
        pytest.param('{"category_id": 42}', id="non-string-id"),
    ],
)
async def test_aclassify_category_invalid_reply_raises(reply: str) -> None:
    client = FakeLLMClient(responses=[reply])
    with pytest.raises(LLMOutputError, match="knowledge category_classify reply is not a valid CategoryReply"):
        await aclassify_category(client, "Title", "Summary", _make_taxonomy())


async def test_aclassify_category_empty_taxonomy_short_circuits() -> None:
    client = FakeLLMClient(responses=[])  # never consumed
    out = await aclassify_category(client, "Title", "Summary", [])
    assert out == ""
    assert client.call_count == 0


async def test_aclassify_category_explicit_empty_response_is_empty() -> None:
    # Model legitimately decided "none of the categories fit" and returned "".
    client = FakeLLMClient(responses=['{"category_id": ""}'])
    out = await aclassify_category(client, "Title", "Summary", _make_taxonomy())
    assert out == ""


async def test_aclassify_category_strips_whitespace() -> None:
    client = FakeLLMClient(responses=['{"category_id": "  how-to  "}'])
    out = await aclassify_category(client, "Title", "Summary", _make_taxonomy())
    assert out == "how-to"


async def test_aclassify_category_taxonomy_closes_the_system_message_and_the_document_is_the_user_message() -> None:
    """The taxonomy is the system message's last block, the same for every document of a run; the document is data."""
    client = FakeLLMClient(responses=['{"category_id": "how-to"}', '{"category_id": "news"}'])
    taxonomy = _make_taxonomy()
    await aclassify_category(client, "Title", "Some summary.", taxonomy)
    await aclassify_category(client, "Other title", "Another summary.", taxonomy)

    first, second = ([message.content for message in call.messages] for call in client.calls)
    assert [message.role for message in client.calls[0].messages] == ["system", "user"]
    assert first[0] == second[0]
    taxonomy_block = "\n".join(f"- {spec.id}: {spec.description}" for spec in taxonomy)
    assert isinstance(first[0], str)
    assert first[0].endswith(f"<taxonomy>\n{taxonomy_block}\n</taxonomy>")
    assert (
        first[1]
        == "<document_title>\nTitle\n</document_title>\n\n<document_summary>\nSome summary.\n</document_summary>"
    )


async def test_aclassify_category_custom_instructions_follow_the_taxonomy_and_close_the_system_message() -> None:
    client = FakeLLMClient(responses=['{"category_id": "how-to"}'] * 2)
    text = "Treat changelogs as news. Keep {{title}} literal."
    await aclassify_category(client, "Title", "Summary", _make_taxonomy())
    await aclassify_category(client, "Title", "Summary", _make_taxonomy(), custom_instructions=text)

    default, with_custom = (str(call.messages[0].content) for call in client.calls)
    assert default.startswith(CATEGORY_CLASSIFY_SYSTEM_PROMPT)
    assert default.endswith("\n</taxonomy>")
    assert with_custom == f"{default}\n\n<custom_instructions>\n{text}\n</custom_instructions>"
    assert client.calls[0].messages[1] == client.calls[1].messages[1]


async def test_aclassify_category_rejects_the_removed_prompt_override() -> None:
    client = FakeLLMClient(responses=[])
    with pytest.raises(TypeError, match="prompt"):
        await aclassify_category(client, "Title", "Summary", _make_taxonomy(), prompt="Custom rules.")  # type: ignore[call-arg]  # the removed keyword
    assert client.call_count == 0
