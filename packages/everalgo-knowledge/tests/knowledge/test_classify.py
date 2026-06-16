"""Unit tests for ``aclassify_category`` (FakeLLMClient-driven)."""

from __future__ import annotations

from everalgo.knowledge import aclassify_category
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


async def test_aclassify_category_out_of_set_falls_back_to_empty() -> None:
    client = FakeLLMClient(responses=['{"category_id": "tutorial"}'])  # not in taxonomy
    out = await aclassify_category(client, "Title", "Summary", _make_taxonomy())
    assert out == ""


async def test_aclassify_category_parse_failure_falls_back_to_empty() -> None:
    client = FakeLLMClient(responses=["not json at all, just prose"])
    out = await aclassify_category(client, "Title", "Summary", _make_taxonomy())
    assert out == ""


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


async def test_aclassify_category_non_string_id_falls_back_to_empty() -> None:
    client = FakeLLMClient(responses=['{"category_id": 42}'])
    out = await aclassify_category(client, "Title", "Summary", _make_taxonomy())
    assert out == ""


async def test_aclassify_category_strips_whitespace() -> None:
    client = FakeLLMClient(responses=['{"category_id": "  how-to  "}'])
    out = await aclassify_category(client, "Title", "Summary", _make_taxonomy())
    assert out == "how-to"


async def test_aclassify_category_taxonomy_rendered_in_prompt() -> None:
    """Verify each taxonomy id and description appears in the LLM message.

    The taxonomy is meant to be a stable prefix so prompt caching takes effect.
    """
    captured: list[str] = []

    async def handler(messages, **_):  # type: ignore[no-untyped-def]  # test-only callback; exact typing irrelevant
        from everalgo.llm.types import ChatResponse

        captured.append(messages[0].content)
        return ChatResponse(content='{"category_id": "how-to"}', model="fake", finish_reason="stop")

    client = FakeLLMClient(handler=handler)
    taxonomy = _make_taxonomy()
    await aclassify_category(client, "Title", "Some summary.", taxonomy)

    assert len(captured) == 1
    rendered = captured[0]
    for spec in taxonomy:
        assert spec.id in rendered
        assert spec.description in rendered
    assert "Title" in rendered
    assert "Some summary." in rendered
