"""Tests for everalgo.user_memory.principle — PrincipleExtractor.

Empty cluster is success without an LLM call. source_entry_ids are a subset of caller-supplied ids.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.llm.types import ChatResponse
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import Decision, Principle
from everalgo.user_memory import OutputLanguage


def _dc(ts: int, **overrides: Any) -> Decision:
    base: dict[str, Any] = {
        "owner_id": None,
        "title": "topic",
        "decision": "some decision",
        "reason": "some reason",
        "timestamp": ts,
    }
    base.update(overrides)
    return Decision.model_validate(base)


def _cluster() -> list[tuple[str, Decision]]:
    return [
        (
            "dc_001",
            _dc(
                1000,
                title="Agent Core language",
                decision="Use Python for the core Agent Runtime.",
                reason="Faster iteration on agent capability.",
            ),
        ),
        (
            "dc_002",
            _dc(
                2000,
                title="Experiment velocity",
                decision="Prefer shipping experiments quickly.",
                reason="The agent surface is still changing weekly.",
            ),
        ),
        (
            "dc_003",
            _dc(
                1500,
                title="Defer Rust on the core",
                decision="Do not rewrite the core in Rust yet.",
                reason="Premature optimisation would slow iteration.",
            ),
        ),
    ]


def _principle_json(**overrides: Any) -> str:
    item: dict[str, Any] = {
        "title": "Iteration over premature optimisation",
        "statement": "Agent architecture prioritises iteration speed.",
        "source_entry_ids": ["dc_001", "dc_002", "dc_003"],
    }
    item.update(overrides)
    return json.dumps({"principles": [item]})


# ==========================================================================
# Validation / empty cluster
# ==========================================================================


async def test_empty_cluster_returns_empty_list_without_llm() -> None:
    fake = FakeLLMClient(responses=[_principle_json()])
    from everalgo.user_memory.principle import PrincipleExtractor

    result = await PrincipleExtractor(llm=fake).aextract([], owner_id="u1")

    assert result == []
    assert fake.call_count == 0


async def test_empty_owner_id_raises_before_llm() -> None:
    fake = FakeLLMClient(responses=[_principle_json()])
    from everalgo.user_memory.principle import PrincipleExtractor

    with pytest.raises(ValueError, match="owner_id"):
        await PrincipleExtractor(llm=fake).aextract(_cluster(), owner_id="  ")

    assert fake.call_count == 0


async def test_empty_entry_id_raises_before_llm() -> None:
    fake = FakeLLMClient(responses=[_principle_json()])
    from everalgo.user_memory.principle import PrincipleExtractor

    with pytest.raises(ValueError, match="entry_id"):
        await PrincipleExtractor(llm=fake).aextract([("  ", _dc(1))], owner_id="u1")

    assert fake.call_count == 0


async def test_duplicate_entry_id_raises_before_llm() -> None:
    fake = FakeLLMClient(responses=[_principle_json()])
    from everalgo.user_memory.principle import PrincipleExtractor

    with pytest.raises(ValueError, match="duplicate entry_id"):
        await PrincipleExtractor(llm=fake).aextract([("dc_001", _dc(1)), ("dc_001", _dc(2))], owner_id="u1")

    assert fake.call_count == 0


# ==========================================================================
# Synthesis
# ==========================================================================


async def test_aextract_synthesises_principles_from_a_cluster() -> None:
    fake = FakeLLMClient(responses=[_principle_json()])
    from everalgo.user_memory.principle import PrincipleExtractor

    result = await PrincipleExtractor(llm=fake).aextract(_cluster(), owner_id="u1")

    assert len(result) == 1
    pr = result[0]
    assert isinstance(pr, Principle)
    assert pr.owner_id == "u1"
    assert pr.title == "Iteration over premature optimisation"
    assert pr.statement == "Agent architecture prioritises iteration speed."
    assert pr.source_entry_ids == ["dc_001", "dc_002", "dc_003"]
    assert pr.timestamp == 2000  # max of cited sources
    assert not hasattr(pr, "decision")


async def test_aextract_sends_entry_ids_in_the_prompt() -> None:
    fake = FakeLLMClient(responses=[_principle_json()])
    from everalgo.user_memory.principle import PrincipleExtractor

    await PrincipleExtractor(llm=fake).aextract(_cluster(), owner_id="u1")

    prompt_text = fake.calls[0].messages[0].content
    assert "id=dc_001" in prompt_text
    assert "Use Python for the core Agent Runtime." in prompt_text


async def test_source_entry_ids_unknown_ids_are_stripped() -> None:
    fake = FakeLLMClient(responses=[_principle_json(source_entry_ids=["dc_001", "dc_unknown", "dc_002", "dc_001"])])
    from everalgo.user_memory.principle import PrincipleExtractor

    result = await PrincipleExtractor(llm=fake).aextract(_cluster(), owner_id="u1")

    assert result[0].source_entry_ids == ["dc_001", "dc_002"]
    assert result[0].timestamp == 2000


async def test_principle_with_only_unknown_source_ids_is_dropped() -> None:
    fake = FakeLLMClient(responses=[_principle_json(source_entry_ids=["dc_unknown"])])
    from everalgo.user_memory.principle import PrincipleExtractor

    result = await PrincipleExtractor(llm=fake).aextract(_cluster(), owner_id="u1")

    assert result == []
    assert fake.call_count == 1


async def test_empty_principles_array_is_success() -> None:
    fake = FakeLLMClient(responses=[ChatResponse(content='{"principles": []}', model="fake")])
    from everalgo.user_memory.principle import PrincipleExtractor

    result = await PrincipleExtractor(llm=fake).aextract(_cluster(), owner_id="u1")

    assert result == []
    assert fake.call_count == 1


async def test_aextract_raises_on_bad_json() -> None:
    fake = FakeLLMClient(responses=[ChatResponse(content="not json", model="fake")])
    from everalgo.user_memory.principle import PrincipleExtractor

    with pytest.raises(ValueError):
        await PrincipleExtractor(llm=fake).aextract(_cluster(), owner_id="u1")

    assert fake.call_count == 1


async def test_aextract_raises_when_principles_key_missing() -> None:
    fake = FakeLLMClient(responses=[ChatResponse(content='{"decisions": []}', model="fake")])
    from everalgo.user_memory.principle import PrincipleExtractor

    with pytest.raises(ValueError, match="principles key missing"):
        await PrincipleExtractor(llm=fake).aextract(_cluster(), owner_id="u1")


async def test_aextract_skips_items_missing_title_or_statement() -> None:
    payload = {
        "principles": [
            {
                "title": "kept",
                "statement": "A standing rule.",
                "source_entry_ids": ["dc_001"],
            },
            {"title": "", "statement": "x", "source_entry_ids": ["dc_001"]},
            {"title": "x", "statement": "", "source_entry_ids": ["dc_001"]},
        ]
    }
    fake = FakeLLMClient(responses=[ChatResponse(content=json.dumps(payload), model="fake")])
    from everalgo.user_memory.principle import PrincipleExtractor

    result = await PrincipleExtractor(llm=fake).aextract(_cluster(), owner_id="u1")
    assert len(result) == 1
    assert result[0].title == "kept"
    assert result[0].timestamp == 1000


async def test_aextract_truncates_when_more_than_10_principles() -> None:
    items = [{"title": f"p-{i}", "statement": "s", "source_entry_ids": ["dc_001"]} for i in range(12)]
    fake = FakeLLMClient(responses=[ChatResponse(content=json.dumps({"principles": items}), model="fake")])
    from everalgo.user_memory.principle import PrincipleExtractor

    result = await PrincipleExtractor(llm=fake).aextract(_cluster(), owner_id="u1")
    assert len(result) == 10


async def test_aextract_per_call_prompt_overrides_default() -> None:
    captured: dict[str, Any] = {}

    def handler(messages: list[LLMChatMessage], **kwargs: Any) -> ChatResponse:
        captured["content"] = messages[0].content
        return ChatResponse(content='{"principles": []}', model="fake")

    fake = FakeLLMClient(handler=handler)
    from everalgo.user_memory.principle import PrincipleExtractor

    custom = "CUSTOM PRINCIPLE cluster={DECISION_CLUSTER}"
    await PrincipleExtractor(llm=fake).aextract(_cluster(), owner_id="u1", prompt=custom)

    assert captured["content"].startswith("CUSTOM PRINCIPLE")
    assert "id=dc_001" in captured["content"]


def test_sync_extract_works() -> None:
    fake = FakeLLMClient(responses=[_principle_json()])
    from everalgo.user_memory.principle import PrincipleExtractor

    result = PrincipleExtractor(llm=fake).extract(_cluster(), owner_id="u1")
    assert result[0].statement.endswith("speed.")


# ==========================================================================
# Language
# ==========================================================================


class _PromptCapturedError(Exception):
    """Ends the call once the prompt has been captured."""


async def _render_principle_prompt(**kwargs: object) -> str:
    from everalgo.user_memory.principle import PrincipleExtractor

    captured: list[str] = []

    class Capture:
        async def chat(self, messages: list[LLMChatMessage], **_: object) -> ChatResponse:
            assert isinstance(messages[0].content, str)
            captured.append(messages[0].content)
            raise _PromptCapturedError

    with pytest.raises(_PromptCapturedError):
        await PrincipleExtractor(llm=Capture()).aextract(_cluster(), owner_id="u1", **kwargs)  # type: ignore[arg-type]
    return captured[0]


def test_prompt_carries_the_language_placeholder_at_both_ends() -> None:
    from everalgo.user_memory.prompts.en.principle import PRINCIPLE_GENERATION_PROMPT

    assert PRINCIPLE_GENERATION_PROMPT.count("{language_rule}") == 2


async def test_rendering_inherits_decision_language_when_none_is_named() -> None:
    rendered = await _render_principle_prompt()

    assert rendered.count("CRITICAL LANGUAGE RULE") == 2
    assert "as the decisions you are synthesising from" in rendered
    assert "{language_rule}" not in rendered


async def test_rendering_injects_the_named_language() -> None:
    rendered = await _render_principle_prompt(output_language=OutputLanguage.GERMAN)

    assert rendered.count("CRITICAL LANGUAGE RULE") == 2
    assert "Write ALL output fields in German." in rendered
    assert "as the decisions you are synthesising from" not in rendered


async def test_unsupported_language_is_rejected_before_the_llm_is_called() -> None:
    from everalgo.user_memory.principle import PrincipleExtractor

    class _NeverCalled:
        async def chat(self, **kwargs: Any) -> ChatResponse:
            raise AssertionError("LLM must not be called")

    with pytest.raises(ValueError, match="unsupported output_language"):
        await PrincipleExtractor(llm=_NeverCalled()).aextract(  # type: ignore[arg-type]
            _cluster(), owner_id="u1", output_language="Klingon"
        )


def test_principle_fallback_inherits_rather_than_judge() -> None:
    from everalgo.user_memory.prompts.en import _language as rules_mod

    rule = rules_mod.PRINCIPLES_FROM_DECISIONS_LANGUAGE_RULE
    assert "however much of the conversation it occupies" not in rule
    assert "do not translate" in rule
