"""Tests for AtomicFactExtractor.aextract_from_text, its sync bridge, and the atomic-fact reply contract.

The prompt's system / user split is pinned in ``test_atomic_fact_from_text_prompt_split.py``.
"""

from __future__ import annotations

import inspect
import json
import re

import pytest

from everalgo.llm import LLMOutputError, StrictOutputModel, to_openai_response_format
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.user_memory import OutputLanguage
from everalgo.user_memory._language import SOURCE_TEXT_LANGUAGE_RULE
from everalgo.user_memory._replies import _AtomicFactsBlock, _AtomicFactsReply
from everalgo.user_memory.atomic_fact import AtomicFactExtractor
from everalgo.user_memory.prompts.en.atomic_fact_from_text import (
    ATOMIC_FACT_FROM_TEXT_SYSTEM_PROMPT,
    ATOMIC_FACT_FROM_TEXT_USER_PROMPT,
)

# 2024-03-10 14:00:00 UTC, a Sunday.
_MARCH_10_2024_14H_UTC_MS: int = 1710079200000
_EXPECTED_TIME_STR: str = "2024-03-10 14:00 UTC (Sunday)"
_TIME_SHAPE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC \([A-Z][a-z]+\)$")


def _reply(*facts: str) -> str:
    return json.dumps({"atomic_facts": {"atomic_fact": list(facts)}})


class _ClientError(Exception):
    """A host-side client failure the operator must not wrap or retype."""


async def _sent(text: str = "text", **kwargs: object) -> tuple[str, str]:
    """The system and user message contents of one from-text extraction."""
    fake = FakeLLMClient(responses=[_reply()])
    await AtomicFactExtractor(llm=fake).aextract_from_text(text, timestamp=_MARCH_10_2024_14H_UTC_MS, **kwargs)  # type: ignore[arg-type]
    system, user = fake.calls[0].messages
    assert (system.role, user.role) == ("system", "user")
    return str(system.content), str(user.content)


# ---------------------------------------------------------------------------
# Happy-path and edge cases
# ---------------------------------------------------------------------------


async def test_returns_atomic_fact_list_on_happy_path() -> None:
    """FakeLLM returns two facts inside nested schema; result equals that list."""
    fake = FakeLLMClient(responses=[_reply("fact 1", "fact 2")])
    result = await AtomicFactExtractor(llm=fake).aextract_from_text("some text", timestamp=_MARCH_10_2024_14H_UTC_MS)
    assert [af.content for af in result] == ["fact 1", "fact 2"]
    assert all(af.owner_id is None and af.timestamp == _MARCH_10_2024_14H_UTC_MS for af in result)


async def test_empty_atomic_fact_list_returns_empty() -> None:
    """FakeLLM returns empty nested list; result is empty list."""
    fake = FakeLLMClient(responses=[_reply()])
    result = await AtomicFactExtractor(llm=fake).aextract_from_text(
        "greeting only", timestamp=_MARCH_10_2024_14H_UTC_MS
    )
    assert result == []


async def test_blank_fact_sentences_are_dropped_and_the_rest_stripped() -> None:
    fake = FakeLLMClient(responses=[_reply(" fact 1 ", "", "  ")])
    result = await AtomicFactExtractor(llm=fake).aextract_from_text("text", timestamp=_MARCH_10_2024_14H_UTC_MS)
    assert [af.content for af in result] == ["fact 1"]


async def test_one_call_requests_the_reply_model() -> None:
    fake = FakeLLMClient(responses=[_reply()])
    await AtomicFactExtractor(llm=fake).aextract_from_text("text", timestamp=_MARCH_10_2024_14H_UTC_MS)
    assert fake.call_count == 1
    assert fake.calls[0].response_format is _AtomicFactsReply


# ---------------------------------------------------------------------------
# Error cases — every invalid reply is an LLMOutputError on the one call
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reply", "location"),
    [
        pytest.param('{"event_log": {"atomic_fact": ["fact A"]}}', "atomic_facts", id="event_log-alias"),
        pytest.param('{"foo": "bar"}', "atomic_facts", id="missing-top-level-key"),
        pytest.param('{"atomic_facts": {}}', "atomic_facts.atomic_fact", id="missing-inner-list"),
        pytest.param('{"atomic_facts": "not a dict"}', "atomic_facts", id="block-not-an-object"),
        pytest.param('{"atomic_facts": {"atomic_fact": "x"}}', "atomic_facts.atomic_fact", id="not-a-list"),
        pytest.param('{"atomic_facts": {"atomic_fact": [1]}}', "atomic_facts.atomic_fact.0", id="non-str"),
        pytest.param('```json\n{"atomic_facts": {"atomic_fact": []}}\n```', "<root>", id="fenced"),
        pytest.param("not json at all", "<root>", id="not-json"),
    ],
)
async def test_an_invalid_reply_raises_llm_output_error_naming_the_location(reply: str, location: str) -> None:
    fake = FakeLLMClient(responses=[reply])

    with pytest.raises(LLMOutputError, match="atomic_fact reply is not a valid _AtomicFactsReply") as excinfo:
        await AtomicFactExtractor(llm=fake).aextract_from_text("text", timestamp=_MARCH_10_2024_14H_UTC_MS)

    assert f"{location}:" in str(excinfo.value)
    assert fake.call_count == 1


async def test_a_time_echo_is_an_extra_key() -> None:
    """The reply no longer echoes ``<time>``; a model that still writes ``time`` breaks the strict skeleton."""
    fake = FakeLLMClient(responses=['{"atomic_facts": {"time": "T", "atomic_fact": ["fact"]}}'])

    with pytest.raises(LLMOutputError, match=r"atomic_facts\.time: Extra inputs are not permitted"):
        await AtomicFactExtractor(llm=fake).aextract_from_text("text", timestamp=_MARCH_10_2024_14H_UTC_MS)


async def test_the_error_message_never_quotes_the_reply() -> None:
    fake = FakeLLMClient(responses=['{"atomic_facts": {"atomic_fact": ["SECRET-FACT", 7]}}'])

    with pytest.raises(LLMOutputError) as excinfo:
        await AtomicFactExtractor(llm=fake).aextract_from_text("text", timestamp=_MARCH_10_2024_14H_UTC_MS)

    assert "SECRET-FACT" not in str(excinfo.value)


async def test_client_exception_propagates_unchanged() -> None:
    """The host recognises call failures by its own types, so the operator neither wraps nor retypes them."""
    error = _ClientError("upstream down")

    with pytest.raises(_ClientError) as excinfo:
        await AtomicFactExtractor(llm=FakeLLMClient(responses=[error])).aextract_from_text(
            "text", timestamp=_MARCH_10_2024_14H_UTC_MS
        )

    assert excinfo.value is error


# ---------------------------------------------------------------------------
# What the model receives
# ---------------------------------------------------------------------------


async def test_timestamp_and_text_fill_the_user_message() -> None:
    _, user = await _sent("UNIQUE_TEXT_MARKER_42")

    assert user == f"<time>\n{_EXPECTED_TIME_STR}\n</time>\n\n<episode_text>\nUNIQUE_TEXT_MARKER_42\n</episode_text>"


@pytest.mark.parametrize("timestamp", [_MARCH_10_2024_14H_UTC_MS, 1_700_000_000_000, 0], ids=["14h", "22h", "epoch"])
async def test_time_is_the_utc_minute_with_its_weekday(timestamp: int) -> None:
    """``<time>`` has the one shape every instant in a prompt has, 24-hour and zone-labelled, plus the weekday."""
    fake = FakeLLMClient(responses=[_reply()])
    await AtomicFactExtractor(llm=fake).aextract_from_text("text", timestamp=timestamp)
    user = str(fake.calls[0].messages[1].content)

    assert _TIME_SHAPE.match(user.split("<time>\n", 1)[1].split("\n</time>", 1)[0])


async def test_default_system_message_is_the_static_rules_then_the_language_rule() -> None:
    system, user = await _sent("MARKER_DEFAULT_PROMPT_CHECK")

    assert system == (
        f"{ATOMIC_FACT_FROM_TEXT_SYSTEM_PROMPT}\n\n<output_language>\n{SOURCE_TEXT_LANGUAGE_RULE}\n</output_language>"
    )
    assert user == ATOMIC_FACT_FROM_TEXT_USER_PROMPT.replace("{{time}}", _EXPECTED_TIME_STR).replace(
        "{{episode_text}}", "MARKER_DEFAULT_PROMPT_CHECK"
    )


async def test_system_message_is_the_same_for_every_text() -> None:
    """With the host's arguments fixed, the system message is a constant a prompt cache can share whole."""
    first, _ = await _sent("hello")
    second, _ = await _sent("something else entirely")

    assert first == second


async def test_text_is_never_rendered_as_a_template() -> None:
    """A slot name inside the caller's text stays literal and never pulls another field in."""
    _, user = await _sent("Rules {{episode_text}} and {{time}} stay literal.")

    assert "<episode_text>\nRules {{episode_text}} and {{time}} stay literal.\n</episode_text>" in user


# ---------------------------------------------------------------------------
# custom_instructions — appended to the rules, never a template, never an override
# ---------------------------------------------------------------------------


async def test_default_system_message_has_no_custom_instructions_block() -> None:
    system, _ = await _sent("hello")

    assert "custom_instructions" not in system


async def test_custom_instructions_land_verbatim_after_the_language_rule_closing_the_message() -> None:
    """The static rules stay a shared prefix; the caller's block follows ``<output_language>`` and ends the message."""
    text = "Keep product names in English.\nSkip greetings."
    system, user = await _sent("hello", custom_instructions=text)

    assert system == (
        f"{ATOMIC_FACT_FROM_TEXT_SYSTEM_PROMPT}\n\n<output_language>\n{SOURCE_TEXT_LANGUAGE_RULE}\n</output_language>"
        f"\n\n<custom_instructions>\n{text}\n</custom_instructions>"
    )
    assert system.endswith("\n</custom_instructions>")
    assert text not in user


@pytest.mark.parametrize("blank", ["", " ", "\n\t  \n"])
async def test_blank_custom_instructions_are_the_same_as_none(blank: str) -> None:
    default, _ = await _sent("hello")
    system, _ = await _sent("hello", custom_instructions=blank)

    assert system == default


async def test_custom_instructions_are_never_rendered_as_a_template() -> None:
    """Slot names, data slots and the retired single-brace shape all stay literal instead of filling or raising."""
    text = "Keep {{LANGUAGE_RULE}}, {{time}}, {{episode_text}} and {TEXT} literal."
    system, _ = await _sent("hello", custom_instructions=text)

    assert f"<custom_instructions>\n{text}\n</custom_instructions>" in system
    assert system.count(SOURCE_TEXT_LANGUAGE_RULE) == 1


def test_aextract_from_text_takes_no_prompt_override() -> None:
    """The bundled rules are the only rules; a host adds to them through ``custom_instructions``."""
    assert list(inspect.signature(AtomicFactExtractor.aextract_from_text).parameters) == [
        "self",
        "text",
        "timestamp",
        "custom_instructions",
        "output_language",
    ]


# ---------------------------------------------------------------------------
# Sync bridge
# ---------------------------------------------------------------------------


def test_sync_bridge_extract_from_text_works() -> None:
    """extract_from_text (sync bridge) called outside event loop returns valid list."""
    fake = FakeLLMClient(responses=[_reply("sync fact")])
    result = AtomicFactExtractor(llm=fake).extract_from_text("some text", timestamp=_MARCH_10_2024_14H_UTC_MS)
    assert [af.content for af in result] == ["sync fact"]


# ==========================================================================
# Language rule — the from-text prompt inherits the <episode_text> language
# ==========================================================================


async def test_rendering_inherits_the_source_text_language_when_none_is_named() -> None:
    """``text`` is normally an already-extracted narrative, so the fallback inherits instead of judging.

    The conversation rule's paragraphs — pasted material, a second speaker, a contradicting identifier —
    have nothing to adjudicate here, and asking for that judgement again could only disagree with the
    extraction that already made it.
    """
    system, user = await _sent("li moved to Hangzhou last month.")

    assert system.count("CRITICAL LANGUAGE RULE") == 1
    assert "SAME language <episode_text> itself is written in" in system
    assert "dominate" not in system
    assert "{{LANGUAGE_RULE}}" not in system
    assert "CRITICAL LANGUAGE RULE" not in user


async def test_rendering_injects_the_named_language() -> None:
    system, _ = await _sent("li moved to Hangzhou last month.", output_language=OutputLanguage.GERMAN)

    assert (
        system.count(
            "Write every output field in German, whatever language the input is in, including anything already stored."
        )
        == 1
    )
    assert "German, whatever language the input is in." not in system
    assert "CRITICAL LANGUAGE RULE" not in system
    assert "SAME language <episode_text> itself is written in" not in system


async def test_an_unsupported_language_is_rejected_before_the_llm_is_called() -> None:
    fake = FakeLLMClient(responses=[_reply()])

    with pytest.raises(ValueError, match="names no supported language"):
        await AtomicFactExtractor(llm=fake).aextract_from_text(
            "text", timestamp=_MARCH_10_2024_14H_UTC_MS, output_language="Klingon"
        )

    assert fake.call_count == 0


# ===========================================================================
# LLM contract — the operator-side obligations of docs/concepts/llm-contract.md §4.3
# ===========================================================================


_REPLY_MODELS: tuple[type[StrictOutputModel], ...] = (_AtomicFactsReply, _AtomicFactsBlock)


@pytest.mark.parametrize("model", _REPLY_MODELS, ids=lambda model: model.__name__)
def test_reply_models_are_strict_one_level_and_default_free(model: type[StrictOutputModel]) -> None:
    """The reply model and its nested block inherit StrictOutputModel directly, require every field, no validators."""
    assert model.__bases__ == (StrictOutputModel,)
    assert model.model_config.get("strict") is True
    assert model.model_config.get("extra") == "forbid"
    for name, field in model.model_fields.items():
        assert field.is_required(), f"{model.__name__}.{name} has a default"
    assert not model.__pydantic_decorators__.field_validators
    assert not model.__pydantic_decorators__.model_validators


def test_reply_model_fits_the_strict_subset() -> None:
    """The request builders raise on anything outside the strict subset or a non-object top level (§8.3)."""
    response_format = to_openai_response_format(_AtomicFactsReply)

    assert response_format is not None
    assert response_format["json_schema"]["schema"]["type"] == "object"


def test_reply_fields_keep_the_skeleton_order() -> None:
    assert list(_AtomicFactsReply.model_fields) == ["atomic_facts"]
    assert list(_AtomicFactsBlock.model_fields) == ["atomic_fact"]
