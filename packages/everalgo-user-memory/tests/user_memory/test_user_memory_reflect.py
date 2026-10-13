"""Tests for everalgo.user_memory.reflect — EpisodeReflector — and the reflect reply contract.

The prompts' system / user split is pinned in ``test_reflect_prompt_split.py``.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from everalgo.llm import LLMMessage, LLMOutputError, StrictOutputModel, to_openai_response_format
from everalgo.testing import FakeLLMClient
from everalgo.types import Episode
from everalgo.user_memory._replies import _ReflectOutput


class _ClientError(Exception):
    """A host-side client failure the operator must not wrap or retype."""


def _ep(ts: int, text: str = "some episode text") -> Episode:
    """Minimal Episode fixture."""
    return Episode(owner_id=None, episode=text, subject="topic", summary="a preview", timestamp=ts)


class TestValidateInputs:
    def test_passes_with_two_sorted_episodes(self) -> None:
        from everalgo.user_memory.reflect import _validate_inputs

        _validate_inputs([_ep(1000), _ep(2000)], min_count=2)

    def test_raises_when_fewer_than_min_count(self) -> None:
        from everalgo.user_memory.reflect import _validate_inputs

        with pytest.raises(ValueError, match="at least 2"):
            _validate_inputs([_ep(1000)], min_count=2)

    def test_raises_on_empty_list(self) -> None:
        from everalgo.user_memory.reflect import _validate_inputs

        with pytest.raises(ValueError, match="at least 1"):
            _validate_inputs([], min_count=1)

    def test_raises_when_timestamps_not_ascending(self) -> None:
        from everalgo.user_memory.reflect import _validate_inputs

        with pytest.raises(ValueError, match=r"sorted.*ascending"):
            _validate_inputs([_ep(3000), _ep(1000)], min_count=2)

    def test_allows_equal_timestamps(self) -> None:
        from everalgo.user_memory.reflect import _validate_inputs

        _validate_inputs([_ep(1000), _ep(1000)], min_count=2)


class TestRenderTimeline:
    def test_numbered_format_without_bracketed_timestamp(self) -> None:
        """No bracketed timestamp: ``ep.episode`` already carries the times its own prompt pinned."""
        from everalgo.user_memory.reflect import _render_timeline

        result = _render_timeline([_ep(1700000000000, "First event"), _ep(1700100000000, "Second event")])
        lines = result.strip().split("\n")
        assert len(lines) == 2
        assert lines[0] == "1. First event"
        assert lines[1] == "2. Second event"

    def test_single_episode(self) -> None:
        from everalgo.user_memory.reflect import _render_timeline

        result = _render_timeline([_ep(1700000000000, "Only event")])
        assert result == "1. Only event"

    def test_no_am_pm_or_bracket(self) -> None:
        """Regression guard: the removed bracket used to carry a 12-hour AM/PM format."""
        from everalgo.user_memory.reflect import _render_timeline

        result = _render_timeline([_ep(1700000000000, "First event"), _ep(1700100000000, "Second event")])
        assert "AM" not in result
        assert "PM" not in result
        assert "[" not in result


class TestReflectInit:
    async def test_init_returns_merged_episode(self) -> None:
        fake = FakeLLMClient(
            responses=['{"content": "Merged narrative.", "summary": "Preview of Merged narrative.", "title": "Pets"}']
        )
        from everalgo.user_memory.reflect import EpisodeReflector

        reflector = EpisodeReflector(llm=fake)
        episodes = [_ep(1000, "Got a dog"), _ep(2000, "Got a cat")]
        result = await reflector.areflect(episodes)

        assert result.episode == "Merged narrative."
        assert result.subject == "Pets"
        assert result.timestamp == 2000
        assert result.owner_id is None

    async def test_init_sends_timeline_in_prompt(self) -> None:
        fake = FakeLLMClient(responses=['{"content": "merged", "summary": "Preview of merged", "title": "t"}'])
        from everalgo.user_memory.reflect import EpisodeReflector

        await EpisodeReflector(llm=fake).areflect([_ep(1000, "Event A"), _ep(2000, "Event B")])

        assert fake.call_count == 1
        prompt_text = fake.calls[0].messages[1].content
        assert "Event A" in prompt_text
        assert "Event B" in prompt_text
        assert "1. Event A" in prompt_text

    @pytest.mark.parametrize("old_episode", [None, _ep(500)])
    async def test_both_modes_request_the_reply_model(self, old_episode: Episode | None) -> None:
        fake = FakeLLMClient(responses=['{"content": "merged", "summary": "Preview of merged", "title": "t"}'])
        from everalgo.user_memory.reflect import EpisodeReflector

        await EpisodeReflector(llm=fake).areflect([_ep(1000), _ep(2000)], old_episode=old_episode)

        assert fake.call_count == 1
        assert fake.calls[0].response_format is _ReflectOutput

    async def test_the_merged_episode_is_deserialised_from_the_reply_content(self) -> None:
        """The reply body is the only source: nothing else carries the merged narrative back."""
        fake = FakeLLMClient(responses=['{"content": "merged body", "summary": "a preview", "title": "T"}'])
        from everalgo.user_memory.reflect import EpisodeReflector

        result = await EpisodeReflector(llm=fake).areflect([_ep(1000), _ep(2000)])

        assert (result.episode, result.summary, result.subject) == ("merged body", "a preview", "T")

    @pytest.mark.parametrize(
        ("reply", "location"),
        [
            pytest.param("not json at all", "<root>", id="not-json"),
            pytest.param('{"content": "c", "summary": "s"}', "title", id="missing-title"),
            pytest.param('{"content": "c", "summary": "s", "title": "t", "extra": 1}', "extra", id="extra-key"),
            pytest.param('{"content": 1, "summary": "s", "title": "t"}', "content", id="content-not-a-string"),
        ],
    )
    async def test_a_reply_that_does_not_match_the_schema_is_rejected(self, reply: str, location: str) -> None:
        """Every invalid reply is an LLMOutputError naming the location; a missing title no longer defaults."""
        fake = FakeLLMClient(responses=[reply])
        from everalgo.user_memory.reflect import EpisodeReflector

        with pytest.raises(LLMOutputError, match="reflect reply is not a valid _ReflectOutput") as excinfo:
            await EpisodeReflector(llm=fake).areflect([_ep(1000), _ep(2000)])

        assert f"{location}:" in str(excinfo.value)
        assert fake.call_count == 1

    async def test_the_error_message_never_quotes_the_reply(self) -> None:
        fake = FakeLLMClient(responses=['{"content": "SECRET-NARRATIVE", "summary": 7, "title": "t"}'])
        from everalgo.user_memory.reflect import EpisodeReflector

        with pytest.raises(LLMOutputError) as excinfo:
            await EpisodeReflector(llm=fake).areflect([_ep(1000), _ep(2000)])

        assert "SECRET-NARRATIVE" not in str(excinfo.value)

    async def test_init_fewer_than_2_raises(self) -> None:
        fake = FakeLLMClient(responses=['{"content": "x", "summary": "Preview of x", "title": "t"}'])
        from everalgo.user_memory.reflect import EpisodeReflector

        with pytest.raises(ValueError, match="at least 2"):
            await EpisodeReflector(llm=fake).areflect([_ep(1000)])

    async def test_init_unsorted_raises(self) -> None:
        fake = FakeLLMClient(responses=['{"content": "x", "summary": "Preview of x", "title": "t"}'])
        from everalgo.user_memory.reflect import EpisodeReflector

        with pytest.raises(ValueError, match=r"sorted.*ascending"):
            await EpisodeReflector(llm=fake).areflect([_ep(3000), _ep(1000)])


class TestReflectUpdate:
    async def test_update_returns_merged_episode(self) -> None:
        fake = FakeLLMClient(
            responses=[
                '{"content": "Updated narrative.", "summary": "Preview of Updated narrative.", "title": "Pets v2"}'
            ]
        )
        from everalgo.user_memory.reflect import EpisodeReflector

        old = _ep(1000, "Had a dog named Toby.")
        new_episodes = [_ep(2000, "Got a cat named Whiskers")]
        result = await EpisodeReflector(llm=fake).areflect(
            new_episodes,
            old_episode=old,
        )

        assert result.episode == "Updated narrative."
        assert result.subject == "Pets v2"
        assert result.timestamp == 2000
        assert result.owner_id is None

    async def test_update_sends_old_episode_in_prompt(self) -> None:
        fake = FakeLLMClient(responses=['{"content": "merged", "summary": "Preview of merged", "title": "t"}'])
        from everalgo.user_memory.reflect import EpisodeReflector

        old = _ep(1000, "Original narrative about pets.")
        await EpisodeReflector(llm=fake).areflect([_ep(2000, "New info")], old_episode=old)

        prompt_text = fake.calls[0].messages[1].content
        assert "Original narrative about pets." in prompt_text
        assert "New info" in prompt_text

    async def test_update_single_episode_sufficient(self) -> None:
        fake = FakeLLMClient(responses=['{"content": "ok", "summary": "Preview of ok", "title": "t"}'])
        from everalgo.user_memory.reflect import EpisodeReflector

        result = await EpisodeReflector(llm=fake).areflect([_ep(2000)], old_episode=_ep(1000))
        assert result.episode.endswith("ok")

    async def test_update_empty_raises(self) -> None:
        fake = FakeLLMClient(responses=['{"content": "x", "summary": "Preview of x", "title": "t"}'])
        from everalgo.user_memory.reflect import EpisodeReflector

        with pytest.raises(ValueError, match="at least 1"):
            await EpisodeReflector(llm=fake).areflect([], old_episode=_ep(1000))


# 2026-05-29 12:25:10 UTC (Friday) .. +70 minutes — mirrors test_user_memory_episode.py's
# _memcell_spanning_70_minutes, so span start and span end are visibly different instants rather
# than two epoch-adjacent millisecond values.
_TS_SPAN_START = 1780057510000
_TS_SPAN_END = _TS_SPAN_START + 70 * 60 * 1000


class TestReflectMergedTimestamp:
    """The merged episode's ``timestamp`` is the span end, matching ``_build_episode``'s field choice."""

    async def test_init_merge_timestamp_is_span_end_not_span_start(self) -> None:
        fake = FakeLLMClient(responses=['{"content": "Merged.", "summary": "Preview of Merged.", "title": "t"}'])
        from everalgo.user_memory.reflect import EpisodeReflector

        episodes = [_ep(_TS_SPAN_START, "first"), _ep(_TS_SPAN_END, "last")]
        result = await EpisodeReflector(llm=fake).areflect(episodes)

        assert result.episode == "Merged."
        assert result.timestamp == _TS_SPAN_END

    async def test_update_merge_timestamp_is_new_episodes_span_end(self) -> None:
        fake = FakeLLMClient(responses=['{"content": "Updated.", "summary": "Preview of Updated.", "title": "t"}'])
        from everalgo.user_memory.reflect import EpisodeReflector

        old = _ep(_TS_SPAN_START - 1, "older narrative")
        new_episodes = [_ep(_TS_SPAN_START, "first new"), _ep(_TS_SPAN_END, "last new")]
        result = await EpisodeReflector(llm=fake).areflect(new_episodes, old_episode=old)

        assert result.episode == "Updated."
        assert result.timestamp == _TS_SPAN_END


class TestReflectErrors:
    @pytest.mark.parametrize("old_episode", [None, _ep(500)])
    async def test_client_exception_propagates_unchanged(self, old_episode: Episode | None) -> None:
        """The host recognises call failures by its own types, so the operator neither wraps nor retypes them."""
        error = _ClientError("upstream down")
        fake = FakeLLMClient(responses=[error])
        from everalgo.user_memory.reflect import EpisodeReflector

        with pytest.raises(_ClientError) as excinfo:
            await EpisodeReflector(llm=fake).areflect([_ep(1000), _ep(2000)], old_episode=old_episode)

        assert excinfo.value is error
        assert fake.call_count == 1


class TestReflectMisc:
    def test_sync_reflect_works(self) -> None:
        fake = FakeLLMClient(responses=['{"content": "synced", "summary": "Preview of synced", "title": "t"}'])
        from everalgo.user_memory.reflect import EpisodeReflector

        result = EpisodeReflector(llm=fake).reflect([_ep(1000), _ep(2000)])
        assert result.episode.endswith("synced")

    def test_areflect_takes_no_prompt_override(self) -> None:
        """The bundled rules are the only rules; a host adds to them through ``custom_instructions``."""
        from everalgo.user_memory.reflect import EpisodeReflector

        assert list(inspect.signature(EpisodeReflector.areflect).parameters) == [
            "self",
            "episodes",
            "old_episode",
            "custom_instructions",
            "output_language",
        ]


# ==========================================================================
# Language rule — reflect inherits the merged episodes' language
# ==========================================================================


# ==========================================================================
# summary — merged episodes are displayed like extracted ones, so they carry a preview too
# ==========================================================================


async def test_init_carries_the_models_summary_onto_the_merged_episode() -> None:
    fake = FakeLLMClient(responses=['{"content": "Merged.", "summary": "A merged preview.", "title": "t"}'])
    from everalgo.user_memory.reflect import EpisodeReflector

    result = await EpisodeReflector(llm=fake).areflect([_ep(1000), _ep(2000)])

    assert result.summary == "A merged preview."


async def test_update_carries_the_models_summary_onto_the_merged_episode() -> None:
    fake = FakeLLMClient(responses=['{"content": "Updated.", "summary": "An updated preview.", "title": "t"}'])
    from everalgo.user_memory.reflect import EpisodeReflector

    result = await EpisodeReflector(llm=fake).areflect([_ep(2000)], old_episode=_ep(1000))

    assert result.summary == "An updated preview."


def test_summary_is_declared_after_content_in_the_structured_output_schema() -> None:
    """Structured Output generates fields in schema order, so a preview declared first previews nothing."""
    order = list(_ReflectOutput.model_fields)
    assert order.index("summary") > order.index("content")


def test_reflect_output_stays_importable_from_the_reflect_module() -> None:
    """Hosts import it from ``everalgo.user_memory.reflect`` to check their own prompts against its fields."""
    from everalgo.user_memory import reflect

    assert reflect._ReflectOutput is _ReflectOutput
    assert list(_ReflectOutput.model_fields) == ["content", "summary", "title"]


@pytest.mark.parametrize("name", ["REFLECT_EPISODE_SYSTEM_PROMPT", "REFLECT_EPISODE_UPDATE_SYSTEM_PROMPT"])
def test_both_reflect_prompts_bound_the_summary(name: str) -> None:
    """Same three rigid clauses as the extractor's spec, worded for a merged narrative."""
    import everalgo.user_memory.prompts.en.reflect as en_mod

    prompt = getattr(en_mod, name)
    assert "Under 50 words" in prompt
    assert "Introduce no fact the narrative does not already carry" in prompt
    assert "do not refer to the record itself" in prompt


_SYSTEM_PROMPTS = ("REFLECT_EPISODE_SYSTEM_PROMPT", "REFLECT_EPISODE_UPDATE_SYSTEM_PROMPT")


async def _render_reflect_prompt(
    *,
    old_episode: Episode | None = None,
    output_language: str | None = None,
) -> str:
    """Return the system message ``areflect`` actually sends, so the language rule is checked after splicing."""
    from everalgo.user_memory.reflect import EpisodeReflector

    captured: list[LLMMessage] = []

    def handler(messages: list[LLMMessage], **_: Any) -> str:
        captured.extend(messages)
        return '{"content": "c", "summary": "s", "title": "t"}'

    await EpisodeReflector(llm=FakeLLMClient(handler=handler)).areflect(
        [_ep(1000), _ep(2000)], old_episode=old_episode, output_language=output_language
    )
    system, user = captured
    assert "LANGUAGE RULE" not in str(user.content)
    return str(system.content)


@pytest.mark.parametrize("old_episode", [None, _ep(500, "existing narrative")])
async def test_the_language_rule_closes_the_system_message_once(old_episode: Episode | None) -> None:
    """With no caller instructions the rule sits in ``<output_language>``, the last block, once."""
    rendered = await _render_reflect_prompt(old_episode=old_episode)

    assert rendered.count("<output_language>") == 1
    assert rendered.endswith("\n</output_language>")
    assert rendered.split("<output_language>\n", 1)[1].count("CRITICAL LANGUAGE RULE") == 1


async def test_init_inherits_the_merged_episodes_language_when_none_is_named() -> None:
    rendered = await _render_reflect_prompt()

    assert rendered.count("CRITICAL LANGUAGE RULE") == 1
    assert "as the episodes you are merging" in rendered
    assert "{{language_rule}}" not in rendered


async def test_update_inherits_the_existing_narratives_language_when_none_is_named() -> None:
    rendered = await _render_reflect_prompt(old_episode=_ep(500, "existing narrative"))

    assert rendered.count("CRITICAL LANGUAGE RULE") == 1
    assert "as the existing narrative you are updating" in rendered
    assert "{{language_rule}}" not in rendered


@pytest.mark.parametrize("old_episode", [None, _ep(500, "existing narrative")])
async def test_both_modes_honour_a_named_language(old_episode: Episode | None) -> None:
    rendered = await _render_reflect_prompt(old_episode=old_episode, output_language="German")

    assert (
        rendered.count(
            "Write every output field in German, whatever language the input is in, including anything already stored."
        )
        == 1
    )
    assert "German, whatever language the input is in." not in rendered
    assert "CRITICAL LANGUAGE RULE" not in rendered
    assert "as the episodes you are merging" not in rendered
    assert "as the existing narrative you are updating" not in rendered


@pytest.mark.parametrize("old_episode", [None, _ep(500, "existing narrative")])
async def test_an_unsupported_language_is_rejected_before_the_llm_is_called(old_episode: Episode | None) -> None:
    """The value is interpolated into prompt text, so it must never reach the model unvalidated."""
    from everalgo.user_memory.reflect import EpisodeReflector

    fake = FakeLLMClient(responses=['{"content": "c", "summary": "s", "title": "t"}'])

    with pytest.raises(ValueError, match="names no supported language"):
        await EpisodeReflector(llm=fake).areflect(
            [_ep(1000), _ep(2000)], old_episode=old_episode, output_language="Klingon"
        )

    assert fake.call_count == 0


@pytest.mark.parametrize(
    "rule_name",
    ["MERGED_EPISODES_LANGUAGE_RULE", "EXISTING_NARRATIVE_LANGUAGE_RULE"],
)
def test_reflect_fallbacks_inherit_rather_than_judge(rule_name: str) -> None:
    """Reflect merges already-extracted episodes, so it inherits rather than judges the language.

    Judgement belongs to the extractor that reads the raw conversation. Re-judging here would be inert on
    single-language narratives, and a disagreeing judgement would translate a merged episode out of the
    language its sources were written in. Asserted against the rule texts rather than the prompt constants:
    the rules are spliced in at call time, so a prompt constant cannot show which one was chosen.
    """
    from everalgo.user_memory.prompts.en import _language as rules_mod

    rule = getattr(rules_mod, rule_name)
    assert "however much of the conversation it occupies" not in rule
    assert "read only the message contents" not in rule
    assert "do not translate" in rule


def test_reflect_prompts_keep_times_as_written_without_touching_zone_labels() -> None:
    """Reflection must carry every time over as the episodes wrote it, zone label included or not.

    It rewrites the narrative wholesale, and ``Episode.timestamp`` never reaches answer context, so a
    time that reflection drops or reformats is not recoverable from anywhere else. Episode extraction
    (v2.2) labels only the clock times it derives itself and leaves speaker-quoted ones unlabelled,
    because their timezone is unknown; a reflect rule demanding a UTC label on every clock time would
    invent one for exactly those times.
    """
    import everalgo.user_memory.prompts.en.reflect as en_mod

    for name in _SYSTEM_PROMPTS:
        prompt = getattr(en_mod, name)
        assert "do not add or drop a zone label" in prompt, name
        assert "MUST carry the UTC zone label" not in prompt, name


def test_reflect_prompts_ask_for_a_bare_json_object() -> None:
    """The schema constrains the reply upstream, but only when the server honours it.

    OpenRouter treats ``response_format`` as a routing preference, so the prompt has to carry the same
    instruction the other operators' prompts carry rather than relying on the schema alone.
    """
    import everalgo.user_memory.prompts.en.reflect as en_mod

    for name in _SYSTEM_PROMPTS:
        assert "Reply with one JSON object, nothing else." in getattr(en_mod, name), name


# ===========================================================================
# LLM contract — the operator-side obligations of docs/concepts/llm-contract.md §4.3
# ===========================================================================


def test_reply_model_is_strict_one_level_and_default_free() -> None:
    """``_ReflectOutput`` inherits StrictOutputModel directly, requires every field and carries no validators."""
    assert _ReflectOutput.__bases__ == (StrictOutputModel,)
    assert _ReflectOutput.model_config.get("strict") is True
    assert _ReflectOutput.model_config.get("extra") == "forbid"
    for name, field in _ReflectOutput.model_fields.items():
        assert field.is_required(), f"_ReflectOutput.{name} has a default"
    assert not _ReflectOutput.__pydantic_decorators__.field_validators
    assert not _ReflectOutput.__pydantic_decorators__.model_validators


def test_reply_model_fits_the_strict_subset() -> None:
    """The request builders raise on anything outside the strict subset or a non-object top level (§8.3)."""
    response_format = to_openai_response_format(_ReflectOutput)

    assert response_format is not None
    assert response_format["json_schema"]["schema"]["type"] == "object"
