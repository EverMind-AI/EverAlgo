"""Tests for everalgo.user_memory.reflect — EpisodeReflector."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from everalgo.llm.errors import LLMError
from everalgo.testing import FakeLLMClient
from everalgo.types import Episode

if TYPE_CHECKING:
    from everalgo.llm.types import ChatMessage as LLMChatMessage
    from everalgo.llm.types import ChatResponse


def _ep(ts: int, text: str = "some episode text") -> Episode:
    """Minimal Episode fixture."""
    return Episode(owner_id=None, episode=text, subject="topic", timestamp=ts)


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
        fake = FakeLLMClient(responses=['{"content": "Merged narrative.", "title": "Pets"}'])
        from everalgo.user_memory.reflect import EpisodeReflector

        reflector = EpisodeReflector(llm=fake)
        episodes = [_ep(1000, "Got a dog"), _ep(2000, "Got a cat")]
        result = await reflector.areflect(episodes)

        assert result.episode == "Merged narrative."
        assert result.subject == "Pets"
        assert result.timestamp == 2000
        assert result.owner_id is None

    async def test_init_sends_timeline_in_prompt(self) -> None:
        fake = FakeLLMClient(responses=['{"content": "merged", "title": "t"}'])
        from everalgo.user_memory.reflect import EpisodeReflector

        await EpisodeReflector(llm=fake).areflect([_ep(1000, "Event A"), _ep(2000, "Event B")])

        assert fake.call_count == 1
        prompt_text = fake.calls[0].messages[0].content
        assert "Event A" in prompt_text
        assert "Event B" in prompt_text
        assert "1. Event A" in prompt_text

    async def test_init_uses_structured_output(self) -> None:
        fake = FakeLLMClient(responses=['{"content": "merged", "title": "t"}'])
        from everalgo.user_memory.reflect import EpisodeReflector

        await EpisodeReflector(llm=fake).areflect([_ep(1000), _ep(2000)])

        assert fake.calls[0].response_format is not None

    async def test_init_fewer_than_2_raises(self) -> None:
        fake = FakeLLMClient(responses=['{"content": "x", "title": "t"}'])
        from everalgo.user_memory.reflect import EpisodeReflector

        with pytest.raises(ValueError, match="at least 2"):
            await EpisodeReflector(llm=fake).areflect([_ep(1000)])

    async def test_init_unsorted_raises(self) -> None:
        fake = FakeLLMClient(responses=['{"content": "x", "title": "t"}'])
        from everalgo.user_memory.reflect import EpisodeReflector

        with pytest.raises(ValueError, match=r"sorted.*ascending"):
            await EpisodeReflector(llm=fake).areflect([_ep(3000), _ep(1000)])


class TestReflectUpdate:
    async def test_update_returns_merged_episode(self) -> None:
        fake = FakeLLMClient(responses=['{"content": "Updated narrative.", "title": "Pets v2"}'])
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
        fake = FakeLLMClient(responses=['{"content": "merged", "title": "t"}'])
        from everalgo.user_memory.reflect import EpisodeReflector

        old = _ep(1000, "Original narrative about pets.")
        await EpisodeReflector(llm=fake).areflect([_ep(2000, "New info")], old_episode=old)

        prompt_text = fake.calls[0].messages[0].content
        assert "Original narrative about pets." in prompt_text
        assert "New info" in prompt_text

    async def test_update_single_episode_sufficient(self) -> None:
        fake = FakeLLMClient(responses=['{"content": "ok", "title": "t"}'])
        from everalgo.user_memory.reflect import EpisodeReflector

        result = await EpisodeReflector(llm=fake).areflect([_ep(2000)], old_episode=_ep(1000))
        assert result.episode.endswith("ok")

    async def test_update_empty_raises(self) -> None:
        fake = FakeLLMClient(responses=['{"content": "x", "title": "t"}'])
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
        fake = FakeLLMClient(responses=['{"content": "Merged.", "title": "t"}'])
        from everalgo.user_memory.reflect import EpisodeReflector

        episodes = [_ep(_TS_SPAN_START, "first"), _ep(_TS_SPAN_END, "last")]
        result = await EpisodeReflector(llm=fake).areflect(episodes)

        assert result.episode == "Merged."
        assert result.timestamp == _TS_SPAN_END

    async def test_update_merge_timestamp_is_new_episodes_span_end(self) -> None:
        fake = FakeLLMClient(responses=['{"content": "Updated.", "title": "t"}'])
        from everalgo.user_memory.reflect import EpisodeReflector

        old = _ep(_TS_SPAN_START - 1, "older narrative")
        new_episodes = [_ep(_TS_SPAN_START, "first new"), _ep(_TS_SPAN_END, "last new")]
        result = await EpisodeReflector(llm=fake).areflect(new_episodes, old_episode=old)

        assert result.episode == "Updated."
        assert result.timestamp == _TS_SPAN_END


class TestReflectErrors:
    async def test_llm_error_propagates_init(self) -> None:
        def handler(messages: list[LLMChatMessage], **kwargs: Any) -> ChatResponse:
            raise LLMError("upstream failure")

        fake = FakeLLMClient(handler=handler)
        from everalgo.user_memory.reflect import EpisodeReflector

        with pytest.raises(LLMError, match="upstream failure"):
            await EpisodeReflector(llm=fake).areflect([_ep(1000), _ep(2000)])

        assert fake.call_count == 1

    async def test_llm_error_propagates_update(self) -> None:
        def handler(messages: list[LLMChatMessage], **kwargs: Any) -> ChatResponse:
            raise LLMError("network down")

        fake = FakeLLMClient(handler=handler)
        from everalgo.user_memory.reflect import EpisodeReflector

        with pytest.raises(LLMError, match="network down"):
            await EpisodeReflector(llm=fake).areflect([_ep(2000)], old_episode=_ep(1000))


class TestReflectMisc:
    def test_sync_reflect_works(self) -> None:
        fake = FakeLLMClient(responses=['{"content": "synced", "title": "t"}'])
        from everalgo.user_memory.reflect import EpisodeReflector

        result = EpisodeReflector(llm=fake).reflect([_ep(1000), _ep(2000)])
        assert result.episode.endswith("synced")

    async def test_prompt_override_init(self) -> None:
        fake = FakeLLMClient(responses=['{"content": "ok", "title": "t"}'])
        from everalgo.user_memory.reflect import EpisodeReflector

        custom = "Custom prompt: {timeline}"
        await EpisodeReflector(llm=fake).areflect([_ep(1000), _ep(2000)], prompt=custom)

        prompt_text = fake.calls[0].messages[0].content
        assert isinstance(prompt_text, str)
        assert prompt_text.startswith("Custom prompt:")
        assert "memory consolidation" not in prompt_text

    async def test_prompt_override_update(self) -> None:
        fake = FakeLLMClient(responses=['{"content": "ok", "title": "t"}'])
        from everalgo.user_memory.reflect import EpisodeReflector

        custom = "Update: {old_episode} + {new_episodes}"
        await EpisodeReflector(llm=fake).areflect(
            [_ep(2000, "new stuff")], old_episode=_ep(1000, "old stuff"), prompt=custom
        )

        prompt_text = fake.calls[0].messages[0].content
        assert isinstance(prompt_text, str)
        assert "old stuff" in prompt_text
        assert "new stuff" in prompt_text
        assert "updating an existing" not in prompt_text.lower()


# ==========================================================================
# Language rule — reflect inherits the merged episodes' language
# ==========================================================================


def test_reflect_prompts_state_the_language_rule_at_both_ends() -> None:
    """Long prompts lose middle instructions, so each rule appears at head and tail."""
    import everalgo.user_memory.prompts.en.reflect as en_mod

    for name in ("REFLECT_EPISODE_PROMPT", "REFLECT_EPISODE_UPDATE_PROMPT"):
        assert getattr(en_mod, name).count("CRITICAL LANGUAGE RULE") == 2, name


def test_reflect_prompts_carry_no_mixed_input_judgement() -> None:
    """Reflect merges already-extracted episodes, so it inherits rather than judges the language.

    Judgement belongs to the extractor that reads the raw conversation. Re-judging here would be
    inert on single-language narratives, and a disagreeing judgement would translate a merged
    episode out of the language its sources were written in.
    """
    import everalgo.user_memory.prompts.en.reflect as en_mod

    for name in ("REFLECT_EPISODE_PROMPT", "REFLECT_EPISODE_UPDATE_PROMPT"):
        assert "dominate" not in getattr(en_mod, name), name


def test_reflect_prompts_pin_the_utc_label_on_carried_over_times() -> None:
    """Reflection is the one path that can reintroduce time drift, so its prompts must pin the format.

    It rewrites the narrative wholesale, and ``Episode.timestamp`` never reaches answer context, so a
    time that reflection drops or reformats is not recoverable from anywhere else. Both variants must
    require the ``UTC`` label and forbid dropping a time the input already carried.
    """
    import everalgo.user_memory.prompts.en.reflect as en_mod

    for name in ("REFLECT_EPISODE_PROMPT", "REFLECT_EPISODE_UPDATE_PROMPT"):
        assert "MUST carry the UTC zone label" in getattr(en_mod, name), name
