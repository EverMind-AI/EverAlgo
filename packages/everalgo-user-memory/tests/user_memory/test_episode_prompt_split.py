"""The episode prompt against the EverAlgo LoCoMo reference run's: every block word for word but one sentence.

The reference prompt (``_episode_reference_d9fe22e``) was one user message: the role sentence, the data and
custom-instruction slots, then the rules under three headings and two unheaded paragraphs. The prompt now keeps its
sentences in the operators' layout:

- Text         — the system message is the role sentence, then ``<important_time_handling>``, ``<what_to_write>``,
                 ``<requirements>``, ``<example>`` and ``<reply_format>``, each the reference text word for word; the
                 only reference lines outside them are the three headings the tags replace and the five slot lines.
                 One sentence differs: the worked example names its start time in the shared time shape
                 (``_REWORDED``).
- Instructions — ``<custom_instructions>`` closes the system message: the caller's text, or the reference run's five
                 default principles when the caller passes none.
- Language     — no ``<output_language>`` unless a language is named; a named one comes after ``<reply_format>``.
- Data         — the user message is ``<conversation_start_time>`` and ``<conversation>``, each message laid out as
                 the reference run laid it out, every time in the shared shape with its weekday and in the caller's
                 zone, then the static ``<reminder>``.
- Reply        — the skeleton and the example the prompt shows are valid ``_EpisodeReply`` objects, title first.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, timedelta, timezone
from typing import Any

import pytest

from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import ChatMessage, ConversationItem, MemCell
from everalgo.user_memory import EpisodeExtractor, OutputLanguage
from everalgo.user_memory._language import PARTICIPANT_LANGUAGE_RULE, build_language_rule
from everalgo.user_memory.episode._replies import _EpisodeReply
from everalgo.user_memory.prompts.en.episode import (
    DEFAULT_CUSTOM_INSTRUCTIONS,
    EPISODE_SYSTEM_PROMPT,
    EPISODE_USER_PROMPT,
    assemble_system_prompt,
)
from everalgo.utils import format_time, paragraphs, tagged

from . import _episode_reference_d9fe22e as reference

#: The static blocks, in the reference run's order; the role sentence precedes them.
_STATIC_BLOCKS = ["important_time_handling", "what_to_write", "requirements", "example", "reply_format"]

#: The reference lines no block holds, in order: the slot lines, whose data moved to the user message and whose
#: instructions moved to ``<custom_instructions>``, and the three headings the tags replace.
_OUTSIDE_THE_BLOCKS = [
    "Conversation start time: {conversation_start_time}",
    "Conversation content:",
    "{conversation}",
    "Custom instructions:",
    "{custom_instructions}",
    "IMPORTANT TIME HANDLING:",
    "Requirements:",
    "Example:",
]

#: The one reference sentence the prompt rewords: the worked example's start time, in the shared time shape the
#: user message now shows (``2024-03-14 15:00 UTC (Thursday)``) instead of the reference run's long form.
_REWORDED = {
    'If the conversation start time is "March 14, 2024 (Thursday) at 3:00 PM UTC" and the conversation is about '
    "Caroline planning to go hiking:": 'If the conversation start time is "2024-03-14 15:00 UTC (Thursday)" and the '
    "conversation is about Caroline planning to go hiking:",
}

#: The static block the user message closes with, word for word.
_REMINDER = (
    "Write the episode for the conversation above, following every rule in the system message: give every relative "
    "time expression its absolute date in parentheses, and keep every fact, name, number, date and stated reason "
    "from the conversation."
)

_TAGGED = re.compile(r"^<([a-z_]+)>\n(.*?)\n</\1>$", re.MULTILINE | re.DOTALL)
_TS_MS = 1_700_000_000_000  # 2023-11-14 22:13:20 UTC, a Tuesday


def _reference_text() -> str:
    """The reference prompt as the model read it: each ``{{`` / ``}}`` the single brace it stands for."""
    return reference.EPISODE_GENERATION_PROMPT.replace("{{", "{").replace("}}", "}")


def _reference_text_reworded() -> str:
    """The reference prompt with the ``_REWORDED`` sentence replaced, each found exactly once."""
    text = _reference_text()
    for old, new in _REWORDED.items():
        assert text.count(old) == 1, old[:60]
        text = text.replace(old, new)
    return text


def _parts(system: str) -> tuple[str, list[tuple[str, str]]]:
    """A system message as its role sentence and its ``(tag, body)`` blocks, checked to hold nothing else."""
    role, rest = system.split("\n\n", 1)
    blocks = _TAGGED.findall(rest)
    assert paragraphs(role, *(tagged(tag, body) for tag, body in blocks)) == system
    return role, blocks


def _static_texts() -> list[str]:
    """The role sentence and each static block's body, in order."""
    role, blocks = _parts(EPISODE_SYSTEM_PROMPT)
    return [role, *(body for _, body in blocks)]


def _reference_conversation(memcell: MemCell, *, tz: timezone = UTC) -> str:
    """The reference run's ``{conversation}`` blocks, one per chat message with text, each time in the shared shape."""
    return "\n".join(
        reference.MESSAGE_BLOCK.format(
            timestamp=format_time(item.timestamp, tz=tz, weekday=True),
            speaker=item.sender_name or item.sender_id,
            content=item.content,
        )
        for item in memcell.items
        if isinstance(item, ChatMessage) and item.content
    )


def _memcell(*texts: str) -> MemCell:
    items: list[ConversationItem] = [
        ChatMessage(
            id=f"m{index}",
            role="user",
            content=text,
            timestamp=_TS_MS + index * 60_000,
            sender_id=f"u{index}",
            sender_name=f"Speaker{index}",
        )
        for index, text in enumerate(texts)
    ]
    return MemCell(items=items, timestamp=items[-1].timestamp)


async def _sent(memcell: MemCell, **kwargs: Any) -> tuple[str, str]:
    """The system and user message contents of one extraction over ``memcell``."""
    fake = FakeLLMClient(responses=[json.dumps({"title": "t", "content": "c"})])
    await EpisodeExtractor(llm=fake).aextract(memcell, sender_id=None, **kwargs)
    system, user = fake.calls[0].messages
    assert (system.role, user.role) == ("system", "user")
    return str(system.content), str(user.content)


# ── Text: the reference run's, block for block ──────────────────────────────────────────────────────


def test_reference_braces_are_escapes_around_three_slots() -> None:
    """``_reference_text`` reads the doubled braces as the run's renderer did: filling the slots gives the same text."""
    slots = {"conversation_start_time": "START", "conversation": "CONVERSATION", "custom_instructions": "RULES"}
    expected = _reference_text()
    for name, value in slots.items():
        expected = expected.replace(f"{{{name}}}", value)

    assert reference.EPISODE_GENERATION_PROMPT.format(**slots) == expected


def test_system_message_is_the_role_sentence_and_the_blocks_in_order() -> None:
    role, blocks = _parts(assemble_system_prompt(custom_instructions=None, language_rule=None))

    assert role == _reference_text().strip().split("\n", 1)[0]
    assert [tag for tag, _ in blocks] == [*_STATIC_BLOCKS, "custom_instructions"]


def test_every_static_block_is_reference_text_word_for_word() -> None:
    """Whole reference lines, single braces, found once each and in the reference run's order; ``_REWORDED`` aside."""
    text = f"\n{_reference_text_reworded()}\n"
    positions = []
    for body in _static_texts():
        assert text.count(f"\n{body}\n") == 1, body[:60]
        positions.append(text.index(f"\n{body}\n"))

    assert positions == sorted(positions)


def test_only_the_headings_and_the_slots_stay_outside_the_blocks() -> None:
    """Cut every block out of the reference prompt: exactly the eight lines the layout replaced remain."""
    remaining = f"\n{_reference_text_reworded()}\n"
    for body in _static_texts():
        remaining = remaining.replace(f"\n{body}\n", "\n\n", 1)

    assert [line for line in remaining.splitlines() if line.strip()] == _OUTSIDE_THE_BLOCKS


def test_static_rules_name_no_slot() -> None:
    """Nothing is rendered into the rules: no ``{{name}}`` and none of the reference run's ``{name}`` slots."""
    assert re.findall(r"\{\{\w+\}\}", EPISODE_SYSTEM_PROMPT) == []
    assert re.findall(r"\{(?:conversation_start_time|conversation|custom_instructions)\}", EPISODE_SYSTEM_PROMPT) == []


# ── Instructions: the caller's, or the reference run's default principles ───────────────────────────


def test_default_principles_are_the_reference_run_s() -> None:
    assert reference.DEFAULT_CUSTOM_INSTRUCTIONS.strip("\n") == DEFAULT_CUSTOM_INSTRUCTIONS


@pytest.mark.parametrize("custom_instructions", [None, "", "  \n"])
async def test_no_or_blank_custom_instructions_send_the_default_principles(custom_instructions: str | None) -> None:
    system, _ = await _sent(_memcell("hello"), custom_instructions=custom_instructions)

    assert system == paragraphs(EPISODE_SYSTEM_PROMPT, tagged("custom_instructions", DEFAULT_CUSTOM_INSTRUCTIONS))


async def test_custom_instructions_replace_the_default_principles_verbatim() -> None:
    """The caller's text takes the principles' place as the last block, never rendered as a template."""
    text = "Rules {{conversation}} stay literal."

    system, user = await _sent(_memcell("hello"), custom_instructions=text)

    assert system == f"{EPISODE_SYSTEM_PROMPT}\n\n<custom_instructions>\n{text}\n</custom_instructions>"
    assert DEFAULT_CUSTOM_INSTRUCTIONS.splitlines()[0] not in system
    assert text not in user


# ── Language: a rule only when the caller names one ─────────────────────────────────────────────────


async def test_no_language_named_sends_no_language_rule() -> None:
    """The reference run sent none, so the shared participant rule does not stand in for it."""
    system, _ = await _sent(_memcell("hello"))

    assert "<output_language>" not in system
    assert PARTICIPANT_LANGUAGE_RULE not in system


@pytest.mark.parametrize("custom_instructions", [None, "Keep brand names in English."])
async def test_a_named_language_comes_after_reply_format_and_before_custom_instructions(
    custom_instructions: str | None,
) -> None:
    system, _ = await _sent(
        _memcell("hello"), custom_instructions=custom_instructions, output_language=OutputLanguage.CHINESE
    )

    _, blocks = _parts(system)
    assert [tag for tag, _ in blocks] == [*_STATIC_BLOCKS, "output_language", "custom_instructions"]
    assert dict(blocks)["output_language"] == build_language_rule(OutputLanguage.CHINESE)


# ── Data: the start time and the messages, as the reference run rendered them ───────────────────────


async def test_user_message_is_the_start_time_the_reference_message_blocks_and_the_reminder() -> None:
    """Empty messages dropped; the blank line the reference run put before the first block is gone."""
    cell = _memcell("first", "", "third")
    conversation = _reference_conversation(cell)
    blocks = conversation.removeprefix("\n")

    _, user = await _sent(cell)

    assert blocks != conversation
    assert user == (
        "<conversation_start_time>\n2023-11-14 22:13 UTC (Tuesday)\n</conversation_start_time>\n\n"
        f"<conversation>\n{blocks}\n</conversation>\n\n<reminder>\n{_REMINDER}\n</reminder>"
    )
    assert "count" not in EPISODE_USER_PROMPT


async def test_times_are_shown_in_the_zone_passed() -> None:
    """22:13 UTC on Tuesday is 07:13 on Wednesday in UTC+9: the start time and every message time move."""
    tz = timezone(timedelta(hours=9))
    cell = _memcell("first", "second")

    _, user = await _sent(cell, tz=tz)

    assert "<conversation_start_time>\n2023-11-15 07:13 UTC+09:00 (Wednesday)\n</conversation_start_time>" in user
    assert _reference_conversation(cell, tz=tz).removeprefix("\n") in user
    assert '"timestamp": 2023-11-15 07:14 UTC+09:00 (Wednesday),' in user


def test_the_reworded_sentence_is_the_only_departure_from_the_reference_lines() -> None:
    """Without ``_REWORDED`` the example's opening sentence is the one reference line no block holds verbatim."""
    text = f"\n{_reference_text()}\n"
    missing = [body for body in _static_texts() if text.count(f"\n{body}\n") != 1]

    assert len(missing) == 1
    assert next(iter(_REWORDED.values())) in missing[0]


async def test_system_message_is_the_same_for_every_conversation() -> None:
    """The data lives in the user message only, so with the host's arguments fixed the system message is a constant."""
    first, _ = await _sent(_memcell("hello"))
    second, _ = await _sent(_memcell("something else", "and more"))

    assert first == second


# ── Reply: what the prompt shows is what the code accepts ───────────────────────────────────────────


@pytest.mark.parametrize("tag", ["what_to_write", "example"])
def test_shown_replies_are_valid_replies_title_first(tag: str) -> None:
    """The described skeleton and the example reply, each after its block's opening sentence."""
    shown = dict(_parts(EPISODE_SYSTEM_PROMPT)[1])[tag].split("\n", 1)[1]

    _EpisodeReply.model_validate_json(shown)
    assert list(json.loads(shown)) == ["title", "content"]
