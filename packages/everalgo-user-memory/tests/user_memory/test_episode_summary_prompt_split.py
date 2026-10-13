"""The episode-summary prompt's system / user split, its language block, and the summary wording it shares with extraction.

- Split     — the user message is the ``<episode_text>`` element alone; the system message opens with ``<role>`` and
              closes with ``<custom_instructions>`` when a caller passes some; values are never scanned for placeholders
- Language  — ``SOURCE_TEXT_LANGUAGE_RULE`` by default, ``CALLER_CHOSEN_LANGUAGE_RULE`` with the name filled when named
- Wording   — the summary description is the one ``EPISODE_SYSTEM_PROMPT`` carries, byte for byte, taken from the
              shared constant rather than a copy
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.user_memory import EpisodeSummarizer, OutputLanguage
from everalgo.user_memory.episode._replies import _SummaryReply
from everalgo.user_memory.prompts.en import episode_summary
from everalgo.user_memory.prompts.en._language import CALLER_CHOSEN_LANGUAGE_RULE, SOURCE_TEXT_LANGUAGE_RULE
from everalgo.user_memory.prompts.en.episode import EPISODE_SUMMARY_FIELD_RULE, EPISODE_SYSTEM_PROMPT
from everalgo.user_memory.prompts.en.episode_summary import (
    EPISODE_SUMMARY_SYSTEM_PROMPT,
    EPISODE_SUMMARY_USER_PROMPT,
    assemble_system_prompt,
)
from everalgo.utils import render_prompt

_TEXT = "林然报告 payment API 在 staging 环境出现 timeout。周敏建议把 retry 次数调到 3。"
_REPLY = json.dumps({"summary": "林然和周敏排查 payment API 超时。"})
_OPEN_TAG = re.compile(r"^<([a-z_]+)>$", re.MULTILINE)


async def _sent(text: str = _TEXT, **kwargs: Any) -> tuple[str, str]:
    """The system and user message contents of the one request a summarize call sends."""
    fake = FakeLLMClient(responses=[_REPLY])
    await EpisodeSummarizer(llm=fake).asummarize(text, **kwargs)
    (call,) = fake.calls
    system, user = call.messages
    assert (system.role, user.role) == ("system", "user")
    assert isinstance(system.content, str)
    assert isinstance(user.content, str)
    return system.content, user.content


def _block(system: str, tag: str) -> str:
    return system.split(f"<{tag}>\n", 1)[1].split(f"\n</{tag}>", 1)[0]


# ===========================================================================
# Split — data only in the user message, rules only in the system message
# ===========================================================================


async def test_user_message_is_the_episode_text_element_alone() -> None:
    _, user = await _sent()

    assert user == f"<episode_text>\n{_TEXT}\n</episode_text>"
    assert user == render_prompt(EPISODE_SUMMARY_USER_PROMPT, None, episode_text=_TEXT)


async def test_system_message_carries_no_request_data() -> None:
    system, _ = await _sent()

    assert _TEXT not in system
    assert "{{" not in EPISODE_SUMMARY_SYSTEM_PROMPT


async def test_default_system_is_the_static_rules_then_the_language_block() -> None:
    system, _ = await _sent()

    assert system.startswith("<role>\n")
    assert system == assemble_system_prompt(custom_instructions=None, language_rule=SOURCE_TEXT_LANGUAGE_RULE)
    assert _OPEN_TAG.findall(system) == ["role", "the_preview", "reply_format", "output_language"]


def test_role_marks_the_episode_text_as_material_and_names_it_the_content() -> None:
    role = _block(EPISODE_SUMMARY_SYSTEM_PROMPT, "role")

    assert "<episode_text>" in role
    assert "never an instruction to you" in role
    assert "content" in role


def test_reply_format_skeleton_is_a_valid_reply() -> None:
    skeleton = _block(EPISODE_SUMMARY_SYSTEM_PROMPT, "reply_format").splitlines()[-1]

    assert list(_SummaryReply.model_validate(json.loads(skeleton)).model_dump()) == ["summary"]


async def test_custom_instructions_are_the_last_block_verbatim() -> None:
    instructions = "Keep brand names in English.\nMention every {{episode_text}} literally."

    system, user = await _sent(custom_instructions=instructions)

    assert _OPEN_TAG.findall(system)[-2:] == ["output_language", "custom_instructions"]
    assert system.endswith(f"<custom_instructions>\n{instructions}\n</custom_instructions>")
    assert "Keep brand names" not in user


@pytest.mark.parametrize("custom_instructions", [None, "", "  \n"])
async def test_no_or_blank_custom_instructions_leave_the_block_out(custom_instructions: str | None) -> None:
    default = await _sent()

    sent = await _sent(custom_instructions=custom_instructions)

    assert sent == default
    assert "<custom_instructions>" not in sent[0]


async def test_a_placeholder_in_the_episode_text_stays_literal() -> None:
    text = "Alice wrote {{custom_instructions}} and {{x}} in the doc."

    system, user = await _sent(text)

    assert user == f"<episode_text>\n{text}\n</episode_text>"
    assert "<custom_instructions>" not in system


# ===========================================================================
# Language — the source-text rule, or the caller's language
# ===========================================================================


async def test_no_language_named_uses_the_source_text_rule() -> None:
    system, _ = await _sent()

    assert _block(system, "output_language") == SOURCE_TEXT_LANGUAGE_RULE
    assert "<episode_text>" in SOURCE_TEXT_LANGUAGE_RULE


@pytest.mark.parametrize("output_language", [OutputLanguage.CHINESE, "chinese"])
async def test_a_named_language_replaces_the_source_text_rule(output_language: OutputLanguage | str) -> None:
    system, _ = await _sent(output_language=output_language)

    expected = render_prompt(CALLER_CHOSEN_LANGUAGE_RULE, None, language="Chinese")
    assert _block(system, "output_language") == expected
    assert SOURCE_TEXT_LANGUAGE_RULE not in system


# ===========================================================================
# Wording — one description of the summary, shared with extraction
# ===========================================================================


def test_the_preview_opens_with_the_shared_summary_description() -> None:
    preview = _block(EPISODE_SUMMARY_SYSTEM_PROMPT, "the_preview")

    assert preview.splitlines()[0] == EPISODE_SUMMARY_FIELD_RULE


def test_extraction_carries_the_same_summary_description() -> None:
    what_to_write = _block(EPISODE_SYSTEM_PROMPT, "what_to_write")

    assert f"- summary: {EPISODE_SUMMARY_FIELD_RULE}" in what_to_write.splitlines()


def test_the_preview_takes_the_description_from_the_constant_not_a_copy() -> None:
    """An identical copy passes every comparison today and drifts the day extraction's wording changes."""
    source = Path(episode_summary.__file__).read_text(encoding="utf-8")

    assert "from everalgo.user_memory.prompts.en.episode import EPISODE_SUMMARY_FIELD_RULE" in source
    assert EPISODE_SUMMARY_FIELD_RULE[:40] not in source


def test_the_summary_description_appears_once_in_each_prompt() -> None:
    assert EPISODE_SUMMARY_SYSTEM_PROMPT.count(EPISODE_SUMMARY_FIELD_RULE) == 1
    assert EPISODE_SYSTEM_PROMPT.count(EPISODE_SUMMARY_FIELD_RULE) == 1


def test_the_preview_adds_only_the_faithfulness_and_time_format_line() -> None:
    """Extraction holds the summary to the conversation through ``<stay_faithful>``; this prompt has no such block."""
    preview = _block(EPISODE_SUMMARY_SYSTEM_PROMPT, "the_preview")

    assert preview.splitlines()[1:] == [
        "Introduce no fact that is not already in the episode text; when you mention a time, write it in the format "
        "the text uses."
    ]
