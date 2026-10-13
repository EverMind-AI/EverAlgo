"""The reflect prompts' system / user split: every old rule line kept as written, data only in the user message.

The two single-message prompts the split replaced (HEAD ``4a6f5ea``) are pinned below verbatim. Every non-blank line
of them is a line of its mode's new system prompt once the backticks around ``summary`` are dropped, or listed in
``_RESTRUCTURED`` under the structural change that removed it: F1 the data and the label in front of it moved to the
user message, F3 the INIT line naming "the following episode summaries" now points at ``<episodes>``, F4 the language
rule spliced in twice is kept once. One line was added, the ``<reply_format>`` skeleton: the old prompts named no
field, which reached the model only through the schema, and the contract asks every prompt for a parseable sample
(``docs/concepts/llm-contract.md`` §4.3). One line per mode was later rewritten for its content, C1: the time rule
follows episode v2.2 and keeps every time as written, adding or dropping no zone label, where the old rule demanded a
UTC label on every clock time and so invented one for the clock times speakers quoted.
"""

from __future__ import annotations

import json
import re

import pytest

from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import Episode
from everalgo.user_memory import EpisodeReflector
from everalgo.user_memory._language import EXISTING_NARRATIVE_LANGUAGE_RULE, MERGED_EPISODES_LANGUAGE_RULE
from everalgo.user_memory._replies import _ReflectOutput
from everalgo.user_memory.prompts.en.reflect import (
    REFLECT_EPISODE_SYSTEM_PROMPT,
    REFLECT_EPISODE_UPDATE_SYSTEM_PROMPT,
    REFLECT_EPISODE_UPDATE_USER_PROMPT,
    REFLECT_EPISODE_USER_PROMPT,
    assemble_system_prompt,
)

# Verbatim from ``git show 4a6f5ea:packages/everalgo-user-memory/src/everalgo/user_memory/prompts/en/reflect.py``.
_OLD_INIT_PROMPT = """{{language_rule}}

You are a memory consolidation assistant.

Below are the following episode summaries about the same topic, listed chronologically.
Each episode was written at a point in time with limited context. You can now
see the full timeline and produce a narrative that is more accurate and complete
than any individual episode.

Merge them into a single coherent narrative that:
- Preserves ALL factual details: names, dates, locations, specific actions, quantities, and status changes
- Resolves contradictions by keeping the latest state
- Maintains chronological flow with dates preserved
- Keeps every time exactly as the episodes wrote it. Every absolute time that states a clock time MUST carry the UTC zone label ("2024-03-14 15:00 UTC", never "2024-03-14 15:00" and never a bare "15:00"); a date with no clock time needs none. Do NOT reformat, convert, or drop a time the episodes already carry — no episode may lose its time in the merge
- Removes redundant information
- Ends with a brief summary of the current state as of the latest episode

Also return a `summary`: a faithful preview of the merged narrative, readable on its own without it. Name the main participants and what happened, ending on the current state as of the latest episode. Introduce no fact the narrative does not already carry, and do not refer to the record itself — no "this narrative", "the above". When you mention a time, write it in the format the narrative uses. Under 50 words; use as few as the narrative needs.

Reply with one JSON object, nothing else.

{{language_rule}}

Episodes:
{{timeline}}"""

_OLD_UPDATE_PROMPT = """{{language_rule}}

You are updating an existing memory narrative with new information.

Current narrative:
{{old_episode}}

New episodes (chronologically ordered):
{{new_episodes}}

Update the narrative to incorporate the new information:
- Correct any statements that are now outdated
- Append new events in chronological position
- Preserve content that is still accurate
- Maintain all factual details: names, dates, locations, specific actions
- Keep every time exactly as the narrative and the new episodes wrote it. Every absolute time that states a clock time MUST carry the UTC zone label ("2024-03-14 15:00 UTC", never "2024-03-14 15:00" and never a bare "15:00"); a date with no clock time needs none. Do NOT reformat, convert, or drop a time that is already there
- End with an updated summary of the current state

Also return a `summary`: a faithful preview of the merged narrative, readable on its own without it. Name the main participants and what happened, ending on the current state as of the latest episode. Introduce no fact the narrative does not already carry, and do not refer to the record itself — no "this narrative", "the above". When you mention a time, write it in the format the narrative uses. Under 50 words; use as few as the narrative needs.

Reply with one JSON object, nothing else.

{{language_rule}}"""

_OLD_POINTER = "Below are the following episode summaries about the same topic, listed chronologically."
_NEW_POINTER = _OLD_POINTER.replace("the following episode summaries", "the episode summaries in <episodes>")
_SKELETON = '{"content": "...", "summary": "...", "title": "..."}'
_REPLY_SENTENCE = "Reply with one JSON object, nothing else."

_OLD = {"init": _OLD_INIT_PROMPT, "update": _OLD_UPDATE_PROMPT}
_SYSTEM = {"init": REFLECT_EPISODE_SYSTEM_PROMPT, "update": REFLECT_EPISODE_UPDATE_SYSTEM_PROMPT}
_USER = {"init": REFLECT_EPISODE_USER_PROMPT, "update": REFLECT_EPISODE_UPDATE_USER_PROMPT}

# Old lines, backticks dropped, that the split removed on purpose, each with the structural change that removed it.
_RESTRUCTURED: dict[str, dict[str, str]] = {
    "init": {
        "{{language_rule}}": "F4",
        _OLD_POINTER: "F3",
        "Episodes:": "F1",
        "{{timeline}}": "F1",
    },
    "update": {
        "{{language_rule}}": "F4",
        "Current narrative:": "F1",
        "{{old_episode}}": "F1",
        "New episodes (chronologically ordered):": "F1",
        "{{new_episodes}}": "F1",
    },
}
_STRUCTURAL_CHANGES = {"F1", "F3", "F4"}
# C1: the one old line per mode rewritten for its content, found by the phrase it dropped, and the line replacing it.
_C1_DROPPED_PHRASE = "MUST carry the UTC zone label"
_C1_TIME_RULE = {
    "init": (
        "- Keeps every time exactly as the episodes wrote it: do not reformat or convert it, and do not add or drop a "
        'zone label — a time written with "UTC" keeps it, a clock time written without one ("at 4", "6:40") stays '
        "without, because its timezone is unknown. No episode may lose its time in the merge"
    ),
    "update": (
        "- Keep every time exactly as the narrative and the new episodes wrote it: do not reformat or convert it, and "
        'do not add or drop a zone label — a time written with "UTC" keeps it, a clock time written without one ("at '
        '4", "6:40") stays without, because its timezone is unknown. Never drop a time that is already there'
    ),
}
_ADDED = {
    "init": {_NEW_POINTER, _SKELETON, _C1_TIME_RULE["init"]},
    "update": {_SKELETON, _C1_TIME_RULE["update"]},
}

_STATIC_BLOCK_ORDER = ["narrative_rules", "summary_rules", "reply_format"]
_LANGUAGE_RULE = "LANGUAGE RULE SENTINEL"
_FALLBACK_RULE = {"init": MERGED_EPISODES_LANGUAGE_RULE, "update": EXISTING_NARRATIVE_LANGUAGE_RULE}
_REPLY = '{"content": "c", "summary": "s", "title": "t"}'
_DATA_SLOTS = {"init": ["timeline"], "update": ["old_episode", "new_episodes"]}
_USER_ELEMENTS = {"init": ["episodes"], "update": ["current_narrative", "new_episodes"]}

_SLOT = re.compile(r"\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}")
_TAG_LINE = re.compile(r"^</?[a-z_]+(?: [^>]*)?>$")

_MODES = pytest.mark.parametrize("mode", ["init", "update"])


def _old_lines(mode: str) -> list[str]:
    return [line.strip().replace("`", "") for line in _OLD[mode].splitlines() if line.strip()]


def _c1_old_line(mode: str) -> str:
    (line,) = [line for line in _old_lines(mode) if _C1_DROPPED_PHRASE in line]
    return line


def _new_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _block(tag: str, text: str) -> str:
    return text.split(f"<{tag}>\n", 1)[1].split(f"\n</{tag}>", 1)[0]


def _ep(ts: int, text: str) -> Episode:
    return Episode(owner_id=None, episode=text, subject="topic", summary="a preview", timestamp=ts)


# ── Every old rule line is kept ─────────────────────────────────────────────────────────────────────


@_MODES
def test_every_kept_old_line_is_a_line_of_the_system_prompt(mode: str) -> None:
    system_lines = set(_new_lines(_SYSTEM[mode]))
    kept = [line for line in _old_lines(mode) if line not in _RESTRUCTURED[mode] and line != _c1_old_line(mode)]
    missing = [line for line in kept if line not in system_lines]

    assert len(kept) >= 9, "the pinned old prompt yielded too few lines, so little was checked"
    assert not missing


@_MODES
def test_every_restructured_line_is_a_line_of_the_old_prompt(mode: str) -> None:
    """An exclusion that matches no old line would hide nothing today and anything after the next edit."""
    assert set(_RESTRUCTURED[mode]) <= set(_old_lines(mode))


@_MODES
def test_every_removal_is_structural(mode: str) -> None:
    assert set(_RESTRUCTURED[mode].values()) <= _STRUCTURAL_CHANGES


@_MODES
def test_the_system_prompt_adds_only_the_pointer_the_skeleton_and_the_c1_time_rule(mode: str) -> None:
    old = set(_old_lines(mode))
    added = {line for line in _new_lines(_SYSTEM[mode]) if not _TAG_LINE.match(line) and line not in old}

    assert added == _ADDED[mode]


@_MODES
def test_c1_replaces_the_utc_label_rule_with_the_as_written_rule(mode: str) -> None:
    system_lines = set(_new_lines(_SYSTEM[mode]))

    assert _c1_old_line(mode) not in system_lines
    assert _C1_TIME_RULE[mode] in system_lines
    assert _C1_DROPPED_PHRASE not in _SYSTEM[mode]


def test_init_pointer_names_the_user_element() -> None:
    assert _OLD_POINTER not in REFLECT_EPISODE_SYSTEM_PROMPT
    assert REFLECT_EPISODE_SYSTEM_PROMPT.split("\n\n")[1].splitlines()[0] == _NEW_POINTER


@_MODES
def test_role_sentence_opens_the_system_prompt_unchanged(mode: str) -> None:
    assert _SYSTEM[mode].split("\n", 1)[0] == _old_lines(mode)[1]


@_MODES
def test_language_rule_spliced_in_twice_is_kept_once(mode: str) -> None:
    system = assemble_system_prompt(_SYSTEM[mode], custom_instructions=None, language_rule=_LANGUAGE_RULE)

    assert _OLD[mode].count("{{language_rule}}") == 2
    assert system.count(_LANGUAGE_RULE) == 1
    assert _block("output_language", system) == _LANGUAGE_RULE
    assert "output_language" not in _SYSTEM[mode]


def test_both_modes_share_the_summary_rules_and_the_reply_format() -> None:
    for tag in ("summary_rules", "reply_format"):
        assert _block(tag, REFLECT_EPISODE_SYSTEM_PROMPT) == _block(tag, REFLECT_EPISODE_UPDATE_SYSTEM_PROMPT)


# ── Structure: rules in system, data in user ────────────────────────────────────────────────────────


@_MODES
def test_system_blocks_follow_the_documented_order(mode: str) -> None:
    """Static blocks first, then the language rule, the caller's block last; a prefix cache covers the static ones."""
    default = assemble_system_prompt(_SYSTEM[mode], custom_instructions=None, language_rule=_LANGUAGE_RULE)
    with_custom = assemble_system_prompt(_SYSTEM[mode], custom_instructions="Be brief.", language_rule=_LANGUAGE_RULE)

    for text, expected in (
        (_SYSTEM[mode], _STATIC_BLOCK_ORDER),
        (default, [*_STATIC_BLOCK_ORDER, "output_language"]),
        (with_custom, [*_STATIC_BLOCK_ORDER, "output_language", "custom_instructions"]),
    ):
        opened = re.findall(r"^<([a-z_]+)>$", text, re.MULTILINE)
        assert opened == expected
        assert opened == re.findall(r"^</([a-z_]+)>$", text, re.MULTILINE)
        assert text.startswith(_SYSTEM[mode])


@_MODES
def test_system_prompt_carries_no_markdown(mode: str) -> None:
    assert "**" not in _SYSTEM[mode]
    assert "`" not in _SYSTEM[mode]
    assert re.search(r"^#", _SYSTEM[mode], re.MULTILINE) is None


@_MODES
def test_reply_format_is_the_sentence_and_a_skeleton_in_schema_order(mode: str) -> None:
    sentence, skeleton = _block("reply_format", _SYSTEM[mode]).split("\n")

    assert sentence == _REPLY_SENTENCE
    assert list(json.loads(skeleton)) == list(_ReflectOutput.model_fields)
    _ReflectOutput.model_validate_json(skeleton)


@_MODES
def test_system_prompt_names_no_slot(mode: str) -> None:
    """Every call-level value is appended by ``assemble_system_prompt``, so the static rules carry no slot."""
    assert _SLOT.findall(_SYSTEM[mode]) == []


@_MODES
def test_user_prompt_carries_only_data_slots_in_the_old_order(mode: str) -> None:
    assert _SLOT.findall(_USER[mode]) == _DATA_SLOTS[mode]
    assert re.findall(r"^<([a-z_]+)", _USER[mode], re.MULTILINE) == _USER_ELEMENTS[mode]
    assert re.sub(r"</?[a-z_]+[^>]*>|\{\{[a-z_]+\}\}", "", _USER[mode]).strip() == ""


def test_the_update_label_parenthesis_became_the_order_attribute() -> None:
    assert '<new_episodes order="chronological">' in REFLECT_EPISODE_UPDATE_USER_PROMPT


# ── What the model receives ─────────────────────────────────────────────────────────────────────────


async def test_init_user_message_is_the_numbered_timeline() -> None:
    fake = FakeLLMClient(responses=['{"content": "c", "summary": "s", "title": "t"}'])
    await EpisodeReflector(llm=fake).areflect([_ep(1000, "First."), _ep(2000, "Second.")])

    system, user = fake.calls[0].messages
    assert (system.role, user.role) == ("system", "user")
    assert user.content == "<episodes>\n1. First.\n2. Second.\n</episodes>"


async def test_update_user_message_holds_the_narrative_then_the_new_episodes() -> None:
    fake = FakeLLMClient(responses=['{"content": "c", "summary": "s", "title": "t"}'])
    await EpisodeReflector(llm=fake).areflect([_ep(2000, "New.")], old_episode=_ep(1000, "Old narrative."))

    system, user = fake.calls[0].messages
    assert (system.role, user.role) == ("system", "user")
    assert user.content == (
        "<current_narrative>\nOld narrative.\n</current_narrative>\n\n"
        '<new_episodes order="chronological">\n1. New.\n</new_episodes>'
    )


@pytest.mark.parametrize("old_episode", [None, _ep(500, "Old narrative.")])
async def test_system_message_is_the_same_for_every_set_of_episodes(old_episode: Episode | None) -> None:
    """With the host's arguments fixed, the system message is a constant a prompt cache can share whole."""
    systems = []
    for texts in (["a", "b"], ["something else", "and more", "and more still"]):
        fake = FakeLLMClient(responses=['{"content": "c", "summary": "s", "title": "t"}'])
        episodes = [_ep(1000 + index, text) for index, text in enumerate(texts)]
        await EpisodeReflector(llm=fake).areflect(episodes, old_episode=old_episode)
        systems.append(fake.calls[0].messages[0].content)

    assert systems[0] == systems[1]


async def test_episode_text_is_never_rendered_as_a_template() -> None:
    """A slot name inside an episode stays literal and never pulls another field in."""
    fake = FakeLLMClient(responses=['{"content": "c", "summary": "s", "title": "t"}'])
    await EpisodeReflector(llm=fake).areflect([_ep(1000, "{{language_rule}}"), _ep(2000, "{{timeline}}")])

    assert fake.calls[0].messages[1].content == "<episodes>\n1. {{language_rule}}\n2. {{timeline}}\n</episodes>"


# ── custom_instructions: appended to the mode's rules, never a template, never an override ─────────


async def _system(mode: str, **kwargs: object) -> str:
    """The system message one ``areflect`` call of ``mode`` sends."""
    fake = FakeLLMClient(responses=[_REPLY])
    old_episode = _ep(500, "Old narrative.") if mode == "update" else None
    await EpisodeReflector(llm=fake).areflect(
        [_ep(1000, "First."), _ep(2000, "Second.")],
        old_episode=old_episode,
        **kwargs,  # type: ignore[arg-type]  # each test passes only areflect's own keywords
    )
    return str(fake.calls[0].messages[0].content)


@_MODES
async def test_default_system_message_is_the_static_rules_then_the_language_rule(mode: str) -> None:
    system = await _system(mode)

    assert system == f"{_SYSTEM[mode]}\n\n<output_language>\n{_FALLBACK_RULE[mode]}\n</output_language>"
    assert "custom_instructions" not in system


@_MODES
async def test_custom_instructions_land_verbatim_after_the_language_rule_closing_the_message(mode: str) -> None:
    """The mode's rules stay a shared prefix; the caller's block follows ``<output_language>`` and ends the message."""
    text = "Keep product names in English.\nNever drop a price."
    system = await _system(mode, custom_instructions=text)

    assert system == (
        f"{_SYSTEM[mode]}\n\n<output_language>\n{_FALLBACK_RULE[mode]}\n</output_language>"
        f"\n\n<custom_instructions>\n{text}\n</custom_instructions>"
    )
    assert system.endswith("\n</custom_instructions>")


@_MODES
@pytest.mark.parametrize("blank", ["", " ", "\n\t  \n"])
async def test_blank_custom_instructions_are_the_same_as_none(mode: str, blank: str) -> None:
    assert await _system(mode, custom_instructions=blank) == await _system(mode)


@_MODES
async def test_custom_instructions_are_never_rendered_as_a_template(mode: str) -> None:
    """Slot names, data slots and the retired single-brace shape all stay literal instead of filling or raising."""
    text = "Keep {{language_rule}}, {{timeline}}, {{old_episode}} and {timeline} literal."
    system = await _system(mode, custom_instructions=text)

    assert f"<custom_instructions>\n{text}\n</custom_instructions>" in system
    assert system.count(_FALLBACK_RULE[mode]) == 1
