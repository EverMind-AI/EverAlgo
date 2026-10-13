"""The from-text atomic-fact prompt's system / user split: every old rule line kept as written, data only in the user.

The single-message prompt the split replaced (HEAD ``4a6f5ea``) is pinned below verbatim. Every non-blank line of it,
its JSON skeleton and worked example included, is a line of the new system prompt once its Markdown marks are dropped
(``**``, inline backticks, ``*`` bullets written ``-``), or listed in ``_RESTRUCTURED`` under the structural change that
removed it: F1 the data moved to the user message, F2 a heading or label became a tag, F4 the language rule spliced in
twice is kept once, F5 the ``### OUTPUT`` sentence became the ``<reply_format>`` sentence, F6 the skeleton lost what
kept it from being valid JSON, F7 a ``---`` rule was dropped, F8 the example's input is shown as the user message
carries it. (F3, a role sentence pointed at a tag, is not used: the sentences that name an input are C rewrites.)
Lines rewritten for their content, the C class, are listed in ``_REWRITTEN`` with what replaced them: ``TIME`` moved
to ``format_utc_time(..., weekday=True)`` and resolved dates to ISO; the reply lost its ``time`` echo; the inputs are
the snake_case ``<episode_text>`` and ``<time>`` elements, and every sentence that named ``EPISODE_TEXT`` or ``TIME``
names the element instead; and the closing instruction, which repeated what ``<reply_format>`` asks, was dropped.
"""

from __future__ import annotations

import json
import re

import everalgo.user_memory.prompts.en.atomic_fact_from_text as prompt_module
from everalgo.user_memory._replies import _AtomicFactsReply
from everalgo.user_memory.prompts.en.atomic_fact_from_text import (
    ATOMIC_FACT_FROM_TEXT_SYSTEM_PROMPT,
    ATOMIC_FACT_FROM_TEXT_USER_PROMPT,
    assemble_system_prompt,
)
from everalgo.utils import format_utc_time

# Verbatim from ``git show 4a6f5ea:packages/everalgo-user-memory/src/everalgo/user_memory/prompts/en/atomic_fact_from_text.py``
# (``ATOMIC_FACT_FROM_TEXT_PROMPT_EN``, its three concatenated parts joined).
_OLD_PROMPT = """{{LANGUAGE_RULE}}

You are an expert episodic memory extraction analyst and information architect.
Your task is to analyze the given narrative or multi-turn conversation (called "EPISODE_TEXT") and extract a set of atomic fact optimized for factual retrieval.

---

### INPUT
- EPISODE_TEXT: The memory text saved through the "episode memory".
- TIME: The start time of the episode, e.g., "March 10, 2024(Sunday) at 2:00 PM".

---

### OUTPUT
Return **only** one valid JSON object, with the following exact structure:

{
  "atomic_facts": {
    "time": "<THE EXACT TIME STRING FROM INPUT TIME>",
    "atomic_fact": [
      "<Atomic fact sentence 1>",
      "<Atomic fact sentence 2>",
      ...
    ]
  }
}

---

### EXTRACTION RULES

#### 1. Atomicity
* Each entry in `"atomic_fact"` must express **exactly one coherent unit of meaning** — an action, emotion, reason, plan, decision, or statement.
* If a speaker expresses multiple ideas (e.g., an event and its reason), split them into multiple atomic facts.
* Each atomic_fact must be **independent and retrievable on its own**.

#### 2. Time & Date Handling
* Do **not** add or prepend timestamps such as "On March 10, 2024".
* The `"time"` field (at the top level) already represents the episode's start time.
* However, when the original text mentions **explicit or relative time expressions**, you must:
  - **Preserve** explicit dates verbatim (e.g., "October 6, 2023").
  - **Resolve** relative or vague times (e.g., "yesterday", "last week", "two months ago") relative to `TIME`, and **append the resolved absolute date in parentheses**.
    Example: "Gina said she launched the campaign yesterday (March 9, 2024)."
  - If the exact resolution is uncertain, use a normalized vague phrase (e.g., "in early 2023", "during the summer of 2021") instead of guessing.

#### 3. Content Preservation
* Preserve **all semantically meaningful information**, including:
  - emotions, attitudes, reasons, intentions, consequences, conditions, and comparisons
* Resolve pronouns into explicit names or entities when unambiguous.
* Keep original wording as much as possible — only fix grammar for clarity.

#### 4. Expression Format
* Write each atomic_fact as a **single, complete sentence** in **third-person** form.
  - e.g., "Gina said that she had launched an ad campaign for her clothing store yesterday (March 9, 2024)."
* Do **not** simplify, paraphrase, or merge logically distinct ideas.

#### 5. Retrieval Clarity
* Each atomic_fact must be concise, factual, and self-contained.
* Avoid small talk or filler unless it conveys meaningful emotional or informational content.
* Ensure entities and actions are explicit and unambiguous.

#### 6. Output Requirements
* Output **only** the JSON object — no additional explanation, markdown, or commentary.
* Ensure valid JSON (proper quotes, commas, and escaping).
* The `"atomic_fact"` list should include all meaningful facts extracted from the episode.

---

### QUALITY CHECKS
Before returning the final output, verify that:
1. Every meaningful fact, intention, and emotion is included.
2. Each `atomic_fact` contains exactly one idea.
3. All relevant time references are preserved or normalized.
4. Wording remains faithful to the source.
5. The output JSON is valid and follows the exact schema.

---

### EXAMPLE

**Input:**
TIME = "March 10, 2024(Sunday) at 2:00 PM"
EPISODE_TEXT =
"Gina said she just launched an ad campaign for her clothing store yesterday.
Jon congratulated her and asked how her dance studio search was going.
Gina explained that she was focusing on the clothing store for now but still hoped to find a suitable studio soon."

**Output:**
{
  "atomic_facts": {
    "time": "March 10, 2024(Sunday) at 2:00 PM",
    "atomic_fact": [
      "Gina said that she had launched an ad campaign for her clothing store yesterday (March 9, 2024).",
      "Jon congratulated Gina on her new campaign.",
      "Jon asked Gina about the progress of her dance studio search.",
      "Gina explained that she was focusing on the clothing store for now.",
      "Gina said that she still hoped to find a suitable studio in the future."
    ]
  }
}

---

Now analyze the provided EPISODE_TEXT and TIME carefully, apply all rules above, and return **only the JSON object** in the specified format.
---

### INPUT
- EPISODE_TEXT: "{{EPISODE_TEXT}}"
- TIME: "{{TIME}}"  (the start time of the episode, e.g., "March 10, 2024(Sunday) at 2:00 PM")

{{LANGUAGE_RULE}}
"""

_OLD_REPLY_SENTENCE = "Return only one valid JSON object, with the following exact structure:"
_NEW_REPLY_SENTENCE = "Reply with one JSON object, nothing else."

# Old lines, Markdown marks dropped, that the split removed on purpose, each with the structural change that removed it.
_RESTRUCTURED: dict[str, str] = {
    "{{LANGUAGE_RULE}}": "F4",
    '- EPISODE_TEXT: "{{EPISODE_TEXT}}"': "F1",
    '- TIME: "{{TIME}}"  (the start time of the episode, e.g., "March 10, 2024(Sunday) at 2:00 PM")': "F1",
    "### INPUT": "F2",
    "### OUTPUT": "F2",
    "### EXTRACTION RULES": "F2",
    "#### 1. Atomicity": "F2",
    "#### 2. Time & Date Handling": "F2",
    "#### 3. Content Preservation": "F2",
    "#### 4. Expression Format": "F2",
    "#### 5. Retrieval Clarity": "F2",
    "#### 6. Output Requirements": "F2",
    "### QUALITY CHECKS": "F2",
    "### EXAMPLE": "F2",
    "Input:": "F2",
    "Output:": "F2",
    _OLD_REPLY_SENTENCE: "F5",
    '"<Atomic fact sentence 2>",': "F6",
    "...": "F6",
    "---": "F7",
    'TIME = "March 10, 2024(Sunday) at 2:00 PM"': "F8",
    "EPISODE_TEXT =": "F8",
    '"Gina said she just launched an ad campaign for her clothing store yesterday.': "F8",
    'Gina explained that she was focusing on the clothing store for now but still hoped to find a suitable studio soon."': "F8",
}
_STRUCTURAL_CHANGES = {"F1", "F2", "F4", "F5", "F6", "F7", "F8"}

# C: old lines, Markdown marks dropped, rewritten for their content, each with the line that replaced it, or None where
# the line was dropped: with the ``time`` echo, or as the closing instruction ``<reply_format>`` already covers.
_REWRITTEN: dict[str, str | None] = {
    (
        'Your task is to analyze the given narrative or multi-turn conversation (called "EPISODE_TEXT") and extract a set '
        "of atomic fact optimized for factual retrieval."
    ): (
        "Your task is to analyze the given narrative or multi-turn conversation (given in <episode_text>) and extract a "
        "set of atomic fact optimized for factual retrieval."
    ),
    '- EPISODE_TEXT: The memory text saved through the "episode memory".': (
        '- <episode_text>: The memory text saved through the "episode memory".'
    ),
    '- TIME: The start time of the episode, e.g., "March 10, 2024(Sunday) at 2:00 PM".': (
        '- <time>: The start time of the episode, e.g., "2024-03-10 14:00 UTC (Sunday)".'
    ),
    """- The "time" field (at the top level) already represents the episode's start time.""": (
        "- The <time> input already represents the episode's start time."
    ),
    (
        '- Resolve relative or vague times (e.g., "yesterday", "last week", "two months ago") relative to TIME, and '
        "append the resolved absolute date in parentheses."
    ): (
        '- Resolve relative or vague times (e.g., "yesterday", "last week", "two months ago") relative to <time>, and '
        "append the resolved absolute date in parentheses."
    ),
    'Example: "Gina said she launched the campaign yesterday (March 9, 2024)."': (
        'Example: "Gina said she launched the campaign yesterday (2024-03-09)."'
    ),
    '- e.g., "Gina said that she had launched an ad campaign for her clothing store yesterday (March 9, 2024)."': (
        '- e.g., "Gina said that she had launched an ad campaign for her clothing store yesterday (2024-03-09)."'
    ),
    '"Gina said that she had launched an ad campaign for her clothing store yesterday (March 9, 2024).",': (
        '"Gina said that she had launched an ad campaign for her clothing store yesterday (2024-03-09).",'
    ),
    '"time": "<THE EXACT TIME STRING FROM INPUT TIME>",': None,
    '"time": "March 10, 2024(Sunday) at 2:00 PM",': None,
    (
        "Now analyze the provided EPISODE_TEXT and TIME carefully, apply all rules above, and return only the JSON object "
        "in the specified format."
    ): None,
}
# The old example's TIME, March 10, 2024(Sunday) at 2:00 PM, as epoch milliseconds.
_EXAMPLE_INSTANT_MS = 1_710_079_200_000
_EXAMPLE_TIME = format_utc_time(_EXAMPLE_INSTANT_MS, weekday=True)

_STATIC_BLOCK_ORDER = [
    "input",
    "extraction_rules",
    "quality_checks",
    "example",
    "reply_format",
]
_RULE_SECTIONS = [
    "atomicity",
    "time_and_date_handling",
    "content_preservation",
    "expression_format",
    "retrieval_clarity",
    "output_requirements",
]
_LANGUAGE_RULE = "LANGUAGE RULE SENTINEL"
_DATA_SLOTS = {"time", "episode_text"}

_SLOT = re.compile(r"\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}")
_TAG_LINE = re.compile(r"^</?[A-Za-z_]+(?: [^>]*)?>$")
_SNAKE_CASE = re.compile(r"[a-z]+(?:_[a-z]+)*")
_JSON_OBJECT = re.compile(r"^\{\n.*?^\}$", re.DOTALL | re.MULTILINE)


def _markdown_free(line: str) -> str:
    """A stripped line without ``**`` or backticks, a leading ``*`` bullet written ``-``."""
    text = line.strip().replace("**", "").replace("`", "")
    return f"- {text[2:]}" if text.startswith("* ") else text


def _old_lines() -> list[str]:
    return [_markdown_free(line) for line in _OLD_PROMPT.splitlines() if line.strip()]


def _new_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _block(tag: str, text: str = ATOMIC_FACT_FROM_TEXT_SYSTEM_PROMPT) -> str:
    return text.split(f"<{tag}>\n", 1)[1].split(f"\n</{tag}>", 1)[0]


def _shown_replies() -> list[str]:
    """The example's reply, then the skeleton under the ``<reply_format>`` sentence."""
    return [_block("reply", _block("example")), _block("reply_format").split("\n", 1)[1]]


# ── Every old rule line is kept ─────────────────────────────────────────────────────────────────────


def test_every_kept_old_line_is_a_line_of_the_system_prompt() -> None:
    system_lines = set(_new_lines(ATOMIC_FACT_FROM_TEXT_SYSTEM_PROMPT))
    kept = [line for line in _old_lines() if line not in _RESTRUCTURED and line not in _REWRITTEN]
    missing = [line for line in kept if line not in system_lines]

    assert len(kept) > 40, "the pinned old prompt yielded too few lines, so little was checked"
    assert not missing


def test_every_restructured_line_is_a_line_of_the_old_prompt() -> None:
    """An exclusion that matches no old line would hide nothing today and anything after the next edit."""
    assert set(_RESTRUCTURED) <= set(_old_lines())
    assert set(_REWRITTEN) <= set(_old_lines())
    assert not set(_RESTRUCTURED) & set(_REWRITTEN)


def test_every_removal_is_structural() -> None:
    assert set(_RESTRUCTURED.values()) <= _STRUCTURAL_CHANGES


def test_every_rewritten_line_is_replaced_as_registered() -> None:
    system_lines = set(_new_lines(ATOMIC_FACT_FROM_TEXT_SYSTEM_PROMPT))

    for old, new in _REWRITTEN.items():
        assert old not in system_lines
        assert new is None or new in system_lines


def test_the_time_echo_is_gone_and_no_old_time_shape_is_left() -> None:
    assert '"time"' not in ATOMIC_FACT_FROM_TEXT_SYSTEM_PROMPT
    assert "2:00 PM" not in ATOMIC_FACT_FROM_TEXT_SYSTEM_PROMPT
    assert "March 9, 2024" not in ATOMIC_FACT_FROM_TEXT_SYSTEM_PROMPT


def test_the_system_prompt_adds_no_wording_of_its_own() -> None:
    """Besides tags, each new line is an old one, the reply sentence, a line F6 / F8 reshaped, or a C replacement."""
    old = set(_old_lines())
    reshaped = {
        _NEW_REPLY_SENTENCE,
        '"<Atomic fact sentence 2>"',
        _EXAMPLE_TIME,
        "Gina said she just launched an ad campaign for her clothing store yesterday.",
        "Gina explained that she was focusing on the clothing store for now but still hoped to find a suitable studio soon.",
    }
    replacements = {new for new in _REWRITTEN.values() if new is not None}
    added = [
        line
        for line in _new_lines(ATOMIC_FACT_FROM_TEXT_SYSTEM_PROMPT)
        if not _TAG_LINE.match(line) and line not in old | reshaped | replacements
    ]

    assert added == []


def test_role_sentences_open_the_system_prompt() -> None:
    """The first unchanged, the second naming ``<episode_text>`` where it named ``EPISODE_TEXT``."""
    role = ATOMIC_FACT_FROM_TEXT_SYSTEM_PROMPT.split("\n\n", 1)[0].splitlines()
    first, second = _old_lines()[1:3]

    assert role == [first, _REWRITTEN[second]]


def test_output_sentence_became_the_reply_format_sentence() -> None:
    assert _OLD_REPLY_SENTENCE not in ATOMIC_FACT_FROM_TEXT_SYSTEM_PROMPT
    assert _block("reply_format").split("\n", 1)[0] == _NEW_REPLY_SENTENCE


def test_language_rule_spliced_in_twice_is_kept_once() -> None:
    system = assemble_system_prompt(custom_instructions=None, language_rule=_LANGUAGE_RULE)

    assert _OLD_PROMPT.count("{{LANGUAGE_RULE}}") == 2
    assert system.count(_LANGUAGE_RULE) == 1
    assert _block("output_language", system) == _LANGUAGE_RULE
    assert "output_language" not in ATOMIC_FACT_FROM_TEXT_SYSTEM_PROMPT


def _without_time_echo(reply: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    return {"atomic_facts": {key: value for key, value in reply["atomic_facts"].items() if key != "time"}}


def test_old_json_objects_are_the_new_ones() -> None:
    """The skeleton and the worked example, compared as JSON with ``time`` dropped from both.

    The skeleton also loses its ``...`` line and the comma before it; the example's resolved date is written in ISO.
    """
    old_skeleton, old_example = _JSON_OBJECT.findall(_OLD_PROMPT)
    skeleton, example = (json.loads(block) for block in reversed(_shown_replies()))

    assert skeleton == _without_time_echo(json.loads(re.sub(r",\n\s*\.\.\.", "", old_skeleton)))
    assert example == _without_time_echo(json.loads(old_example.replace("(March 9, 2024)", "(2024-03-09)")))


def test_example_input_is_the_old_one_shown_as_the_user_message_carries_it() -> None:
    old_time = re.search(r'^TIME = "([^"]+)"$', _OLD_PROMPT, re.MULTILINE)
    old_text = re.search(r'^EPISODE_TEXT =\n"(.*?)"$', _OLD_PROMPT, re.DOTALL | re.MULTILINE)
    example = _block("example")

    assert old_time is not None
    assert old_text is not None
    assert old_time.group(1) == "March 10, 2024(Sunday) at 2:00 PM"
    assert _block("time", example) == _EXAMPLE_TIME == "2024-03-10 14:00 UTC (Sunday)"
    assert _block("episode_text", example) == old_text.group(1)


def test_the_event_log_variant_is_gone() -> None:
    """``EVENT_LOG_PROMPT`` and its ``event_log`` key no longer ship; the strict reply model accepts one key only."""
    assert not hasattr(prompt_module, "EVENT_LOG_PROMPT")
    assert not hasattr(prompt_module, "ATOMIC_FACT_FROM_TEXT_PROMPT_EN")
    assert set(prompt_module.__all__) == {
        "ATOMIC_FACT_FROM_TEXT_SYSTEM_PROMPT",
        "ATOMIC_FACT_FROM_TEXT_USER_PROMPT",
        "assemble_system_prompt",
    }


# ── Structure: rules in system, data in user ────────────────────────────────────────────────────────


def _top_level_blocks(text: str) -> list[str]:
    return re.findall(r"(?:\A|\n\n)<([A-Za-z_]+)>\n", text)


def test_system_blocks_follow_the_documented_order() -> None:
    """Static blocks first, then the language rule, the caller's block last; a prefix cache covers the static ones."""
    default = assemble_system_prompt(custom_instructions=None, language_rule=_LANGUAGE_RULE)
    with_custom = assemble_system_prompt(custom_instructions="Be brief.", language_rule=_LANGUAGE_RULE)

    assert _top_level_blocks(ATOMIC_FACT_FROM_TEXT_SYSTEM_PROMPT) == _STATIC_BLOCK_ORDER
    assert _top_level_blocks(default) == [*_STATIC_BLOCK_ORDER, "output_language"]
    assert _top_level_blocks(with_custom) == [*_STATIC_BLOCK_ORDER, "output_language", "custom_instructions"]
    assert default.startswith(ATOMIC_FACT_FROM_TEXT_SYSTEM_PROMPT)
    assert with_custom.startswith(ATOMIC_FACT_FROM_TEXT_SYSTEM_PROMPT)
    assert re.findall(r"^<([a-z_]+)>$", _block("extraction_rules"), re.MULTILINE) == _RULE_SECTIONS


def test_every_tag_is_closed() -> None:
    opened = re.findall(r"^<([A-Za-z_]+)>$", ATOMIC_FACT_FROM_TEXT_SYSTEM_PROMPT, re.MULTILINE)

    assert sorted(opened) == sorted(re.findall(r"^</([A-Za-z_]+)>$", ATOMIC_FACT_FROM_TEXT_SYSTEM_PROMPT, re.MULTILINE))


def test_system_prompt_carries_no_markdown() -> None:
    assert "**" not in ATOMIC_FACT_FROM_TEXT_SYSTEM_PROMPT
    assert "`" not in ATOMIC_FACT_FROM_TEXT_SYSTEM_PROMPT
    assert re.search(r"^(#|---|\s*\* )", ATOMIC_FACT_FROM_TEXT_SYSTEM_PROMPT, re.MULTILINE) is None


def test_input_block_names_exactly_the_user_message_elements() -> None:
    """``<input>`` describes each input by the tag of the user message element that carries it."""
    described = set(re.findall(r"^- <([A-Za-z_]+)>:", _block("input"), re.MULTILINE))
    user_elements = set(re.findall(r"^<([A-Za-z_]+)>$", ATOMIC_FACT_FROM_TEXT_USER_PROMPT, re.MULTILINE))

    assert described == user_elements == _DATA_SLOTS


def test_every_tag_is_snake_case() -> None:
    """Data tags follow the block tags; the example's input tags are the user message's."""
    tags = re.findall(
        r"^</?([^\s>]+)(?: [^>]*)?>$",
        ATOMIC_FACT_FROM_TEXT_SYSTEM_PROMPT + "\n" + ATOMIC_FACT_FROM_TEXT_USER_PROMPT,
        re.MULTILINE,
    )

    assert {"time", "episode_text"} <= set(tags)
    assert [tag for tag in tags if not _SNAKE_CASE.fullmatch(tag)] == []


def test_the_rules_name_the_inputs_by_their_tags() -> None:
    """No sentence calls an input by an upper-case word the user message no longer carries."""
    system = assemble_system_prompt(custom_instructions=None, language_rule=_LANGUAGE_RULE)

    assert re.findall(r"\bEPISODE_TEXT\b|\bTIME\b", system) == []
    assert "relative to <time>" in system
    assert "(given in <episode_text>)" in system


def test_the_closing_instruction_is_gone() -> None:
    """``<reply_format>`` already ends the rules asking for only the JSON object; the restatement after it is dropped."""
    assert "final_instruction" not in ATOMIC_FACT_FROM_TEXT_SYSTEM_PROMPT
    assert "Now analyze" not in ATOMIC_FACT_FROM_TEXT_SYSTEM_PROMPT
    assert ATOMIC_FACT_FROM_TEXT_SYSTEM_PROMPT.endswith("</reply_format>")


def test_system_prompt_names_no_slot() -> None:
    """Every call-level value is appended by ``assemble_system_prompt``, so the static rules carry no slot."""
    assert _SLOT.findall(ATOMIC_FACT_FROM_TEXT_SYSTEM_PROMPT) == []


def test_user_prompt_carries_only_data_slots_time_first() -> None:
    assert _SLOT.findall(ATOMIC_FACT_FROM_TEXT_USER_PROMPT) == ["time", "episode_text"]
    assert re.sub(r"</?[A-Za-z_]+>|\{\{[A-Za-z_]+\}\}", "", ATOMIC_FACT_FROM_TEXT_USER_PROMPT).strip() == ""


def test_every_reply_shown_validates_against_the_reply_model() -> None:
    """The worked example and the skeleton are what the model imitates; each must pass the model requested."""
    for block in _shown_replies():
        _AtomicFactsReply.model_validate_json(block)
