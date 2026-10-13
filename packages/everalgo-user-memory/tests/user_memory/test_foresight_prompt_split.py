"""The split foresight prompt against the single-message prompt it replaces: every old line accounted for, and the new rules hold.

The single-message prompt (``FORESIGHT_GENERATION_PROMPT``, the working tree before this change, whose worked examples
already showed the weekday) is pinned below verbatim. Every non-blank line of it is a line of the new system message,
or listed in ``_RESTRUCTURED`` with the item of the rewrite that removed it. Each is one of the six:

    1  the head and tail language-rule copies became one ``<output_language>`` block at the end of the system message
    2  the Markdown headings became tags
    3  the worked examples show their input as the user message renders it (``<user_id>``, ``<user_name>``,
       ``<conversation>`` elements) and their reply in a ``<reply>`` element
    4  the ``...`` lines that stood for further items are gone, so the example replies and the reply skeleton parse
    5  the input is no longer Markdown: ``<input>`` and the user message replace the "Input (Markdown)" block
    6  the closing request restated objectives 1 and 6 and addressed data that now sits in the user message;
       ``<reply_format>`` ends the rules
"""

from __future__ import annotations

import datetime as dt
import json
import re

import pytest

from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import ChatMessage, MemCell
from everalgo.user_memory import ForesightExtractor, OutputLanguage
from everalgo.user_memory._language import PARTICIPANT_LANGUAGE_RULE
from everalgo.user_memory._replies import _ForesightsReply
from everalgo.user_memory.prompts.en.foresight import (
    FORESIGHT_SYSTEM_PROMPT,
    FORESIGHT_USER_PROMPT,
    assemble_system_prompt,
)

# Verbatim: the prompt as it stood before this change.
_OLD_PROMPT = """
{{language_rule}}

You are an advanced personal foresight analysis agent. Your task is to predict the specific impacts that a user's latest MemCell event might have on their future personal behaviors, habits, decisions, and lifestyle.

## Task Objectives:
1. **Personal-Level Association**: Analyze the event's potential impact on the user's future behavior, thinking patterns, life habits, or decision preferences from the personal perspective.
2. **Associative Prediction, Not Summary**: Based on event content, predict potential personal changes rather than repeating or summarizing the original content.
3. **Scenario Style Matching**: Predictions must match the scenario style of the event:
   - Life scenarios (e.g., health, family, leisure, learning) → Use casual language, focus on personal habits, emotional states, lifestyle, personal growth, etc.
   - Work scenarios (e.g., career development, skill improvement, work style) → Use professional language, focus on career planning, capability enhancement, work habits, professional development, etc.
4. **Personal Behavior-Oriented**: Each association should reflect the user's "potential changes" or "behavioral tendencies," focusing on individual-level future development.
5. **Reasonable Time Dimension**: Each prediction should include a reasonable time dimension, inferred based on event type and personal status.
6. **Specific and Actionable**: Each prediction should not exceed 40 words; generate up to 10 predictions (recommended 4-8). Content must be specific and verifiable.
7. **Prefer user_name**: Prefer using user_name when provided; otherwise use user_id (e.g., user_1). Avoid using generic terms like "the user."
8. **Semantic Grounding**: Predictions must remain semantically related to the input; store grounded supporting facts in evidence so the system can trace back the source.
## Output Format:
Return as a JSON object with a "foresights" array; each association includes time information and evidence:
{
  "foresights": [
    {
      "content": "XiaoMing will avoid hot/spicy food for the next week",
      "evidence": "Doctor advice: keep oral hygiene; avoid hot/spicy food for a week",
      "start_time": "2025-10-21",
      "end_time": "2025-10-28",
      "duration_days": 7
    },
    ...
  ]
}

## Example Input (Life Scenario):
- user_id: xiaoming-001
- user_name: XiaoMing
- conversation:
```text
[2025-10-21 14:05 UTC (Tuesday)] XiaoMing: The extraction went fine, but it's still sore.
[2025-10-21 14:06 UTC (Tuesday)] Doctor: Keep oral hygiene, avoid hot/spicy food for a week, and follow up if swelling worsens.
[2025-10-21 14:07 UTC (Tuesday)] XiaoMing: Got it, I'll follow the instructions and watch for symptoms.
```

## Example Output (Life Scenario):
{
  "foresights": [
    {
      "content": "XiaoMing will avoid hot/spicy food for the next week",
      "evidence": "Doctor advice: avoid hot/spicy food; keep oral hygiene for a week",
      "start_time": "2025-10-21",
      "end_time": "2025-10-28",
      "duration_days": 7
    },
    {
      "content": "XiaoMing will pay more attention to oral hygiene this week",
      "evidence": "Doctor advice: keep oral hygiene for a week",
      "start_time": "2025-10-21",
      "end_time": "2025-10-28",
      "duration_days": 7
    },
    {
      "content": "If swelling/pain worsens, XiaoMing will seek a follow-up soon",
      "evidence": "Doctor: follow up if swelling worsens",
      "start_time": "2025-10-21",
      "end_time": "2025-11-04",
      "duration_days": 14
    },
    {
      "content": "XiaoMing will avoid hard chewing for the next few days",
      "evidence": "XiaoMing said it is still sore after the extraction",
      "start_time": "2025-10-21",
      "end_time": "2025-10-25",
      "duration_days": 4
    }
    ...
  ]
}

## Example Input (Work Scenario):
- user_id: LiHua-001
- user_name: LiHua
- conversation:
```text
[2025-10-21 10:00 UTC (Tuesday)] Trainer: Today we'll cover agile planning and sprint rituals.
[2025-10-21 11:15 UTC (Tuesday)] LiHua: The daily standup structure is clearer—I can apply it to my team.
[2025-10-23 16:40 UTC (Thursday)] Trainer: Review metrics and improve collaboration after each sprint.
```

## Example Output (Work Scenario):
{
  "foresights": [
    {
      "content": "LiHua will trial a more structured standup in the team over the next two weeks",
      "evidence": "LiHua: the daily standup structure is clearer and can be applied to the team",
      "start_time": "2025-10-21",
      "end_time": "2025-11-04",
      "duration_days": 14
    },
    {
      "content": "LiHua will try to introduce sprint rituals in the next month",
      "evidence": "Training covered sprint rituals; LiHua intends to apply learnings",
      "start_time": "2025-10-21",
      "end_time": "2025-11-21",
      "duration_days": 31
    },
    {
      "content": "After the next iteration, LiHua will try to review metrics and do a retrospective",
      "evidence": "Trainer: review metrics and improve collaboration after each sprint",
      "start_time": "2025-10-21",
      "end_time": "2025-11-21",
      "duration_days": 31
    },
    {
      "content": "LiHua will pay more attention to concrete collaboration improvement actions this month",
      "evidence": "Trainer emphasized improving collaboration after each sprint",
      "start_time": "2025-10-21",
      "end_time": "2025-11-21",
      "duration_days": 31
    }
    ...
  ]
}

## Important Notes:
- **Personal-Oriented**: Focus on "personal-level future changes," content can cover life, learning, work, emotions, habits, and other personal development areas.
- **Associative Innovation**: Don't repeat original content; generate personal behavioral, habitual, or decision-making changes that the event might trigger.
- **Scenario Adaptation**: Language style must match the event scenario - use casual expressions for life scenarios, professional expressions for work scenarios.
- **Time Inference**: Reasonably infer time ranges based on event type, personal status, and common sense - don't rigidly apply fixed times.
- **Content Practicality**: Content must be specific, reasonable, practical, and usable by the system for personal foresight modeling.
- **Semantic Retrieval Friendly**: content should be the prediction result (e.g., "will choose soft food"), evidence stores the original fact (e.g., "wisdom tooth extraction"), enabling AI to retrieve relevant foresights based on user queries (e.g., "recommend food") and trace back reasons.
- **Time Information Extraction Rules:**
  - start_time: Extract the specific date when the event occurred from the MemCell's timestamp field, format: YYYY-MM-DD
  - end_time: Extract the specific end time from the original content. If there's an explicit end time (e.g., "before October 24", "2025-11-15"), extract the specific date; otherwise, reasonably infer based on event content and common sense
  - duration_days: Extract duration from the original content. If there's explicit time description (e.g., "within a week", "7 days", "one month"), extract days; otherwise, reasonably infer based on event content and common sense
  - evidence: Provide a short grounded summary (1–2 sentences) of the key supporting facts (can merge multiple lines from the transcript/summary); do not introduce new facts; keep it concise (≤40 words)
  - **Important**: Prioritize extracting explicit time information from the original text; if not available, make reasonable inferences based on event content and common sense. Time cannot be null

## Input (Markdown):
You will receive the following Markdown structure:
- user_id: {{USER_ID}}
- user_name: {{USER_NAME}}
- conversation:
```text
{{CONVERSATION_TEXT}}
```

{{language_rule}}

## Please generate 4-8 (up to 10) associations that may impact the user's future life and decisions based on the above content:

"""  # noqa: RUF001 - the en dash is the old prompt's own

_LANGUAGE_RULE = "LANGUAGE RULE"
_SYSTEM = assemble_system_prompt(language_rule=_LANGUAGE_RULE)
_SYSTEM_LINES = {line.strip() for line in _SYSTEM.splitlines() if line.strip()}
_OLD_LINES = list(dict.fromkeys(line.strip() for line in _OLD_PROMPT.splitlines() if line.strip()))

_RESTRUCTURED: dict[str, str] = {
    "{{language_rule}}": "1",
    "## Task Objectives:": "2",
    "## Output Format:": "2",
    "## Example Input (Life Scenario):": "3",
    "## Example Output (Life Scenario):": "3",
    "## Example Input (Work Scenario):": "3",
    "## Example Output (Work Scenario):": "3",
    "## Important Notes:": "2",
    "- user_id: xiaoming-001": "3",
    "- user_name: XiaoMing": "3",
    "- user_id: LiHua-001": "3",
    "- user_name: LiHua": "3",
    "- conversation:": "3",
    "```text": "3",
    "```": "3",
    "...": "4",
    "## Input (Markdown):": "5",
    "You will receive the following Markdown structure:": "5",
    "- user_id: {{USER_ID}}": "5",
    "- user_name: {{USER_NAME}}": "5",
    "{{CONVERSATION_TEXT}}": "5",
    "## Please generate 4-8 (up to 10) associations that may impact the user's future life and decisions based on the above content:": "6",
}

# What the new system message holds that the old prompt did not: the ``<input>`` block, the examples' data values.
_NEW_LINES = {
    "- user_id: the ID of the user",
    "- user_name: the name of the user",
    "- conversation: the messages, one per line",
    "xiaoming-001",
    "XiaoMing",
    "LiHua-001",
    "LiHua",
    _LANGUAGE_RULE,
}
_TAG_LINE = re.compile(r"^</?[a-z_]+(?: [a-z_]+=\"[a-z_]+\")?>$")

_CELL = MemCell(
    items=[
        ChatMessage(
            id="m1",
            role="user",
            content="I'll follow up with Bob next week.",
            timestamp=1700000000000,
            sender_id="u_alice",
            sender_name="Alice",
        )
    ],
    timestamp=1700000000000,
)


def test_every_old_line_is_in_the_new_system_message_or_registered() -> None:
    missing = [line for line in _OLD_LINES if line not in _SYSTEM_LINES and line not in _RESTRUCTURED]

    assert missing == []


def test_a_registered_line_is_gone_from_the_new_system_message() -> None:
    """A stale registration would hide a line that has since come back, or never left."""
    assert [line for line in _RESTRUCTURED if line in _SYSTEM_LINES] == []


def test_every_registered_line_is_a_line_of_the_old_prompt() -> None:
    assert [line for line in _RESTRUCTURED if line not in _OLD_LINES] == []


def test_every_new_system_line_is_an_old_line_a_tag_or_a_registered_addition() -> None:
    unexplained = [
        line
        for line in sorted(_SYSTEM_LINES)
        if line not in _OLD_LINES and line not in _NEW_LINES and not _TAG_LINE.match(line)
    ]

    assert unexplained == []


def test_the_head_and_tail_language_rule_copies_became_one_block() -> None:
    assert _OLD_PROMPT.count("{{language_rule}}") == 2
    assert _SYSTEM.count(_LANGUAGE_RULE) == 1
    assert _SYSTEM.endswith(f"<output_language>\n{_LANGUAGE_RULE}\n</output_language>")


def test_the_static_system_prompt_names_no_placeholder() -> None:
    assert "{{" not in FORESIGHT_SYSTEM_PROMPT


def test_the_old_rules_keep_their_order() -> None:
    rule_lines = [line for line in _OLD_LINES if re.match(r"\d\. \*\*|- \*\*", line)]
    system_lines = [line.strip() for line in _SYSTEM.splitlines()]
    positions = [system_lines.index(line) for line in rule_lines]

    assert len(rule_lines) == 16
    assert positions == sorted(positions)


# ---------------------------------------------------------------------------
# What goes where
# ---------------------------------------------------------------------------


async def _sent(**kwargs: object) -> tuple[str, str]:
    fake = FakeLLMClient(responses=['{"foresights": []}'])
    await ForesightExtractor(llm=fake).aextract(_CELL, sender_id="u_alice", **kwargs)  # type: ignore[arg-type]
    system, user = fake.calls[0].messages
    return str(system.content), str(user.content)


async def test_the_system_message_is_static_apart_from_the_language_rule() -> None:
    first, _ = await _sent()
    second, _ = await _sent()
    named, _ = await _sent(output_language=OutputLanguage.GERMAN)

    assert first == second == assemble_system_prompt(language_rule=PARTICIPANT_LANGUAGE_RULE)
    assert named.startswith(FORESIGHT_SYSTEM_PROMPT)
    assert first.startswith(FORESIGHT_SYSTEM_PROMPT)


async def test_the_user_message_holds_the_data_and_nothing_else() -> None:
    _, user = await _sent()

    assert user == (
        "<user_id>\nu_alice\n</user_id>\n\n"
        "<user_name>\nAlice\n</user_name>\n\n"
        "<conversation>\n[2023-11-14 22:13 UTC (Tuesday)] Alice: I'll follow up with Bob next week.\n</conversation>"
    )


def test_the_user_template_names_its_three_placeholders_in_data_order() -> None:
    assert re.findall(r"\{\{(\w+)\}\}", FORESIGHT_USER_PROMPT) == ["user_id", "user_name", "conversation"]


async def test_data_that_looks_like_a_placeholder_stays_verbatim() -> None:
    cell = MemCell(
        items=[
            ChatMessage(
                id="m1",
                role="user",
                content="{{user_id}} </conversation> {{language_rule}}",
                timestamp=1700000000000,
                sender_id="u_alice",
                sender_name="Alice",
            )
        ],
        timestamp=1700000000000,
    )
    fake = FakeLLMClient(responses=['{"foresights": []}'])

    await ForesightExtractor(llm=fake).aextract(cell, sender_id="u_alice")

    user = str(fake.calls[0].messages[1].content)
    assert "Alice: {{user_id}} </conversation> {{language_rule}}" in user
    assert "{{" not in str(fake.calls[0].messages[0].content)


# ---------------------------------------------------------------------------
# The worked examples and the reply skeleton
# ---------------------------------------------------------------------------

_EXAMPLE = re.compile(
    r'<example scenario="(?P<scenario>\w+)">\n'
    r"<user_id>\n(?P<user_id>[^\n]+)\n</user_id>\n\n"
    r"<user_name>\n(?P<user_name>[^\n]+)\n</user_name>\n\n"
    r"<conversation>\n(?P<conversation>.+?)\n</conversation>\n"
    r"<reply>\n(?P<reply>.+?)\n</reply>\n</example>",
    re.DOTALL,
)
_EXAMPLES = list(_EXAMPLE.finditer(_SYSTEM))
_CONVERSATION_LINE = re.compile(r"^\[(\d{4}-\d{2}-\d{2}) \d{2}:\d{2} UTC \((\w+)\)\] [^:]+: .+$")


def test_there_are_a_life_and_a_work_example() -> None:
    assert [m["scenario"] for m in _EXAMPLES] == ["life", "work"]


@pytest.mark.parametrize("example", _EXAMPLES, ids=lambda m: m["scenario"])
def test_an_example_reply_is_a_valid_reply(example: re.Match[str]) -> None:
    reply = _ForesightsReply.model_validate_json(example["reply"])

    assert len(reply.foresights) == 4


@pytest.mark.parametrize("example", _EXAMPLES, ids=lambda m: m["scenario"])
def test_an_example_reply_keeps_its_own_time_arithmetic(example: re.Match[str]) -> None:
    for item in _ForesightsReply.model_validate_json(example["reply"]).foresights:
        assert item.start_time is not None
        assert item.end_time is not None
        span = dt.date.fromisoformat(item.end_time) - dt.date.fromisoformat(item.start_time)
        assert span.days == item.duration_days


@pytest.mark.parametrize("example", _EXAMPLES, ids=lambda m: m["scenario"])
def test_an_example_conversation_has_the_shape_the_user_message_renders(example: re.Match[str]) -> None:
    for line in example["conversation"].splitlines():
        match = _CONVERSATION_LINE.match(line)
        assert match, line
        assert dt.date.fromisoformat(match[1]).strftime("%A") == match[2]


@pytest.mark.parametrize("example", _EXAMPLES, ids=lambda m: m["scenario"])
def test_an_example_reply_starts_inside_its_conversation(example: re.Match[str]) -> None:
    first_day = _CONVERSATION_LINE.match(example["conversation"].splitlines()[0])[1]  # type: ignore[index]
    starts = {item.start_time for item in _ForesightsReply.model_validate_json(example["reply"]).foresights}

    assert starts == {first_day}


def test_the_reply_skeleton_is_one_valid_item() -> None:
    skeleton = _SYSTEM.split("<reply_format>\n", 1)[1].split("\n</reply_format>", 1)[0]
    sentence, _, body = skeleton.partition("\n")

    assert sentence.startswith('Return as a JSON object with a "foresights" array')
    assert list(json.loads(body)) == ["foresights"]
    assert len(_ForesightsReply.model_validate_json(body).foresights) == 1


def test_the_rules_state_the_json_shape_once_and_name_the_five_fields() -> None:
    assert _SYSTEM.count("<reply_format>") == 1
    for field in ("content", "evidence", "start_time", "end_time", "duration_days"):
        assert f'"{field}"' in _SYSTEM
