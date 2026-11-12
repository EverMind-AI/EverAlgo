"""Request contracts for the batch and incremental step paths.

Both paths judge by the step rule blocks, which keep their historical sentence-by-sentence checks; the batch prompt
and the strict schema's structure are frozen to the version measured on LoCoMo. Both paths exercise the assembled
system and data messages.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

import pytest

from everalgo.boundary import BoundaryDetector
from everalgo.boundary._replies import Reply, _StepReply
from everalgo.boundary.prompts.en.chat import (
    BATCH_BOUNDARY_DETECT_SYSTEM_PROMPT,
    BATCH_BOUNDARY_DETECT_USER_PROMPT,
    STEP_BOUNDARY_DETECT_SYSTEM_PROMPT,
    STEP_BOUNDARY_DETECT_USER_PROMPT,
)
from everalgo.llm import to_openai_response_format
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import ChatMessage
from everalgo.utils import format_time

if TYPE_CHECKING:
    from collections.abc import Callable

    from everalgo.llm import StrictOutputModel

_OLD_STEP_PROMPT = """
You are an episodic memory boundary detection expert. You need to determine if the newly added dialogue should end the current episode and start a new one.

Current conversation history:
{{conversation_history}}

Time gap information:
{{time_gap_info}}

Newly added messages:
{{new_messages}}

Please carefully analyze the following aspects to determine if a new episode should begin:

1. **Substantive Topic Change** (Highest Priority):
   - Do the new messages introduce a completely different substantive topic with meaningful content?
   - Is there a shift from one specific event/experience to another distinct event/experience?
   - Has the conversation moved from one meaningful question to an unrelated new question?

2. **Intent and Purpose Transition**:
   - Has the fundamental purpose of the conversation changed significantly?
   - Has the core question or issue of the current topic been fully resolved and a new substantial topic begun?

3. **Meaningful Content Assessment**:
   - **IMPORTANT**: Ignore pure greetings, small talk, transition phrases, and social pleasantries
   - Focus only on content that would be memorable and worth recalling later
   - Consider: Would a person remember this as part of the main conversation topic or as a separate discussion?

4. **Structural and Temporal Signals**:
   - Are there explicit topic transition phrases introducing substantial new content?
   - Are there clear concluding statements followed by genuinely new topics?
   - Is there a significant time gap between messages?

5. **Content Relevance and Independence**:
   - How related is the new substantive content to the previous meaningful discussion?
   - Does it involve completely different events, experiences, or substantial topics?

**Special Rules for Common Patterns**:
- **Greetings + Topic**: "Hey!" followed by actual content should be ONE episode
- **Transition Phrases**: "By the way", "Oh, also", "Speaking of which" usually continue the same episode unless introducing major topic shifts
- **Social Closures and Farewells**: "Thanks!", "Take care!", "Talk to you soon!", "I'm off to go...", "See you later!" should continue the current episode as natural conversation endings
- **Supportive Responses**: Brief encouragement or acknowledgment should usually continue the current episode

Decision Principles:
- **Prioritize meaningful content**: Each episode should contain substantive, memorable content
- **Ignore social formalities**: Don't split on greetings, pleasantries, brief transitions, or conversation closures
- **Treat closures as episode endings**: Messages that announce departure ("I'm off to go...", "Talk to you soon!") or provide closure ("Thanks!", "Take care!") should stay with the current episode as natural endings
- **Consider time gaps**: Long time gaps (hours or days) strongly suggest new episodes, while short gaps (minutes) usually indicate continuing conversation
- **Episodic memory focus**: Think about what a person would naturally group together when recalling this conversation
- **Reasonable episode length**: Aim for episodes with 3-20 meaningful exchanges
- **When in doubt, consider context**: If unsure, keep related content together rather than over-splitting

Please return your judgment in JSON format:
{
    "reasoning": "One sentence summary of your reasoning process",
    "should_end": true/false,
    "confidence": 0.0-1.0,
    "topic_summary": "If should_end = true, summarize the core meaningful topic of the current episode, otherwise leave it blank"
}

Note:
- If conversation history is empty, this is the first message, return false
- Focus on episodic memory principles: what would people naturally remember as distinct experiences?
- Each episode should contain substantive content that stands alone as a meaningful memory unit
"""

_STEP_RESTRUCTURED = {
    "Current conversation history:": "F1",
    "{{conversation_history}}": "F1",
    "Time gap information:": "F1",
    "{{time_gap_info}}": "F1",
    "Newly added messages:": "F1",
    "{{new_messages}}": "F1",
    "Special Rules for Common Patterns:": "F2",
    "Decision Principles:": "F2",
    "Note:": "F2",
    "Please return your judgment in JSON format:": "F5",
}

_REFERENCE_STEP_TASK = (
    "You need to determine if the newly added dialogue should end the current episode and start a new one."
)

_SYSTEM = {"batch": BATCH_BOUNDARY_DETECT_SYSTEM_PROMPT, "step": STEP_BOUNDARY_DETECT_SYSTEM_PROMPT}
_USER = {"batch": BATCH_BOUNDARY_DETECT_USER_PROMPT, "step": STEP_BOUNDARY_DETECT_USER_PROMPT}
_REPLY_MODEL: dict[str, type[StrictOutputModel]] = {"batch": Reply, "step": _StepReply}
_DATA_SLOTS = {"batch": {"count", "messages"}, "step": {"conversation_history", "time_gap_info", "new_messages"}}
_PATHS = ["batch", "step"]

_FENCED_JSON = re.compile(r"```json\n(.*?)\n```", re.DOTALL)
_BARE_JSON = re.compile(r"^\{\n.*?^\}$", re.DOTALL | re.MULTILINE)
_ISO_TIME = re.compile(r"(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}):\d{2}Z")
_SENTENCE_BREAK = re.compile(r"(?<=\.)\s+(?=[A-Z])")
_SLOT = re.compile(r"\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}")
_REPLY = re.compile(r"<reply>\n(.*?)\n</reply>", re.DOTALL)
_BATCH_LINE = re.compile(r"\[\d+\] \[\d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC \([A-Z][a-z]+day\)\] ")
_STEP_LINE = re.compile(r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC \([A-Z][a-z]+day\)\] ")

_BASE_MS = 1_710_064_800_000  # 2024-03-10 10:00 UTC


def _old_lines(old: str) -> list[str]:
    """The old prompt's non-blank lines outside JSON objects, with the split's prompt-wide changes applied."""
    text = _BARE_JSON.sub("", _FENCED_JSON.sub("", old))
    text = _ISO_TIME.sub(_as_rendered, text).replace("**", "").replace("`", "")
    return [line.strip() for line in text.splitlines() if line.strip()]


def _as_rendered(match: re.Match[str]) -> str:
    """An old ISO 8601 example time in the shape the renderer writes, weekday included."""
    instant = datetime.fromisoformat(match.group(0))
    return format_time(int(instant.timestamp() * 1000), weekday=True)


def _msg(index: int, text: str, *, minutes: int) -> ChatMessage:
    return ChatMessage(
        id=f"m{index}",
        role="user",
        content=text,
        timestamp=_BASE_MS + minutes * 60_000 + 59_000,
        sender_id=f"u{index}",
        sender_name=f"Speaker{index}",
    )


def _dialogue(n: int) -> list[ChatMessage]:
    return [_msg(i, f"text {i}", minutes=i) for i in range(n)]


def _batch_client() -> FakeLLMClient:
    return FakeLLMClient(responses=[json.dumps({"reasoning": "r", "boundaries": []})])


def _step_client() -> FakeLLMClient:
    return FakeLLMClient(responses=[json.dumps({"reasoning": "r", "should_end": False})])


def test_step_keeps_its_historical_rule_sentences() -> None:
    kept = [line for line in _old_lines(_OLD_STEP_PROMPT) if line not in _STEP_RESTRUCTURED]
    missing = [
        sentence
        for line in kept
        for sentence in _SENTENCE_BREAK.split(line)
        if sentence not in STEP_BOUNDARY_DETECT_SYSTEM_PROMPT
    ]
    assert kept
    assert missing == [_REFERENCE_STEP_TASK]
    assert set(_STEP_RESTRUCTURED) <= set(_old_lines(_OLD_STEP_PROMPT))


def test_batch_request_uses_the_evaluated_prompt_and_schema() -> None:
    """Guard the evaluated prompt and the schema the model sees: structure only, no docstring or derived title."""
    assert hashlib.sha256(BATCH_BOUNDARY_DETECT_SYSTEM_PROMPT.encode()).hexdigest() == (
        "cbee45b8b9c688964cc0731248719ce8aff8fd60e4305633918ba8a73da286fc"
    )
    assert to_openai_response_format(Reply) == {
        "type": "json_schema",
        "json_schema": {
            "name": "Reply",
            "schema": {
                "additionalProperties": False,
                "properties": {
                    "reasoning": {"type": "string"},
                    "boundaries": {"items": {"type": "integer"}, "type": "array"},
                },
                "required": ["reasoning", "boundaries"],
                "type": "object",
            },
            "strict": True,
        },
    }


def _block(path: str, tag: str) -> str:
    """The text inside one of a system prompt's tagged blocks."""
    return _SYSTEM[path].split(f"<{tag}>\n", 1)[1].split(f"\n</{tag}>", 1)[0]


@pytest.mark.parametrize("tag", ["analysis_aspects", "special_rules_for_common_patterns", "decision_principles"])
def test_batch_judges_by_the_step_rule_blocks_verbatim(tag: str) -> None:
    assert _block("batch", tag) == _block("step", tag)


def test_batch_notes_replace_only_the_first_step_note() -> None:
    batch, step = _block("batch", "notes").splitlines(), _block("step", "notes").splitlines()

    assert batch[0] == "- Message [0] always belongs to the current episode, so it is never a boundary"
    assert batch[1:] == step[1:]


# ── Structure: rules in system, data in user ────────────────────────────────────────────────────────


@pytest.mark.parametrize("path", _PATHS)
def test_system_prompt_carries_no_markdown(path: str) -> None:
    system = _SYSTEM[path]

    assert "**" not in system
    assert "`" not in system
    assert re.search(r"^#", system, re.MULTILINE) is None


@pytest.mark.parametrize("path", _PATHS)
def test_system_prompt_blocks_are_closed(path: str) -> None:
    """Block tags stand alone on their lines; a tag named inside a sentence is a reference, not a block."""
    system = _SYSTEM[path]
    opened = re.findall(r"^<([a-z_]+)(?: [^>]*)?>$", system, re.MULTILINE)

    assert opened
    assert sorted(opened) == sorted(re.findall(r"^</([a-z_]+)>$", system, re.MULTILINE))


def test_the_step_role_sentence_asks_whether_the_new_dialogue_starts_an_episode() -> None:
    """The step role sentence asks whether the new dialogue starts an episode, as the batch one does.

    The reference run's task sentence asked whether it should end the current episode; ``should_end`` reports the start.
    The sentence names no data block.
    """
    role = _SYSTEM["step"].split("\n", 1)[0]

    assert role == (
        "You are an episodic memory boundary detection expert. You need to determine if the newly added dialogue "
        "starts a new episode."
    )
    assert _REFERENCE_STEP_TASK in _OLD_STEP_PROMPT
    assert not re.findall(r"<([a-z_]+)>", role)


@pytest.mark.parametrize("path", _PATHS)
def test_system_prompt_names_no_placeholder(path: str) -> None:
    assert not _SLOT.findall(_SYSTEM[path])


@pytest.mark.parametrize("path", _PATHS)
def test_user_prompt_carries_only_data_slots(path: str) -> None:
    user = _USER[path]

    assert set(_SLOT.findall(user)) == _DATA_SLOTS[path]
    assert re.sub(r"</?[a-z_]+[^>]*>|\{\{[a-z_]+\}\}", "", user).strip() == ""


def test_the_batch_reply_skeleton_shows_no_example_label() -> None:
    """The batch skeleton's example ``boundaries`` is ``[]``.

    With ``reasoning`` naming start labels, a model copied the former ``[7, 15]`` into windows with no new episode.
    """
    assert json.loads(_block("batch", "reply_format").splitlines()[1])["boundaries"] == []


def test_the_batch_reply_skeleton_parses_reasoning_first_and_validates() -> None:
    """The skeleton is what the model imitates; it must parse and pass the model requested (llm-contract §4.3)."""
    skeleton = _block("batch", "reply_format").splitlines()[1]

    assert list(json.loads(skeleton)) == ["reasoning", "boundaries"]
    _REPLY_MODEL["batch"].model_validate_json(skeleton)


_STEP_REPLY_FORMAT = """\
Please return your judgment in JSON format:
{
    "reasoning": "One sentence of at most 25 words saying whether the newly added dialogue starts a new episode, and why",
    "should_end": true/false
}"""


def test_the_step_reply_format_is_the_measured_two_field_skeleton_and_shows_no_example_reply() -> None:
    """Exactly the measured text, and no example reply is added (llm-contract §4.3 exception).

    Every example reply tried moved the step decisions away from the reference run's.
    """
    block = _block("step", "reply_format")

    assert block == _STEP_REPLY_FORMAT
    with pytest.raises(json.JSONDecodeError):
        json.loads(block.split("\n", 1)[1])
    assert not _REPLY.findall(_SYSTEM["step"])


def test_the_step_reply_format_departs_from_the_reference_skeleton_in_two_measured_ways_only() -> None:
    """The reference run's skeleton with ``confidence`` and ``topic_summary`` dropped and ``reasoning`` reworded."""
    reference = _OLD_STEP_PROMPT.split("Please return your judgment in JSON format:\n{\n", 1)[1].split("\n}", 1)[0]
    reasoning, should_end, *dropped = reference.splitlines()
    capped = (
        '    "reasoning": "One sentence of at most 25 words saying whether the newly added dialogue starts a new '
        'episode, and why",'
    )

    assert reasoning == '    "reasoning": "One sentence summary of your reasoning process",'
    assert _block("step", "reply_format").splitlines()[2:] == [capped, should_end.rstrip(","), "}"]
    assert [line.split(":", 1)[0].strip() for line in dropped] == ['"confidence"', '"topic_summary"']


# ── What the model receives ─────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("message_count", [1, 3])
async def test_batch_call_sends_the_rules_then_the_counted_messages(message_count: int) -> None:
    client = _batch_client()

    await BoundaryDetector(llm=client).adetect_batch(_dialogue(message_count))

    system, user = client.calls[0].messages
    body, count = str(user.content).split("\n</messages>\n\n", 1)
    lines = body.removeprefix("<messages>\n").splitlines()
    records = [line for line in lines if not line.startswith("<time_gap>")]
    assert (system.role, system.content) == ("system", BATCH_BOUNDARY_DETECT_SYSTEM_PROMPT)
    assert user.role == "user"
    assert str(user.content).startswith("<messages>\n[0] ")
    assert count == f"<message_count>\n{message_count}\n</message_count>"
    assert len(records) == message_count
    assert [line.startswith("<time_gap>") for line in lines] == [i % 2 == 1 for i in range(2 * message_count - 1)]


async def test_batch_count_is_the_window_the_model_sees_after_the_pre_cut() -> None:
    client = _batch_client()

    await BoundaryDetector(llm=client).adetect_batch(_dialogue(10), hard_message_limit=5)

    user = str(client.calls[0].messages[1].content)
    assert user.startswith("<messages>\n[0] ")
    assert user.endswith(
        "[1] [2024-03-10 10:09 UTC (Sunday)] Speaker9: text 9\n</messages>\n\n<message_count>\n2\n</message_count>"
    )


async def test_a_window_that_only_grew_keeps_its_request_prefix() -> None:
    """The count follows the records, so a longer window opens with the shorter one's records (prompt caching)."""
    short, long = _batch_client(), _batch_client()

    await BoundaryDetector(llm=short).adetect_batch(_dialogue(3))
    await BoundaryDetector(llm=long).adetect_batch(_dialogue(5))

    short_records = str(short.calls[0].messages[1].content).split("\n</messages>", 1)[0]
    assert short.calls[0].messages[0] == long.calls[0].messages[0]
    assert str(long.calls[0].messages[1].content).startswith(short_records + "\n")


async def test_step_call_sends_the_rules_then_three_data_blocks() -> None:
    client = _step_client()

    await BoundaryDetector(llm=client).adetect_step([*_dialogue(2), _msg(2, "text 2", minutes=2)])

    system, user = client.calls[0].messages
    assert (system.role, system.content) == ("system", STEP_BOUNDARY_DETECT_SYSTEM_PROMPT)
    assert (user.role, user.content) == (
        "user",
        "<conversation_history>\n"
        "[2024-03-10 10:00 UTC (Sunday)] Speaker0: text 0\n"
        "[2024-03-10 10:01 UTC (Sunday)] Speaker1: text 1\n"
        "</conversation_history>\n\n"
        "<time_gap>\nTime gap: 1 minutes (recent conversation)\n</time_gap>\n\n"
        "<new_messages>\n[2024-03-10 10:02 UTC (Sunday)] Speaker2: text 2\n</new_messages>",
    )


# ── Custom instructions ─────────────────────────────────────────────────────────────────────────────

_CUSTOM = "Treat a switch between {{project}} names as a topic change.\nKeep  it short."
_ENTRY_PATH = {
    "BoundaryDetector.adetect_batch": "batch",
    "BoundaryDetector.adetect_step": "step",
}
_ENTRIES = list(_ENTRY_PATH)


async def _sent(entry: str, **kwargs: Any) -> tuple[str, str]:
    """Drive one async entry point over a fixed conversation; return the system and user messages it sent."""
    client = _batch_client() if _ENTRY_PATH[entry] == "batch" else _step_client()
    history, new = _dialogue(2), _msg(2, "text 2", minutes=2)
    if entry == "BoundaryDetector.adetect_batch":
        await BoundaryDetector(llm=client).adetect_batch(history, **kwargs)
    else:
        await BoundaryDetector(llm=client).adetect_step([*history, new], **kwargs)
    system, user = client.calls[0].messages
    assert (system.role, user.role) == ("system", "user")
    return str(system.content), str(user.content)


@pytest.mark.parametrize("entry", _ENTRIES)
async def test_custom_instructions_close_the_system_message_verbatim(entry: str) -> None:
    """The block comes after every bundled block, its text untouched: no template rendering, no whitespace change."""
    system, user = await _sent(entry, custom_instructions=_CUSTOM)
    _, default_user = await _sent(entry)

    assert system == f"{_SYSTEM[_ENTRY_PATH[entry]]}\n\n<custom_instructions>\n{_CUSTOM}\n</custom_instructions>"
    assert system.endswith("\n</custom_instructions>")
    assert system.index("<custom_instructions>") > len(_SYSTEM[_ENTRY_PATH[entry]])
    assert "{{project}}" in system
    assert user == default_user


@pytest.mark.parametrize("custom_instructions", [None, "", "  \n\t"], ids=["none", "empty", "blank"])
@pytest.mark.parametrize("entry", _ENTRIES)
async def test_no_or_blank_custom_instructions_send_the_bundled_rules_byte_for_byte(
    entry: str, custom_instructions: str | None
) -> None:
    system, _ = await _sent(entry, custom_instructions=custom_instructions)

    assert system == _SYSTEM[_ENTRY_PATH[entry]]
    assert "<custom_instructions>" not in system


@pytest.mark.parametrize(
    "operator",
    [BoundaryDetector.adetect_batch, BoundaryDetector.adetect_step],
    ids=lambda operator: operator.__qualname__,
)
def test_custom_instructions_is_keyword_only_and_prompt_is_gone(operator: Callable[..., Any]) -> None:
    parameters = inspect.signature(operator).parameters

    assert "prompt" not in parameters
    assert parameters["custom_instructions"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["custom_instructions"].default is None


@pytest.mark.parametrize("entry", _ENTRIES)
async def test_passing_prompt_raises_type_error(entry: str) -> None:
    with pytest.raises(TypeError, match="prompt"):
        await _sent(entry, prompt="CUSTOM RULES")


def test_sync_bridges_forward_custom_instructions_and_reject_prompt() -> None:
    detector_batch, detector_step = _batch_client(), _step_client()
    block = f"<custom_instructions>\n{_CUSTOM}\n</custom_instructions>"

    BoundaryDetector(llm=detector_batch).detect_batch(_dialogue(2), custom_instructions=_CUSTOM)
    BoundaryDetector(llm=detector_step).detect_step(_dialogue(3), custom_instructions=_CUSTOM)

    for client in (detector_batch, detector_step):
        assert str(client.calls[0].messages[0].content).endswith(block)
    with pytest.raises(TypeError, match="prompt"):
        # The removed keyword, on purpose: a type checker rejects it too.
        BoundaryDetector(llm=_batch_client()).detect_batch(_dialogue(2), prompt="CUSTOM RULES")  # type: ignore[call-arg]


# ── Time shape ──────────────────────────────────────────────────────────────────────────────────────


_UTC_PLUS_14 = timezone(timedelta(hours=14))


async def test_every_entry_point_shows_message_times_in_the_zone_passed() -> None:
    """10:00 UTC on Sunday is 00:00 on Monday in UTC+14: the date, clock and weekday move, the offset is labelled."""
    batch, step = _batch_client(), _step_client()

    await BoundaryDetector(llm=batch).adetect_batch(_dialogue(1), tz=_UTC_PLUS_14)
    await BoundaryDetector(llm=step).adetect_step(_dialogue(3), tz=_UTC_PLUS_14)

    assert "[0] [2024-03-11 00:00 UTC+14:00 (Monday)] Speaker0: text 0" in str(batch.calls[0].messages[1].content)
    assert "[2024-03-11 00:01 UTC+14:00 (Monday)] Speaker1: text 1" in str(step.calls[0].messages[1].content)
    assert "[2024-03-11 00:02 UTC+14:00 (Monday)] Speaker2: text 2" in str(step.calls[0].messages[1].content)


async def test_rendered_lines_show_minutes_and_the_utc_label() -> None:
    """Seconds are truncated, not rounded: every timestamp here sits at :59 seconds."""
    batch, step = _batch_client(), _step_client()

    await BoundaryDetector(llm=batch).adetect_batch(_dialogue(3))
    await BoundaryDetector(llm=step).adetect_step(_dialogue(3))

    batch_lines = [line for line in str(batch.calls[0].messages[1].content).splitlines() if line.startswith("[")]
    step_lines = [line for line in str(step.calls[0].messages[1].content).splitlines() if line.startswith("[")]
    assert batch_lines and all(_BATCH_LINE.match(line) for line in batch_lines)
    assert len(step_lines) == 3 and all(_STEP_LINE.match(line) for line in step_lines)
    assert batch_lines[0] == "[0] [2024-03-10 10:00 UTC (Sunday)] Speaker0: text 0"
