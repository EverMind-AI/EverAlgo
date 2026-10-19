"""The boundary prompts' system / user split: every old rule sentence kept, data only in the user message.

The single-message prompts the split replaced (HEAD ``e5a9df1``) are pinned below verbatim. Every non-blank line
of them is either in the new system prompt, sentence by sentence, after the changes the split made everywhere
(``**`` and inline backticks dropped, example times in the rendered ``YYYY-MM-DD HH:MM UTC (Weekday)`` shape), or
listed in
``_RESTRUCTURED`` under the change of the verbatim draft that removed it: F1-F6 restructured a line, C rewrote its
content on purpose, so a C line must be gone and its replacement present, D dropped a rule outright, so a D
line must be gone with nothing in its place, and E rewrote the first batch worked example, so an E line must be
gone and the new example present. The old prompts' JSON objects are compared as JSON: the worked examples, with the
field C renamed and the split E moved, must open the new ``<reply>`` objects, followed only by the examples C added;
the one skeleton each prompt had gave way to a parseable one. v6 then rewrote the
batch ``reasoning``: its definition, whose old wording lived only in that skeleton, and every worked example's,
which now lists the topics first. The old wording must be gone, and batch replies are compared without
``reasoning``.

A caller's ``custom_instructions`` close the system message as its last block, verbatim; without them the system
message is the bundled rules byte for byte. ``prompt=``, which replaced the rules, is gone from every entry point.
"""

from __future__ import annotations

import inspect
import json
import re
from datetime import datetime
from typing import TYPE_CHECKING, Any

import pytest

from everalgo.boundary import (
    BoundaryDetector,
    adetect_boundaries,
    adetect_boundary_step,
    detect_boundaries,
)
from everalgo.boundary._replies import _BatchReply, _StepReply
from everalgo.boundary.prompts.en.chat import (
    BATCH_BOUNDARY_DETECT_SYSTEM_PROMPT,
    BATCH_BOUNDARY_DETECT_USER_PROMPT,
    STEP_BOUNDARY_DETECT_SYSTEM_PROMPT,
    STEP_BOUNDARY_DETECT_USER_PROMPT,
)
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import ChatMessage
from everalgo.utils import format_utc_time

if TYPE_CHECKING:
    from collections.abc import Callable

    from everalgo.llm import StrictOutputModel

# Verbatim from ``git show e5a9df1:packages/everalgo-boundary/src/everalgo/boundary/prompts/en/chat.py``.
_OLD_BATCH_PROMPT = """
**CRITICAL LANGUAGE RULE**: You MUST output in the SAME language as the input conversation content. If the conversation content is in Chinese, ALL output MUST be in Chinese. If in English, output in English. This is mandatory.

You are an episodic memory boundary detection expert. Your task is to find all natural "episode boundaries" in a continuous conversation and split it into meaningful, independently memorable segments (MemCells). Your core principle is **"default to merging, split cautiously"**.

### Input Format
The following is a complete chronological conversation log. Each message is prefixed with a 1-based index and timestamp:

```
{{messages}}
```

### When to split

Add a boundary (by message number) only when **clear signals** appear:
- **Cross-day split (highest priority):** Adjacent messages have different calendar dates — MUST split at the date boundary.
- **Substantive topic change:** Conversation shifts from one concrete topic to a completely unrelated one (e.g., project architecture → weekend plans).
- **Task completion + new topic:** A closing message ("migration done") belongs to its task's episode; split only when the **next** message opens a genuinely unrelated topic.
- **Long gap + new topic:** Time gap > 4 hours AND new messages have no clear connection to prior conversation.

**Do NOT split for:**
- Greetings, farewells ("bye", "thanks") — keep with the main episode
- Transition phrases ("by the way", "oh also") — usually continue the current episode
- Brief pauses (< 4 hours) followed by the same topic

### `should_wait`
Set to `true` when the **last segment** has insufficient information to determine its episode context:
- **Non-text messages:** Only media placeholders (`[image]`, `[video]`, `[file]`) with no accompanying text
- **Intent-free short replies:** Minimal responses like "ok", "sure", "got it", "😂"
- **System or non-conversational messages:** System notifications (join/leave group, payment reminders, etc.) cannot themselves determine episode boundaries — wait for the next human message to decide
- **Ambiguous intermediate state:** Gap of 30 min–4 hours with content that is neither clearly continuing nor clearly starting a new topic

### Decision Principles
- **Merge by default:** When in doubt, do not split; only split on clear signals
- **Content over form:** Greetings and farewells belong to the episode they serve, not their own
- **Process continuity:** Consecutive actions toward the same goal (e.g., create group → post first instruction) form one episode
- **System messages don't trigger splits:** System messages (e.g., `"[User X] joined the group"`) have their episode context determined by the next human message that follows them

### Examples

**Example 1 — one boundary:**
Input messages:
```
[1] [2024-03-10T09:00:00Z] Alice: Can you help me debug the login issue?
[2] [2024-03-10T09:01:00Z] Bob: Sure, let me check the logs.
[3] [2024-03-10T09:05:00Z] Bob: Found it — a null pointer in AuthService line 42.
[4] [2024-03-10T09:06:00Z] Alice: Fixed, thanks!
[5] [2024-03-11T10:00:00Z] Alice: Hey, are you free for lunch today?
[6] [2024-03-11T10:01:00Z] Bob: Sure, 12:30?
```
Output:
```json
{
    "reasoning": "Messages 1-4 are a complete bug-fix episode; message 5 starts a new day with an unrelated lunch topic.",
    "boundaries": [4],
    "should_wait": false
}
```

**Example 2 — no boundary:**
Input messages:
```
[1] [2024-03-10T06:00:00Z] Alice: What's the status of the Q2 roadmap?
[2] [2024-03-10T06:02:00Z] Bob: About 60% done. Need to finalize the API specs.
[3] [2024-03-10T06:10:00Z] Alice: OK, let's review the specs tomorrow.
```
Output:
```json
{
    "reasoning": "All messages are part of the same Q2 roadmap discussion with no topic change.",
    "boundaries": [],
    "should_wait": false
}
```

### Output Format
Return strictly in the following JSON format:
```json
{
    "reasoning": "<one sentence explaining all boundary decisions>",
    "boundaries": [<1-indexed message numbers after which to split>],
    "should_wait": <boolean, whether the last segment has insufficient information>
}
```

**`boundaries: []` means all messages belong to the same episode — no split.**

**CRITICAL LANGUAGE RULE**: You MUST output in the SAME language as the input conversation content. If the conversation content is in Chinese, ALL output MUST be in Chinese. If in English, output in English. This is mandatory.
"""  # noqa: RUF001 — verbatim old prompt, en dash included

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

# Old lines, normalized as ``_old_lines`` does, that the split removed on purpose, each with the draft change that
# removed it: F1 data moved to the user message, F2 Markdown headings became tags, F3 the input description merged
# into the role sentence, F4 the output-language rule dropped, F5 the reply skeleton made parseable JSON, C the
# batch should_wait rule replaced by tail_complete, "the trailing segment already forms a complete episode", with
# the opposite polarity (draft §0a, 2026-10-10), D a batch rule dropped: the cross-day rule (2026-10-16), since the
# host already closes an idle conversation and the UTC date the timestamps carry is not the user's day, and the
# transition-phrase line (r13), a prior rather than a test that the model applied in reverse, E the first batch
# example rewritten (r13): the other speaker's courtesy reply joins the topic it closes, on the same day.
_LANGUAGE_RULE = (
    "CRITICAL LANGUAGE RULE: You MUST output in the SAME language as the input conversation content. If the "
    "conversation content is in Chinese, ALL output MUST be in Chinese. If in English, output in English. This is "
    "mandatory."
)
_CROSS_DAY_RULE = (
    "- Cross-day split (highest priority): Adjacent messages have different calendar dates — MUST split at the date "
    "boundary."
)
_TRANSITION_PHRASE_RULE = '- Transition phrases ("by the way", "oh also") — usually continue the current episode'
_RESTRUCTURED: dict[str, dict[str, str]] = {
    "batch": {
        _LANGUAGE_RULE: "F4",
        "### Input Format": "F2",
        "The following is a complete chronological conversation log. Each message is prefixed with a 1-based index "
        "and timestamp:": "F3",
        "{{messages}}": "F1",
        "### When to split": "F2",
        _CROSS_DAY_RULE: "D",
        "Do NOT split for:": "F2",
        _TRANSITION_PHRASE_RULE: "D",
        "### should_wait": "C",
        "### Decision Principles": "F2",
        "### Examples": "F2",
        "Example 1 — one boundary:": "F2",
        "[5] [2024-03-11 10:00 UTC (Monday)] Alice: Hey, are you free for lunch today?": "E",
        "[6] [2024-03-11 10:01 UTC (Monday)] Bob: Sure, 12:30?": "E",
        "Example 2 — no boundary:": "F2",
        "Input messages:": "F2",
        "Output:": "F2",
        "### Output Format": "F2",
        "Return strictly in the following JSON format:": "F5",
        "Set to true when the last segment has insufficient information to determine its episode context:": "C",
        "- Non-text messages: Only media placeholders ([image], [video], [file]) with no accompanying text": "C",
        '- Intent-free short replies: Minimal responses like "ok", "sure", "got it", "😂"': "C",
        "- System or non-conversational messages: System notifications (join/leave group, payment reminders, etc.) "
        "cannot themselves determine episode boundaries — wait for the next human message to decide": "C",
        "- Ambiguous intermediate state: Gap of 30 min–4 hours with content that is neither clearly continuing nor "  # noqa: RUF001 — verbatim old rule, en dash included
        "clearly starting a new topic": "C",
    },
    "step": {
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
    },
}

# What C put in place of its lines, and the worked examples it added after the old ones.
_REWRITTEN_TAIL_COMPLETE = """\
<tail_complete>
tail_complete says whether the messages after the last boundary (all of them when there is no boundary) already form a complete episode. An episode is complete when what it is about has been said and the messages that follow will not belong to it. Set it to false when that trailing segment:
- Only opens a topic that the next messages will continue, however clear the opening is
- Is only media placeholders ([image], [video], [file]) with no accompanying text
- Is only intent-free short replies ("ok", "sure", "got it", "😂") with nothing before them in the segment that they answer
- Is only system or non-conversational messages (join/leave group, payment reminders, etc.) — wait for the next human message to decide
- Follows a gap of 30 min–4 hours with content that is neither clearly continuing nor clearly starting a new topic
A segment that ends with an acknowledgement or farewell closing its own topic is complete: set it to true.
</tail_complete>"""  # noqa: RUF001 — en dash kept from the old rule
_REWRITTEN_REPLY_FIELD = "tail_complete says whether the trailing segment is already a complete episode."
_ADDED_EXAMPLES = {"batch": 1, "step": 0}

# What E put in place of the first batch example: a courtesy reply after the thanks, every message on one day, and
# the split after that reply.
_REWRITTEN_EXAMPLE_ONE = """\
<messages count="7">
[1] [2024-03-10 09:00 UTC (Sunday)] Alice: Can you help me debug the login issue?
[2] [2024-03-10 09:01 UTC (Sunday)] Bob: Sure, let me check the logs.
[3] [2024-03-10 09:05 UTC (Sunday)] Bob: Found it — a null pointer in AuthService line 42.
[4] [2024-03-10 09:06 UTC (Sunday)] Alice: Fixed, thanks!
[5] [2024-03-10 09:06 UTC (Sunday)] Bob: Anytime.
[6] [2024-03-10 11:30 UTC (Sunday)] Alice: Hey, are you free for lunch today?
[7] [2024-03-10 11:31 UTC (Sunday)] Bob: Sure, 12:30?
</messages>"""
_EXAMPLE_ONE_DAY = "2024-03-10"
_REWRITTEN_EXAMPLE_ONE_SPLIT = {"boundaries": [5]}

# v6 (draft §0c) asks for the topics in order before the split points, in place of one sentence on all decisions;
# r8 bounds that list: each topic is a short label with its message range, then only the split points follow.
# Every batch worked example's reasoning takes that shape, and its split points are the reply's boundaries.
_OLD_REASONING_DEFINITION = "one sentence explaining all boundary decisions"
_V6_REASONING_DEFINITION = (
    "reasoning first names the topics the conversation goes through, in order, then states where they change;"
)
_REWRITTEN_REASONING_DEFINITION = (
    "reasoning names the topics the conversation goes through, in order, each as a short label with its message "
    "range, then the message numbers after which the topic changes, and nothing more;"
)
_V6_TOPICS_FIRST = "Topics in order: "
_LABELLED_TOPICS = re.compile(r"^[^();]+ \(\d+(?:-\d+)?\)(?:; [^();]+ \(\d+(?:-\d+)?\))*\. ")
_CHANGES_AFTER = re.compile(r"^Changes after (\d+(?:, \d+)*)(?=[,;.])")
_NO_CHANGE = "No change."

_OLD = {"batch": _OLD_BATCH_PROMPT, "step": _OLD_STEP_PROMPT}
_SYSTEM = {"batch": BATCH_BOUNDARY_DETECT_SYSTEM_PROMPT, "step": STEP_BOUNDARY_DETECT_SYSTEM_PROMPT}
_USER = {"batch": BATCH_BOUNDARY_DETECT_USER_PROMPT, "step": STEP_BOUNDARY_DETECT_USER_PROMPT}
_REPLY_MODEL: dict[str, type[StrictOutputModel]] = {"batch": _BatchReply, "step": _StepReply}
_DATA_SLOTS = {"batch": {"count", "messages"}, "step": {"conversation_history", "time_gap_info", "new_messages"}}
_PATHS = ["batch", "step"]

_FENCED_JSON = re.compile(r"```json\n(.*?)\n```", re.DOTALL)
_BARE_JSON = re.compile(r"^\{\n.*?^\}$", re.DOTALL | re.MULTILINE)
_ISO_TIME = re.compile(r"(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}):\d{2}Z")
_SENTENCE_BREAK = re.compile(r"(?<=\.)\s+(?=[A-Z])")
_SLOT = re.compile(r"\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}")
_REPLY = re.compile(r"<reply>\n(.*?)\n</reply>", re.DOTALL)
_EXAMPLE_MESSAGES = re.compile(r'<messages count="(\d+)">\n(.*?)\n</messages>', re.DOTALL)
_BATCH_LINE = re.compile(r"\[\d+\] \[\d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC \([A-Z][a-z]+day\)\] ")
_STEP_LINE = re.compile(r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC \([A-Z][a-z]+day\)\] ")

_BASE_MS = 1_710_064_800_000  # 2024-03-10 10:00 UTC


def _old_json_objects(old: str) -> list[str]:
    """The old prompt's JSON objects: fenced worked examples and skeletons, then the bare skeleton."""
    fenced = _FENCED_JSON.findall(old)
    return fenced + _BARE_JSON.findall(_FENCED_JSON.sub("", old))


def _old_lines(old: str) -> list[str]:
    """The old prompt's non-blank lines outside JSON objects, with the split's prompt-wide changes applied."""
    text = _BARE_JSON.sub("", _FENCED_JSON.sub("", old))
    text = _ISO_TIME.sub(_as_rendered, text).replace("**", "").replace("`", "")
    return [line.strip() for line in text.splitlines() if line.strip()]


def _as_rendered(match: re.Match[str]) -> str:
    """An old ISO 8601 example time in the shape the renderer writes, weekday included."""
    instant = datetime.fromisoformat(match.group(0))
    return format_utc_time(int(instant.timestamp() * 1000), weekday=True)


def _as_rewritten_by_c(reply: dict[str, object]) -> dict[str, object]:
    """An old batch reply as C rewrote it: ``should_wait`` became ``tail_complete`` with the opposite polarity."""
    if "should_wait" not in reply:
        return reply
    rewritten = {key: value for key, value in reply.items() if key != "should_wait"}
    return rewritten | {"tail_complete": not reply["should_wait"]}


def _as_rewritten_by_e(path: str, index: int, reply: dict[str, object]) -> dict[str, object]:
    """An old worked example's reply as E rewrote it: the first batch example splits after the added reply."""
    if path != "batch" or index != 0:
        return reply
    return reply | _REWRITTEN_EXAMPLE_ONE_SPLIT


def _without_rewritten_reasoning(path: str, reply: dict[str, object]) -> dict[str, object]:
    """A reply minus the field v6 rewrote in every batch worked example: ``reasoning`` now lists the topics first."""
    if path != "batch":
        return reply
    return {key: value for key, value in reply.items() if key != "reasoning"}


def _split_points(rest: str) -> list[int]:
    """The message numbers a worked example's reasoning says the topic changes after, from what follows its topics."""
    if rest == _NO_CHANGE:
        return []
    changes = _CHANGES_AFTER.match(rest)
    assert changes, rest
    return [int(number) for number in changes.group(1).split(", ")]


def _parses(block: str) -> bool:
    try:
        json.loads(block)
    except json.JSONDecodeError:
        return False
    return True


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
    return FakeLLMClient(responses=[json.dumps({"reasoning": "r", "boundaries": [], "tail_complete": True})])


def _step_client() -> FakeLLMClient:
    reply = {"reasoning": "r", "should_end": False, "confidence": 0.9, "topic_summary": ""}
    return FakeLLMClient(responses=[json.dumps(reply)])


# ── Every old rule sentence is kept ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("path", _PATHS)
def test_every_kept_old_sentence_is_in_the_system_prompt(path: str) -> None:
    kept = [line for line in _old_lines(_OLD[path]) if line not in _RESTRUCTURED[path]]
    missing = [sentence for line in kept for sentence in _SENTENCE_BREAK.split(line) if sentence not in _SYSTEM[path]]

    assert kept, "the pinned old prompt yielded no lines, so nothing was checked"
    assert not missing


@pytest.mark.parametrize("path", _PATHS)
def test_every_restructured_line_is_a_line_of_the_old_prompt(path: str) -> None:
    """An exclusion that matches no old line would hide nothing today and anything after the next edit."""
    assert set(_RESTRUCTURED[path]) <= set(_old_lines(_OLD[path]))


def test_rewritten_old_lines_are_gone_from_the_batch_system_prompt() -> None:
    rewritten = [line for line, change in _RESTRUCTURED["batch"].items() if change == "C"]

    assert len(rewritten) == 6, "the old should_wait block: its heading, lead sentence and four cases"
    assert not [line for line in rewritten if line in BATCH_BOUNDARY_DETECT_SYSTEM_PROMPT]


def test_dropped_rules_are_gone_from_the_batch_system_prompt() -> None:
    dropped = [line for line, change in _RESTRUCTURED["batch"].items() if change == "D"]

    assert dropped == [_CROSS_DAY_RULE, _TRANSITION_PHRASE_RULE]
    assert "Cross-day" not in BATCH_BOUNDARY_DETECT_SYSTEM_PROMPT
    assert "new day" not in BATCH_BOUNDARY_DETECT_SYSTEM_PROMPT, "no worked example cites a date change as its reason"
    assert "Transition phrases" not in BATCH_BOUNDARY_DETECT_SYSTEM_PROMPT
    assert "by the way" not in BATCH_BOUNDARY_DETECT_SYSTEM_PROMPT


def test_rewritten_first_example_replaces_the_old_one_on_a_single_day() -> None:
    replaced = [line for line, change in _RESTRUCTURED["batch"].items() if change == "E"]
    first = _EXAMPLE_MESSAGES.search(BATCH_BOUNDARY_DETECT_SYSTEM_PROMPT)

    assert len(replaced) == 2, "the old second-day lunch lines"
    assert not [line for line in replaced if line in BATCH_BOUNDARY_DETECT_SYSTEM_PROMPT]
    assert first
    assert first.group(0) == _REWRITTEN_EXAMPLE_ONE
    assert set(re.findall(r"\d{4}-\d{2}-\d{2}", first.group(2))) == {_EXAMPLE_ONE_DAY}


def test_rewritten_tail_complete_rule_is_in_the_batch_system_prompt() -> None:
    assert _REWRITTEN_TAIL_COMPLETE in BATCH_BOUNDARY_DETECT_SYSTEM_PROMPT
    assert _REWRITTEN_REPLY_FIELD in BATCH_BOUNDARY_DETECT_SYSTEM_PROMPT


def test_rewritten_reasoning_definition_replaces_the_old_skeleton_wording() -> None:
    old_skeletons = [block for block in _old_json_objects(_OLD_BATCH_PROMPT) if not _parses(block)]

    assert [_OLD_REASONING_DEFINITION in block for block in old_skeletons] == [True]
    assert _OLD_REASONING_DEFINITION not in BATCH_BOUNDARY_DETECT_SYSTEM_PROMPT
    assert _V6_REASONING_DEFINITION not in BATCH_BOUNDARY_DETECT_SYSTEM_PROMPT
    assert _REWRITTEN_REASONING_DEFINITION in BATCH_BOUNDARY_DETECT_SYSTEM_PROMPT


def test_every_batch_example_reasoning_lists_labelled_topics_then_its_boundaries() -> None:
    old = {json.loads(block)["reasoning"] for block in _old_json_objects(_OLD_BATCH_PROMPT) if _parses(block)}
    replies = [json.loads(block) for block in _REPLY.findall(BATCH_BOUNDARY_DETECT_SYSTEM_PROMPT)]

    assert replies
    for reply in replies:
        reasoning = reply["reasoning"]
        topics = _LABELLED_TOPICS.match(reasoning)
        assert topics, reasoning
        assert not reasoning.startswith(_V6_TOPICS_FIRST)
        assert _split_points(reasoning[topics.end() :]) == reply["boundaries"], reasoning
    assert not old & {reply["reasoning"] for reply in replies}


@pytest.mark.parametrize("path", _PATHS)
def test_old_json_objects_are_the_new_replies_plus_one_skeleton(path: str) -> None:
    old_objects = _old_json_objects(_OLD[path])
    old_replies = [json.loads(block) for block in old_objects if _parses(block)]
    worked_examples = [
        _without_rewritten_reasoning(path, _as_rewritten_by_e(path, index, _as_rewritten_by_c(reply)))
        for index, reply in enumerate(old_replies)
    ]
    new_replies = [_without_rewritten_reasoning(path, json.loads(block)) for block in _REPLY.findall(_SYSTEM[path])]

    assert new_replies[: len(worked_examples)] == worked_examples
    assert len(new_replies) - len(worked_examples) == _ADDED_EXAMPLES[path]
    assert len([block for block in old_objects if not _parses(block)]) == 1, "F5: one unparseable skeleton"


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


@pytest.mark.parametrize("path", _PATHS)
def test_role_sentence_names_exactly_the_user_message_blocks(path: str) -> None:
    role = _SYSTEM[path].split("\n", 1)[0]
    user_blocks = set(re.findall(r"^<([a-z_]+)", _USER[path], re.MULTILINE))

    assert set(re.findall(r"<([a-z_]+)>", role)) == user_blocks


@pytest.mark.parametrize("path", _PATHS)
def test_system_prompt_names_no_placeholder(path: str) -> None:
    assert not _SLOT.findall(_SYSTEM[path])


@pytest.mark.parametrize("path", _PATHS)
def test_user_prompt_carries_only_data_slots(path: str) -> None:
    user = _USER[path]

    assert set(_SLOT.findall(user)) == _DATA_SLOTS[path]
    assert re.sub(r"</?[a-z_]+[^>]*>|\{\{[a-z_]+\}\}", "", user).strip() == ""


@pytest.mark.parametrize("path", _PATHS)
def test_every_reply_shown_validates_against_the_reply_model(path: str) -> None:
    """The worked examples and the skeleton are what the model imitates; each must pass the model requested."""
    shown = [*_REPLY.findall(_SYSTEM[path]), _SYSTEM[path].split("\n</reply_format>")[0].rsplit("\n", 1)[1]]

    for block in shown:
        _REPLY_MODEL[path].model_validate(json.loads(block))
    assert len(shown) == (4 if path == "batch" else 1)


def test_every_batch_example_counts_its_messages() -> None:
    examples = _EXAMPLE_MESSAGES.findall(BATCH_BOUNDARY_DETECT_SYSTEM_PROMPT)

    assert len(examples) == 3
    for count, body in examples:
        lines = body.splitlines()
        assert len(lines) == int(count)
        assert all(line.startswith(f"[{index}] ") and _BATCH_LINE.match(line) for index, line in enumerate(lines, 1))


# ── What the model receives ─────────────────────────────────────────────────────────────────────────


async def test_batch_call_sends_the_rules_then_the_counted_messages() -> None:
    client = _batch_client()

    await adetect_boundaries(_dialogue(3), llm=client)

    system, user = client.calls[0].messages
    lines = str(user.content).splitlines()[1:-1]
    assert (system.role, system.content) == ("system", BATCH_BOUNDARY_DETECT_SYSTEM_PROMPT)
    assert user.role == "user"
    assert str(user.content).startswith(f'<messages count="{len(lines)}">\n')
    assert len(lines) == 3


async def test_batch_count_is_the_window_the_model_sees_after_the_pre_cut() -> None:
    client = _batch_client()

    await adetect_boundaries(_dialogue(10), llm=client, hard_msg_limit=5)

    user = str(client.calls[0].messages[1].content)
    assert user.startswith('<messages count="2">\n[1] ')
    assert user.endswith("[2] [2024-03-10 10:09 UTC (Sunday)] Speaker9: text 9\n</messages>")


async def test_step_call_sends_the_rules_then_three_data_blocks() -> None:
    client = _step_client()

    await adetect_boundary_step(_dialogue(2), [_msg(2, "text 2", minutes=2)], llm=client, time_gap_info="1 minute")

    system, user = client.calls[0].messages
    assert (system.role, system.content) == ("system", STEP_BOUNDARY_DETECT_SYSTEM_PROMPT)
    assert (user.role, user.content) == (
        "user",
        "<conversation_history>\n"
        "[2024-03-10 10:00 UTC (Sunday)] Speaker0: text 0\n"
        "[2024-03-10 10:01 UTC (Sunday)] Speaker1: text 1\n"
        "</conversation_history>\n\n"
        "<time_gap>\n1 minute\n</time_gap>\n\n"
        "<new_messages>\n[2024-03-10 10:02 UTC (Sunday)] Speaker2: text 2\n</new_messages>",
    )


# ── Custom instructions ─────────────────────────────────────────────────────────────────────────────

_CUSTOM = "Treat a switch between {{project}} names as a topic change.\nKeep  it short."
_ENTRY_PATH = {
    "adetect_boundaries": "batch",
    "BoundaryDetector.adetect": "batch",
    "adetect_boundary_step": "step",
    "BoundaryDetector.adetect_step": "step",
}
_ENTRIES = list(_ENTRY_PATH)


async def _sent(entry: str, **kwargs: Any) -> tuple[str, str]:
    """Drive one async entry point over a fixed conversation; return the system and user messages it sent."""
    client = _batch_client() if _ENTRY_PATH[entry] == "batch" else _step_client()
    history, new = _dialogue(2), _msg(2, "text 2", minutes=2)
    if entry == "adetect_boundaries":
        await adetect_boundaries(history, llm=client, **kwargs)
    elif entry == "BoundaryDetector.adetect":
        await BoundaryDetector(llm=client).adetect(history, **kwargs)
    elif entry == "adetect_boundary_step":
        await adetect_boundary_step(history, [new], llm=client, time_gap_info="1 minute", **kwargs)
    else:
        await BoundaryDetector(llm=client).adetect_step(history, new, **kwargs)
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
    assert system.index("</reply_format>") < system.index("<custom_instructions>")
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
    [adetect_boundaries, adetect_boundary_step, BoundaryDetector.adetect, BoundaryDetector.adetect_step],
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
    batch, detector_batch, detector_step = _batch_client(), _batch_client(), _step_client()
    block = f"<custom_instructions>\n{_CUSTOM}\n</custom_instructions>"

    detect_boundaries(_dialogue(2), llm=batch, custom_instructions=_CUSTOM)
    BoundaryDetector(llm=detector_batch).detect(_dialogue(2), custom_instructions=_CUSTOM)
    BoundaryDetector(llm=detector_step).detect_step(
        _dialogue(2), _msg(2, "text 2", minutes=2), custom_instructions=_CUSTOM
    )

    for client in (batch, detector_batch, detector_step):
        assert str(client.calls[0].messages[0].content).endswith(block)
    with pytest.raises(TypeError, match="prompt"):
        # The removed keyword, on purpose: a type checker rejects it too.
        detect_boundaries(_dialogue(2), llm=_batch_client(), prompt="CUSTOM RULES")  # type: ignore[call-arg]


# ── Time shape ──────────────────────────────────────────────────────────────────────────────────────


async def test_rendered_lines_show_minutes_and_the_utc_label() -> None:
    """Seconds are truncated, not rounded: every timestamp here sits at :59 seconds."""
    batch, step = _batch_client(), _step_client()

    await adetect_boundaries(_dialogue(3), llm=batch)
    await adetect_boundary_step(_dialogue(2), [_msg(2, "text 2", minutes=2)], llm=step)

    batch_lines = str(batch.calls[0].messages[1].content).splitlines()[1:-1]
    step_lines = [line for line in str(step.calls[0].messages[1].content).splitlines() if line.startswith("[")]
    assert batch_lines and all(_BATCH_LINE.match(line) for line in batch_lines)
    assert len(step_lines) == 3 and all(_STEP_LINE.match(line) for line in step_lines)
    assert batch_lines[0] == "[1] [2024-03-10 10:00 UTC (Sunday)] Speaker0: text 0"
