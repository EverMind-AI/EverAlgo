"""The six knowledge prompts' system / user split: every old rule line kept as written, data only in the user message.

The single-message prompts the split replaced are pinned verbatim in ``_old_prompts``. Every non-blank line of each,
its JSON skeleton included, is a line of the new system prompt once its Markdown marks are dropped (``**``, inline
backticks) and its doubled skeleton braces are written single (F6: the old ``topic_split`` / ``uncovered_assign``
skeletons reached the model as ``{{``, which is not JSON), or it is listed in ``_RESTRUCTURED`` under the structural
change that removed it: F1 the data moved to the user message, F2 a heading or data label became a tag, F5 a lead-in
or closing format sentence became the ``<reply_format>`` sentence, F9 the code fence around the data was dropped. Lines
rewritten for their content, the C class, are listed in ``_REWRITTEN`` with the line that replaced them.
"""

from __future__ import annotations

import inspect
import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest

from everalgo.knowledge import aclassify_category
from everalgo.knowledge._batch_merge import amerge_doc_summary, amerge_topics
from everalgo.knowledge._postprocess import aassign_uncovered_blocks, apostprocess_topics, asplit_unsplit_leaves
from everalgo.knowledge._replies import (
    CategoryReply,
    ContentMergeReply,
    TopicMergeReply,
    TopicSplitReply,
    TopicTreeReply,
    UncoveredAssignReply,
)
from everalgo.knowledge.extractor import _aextract_batch
from everalgo.knowledge.prompts.en import (
    category_classify,
    content_merge,
    topic_merge,
    topic_split,
    topic_tree_extract,
    uncovered_assign,
)
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import CategorySpec

from . import _old_prompts
from ._scripted import assign_reply, content_merge_reply, topic_tree_reply

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from everalgo.llm import StrictOutputModel

_REPLY_SENTENCE = "Reply with one JSON object, nothing else."
_RETURN_ONLY = "Return only the JSON object, no additional text or formatting."
_STRUCTURAL_CHANGES = {"F1", "F2", "F5", "F9"}


@dataclass(frozen=True)
class _Split:
    """One prompt's old text, its new pair, and the registry of every old line the new system does not repeat."""

    old: str
    system: str
    user: str
    blocks: list[str]
    data_slots: set[str]
    reply_model: type[StrictOutputModel]
    old_reply_lead_in: str
    restructured: dict[str, str]
    rewritten: dict[str, str]
    added: set[str] = field(default_factory=set)


_SPLITS: dict[str, _Split] = {
    "topic_tree_extract": _Split(
        old=_old_prompts.OLD_TOPIC_TREE_EXTRACT,
        system=topic_tree_extract.TOPIC_TREE_EXTRACT_SYSTEM_PROMPT,
        user=topic_tree_extract.TOPIC_TREE_EXTRACT_USER_PROMPT,
        blocks=[
            "topic_segmentation_requirements",
            "overall_analysis_requirements",
            "content_labels",
            "critical_language_rule",
            "reply_format",
        ],
        data_slots={"title", "count", "max_id", "numbered_paragraphs"},
        reply_model=TopicTreeReply,
        old_reply_lead_in="Return a JSON object with the following structure:",
        restructured={
            "Document Title: {{title}}": "F1",
            "Total paragraphs: {{num_paragraphs}} (IDs 0 to {{max_id}})": "F1",
            "{{numbered_paragraphs}}": "F1",
            "Paragraphs (format: ID: content):": "F2",
            "Topic Segmentation Requirements:": "F2",
            "Overall Analysis Requirements:": "F2",
            "CRITICAL LANGUAGE RULE:": "F2",
            "Return a JSON object with the following structure:": "F5",
            _RETURN_ONLY: "F5",
        },
        rewritten={
            "- children key is optional — omit or use [] for leaf topics.": (
                "- children key is required — use [] for leaf topics."
            ),
            (
                "- Every paragraph ID (0 to {{max_id}}) MUST appear in at least one topic's block_refs. Document metadata "
                '(title, version, date, author) goes in the most relevant topic. Only pure separators (e.g., "---") may be '
                "omitted."
            ): (
                "- Every paragraph ID (0 to the max_id of <paragraphs>) MUST appear in at least one topic's block_refs. "
                "Document metadata (title, version, date, author) goes in the most relevant topic. Only pure separators "
                '(e.g., "---") may be omitted.'
            ),
        },
    ),
    "content_merge": _Split(
        old=_old_prompts.OLD_CONTENT_MERGE,
        system=content_merge.CONTENT_MERGE_SYSTEM_PROMPT,
        user=content_merge.CONTENT_MERGE_USER_PROMPT,
        blocks=["requirements", "critical_language_rule", "reply_format"],
        data_slots={"title", "count", "partial_results"},
        reply_model=ContentMergeReply,
        old_reply_lead_in="Please create a comprehensive analysis that merges all the partial results into:",
        restructured={
            "Document Title: {{title}}": "F1",
            "{{partial_results}}": "F1",
            "Partial Analysis Results:": "F2",
            "Requirements:": "F2",
            "Please create a comprehensive analysis that merges all the partial results into:": "F5",
            _RETURN_ONLY: "F5",
            "```": "F9",
        },
        rewritten={
            (
                'CRITICAL LANGUAGE RULE: First identify the language in the "language" field. ALL outputs (summary, '
                "subject, keywords) MUST be in that language. Do NOT translate."
            ): (
                'First identify the language in the "language" field. ALL outputs (summary, subject, keywords) MUST be in '
                "that language. Do NOT translate."
            ),
        },
    ),
    "topic_merge": _Split(
        old=_old_prompts.OLD_TOPIC_MERGE,
        system=topic_merge.TOPIC_MERGE_SYSTEM_PROMPT,
        user=topic_merge.TOPIC_MERGE_USER_PROMPT,
        blocks=["instructions", "reply_format"],
        data_slots={"title", "all_topics_json"},
        reply_model=TopicMergeReply,
        old_reply_lead_in="Return a JSON object:",
        restructured={
            "Document Title: {{title}}": "F1",
            "{{all_topics_json}}": "F1",
            "All topics (each with an index number and optional parent_topic for context):": "F2",
            "Instructions:": "F2",
            "Return a JSON object:": "F5",
            _RETURN_ONLY: "F5",
            "```": "F9",
        },
        rewritten={},
    ),
    "category_classify": _Split(
        old=_old_prompts.OLD_CATEGORY_CLASSIFY,
        system=category_classify.CATEGORY_CLASSIFY_SYSTEM_PROMPT,
        user=category_classify.CATEGORY_CLASSIFY_USER_PROMPT,
        blocks=["task", "reply_format"],
        data_slots={"title", "doc_summary"},
        reply_model=CategoryReply,
        old_reply_lead_in="Output strictly this JSON shape and nothing else:",
        restructured={
            "{{taxonomy}}": "F1",
            "Title: {{title}}": "F1",
            "Summary: {{doc_summary}}": "F1",
            "# Taxonomy": "F2",
            "# Document": "F2",
            "# Task": "F2",
            "Output strictly this JSON shape and nothing else:": "F5",
        },
        rewritten={
            "matching category from the taxonomy above. If none of the categories fit, return the empty": (
                "matching category from <taxonomy>. If none of the categories fit, return the empty"
            ),
        },
    ),
    "topic_split": _Split(
        old=_old_prompts.OLD_TOPIC_SPLIT,
        system=topic_split.TOPIC_SPLIT_SYSTEM_PROMPT,
        user=topic_split.TOPIC_SPLIT_USER_PROMPT,
        blocks=["instructions", "reply_format"],
        data_slots={"title", "count", "topics_section"},
        reply_model=TopicSplitReply,
        old_reply_lead_in="Return a JSON object:",
        restructured={
            "Document Title: {{title}}": "F1",
            "{{topics_section}}": "F1",
            "Instructions:": "F2",
            "Return a JSON object:": "F5",
            _RETURN_ONLY: "F5",
        },
        rewritten={
            '"reason": "Brief explanation"': '"reason": "Brief explanation",',
            "For topics with should_split=false, only original_topic, should_split, and reason are needed.": (
                "For topics with should_split=false, only original_topic, should_split, and reason carry content; write "
                "block_refs and children as null."
            ),
        },
        added={'"block_refs": null,', '"children": null'},
    ),
    "uncovered_assign": _Split(
        old=_old_prompts.OLD_UNCOVERED_ASSIGN,
        system=uncovered_assign.UNCOVERED_ASSIGN_SYSTEM_PROMPT,
        user=uncovered_assign.UNCOVERED_ASSIGN_USER_PROMPT,
        blocks=["instructions", "reply_format"],
        data_slots={"title", "topic_count", "topics_text", "block_count", "uncovered_text"},
        reply_model=UncoveredAssignReply,
        old_reply_lead_in="Return a JSON object:",
        restructured={
            "Document Title: {{title}}": "F1",
            "{{topics_text}}": "F1",
            "{{uncovered_text}}": "F1",
            "Existing topics:": "F2",
            "Uncovered blocks:": "F2",
            "Instructions:": "F2",
            "Return a JSON object:": "F5",
            _RETURN_ONLY: "F5",
        },
        rewritten={
            "- Use the topic index (0-based) from the list above.": "- Use the topic index (0-based) from <existing_topics>.",
        },
    ),
}
_NAMES = list(_SPLITS)

_SLOT = re.compile(r"\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}")
_TAG_LINE = re.compile(r"^</?[a-z]+(?:_[a-z]+)*(?: [a-z_]+=\"[^\"]*\")*>$")
_SNAKE_TAG = re.compile(r"[a-z]+(?:_[a-z]+)*")


def _single_braces(text: str) -> str:
    """Write a skeleton brace pair that was doubled (``{{`` not opening a slot, ``}}`` not closing one) single."""
    return re.sub(r"(?<![A-Za-z0-9_])\}\}", "}", re.sub(r"\{\{(?![A-Za-z_])", "{", text))


def _normalized(line: str) -> str:
    """A stripped old line without ``**`` or inline backticks, its skeleton braces single; a code fence stays as is."""
    text = line.strip()
    if text.startswith("```"):
        return text
    return _single_braces(text.replace("**", "").replace("`", ""))


def _old_lines(split: _Split) -> list[str]:
    return [_normalized(line) for line in split.old.splitlines() if line.strip()]


def _new_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _block(tag: str, text: str) -> str:
    return text.split(f"<{tag}>\n", 1)[1].split(f"\n</{tag}>", 1)[0]


def _top_level_blocks(text: str) -> list[str]:
    return re.findall(r"(?:\A|\n\n)<([a-z_]+)(?: [^>]*)?>\n", text)


def _decode_first_object(text: str) -> tuple[Any, str]:
    """The first JSON object in ``text`` and whatever follows it."""
    start = text.index("{")
    value, end = json.JSONDecoder().raw_decode(text, start)
    return value, text[end:]


# ── Every old rule line is kept ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", _NAMES)
def test_every_kept_old_line_is_a_line_of_the_system_prompt(name: str) -> None:
    split = _SPLITS[name]
    kept = [line for line in _old_lines(split) if line not in split.restructured and line not in split.rewritten]
    missing = [line for line in kept if line not in set(_new_lines(split.system))]

    assert len(kept) >= 4, "the pinned old prompt yielded too few lines, so little was checked"
    assert not missing


@pytest.mark.parametrize("name", _NAMES)
def test_every_registered_line_is_a_line_of_the_old_prompt(name: str) -> None:
    """An exclusion that matches no old line would hide nothing today and anything after the next edit."""
    split = _SPLITS[name]

    assert set(split.restructured) <= set(_old_lines(split))
    assert set(split.rewritten) <= set(_old_lines(split))
    assert not set(split.restructured) & set(split.rewritten)


@pytest.mark.parametrize("name", _NAMES)
def test_every_removal_is_structural(name: str) -> None:
    assert set(_SPLITS[name].restructured.values()) <= _STRUCTURAL_CHANGES


@pytest.mark.parametrize("name", _NAMES)
def test_every_rewritten_line_is_replaced_as_registered(name: str) -> None:
    split = _SPLITS[name]
    system_lines = set(_new_lines(split.system))

    for old, new in split.rewritten.items():
        assert old not in system_lines
        assert new in system_lines


@pytest.mark.parametrize("name", _NAMES)
def test_the_system_prompt_adds_no_wording_of_its_own(name: str) -> None:
    """Besides tags, each new line is an old one, the reply sentence, a C replacement, or a registered addition."""
    split = _SPLITS[name]
    allowed = set(_old_lines(split)) | {_REPLY_SENTENCE} | set(split.rewritten.values()) | split.added
    added = [line for line in _new_lines(split.system) if not _TAG_LINE.match(line) and line not in allowed]

    assert added == []


# ── Structure: rules in system, data in user ────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", _NAMES)
def test_system_blocks_follow_the_documented_order(name: str) -> None:
    split = _SPLITS[name]
    assert _top_level_blocks(split.system) == split.blocks


@pytest.mark.parametrize("name", _NAMES)
def test_system_prompt_names_no_data(name: str) -> None:
    split = _SPLITS[name]
    assert _SLOT.findall(split.system) == []
    assert "{{" not in split.system


@pytest.mark.parametrize("name", _NAMES)
def test_user_prompt_is_only_tagged_data_slots(name: str) -> None:
    """Every user line is a tag or one bare slot; the slots are exactly the data the old prompt interpolated."""
    split = _SPLITS[name]
    body = [line for line in _new_lines(split.user) if not _TAG_LINE.match(_SLOT.sub("x", line))]

    assert all(_SLOT.fullmatch(line) for line in body)
    assert set(_SLOT.findall(split.user)) == split.data_slots
    assert all(_SNAKE_TAG.fullmatch(tag) for tag in _top_level_blocks(split.user))


# ── The reply skeleton ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", _NAMES)
def test_reply_format_is_one_sentence_and_a_skeleton_the_reply_model_accepts(name: str) -> None:
    split = _SPLITS[name]
    sentence, rest = _block("reply_format", split.system).split("\n", 1)
    skeleton, _ = _decode_first_object(rest)

    assert sentence == _REPLY_SENTENCE
    split.reply_model.model_validate_json(json.dumps(skeleton))


@pytest.mark.parametrize("name", _NAMES)
def test_old_skeleton_is_the_new_one(name: str) -> None:
    """Compared as JSON; the split skeleton's should_split=false result gains its two null fields."""
    split = _SPLITS[name]
    old_text = _single_braces(split.old)
    old_skeleton, _ = _decode_first_object(old_text[old_text.index(split.old_reply_lead_in) :])
    new_skeleton, _ = _decode_first_object(_block("reply_format", split.system))

    if name == "topic_split":
        kept_result = new_skeleton["results"][1]
        assert (kept_result.pop("block_refs"), kept_result.pop("children")) == (None, None)
    assert new_skeleton == old_skeleton


def test_only_the_split_skeleton_is_followed_by_a_rule() -> None:
    trailing = {name: _decode_first_object(_block("reply_format", s.system))[1].strip() for name, s in _SPLITS.items()}

    assert (
        trailing.pop("topic_split")
        == _SPLITS["topic_split"].rewritten[
            "For topics with should_split=false, only original_topic, should_split, and reason are needed."
        ]
    )
    assert set(trailing.values()) == {""}


# ── The call sites send the pair ────────────────────────────────────────────────────────────────────

_SENTINEL = "SENTINEL-TITLE-5c1e"


async def _extract(client: FakeLLMClient, custom_instructions: str | None = None) -> None:
    await _aextract_batch(client, _SENTINEL, [(0, "first"), (1, "second")], custom_instructions)


async def _content_merge(client: FakeLLMClient, custom_instructions: str | None = None) -> None:
    await amerge_doc_summary(
        client, _SENTINEL, [{"summary": "s1"}, {"summary": "s2"}], custom_instructions=custom_instructions
    )


async def _topic_merge(client: FakeLLMClient, custom_instructions: str | None = None) -> None:
    topics = [{"topic": f"T{i}", "summary": "s", "block_refs": str(i)} for i in range(4)]
    await amerge_topics(client, _SENTINEL, topics, custom_instructions=custom_instructions)


async def _classify(client: FakeLLMClient, custom_instructions: str | None = None) -> None:
    categories = [CategorySpec(id="a", description="A.")]
    await aclassify_category(client, _SENTINEL, "summary", categories, custom_instructions=custom_instructions)


async def _split(client: FakeLLMClient, custom_instructions: str | None = None) -> None:
    unsplit = [{"topic_name": "Big", "path": [0], "block_ids": [0, 1], "heading_ids": [0, 1], "total_chars": 1}]
    topics: list[dict[str, Any]] = [{"topic": "Big", "block_refs": "0-1"}]
    await asplit_unsplit_leaves(
        client, _SENTINEL, topics, unsplit, {0: "a", 1: "b"}, custom_instructions=custom_instructions
    )


async def _assign(client: FakeLLMClient, custom_instructions: str | None = None) -> None:
    topics: list[dict[str, Any]] = [{"topic": "T", "block_refs": "0"}]
    uncovered = [{"block_id": 1, "text": "o"}]
    await aassign_uncovered_blocks(
        client, _SENTINEL, topics, uncovered, {0: "a", 1: "o"}, custom_instructions=custom_instructions
    )


_CALL_SITES: dict[str, tuple[Callable[..., Awaitable[None]], str]] = {
    "topic_tree_extract": (_extract, topic_tree_reply("s")),
    "content_merge": (_content_merge, content_merge_reply("merged")),
    "topic_merge": (_topic_merge, '{"merges": []}'),
    "category_classify": (_classify, '{"category_id": "a"}'),
    "topic_split": (_split, '{"results": []}'),
    "uncovered_assign": (_assign, assign_reply()),
}


@pytest.mark.parametrize("name", _NAMES)
async def test_call_site_sends_the_rules_as_system_and_the_data_as_user(name: str) -> None:
    drive, reply = _CALL_SITES[name]
    client = FakeLLMClient(responses=[reply])

    await drive(client)

    (call,) = client.calls
    system, user = call.messages
    assert (system.role, user.role) == ("system", "user")
    assert isinstance(system.content, str)
    assert isinstance(user.content, str)
    assert system.content.startswith(_SPLITS[name].system)
    assert _SENTINEL not in system.content
    assert _block("document_title", user.content) == _SENTINEL
    assert call.response_format is _SPLITS[name].reply_model


@pytest.mark.parametrize("name", [n for n in _NAMES if n != "category_classify"])
async def test_call_site_system_is_exactly_the_static_rules(name: str) -> None:
    """Only classification appends to the rules (its taxonomy); every other system message is the constant itself."""
    drive, reply = _CALL_SITES[name]
    client = FakeLLMClient(responses=[reply])

    await drive(client)

    assert client.calls[0].messages[0].content == _SPLITS[name].system


# ── custom_instructions: appended to the rules, never a template, never an override ───────────────


async def _sent(name: str, custom_instructions: str | None = None) -> tuple[str, str]:
    drive, reply = _CALL_SITES[name]
    client = FakeLLMClient(responses=[reply])
    await drive(client, custom_instructions)
    system, user = client.calls[0].messages
    return str(system.content), str(user.content)


@pytest.mark.parametrize("name", _NAMES)
async def test_call_site_custom_instructions_close_the_system_message(name: str) -> None:
    """The block follows everything the default system carries (the taxonomy too) and is the message's last."""
    text = 'Use the glossary terms.\nA JSON sample {"a": 1} stays.'
    default_system, default_user = await _sent(name)

    system, user = await _sent(name, text)

    assert system == f"{default_system}\n\n<custom_instructions>\n{text}\n</custom_instructions>"
    assert _top_level_blocks(system)[-1] == "custom_instructions"
    assert user == default_user


@pytest.mark.parametrize("name", _NAMES)
@pytest.mark.parametrize("blank", ["", "  \n\t "])
async def test_call_site_blank_custom_instructions_are_the_same_as_none(name: str, blank: str) -> None:
    assert await _sent(name, blank) == await _sent(name)


@pytest.mark.parametrize("name", _NAMES)
@pytest.mark.parametrize("slot", ["{{title}}", "{{taxonomy}}", "{x}"])
async def test_call_site_custom_instructions_are_never_rendered_as_a_template(name: str, slot: str) -> None:
    """A slot in the caller's text is sent as written: neither substituted nor rejected."""
    system, _ = await _sent(name, f"Keep {slot} literal.")

    assert system.endswith(f"<custom_instructions>\nKeep {slot} literal.\n</custom_instructions>")


@pytest.mark.parametrize(
    "function",
    [amerge_doc_summary, amerge_topics, aclassify_category, asplit_unsplit_leaves, aassign_uncovered_blocks],
    ids=lambda f: f.__name__,
)
def test_call_site_takes_custom_instructions_and_no_prompt(function: Callable[..., Any]) -> None:
    parameters = inspect.signature(function).parameters

    assert parameters["custom_instructions"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["custom_instructions"].default is None
    assert not [name for name in parameters if name.endswith("prompt")]


def test_postprocess_takes_one_custom_instructions_for_both_passes() -> None:
    parameters = inspect.signature(apostprocess_topics).parameters

    assert parameters["custom_instructions"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["custom_instructions"].default is None
    assert "split_prompt" not in parameters
    assert "assign_prompt" not in parameters
