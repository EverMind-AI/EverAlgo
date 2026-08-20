"""Tests for everalgo.user_memory.profile — ProfileExtractor.

Uses a single LLM call against ``PROFILE_INITIAL_EXTRACTION_PROMPT`` returning
``{explicit_info, implicit_traits}``. No internal retry — exceptions propagate directly to the caller.

UPDATE-mode tests cover:
- update no compact: ops applied, LLM called once, result has merged items.
- update triggers compact: post-merge item count > threshold, second LLM call runs.
- update preserves new_explicit_info merge correctness: add + update + delete ops each applied correctly.
"""

from __future__ import annotations

import json
import logging
from typing import Any, cast

import pytest

from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.llm.types import ChatResponse
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import ChatMessage, MemCell, Profile, ToolCall, ToolCallFunction, ToolCallRequest, ToolCallResult
from everalgo.user_memory import OutputLanguage
from everalgo.user_memory.profile import (
    _PROFILE_COMPACT_THRESHOLD,
    ProfileExtractor,
    _apply_ops,
    _build_summary,
    _render_conversation,
)

_TS = 1700000010000
_TS_OLD_1 = 1690000000000
_TS_OLD_2 = 1695000000000

_msg_id = 0


def _msg(
    text: str,
    role: str = "user",
    ts: int = _TS,
    sender: str = "u_alice",
    sender_name: str | None = "Alice",
    msg_id: str | None = None,
) -> ChatMessage:
    global _msg_id
    _msg_id += 1
    return ChatMessage(
        id=msg_id if msg_id is not None else f"m{_msg_id}",
        role=role,  # type: ignore[arg-type]
        content=text,
        timestamp=ts,
        sender_id=sender,
        sender_name=sender_name,
    )


def _memcell(ts: int = _TS, content: str = "Default content") -> MemCell:
    return MemCell(
        items=[_msg(content, ts=ts)],
        timestamp=ts,
    )


def _cluster() -> list[MemCell]:
    return [
        MemCell(
            items=[_msg("User asked about Python async patterns.", ts=_TS_OLD_1)],
            timestamp=_TS_OLD_1,
        ),
        MemCell(
            items=[_msg("User mentioned they prefer ruff over black.", ts=_TS_OLD_2)],
            timestamp=_TS_OLD_2,
        ),
    ]


def _payload(explicit_info: list[dict[str, Any]], implicit_traits: list[dict[str, Any]]) -> str:
    return json.dumps({"explicit_info": explicit_info, "implicit_traits": implicit_traits})


def _implicit_trait(
    trait: str = "[Test]", description: str = "Test trait", basis: str = "test basis", evidence: str = "test evidence"
) -> dict[str, Any]:
    """Helper to create a complete implicit_traits item with all required fields."""
    return {"trait": trait, "description": description, "basis": basis, "evidence": evidence}


async def test_aextract_builds_profile_from_explicit_info() -> None:
    """``summary`` derives from first explicit_info description; lists preserved as extras."""
    payload = _payload(
        explicit_info=[
            {
                "category": "Technical Skills",
                "description": "Alice is a Python developer focusing on async patterns.",
                "evidence": "Alice asked about async patterns.",
            },
        ],
        implicit_traits=[
            _implicit_trait(
                trait="[Pragmatic]",
                description="Prefers tooling that minimises ceremony.",
                basis="Repeated preference for ruff over black.",
                evidence="Alice mentioned ruff preference.",
            ),
        ],
    )
    fake = FakeLLMClient(responses=[ChatResponse(content=payload, model="fake")])

    profile = await ProfileExtractor(llm=fake).aextract([*_cluster(), _memcell()], sender_id="u_alice")

    assert profile.owner_id == "u_alice"
    assert "Python developer" in profile.summary
    # Extras preserved via Profile.model_config extra="allow"
    assert profile.explicit_info[0]["category"] == "Technical Skills"  # type: ignore[attr-defined]
    assert profile.implicit_traits[0]["trait"] == "[Pragmatic]"  # type: ignore[attr-defined]


async def test_aextract_summary_falls_back_to_implicit_trait_when_explicit_empty() -> None:
    payload = _payload(
        explicit_info=[],
        implicit_traits=[
            _implicit_trait(trait="[Pragmatic]", description="Prefers minimal-ceremony tooling."),
        ],
    )
    fake = FakeLLMClient(responses=[ChatResponse(content=payload, model="fake")])

    profile = await ProfileExtractor(llm=fake).aextract([_memcell()], sender_id="u_alice")

    assert profile.summary == "Prefers minimal-ceremony tooling."


async def test_aextract_raises_on_payload_missing_required_keys() -> None:
    """Payload without both explicit_info and implicit_traits → ValueError on first attempt (no internal retry)."""
    bad_responses: list[str | ChatResponse] = [ChatResponse(content="{}", model="fake")]
    fake = FakeLLMClient(responses=bad_responses)

    with pytest.raises(ValueError):
        await ProfileExtractor(llm=fake).aextract([_memcell()], sender_id="u_alice")

    assert fake.call_count == 1


async def test_aextract_raises_on_bad_json() -> None:
    """Unparseable JSON → ValueError on first attempt (no internal retry)."""
    bad_responses: list[str | ChatResponse] = [ChatResponse(content="not json", model="fake")]
    fake = FakeLLMClient(responses=bad_responses)

    with pytest.raises(ValueError):
        await ProfileExtractor(llm=fake).aextract([_memcell()], sender_id="u_alice")

    assert fake.call_count == 1


async def test_aextract_owner_id_equals_sender_id() -> None:
    """``Profile.owner_id`` must equal the ``sender_id`` argument."""
    payload = _payload(
        explicit_info=[{"category": "x", "description": "y", "evidence": "z"}],
        implicit_traits=[],
    )
    fake = FakeLLMClient(responses=[ChatResponse(content=payload, model="fake")])
    cell = MemCell(items=[_msg("Default content", sender="u_custom")], timestamp=_TS)

    profile = await ProfileExtractor(llm=fake).aextract([cell], sender_id="u_custom")

    assert profile.owner_id == "u_custom"


async def test_aextract_renders_cluster_into_conversation_text() -> None:
    """Cluster ChatMemCells are stitched into the {conversation_text} placeholder."""
    captured: dict[str, str] = {}
    payload = _payload(
        explicit_info=[{"category": "x", "description": "y", "evidence": "z"}],
        implicit_traits=[],
    )

    def handler(messages: list[LLMChatMessage], **kwargs: Any) -> ChatResponse:
        assert isinstance(messages[0].content, str)  # narrow for test
        captured["prompt"] = messages[0].content
        return ChatResponse(content=payload, model="fake")

    fake = FakeLLMClient(handler=handler)

    await ProfileExtractor(llm=fake).aextract([*_cluster(), _memcell()], sender_id="u_alice")

    assert "Python async patterns" in captured["prompt"]


async def test_aextract_per_call_prompt_overrides_default() -> None:
    captured: dict[str, str] = {}
    payload = _payload(
        explicit_info=[{"category": "x", "description": "y", "evidence": "z"}],
        implicit_traits=[],
    )

    def handler(messages: list[LLMChatMessage], **kwargs: Any) -> ChatResponse:
        assert isinstance(messages[0].content, str)  # narrow for test
        captured["prompt"] = messages[0].content
        return ChatResponse(content=payload, model="fake")

    fake = FakeLLMClient(handler=handler)
    custom = "CUSTOM PROFILE conv={conversation_text}"

    await ProfileExtractor(llm=fake).aextract([_memcell()], sender_id="u_alice", prompt=custom)

    assert captured["prompt"].startswith("CUSTOM PROFILE conv=")
    assert "Default content" in captured["prompt"]


# ==========================================================================
# _render_conversation helper
# ==========================================================================


def test_render_conversation_skips_empty_content() -> None:
    """Messages with empty content are silently dropped."""
    cell = MemCell(
        items=[
            ChatMessage(
                id="m1", role="user", content="hello", timestamp=1700000000000, sender_id="u_alice", sender_name="Alice"
            ),
            ChatMessage(
                id="m2", role="user", content="", timestamp=1700000001000, sender_id="u_bob", sender_name="Bob"
            ),
        ],
        timestamp=1700000001000,
    )
    rendered = _render_conversation([cell])
    assert "Alice" in rendered
    assert "Bob" not in rendered


def test_render_conversation_uses_sentinel_when_all_inputs_empty() -> None:
    """Empty cluster + empty messages → sentinel line."""
    empty_cell = MemCell(
        items=[],
        timestamp=1700000000000,
    )
    rendered = _render_conversation([empty_cell])
    assert rendered == "(no prior MemCells in the cluster)"


# ==========================================================================
# _build_summary defensive branches
# ==========================================================================


def test_build_summary_skips_non_dict_items_in_explicit_info() -> None:
    """Non-dict entries inside explicit_info are skipped."""
    summary = _build_summary(
        explicit_info=["not a dict", 42, {"description": "real one"}],
        implicit_traits=[],
    )
    assert summary == "real one"


def test_build_summary_skips_non_dict_items_in_implicit_traits() -> None:
    """Non-dict entries inside implicit_traits are skipped."""
    summary = _build_summary(
        explicit_info=[],
        implicit_traits=["not a dict", {"description": "trait desc"}],
    )
    assert summary == "trait desc"


def test_build_summary_returns_sentinel_when_no_usable_description() -> None:
    """Empty / whitespace-only / non-string descriptions → sentinel."""
    summary = _build_summary(
        explicit_info=[{"description": ""}, {"description": "   "}, {"description": 42}],
        implicit_traits=[{"description": None}],
    )
    assert summary == "(no summary)"


# ==========================================================================
# Empty-memcells guard
# ==========================================================================


async def test_aextract_raises_on_empty_memcells() -> None:
    """Passing an empty sequence raises ValueError before any LLM call."""
    fake = FakeLLMClient(handler=lambda *_a, **_kw: ChatResponse(content="{}", model="fake"))
    with pytest.raises(ValueError, match="at least one"):
        await ProfileExtractor(llm=fake).aextract([], sender_id="u_alice")


# ==========================================================================
# Silent-skip contract — agent → user-memory pipeline
# ==========================================================================


async def test_aextract_silently_skips_non_chat_items() -> None:
    """ProfileExtractor must silently skip ToolCallRequest / ToolCallResult items.

    Locks the agent → user-memory pipeline contract: a sequence of MemCells with mixed items
    must render only the ChatMessage text into the profile prompt.
    """
    payload = json.dumps(
        {
            "explicit_info": [{"category": "Skills", "description": "Alice writes Python.", "evidence": "x"}],
            "implicit_traits": [],
        }
    )
    captured: dict[str, str] = {}

    def handler(messages: list[LLMChatMessage], **kwargs: Any) -> ChatResponse:
        assert isinstance(messages[0].content, str)  # narrow for test
        captured["prompt"] = messages[0].content
        return ChatResponse(content=payload, model="fake")

    fake = FakeLLMClient(handler=handler)

    mixed_cell = MemCell(
        items=[
            ChatMessage(
                id="c1",
                role="user",
                content="I write Python for data pipelines.",
                timestamp=1700000000000,
                sender_id="u_alice",
                sender_name="Alice",
            ),
            ToolCallRequest(
                tool_calls=[
                    ToolCall(id="tc1", function=ToolCallFunction(name="search.docs", arguments='{"q": "python"}'))
                ],
                timestamp=1700000001000,
                sender_id="assistant",
            ),
            ToolCallResult(
                tool_call_id="tc1",
                content="Python docs returned.",
                timestamp=1700000002000,
            ),
        ],
        timestamp=1700000002000,
    )

    profile = await ProfileExtractor(llm=fake).aextract([mixed_cell], sender_id="u_alice")

    assert profile.owner_id == "u_alice"
    # ChatMessage content must be present in the rendered prompt
    assert "I write Python for data pipelines" in captured["prompt"]
    # Tool call content must NOT appear — ToolCallRequest / ToolCallResult were silently skipped
    assert "search.docs" not in captured["prompt"]
    assert "Python docs returned" not in captured["prompt"]


# ==========================================================================
# UPDATE mode
# ==========================================================================


def _old_profile(
    explicit_info: list[dict[str, Any]] | None = None,
    implicit_traits: list[dict[str, Any]] | None = None,
) -> Profile:
    ei: list[dict[str, Any]] = explicit_info or [
        {"category": "Skills", "description": "Alice writes Python.", "evidence": "x"}
    ]
    it: list[dict[str, Any]] = implicit_traits or [
        _implicit_trait(trait="[Pragmatic]", description="Minimal ceremony.", basis="test basis", evidence="y")
    ]
    return Profile.model_validate(
        {
            "owner_id": "u_alice",
            "summary": "Alice writes Python.",
            "timestamp": _TS_OLD_1,
            "explicit_info": ei,
            "implicit_traits": it,
        }
    )


async def test_update_no_compact_applies_ops_and_returns_merged_profile() -> None:
    """UPDATE mode: ops applied, single LLM call, item count stays below compact threshold."""
    ops_json = json.dumps(
        {
            "operations": [
                {
                    "action": "add",
                    "type": "explicit_info",
                    "data": {"category": "Location", "description": "Lives in Berlin.", "evidence": "z"},
                },
            ],
            "update_note": "added 1",
        }
    )
    fake = FakeLLMClient(responses=[ChatResponse(content=ops_json, model="fake")])

    profile = await ProfileExtractor(llm=fake).aextract(
        [_memcell()],
        sender_id="u_alice",
        old_profile=_old_profile(),
    )

    assert fake.call_count == 1
    assert profile.owner_id == "u_alice"
    ei = cast("list[dict[str, Any]]", profile.explicit_info)  # type: ignore[attr-defined]
    assert len(ei) == 2
    assert ei[1]["category"] == "Location"
    assert ei[1]["description"] == "Lives in Berlin."


async def test_update_triggers_compact_when_item_count_exceeds_threshold() -> None:
    """UPDATE mode: post-merge item count > _PROFILE_COMPACT_THRESHOLD triggers second LLM compact call."""
    # Build an old profile with enough items to push total over the threshold after one add.
    n = _PROFILE_COMPACT_THRESHOLD  # one add brings total to threshold + 1
    ei_items = [{"category": f"Cat{i}", "description": f"Desc{i}.", "evidence": "x"} for i in range(n)]
    old_p = _old_profile(explicit_info=ei_items, implicit_traits=[])

    ops_json = json.dumps(
        {
            "operations": [
                {
                    "action": "add",
                    "type": "explicit_info",
                    "data": {"category": "New", "description": "One more.", "evidence": "z"},
                }
            ],
            "update_note": "added 1",
        }
    )
    compact_json = json.dumps(
        {
            "explicit_info": [{"category": "Merged", "description": "Compacted profile.", "evidence": "all"}],
            "implicit_traits": [],
            "compact_note": "merged many into one",
        }
    )
    fake = FakeLLMClient(
        responses=[
            ChatResponse(content=ops_json, model="fake"),
            ChatResponse(content=compact_json, model="fake"),
        ]
    )

    profile = await ProfileExtractor(llm=fake).aextract(
        [_memcell()],
        sender_id="u_alice",
        old_profile=old_p,
    )

    assert fake.call_count == 2  # update call + compact call
    ei = cast("list[dict[str, Any]]", profile.explicit_info)  # type: ignore[attr-defined]
    assert len(ei) == 1
    assert ei[0]["category"] == "Merged"


async def test_update_merge_correctness_add_update_delete() -> None:
    """UPDATE mode: add / update / delete ops each produce the correct merged result."""
    old_p = _old_profile(
        explicit_info=[
            {"category": "Skills", "description": "Writes Python.", "evidence": "x"},
            {"category": "Location", "description": "In Berlin.", "evidence": "y"},
        ],
        implicit_traits=[
            _implicit_trait(trait="[Pragmatic]", description="Minimal ceremony.", basis="test", evidence="z"),
        ],
    )
    ops_json = json.dumps(
        {
            "operations": [
                {
                    "action": "add",
                    "type": "explicit_info",
                    "data": {"category": "Hobby", "description": "Plays chess.", "evidence": "new"},
                },
                {
                    "action": "update",
                    "type": "explicit_info",
                    "index": 0,
                    "data": {"description": "Writes Python and Go."},
                },
                {"action": "delete", "type": "implicit_traits", "index": 0},
            ],
            "update_note": "add, update, delete",
        }
    )
    fake = FakeLLMClient(responses=[ChatResponse(content=ops_json, model="fake")])

    profile = await ProfileExtractor(llm=fake).aextract(
        [_memcell()],
        sender_id="u_alice",
        old_profile=old_p,
    )

    ei = cast("list[dict[str, Any]]", profile.explicit_info)  # type: ignore[attr-defined]
    it = cast("list[dict[str, Any]]", profile.implicit_traits)  # type: ignore[attr-defined]
    # add: 3rd explicit_info item added
    assert len(ei) == 3
    assert ei[2]["category"] == "Hobby"
    # update: first item's description overwritten
    assert ei[0]["description"] == "Writes Python and Go."
    # existing field preserved by update merge
    assert ei[0]["category"] == "Skills"
    # Location (index 1) untouched
    assert ei[1]["category"] == "Location"
    # delete: implicit_traits[0] removed
    assert len(it) == 0


async def test_init_injects_target_user_and_preserves_other_speakers() -> None:
    """INIT prompt labels the target by name and locates them by id; other speakers stay as context."""
    captured: dict[str, str] = {}

    def handler(messages: list[LLMChatMessage], **kwargs: Any) -> ChatResponse:
        assert isinstance(messages[0].content, str)  # narrow for test
        captured["prompt"] = messages[0].content
        return ChatResponse(content=_payload([], []), model="fake")

    fake = FakeLLMClient(handler=handler)
    cell = MemCell(
        items=[
            _msg("I write Python.", sender="u_alice", sender_name="Alice"),
            _msg("I prefer Go.", sender="u_bob", sender_name="Bob"),
        ],
        timestamp=_TS,
    )
    await ProfileExtractor(llm=fake).aextract([cell], sender_id="u_alice")

    assert "TARGET USER: Alice" in captured["prompt"]  # name preferred over the raw id
    assert "equals ``u_alice``" in captured["prompt"]  # id stays the locator
    assert "{target_user_id}" not in captured["prompt"]  # placeholder actually substituted
    assert "u_bob" in captured["prompt"]  # other speaker kept as context (no render filtering)


async def test_update_injects_target_user() -> None:
    """UPDATE prompt labels the target by name and locates them by id."""
    captured: dict[str, str] = {}

    def handler(messages: list[LLMChatMessage], **kwargs: Any) -> ChatResponse:
        assert isinstance(messages[0].content, str)  # narrow for test
        captured["prompt"] = messages[0].content
        return ChatResponse(content=json.dumps({"operations": [{"action": "none"}]}), model="fake")

    fake = FakeLLMClient(handler=handler)
    cell = MemCell(
        items=[
            _msg("I moved to Berlin.", sender="u_alice"),
            _msg("I prefer Go.", sender="u_bob", sender_name="Bob"),
        ],
        timestamp=_TS,
    )
    old = Profile.model_validate(
        {
            "owner_id": "u_alice",
            "summary": "Alice writes Python.",
            "timestamp": _TS_OLD_1,
            "explicit_info": [{"category": "Skills", "description": "Python.", "evidence": "x"}],
            "implicit_traits": [],
        }
    )
    await ProfileExtractor(llm=fake).aextract([cell], sender_id="u_alice", old_profile=old)

    assert "TARGET USER: Alice" in captured["prompt"]
    assert "equals ``u_alice``" in captured["prompt"]
    assert "{target_user_id}" not in captured["prompt"]
    assert "u_bob" in captured["prompt"]  # other speaker kept as context (no render filtering)


async def test_init_falls_back_to_sender_id_when_no_name_is_carried() -> None:
    """``sender_name`` is optional, so the id doubles as the label when the conversation carries none."""
    captured: dict[str, str] = {}

    def handler(messages: list[LLMChatMessage], **kwargs: Any) -> ChatResponse:
        assert isinstance(messages[0].content, str)  # narrow for test
        captured["prompt"] = messages[0].content
        return ChatResponse(content=_payload([], []), model="fake")

    fake = FakeLLMClient(handler=handler)
    cell = MemCell(items=[_msg("I write Python.", sender="u_alice", sender_name=None)], timestamp=_TS)

    await ProfileExtractor(llm=fake).aextract([cell], sender_id="u_alice")

    assert "TARGET USER: u_alice" in captured["prompt"]
    assert "equals ``u_alice``" in captured["prompt"]


async def test_display_name_ignores_a_blank_name_and_other_speakers_names() -> None:
    """A whitespace-only name is no name, and another speaker's name is never the target's label."""
    captured: dict[str, str] = {}

    def handler(messages: list[LLMChatMessage], **kwargs: Any) -> ChatResponse:
        assert isinstance(messages[0].content, str)  # narrow for test
        captured["prompt"] = messages[0].content
        return ChatResponse(content=_payload([], []), model="fake")

    fake = FakeLLMClient(handler=handler)
    cell = MemCell(
        items=[
            _msg("I write Python.", sender="u_alice", sender_name="   "),
            _msg("I prefer Go.", sender="u_bob", sender_name="Bob"),
        ],
        timestamp=_TS,
    )

    await ProfileExtractor(llm=fake).aextract([cell], sender_id="u_alice")

    assert "TARGET USER: u_alice" in captured["prompt"]
    assert "TARGET USER: Bob" not in captured["prompt"]


# ==========================================================================
# sender_id fail-loud validation — semantic exclusion of assistant
# ==========================================================================


async def test_aextract_rejects_sender_id_not_a_user_speaker() -> None:
    """sender_id absent from user speakers raises ValueError before any LLM call."""
    fake = FakeLLMClient(responses=[ChatResponse(content=_payload([], []), model="fake")])
    cell = MemCell(items=[_msg("hi", sender="u_alice")], timestamp=_TS)
    with pytest.raises(ValueError, match="not a user speaker"):
        await ProfileExtractor(llm=fake).aextract([cell], sender_id="u_ghost")


async def test_aextract_rejects_assistant_as_target() -> None:
    """Assistant is not a valid profile owner even when it speaks (semantic exclusion)."""
    fake = FakeLLMClient(responses=[ChatResponse(content=_payload([], []), model="fake")])
    cell = MemCell(
        items=[
            _msg("How can I help?", role="assistant", sender="assistant", sender_name="AI"),
            _msg("I like tea.", role="user", sender="u_alice"),
        ],
        timestamp=_TS,
    )
    with pytest.raises(ValueError, match="not a user speaker"):
        await ProfileExtractor(llm=fake).aextract([cell], sender_id="assistant")


# ==========================================================================
# Language rules — INIT decides the language, UPDATE / COMPACT preserve it
# ==========================================================================


def test_init_prompt_carries_the_language_placeholder_at_both_ends() -> None:
    """Long prompts lose middle instructions, so the rule is spliced at head and tail."""
    from everalgo.user_memory.prompts.en.profile import PROFILE_INITIAL_EXTRACTION_PROMPT

    assert PROFILE_INITIAL_EXTRACTION_PROMPT.count("{language_rule}") == 2


def test_all_three_prompts_carry_the_language_placeholder_at_both_ends() -> None:
    """Every mode honours ``output_language``, so every prompt takes its rule at render time.

    Long prompts lose middle instructions, hence head and tail rather than once.
    """
    from everalgo.user_memory.prompts.en.profile import (
        PROFILE_COMPACT_PROMPT,
        PROFILE_INITIAL_EXTRACTION_PROMPT,
        PROFILE_UPDATE_PROMPT,
    )

    for prompt in (PROFILE_INITIAL_EXTRACTION_PROMPT, PROFILE_UPDATE_PROMPT, PROFILE_COMPACT_PROMPT):
        assert prompt.count("{language_rule}") == 2
        assert "CRITICAL LANGUAGE RULE" not in prompt


async def test_init_rendering_injects_the_participant_rule_when_no_language_is_named() -> None:
    rendered = await _render_init_prompt()

    assert rendered.count("CRITICAL LANGUAGE RULE") == 2
    assert "the language the participants use" in rendered
    assert "{language_rule}" not in rendered


async def test_init_rendering_injects_the_named_language() -> None:
    rendered = await _render_init_prompt(output_language=OutputLanguage.GERMAN)

    assert rendered.count("CRITICAL LANGUAGE RULE") == 2
    assert "Write ALL output fields in German." in rendered
    assert "the language the participants use" not in rendered


async def _render_init_prompt(**kwargs: object) -> str:
    """Capture what the extractor hands the LLM; the rule only exists after rendering."""
    captured: list[str] = []

    class Capture:
        async def chat(self, messages: list[LLMChatMessage], **_: object) -> ChatResponse:
            assert isinstance(messages[0].content, str)  # narrow for test
            captured.append(messages[0].content)
            raise _PromptCapturedError

    with pytest.raises(_PromptCapturedError):
        await ProfileExtractor(llm=Capture()).aextract([_memcell()], sender_id="u_alice", **kwargs)  # type: ignore[arg-type]
    return captured[0]


class _PromptCapturedError(Exception):
    """Ends the call once the prompt has been captured — no LLM response is needed."""


def test_update_and_compact_fall_back_to_inheriting_the_profiles_language() -> None:
    """Without a named language those two inherit rather than re-judge, and carry no judgement clauses.

    Re-judging downstream is how an update in a second language would split a profile's language in half.
    The fallbacks differ from INIT's for that reason — the conversation rule's paragraphs would invite
    exactly the judgement these paths must not make.
    """
    from everalgo.user_memory._language import (
        COMPACTED_PROFILE_LANGUAGE_RULE,
        EXISTING_PROFILE_LANGUAGE_RULE,
        build_language_rule,
    )

    update = build_language_rule(None, fallback=EXISTING_PROFILE_LANGUAGE_RULE)
    compact = build_language_rule(None, fallback=COMPACTED_PROFILE_LANGUAGE_RULE)

    assert "as the existing profile you are updating" in update
    assert "as the profile you are compacting" in compact
    for rule in (update, compact):
        assert "dominate" not in rule
        assert "the language the participants use" not in rule


def test_compact_prompt_no_longer_references_a_conversation_it_never_receives() -> None:
    """`_compact` is called with a Profile only — the old rule cited a non-existent input."""
    from everalgo.user_memory.prompts.en.profile import PROFILE_COMPACT_PROMPT

    assert "input conversation content" not in PROFILE_COMPACT_PROMPT


def test_every_profile_fallback_binds_the_personality_tags() -> None:
    """Tags are the one part of a profile a language rule can miss, so all three paths must name them.

    They appear in the prompts only as English examples, and they are model-authored free text, so a rule
    that binds the descriptions without naming the tags leaves them free to stay English. INIT is included
    deliberately: it is the call that sets the language the other two inherit.
    """
    from everalgo.user_memory._language import (
        COMPACTED_PROFILE_LANGUAGE_RULE,
        EXISTING_PROFILE_LANGUAGE_RULE,
        PROFILE_INIT_LANGUAGE_RULE,
    )

    for rule in (PROFILE_INIT_LANGUAGE_RULE, EXISTING_PROFILE_LANGUAGE_RULE, COMPACTED_PROFILE_LANGUAGE_RULE):
        assert "personality tag" in rule


def test_init_fallback_says_it_is_the_call_that_fixes_the_language() -> None:
    """Only INIT chooses; UPDATE and COMPACT inherit. The prompt has to say which call it is."""
    from everalgo.user_memory._language import PROFILE_INIT_LANGUAGE_RULE

    assert "fixes the profile's language" in PROFILE_INIT_LANGUAGE_RULE


def test_init_fallback_extends_the_measured_participant_rule_verbatim() -> None:
    """The eight-arm measurement describes the participant rule, so INIT must add to it, not reword it."""
    from everalgo.user_memory._language import PROFILE_INIT_LANGUAGE_RULE
    from everalgo.user_memory.prompts.en._language import PARTICIPANT_LANGUAGE_RULE

    assert PROFILE_INIT_LANGUAGE_RULE.startswith(PARTICIPANT_LANGUAGE_RULE)


# ==========================================================================
# INIT language rule — shape guards
#
# 0.4.0 rewrote the INIT rule from a concrete enumeration to an abstract statement and shipped it
# with experimental backing — but the experiments only covered the case that release set out to fix
# (pasted material), never the baseline. On low-information English input the abstract form produced
# whole-profile foreign-language output, and since UPDATE / COMPACT faithfully preserve whatever
# language INIT picked, a mislabelled profile could never recover.
#
# These assertions cost nothing and run on every commit. Had they existed, that release would have
# failed here instead of in production.
# ==========================================================================

# Phrasing that frames the output language as something the model decides. The abstract rule paired
# with "whatever language you choose here" is what turned language into an open variable; once it is
# open, every additional language sentence argues into it rather than closing it down.
_OPEN_VARIABLE_PHRASES_EN = ("whatever language you choose", "language you choose here")


async def test_init_rule_keeps_the_concrete_language_enumeration_at_both_ends() -> None:
    """An abstract "match that language" rule is not enough — name the languages.

    Counted rather than merely present: the rule is spliced at head and tail, and weakening only one
    of the two would otherwise slip past. The rule text itself is guarded in
    ``test_user_memory_language.py``; what this checks is that INIT gets both copies.
    """
    rendered = await _render_init_prompt()

    assert rendered.count("write in Chinese") == 2
    assert rendered.count("if in English, write in English") == 2


@pytest.mark.parametrize("phrase", _OPEN_VARIABLE_PHRASES_EN)
def test_en_profile_prompts_never_frame_language_as_a_choice(phrase: str) -> None:
    """No prompt may describe the output language as the model's to pick."""
    import everalgo.user_memory.prompts.en.profile as en_mod

    for name in ("PROFILE_UPDATE_PROMPT", "PROFILE_COMPACT_PROMPT", "PROFILE_INITIAL_EXTRACTION_PROMPT"):
        assert phrase not in getattr(en_mod, name), name


# ==========================================================================
# Prompt shape
#
# Every rule below answers a measured failure, and the rewrite put the shared ones in two constants that
# all three prompts splice in — which is what stops compaction from drifting behind the other two paths, as
# it had (brackets on 40/40 traits where the others had 0/44; capability items merged rather than deleted).
# Guards therefore assert against `_PORTRAIT` / `_ITEM_RULES` once, plus the per-prompt parts separately.
#
# Numbers, all from 20-run measurements on a corpus of eight purely operational conversations (nothing the
# person said about themselves, so the correct answer is nothing at all):
#   before: 8.00 explicit items per run, 86% of them an action restated as a standing capability
#   after:  0.00
# ==========================================================================

_PORTRAIT_CLAUSES = (
    "A profile is a **portrait of a person**, not a log of what happened.",
    "only because it is still true the next time you meet this person",
    "can correctly yield nothing at all",
    # explicit_info: stated, and still true next time.
    "what this person **stated** about themselves",
    "I'm on leave next week",
    "there is nowhere to record when they expire",
    # implicit_traits: the unstated half, with a threshold on what counts as evidence.
    "the half of the portrait they did not state outright",
    "should not leave this empty",
    "**two or more signals**",
    "**chose or asserted** — never an operation they carried out",
    "two actions are not two signals about who someone is",
    # Which bucket: decided by whether the person said it, not by what the fact is about.
    "decided by whether this person said it, never by what the fact is about",
    "including how their team requires work to be done, what they refuse to touch",
    "implicit_traits holds only what no sentence of theirs states",
    "does not move it across",
    # The capability ban, and why an evidence line does not rescue one already stored.
    "**Never restate an action as a capability, duty, skill, familiarity, interest or attitude.**",
    "not caring about test results or an interest in testing either",
    "asking what a skipped test was is a question, not an attitude towards testing",
    "interested in, attentive to or concerned with anything unless they said so themselves",
    # The mechanism behind the leftover noise: what the assistant explained came back as the person's own
    # knowledge, so the ban on capabilities never reached it.
    "**A question records nothing but the question, and being told something is not knowing it.**",
    "it never becomes this person's knowledge, familiarity or concern",
    "leaves the portrait exactly as it was",
    "an occasion cannot establish one",
    "to be removed, not rewritten and not merged",
)

_ITEM_CLAUSES = (
    # One dimension, one item — with the name-reuse and granularity tests that make it bind.
    "**One item per dimension.**",
    "is ONE item, not three",
    "Reuse a category or trait name already in use rather than coining a near-synonym",
    "Dimensions are coarse",
    "When unsure whether a dimension is new, it is not.",
    # No cross-dimension chaining. The connective list is load-bearing; see the note below.
    "**One item, one dimension.**",
    '"also", "as well as" or a comma-list',
    "Where neither belongs in a portrait, write nothing.",
    # No stand-in for the subject, in either field.
    "**Never name the subject.**",
    'never "{target_user} works mainly in Python"',
    # Interpolated, not a literal name. No real name may ship in a PyPI prompt, and a fixed stand-in does not
    # work either: a literal "Alice" — absent from the conversation — took implicit_traits from 0.95 to
    # 0.00/0.05/0.00 per run over three repeats (same-version spread 0.05). The example has to name whoever is
    # actually speaking; a stranger's name reads as another participant and unanchors the subject.
    "give what was said and when, not who said it",
)


@pytest.mark.parametrize("clause", _PORTRAIT_CLAUSES)
def test_portrait_block_carries_its_clauses(clause: str) -> None:
    """The definition of a profile and of the two buckets, shared by all three prompts."""
    from everalgo.user_memory.prompts.en.profile import _PORTRAIT

    assert clause in _PORTRAIT


@pytest.mark.parametrize("clause", _ITEM_CLAUSES)
def test_item_rules_block_carries_its_clauses(clause: str) -> None:
    """How items are shaped: one per dimension, one dimension each, subject never named."""
    from everalgo.user_memory.prompts.en.profile import _ITEM_RULES

    assert clause in _ITEM_RULES


@pytest.mark.parametrize(
    "name", ["PROFILE_INITIAL_EXTRACTION_PROMPT", "PROFILE_UPDATE_PROMPT", "PROFILE_COMPACT_PROMPT"]
)
def test_all_three_prompts_splice_in_the_shared_blocks(name: str) -> None:
    """Compaction fell behind precisely because it carried its own wording; a shared block cannot drift."""
    import everalgo.user_memory.prompts.en.profile as mod

    prompt = getattr(mod, name)
    assert mod._PORTRAIT in prompt, name
    assert mod._ITEM_RULES in prompt, name
    assert mod._ITEM_SHAPE in prompt, name


def test_item_shape_says_what_basis_holds() -> None:
    """Left undefined, "basis" came back as the rule that asks for it rather than the signals it asks for.

    The nearest instruction was the trait threshold, so the model filled the field with a translation of
    "two or more signals" — a claim of having found evidence, in the slot meant to hold it. Naming the
    signals is now the requirement, and restating the threshold is named as the failure.
    """
    from everalgo.user_memory.prompts.en.profile import _ITEM_SHAPE

    assert '"basis" names the signals themselves' in _ITEM_SHAPE
    assert "each one findable in the conversation" in _ITEM_SHAPE
    assert 'Restating the requirement ("two or more signals", "multiple instances", "repeated choices")' in _ITEM_SHAPE
    assert "if you cannot name the signals, the trait does not belong here at all" in _ITEM_SHAPE


def test_init_prompt_permits_producing_nothing() -> None:
    """Without an explicit way out, a conversation with nothing to record still has to yield something.

    INIT's instruction is "build a user profile", so on purely operational input the model filled the gap by
    restating operations as capabilities. Granting permission moved capability phrasing 0.80 -> 0.20 per run
    after four rounds of prohibitions had stalled.
    """
    from everalgo.user_memory.prompts.en.profile import PROFILE_INITIAL_EXTRACTION_PROMPT

    assert "Either list may be empty, and often should be." in PROFILE_INITIAL_EXTRACTION_PROMPT
    assert "That IS the correct answer there" in PROFILE_INITIAL_EXTRACTION_PROMPT
    assert "Do not treat producing items as the goal." in PROFILE_INITIAL_EXTRACTION_PROMPT


def test_update_prompt_reserves_add_for_dimensions_not_yet_on_file() -> None:
    """`add` used to read "completely new user information", which a new fact about a stored dimension is."""
    from everalgo.user_memory.prompts.en.profile import PROFILE_UPDATE_PROMPT

    assert "a dimension not yet on file at all" in PROFILE_UPDATE_PROMPT
    assert "a second item under an existing category or trait name is always wrong" in PROFILE_UPDATE_PROMPT
    assert "This is the operation for a fresh occurrence of something already recorded" in PROFILE_UPDATE_PROMPT


def test_update_prompt_can_delete_a_stored_capability_item() -> None:
    """Deleting these is compaction's job, but UPDATE must be able to when the conversation raises one."""
    from everalgo.user_memory.prompts.en.profile import PROFILE_UPDATE_PROMPT

    assert (
        "asserts a duty, capability, skill, familiarity, interest or attitude this person never claimed"
        in PROFILE_UPDATE_PROMPT
    )


def test_update_prompt_keeps_index_semantics_and_field_consistency() -> None:
    """Index rules predate this work and still carry it; omitted fields keeping stored values does too."""
    from everalgo.user_memory.prompts.en.profile import PROFILE_UPDATE_PROMPT

    assert "every index resolves against the profile snapshot shown above" in PROFILE_UPDATE_PROMPT
    assert "never shift each other's indices" in PROFILE_UPDATE_PROMPT
    assert '"add" takes no index' in PROFILE_UPDATE_PROMPT
    assert 'fields you omit from an "update" keep their stored values' in PROFILE_UPDATE_PROMPT


def test_update_prompt_output_examples_share_one_shape() -> None:
    """Two examples in different shapes produced unparseable JSON in 6/20 runs.

    A flat single-line "none" example beside a three-level indented one had the model writing a single line
    and closing it with the nested example's brace count. Both are indented now, and a sentence states the
    brace count outright.
    """
    from everalgo.user_memory.prompts.en.profile import PROFILE_UPDATE_PROMPT

    assert "Every operation object closes with exactly two braces before the comma" in PROFILE_UPDATE_PROMPT
    # Neither example may be the flat form that got mixed with the nested one.
    assert '{{"operations": [{{"action": "none"}}]' not in PROFILE_UPDATE_PROMPT


def test_compact_threshold_equals_the_item_limit() -> None:
    """Compaction runs at the limit, not 50% past it — it is the only path that removes non-portrait items.

    UPDATE cannot do that job: it sees a snapshot plus one conversation and acts on what the conversation
    raises. Told to audit the whole profile it emitted the deletes in 1 run out of 10 and otherwise answered
    "none, nothing needs cleaning" — it had read the instruction and judged the items acceptable. Measured at
    this threshold, a 32-item profile carrying 29 capability items comes back with 1.9 of them and all 3
    legitimate items intact; at the old threshold it would have had to reach 46 items first.
    """
    from everalgo.user_memory.profile import _PROFILE_COMPACT_THRESHOLD, _PROFILE_MAX_ITEMS

    assert _PROFILE_COMPACT_THRESHOLD == _PROFILE_MAX_ITEMS


def test_compact_prompt_deletes_before_merging() -> None:
    """Ordering matters: the two steps collide on exactly the items that must go.

    Merge-first turned 22 capability items sharing one category into a single surviving false claim. Stating
    delete-first, and that an evidence line does not rescue such an item, takes 29 of them down to 1.9 with
    every legitimate item kept (30/30 across three thresholds).
    """
    from everalgo.user_memory.prompts.en.profile import PROFILE_COMPACT_PROMPT

    assert "Work in this order." in PROFILE_COMPACT_PROMPT
    assert "Delete everything that was never a portrait item, before merging anything." in PROFILE_COMPACT_PROMPT
    assert "Twelve such items become **zero** items, not one." in PROFILE_COMPACT_PROMPT
    assert "Being numerous is not evidence" in PROFILE_COMPACT_PROMPT
    # The delete step must come before the merge step in the text, not merely be present.
    assert PROFILE_COMPACT_PROMPT.index("Delete everything that was never") < PROFILE_COMPACT_PROMPT.index(
        "Then collapse each remaining dimension"
    )


def test_compact_prompt_rehomes_a_trait_that_merely_restates_a_statement() -> None:
    """Compaction is the only path that rewrites both lists, so it is the only one that can move an item.

    A stated fact filed as a disposition ("insists on the team's merge process" for "we must go through MRs")
    is the same fact in the wrong bucket; leaving it there splits one dimension across two lists.
    """
    from everalgo.user_memory.prompts.en.profile import PROFILE_COMPACT_PROMPT

    assert "**Move anything misfiled.**" in PROFILE_COMPACT_PROMPT
    assert "A trait that merely restates something this person said is an explicit_info fact" in PROFILE_COMPACT_PROMPT
    assert "do not leave a copy behind in implicit_traits" in PROFILE_COMPACT_PROMPT
    # Deleting non-portrait items first keeps this step from rehoming an item that should simply go.
    assert PROFILE_COMPACT_PROMPT.index("Delete everything that was never") < PROFILE_COMPACT_PROMPT.index(
        "Move anything misfiled"
    )


def test_no_prompt_demonstrates_a_forbidden_form() -> None:
    """An example beats the rule it contradicts — measured four times over.

    `evidence` named the speaker in 71/71 items while a rule forbade it, because one example showed
    "In Oct 2024 user mentioned...". Compaction bracketed 40/40 traits where the other paths bracketed 0/44,
    the only difference being an "(e.g., [Risk-Averse])". Both examples are gone; this pins them out.
    """
    import everalgo.user_memory.prompts.en.profile as mod

    for name in ("PROFILE_INITIAL_EXTRACTION_PROMPT", "PROFILE_UPDATE_PROMPT", "PROFILE_COMPACT_PROMPT"):
        prompt = getattr(mod, name)
        assert "user mentioned" not in prompt, name
        assert "[Risk-Averse]" not in prompt, name
        assert "current status" not in prompt, name  # the phrase that used to request a passing state


# Each rule was added in its own measurement round, so nothing stopped a later round from restating an
# earlier one. One did: the durability clause landed twice when an interrupted command had already written
# the first wording — invisible to lint, to the tests, and to a diff that reads as two ordinary additions.
#
# NOT listed here: the "also / as well as / comma-list" prohibition, which appears in both the generic item
# rule and the trait rule. That reads as a duplicate and was removed as one; measured, it is load-bearing.
# Explicit items went 0.63-0.75 per run (two repeats) to 1.30-1.40 (three), two non-overlapping ranges
# against a repeat-run spread of 0.05-0.12 — and the regression showed up in `explicit_info`, which the trait
# rule does not govern. The prompt's signals are not partitioned by section. Do not "tidy" that one away.
_SINGLE_OCCURRENCE_PHRASES = (
    "portrait of a person",
    "**One item per dimension.**",
    "**One item, one dimension.**",
    "**Never name the subject.**",
    "**two or more signals**",
)


@pytest.mark.parametrize("phrase", _SINGLE_OCCURRENCE_PHRASES)
def test_no_rule_is_stated_twice_within_one_prompt(phrase: str) -> None:
    """One paragraph per rule, per prompt — a rule stated twice is a rule that got edited twice."""
    import everalgo.user_memory.prompts.en.profile as mod

    for name in ("PROFILE_INITIAL_EXTRACTION_PROMPT", "PROFILE_UPDATE_PROMPT", "PROFILE_COMPACT_PROMPT"):
        prompt = getattr(mod, name)
        assert prompt.count(phrase) == 1, f"{name}: {phrase!r} appears {prompt.count(phrase)}x"


# ==========================================================================
# Label inventory — what the UPDATE prompt is shown before the JSON dump
# ==========================================================================


def test_label_inventory_lists_distinct_labels_in_first_seen_order() -> None:
    """Duplicates collapse and order is preserved, so the inventory reads as a checklist."""
    from everalgo.user_memory.profile import _render_label_inventory

    rendered = _render_label_inventory(
        [
            {"category": "工作内容", "description": "a"},
            {"category": "技术知识", "description": "b"},
            {"category": "工作内容", "description": "c"},
        ],
        [{"trait": "高效务实", "description": "d"}],
    )

    assert "工作内容, 技术知识" in rendered
    assert "高效务实" in rendered
    assert rendered.count("工作内容") == 1


def test_label_inventory_marks_an_empty_bucket_rather_than_leaving_a_blank() -> None:
    """A blank line reads as "nothing to check against"; the sentinel says it outright."""
    from everalgo.user_memory.profile import _render_label_inventory

    rendered = _render_label_inventory([], [])

    assert rendered.count("(none yet)") == 2


def test_label_inventory_skips_non_dict_and_blank_labels() -> None:
    """extra="allow" means a bucket can hold anything; a blank label is no label."""
    from everalgo.user_memory.profile import _render_label_inventory

    rendered = _render_label_inventory(["not a dict", {"category": "   "}, {"description": "no category"}], [])

    assert "(none yet)" in rendered


def test_update_render_puts_the_inventory_before_the_indexed_items() -> None:
    """The inventory is a checklist to read first; after the JSON dump it would be read last or not at all."""
    from everalgo.user_memory.profile import _render_profile_for_update

    profile = Profile.model_validate(
        {
            "owner_id": "u_alice",
            "summary": "s",
            "timestamp": _TS,
            "explicit_info": [{"category": "Skills", "description": "Python.", "evidence": "x"}],
            "implicit_traits": [{"trait": "[Pragmatic]", "description": "d", "basis": "b", "evidence": "e"}],
        }
    )

    rendered = _render_profile_for_update(profile)

    assert rendered.index("categories already in use") < rendered.index("=== explicit_info ===")
    assert rendered.index("trait labels already in use") < rendered.index("=== explicit_info ===")
    assert "[0] " in rendered  # indexed items still there, numbering untouched


# ==========================================================================
# _apply_ops — index semantics, op validation, add dedup
#
# _render_profile_for_update numbers items with enumerate over the profile it is
# handed, so that snapshot is the only index base a model can see. Every op must
# therefore resolve against it, unaffected by the other ops in the same batch.
# ==========================================================================

_BASE = ["A", "B", "C", "D", "E"]
_PROFILE_LOGGER = "everalgo.user_memory.profile"


def _profile_of(descriptions: list[str], *, bucket: str = "explicit_info") -> Profile:
    """Profile holding one item per description in ``bucket``; the other bucket stays empty."""
    identity = "category" if bucket == "explicit_info" else "trait"
    other = "implicit_traits" if bucket == "explicit_info" else "explicit_info"
    return Profile.model_validate(
        {
            "owner_id": "u_alice",
            "summary": "s",
            "timestamp": _TS_OLD_1,
            bucket: [{identity: "c", "description": d, "evidence": "e"} for d in descriptions],
            other: [],
        }
    )


def _descriptions(profile: Profile, bucket: str = "explicit_info") -> list[str]:
    items = cast("list[dict[str, Any]]", getattr(profile, bucket))
    return [item["description"] for item in items]


def test_delete_then_update_both_resolve_against_the_original_snapshot() -> None:
    """A delete must not shift the index a later update refers to.

    Sequential mutation made this edit land on E while D survived — a self-consistent but wrong
    profile the caller had no way to detect afterwards.
    """
    ops: list[dict[str, Any]] = [
        {"action": "delete", "type": "explicit_info", "index": 1},
        {"action": "update", "type": "explicit_info", "index": 3, "data": {"description": "D-updated"}},
    ]
    assert _descriptions(_apply_ops(_profile_of(_BASE), ops, timestamp=_TS)) == ["A", "C", "D-updated", "E"]


def test_two_deletes_both_resolve_against_the_original_snapshot() -> None:
    """Two deletes in one batch each address the original numbering, not the shrinking list."""
    ops: list[dict[str, Any]] = [
        {"action": "delete", "type": "explicit_info", "index": 1},
        {"action": "delete", "type": "explicit_info", "index": 3},
    ]
    assert _descriptions(_apply_ops(_profile_of(_BASE), ops, timestamp=_TS)) == ["A", "C", "E"]


def test_add_does_not_widen_the_index_bound(caplog: pytest.LogCaptureFixture) -> None:
    """An index past the snapshot's end stays out of range however many items this batch adds."""
    ops: list[dict[str, Any]] = [
        # A category no stored item owns, so the add lands as an add rather than being folded in.
        {"action": "add", "type": "explicit_info", "data": {"category": "fresh", "description": "F-new"}},
        {"action": "update", "type": "explicit_info", "index": 5, "data": {"description": "CLOBBERED"}},
    ]
    with caplog.at_level(logging.WARNING, logger=_PROFILE_LOGGER):
        result = _apply_ops(_profile_of(_BASE), ops, timestamp=_TS)

    assert _descriptions(result) == [*_BASE, "F-new"]
    assert "index out of range" in caplog.text


def test_add_under_an_existing_category_is_folded_into_that_item() -> None:
    """One dimension, one item — enforced here, not only asked for in the prompt.

    The prompt states that a second item under an existing category is always wrong, and the model mostly
    complies. Mostly is not enough: an end-to-end walkthrough produced three items under one `职业角色`
    category, which is the duplication this whole area exists to prevent. The incoming fact is real, so it
    is folded into the item that owns the label rather than dropped.
    """
    ops: list[dict[str, Any]] = [
        {"action": "add", "type": "explicit_info", "data": {"category": "c", "description": "refuses front-end"}},
    ]
    result = _apply_ops(_profile_of(["works in Python"]), ops, timestamp=_TS)

    assert _descriptions(result) == ["works in Python; refuses front-end"]


def test_folding_extends_evidence_and_skips_a_restated_fact() -> None:
    """Evidence accumulates with the description; a fact already stated verbatim does not lengthen it."""
    stored = _profile_of(["works in Python"])
    ops: list[dict[str, Any]] = [
        {
            "action": "add",
            "type": "explicit_info",
            "data": {"category": "c", "description": "refuses front-end", "evidence": "e2"},
        },
        # Same description again: nothing to append.
        {"action": "add", "type": "explicit_info", "data": {"category": "c", "description": "works in Python"}},
    ]
    result = _apply_ops(stored, ops, timestamp=_TS)
    item = cast("list[dict[str, Any]]", result.explicit_info)[0]  # type: ignore[attr-defined]

    assert item["description"] == "works in Python; refuses front-end"
    assert item["evidence"] == "e; e2"


def test_folding_applies_to_traits_by_their_own_label_field() -> None:
    """`implicit_traits` is keyed by `trait`, not `category`, so the bucket's own field decides."""
    ops: list[dict[str, Any]] = [
        {"action": "add", "type": "implicit_traits", "data": {"trait": "c", "description": "second observation"}},
    ]
    result = _apply_ops(_profile_of(["first observation"], bucket="implicit_traits"), ops, timestamp=_TS)

    assert _descriptions(result, "implicit_traits") == ["first observation; second observation"]


def test_add_under_a_free_category_still_adds() -> None:
    """Folding must not swallow a genuinely new dimension."""
    ops: list[dict[str, Any]] = [
        {"action": "add", "type": "explicit_info", "data": {"category": "elsewhere", "description": "lives in Berlin"}},
    ]
    result = _apply_ops(_profile_of(["works in Python"]), ops, timestamp=_TS)

    assert _descriptions(result) == ["works in Python", "lives in Berlin"]


def test_unknown_type_is_rejected_rather_than_routed_to_implicit_traits(caplog: pytest.LogCaptureFixture) -> None:
    """A misspelt bucket name used to fall through to implicit_traits by way of an else branch."""
    ops: list[dict[str, Any]] = [
        {"action": "add", "type": "explicit", "data": {"category": "c", "description": "X"}},
    ]
    with caplog.at_level(logging.WARNING, logger=_PROFILE_LOGGER):
        result = _apply_ops(_profile_of(_BASE), ops, timestamp=_TS)

    assert _descriptions(result) == _BASE
    assert _descriptions(result, "implicit_traits") == []
    assert "unknown type" in caplog.text


def test_add_skips_an_item_the_profile_already_holds() -> None:
    """Dedup is a correctness property of the merge, not something the prompt can be trusted with."""
    item = {"category": "diet", "description": "loves kiwifruit", "evidence": "e"}
    ops: list[dict[str, Any]] = [{"action": "add", "type": "explicit_info", "data": dict(item)} for _ in range(3)]
    result = _apply_ops(_profile_of(["loves kiwifruit"]), ops, timestamp=_TS)

    assert _descriptions(result) == ["loves kiwifruit"]


def test_dedup_ignores_the_category_label() -> None:
    """The label is LLM-authored and drifts between rounds, so keying on it would let dupes through."""
    ops: list[dict[str, Any]] = [
        {"action": "add", "type": "explicit_info", "data": {"category": "preferences", "description": "A"}},
    ]
    result = _apply_ops(_profile_of(_BASE), ops, timestamp=_TS)

    assert _descriptions(result) == _BASE


def test_dedup_ignores_case_and_whitespace_differences() -> None:
    """A model restating an item rarely reproduces its spacing and casing byte for byte."""
    ops: list[dict[str, Any]] = [
        {"action": "add", "type": "explicit_info", "data": {"category": " C ", "description": "Loves   Kiwifruit"}},
    ]
    result = _apply_ops(_profile_of(["loves kiwifruit"]), ops, timestamp=_TS)

    assert _descriptions(result) == ["loves kiwifruit"]


def test_dedup_ignores_evidence_because_it_is_provenance_not_identity() -> None:
    """The same fact cited from a second conversation is still the same fact."""
    ops: list[dict[str, Any]] = [
        {
            "action": "add",
            "type": "explicit_info",
            "data": {"category": "c", "description": "loves kiwifruit", "evidence": "a different quote"},
        },
    ]
    result = _apply_ops(_profile_of(["loves kiwifruit"]), ops, timestamp=_TS)

    assert _descriptions(result) == ["loves kiwifruit"]


def test_dedup_keeps_the_earliest_item_so_the_summary_stays_stable() -> None:
    """_build_summary reads explicit_info[0], so dropping the later copy keeps summary steady."""
    ops: list[dict[str, Any]] = [
        {"action": "add", "type": "explicit_info", "data": {"category": "c", "description": "A"}},
    ]
    result = _apply_ops(_profile_of(_BASE), ops, timestamp=_TS)

    assert _descriptions(result) == _BASE
    assert result.summary == "A"


def test_implicit_traits_dedup_reads_description_like_explicit_info_does() -> None:
    """`description` is the field both buckets share, so one identity rule covers both."""
    ops: list[dict[str, Any]] = [
        {"action": "add", "type": "implicit_traits", "data": {"trait": "[Different-Tag]", "description": "A"}},
    ]
    result = _apply_ops(_profile_of(["A"], bucket="implicit_traits"), ops, timestamp=_TS)

    assert _descriptions(result, "implicit_traits") == ["A"]


def test_delete_wins_when_one_index_is_both_updated_and_deleted(caplog: pytest.LogCaptureFixture) -> None:
    """A delete answers to an explicit negation or a contradiction, which outranks a correction."""
    ops: list[dict[str, Any]] = [
        {"action": "update", "type": "explicit_info", "index": 1, "data": {"description": "B-updated"}},
        {"action": "delete", "type": "explicit_info", "index": 1},
    ]
    with caplog.at_level(logging.WARNING, logger=_PROFILE_LOGGER):
        result = _apply_ops(_profile_of(_BASE), ops, timestamp=_TS)

    assert _descriptions(result) == ["A", "C", "D", "E"]
    assert "superseded by delete" in caplog.text


def test_two_updates_on_one_index_merge_in_order() -> None:
    """A single update is a partial merge, so two of them must accumulate rather than overwrite."""
    ops: list[dict[str, Any]] = [
        {"action": "update", "type": "explicit_info", "index": 0, "data": {"description": "A-updated"}},
        {"action": "update", "type": "explicit_info", "index": 0, "data": {"category": "Skills"}},
    ]
    result = _apply_ops(_profile_of(_BASE), ops, timestamp=_TS)
    first = cast("list[dict[str, Any]]", result.explicit_info)[0]  # type: ignore[attr-defined]

    assert first["description"] == "A-updated"  # first patch survives the second
    assert first["category"] == "Skills"
    assert first["evidence"] == "e"  # untouched field preserved


def test_unknown_action_is_rejected(caplog: pytest.LogCaptureFixture) -> None:
    """An action outside the documented four is a malformed response, not a no-op."""
    ops: list[dict[str, Any]] = [
        {"action": "replace", "type": "explicit_info", "index": 0, "data": {"description": "X"}},
    ]
    with caplog.at_level(logging.WARNING, logger=_PROFILE_LOGGER):
        result = _apply_ops(_profile_of(_BASE), ops, timestamp=_TS)

    assert _descriptions(result) == _BASE
    assert "unknown action" in caplog.text


@pytest.mark.parametrize("action", ["add", "update"])
def test_non_object_data_is_rejected(action: str, caplog: pytest.LogCaptureFixture) -> None:
    """`data` is merged into an item with `**`, so a non-object is unusable either way."""
    ops: list[dict[str, Any]] = [{"action": action, "type": "explicit_info", "index": 0, "data": "not an object"}]
    with caplog.at_level(logging.WARNING, logger=_PROFILE_LOGGER):
        result = _apply_ops(_profile_of(_BASE), ops, timestamp=_TS)

    assert _descriptions(result) == _BASE
    assert "data is not an object" in caplog.text


@pytest.mark.parametrize("data", [{"category": "c"}, {"category": "c", "description": "   "}])
def test_add_without_a_usable_description_is_rejected(data: dict[str, Any], caplog: pytest.LogCaptureFixture) -> None:
    """An item with no description cannot be retrieved and cannot be deduplicated against."""
    ops: list[dict[str, Any]] = [{"action": "add", "type": "explicit_info", "data": data}]
    with caplog.at_level(logging.WARNING, logger=_PROFILE_LOGGER):
        result = _apply_ops(_profile_of(_BASE), ops, timestamp=_TS)

    assert _descriptions(result) == _BASE
    assert "missing description" in caplog.text


def test_update_of_a_non_object_item_is_rejected(caplog: pytest.LogCaptureFixture) -> None:
    """`Profile` allows non-object items through `extra="allow"`, and merging into one used to raise."""
    profile = Profile.model_validate(
        {
            "owner_id": "u_alice",
            "summary": "s",
            "timestamp": _TS_OLD_1,
            "explicit_info": ["not an object"],
            "implicit_traits": [],
        }
    )
    ops: list[dict[str, Any]] = [
        {"action": "update", "type": "explicit_info", "index": 0, "data": {"description": "X"}},
    ]
    with caplog.at_level(logging.WARNING, logger=_PROFILE_LOGGER):
        result = _apply_ops(profile, ops, timestamp=_TS)

    assert cast("list[Any]", result.explicit_info) == ["not an object"]  # type: ignore[attr-defined]
    assert "target item is not an object" in caplog.text


def test_boolean_index_is_rejected(caplog: pytest.LogCaptureFixture) -> None:
    """JSON `true` is an `int` subclass in Python, so it would otherwise address item 1."""
    ops: list[dict[str, Any]] = [{"action": "delete", "type": "explicit_info", "index": True}]
    with caplog.at_level(logging.WARNING, logger=_PROFILE_LOGGER):
        result = _apply_ops(_profile_of(_BASE), ops, timestamp=_TS)

    assert _descriptions(result) == _BASE
    assert "index out of range" in caplog.text


def test_none_action_is_silent_because_it_is_the_documented_no_op(caplog: pytest.LogCaptureFixture) -> None:
    """The prompt tells the model to emit `none` when a conversation carries no user info."""
    ops: list[dict[str, Any]] = [{"action": "none"}]
    with caplog.at_level(logging.WARNING, logger=_PROFILE_LOGGER):
        result = _apply_ops(_profile_of(_BASE), ops, timestamp=_TS)

    assert _descriptions(result) == _BASE
    assert caplog.text == ""


def test_update_rewriting_an_item_into_a_copy_of_another_leaves_one(caplog: pytest.LogCaptureFixture) -> None:
    """An update can mint a duplicate just as an add can, so dedup covers the whole bucket."""
    ops: list[dict[str, Any]] = [
        {"action": "update", "type": "explicit_info", "index": 0, "data": {"description": "B"}},
    ]
    with caplog.at_level(logging.WARNING, logger=_PROFILE_LOGGER):
        result = _apply_ops(_profile_of(["A", "B"]), ops, timestamp=_TS)

    assert _descriptions(result) == ["B"]  # the updated item is the earlier one, so it is the survivor
    assert "duplicate item dropped" in caplog.text


def test_update_heals_duplicates_already_stored_in_the_profile(caplog: pytest.LogCaptureFixture) -> None:
    """Profiles written before dedup existed carry duplicates; an update is their only way out.

    `_compact` is the sole other backstop and does not run until the item count passes its threshold,
    so without this a polluted profile stayed polluted for as long as it sat below that line.
    """
    ops: list[dict[str, Any]] = [{"action": "none"}]
    with caplog.at_level(logging.WARNING, logger=_PROFILE_LOGGER):
        result = _apply_ops(_profile_of(["A", "A", "A", "B"]), ops, timestamp=_TS)

    assert _descriptions(result) == ["A", "B"]
    assert "duplicate item dropped" in caplog.text


def test_dedup_keeps_items_that_have_no_identity_rather_than_dropping_them() -> None:
    """A stored item with no description cannot be compared, and discarding it would lose data."""
    profile = Profile.model_validate(
        {
            "owner_id": "u_alice",
            "summary": "s",
            "timestamp": _TS_OLD_1,
            "explicit_info": [{"category": "c"}, {"category": "d"}, "not an object"],
            "implicit_traits": [],
        }
    )
    result = _apply_ops(profile, [{"action": "none"}], timestamp=_TS)

    assert len(cast("list[Any]", result.explicit_info)) == 3  # type: ignore[attr-defined]


async def test_init_deduplicates_what_the_llm_returns(caplog: pytest.LogCaptureFixture) -> None:
    """INIT had no dedup at all, so a profile could be born dirty and stay that way."""
    payload = _payload(
        explicit_info=[
            {"category": "c", "description": "loves kiwifruit"},
            {"category": "other", "description": "Loves   Kiwifruit"},
            {"category": "c", "description": "loves kiwifruit"},
        ],
        implicit_traits=[_implicit_trait(description="dup"), _implicit_trait(description="dup")],
    )
    fake = FakeLLMClient(responses=[ChatResponse(content=payload, model="fake")])

    with caplog.at_level(logging.WARNING, logger=_PROFILE_LOGGER):
        profile = await ProfileExtractor(llm=fake).aextract([_memcell()], sender_id="u_alice")

    assert _descriptions(profile) == ["loves kiwifruit"]
    assert _descriptions(profile, "implicit_traits") == ["dup"]
    assert "duplicate item dropped" in caplog.text


async def test_compact_deduplicates_its_own_result(caplog: pytest.LogCaptureFixture) -> None:
    """Compaction is the backstop for duplicate accumulation, so its output cannot carry duplicates."""
    old_p = _old_profile(
        explicit_info=[{"category": f"C{i}", "description": f"D{i}."} for i in range(_PROFILE_COMPACT_THRESHOLD)],
        implicit_traits=[],
    )
    ops_json = json.dumps(
        {"operations": [{"action": "add", "type": "explicit_info", "data": {"category": "New", "description": "N."}}]}
    )
    compact_json = json.dumps(
        {
            "explicit_info": [{"category": "c", "description": "same"}, {"category": "c", "description": "same"}],
            "implicit_traits": [],
        }
    )
    fake = FakeLLMClient(
        responses=[
            ChatResponse(content=ops_json, model="fake"),
            ChatResponse(content=compact_json, model="fake"),
        ]
    )

    with caplog.at_level(logging.WARNING, logger=_PROFILE_LOGGER):
        profile = await ProfileExtractor(llm=fake).aextract([_memcell()], sender_id="u_alice", old_profile=old_p)

    assert fake.call_count == 2
    assert _descriptions(profile) == ["same"]
    assert "duplicate item dropped" in caplog.text


def test_apply_ops_carries_owner_id_and_takes_the_given_timestamp() -> None:
    """The merged profile keeps its owner and adopts the newest memcell's timestamp."""
    result = _apply_ops(_profile_of(_BASE), [], timestamp=_TS)

    assert result.owner_id == "u_alice"
    assert result.timestamp == _TS
