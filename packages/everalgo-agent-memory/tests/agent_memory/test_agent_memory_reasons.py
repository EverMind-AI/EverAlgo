"""Tests for the diagnostic surface — ``aextract_with_reason`` on both agent-memory extractors.

Covers what ``aextract`` cannot express: *which* gate rejected an input, the structured numbers
behind that rejection, and — on the skill side — the per-operation outcome list whose length always
matches the LLM's ``operations``. The equivalence tests pin the wrapper contract: ``aextract`` must
keep returning exactly what the richer method's success payload holds.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

import pytest

from everalgo.agent_memory import (
    AgentCaseExtractor,
    AgentSkillExtractor,
    CaseExtractionResult,
    CaseSkipReason,
    OpOutcome,
    SkillExtractionResult,
    SkillSkipReason,
)
from everalgo.agent_memory import case as case_module
from everalgo.agent_memory.case import _should_skip, _strip_before_first_user
from everalgo.llm.types import ChatResponse
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import (
    AgentCase,
    AgentSkill,
    ChatMessage,
    ConversationItem,
    MemCell,
    ToolCall,
    ToolCallFunction,
    ToolCallRequest,
    ToolCallResult,
)

if TYPE_CHECKING:
    from enum import StrEnum

# ── Fixture helpers ─────────────────────────────────────────────────────────────────────────────────

_SUFFICIENT_CONTENT = "## Steps\n1. First step\n2. Second step\n3. Third step\n4. Fourth step\n5. Fifth step"


def _user(content: str, ts: int = 1700000000000) -> ChatMessage:
    return ChatMessage(id="u", role="user", content=content, timestamp=ts, sender_id="user")


def _assistant(content: str, ts: int = 1700000001000) -> ChatMessage:
    return ChatMessage(id="a", role="assistant", content=content, timestamp=ts, sender_id="assistant")


def _tool_call(call_id: str = "call_1", ts: int = 1700000000500) -> ToolCallRequest:
    return ToolCallRequest(
        tool_calls=[ToolCall(id=call_id, function=ToolCallFunction(name="search", arguments="{}"))],
        content="thinking",
        timestamp=ts,
        sender_id="assistant",
    )


def _tool_result(call_id: str = "call_1", ts: int = 1700000000600) -> ToolCallResult:
    return ToolCallResult(tool_call_id=call_id, content="result", timestamp=ts)


def _tool_rounds(n: int, base_ts: int = 1700000000100) -> list[ConversationItem]:
    """``n`` paired (request, result) messages — one tool-call round each."""
    msgs: list[ConversationItem] = []
    for i in range(n):
        msgs.append(_tool_call(call_id=f"call_{i}", ts=base_ts + i * 2))
        msgs.append(_tool_result(call_id=f"call_{i}", ts=base_ts + i * 2 + 1))
    return msgs


def _memcell(items: list[ConversationItem]) -> MemCell:
    ts = items[-1].timestamp if items else 1700000000000
    return MemCell(items=items, timestamp=ts)


def _case(quality: float = 0.8, case_id: str = "case_1") -> AgentCase:
    return AgentCase(
        id=case_id,
        timestamp=1700000000000,
        task_intent="Fix the flaky test",
        approach="1. reproduce 2. isolate 3. patch",
        quality_score=quality,
        key_insight="Race on shared fixture",
    )


def _skill(name: str = "Existing Skill", skill_id: str = "skill_1") -> AgentSkill:
    return AgentSkill(
        id=skill_id,
        name=name,
        description="Does a thing",
        content=_SUFFICIENT_CONTENT,
        confidence=0.7,
        maturity_score=0.8,
        source_case_ids=[],
    )


_COMPRESS_OK = json.dumps(
    {
        "task_intent": "Build API",
        "approach": "1. design 2. implement 3. test",
        "quality_score": 0.85,
        "key_insight": "Use caching",
    }
)


# ── The enum docstrings are the consumer-facing contract, so keep them honest ───────────────────────


class TestDocstringContract:
    """`reasons.py` doubles as the interface reference for consumers, so its promises are testable.

    A consumer reads a member's bullet, sees ``detail={...}``, and writes ``result.detail``. If the
    field it names does not exist on the type that actually reports that member, they get an
    `AttributeError` from following the documentation correctly — so the field names in the prose are
    asserted against the real NamedTuple fields here rather than trusted.
    """

    @staticmethod
    def _bullets(enum_cls: type[StrEnum]) -> dict[str, str]:
        """Map member name → its docstring bullet, flattened to one line.

        The pattern must not assume any particular indentation: Python 3.13 strips the common leading
        whitespace from docstrings at compile time, so a bullet that reads ``"    - ``NAME``"`` on
        3.12 reads ``"- ``NAME``"`` on 3.13+. Hard-coding four spaces here matches nothing on the
        newer interpreters and silently reports every member as undocumented.
        """
        doc = enum_cls.__doc__ or ""
        found: dict[str, str] = {}
        pattern = r"^[ \t]*- ``([A-Z_]+)``(?: / ``([A-Z_]+)``)?(.*?)(?=^[ \t]*- ``|\Z)"
        for m in re.finditer(pattern, doc, re.MULTILINE | re.DOTALL):
            body = " ".join(m.group(3).split())
            for name in (m.group(1), m.group(2)):
                if name:
                    found[name] = body
        return found

    @pytest.mark.parametrize(
        ("enum_cls", "expected"),
        [(CaseSkipReason, 13), (SkillSkipReason, 12)],
        ids=["CaseSkipReason", "SkillSkipReason"],
    )
    def test_every_member_is_documented(self, enum_cls: type[StrEnum], expected: int) -> None:
        """A member with no bullet is invisible to the consumer reading the reference."""
        assert len(enum_cls) == expected, "member count changed — update the count and the docstring"
        undocumented = [m.name for m in enum_cls if m.name not in self._bullets(enum_cls)]
        assert not undocumented, f"{enum_cls.__name__} members missing a docstring bullet: {undocumented}"

    def test_case_detail_promises_name_a_real_field(self) -> None:
        """Every case bullet promising ``detail={...}`` points at a field `CaseExtractionResult` has."""
        for name, bullet in self._bullets(CaseSkipReason).items():
            if "``detail={" in bullet:
                assert "detail" in CaseExtractionResult._fields, f"CaseSkipReason.{name}"

    def test_skill_detail_promises_match_the_type_that_reports_the_member(self) -> None:
        """The check that matters: each bullet must name the field on *its own* carrier.

        Both carriers exist and both have a detail-ish field, so "the name exists somewhere" is too
        weak — documenting plain ``detail={...}`` on a `pre_reason`-reported member passes that and
        still sends the consumer to `SkillExtractionResult.detail`, which does not exist. So the
        split is asserted explicitly: members that fire before any operation exists are reported
        through `pre_reason` and must document `pre_detail`; all others ride an `OpOutcome` and must
        document `detail`.
        """
        # Everything before the LLM call. Grown only when a new short-circuit is added to
        # AgentSkillExtractor.aextract_with_reason ahead of the operations loop.
        pre_llm = {SkillSkipReason.CASE_QUALITY_BELOW_THRESHOLD.name}
        assert "pre_detail" in SkillExtractionResult._fields
        assert "detail" in OpOutcome._fields

        for name, bullet in self._bullets(SkillSkipReason).items():
            promises_pre = "``pre_detail={" in bullet
            # "detail={" also matches inside "pre_detail={", so exclude that overlap.
            promises_op = "``detail={" in bullet
            if name in pre_llm:
                assert promises_pre, f"{name} is reported via pre_reason but does not document pre_detail"
                assert not promises_op, (
                    f"{name} documents detail={{...}}, but it fires before any operation exists — "
                    f"there is no OpOutcome to carry it, and SkillExtractionResult has no 'detail' "
                    f"field. It must document pre_detail instead."
                )
            else:
                assert not promises_pre, (
                    f"{name} documents pre_detail={{...}}, but it is reported through an OpOutcome, "
                    f"whose field is 'detail'."
                )


# ── Case: structural gates (no LLM call) ────────────────────────────────────────────────────────────


class TestCaseStructuralReasons:
    async def test_empty_memcell(self) -> None:
        fake = FakeLLMClient(responses=[])
        result = await AgentCaseExtractor(llm=fake).aextract_with_reason(MemCell(items=[], timestamp=1700000000000))
        assert result.cases == []
        assert result.reason is CaseSkipReason.EMPTY_MEMCELL
        assert fake.call_count == 0

    async def test_no_user_message_strips_to_nothing(self) -> None:
        """Only assistant turns → the system-head strip empties the cell before any other gate."""
        result = await AgentCaseExtractor(llm=FakeLLMClient(responses=[])).aextract_with_reason(
            _memcell([_assistant("hello")])
        )
        assert result.reason is CaseSkipReason.NO_MESSAGES_AFTER_STRIP

    async def test_no_assistant_message(self) -> None:
        result = await AgentCaseExtractor(llm=FakeLLMClient(responses=[])).aextract_with_reason(
            _memcell([_user("hi"), _user("still there?")])
        )
        assert result.reason is CaseSkipReason.NO_ASSISTANT_MESSAGE

    async def test_trajectory_not_closed_reports_trailing_item_kind(self) -> None:
        """A cell ending mid-tool-loop is incomplete; detail names what it actually ended on."""
        items: list[ConversationItem] = [_user("do X"), _tool_call(), _tool_result()]
        result = await AgentCaseExtractor(llm=FakeLLMClient(responses=[])).aextract_with_reason(_memcell(items))
        assert result.reason is CaseSkipReason.TRAJECTORY_NOT_CLOSED
        assert result.detail == {"last_item_kind": "tool_result"}

    async def test_trajectory_not_closed_on_a_cell_ending_in_a_user_turn(self) -> None:
        """The force-split shape: a cell cut mid-conversation can end on a user turn, not a tool item.

        ``detail`` distinguishes it from the mid-tool-loop case, which matters upstream: a trailing
        tool item usually means "the agent has not finished yet", while a trailing user turn means
        the cell boundary landed in the wrong place.
        """
        items: list[ConversationItem] = [_user("do X"), *_tool_rounds(3), _user("and also do Y", ts=1700000099999)]
        result = await AgentCaseExtractor(llm=FakeLLMClient(responses=[])).aextract_with_reason(_memcell(items))
        assert result.reason is CaseSkipReason.TRAJECTORY_NOT_CLOSED
        assert result.detail == {"last_item_kind": "text:user"}

    async def test_no_tool_single_user(self) -> None:
        result = await AgentCaseExtractor(llm=FakeLLMClient(responses=[])).aextract_with_reason(
            _memcell([_user("hi"), _assistant("hello")])
        )
        assert result.reason is CaseSkipReason.NO_TOOL_SINGLE_USER
        assert result.detail == {"user_count": 1}

    async def test_too_few_tool_rounds_carries_observed_and_required(self) -> None:
        items: list[ConversationItem] = [_user("search"), *_tool_rounds(2), _assistant("done", ts=1700000099999)]
        fake = FakeLLMClient(responses=[])
        result = await AgentCaseExtractor(llm=fake).aextract_with_reason(_memcell(items))
        assert result.reason is CaseSkipReason.TOO_FEW_TOOL_ROUNDS
        assert result.detail == {"rounds": 2, "min_rounds": 3}
        assert fake.call_count == 0  # gate fires before any LLM call

    def test_no_user_message_is_unreachable_from_the_pipeline(self) -> None:
        """The one defensive member: reachable only by calling the pre-filter directly.

        ``_strip_before_first_user`` returns either ``[]`` or a list starting with a user message, so
        ``aextract_with_reason`` can never surface this — an all-assistant cell reports
        ``NO_MESSAGES_AFTER_STRIP`` instead (see the test above). Pinned here so the day the strip
        step changes, the claim in the enum docstring fails loudly.
        """
        assert _strip_before_first_user([_assistant("hello")]) == []
        skip = _should_skip([_assistant("hello")])  # direct call bypasses the strip step
        assert skip is not None
        assert skip[0] is CaseSkipReason.NO_USER_MESSAGE

    async def test_trajectory_too_large(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Over the hard ceiling even after trimming — the size caps are scaled down for speed."""
        monkeypatch.setattr(case_module, "PRE_COMPRESS_CHUNK_SIZE", 100)  # limit becomes 2 x 100

        bulky: list[ConversationItem] = []
        for i in range(3):
            bulky.append(_tool_call(call_id=f"call_{i}", ts=1700000000100 + i * 2))
            bulky.append(
                ToolCallResult(
                    tool_call_id=f"call_{i}",
                    content=" ".join(f"word_{n}" for n in range(500)),
                    timestamp=1700000000101 + i * 2,
                )
            )
        items: list[ConversationItem] = [_user("dump everything"), *bulky, _assistant("done", ts=1700000099999)]

        fake = FakeLLMClient(responses=[])
        result = await AgentCaseExtractor(llm=fake).aextract_with_reason(_memcell(items))
        assert result.reason is CaseSkipReason.TRAJECTORY_TOO_LARGE
        assert result.detail["limit"] == 200
        assert result.detail["tokens"] > 200
        assert fake.call_count == 0  # bail happens before pre-compression

    async def test_min_rounds_gate_disabled_lets_trajectory_through(self) -> None:
        """``min_tool_call_rounds=0`` turns the gate off — the trajectory reaches the LLM filter."""
        items: list[ConversationItem] = [_user("search"), *_tool_rounds(1), _assistant("done", ts=1700000099999)]
        fake = FakeLLMClient(
            responses=[
                ChatResponse(
                    content='{"has_exploration": false, "has_user_correction": false, "reason": "linear"}', model="fake"
                ),
            ]
        )
        result = await AgentCaseExtractor(llm=fake, min_tool_call_rounds=0).aextract_with_reason(_memcell(items))
        assert result.reason is CaseSkipReason.FILTER_REJECTED


# ── Case: LLM-driven gates ──────────────────────────────────────────────────────────────────────────


class TestCaseLLMReasons:
    @staticmethod
    def _three_round_trajectory() -> MemCell:
        items: list[ConversationItem] = [_user("search"), *_tool_rounds(3), _assistant("done", ts=1700000099999)]
        return _memcell(items)

    async def test_filter_rejected_surfaces_model_rationale(self) -> None:
        fake = FakeLLMClient(
            responses=[
                ChatResponse(
                    content='{"has_exploration": false, "has_user_correction": false, "reason": "linear lookup"}',
                    model="fake",
                )
            ]
        )
        result = await AgentCaseExtractor(llm=fake).aextract_with_reason(self._three_round_trajectory())
        assert result.cases == []
        assert result.reason is CaseSkipReason.FILTER_REJECTED
        assert result.detail == {"llm_reason": "linear lookup"}

    async def test_compress_empty_intent_and_approach_are_distinct(self) -> None:
        """The two compress failures mean different things and must not collapse into one reason."""
        cell = self._three_round_trajectory()
        pass_filter = '{"has_exploration": true, "has_user_correction": false, "reason": "explored"}'

        empty_intent = FakeLLMClient(
            responses=[
                ChatResponse(content=pass_filter, model="fake"),
                ChatResponse(content=json.dumps({"task_intent": "", "approach": "x"}), model="fake"),
            ]
        )
        assert (
            await AgentCaseExtractor(llm=empty_intent).aextract_with_reason(cell)
        ).reason is CaseSkipReason.COMPRESS_EMPTY_INTENT

        empty_approach = FakeLLMClient(
            responses=[
                ChatResponse(content=pass_filter, model="fake"),
                ChatResponse(content=json.dumps({"task_intent": "t", "approach": ""}), model="fake"),
            ]
        )
        assert (
            await AgentCaseExtractor(llm=empty_approach).aextract_with_reason(cell)
        ).reason is CaseSkipReason.COMPRESS_EMPTY_APPROACH

    async def test_success_has_no_reason_and_empty_detail(self) -> None:
        items: list[ConversationItem] = [_user("build"), *_tool_rounds(21), _assistant("done", ts=1700000099999)]
        fake = FakeLLMClient(responses=[ChatResponse(content=_COMPRESS_OK, model="fake")])
        result = await AgentCaseExtractor(llm=fake).aextract_with_reason(_memcell(items))
        assert len(result.cases) == 1
        assert result.reason is None
        assert result.detail == {}


class TestCaseWrapperEquivalence:
    async def test_aextract_returns_the_same_cases(self) -> None:
        items: list[ConversationItem] = [_user("build"), *_tool_rounds(21), _assistant("done", ts=1700000099999)]
        cell = _memcell(items)

        plain = await AgentCaseExtractor(
            llm=FakeLLMClient(responses=[ChatResponse(content=_COMPRESS_OK, model="fake")])
        ).aextract(cell)
        rich = await AgentCaseExtractor(
            llm=FakeLLMClient(responses=[ChatResponse(content=_COMPRESS_OK, model="fake")])
        ).aextract_with_reason(cell)

        assert [c.task_intent for c in plain] == [c.task_intent for c in rich.cases]

    async def test_aextract_returns_empty_where_the_richer_call_reports_a_reason(self) -> None:
        cell = _memcell([_user("hi"), _assistant("hello")])
        assert await AgentCaseExtractor(llm=FakeLLMClient(responses=[])).aextract(cell) == []
        assert (await AgentCaseExtractor(llm=FakeLLMClient(responses=[])).aextract_with_reason(cell)).reason is not None


# ── Skill: pre_reason vs per-op outcomes ────────────────────────────────────────────────────────────


class TestSkillPreReason:
    async def test_low_quality_short_circuits_before_any_llm_call(self) -> None:
        fake = FakeLLMClient(responses=[])
        result = await AgentSkillExtractor(llm=fake).aextract_with_reason(
            _case(quality=0.1), existing_relevant_skills=[], supporting_cases=[]
        )
        assert result.pre_reason is SkillSkipReason.CASE_QUALITY_BELOW_THRESHOLD
        assert result.pre_detail == {"quality": 0.1, "threshold": 0.2}
        assert result.outcomes == []
        assert result.skills == []
        assert fake.call_count == 0

    async def test_pre_detail_is_empty_when_no_short_circuit_fired(self) -> None:
        """``pre_detail`` describes ``pre_reason`` and nothing else — per-op context lives on OpOutcome."""
        ops = json.dumps(
            {"operations": [{"action": "add", "data": {"name": "A", "description": "d", "content": "short"}}]}
        )
        fake = FakeLLMClient(responses=[ChatResponse(content=ops, model="fake")])
        result = await AgentSkillExtractor(llm=fake).aextract_with_reason(
            _case(), existing_relevant_skills=[], supporting_cases=[]
        )
        assert result.pre_reason is None
        assert result.pre_detail == {}
        assert result.outcomes[0].detail  # the dropped op's context is here, not on pre_detail

    async def test_llm_proposing_nothing_is_distinguishable_from_the_short_circuit(self) -> None:
        """Both yield no skills; only ``pre_reason`` tells the caller whether the LLM ever ran."""
        fake = FakeLLMClient(responses=[ChatResponse(content='{"operations": []}', model="fake")])
        result = await AgentSkillExtractor(llm=fake).aextract_with_reason(
            _case(), existing_relevant_skills=[], supporting_cases=[]
        )
        assert result.pre_reason is None
        assert result.outcomes == []
        assert fake.call_count == 1


class TestSkillOpOutcomes:
    async def test_one_outcome_per_proposed_operation(self) -> None:
        """The invariant callers rely on to map outcomes back to operations."""
        ops = json.dumps(
            {
                "operations": [
                    {"action": "add", "data": {"name": "A", "description": "d", "content": "too short"}},
                    {"action": "update", "index": 7, "data": {"name": "B"}},
                    {"action": "add", "data": {"name": "C", "description": "d", "content": _SUFFICIENT_CONTENT}},
                ]
            }
        )
        fake = FakeLLMClient(responses=[ChatResponse(content=ops, model="fake")])
        result = await AgentSkillExtractor(llm=fake).aextract_with_reason(
            _case(), existing_relevant_skills=[_skill()], supporting_cases=[]
        )

        assert len(result.outcomes) == 3
        assert [o.op_index for o in result.outcomes] == [0, 1, 2]

        dropped, kept = result.dropped, result.skills
        assert len(dropped) + len(kept) == len(result.outcomes)
        assert [o.reason for o in dropped] == [
            SkillSkipReason.ADD_CONTENT_INSUFFICIENT,
            SkillSkipReason.UPDATE_INDEX_OUT_OF_RANGE,
        ]
        assert dropped[1].detail == {"index": 7, "size": 1}
        assert [s.name for s in kept] == ["C"]

    async def test_content_insufficiency_detail_reports_observed_and_required(self) -> None:
        ops = json.dumps(
            {"operations": [{"action": "add", "data": {"name": "A", "description": "d", "content": "one line only"}}]}
        )
        fake = FakeLLMClient(responses=[ChatResponse(content=ops, model="fake")])
        result = await AgentSkillExtractor(llm=fake).aextract_with_reason(
            _case(), existing_relevant_skills=[], supporting_cases=[]
        )
        assert result.outcomes[0].detail == {"lines": 1, "chars": 13, "min_lines": 5, "min_chars": 50}

    async def test_action_none_is_recorded_not_silently_dropped(self) -> None:
        """An all-``none`` response means "the case taught us nothing new" — a real answer, not a void."""
        ops = json.dumps({"operations": [{"action": "none"}, {"action": "none"}]})
        fake = FakeLLMClient(responses=[ChatResponse(content=ops, model="fake")])
        result = await AgentSkillExtractor(llm=fake).aextract_with_reason(
            _case(), existing_relevant_skills=[_skill()], supporting_cases=[]
        )
        assert result.skills == []
        assert [o.reason for o in result.outcomes] == [SkillSkipReason.OP_ACTION_NONE] * 2

    async def test_unknown_action_carries_the_offending_value(self) -> None:
        ops = json.dumps({"operations": [{"action": "delete", "index": 0}]})
        fake = FakeLLMClient(responses=[ChatResponse(content=ops, model="fake")])
        result = await AgentSkillExtractor(llm=fake).aextract_with_reason(
            _case(), existing_relevant_skills=[_skill()], supporting_cases=[]
        )
        assert result.outcomes[0].reason is SkillSkipReason.OP_UNKNOWN_ACTION
        assert result.outcomes[0].detail == {"action": "delete"}

    async def test_duplicate_update_on_same_existing_skill_is_dropped(self) -> None:
        ops = json.dumps(
            {
                "operations": [
                    {"action": "update", "index": 0, "data": {"name": "First rename"}},
                    {"action": "update", "index": 0, "data": {"name": "Second rename"}},
                ]
            }
        )
        fake = FakeLLMClient(responses=[ChatResponse(content=ops, model="fake")])
        result = await AgentSkillExtractor(llm=fake).aextract_with_reason(
            _case(), existing_relevant_skills=[_skill()], supporting_cases=[]
        )
        assert result.outcomes[0].skill is not None
        assert result.outcomes[1].reason is SkillSkipReason.UPDATE_DUPLICATE_INDEX
        assert result.outcomes[1].detail == {"index": 0}

    async def test_op_index_is_not_the_existing_skill_index(self) -> None:
        """``op_index`` counts LLM operations; ``op["index"]`` addresses existing skills. Keep them apart."""
        ops = json.dumps({"operations": [{"action": "update", "index": 1, "data": {"name": "Renamed"}}]})
        fake = FakeLLMClient(responses=[ChatResponse(content=ops, model="fake")])
        result = await AgentSkillExtractor(llm=fake).aextract_with_reason(
            _case(),
            existing_relevant_skills=[_skill(name="zero", skill_id="s0"), _skill(name="one", skill_id="s1")],
            supporting_cases=[],
        )
        outcome = result.outcomes[0]
        assert outcome.op_index == 0  # first (and only) operation...
        assert outcome.skill is not None
        assert outcome.skill.id == "s1"  # ...but it targets the second existing skill


class TestSkillRemainingDropReasons:
    """The malformed-LLM-output drop paths — one trajectory each, all through the public method."""

    @staticmethod
    async def _single_op(
        op: object, existing: list[AgentSkill] | None = None
    ) -> tuple[SkillSkipReason | None, dict[str, Any]]:
        fake = FakeLLMClient(responses=[ChatResponse(content=json.dumps({"operations": [op]}), model="fake")])
        result = await AgentSkillExtractor(llm=fake).aextract_with_reason(
            _case(), existing_relevant_skills=existing or [_skill()], supporting_cases=[]
        )
        assert len(result.outcomes) == 1
        return result.outcomes[0].reason, result.outcomes[0].detail

    async def test_op_not_dict(self) -> None:
        reason, _ = await self._single_op("not an object")
        assert reason is SkillSkipReason.OP_NOT_DICT

    async def test_add_content_empty(self) -> None:
        reason, _ = await self._single_op({"action": "add", "data": {"name": "A", "content": ""}})
        assert reason is SkillSkipReason.ADD_CONTENT_EMPTY

    async def test_add_name_and_desc_empty(self) -> None:
        """Content is fine; the skill is unusable because nothing names or describes it."""
        reason, _ = await self._single_op(
            {"action": "add", "data": {"name": "", "description": "", "content": _SUFFICIENT_CONTENT}}
        )
        assert reason is SkillSkipReason.ADD_NAME_AND_DESC_EMPTY

    async def test_update_index_invalid(self) -> None:
        reason, detail = await self._single_op({"action": "update", "index": "abc", "data": {"name": "X"}})
        assert reason is SkillSkipReason.UPDATE_INDEX_INVALID
        assert detail == {"raw": "'abc'"}

    async def test_update_content_insufficient(self) -> None:
        reason, detail = await self._single_op({"action": "update", "index": 0, "data": {"content": "one liner"}})
        assert reason is SkillSkipReason.UPDATE_CONTENT_INSUFFICIENT
        assert detail == {"lines": 1, "chars": 9, "min_lines": 5, "min_chars": 50}

    async def test_update_no_field_changed(self) -> None:
        reason, detail = await self._single_op({"action": "update", "index": 0, "data": {}})
        assert reason is SkillSkipReason.UPDATE_NO_FIELD_CHANGED
        assert detail == {"index": 0}


class TestSkillWrapperEquivalence:
    async def test_aextract_matches_the_skills_property(self) -> None:
        ops = json.dumps(
            {
                "operations": [
                    {"action": "add", "data": {"name": "Kept", "description": "d", "content": _SUFFICIENT_CONTENT}},
                    {"action": "add", "data": {"name": "Dropped", "description": "d", "content": "short"}},
                ]
            }
        )
        plain = await AgentSkillExtractor(
            llm=FakeLLMClient(responses=[ChatResponse(content=ops, model="fake")])
        ).aextract(_case(), existing_relevant_skills=[], supporting_cases=[])
        rich = await AgentSkillExtractor(
            llm=FakeLLMClient(responses=[ChatResponse(content=ops, model="fake")])
        ).aextract_with_reason(_case(), existing_relevant_skills=[], supporting_cases=[])

        assert [s.name for s in plain] == [s.name for s in rich.skills] == ["Kept"]
