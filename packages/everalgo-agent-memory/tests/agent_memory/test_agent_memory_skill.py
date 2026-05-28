"""Tests for everalgo.agent_memory.skill + skill_ops — AgentSkillExtractor.

Contract differences vs a repo-coupled extractor:

- :class:`everalgo.types.AgentCase` / :class:`everalgo.types.AgentSkill` Pydantic models (no vector fields
  on the schema — caller owns embedding lifecycle)
- ``list[AgentSkill]`` return contract — caller decodes add/update/retire via
  ``id ∈ existing_relevant_skills`` + ``confidence < retire_confidence``
- :class:`everalgo.testing.fake_llm.FakeLLMClient` for deterministic LLM replays
- :class:`_SkillCfg` policy thresholds (no per-call prompt kwargs, no internal top-K)
- Caller pre-filters skills externally before passing in as ``existing_relevant_skills`` —
  ``query_vector`` is absent from the signature

Not covered here: persistence write assertions (EverAlgo returns deltas), embedding calls (caller
embeds), case-history loading (caller fetches), per-call prompt overrides (use module monkey-patch).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from everalgo.agent_memory.skill import AgentSkillExtractor, _SkillCfg
from everalgo.agent_memory.skill_ops import (
    _apply_add,
    _apply_update,
    _content_change_ratio,
    _format_cases,
    _format_existing_skills,
    _is_hypothesis_promotion,
    _is_skill_content_sufficient,
)
from everalgo.llm.types import ChatResponse
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import AgentCase, AgentSkill

# ── Fixture helpers ─────────────────────────────────────────────────────────────────────────────────


_DEFAULT_CONTENT = "## Steps\n1. Design schema\n2. Implement\n3. Test\n4. Deploy\n5. Verify"


def _make_case(
    *,
    case_id: str = "case_001",
    task_intent: str = "Build a REST API",
    approach: str = "1. Design schema\n2. Implement endpoints",
    quality_score: float = 0.8,
    key_insight: str = "",
    timestamp: int = 1700000000000,
) -> AgentCase:
    return AgentCase(
        id=case_id,
        timestamp=timestamp,
        task_intent=task_intent,
        approach=approach,
        quality_score=quality_score,
        key_insight=key_insight,
    )


def _make_skill(
    *,
    skill_id: str = "skill_001",
    cluster_id: str = "cluster_001",
    name: str = "API Development",
    description: str = "Build REST APIs",
    content: str = _DEFAULT_CONTENT,
    confidence: float = 0.7,
    maturity_score: float = 0.7,
    source_case_ids: list[str] | None = None,
) -> AgentSkill:
    return AgentSkill(
        id=skill_id,
        cluster_id=cluster_id,
        name=name,
        description=description,
        content=content,
        confidence=confidence,
        maturity_score=maturity_score,
        source_case_ids=source_case_ids or [],
    )


# ── _format_cases ───────────────────────────────────────────────────────────────────────────────────


class TestFormatCases:
    def test_single_case_round_trips_as_json(self) -> None:
        out = _format_cases([_make_case()])
        parsed: list[dict[str, object]] = json.loads(out)
        assert len(parsed) == 1
        assert parsed[0]["task_intent"] == "Build a REST API"
        assert parsed[0]["quality_score"] == 0.8

    def test_multiple_cases(self) -> None:
        cases = [
            _make_case(task_intent="Task A"),
            _make_case(case_id="case_002", task_intent="Task B", quality_score=0.3),
        ]
        parsed = json.loads(_format_cases(cases))
        assert len(parsed) == 2
        assert parsed[1]["task_intent"] == "Task B"
        assert parsed[1]["quality_score"] == 0.3

    def test_empty_input(self) -> None:
        assert json.loads(_format_cases([])) == []

    def test_key_insight_included_when_set(self) -> None:
        parsed = json.loads(_format_cases([_make_case(key_insight="use indexing")]))
        assert parsed[0]["key_insight"] == "use indexing"

    def test_key_insight_omitted_when_empty(self) -> None:
        parsed = json.loads(_format_cases([_make_case(key_insight="")]))
        assert "key_insight" not in parsed[0]


# ── _format_existing_skills ─────────────────────────────────────────────────────────────────────────


class TestFormatExistingSkills:
    def test_empty_returns_marker(self) -> None:
        assert _format_existing_skills([], [], max_description_tokens=400, max_content_tokens=5000) == (
            "(empty — no existing skills)"
        )

    def test_each_skill_has_index(self) -> None:
        skills = [_make_skill(name="A"), _make_skill(skill_id="skill_002", name="B")]
        out = _format_existing_skills(skills, [], max_description_tokens=400, max_content_tokens=5000)
        parsed = json.loads(out)
        assert parsed[0]["index"] == 0
        assert parsed[1]["index"] == 1
        assert parsed[0]["name"] == "A"
        assert parsed[1]["name"] == "B"

    def test_supporting_cases_attached_when_case_history_matches(self) -> None:
        case = _make_case(case_id="cs_42", task_intent="lookup")
        skill = _make_skill(source_case_ids=["cs_42"])
        out = _format_existing_skills([skill], [case], max_description_tokens=400, max_content_tokens=5000)
        parsed = json.loads(out)
        assert parsed[0]["supporting_case_count"] == 1
        assert parsed[0]["supporting_cases"][0]["task_intent"] == "lookup"

    def test_no_supporting_when_case_history_empty(self) -> None:
        skill = _make_skill(source_case_ids=["nope"])
        parsed = json.loads(_format_existing_skills([skill], [], max_description_tokens=400, max_content_tokens=5000))
        assert "supporting_cases" not in parsed[0]

    def test_max_support_cases_caps_rendered_supporting_cases(self) -> None:
        """``max_support_cases`` bounds rendered supporting cases to the most recent N; count stays full."""
        cases = [_make_case(case_id=f"cs_{i}", task_intent=f"task {i}") for i in range(5)]
        skill = _make_skill(source_case_ids=[c.id for c in cases])
        parsed = json.loads(
            _format_existing_skills(
                [skill], cases, max_description_tokens=400, max_content_tokens=5000, max_support_cases=2
            )
        )
        # All 5 matched, but only the last 2 (cs_3, cs_4) are rendered.
        assert parsed[0]["supporting_case_count"] == 5
        rendered = [c["task_intent"] for c in parsed[0]["supporting_cases"]]
        assert rendered == ["task 3", "task 4"]


# ── _is_skill_content_sufficient ────────────────────────────────────────────────────────────────────


class TestIsSkillContentSufficient:
    def test_empty(self) -> None:
        assert _is_skill_content_sufficient("") is False

    def test_too_short(self) -> None:
        assert _is_skill_content_sufficient("short") is False

    def test_too_few_lines(self) -> None:
        assert _is_skill_content_sufficient("A" * 100 + "\n" + "B" * 100) is False

    def test_sufficient(self) -> None:
        text = "## Steps\n1. First\n2. Second\n3. Third\n4. Fourth\n5. Fifth"
        assert _is_skill_content_sufficient(text) is True

    def test_whitespace_only_lines_ignored(self) -> None:
        assert _is_skill_content_sufficient("a\n\n\nb\n\n\nc\n   \n") is False

    def test_custom_thresholds(self) -> None:
        assert _is_skill_content_sufficient("ab\ncd\nef", min_lines=3, min_length=5) is True
        assert _is_skill_content_sufficient("ab\ncd\nef", min_lines=3, min_length=100) is False


# ── _content_change_ratio ───────────────────────────────────────────────────────────────────────────


class TestContentChangeRatio:
    def test_identical_returns_zero(self) -> None:
        assert _content_change_ratio("hello world", "hello world") == 0.0

    def test_completely_different_returns_one(self) -> None:
        assert _content_change_ratio("aaaa", "bbbb") > 0.9

    def test_minor_edit_low_ratio(self) -> None:
        assert _content_change_ratio("hello world", "hello world!") < 0.2

    def test_empty_both_zero(self) -> None:
        assert _content_change_ratio("", "") == 0.0

    def test_empty_one_side_returns_one(self) -> None:
        assert _content_change_ratio("x", "") == 1.0
        assert _content_change_ratio("", "x") == 1.0


# ── _is_hypothesis_promotion ────────────────────────────────────────────────────────────────────────


class TestIsHypothesisPromotion:
    def test_classic_promotion_potential_to_steps(self) -> None:
        old = "## Potential Steps\n1. Try X\n\n## Pitfalls\n- Y failed"
        new = "## Steps\n1. Do X\n2. Do Y\n\n## Pitfalls\n- Y failed"
        assert _is_hypothesis_promotion(old, new) is True

    def test_steps_to_steps_not_promotion(self) -> None:
        assert _is_hypothesis_promotion("## Steps\n1. X", "## Steps\n1. X\n2. Y") is False

    def test_potential_to_potential_not_promotion(self) -> None:
        assert (
            _is_hypothesis_promotion("## Potential Steps\n1. Try X", "## Potential Steps\n1. Try X\n2. Try Y") is False
        )

    def test_new_has_both_potential_and_steps_not_promotion(self) -> None:
        old = "## Potential Steps\n1. Try X"
        new = "## Steps\n1. Do X\n\n## Potential Steps\n1. Try Y"
        assert _is_hypothesis_promotion(old, new) is False

    def test_empty_old_not_promotion(self) -> None:
        assert _is_hypothesis_promotion("", "## Steps\n1. X") is False

    def test_empty_new_not_promotion(self) -> None:
        assert _is_hypothesis_promotion("## Potential Steps\n1. X", "") is False

    def test_case_sensitive_heading(self) -> None:
        assert _is_hypothesis_promotion("## Potential Steps\n1. X", "## steps\n1. X") is False

    def test_extra_whitespace_in_heading_still_matches(self) -> None:
        old = "##  Potential Steps\n1. Try X"
        new = "##  Steps\n1. Do X"
        assert _is_hypothesis_promotion(old, new) is True


# ── _apply_add ──────────────────────────────────────────────────────────────────────────────────────


class TestApplyAdd:
    async def test_success_returns_new_skill(self) -> None:
        op = {
            "action": "add",
            "data": {
                "name": "New Skill",
                "description": "Does things",
                "content": _DEFAULT_CONTENT,
                "confidence": 0.6,
            },
        }
        cfg = _SkillCfg()  # skip_maturity_scoring=True by default → no LLM call needed
        added = await _apply_add(op, ["case_x"], client=FakeLLMClient(responses=[]), cfg=cfg)
        assert added is not None
        assert added.name == "New Skill"
        assert added.confidence == 0.6
        assert added.source_case_ids == ["case_x"]
        # AgentSkill no longer carries vector fields on the schema
        assert "vector" not in added.model_dump()
        # skip_maturity_scoring → maturity = 1.0
        assert added.maturity_score == 1.0

    async def test_empty_content_skipped(self) -> None:
        op = {"action": "add", "data": {"name": "X", "description": "Y", "content": ""}}
        result = await _apply_add(op, [], client=FakeLLMClient(responses=[]), cfg=_SkillCfg())
        assert result is None

    async def test_insufficient_content_skipped(self) -> None:
        op = {"action": "add", "data": {"name": "X", "description": "Y", "content": "too short"}}
        result = await _apply_add(op, [], client=FakeLLMClient(responses=[]), cfg=_SkillCfg())
        assert result is None

    async def test_no_name_and_no_description_skipped(self) -> None:
        op = {"action": "add", "data": {"name": "", "description": "", "content": _DEFAULT_CONTENT}}
        result = await _apply_add(op, [], client=FakeLLMClient(responses=[]), cfg=_SkillCfg())
        assert result is None

    async def test_invalid_confidence_defaults_to_half(self) -> None:
        op = {
            "action": "add",
            "data": {
                "name": "X",
                "description": "Y",
                "content": _DEFAULT_CONTENT,
                "confidence": "not-a-number",
            },
        }
        result = await _apply_add(op, [], client=FakeLLMClient(responses=[]), cfg=_SkillCfg())
        assert result is not None
        assert result.confidence == 0.5

    async def test_maturity_evaluation_invoked_when_opted_in(self) -> None:
        """When skip_maturity_scoring=False, the LLM is called once for maturity scoring."""
        maturity_json = json.dumps({"completeness": 4, "executability": 4, "evidence": 3, "clarity": 4, "reason": ""})
        fake = FakeLLMClient(responses=[ChatResponse(content=maturity_json, model="fake")])
        op = {
            "action": "add",
            "data": {
                "name": "X",
                "description": "Y",
                "content": _DEFAULT_CONTENT,
                "confidence": 0.5,
            },
        }
        result = await _apply_add(op, [], client=fake, cfg=_SkillCfg(skip_maturity_scoring=False))
        assert result is not None
        assert fake.call_count == 1
        # raw_total = 15 / 20 = 0.75
        assert abs(result.maturity_score - 0.75) < 1e-6

    async def test_maturity_llm_exception_propagates(self) -> None:
        """LLM raises → exception propagates from _evaluate_maturity (no swallow)."""

        def raise_llm_error(*_args: object, **_kwargs: object) -> ChatResponse:
            raise RuntimeError("simulated LLM provider failure (e.g. network timeout)")

        fake = FakeLLMClient(handler=raise_llm_error)
        op = {
            "action": "add",
            "data": {
                "name": "X",
                "description": "Y",
                "content": _DEFAULT_CONTENT,
                "confidence": 0.7,
            },
        }
        with pytest.raises(RuntimeError, match="simulated LLM provider failure"):
            await _apply_add(op, [], client=fake, cfg=_SkillCfg(skip_maturity_scoring=False))

    async def test_invalid_index_type_returns_none(self) -> None:
        result = await _apply_update(
            {"index": "abc", "data": {"name": "X"}},
            [_make_skill()],
            source_case_ids=[],
            source_quality=0.5,
            client=FakeLLMClient(responses=[]),
            cfg=_SkillCfg(),
            processed_indices=set(),
        )
        assert result is None

    async def test_out_of_range_index_returns_none(self) -> None:
        result = await _apply_update(
            {"index": 5, "data": {"name": "X"}},
            [_make_skill()],
            source_case_ids=[],
            source_quality=0.5,
            client=FakeLLMClient(responses=[]),
            cfg=_SkillCfg(),
            processed_indices=set(),
        )
        assert result is None

    async def test_negative_index_returns_none(self) -> None:
        result = await _apply_update(
            {"index": -1, "data": {"name": "X"}},
            [_make_skill()],
            source_case_ids=[],
            source_quality=0.5,
            client=FakeLLMClient(responses=[]),
            cfg=_SkillCfg(),
            processed_indices=set(),
        )
        assert result is None

    async def test_duplicate_index_returns_none(self) -> None:
        processed: set[int] = {0}
        result = await _apply_update(
            {"index": 0, "data": {"name": "X"}},
            [_make_skill()],
            source_case_ids=[],
            source_quality=0.5,
            client=FakeLLMClient(responses=[]),
            cfg=_SkillCfg(),
            processed_indices=processed,
        )
        assert result is None

    async def test_no_fields_to_change_returns_none(self) -> None:
        result = await _apply_update(
            {"index": 0, "data": {}},
            [_make_skill()],
            source_case_ids=[],
            source_quality=0.5,
            client=FakeLLMClient(responses=[]),
            cfg=_SkillCfg(),
            processed_indices=set(),
        )
        assert result is None

    async def test_insufficient_new_content_returns_none(self) -> None:
        result = await _apply_update(
            {"index": 0, "data": {"content": "too short"}},
            [_make_skill()],
            source_case_ids=[],
            source_quality=0.5,
            client=FakeLLMClient(responses=[]),
            cfg=_SkillCfg(),
            processed_indices=set(),
        )
        assert result is None

    async def test_retire_branch_low_confidence(self) -> None:
        """Confidence < retire_confidence (0.1) → returned with confidence dropped (caller soft-deletes)."""
        prior = _make_skill(confidence=0.7)
        cfg = _SkillCfg()
        result = await _apply_update(
            {"index": 0, "data": {"confidence": 0.05}},
            [prior],
            source_case_ids=["case_x"],
            source_quality=0.4,
            client=FakeLLMClient(responses=[]),
            cfg=cfg,
            processed_indices=set(),
        )
        assert result is not None
        assert result.id == prior.id
        assert result.confidence == 0.05  # caller decodes RETIRE via this < cfg.retire_confidence
        assert "case_x" in result.source_case_ids

    async def test_update_keeps_prior_id_and_propagates_fields(self) -> None:
        prior = _make_skill(name="Old", description="Old desc")
        cfg = _SkillCfg()
        result = await _apply_update(
            {"index": 0, "data": {"name": "New Name", "description": "New description text content"}},
            [prior],
            source_case_ids=[],
            source_quality=0.5,
            client=FakeLLMClient(responses=[]),
            cfg=cfg,
            processed_indices=set(),
        )
        assert result is not None
        assert result.id == prior.id  # caller decodes UPDATE via id ∈ existing + confidence >= retire
        assert result.name == "New Name"
        assert "New description" in result.description

    async def test_content_only_change(self) -> None:
        prior = _make_skill(content="## Old\n1. a\n2. b\n3. c\n4. d\n5. e")
        new_content = "## Steps\n1. New step 1\n2. New step 2\n3. Step 3\n4. Step 4\n5. Step 5"
        cfg = _SkillCfg()
        result = await _apply_update(
            {"index": 0, "data": {"content": new_content}},
            [prior],
            source_case_ids=[],
            source_quality=0.5,
            client=FakeLLMClient(responses=[]),
            cfg=cfg,
            processed_indices=set(),
        )
        assert result is not None
        assert result.content == new_content

    async def test_source_case_ids_appended_dedup(self) -> None:
        prior = _make_skill(source_case_ids=["case_a"])
        cfg = _SkillCfg()
        result = await _apply_update(
            {"index": 0, "data": {"confidence": 0.8}},
            [prior],
            source_case_ids=["case_a", "case_b"],
            source_quality=0.5,
            client=FakeLLMClient(responses=[]),
            cfg=cfg,
            processed_indices=set(),
        )
        assert result is not None
        assert result.source_case_ids == ["case_a", "case_b"]

    async def test_invalid_confidence_string_no_other_fields_returns_none(self) -> None:
        """Bug 2 fix: op with only an invalid ``confidence`` (no name/desc/content change) returns None.

        Invalid confidence is treated as "no confidence change" → no-fields-to-change guard fires →
        returns None instead of emitting a no-op update.
        """
        prior = _make_skill(confidence=0.7)
        result = await _apply_update(
            {"index": 0, "data": {"confidence": "not-a-number"}},  # only field, and it's invalid
            [prior],
            source_case_ids=[],
            source_quality=0.5,
            client=FakeLLMClient(responses=[]),
            cfg=_SkillCfg(),
            processed_indices=set(),
        )
        assert result is None  # before the fix: would have emitted a no-op update with prior.confidence

    async def test_invalid_confidence_with_real_name_change_still_updates(self) -> None:
        """Bug 2 fix: invalid confidence + real name change still emits an update.

        Confidence stays at prior.confidence, but the name change is what triggers the update path.
        """
        prior = _make_skill(name="Old", confidence=0.7)
        result = await _apply_update(
            {"index": 0, "data": {"name": "New Name", "confidence": "garbage"}},
            [prior],
            source_case_ids=[],
            source_quality=0.5,
            client=FakeLLMClient(responses=[]),
            cfg=_SkillCfg(),
            processed_indices=set(),
        )
        assert result is not None
        assert result.name == "New Name"
        assert result.confidence == 0.7  # invalid confidence ignored, prior preserved

    async def test_invalid_confidence_none_type_also_treated_as_no_change(self) -> None:
        """Bug 2 fix coverage: a None-typed confidence is treated as no confidence change.

        Sloppy LLMs may emit ``{"confidence": None}`` literally; this branch triggers the TypeError
        path and is treated as no confidence change.
        """
        prior = _make_skill(confidence=0.7)
        result = await _apply_update(
            {"index": 0, "data": {"confidence": None}},
            [prior],
            source_case_ids=[],
            source_quality=0.5,
            client=FakeLLMClient(responses=[]),
            cfg=_SkillCfg(),
            processed_indices=set(),
        )
        # data.get("confidence") returns None, and `new_confidence_raw is not None` is False → the
        # pre-parse branch never runs → confidence_changed=False → no-fields guard fires
        assert result is None

    async def test_maturity_llm_exception_in_update_propagates(self) -> None:
        """Maturity re-eval LLM raises → exception propagates from _apply_update.

        Major content change triggers re-eval; LLM raises; error propagates up.
        """

        def raise_llm_error(*_args: object, **_kwargs: object) -> ChatResponse:
            raise RuntimeError("simulated LLM provider failure")

        # Prior has substantive content + maturity_score; we force a "major content change" (>= 0.4
        # change ratio) so the algorithm tries to re-evaluate maturity via LLM.
        prior = _make_skill(
            content="## Steps\n1. Old step one\n2. Old step two\n3. Old step three\n4. Old step four\n5. Old",
            maturity_score=0.55,
            confidence=0.7,
        )
        new_content = (
            "## Steps\n1. Brand new procedure A\n2. Brand new procedure B\n"
            "3. Brand new procedure C\n4. Brand new procedure D\n5. Brand new procedure E"
        )
        fake = FakeLLMClient(handler=raise_llm_error)
        with pytest.raises(RuntimeError, match="simulated LLM provider failure"):
            await _apply_update(
                {"index": 0, "data": {"content": new_content}},
                [prior],
                source_case_ids=[],
                source_quality=0.8,
                client=fake,
                cfg=_SkillCfg(skip_maturity_scoring=False),  # opt into LLM maturity scoring
                processed_indices=set(),
            )


# ── End-to-end AgentSkillExtractor.aextract ─────────────────────────────────────────────────────────


class TestAgentSkillExtractorAExtract:
    async def test_no_operations_returns_empty(self) -> None:
        fake = FakeLLMClient(responses=[ChatResponse(content='{"operations": [], "update_note": ""}', model="fake")])
        result = await AgentSkillExtractor(llm=fake).aextract(
            _make_case(),
            existing_relevant_skills=[],
            supporting_cases=[],
        )
        assert result == []

    async def test_add_op_emits_new_skill(self) -> None:
        ops_response = json.dumps(
            {
                "operations": [
                    {
                        "action": "add",
                        "data": {
                            "name": "API skill",
                            "description": "Build REST APIs",
                            "content": _DEFAULT_CONTENT,
                            "confidence": 0.5,
                        },
                    }
                ],
                "update_note": "",
            }
        )
        fake = FakeLLMClient(responses=[ChatResponse(content=ops_response, model="fake")])
        raw_result = await AgentSkillExtractor(llm=fake).aextract(
            _make_case(),
            existing_relevant_skills=[],
            supporting_cases=[],
        )
        # Caller stamps cluster_id
        result = [s.model_copy(update={"cluster_id": "cl_001"}) for s in raw_result]
        assert len(result) == 1
        skill = result[0]
        assert skill.name == "API skill"
        assert skill.cluster_id == "cl_001"
        assert skill.source_case_ids == ["case_001"]

    async def test_quality_score_routes_to_failure_prompt(self) -> None:
        """case.quality_score=0.3 (< failure_quality_threshold=0.5) → failure prompt used."""
        fake = FakeLLMClient(responses=[ChatResponse(content='{"operations": []}', model="fake")])
        await AgentSkillExtractor(llm=fake).aextract(
            _make_case(quality_score=0.3),
            existing_relevant_skills=[],
            supporting_cases=[],
            prompt_failure="FAILURE_PROMPT_MARKER_{new_case_json}{existing_skills_json}",
            prompt_success="SUCCESS_PROMPT_MARKER_{new_case_json}{existing_skills_json}",
        )
        assert "FAILURE_PROMPT_MARKER_" in fake.calls[0].messages[0].content
        assert "SUCCESS_PROMPT_MARKER_" not in fake.calls[0].messages[0].content

    async def test_quality_score_routes_to_success_prompt_at_threshold(self) -> None:
        """case.quality_score=0.5 (= threshold) → success prompt used."""
        fake = FakeLLMClient(responses=[ChatResponse(content='{"operations": []}', model="fake")])
        await AgentSkillExtractor(llm=fake).aextract(
            _make_case(quality_score=0.5),
            existing_relevant_skills=[],
            supporting_cases=[],
            prompt_failure="FAILURE_PROMPT_MARKER_{new_case_json}{existing_skills_json}",
            prompt_success="SUCCESS_PROMPT_MARKER_{new_case_json}{existing_skills_json}",
        )
        assert "SUCCESS_PROMPT_MARKER_" in fake.calls[0].messages[0].content

    async def test_monkey_patch_prompt_override_still_works(self) -> None:
        """Module-constant monkey-patch is an alternative to per-call kwargs (startup-time override)."""
        import everalgo.agent_memory.skill as skill_mod

        original = skill_mod.AGENT_SKILL_SUCCESS_EXTRACT_PROMPT
        skill_mod.AGENT_SKILL_SUCCESS_EXTRACT_PROMPT = "MONKEY_SUCCESS_{new_case_json}{existing_skills_json}"
        try:
            fake = FakeLLMClient(responses=[ChatResponse(content='{"operations": []}', model="fake")])
            await AgentSkillExtractor(llm=fake).aextract(
                _make_case(),  # quality_score=0.8 → success path
                existing_relevant_skills=[],
                supporting_cases=[],
            )
            assert "MONKEY_SUCCESS_" in fake.calls[0].messages[0].content
        finally:
            skill_mod.AGENT_SKILL_SUCCESS_EXTRACT_PROMPT = original

    async def test_update_retire_encoded_in_returned_list(self) -> None:
        """End-to-end: LLM emits one update with low confidence → retire encoding in returned skill."""
        prior = _make_skill(skill_id="sk_existing", confidence=0.6)
        ops_response = json.dumps(
            {
                "operations": [{"action": "update", "index": 0, "data": {"confidence": 0.05}}],
                "update_note": "",
            }
        )
        fake = FakeLLMClient(responses=[ChatResponse(content=ops_response, model="fake")])
        result = await AgentSkillExtractor(llm=fake).aextract(
            _make_case(),
            existing_relevant_skills=[prior],
            supporting_cases=[],
        )
        assert len(result) == 1
        retired = result[0]
        existing_by_id = {prior.id: prior}
        assert retired.id in existing_by_id
        assert retired.confidence < 0.1  # default retire_confidence

    async def test_existing_relevant_skills_all_passed_through(self) -> None:
        """All caller-supplied skills appear in the prompt — algorithm no longer caps them internally."""
        existing = [_make_skill(skill_id=f"sk_{i}", name=f"NAME_MARKER_{i}") for i in range(5)]
        fake = FakeLLMClient(responses=[ChatResponse(content='{"operations": []}', model="fake")])
        await AgentSkillExtractor(llm=fake).aextract(
            _make_case(),
            existing_relevant_skills=existing,
            supporting_cases=[],
        )
        prompt_content = fake.calls[0].messages[0].content
        # All 5 skills' names should appear — algorithm doesn't filter
        for i in range(5):
            assert f"NAME_MARKER_{i}" in prompt_content

    async def test_update_note_logged_when_present(self) -> None:
        """skill.py: when ``update_note`` is non-empty the debug log fires (no behaviour change)."""
        ops_response = json.dumps(
            {
                "operations": [{"action": "none"}],
                "update_note": "explanation of the decision",
            }
        )
        fake = FakeLLMClient(responses=[ChatResponse(content=ops_response, model="fake")])
        # Just exercise the path; logger.debug fires but doesn't change the return value
        result = await AgentSkillExtractor(llm=fake).aextract(
            _make_case(),
            existing_relevant_skills=[],
            supporting_cases=[],
        )
        assert result == []

    async def test_apply_add_with_non_dict_data_skipped(self) -> None:
        """Non-dict ``op["data"]`` falls back to ``{}`` -> empty content -> ``_apply_add`` skips."""
        op = {"action": "add", "data": "not a dict"}  # ← non-dict data
        result = await _apply_add(op, [], client=FakeLLMClient(responses=[]), cfg=_SkillCfg())
        assert result is None  # empty content from the {} fallback → skip


class TestSummariseCaseForPrompt:
    """skill_ops.py:87-97 — approach-attached branch coverage."""

    def test_case_with_approach_includes_truncated_field(self) -> None:
        case = _make_case(approach="step1 then step2 then step3", key_insight="")
        # _summarise_case_for_prompt is not exported, but we exercise it indirectly via _format_existing_skills
        skill = _make_skill(source_case_ids=[case.id])
        out = _format_existing_skills([skill], [case], max_description_tokens=400, max_content_tokens=5000)
        parsed = json.loads(out)
        # supporting_cases includes the approach field for this case
        assert "approach" in parsed[0]["supporting_cases"][0]
        assert parsed[0]["supporting_cases"][0]["approach"].startswith("step1")

    def test_case_with_empty_approach_omits_approach_field(self) -> None:
        """When ``case.approach`` is empty, the ``if case_record.approach`` branch is False (95→97)."""
        case = _make_case(approach="", key_insight="insight x")
        skill = _make_skill(source_case_ids=[case.id])
        out = _format_existing_skills([skill], [case], max_description_tokens=400, max_content_tokens=5000)
        parsed = json.loads(out)
        # supporting_cases[0] has key_insight but no approach (empty approach was skipped)
        supporting = parsed[0]["supporting_cases"][0]
        assert "approach" not in supporting
        assert supporting["key_insight"] == "insight x"


class TestFormatExistingSkillsEdgeBranches:
    def test_supporting_cases_entry_with_empty_id_skipped(self) -> None:
        """skill_ops.py:120-122 ``if cid:`` False branch — case with empty id doesn't enter case_map.

        This forces the ``case_map`` to stay empty even though ``supporting_cases`` is non-empty, which then
        exercises the ``if case_map:`` False branch on line 134.
        """
        case_no_id = AgentCase(
            id="",  # ← empty id triggers the `if cid:` False branch
            timestamp=1700000000000,
            task_intent="x",
        )
        skill = _make_skill(source_case_ids=["whatever"])
        out = _format_existing_skills([skill], [case_no_id], max_description_tokens=400, max_content_tokens=5000)
        parsed = json.loads(out)
        # No supporting_cases since case_map ended up empty
        assert "supporting_cases" not in parsed[0]

    def test_skill_with_unmatched_source_case_ids(self) -> None:
        """No ``source_case_ids`` overlap with ``supporting_cases`` -> no supporting_cases (skill_ops.py:136-137)."""
        # supporting_cases has one case with id "cs_X"; skill references "cs_Y" — no overlap
        unrelated_case = _make_case(case_id="cs_X")
        skill_referring_other = _make_skill(source_case_ids=["cs_Y"])
        out = _format_existing_skills(
            [skill_referring_other], [unrelated_case], max_description_tokens=400, max_content_tokens=5000
        )
        parsed = json.loads(out)
        # case_map is non-empty (has cs_X) but no matched_ids → no supporting_cases
        assert "supporting_cases" not in parsed[0]
        assert "supporting_case_count" not in parsed[0]


# ── _apply_update — maturity moderate-band three sub-branches (skill_ops.py:471-514) ────────────────


class TestApplyUpdateModerateMaturityBand:
    """Cover the three sub-branches in the moderate band (change_ratio in [0.2, 0.4))."""

    @staticmethod
    def _build_moderate_change_op(*, name: str = "API skill", new_name: str | None = None) -> dict[str, Any]:
        """Build an update op whose content change produces a ratio in [0.2, 0.4)."""
        op_data: dict[str, Any] = {
            "content": (
                "## Steps\n1. Old step one EDITED\n2. New entry here\n3. Old step three\n"
                "4. Step four with edits\n5. Last step"
            ),
        }
        if new_name:
            op_data["name"] = new_name
        return {"action": "update", "index": 0, "data": op_data}

    async def test_moderate_change_already_mature_stable_skips_reeval(self) -> None:
        """Moderate change + mature + stable confidence -> skip re-eval (skill_ops.py:475-485)."""
        prior = _make_skill(
            content=("## Steps\n1. Old step one\n2. Original entry\n3. Old step three\n4. Original step four\n5. Last"),
            maturity_score=0.85,  # ≥ threshold (0.6)
            confidence=0.7,
        )
        op = self._build_moderate_change_op()
        fake = FakeLLMClient(responses=[])  # must NOT be called — reeval is skipped
        result = await _apply_update(
            op,
            [prior],
            source_case_ids=["c1"],
            source_quality=0.8,
            client=fake,
            cfg=_SkillCfg(skip_maturity_scoring=False),  # opt into LLM but skip should bypass
            processed_indices=set(),
        )
        assert result is not None
        # No LLM call happened — moderate + mature + stable skipped re-eval
        assert fake.call_count == 0
        # Maturity preserved from prior
        assert result.maturity_score == 0.85

    async def test_moderate_change_immature_low_source_quality_skips_reeval(self) -> None:
        """Moderate change + immature prior + ``source_quality < 0.3`` -> skip re-eval (skill_ops.py:486-494)."""
        prior = _make_skill(
            content=("## Steps\n1. Old step one\n2. Original entry\n3. Old step three\n4. Original step four\n5. Last"),
            maturity_score=0.4,  # < threshold (0.6)
            confidence=0.6,
        )
        op = self._build_moderate_change_op()
        fake = FakeLLMClient(responses=[])
        result = await _apply_update(
            op,
            [prior],
            source_case_ids=["c1"],
            source_quality=0.2,  # < 0.3 → skip re-eval
            client=fake,
            cfg=_SkillCfg(skip_maturity_scoring=False),
            processed_indices=set(),
        )
        assert result is not None
        assert fake.call_count == 0  # re-eval skipped
        assert result.maturity_score == 0.4  # preserved

    async def test_moderate_change_otherwise_reeval_via_llm(self) -> None:
        """Moderate change + neither skip-branch -> LLM re-eval fires (skill_ops.py:495-514)."""
        prior = _make_skill(
            content=("## Steps\n1. Old step one\n2. Original entry\n3. Old step three\n4. Original step four\n5. Last"),
            maturity_score=0.4,  # < threshold
            confidence=0.6,
        )
        op = self._build_moderate_change_op()
        maturity_json = json.dumps({"completeness": 5, "executability": 5, "evidence": 5, "clarity": 5, "reason": "ok"})
        fake = FakeLLMClient(responses=[ChatResponse(content=maturity_json, model="fake")])
        result = await _apply_update(
            op,
            [prior],
            source_case_ids=["c1"],
            source_quality=0.6,  # ≥ 0.3 → does NOT skip; immature → re-eval fires
            client=fake,
            cfg=_SkillCfg(skip_maturity_scoring=False),
            processed_indices=set(),
        )
        assert result is not None
        # LLM was called for maturity re-eval
        assert fake.call_count == 1
        # maturity_score updated from LLM: 20/20 = 1.0
        assert abs(result.maturity_score - 1.0) < 1e-6

    async def test_moderate_mature_confidence_dropping_to_low_triggers_reeval(self) -> None:
        """Mature prior + confidence dropping to < 0.5 -> falls through to re-eval (skill_ops.py:475)."""
        prior = _make_skill(
            content=("## Steps\n1. Old step one\n2. Original entry\n3. Old step three\n4. Original step four\n5. Last"),
            maturity_score=0.85,  # ≥ threshold (mature)
            confidence=0.7,  # high prior confidence
        )
        # New confidence drops to 0.3 — dropping AND < 0.5 → "not (not dropping or ≥ 0.5)" → re-eval
        op = self._build_moderate_change_op()
        op["data"]["confidence"] = 0.3
        maturity_json = json.dumps({"completeness": 3, "executability": 3, "evidence": 3, "clarity": 3, "reason": "ok"})
        fake = FakeLLMClient(responses=[ChatResponse(content=maturity_json, model="fake")])
        result = await _apply_update(
            op,
            [prior],
            source_case_ids=["c1"],
            source_quality=0.6,  # ≥ 0.3 → does NOT short-circuit via the low-quality branch
            client=fake,
            cfg=_SkillCfg(skip_maturity_scoring=False),
            processed_indices=set(),
        )
        assert result is not None
        # LLM was called — confidence dropping to <0.5 forces re-eval
        assert fake.call_count == 1
        # maturity_score updated: 12/20 = 0.6
        assert abs(result.maturity_score - 0.6) < 1e-6


# ── skip_quality_threshold short-circuit ────────────────────────────────────────────────────────────


class TestSkipQualityThreshold:
    async def test_skip_when_quality_below_threshold(self) -> None:
        """case.quality_score < 0.2 short-circuits to [] without calling the LLM."""
        fake = FakeLLMClient(responses=[])  # no responses; any LLM call would raise StopIteration
        case = _make_case(quality_score=0.1)
        result = await AgentSkillExtractor(llm=fake).aextract(
            case,
            existing_relevant_skills=[],
            supporting_cases=[],
        )
        assert result == []
        assert len(fake.calls) == 0  # critical: no LLM call

    async def test_no_skip_when_quality_equals_threshold(self) -> None:
        """case.quality_score == 0.2 does not short-circuit (strict ``<`` comparison)."""
        fake = FakeLLMClient(responses=[ChatResponse(content='{"operations": [], "update_note": ""}', model="fake")])
        case = _make_case(quality_score=0.2)
        await AgentSkillExtractor(llm=fake).aextract(
            case,
            existing_relevant_skills=[],
            supporting_cases=[],
        )
        assert len(fake.calls) >= 1  # LLM was called

    async def test_caller_override_skip_threshold(self) -> None:
        """Passing skip_quality_threshold=0.0 prevents short-circuit for low-quality cases."""
        fake = FakeLLMClient(responses=[ChatResponse(content='{"operations": [], "update_note": ""}', model="fake")])
        case = _make_case(quality_score=0.1)
        await AgentSkillExtractor(llm=fake).aextract(
            case,
            existing_relevant_skills=[],
            supporting_cases=[],
            skip_quality_threshold=0.0,
        )
        assert len(fake.calls) >= 1  # LLM was called despite low quality_score
