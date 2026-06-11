"""Tests for everalgo.agent_memory.profile — AgentProfileExtractor.

The operator is precision-first (default noop), so coverage is negative-sample-heavy: every gate has
a dedicated drop test, plus structural-validation drops on the proposed patches. Positive paths
assert the exact section-level placement, unified diff shape, and the is_conflict flag. One LLM
call total; structural noops make zero calls.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from everalgo.agent_memory.profile import (
    AgentProfileExtractor,
    _insert_under_section,
    _unified_diff,
)
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import AgentProfileSignal, ChatMessage, MemCell

# ── Fixture helpers ─────────────────────────────────────────────────────────────────────────────────


SOUL_MD = """# Soul

I am EverClaw, a personal AI assistant.

## Personality

- Helpful and friendly
- Curious and eager to learn

## Communication Style

- Be clear and direct
- Explain reasoning when helpful
"""

AGENTS_MD = """# Agent Instructions

You are a helpful AI assistant.

## Operating Rules

- Ask before deleting files
- Keep commit messages in English
"""


def _user(text: str, *, ts: int = 1700000000000, msg_id: str = "u1") -> ChatMessage:
    return ChatMessage(kind="text", id=msg_id, role="user", content=text, timestamp=ts, sender_id="user_1")


def _assistant(text: str, *, ts: int = 1700000001000, msg_id: str = "a1") -> ChatMessage:
    return ChatMessage(kind="text", id=msg_id, role="assistant", content=text, timestamp=ts, sender_id="agent")


def _memcell(*items: ChatMessage) -> MemCell:
    return MemCell(items=list(items), timestamp=items[-1].timestamp if items else 1700000000000)


def _llm_response(candidates: list[dict[str, Any]]) -> str:
    return json.dumps({"candidates": candidates})


def _candidate(
    *,
    target: str = "agents",
    signal: str = "Never auto-commit after code edits",
    evidence: str = "以后改完代码不要自动 commit",
    speech_act: str = "directive",
    persistence: str = "explicit",
    directed_at: str = "agent",
    novelty: str = "new",
    key: str = "no-auto-commit",
    matched_pending_key: str | None = None,
    patch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "target": target,
        "signal": signal,
        "key": key,
        "evidence": evidence,
        "speech_act": speech_act,
        "persistence": persistence,
        "directed_at": directed_at,
        "novelty": novelty,
        "conflict_excerpt": None,
        "matched_pending_key": matched_pending_key,
        "reason": "explicit durable instruction",
        "patch": patch,
    }


_EXPLICIT_USER_TEXT = "以后改完代码不要自动 commit"

_ADD_RULE_PATCH: dict[str, Any] = {
    "action": "add",
    "section": "Operating Rules",
    "old_text": "",
    "new_text": "- Never auto-commit after editing code",
}


# ── Structural noop paths (zero LLM calls) ──────────────────────────────────────────────────────────


class TestStructuralNoop:
    async def test_empty_memcell_is_noop_without_llm_call(self) -> None:
        llm = FakeLLMClient(responses=[])
        extractor = AgentProfileExtractor(llm=llm)
        update = await extractor.aextract(
            MemCell(items=[], timestamp=1700000000000), soul_md=SOUL_MD, agents_md=AGENTS_MD
        )
        assert update.patches == []
        assert update.soul_diff == "" and update.agents_diff == ""
        assert update.new_soul_md == SOUL_MD and update.new_agents_md == AGENTS_MD
        assert llm.call_count == 0

    async def test_empty_content_user_message_is_noop_without_llm_call(self) -> None:
        llm = FakeLLMClient(responses=[])
        extractor = AgentProfileExtractor(llm=llm)
        update = await extractor.aextract(_memcell(_user("")), soul_md=SOUL_MD, agents_md=AGENTS_MD)
        assert update.patches == []
        assert llm.call_count == 0

    async def test_assistant_only_memcell_is_noop_without_llm_call(self) -> None:
        llm = FakeLLMClient(responses=[])
        extractor = AgentProfileExtractor(llm=llm)
        update = await extractor.aextract(
            _memcell(_assistant("我帮你看看这个问题。")), soul_md=SOUL_MD, agents_md=AGENTS_MD
        )
        assert update.patches == []
        assert llm.call_count == 0

    async def test_assistant_turns_rendered_as_context_in_prompt(self) -> None:
        # Assistant text turns are part of the conversation view (context); authority checks are
        # covered by test_evidence_from_assistant_message_dropped.
        llm = FakeLLMClient(responses=[_llm_response([])])
        extractor = AgentProfileExtractor(llm=llm)
        await extractor.aextract(
            _memcell(_user("帮我查个东西"), _assistant("好的, 这是结果。")), soul_md=SOUL_MD, agents_md=AGENTS_MD
        )
        prompt_text = llm.calls[0].messages[0].content
        assert "帮我查个东西" in prompt_text
        assert "这是结果" in prompt_text


# ── Gate drops (one LLM call, no patch emitted) ─────────────────────────────────────────────────────


class TestGateDrops:
    async def _run_single_candidate(self, candidate: dict[str, Any], user_text: str) -> Any:
        llm = FakeLLMClient(responses=[_llm_response([candidate])])
        extractor = AgentProfileExtractor(llm=llm)
        return await extractor.aextract(_memcell(_user(user_text)), soul_md=SOUL_MD, agents_md=AGENTS_MD)

    async def test_no_candidates_is_noop(self) -> None:
        llm = FakeLLMClient(responses=[_llm_response([])])
        extractor = AgentProfileExtractor(llm=llm)
        update = await extractor.aextract(_memcell(_user("帮我看看这个函数")), soul_md=SOUL_MD, agents_md=AGENTS_MD)
        assert update.patches == [] and update.signals == []
        assert llm.call_count == 1

    async def test_target_none_dropped(self) -> None:
        cand = _candidate(target="none", evidence="我是后端工程师", signal="User is a backend engineer")
        update = await self._run_single_candidate(cand, "我是后端工程师")
        assert update.patches == []

    async def test_one_off_persistence_dropped(self) -> None:
        cand = _candidate(
            persistence="one_off", evidence="这次帮我用 patch=8", signal="Use patch=8", patch=_ADD_RULE_PATCH
        )
        update = await self._run_single_candidate(cand, "这次帮我用 patch=8")
        assert update.patches == [] and update.new_agents_md == AGENTS_MD

    async def test_task_directed_dropped(self) -> None:
        cand = _candidate(directed_at="task", evidence="这个函数太长了", signal="Function too long")
        update = await self._run_single_candidate(cand, "这个函数太长了")
        assert update.patches == []

    async def test_non_directive_speech_act_dropped(self) -> None:
        # Gate 0: config-shaped content inside a question never produces a patch.
        cand = _candidate(
            speech_act="question",
            evidence="能不能设置成每次保存都自动跑 lint?",
            signal="Always run lint on save",
            patch=_ADD_RULE_PATCH,
        )
        update = await self._run_single_candidate(cand, "能不能设置成每次保存都自动跑 lint?")
        assert update.patches == [] and update.new_agents_md == AGENTS_MD

    async def test_missing_speech_act_dropped(self) -> None:
        # White-list semantics: a candidate without the speech_act field is dropped, not trusted.
        cand = _candidate(patch=_ADD_RULE_PATCH)
        del cand["speech_act"]
        update = await self._run_single_candidate(cand, _EXPLICIT_USER_TEXT)
        assert update.patches == []

    async def test_redundant_novelty_dropped(self) -> None:
        cand = _candidate(novelty="redundant", evidence=_EXPLICIT_USER_TEXT, patch=_ADD_RULE_PATCH)
        update = await self._run_single_candidate(cand, _EXPLICIT_USER_TEXT)
        assert update.patches == []

    async def test_hallucinated_evidence_dropped(self) -> None:
        cand = _candidate(evidence="never auto commit please", patch=_ADD_RULE_PATCH)
        update = await self._run_single_candidate(cand, "随便聊聊天气")
        assert update.patches == []

    async def test_evidence_from_assistant_message_dropped(self) -> None:
        # The quote exists in the conversation but only in an assistant message — user messages
        # are the sole input, so the evidence check must fail.
        llm = FakeLLMClient(responses=[_llm_response([_candidate(evidence="以后不再自动提交", patch=_ADD_RULE_PATCH)])])
        extractor = AgentProfileExtractor(llm=llm)
        update = await extractor.aextract(
            _memcell(_user("帮我改下代码"), _assistant("好的, 以后不再自动提交。")),
            soul_md=SOUL_MD,
            agents_md=AGENTS_MD,
        )
        assert update.patches == []

    async def test_implicit_single_occurrence_recorded_not_patched(self) -> None:
        cand = _candidate(
            persistence="implicit",
            evidence="你回答太长了",
            signal="Answers are too long",
            key="be-concise",
            patch=_ADD_RULE_PATCH,
        )
        update = await self._run_single_candidate(cand, "你回答太长了")
        assert update.patches == [] and update.new_agents_md == AGENTS_MD
        assert len(update.signals) == 1
        sig = update.signals[0]
        assert sig.key == "be-concise" and sig.occurrences == 1 and sig.target == "agents"


# ── Gate 3 recurrence accumulation via pending_signals ──────────────────────────────────────────────


class TestRecurrenceGate:
    async def test_implicit_matching_pending_signal_reaches_gate(self) -> None:
        pending = AgentProfileSignal(
            key="be-concise",
            description="Answers are too long",
            target="soul",
            evidence="你回答太长了",
            occurrences=1,
            timestamp=1690000000000,
        )
        llm = FakeLLMClient(
            responses=[
                _llm_response(
                    [
                        _candidate(
                            target="soul",
                            persistence="implicit",
                            evidence="你太啰嗦了",
                            signal="Keep replies short",
                            matched_pending_key="be-concise",
                            patch={
                                "action": "add",
                                "section": "Communication Style",
                                "old_text": "",
                                "new_text": "- Keep replies short by default",
                            },
                        )
                    ]
                )
            ]
        )
        extractor = AgentProfileExtractor(llm=llm, min_recurrence=2)
        update = await extractor.aextract(
            _memcell(_user("你太啰嗦了")), soul_md=SOUL_MD, agents_md=AGENTS_MD, pending_signals=[pending]
        )
        assert llm.call_count == 1
        assert len(update.patches) == 1
        assert update.patches[0].file == "soul" and update.patches[0].action == "add"
        assert "- Keep replies short by default" in update.new_soul_md
        assert update.signals == []  # consumed by the patch, not re-emitted

    async def test_implicit_unmatched_pending_starts_fresh_count(self) -> None:
        pending = AgentProfileSignal(
            key="unrelated-key",
            description="Something else",
            target="agents",
            evidence="x",
            occurrences=5,
            timestamp=1690000000000,
        )
        llm = FakeLLMClient(
            responses=[
                _llm_response(
                    [
                        _candidate(
                            persistence="implicit",
                            evidence="你回答太长了",
                            signal="Answers too long",
                            key="be-concise",
                            matched_pending_key=None,
                        )
                    ]
                )
            ]
        )
        extractor = AgentProfileExtractor(llm=llm, min_recurrence=2)
        update = await extractor.aextract(
            _memcell(_user("你回答太长了")), soul_md=SOUL_MD, agents_md=AGENTS_MD, pending_signals=[pending]
        )
        assert update.patches == []
        assert len(update.signals) == 1 and update.signals[0].occurrences == 1


# ── Positive path: explicit instruction → validated patch + diff ────────────────────────────────────


class TestExplicitPatch:
    async def test_explicit_add_appends_under_existing_section(self) -> None:
        llm = FakeLLMClient(responses=[_llm_response([_candidate(patch=_ADD_RULE_PATCH)])])
        extractor = AgentProfileExtractor(llm=llm)
        update = await extractor.aextract(_memcell(_user(_EXPLICIT_USER_TEXT)), soul_md=SOUL_MD, agents_md=AGENTS_MD)

        assert llm.call_count == 1
        assert len(update.patches) == 1
        patch = update.patches[0]
        assert patch.is_conflict is False
        assert patch.evidence == _EXPLICIT_USER_TEXT
        # New rule lands inside the Operating Rules section, after the existing bullets.
        section = update.new_agents_md.split("## Operating Rules")[1]
        assert "- Keep commit messages in English\n- Never auto-commit after editing code" in section
        assert update.soul_diff == "" and update.new_soul_md == SOUL_MD
        assert update.agents_diff.startswith("--- a/AGENTS.md")
        assert "+- Never auto-commit after editing code" in update.agents_diff

    async def test_add_with_unknown_section_appends_new_section(self) -> None:
        patch = dict(_ADD_RULE_PATCH, section="Git Workflow")
        llm = FakeLLMClient(responses=[_llm_response([_candidate(patch=patch)])])
        extractor = AgentProfileExtractor(llm=llm)
        update = await extractor.aextract(_memcell(_user(_EXPLICIT_USER_TEXT)), soul_md=SOUL_MD, agents_md=AGENTS_MD)
        assert "## Git Workflow\n\n- Never auto-commit after editing code" in update.new_agents_md

    async def test_modify_replaces_verbatim_block(self) -> None:
        llm = FakeLLMClient(
            responses=[
                _llm_response(
                    [
                        _candidate(
                            target="agents",
                            signal="Commit messages must be in Chinese",
                            evidence="提交信息一律用中文",
                            patch={
                                "action": "modify",
                                "section": "Operating Rules",
                                "old_text": "- Keep commit messages in English",
                                "new_text": "- Keep commit messages in Chinese",
                            },
                        )
                    ]
                )
            ]
        )
        extractor = AgentProfileExtractor(llm=llm)
        update = await extractor.aextract(_memcell(_user("提交信息一律用中文")), soul_md=SOUL_MD, agents_md=AGENTS_MD)
        assert len(update.patches) == 1 and update.patches[0].action == "modify"
        assert "- Keep commit messages in Chinese" in update.new_agents_md
        assert "- Keep commit messages in English" not in update.new_agents_md
        assert "-- Keep commit messages in English" in update.agents_diff

    def test_sync_bridge_extract(self) -> None:
        llm = FakeLLMClient(responses=[_llm_response([])])
        extractor = AgentProfileExtractor(llm=llm)
        update = extractor.extract(_memcell(_user("聊聊天")), soul_md=SOUL_MD, agents_md=AGENTS_MD)
        assert update.patches == []


# ── Conflict handling: applied like any other patch, flagged for caller confirmation / debugging ───


class TestConflictFlag:
    async def test_conflict_patch_applied_and_flagged(self) -> None:
        llm = FakeLLMClient(
            responses=[
                _llm_response(
                    [
                        _candidate(
                            target="agents",
                            signal="Commit messages must be in Chinese",
                            evidence="提交信息一律用中文",
                            novelty="conflict",
                            patch={
                                "action": "modify",
                                "section": "Operating Rules",
                                "old_text": "- Keep commit messages in English",
                                "new_text": "- Keep commit messages in Chinese",
                            },
                        )
                    ]
                )
            ]
        )
        extractor = AgentProfileExtractor(llm=llm)
        update = await extractor.aextract(_memcell(_user("提交信息一律用中文")), soul_md=SOUL_MD, agents_md=AGENTS_MD)
        assert len(update.patches) == 1
        # Conflicting (mind-changed) patches are applied like any other; the flag lets the caller
        # route them through a confirmation step or surface them in debugging.
        assert update.patches[0].is_conflict is True
        assert "- Keep commit messages in Chinese" in update.new_agents_md
        assert "- Keep commit messages in English" not in update.new_agents_md
        assert "-- Keep commit messages in English" in update.agents_diff


# ── Patch structural validation drops ───────────────────────────────────────────────────────────────


class TestPatchValidationDrops:
    async def _run_with_patch(self, patch: dict[str, Any] | None) -> Any:
        llm = FakeLLMClient(responses=[_llm_response([_candidate(patch=patch)])])
        extractor = AgentProfileExtractor(llm=llm)
        return await extractor.aextract(_memcell(_user(_EXPLICIT_USER_TEXT)), soul_md=SOUL_MD, agents_md=AGENTS_MD)

    async def test_missing_patch_dropped(self) -> None:
        update = await self._run_with_patch(None)
        assert update.patches == [] and update.new_agents_md == AGENTS_MD

    async def test_modify_with_unmatched_old_text_dropped(self) -> None:
        update = await self._run_with_patch(
            {
                "action": "modify",
                "section": "Operating Rules",
                "old_text": "- This bullet does not exist",
                "new_text": "- Never auto-commit",
            }
        )
        assert update.patches == [] and update.new_agents_md == AGENTS_MD

    async def test_add_already_present_dropped(self) -> None:
        update = await self._run_with_patch(
            {
                "action": "add",
                "section": "Operating Rules",
                "old_text": "",
                "new_text": "- Keep commit messages in English",
            }
        )
        assert update.patches == [] and update.new_agents_md == AGENTS_MD

    async def test_invalid_action_dropped(self) -> None:
        update = await self._run_with_patch(
            {"action": "delete", "section": "Operating Rules", "old_text": "- Ask before deleting files"}
        )
        assert update.patches == []

    async def test_token_budget_guard_drops_oversized_patch(self) -> None:
        oversized = dict(_ADD_RULE_PATCH, new_text="- " + "never auto commit " * 200)
        llm = FakeLLMClient(responses=[_llm_response([_candidate(patch=oversized)])])
        extractor = AgentProfileExtractor(llm=llm, max_file_tokens=100)
        update = await extractor.aextract(_memcell(_user(_EXPLICIT_USER_TEXT)), soul_md=SOUL_MD, agents_md=AGENTS_MD)
        assert update.patches == [] and update.new_agents_md == AGENTS_MD

    async def test_multiple_survivors_all_applied(self) -> None:
        # No per-run cap: every patch that passes the gates and validation is applied; precision is
        # guarded per-patch (gates + verbatim evidence + max_file_tokens), not by batch size.
        cands = [
            _candidate(
                signal=f"Rule {i}",
                key=f"rule-{i}",
                patch=dict(_ADD_RULE_PATCH, new_text=f"- Never auto-commit (rule {i})"),
            )
            for i in range(3)
        ]
        llm = FakeLLMClient(responses=[_llm_response(cands)])
        extractor = AgentProfileExtractor(llm=llm)
        update = await extractor.aextract(_memcell(_user(_EXPLICIT_USER_TEXT)), soul_md=SOUL_MD, agents_md=AGENTS_MD)
        assert len(update.patches) == 3
        for i in range(3):
            assert f"- Never auto-commit (rule {i})" in update.new_agents_md

    async def test_malformed_response_raises(self) -> None:
        llm = FakeLLMClient(responses=['{"no_candidates_key": []}'])
        extractor = AgentProfileExtractor(llm=llm)
        with pytest.raises(ValueError, match="candidates"):
            await extractor.aextract(_memcell(_user(_EXPLICIT_USER_TEXT)), soul_md=SOUL_MD, agents_md=AGENTS_MD)


# ── Defensive branches: malformed LLM output and edge interactions ──────────────────────────────────


class TestDefensiveBranches:
    async def _run(self, response: str) -> Any:
        llm = FakeLLMClient(responses=[response])
        extractor = AgentProfileExtractor(llm=llm)
        return await extractor.aextract(_memcell(_user(_EXPLICIT_USER_TEXT)), soul_md=SOUL_MD, agents_md=AGENTS_MD)

    async def test_non_dict_candidate_skipped(self) -> None:
        update = await self._run(_llm_response(["not a dict"]))  # type: ignore[list-item]
        assert update.patches == [] and update.signals == []

    async def test_implicit_without_key_not_recorded(self) -> None:
        cand = _candidate(persistence="implicit", evidence=_EXPLICIT_USER_TEXT, key="", matched_pending_key=None)
        update = await self._run(_llm_response([cand]))
        assert update.patches == [] and update.signals == []  # no key -> nothing for the caller to persist

    async def test_patch_missing_section_dropped(self) -> None:
        update = await self._run(_llm_response([_candidate(patch=dict(_ADD_RULE_PATCH, section=""))]))
        assert update.patches == [] and update.new_agents_md == AGENTS_MD

    async def test_modify_identical_new_text_dropped(self) -> None:
        patch = {
            "action": "modify",
            "section": "Operating Rules",
            "old_text": "- Ask before deleting files",
            "new_text": "- Ask before deleting files",
        }
        update = await self._run(_llm_response([_candidate(patch=patch)]))
        assert update.patches == [] and update.new_agents_md == AGENTS_MD

    async def test_overlapping_modify_dropped_at_apply_stage(self) -> None:
        # Both patches validate against the ORIGINAL file, but the first one consumes the anchor;
        # the second must be re-checked against the working text and dropped.
        def _mod(i: int) -> dict[str, Any]:
            return {
                "action": "modify",
                "section": "Operating Rules",
                "old_text": "- Keep commit messages in English",
                "new_text": f"- Keep commit messages in Chinese (v{i})",
            }

        cands = [
            _candidate(signal=f"Rule {i}", key=f"rule-{i}", evidence=_EXPLICIT_USER_TEXT, patch=_mod(i))
            for i in range(2)
        ]
        update = await self._run(_llm_response(cands))
        assert len(update.patches) == 1
        assert "- Keep commit messages in Chinese (v0)" in update.new_agents_md
        assert "(v1)" not in update.new_agents_md

    async def test_candidates_not_a_list_raises(self) -> None:
        with pytest.raises(ValueError, match="must be a list"):
            await self._run('{"candidates": "nope"}')

    async def test_response_without_json_raises(self) -> None:
        with pytest.raises(ValueError, match="No JSON object"):
            await self._run("sorry, I cannot help with that")

    async def test_unbalanced_json_raises(self) -> None:
        with pytest.raises(ValueError, match="Unbalanced"):
            await self._run('{"candidates": [')


# ── Pure helpers ────────────────────────────────────────────────────────────────────────────────────


class TestHelpers:
    def test_insert_under_section_before_next_heading(self) -> None:
        out = _insert_under_section(SOUL_MD, "Personality", "- Calm under pressure")
        section = out.split("## Personality")[1].split("## Communication Style")[0]
        assert "- Curious and eager to learn\n- Calm under pressure" in section

    def test_insert_under_missing_section_appends(self) -> None:
        out = _insert_under_section("# Soul\n\nIntro.\n", "Values", "- Honesty")
        assert out.endswith("## Values\n\n- Honesty\n")

    def test_unified_diff_empty_when_identical(self) -> None:
        assert _unified_diff(SOUL_MD, SOUL_MD, "SOUL.md") == ""

    def test_unified_diff_labels_files(self) -> None:
        diff = _unified_diff("a\n", "b\n", "SOUL.md")
        assert diff.startswith("--- a/SOUL.md") and "+++ b/SOUL.md" in diff
