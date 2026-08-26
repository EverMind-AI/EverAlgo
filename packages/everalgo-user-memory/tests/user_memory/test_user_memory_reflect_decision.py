"""Tests for everalgo.user_memory.reflect_decision — DecisionReflector.

INIT/UPDATE, too-few inputs, unsorted timestamps. Output is a Decision DTO, not a Principle.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from everalgo.llm.errors import LLMError
from everalgo.testing import FakeLLMClient
from everalgo.types import Decision

if TYPE_CHECKING:
    from everalgo.llm.types import ChatMessage as LLMChatMessage
    from everalgo.llm.types import ChatResponse


def _dc(
    ts: int,
    *,
    title: str = "topic",
    decision: str = "some decision",
    reason: str = "some reason",
    impact: str | None = None,
    tags: list[str] | None = None,
    owner_id: str | None = None,
) -> Decision:
    return Decision(
        owner_id=owner_id,
        title=title,
        decision=decision,
        reason=reason,
        impact=impact,
        tags=tags or [],
        timestamp=ts,
    )


def _merged_json(**overrides: Any) -> str:
    item: dict[str, Any] = {
        "decision": "Use a self-developed Agent Runtime.",
        "reason": "Need control over the loop.",
        "title": "Agent Runtime",
        "impact": "Device talks through our APIs.",
        "tags": ["runtime"],
    }
    item.update(overrides)
    return json.dumps(item)


# ==========================================================================
# Input validation
# ==========================================================================


class TestValidateInputs:
    def test_passes_with_two_sorted_decisions(self) -> None:
        from everalgo.user_memory.reflect_decision import _validate_inputs

        _validate_inputs([_dc(1000), _dc(2000)], min_count=2)

    def test_raises_when_fewer_than_min_count(self) -> None:
        from everalgo.user_memory.reflect_decision import _validate_inputs

        with pytest.raises(ValueError, match="at least 2"):
            _validate_inputs([_dc(1000)], min_count=2)

    def test_raises_on_empty_list(self) -> None:
        from everalgo.user_memory.reflect_decision import _validate_inputs

        with pytest.raises(ValueError, match="at least 1"):
            _validate_inputs([], min_count=1)

    def test_raises_when_timestamps_not_ascending(self) -> None:
        from everalgo.user_memory.reflect_decision import _validate_inputs

        with pytest.raises(ValueError, match=r"sorted.*ascending"):
            _validate_inputs([_dc(3000), _dc(1000)], min_count=2)

    def test_allows_equal_timestamps(self) -> None:
        from everalgo.user_memory.reflect_decision import _validate_inputs

        _validate_inputs([_dc(1000), _dc(1000)], min_count=2)


class TestRenderTimeline:
    def test_numbered_fields_without_owner_or_timestamp(self) -> None:
        from everalgo.user_memory.reflect_decision import _render_timeline

        result = _render_timeline(
            [
                _dc(
                    1, title="LangChain", decision="Use LangChain", reason="Ship faster", impact="Coupled", tags=["fw"]
                ),
                _dc(2, title="In-house", decision="Self-developed runtime", reason="Control the loop"),
            ]
        )
        assert "1. Title: LangChain" in result
        assert "   Decision: Use LangChain" in result
        assert "   Reason: Ship faster" in result
        assert "   Impact: Coupled" in result
        assert "   Tags: fw" in result
        assert "2. Title: In-house" in result
        assert "   Impact: (none)" in result
        assert "owner_id" not in result
        assert "timestamp" not in result.lower()


# ==========================================================================
# INIT
# ==========================================================================


class TestReflectInit:
    async def test_init_returns_merged_decision(self) -> None:
        fake = FakeLLMClient(responses=[_merged_json()])
        from everalgo.user_memory.reflect_decision import DecisionReflector

        result = await DecisionReflector(llm=fake).areflect(
            [
                _dc(1000, decision="Use LangChain", owner_id="u_alice"),
                _dc(2000, decision="Self-developed runtime"),
            ]
        )

        assert result.title == "Agent Runtime"
        assert result.decision == "Use a self-developed Agent Runtime."
        assert result.reason == "Need control over the loop."
        assert result.impact == "Device talks through our APIs."
        assert result.tags == ["runtime"]
        assert result.timestamp == 2000
        assert result.owner_id is None
        assert not hasattr(result, "statement")
        assert not hasattr(result, "source_entry_ids")

    async def test_init_sends_timeline_in_prompt(self) -> None:
        fake = FakeLLMClient(responses=[_merged_json()])
        from everalgo.user_memory.reflect_decision import DecisionReflector

        await DecisionReflector(llm=fake).areflect(
            [_dc(1000, decision="Use LangChain"), _dc(2000, decision="Self-developed runtime")]
        )

        assert fake.call_count == 1
        prompt_text = fake.calls[0].messages[0].content
        assert "Use LangChain" in prompt_text
        assert "Self-developed runtime" in prompt_text
        assert "1. Title:" in prompt_text

    async def test_init_uses_structured_output(self) -> None:
        fake = FakeLLMClient(responses=[_merged_json()])
        from everalgo.user_memory.reflect_decision import DecisionReflector

        await DecisionReflector(llm=fake).areflect([_dc(1000), _dc(2000)])

        assert fake.calls[0].response_format is not None

    async def test_init_fewer_than_2_raises(self) -> None:
        fake = FakeLLMClient(responses=[_merged_json()])
        from everalgo.user_memory.reflect_decision import DecisionReflector

        with pytest.raises(ValueError, match="at least 2"):
            await DecisionReflector(llm=fake).areflect([_dc(1000)])

        assert fake.call_count == 0

    async def test_init_unsorted_raises(self) -> None:
        fake = FakeLLMClient(responses=[_merged_json()])
        from everalgo.user_memory.reflect_decision import DecisionReflector

        with pytest.raises(ValueError, match=r"sorted.*ascending"):
            await DecisionReflector(llm=fake).areflect([_dc(3000), _dc(1000)])

        assert fake.call_count == 0


# ==========================================================================
# UPDATE
# ==========================================================================


class TestReflectUpdate:
    async def test_update_returns_merged_decision(self) -> None:
        fake = FakeLLMClient(responses=[_merged_json(title="Agent Runtime v2")])
        from everalgo.user_memory.reflect_decision import DecisionReflector

        old = _dc(1000, decision="Use LangChain")
        result = await DecisionReflector(llm=fake).areflect(
            [_dc(2000, decision="Self-developed runtime")],
            old_decision=old,
        )

        assert result.title == "Agent Runtime v2"
        assert result.decision == "Use a self-developed Agent Runtime."
        assert result.timestamp == 2000
        assert result.owner_id is None

    async def test_update_sends_old_decision_in_prompt(self) -> None:
        fake = FakeLLMClient(responses=[_merged_json()])
        from everalgo.user_memory.reflect_decision import DecisionReflector

        old = _dc(1000, decision="Use LangChain for the first version.")
        await DecisionReflector(llm=fake).areflect(
            [_dc(2000, decision="Replace LangChain with an in-house runtime.")],
            old_decision=old,
        )

        prompt_text = fake.calls[0].messages[0].content
        assert "Use LangChain for the first version." in prompt_text
        assert "Replace LangChain with an in-house runtime." in prompt_text

    async def test_update_single_decision_sufficient(self) -> None:
        fake = FakeLLMClient(responses=[_merged_json()])
        from everalgo.user_memory.reflect_decision import DecisionReflector

        result = await DecisionReflector(llm=fake).areflect([_dc(2000)], old_decision=_dc(1000))
        assert result.decision.endswith("Runtime.")

    async def test_update_empty_raises(self) -> None:
        fake = FakeLLMClient(responses=[_merged_json()])
        from everalgo.user_memory.reflect_decision import DecisionReflector

        with pytest.raises(ValueError, match="at least 1"):
            await DecisionReflector(llm=fake).areflect([], old_decision=_dc(1000))

        assert fake.call_count == 0

    async def test_update_timestamp_is_new_decisions_span_end(self) -> None:
        fake = FakeLLMClient(responses=[_merged_json()])
        from everalgo.user_memory.reflect_decision import DecisionReflector

        old = _dc(500, decision="older")
        result = await DecisionReflector(llm=fake).areflect(
            [_dc(1000, decision="first new"), _dc(3000, decision="last new")],
            old_decision=old,
        )
        assert result.timestamp == 3000


# ==========================================================================
# Errors / misc
# ==========================================================================


class TestReflectErrors:
    async def test_llm_error_propagates_init(self) -> None:
        def handler(messages: list[LLMChatMessage], **kwargs: Any) -> ChatResponse:
            raise LLMError("upstream failure")

        fake = FakeLLMClient(handler=handler)
        from everalgo.user_memory.reflect_decision import DecisionReflector

        with pytest.raises(LLMError, match="upstream failure"):
            await DecisionReflector(llm=fake).areflect([_dc(1000), _dc(2000)])

        assert fake.call_count == 1

    async def test_empty_required_fields_raise(self) -> None:
        fake = FakeLLMClient(responses=[_merged_json(title="  ", decision="kept", reason="kept")])
        from everalgo.user_memory.reflect_decision import DecisionReflector

        with pytest.raises(ValueError, match="empty title, decision, or reason"):
            await DecisionReflector(llm=fake).areflect([_dc(1000), _dc(2000)])


class TestReflectMisc:
    def test_sync_reflect_works(self) -> None:
        fake = FakeLLMClient(responses=[_merged_json()])
        from everalgo.user_memory.reflect_decision import DecisionReflector

        result = DecisionReflector(llm=fake).reflect([_dc(1000), _dc(2000)])
        assert result.decision.endswith("Runtime.")

    async def test_prompt_override_init(self) -> None:
        fake = FakeLLMClient(responses=[_merged_json()])
        from everalgo.user_memory.reflect_decision import DecisionReflector

        custom = "Custom merge: {timeline}"
        await DecisionReflector(llm=fake).areflect([_dc(1000), _dc(2000)], prompt=custom)

        prompt_text = fake.calls[0].messages[0].content
        assert isinstance(prompt_text, str)
        assert prompt_text.startswith("Custom merge:")
        assert "already-extracted decisions" not in prompt_text

    async def test_prompt_override_update(self) -> None:
        fake = FakeLLMClient(responses=[_merged_json()])
        from everalgo.user_memory.reflect_decision import DecisionReflector

        custom = "Update: {old_decision} + {new_decisions}"
        await DecisionReflector(llm=fake).areflect(
            [_dc(2000, decision="new stuff")], old_decision=_dc(1000, decision="old stuff"), prompt=custom
        )

        prompt_text = fake.calls[0].messages[0].content
        assert isinstance(prompt_text, str)
        assert "old stuff" in prompt_text
        assert "new stuff" in prompt_text
        assert "updating an existing" not in prompt_text.lower()

    def test_title_is_declared_after_decision_in_the_structured_output_schema(self) -> None:
        from everalgo.user_memory.reflect_decision import _DecisionReflectOutput

        order = list(_DecisionReflectOutput.model_fields)
        assert order.index("title") > order.index("decision")
        assert order.index("title") > order.index("reason")


# ==========================================================================
# Language rule — inherit, do not re-judge the conversation
# ==========================================================================


_LANGUAGE_PROMPTS = ("REFLECT_DECISION_PROMPT", "REFLECT_DECISION_UPDATE_PROMPT")


class _PromptCapturedError(Exception):
    """Ends the call once the prompt has been captured — no LLM response is needed."""


async def _render_reflect_decision_prompt(
    *,
    old_decision: Decision | None = None,
    output_language: str | None = None,
) -> str:
    from everalgo.user_memory.reflect_decision import DecisionReflector

    captured: list[str] = []

    class _Capture:
        async def chat(self, *, messages: list[LLMChatMessage], **kwargs: Any) -> ChatResponse:
            content = messages[0].content
            assert isinstance(content, str)
            captured.append(content)
            raise _PromptCapturedError

    kwargs: dict[str, Any] = {"old_decision": old_decision}
    if output_language is not None:
        kwargs["output_language"] = output_language

    with pytest.raises(_PromptCapturedError):
        await DecisionReflector(llm=_Capture()).areflect([_dc(1000), _dc(2000)], **kwargs)  # type: ignore[arg-type]
    return captured[0]


@pytest.mark.parametrize("name", _LANGUAGE_PROMPTS)
def test_reflect_prompts_carry_the_language_placeholder_at_both_ends(name: str) -> None:
    import everalgo.user_memory.prompts.en.reflect_decision as en_mod

    assert getattr(en_mod, name).count("{language_rule}") == 2


async def test_init_inherits_the_merged_decisions_language_when_none_is_named() -> None:
    rendered = await _render_reflect_decision_prompt()

    assert rendered.count("CRITICAL LANGUAGE RULE") == 2
    assert "as the decisions you are merging" in rendered
    assert "{language_rule}" not in rendered


async def test_update_inherits_the_existing_decision_language_when_none_is_named() -> None:
    rendered = await _render_reflect_decision_prompt(old_decision=_dc(500, decision="existing"))

    assert rendered.count("CRITICAL LANGUAGE RULE") == 2
    assert "as the existing decision you are updating" in rendered
    assert "{language_rule}" not in rendered


@pytest.mark.parametrize("old_decision", [None, _dc(500, decision="existing")])
async def test_both_modes_honour_a_named_language(old_decision: Decision | None) -> None:
    rendered = await _render_reflect_decision_prompt(old_decision=old_decision, output_language="German")

    assert rendered.count("CRITICAL LANGUAGE RULE") == 2
    assert "Write ALL output fields in German." in rendered
    assert "as the decisions you are merging" not in rendered
    assert "as the existing decision you are updating" not in rendered


@pytest.mark.parametrize("old_decision", [None, _dc(500, decision="existing")])
async def test_an_unsupported_language_is_rejected_before_the_llm_is_called(old_decision: Decision | None) -> None:
    from everalgo.user_memory.reflect_decision import DecisionReflector

    class _NeverCalled:
        async def chat(self, **kwargs: Any) -> ChatResponse:
            raise AssertionError("LLM must not be called")

    with pytest.raises(ValueError, match="unsupported output_language"):
        await DecisionReflector(llm=_NeverCalled()).areflect(  # type: ignore[arg-type]
            [_dc(1000), _dc(2000)], old_decision=old_decision, output_language="Klingon"
        )


@pytest.mark.parametrize(
    "rule_name",
    ["MERGED_DECISIONS_LANGUAGE_RULE", "EXISTING_DECISION_LANGUAGE_RULE"],
)
def test_reflect_fallbacks_inherit_rather_than_judge(rule_name: str) -> None:
    from everalgo.user_memory.prompts.en import _language as rules_mod

    rule = getattr(rules_mod, rule_name)
    assert "however much of the conversation it occupies" not in rule
    assert "read only the message contents" not in rule
    assert "do not translate" in rule
