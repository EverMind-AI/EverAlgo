"""The four rank LLM callsites against the LLM contract: reply models, parsing, and the ``[system, user]`` pair.

Each agentic call is driven through its real helper with a ``FakeLLMClient``: a valid reply comes back as the
operator's result, a reply that is not JSON, misses a field or adds one raises ``LLMOutputError``, the requested
``response_format`` and temperature are the call's own, the system message is the default rules, closed by the
caller's ``<custom_instructions>`` block when one is given, and the user message is the data rendered into its tags.
The rerank call has the same checks in ``test_rank_rerank.py``.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import pytest

from everalgo.llm import LLMOutputError, StrictOutputModel, to_openai_response_format
from everalgo.rank import aagentic_retrieve
from everalgo.rank import agentic as agentic_mod
from everalgo.rank._replies import (
    _MultiQueryReply,
    _RankedCandidate,
    _RefinedQueryReply,
    _RerankReply,
    _SufficiencyReply,
)
from everalgo.rank.prompts.en.multi_query import MULTI_QUERY_PROMPT_EN, MULTI_QUERY_USER_PROMPT
from everalgo.rank.prompts.en.refined_query import REFINED_QUERY_PROMPT_EN, REFINED_QUERY_USER_PROMPT
from everalgo.rank.prompts.en.sufficiency import SUFFICIENCY_CHECK_PROMPT_EN, SUFFICIENCY_CHECK_USER_PROMPT
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import Candidate
from everalgo.utils import render_prompt

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

_QUERY = "When did Alice move to New York?"
_CANDIDATES = [
    Candidate(
        id="ep1",
        score=0.9,
        metadata={"episode": {"subject": "New flat", "content": "Alice lives in Brooklyn."}, "timestamp": 1_000},
    )
]
_DOCS = agentic_mod._format_docs(_CANDIDATES)
_REPLY_MODELS: list[type[StrictOutputModel]] = [
    _SufficiencyReply,
    _MultiQueryReply,
    _RefinedQueryReply,
    _RankedCandidate,
    _RerankReply,
]


@dataclass(frozen=True)
class _Call:
    """One agentic callsite: how to drive it, what it must request, and the user message it must send."""

    run: Callable[[FakeLLMClient, str | None], Awaitable[Any]]
    model: type[StrictOutputModel]
    temperature: float
    good_reply: dict[str, Any]
    default_system: str
    expected_user: str


async def _sufficiency(llm: FakeLLMClient, custom_instructions: str | None) -> Any:
    return await agentic_mod._check_sufficiency(
        query=_QUERY, candidates=_CANDIDATES, llm=llm, custom_instructions=custom_instructions
    )


async def _multi_query(llm: FakeLLMClient, custom_instructions: str | None) -> Any:
    return await agentic_mod._generate_multi_queries(
        original_query=_QUERY,
        candidates=_CANDIDATES,
        missing_info=["move date"],
        key_info=["lives in Brooklyn", "has a flat"],
        llm=llm,
        custom_instructions=custom_instructions,
        count=3,
    )


async def _refined_query(llm: FakeLLMClient, custom_instructions: str | None) -> Any:
    return await agentic_mod._generate_refined_query(
        original_query=_QUERY,
        candidates=_CANDIDATES,
        missing_info=[],
        llm=llm,
        custom_instructions=custom_instructions,
    )


_CALLS = {
    "sufficiency": _Call(
        run=_sufficiency,
        model=_SufficiencyReply,
        temperature=0.0,
        good_reply={
            "is_sufficient": False,
            "reasoning": "no date",
            "key_information_found": ["lives in Brooklyn"],
            "missing_information": ["move date"],
        },
        default_system=SUFFICIENCY_CHECK_PROMPT_EN,
        expected_user=render_prompt(SUFFICIENCY_CHECK_USER_PROMPT, None, query=_QUERY, retrieved_docs=_DOCS),
    ),
    "multi_query": _Call(
        run=_multi_query,
        model=_MultiQueryReply,
        temperature=0.4,
        good_reply={"queries": ["Alice relocation date", "hi"], "reasoning": "Q1: Temporal"},
        default_system=MULTI_QUERY_PROMPT_EN,
        expected_user=render_prompt(
            MULTI_QUERY_USER_PROMPT,
            None,
            original_query=_QUERY,
            key_info="lives in Brooklyn, has a flat",
            missing_info="move date",
            retrieved_docs=_DOCS,
        ),
    ),
    "refined_query": _Call(
        run=_refined_query,
        model=_RefinedQueryReply,
        temperature=0.3,
        good_reply={"text": "When exactly did Alice move to Brooklyn?"},
        default_system=REFINED_QUERY_PROMPT_EN,
        expected_user=render_prompt(
            REFINED_QUERY_USER_PROMPT, None, original_query=_QUERY, retrieved_docs=_DOCS, missing_info="N/A"
        ),
    ),
}
_STAGES = pytest.mark.parametrize("stage", list(_CALLS))


def _missing_one_field(reply: dict[str, Any]) -> str:
    return json.dumps(dict(list(reply.items())[1:]))


def _with_extra_field(reply: dict[str, Any]) -> str:
    return json.dumps({**reply, "confidence": 0.9})


# ── Reply models (llm-contract §4.3) ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("model", _REPLY_MODELS, ids=lambda m: m.__name__)
def test_reply_model_is_strict_with_no_defaults_and_fits_the_strict_schema(model: type[StrictOutputModel]) -> None:
    assert issubclass(model, StrictOutputModel)
    assert all(field.is_required() for field in model.model_fields.values())
    to_openai_response_format(model)


# ── Parsing ─────────────────────────────────────────────────────────────────────────────────────────


async def test_sufficiency_reply_becomes_the_check_result() -> None:
    client = FakeLLMClient(responses=[json.dumps(_CALLS["sufficiency"].good_reply)])

    result = await _sufficiency(client, None)

    assert result == agentic_mod.SufficiencyCheckResponse(
        is_sufficient=False,
        reasoning="no date",
        key_information_found=["lives in Brooklyn"],
        missing_information=["move date"],
    )


async def test_multi_query_reply_keeps_only_the_usable_queries() -> None:
    client = FakeLLMClient(responses=[json.dumps(_CALLS["multi_query"].good_reply)])

    result = await _multi_query(client, None)

    assert result == agentic_mod.MultiQueryResponse(queries=["Alice relocation date"], reasoning="Q1: Temporal")


async def test_refined_query_reply_text_becomes_the_refined_query() -> None:
    client = FakeLLMClient(responses=[json.dumps(_CALLS["refined_query"].good_reply)])

    result = await _refined_query(client, None)

    assert result == agentic_mod.RefinedQueryResponse(
        refined_query="When exactly did Alice move to Brooklyn?", reasoning=""
    )


@_STAGES
@pytest.mark.parametrize(
    "make_reply",
    [lambda _: "not json at all", _missing_one_field, _with_extra_field],
    ids=["not_json", "missing_field", "extra_field"],
)
async def test_invalid_reply_raises_llm_output_error(stage: str, make_reply: Callable[[dict[str, Any]], str]) -> None:
    call = _CALLS[stage]
    client = FakeLLMClient(responses=[make_reply(call.good_reply)])

    with pytest.raises(LLMOutputError, match=call.model.__name__):
        await call.run(client, None)


@_STAGES
async def test_the_error_message_never_quotes_the_reply(stage: str) -> None:
    call = _CALLS[stage]
    client = FakeLLMClient(responses=[_with_extra_field({**call.good_reply, "leak": "SECRET-REPLY-TEXT"})])

    with pytest.raises(LLMOutputError) as caught:
        await call.run(client, None)

    assert "SECRET-REPLY-TEXT" not in str(caught.value)


@_STAGES
async def test_client_exception_propagates_unchanged(stage: str) -> None:
    class _HostError(Exception):
        pass

    client = FakeLLMClient(responses=[_HostError("upstream down")])

    with pytest.raises(_HostError, match="upstream down"):
        await _CALLS[stage].run(client, None)


# ── What the call requests ──────────────────────────────────────────────────────────────────────────


@_STAGES
async def test_call_requests_its_reply_model_at_its_temperature(stage: str) -> None:
    call = _CALLS[stage]
    client = FakeLLMClient(responses=[json.dumps(call.good_reply)])

    await call.run(client, None)

    record = client.calls[0]
    assert record.response_format is call.model
    assert record.temperature == call.temperature
    assert record.max_tokens is None


@_STAGES
async def test_call_sends_the_default_rules_then_the_tagged_data(stage: str) -> None:
    call = _CALLS[stage]
    client = FakeLLMClient(responses=[json.dumps(call.good_reply)])

    await call.run(client, None)

    system, user = client.calls[0].messages
    assert (system.role, system.content) == ("system", call.default_system)
    assert (user.role, user.content) == ("user", call.expected_user)


# ── custom_instructions: appended to the rules, never a template, never an override ───────────────


@_STAGES
async def test_custom_instructions_close_the_system_message_and_leave_the_data(stage: str) -> None:
    call = _CALLS[stage]
    client = FakeLLMClient(responses=[json.dumps(call.good_reply)])
    text = 'Prefer recent documents.\nA JSON sample {"a": 1} stays.'

    await call.run(client, text)

    system, user = client.calls[0].messages
    assert system.content == f"{call.default_system}\n\n<custom_instructions>\n{text}\n</custom_instructions>"
    assert str(system.content).endswith("\n</custom_instructions>")
    assert user.content == call.expected_user


@_STAGES
@pytest.mark.parametrize("blank", ["", "  \n\t "])
async def test_blank_custom_instructions_are_the_same_as_none(stage: str, blank: str) -> None:
    call = _CALLS[stage]
    client = FakeLLMClient(responses=[json.dumps(call.good_reply)] * 2)

    await call.run(client, blank)
    await call.run(client, None)

    blank_call, none_call = client.calls
    assert blank_call.messages == none_call.messages
    assert "custom_instructions" not in str(none_call.messages[0].content)


@_STAGES
@pytest.mark.parametrize("slot", ["{{query}}", "{{original_query}}", "{{retrieved_docs}}", "{x}"])
async def test_custom_instructions_are_never_rendered_as_a_template(stage: str, slot: str) -> None:
    """A slot in the caller's text is sent as written: it is neither substituted nor rejected."""
    client = FakeLLMClient(responses=[json.dumps(_CALLS[stage].good_reply)])

    await _CALLS[stage].run(client, f"Keep {slot} literal.")

    system = str(client.calls[0].messages[0].content)
    assert system.endswith(f"<custom_instructions>\nKeep {slot} literal.\n</custom_instructions>")


async def test_query_text_is_never_rendered_as_a_template() -> None:
    client = FakeLLMClient(responses=[json.dumps(_CALLS["sufficiency"].good_reply)])

    await agentic_mod._check_sufficiency(
        query="what is {{retrieved_docs}}?", candidates=_CANDIDATES, llm=client, custom_instructions=None
    )

    user = str(client.calls[0].messages[1].content)
    assert "<user_query>\nwhat is {{retrieved_docs}}?\n</user_query>" in user


@pytest.mark.parametrize("strategy", ["multi_query", "refined_query"])
async def test_aagentic_retrieve_appends_one_custom_instructions_to_every_call(
    strategy: Literal["multi_query", "refined_query"],
) -> None:
    async def base_retrieve(q: str, k: int) -> list[Candidate]:
        return list(_CANDIDATES)

    second = _CALLS[strategy]
    client = FakeLLMClient(responses=[json.dumps(_CALLS["sufficiency"].good_reply), json.dumps(second.good_reply)])

    await aagentic_retrieve(
        _QUERY,
        base_retrieve=base_retrieve,
        llm=client,
        refinement_strategy=strategy,
        custom_instructions="Prefer recent documents.",
    )

    block = "\n\n<custom_instructions>\nPrefer recent documents.\n</custom_instructions>"
    assert [record.messages[0].content for record in client.calls] == [
        _CALLS["sufficiency"].default_system + block,
        second.default_system + block,
    ]
    assert [record.response_format for record in client.calls] == [_SufficiencyReply, second.model]


def test_aagentic_retrieve_takes_no_prompt_override() -> None:
    parameters = inspect.signature(aagentic_retrieve).parameters

    assert "custom_instructions" in parameters
    assert parameters["custom_instructions"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["custom_instructions"].default is None
    assert not [name for name in parameters if name.endswith("prompt")]


@pytest.mark.parametrize("removed", ["sufficiency_prompt", "multi_query_prompt", "refined_query_prompt"])
async def test_aagentic_retrieve_rejects_a_removed_prompt_override(removed: str) -> None:
    async def base_retrieve(q: str, k: int) -> list[Candidate]:
        return list(_CANDIDATES)

    removed_kwarg: dict[str, Any] = {removed: "Rules."}

    with pytest.raises(TypeError, match=removed):
        await aagentic_retrieve(_QUERY, base_retrieve=base_retrieve, llm=FakeLLMClient(responses=[]), **removed_kwarg)
