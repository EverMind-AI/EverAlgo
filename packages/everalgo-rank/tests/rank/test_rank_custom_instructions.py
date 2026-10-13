"""``custom_instructions`` on the rerank entry points: one block closing the rerank system message.

Driven through every public way into the rerank call — ``arerank`` / ``rerank``, the ``case`` / ``skill`` /
``episodic`` rankers in their module and class forms, and the top-level ``arank`` dispatch — with a
``FakeLLMClient`` reading what was sent: the block follows the rendered rules and is the system message's last,
its text is verbatim and never a template, a blank value sends exactly what ``None`` sends, the user message does
not change, and the old ``prompt=`` override is gone.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable
from typing import Any, Literal

import pytest

from everalgo.rank import CaseRanker, EpisodicRanker, SkillRanker, arank, case, episodic, skill
from everalgo.rank import rerank as rerank_mod
from everalgo.rank.prompts.en.case import CASE_RERANK_PROMPT_EN
from everalgo.rank.prompts.en.episodic import EPISODIC_RERANK_PROMPT_EN
from everalgo.rank.prompts.en.skill import SKILL_RERANK_PROMPT_EN
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import Candidate, RankInput, RankOutput
from everalgo.utils import render_prompt

_TOP_K = 2
_TEXT = "Prefer the most recent items.\nKeep {{top_k}} and {x} literal."
_BLOCK = f"\n\n<custom_instructions>\n{_TEXT}\n</custom_instructions>"
_REPLY = json.dumps({"ranked": [{"id": "c1", "score": 0.9}, {"id": "c0", "score": 0.8}]})
_CANDIDATES = [Candidate(id=f"c{i}", score=1.0 - i / 10, metadata={"text": f"item {i}"}) for i in range(3)]

type _Facade = Callable[..., Awaitable[RankOutput]]
type _MemoryType = Literal["case", "skill", "episodic"]


def _rank_input(memory_type: _MemoryType) -> RankInput:
    return RankInput(query="what changed?", memory_type=memory_type, dense_candidates=list(_CANDIDATES), top_k=_TOP_K)


def _module_level(function: _Facade) -> Callable[[FakeLLMClient], _Facade]:
    def bind(llm: FakeLLMClient) -> _Facade:
        async def run(rank_input: RankInput, **kwargs: Any) -> RankOutput:
            return await function(rank_input, llm=llm, **kwargs)

        return run

    return bind


def _class_form(ranker: type[CaseRanker | SkillRanker | EpisodicRanker]) -> Callable[[FakeLLMClient], _Facade]:
    def bind(llm: FakeLLMClient) -> _Facade:
        return ranker(llm=llm).arank

    return bind


# (bind the facade to a client, memory_type, the bundled rerank rules it sends)
_FACADES: dict[str, tuple[Callable[[FakeLLMClient], _Facade], _MemoryType, str]] = {
    "case.arank": (_module_level(case.arank), "case", CASE_RERANK_PROMPT_EN),
    "skill.arank": (_module_level(skill.arank), "skill", SKILL_RERANK_PROMPT_EN),
    "episodic.arank": (_module_level(episodic.arank), "episodic", EPISODIC_RERANK_PROMPT_EN),
    "CaseRanker.arank": (_class_form(CaseRanker), "case", CASE_RERANK_PROMPT_EN),
    "SkillRanker.arank": (_class_form(SkillRanker), "skill", SKILL_RERANK_PROMPT_EN),
    "EpisodicRanker.arank": (_class_form(EpisodicRanker), "episodic", EPISODIC_RERANK_PROMPT_EN),
    "rank.arank (dispatch)": (_module_level(arank), "episodic", EPISODIC_RERANK_PROMPT_EN),
}
_EACH_FACADE = pytest.mark.parametrize("facade", list(_FACADES))
_SIGNATURES: dict[str, Callable[..., Any]] = {
    "arerank": rerank_mod.arerank,
    "_basic_arank": rerank_mod._basic_arank,
    "case.arank": case.arank,
    "skill.arank": skill.arank,
    "episodic.arank": episodic.arank,
    "CaseRanker.arank": CaseRanker.arank,
    "SkillRanker.arank": SkillRanker.arank,
    "EpisodicRanker.arank": EpisodicRanker.arank,
}


async def _facade_messages(facade: str, **kwargs: Any) -> tuple[str, str]:
    bind, memory_type, _ = _FACADES[facade]
    client = FakeLLMClient(responses=[_REPLY])
    await bind(client)(_rank_input(memory_type), enable_rerank=True, **kwargs)
    system, user = client.calls[0].messages
    return str(system.content), str(user.content)


async def _arerank_messages(**kwargs: Any) -> tuple[str, str]:
    client = FakeLLMClient(responses=[_REPLY])
    await rerank_mod.arerank(
        _CANDIDATES, query="what changed?", prompt=EPISODIC_RERANK_PROMPT_EN, top_k=_TOP_K, llm=client, **kwargs
    )
    system, user = client.calls[0].messages
    return str(system.content), str(user.content)


# ── arerank ─────────────────────────────────────────────────────────────────────────────────────────


async def test_arerank_appends_the_block_after_the_rendered_rules() -> None:
    system, user = await _arerank_messages(custom_instructions=_TEXT)

    assert system == render_prompt(EPISODIC_RERANK_PROMPT_EN, None, top_k=_TOP_K) + _BLOCK
    assert system.endswith("\n</custom_instructions>")
    assert user == (await _arerank_messages())[1]


@pytest.mark.parametrize("blank", [None, "", "  \n\t "])
async def test_arerank_without_custom_instructions_sends_the_rendered_rules_only(blank: str | None) -> None:
    system, _ = await _arerank_messages(custom_instructions=blank)

    assert system == render_prompt(EPISODIC_RERANK_PROMPT_EN, None, top_k=_TOP_K)


def test_sync_rerank_bridge_forwards_custom_instructions() -> None:
    client = FakeLLMClient(responses=[_REPLY])

    rerank_mod.rerank(
        _CANDIDATES,
        query="q",
        prompt=EPISODIC_RERANK_PROMPT_EN,
        top_k=_TOP_K,
        llm=client,
        custom_instructions=_TEXT,
    )

    assert str(client.calls[0].messages[0].content).endswith(_BLOCK)


# ── facades ─────────────────────────────────────────────────────────────────────────────────────────


@_EACH_FACADE
async def test_facade_appends_the_block_to_its_bundled_rerank_rules(facade: str) -> None:
    rules = render_prompt(_FACADES[facade][2], None, top_k=_TOP_K)

    system, user = await _facade_messages(facade, custom_instructions=_TEXT)

    assert system == rules + _BLOCK
    assert user == (await _facade_messages(facade))[1]


@_EACH_FACADE
@pytest.mark.parametrize("blank", ["", "  \n\t "])
async def test_facade_blank_custom_instructions_are_the_same_as_none(facade: str, blank: str) -> None:
    default = await _facade_messages(facade)

    assert await _facade_messages(facade, custom_instructions=blank) == default
    assert default[0] == render_prompt(_FACADES[facade][2], None, top_k=_TOP_K)


@_EACH_FACADE
async def test_facade_without_rerank_ignores_custom_instructions(facade: str) -> None:
    bind, memory_type, _ = _FACADES[facade]
    client = FakeLLMClient(responses=[])

    await bind(client)(_rank_input(memory_type), custom_instructions=_TEXT)

    assert client.call_count == 0


@_EACH_FACADE
async def test_facade_rejects_the_removed_prompt_override(facade: str) -> None:
    bind, memory_type, _ = _FACADES[facade]

    with pytest.raises(TypeError, match="prompt"):
        await bind(FakeLLMClient(responses=[]))(_rank_input(memory_type), enable_rerank=True, prompt="Rules {{top_k}}.")


@pytest.mark.parametrize("name", list(_SIGNATURES))
def test_custom_instructions_is_an_optional_keyword(name: str) -> None:
    parameter = inspect.signature(_SIGNATURES[name]).parameters["custom_instructions"]

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None


@pytest.mark.parametrize("name", [name for name in _SIGNATURES if name != "arerank"])
def test_facade_takes_no_prompt_override(name: str) -> None:
    assert "prompt" not in inspect.signature(_SIGNATURES[name]).parameters
