"""Every rank call site sends the model a prompt with no slot left in it.

Asserting on the template alone would not catch the failure this package is exposed to: the
prompts and their call sites were migrated together, and a call site that went back to
``str.format`` would turn the template's ``{{query}}`` into a literal ``{query}`` and ship it —
no exception, no missing field, just the word ``{query}`` where the user's question belonged.
So each test drives the real function against a ``FakeLLMClient`` and reads the prompt the client
actually received.

The check rejects both brace shapes for that reason: ``{{query}}`` means the renderer never
substituted it, ``{query}`` means something other than the renderer did. Both messages of each
call are checked: the system message renders ``{{top_k}}`` (rerank), the user message the data.
The Chinese rerank mirrors get a case each: they are the rules ``arerank`` takes as ``prompt``, like the English.
"""

from __future__ import annotations

import json
import re

import pytest

from everalgo.rank import agentic as agentic_mod
from everalgo.rank import rerank as rerank_mod
from everalgo.rank.prompts.en.case import CASE_RERANK_PROMPT_EN
from everalgo.rank.prompts.en.episodic import EPISODIC_RERANK_PROMPT_EN
from everalgo.rank.prompts.en.skill import SKILL_RERANK_PROMPT_EN
from everalgo.rank.prompts.zh.case import CASE_RERANK_PROMPT_ZH
from everalgo.rank.prompts.zh.episodic import EPISODIC_RERANK_PROMPT_ZH
from everalgo.rank.prompts.zh.skill import SKILL_RERANK_PROMPT_ZH
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import Candidate

_SLOT_LEFT_BEHIND = re.compile(r"\{\{[A-Za-z_][A-Za-z0-9_]*\}\}|(?<!\{)\{[A-Za-z_][A-Za-z0-9_]*\}(?!\})")


def _assert_no_slot_reached_the_model(client: FakeLLMClient) -> None:
    assert client.call_count > 0, "the call site issued no LLM call, so nothing was checked"
    for record in client.calls:
        assert [message.role for message in record.messages] == ["system", "user"]
        for message in record.messages:
            leftover = _SLOT_LEFT_BEHIND.findall(str(message.content))
            assert not leftover, f"unsubstituted slots {leftover} in: {message.content}"


def _candidates() -> list[Candidate]:
    return [Candidate(id="c1", score=0.9, metadata={"episode": {"subject": "Alice", "content": "Alice ate cake"}})]


@pytest.mark.parametrize(
    "prompt",
    [
        pytest.param(CASE_RERANK_PROMPT_EN, id="case_en"),
        pytest.param(EPISODIC_RERANK_PROMPT_EN, id="episodic_en"),
        pytest.param(SKILL_RERANK_PROMPT_EN, id="skill_en"),
        pytest.param(CASE_RERANK_PROMPT_ZH, id="case_zh"),
        pytest.param(EPISODIC_RERANK_PROMPT_ZH, id="episodic_zh"),
        pytest.param(SKILL_RERANK_PROMPT_ZH, id="skill_zh"),
    ],
)
async def test_arerank_sends_a_fully_rendered_prompt(prompt: str) -> None:
    client = FakeLLMClient(responses=[json.dumps({"ranked": []})])

    await rerank_mod.arerank(_candidates(), query="who ate cake?", prompt=prompt, top_k=5, llm=client)

    _assert_no_slot_reached_the_model(client)


async def test_check_sufficiency_sends_a_fully_rendered_prompt() -> None:
    reply = {"is_sufficient": True, "reasoning": "", "key_information_found": [], "missing_information": []}
    client = FakeLLMClient(responses=[json.dumps(reply)])

    await agentic_mod._check_sufficiency(
        query="who ate cake?", candidates=_candidates(), llm=client, custom_instructions=None
    )

    _assert_no_slot_reached_the_model(client)


async def test_generate_multi_queries_sends_a_fully_rendered_prompt() -> None:
    client = FakeLLMClient(responses=[json.dumps({"queries": [], "reasoning": ""})])

    await agentic_mod._generate_multi_queries(
        original_query="who ate cake?",
        candidates=_candidates(),
        missing_info=[],
        key_info=[],
        llm=client,
        custom_instructions=None,
        count=2,
    )

    _assert_no_slot_reached_the_model(client)


async def test_generate_refined_query_sends_a_fully_rendered_prompt() -> None:
    client = FakeLLMClient(responses=[json.dumps({"text": "who ate the cake at the party?"})])

    await agentic_mod._generate_refined_query(
        original_query="who ate cake?",
        candidates=_candidates(),
        missing_info=[],
        llm=client,
        custom_instructions=None,
    )

    _assert_no_slot_reached_the_model(client)
