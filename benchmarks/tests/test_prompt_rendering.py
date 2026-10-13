"""The harness call sites send the model a prompt with no slot left in it.

Asserting on the templates alone would not catch the failure these calls are exposed to: the
prompts and their call sites were migrated together, and a call site that went back to
``str.format`` would turn the template's ``{{question}}`` into a literal ``{question}`` and ship
it — no exception, no missing field. So each test drives the real function and reads the prompt
it produced.

``_answer_one_question`` needs a whole ``StageContext`` to run, so its prompt is checked inside
``test_stage_answer.py``'s end-to-end case, which already builds one.

The sweep at the bottom is the harness-side half of
``everalgo-core/tests/utils/test_no_legacy_placeholders.py``: that one walks the library's
prompt packages, this one walks the harness's own templates, so neither side can drift back.
"""

from __future__ import annotations

import json
import re

import pytest

from benchmarks.common.judge import allm_judge
from benchmarks.common.stages.answer import _CONTEXT_TEMPLATE, _build_context
from benchmarks.datasets.locomo.prompts import ANSWER_PROMPT, JUDGE_PROMPT
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.utils import render_prompt

_SLOT_LEFT_BEHIND = re.compile(r"\{\{[A-Za-z_][A-Za-z0-9_]*\}\}|(?<!\{)\{[A-Za-z_][A-Za-z0-9_]*\}(?!\})")
_LEGACY_PLACEHOLDER = re.compile(r"(?<!\{)\{([A-Za-z_][A-Za-z0-9_]*)\}(?!\})")


async def test_judge_sends_a_fully_rendered_prompt() -> None:
    client = FakeLLMClient(responses=[json.dumps({"label": "CORRECT", "reasoning": "ok"})])

    await allm_judge(
        question="What did Alice eat?",
        golden_answer="cake",
        generated_answer="cake",
        judge_prompt=JUDGE_PROMPT,
        llm=client,
        num_runs=1,
    )

    sent = str(client.calls[0].messages[0].content)
    assert sent == render_prompt(
        JUDGE_PROMPT, None, question="What did Alice eat?", gold_answer="cake", response="cake"
    )


def test_build_context_renders_through_the_renderer() -> None:
    episodes = [{"subject": "Alice", "episode": "ate cake"}]

    context = _build_context(episodes, ("Alice", "Bob"))

    assert context == render_prompt(
        _CONTEXT_TEMPLATE, None, speaker_a="Alice", speaker_b="Bob", episodes="Alice: ate cake\n---"
    )
    assert _SLOT_LEFT_BEHIND.search(context) is None, context


@pytest.mark.parametrize(
    "template",
    [
        pytest.param(ANSWER_PROMPT, id="locomo.ANSWER_PROMPT"),
        pytest.param(JUDGE_PROMPT, id="locomo.JUDGE_PROMPT"),
        pytest.param(_CONTEXT_TEMPLATE, id="stages.answer._CONTEXT_TEMPLATE"),
    ],
)
def test_prompt_has_no_single_brace_placeholder(template: str) -> None:
    found = sorted(set(_LEGACY_PLACEHOLDER.findall(template)))
    assert not found, f"single-brace placeholders {found}; write them as {{{{name}}}}"
