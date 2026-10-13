"""Every shipped USER template renders against the field set its operator supplies.

A template and its call site can drift apart — a renamed slot, a field dropped from the call — and
the only visible symptom would be a literal ``{{new_case_json}}`` reaching the model. Rendering each
template here against the fields its operator passes turns that drift into a failing test: a slot with
no field raises out of ``render_prompt``, and a slot that survives substitution is caught by the
residue assertion. The SYSTEM prompts name no slot at all, which the split test pins.
"""

from __future__ import annotations

import re

import pytest

from everalgo.agent_memory.prompts.case_compress import AGENT_CASE_COMPRESS_USER_PROMPT
from everalgo.agent_memory.prompts.case_filter import AGENT_CASE_FILTER_USER_PROMPT
from everalgo.agent_memory.prompts.profile_update import AGENT_PROFILE_UPDATE_USER_PROMPT
from everalgo.agent_memory.prompts.skill_maturity import AGENT_SKILL_MATURITY_SCORE_USER_PROMPT
from everalgo.agent_memory.prompts.skill_success import SKILL_EXTRACT_USER_PROMPT
from everalgo.agent_memory.prompts.tool_pre_compress import AGENT_TOOL_PRE_COMPRESS_USER_PROMPT
from everalgo.utils import render_prompt

_PLACEHOLDER_LEFT_BEHIND = re.compile(r"\{\{[A-Za-z_][A-Za-z0-9_]*\}\}")

_SKILL_FIELDS = {"new_case_json": "CASE", "existing_skills_json": "SKILLS"}

#: Field sets mirror the ``render_prompt`` call sites in ``case.py`` / ``profile.py`` / ``skill.py``
#: / ``skill_ops.py``.
_PROMPTS = [
    (AGENT_CASE_COMPRESS_USER_PROMPT, {"messages": "MSGS"}),
    (AGENT_CASE_FILTER_USER_PROMPT, {"messages": "MSGS"}),
    (AGENT_TOOL_PRE_COMPRESS_USER_PROMPT, {"messages_json": "MSGS", "new_count": 3}),
    (
        AGENT_PROFILE_UPDATE_USER_PROMPT,
        {"soul_md": "SOUL", "agents_md": "AGENTS", "pending_signals": "[]", "conversation": "CONV"},
    ),
    (SKILL_EXTRACT_USER_PROMPT, _SKILL_FIELDS),
    (
        AGENT_SKILL_MATURITY_SCORE_USER_PROMPT,
        {"name": "N", "description": "D", "content": "C", "confidence": 0.5},
    ),
]


@pytest.mark.parametrize(("template", "fields"), _PROMPTS)
def test_prompt_renders_with_no_placeholder_left_behind(template: str, fields: dict[str, object]) -> None:
    rendered = render_prompt(template, None, **fields)
    assert _PLACEHOLDER_LEFT_BEHIND.search(rendered) is None, rendered
