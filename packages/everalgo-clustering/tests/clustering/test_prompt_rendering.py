"""The cluster-assignment user prompt renders against the field set ``algorithm.py`` supplies.

A template and its call site can drift apart — a renamed slot, a field dropped from the call — and
the only visible symptom would be a literal ``{{clusters_json}}`` reaching the model.
"""

from __future__ import annotations

import re

from everalgo.clustering.prompts.en.cluster import CLUSTER_LLM_ASSIGN_SYSTEM_PROMPT, CLUSTER_LLM_ASSIGN_USER_PROMPT
from everalgo.utils import render_prompt

_PLACEHOLDER_LEFT_BEHIND = re.compile(r"\{\{[A-Za-z_][A-Za-z0-9_]*\}\}")


def test_user_prompt_renders_with_no_placeholder_left_behind() -> None:
    rendered = render_prompt(
        CLUSTER_LLM_ASSIGN_USER_PROMPT,
        None,
        memcell_text="fix the payment timeout",
        clusters_json='[{"idx": 0, "intent": "payments"}]',
    )
    assert _PLACEHOLDER_LEFT_BEHIND.search(rendered) is None, rendered


def test_system_prompt_renders_with_no_field() -> None:
    assert render_prompt(CLUSTER_LLM_ASSIGN_SYSTEM_PROMPT, None) == CLUSTER_LLM_ASSIGN_SYSTEM_PROMPT
