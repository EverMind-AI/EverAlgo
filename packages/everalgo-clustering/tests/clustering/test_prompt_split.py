"""The cluster-assignment prompt's system / user split: every old line kept or registered, data only in the user message.

The single-message prompt the split replaced (HEAD ``c00ebe8``) is pinned below verbatim. Every non-blank line of it
is a line of the new system message or of the new user template, or listed in ``_RESTRUCTURED`` with the change that
removed it. The rules and the reply skeleton must be in the system message and nowhere in the user message; the two
data placeholders must be in the user template and nowhere in the system message. The skeleton is the one line
rewritten in place (F): the old ``{"idx": <existing_idx or -1>, ...}`` was no JSON a parser accepts, so the new one
is a parseable object the reply model validates, and what its stand-in said moved into the sentence before it.
"""

from __future__ import annotations

import json

import numpy as np

from everalgo.clustering import Cluster, acluster_by_llm
from everalgo.clustering._replies import _ClusterAssignmentReply
from everalgo.clustering.prompts.en.cluster import CLUSTER_LLM_ASSIGN_SYSTEM_PROMPT, CLUSTER_LLM_ASSIGN_USER_PROMPT
from everalgo.testing import FakeLLMClient

# Verbatim from ``git show c00ebe8:packages/everalgo-clustering/src/everalgo/clustering/prompts/en/cluster.py``.
_OLD_PROMPT = """You are a clustering expert. Your goal is to group similar and related tasks together so that patterns and reusable strategies can be extracted from each cluster. Assign the new task intent to an existing cluster, or signal a new cluster if no existing cluster fits.

[How to decide]
The goal of clustering is to group cases that would produce a **specific, actionable skill** — not generic advice. Use this test: "Would an agent who solved one task in this cluster have a **concrete advantage** (reusable tools, domain knowledge, verified strategies) when facing the other tasks?"

1. **Identify two dimensions**: the task's **subject domain** (e.g., medical research, urban planning, e-commerce) and its **problem-solving pattern** (e.g., root cause analysis, constraint satisfaction, data pipeline design).
2. **Cluster by the more specific dimension**. If the domain is already narrow (e.g., "clinical trial data extraction"), domain alone is enough. If the domain is broad (e.g., "software engineering"), use the problem-solving pattern to differentiate (e.g., "performance profiling" vs. "schema migration").
3. **Do NOT merge across unrelated domains just because the strategy is similar.** "Diagnose a patient's symptoms via differential diagnosis" and "diagnose a supply chain bottleneck via constraint analysis" both use diagnostic reasoning, but involve completely different domain knowledge and belong in separate clusters.
4. Scan candidate clusters. Prefer the cluster whose existing items would **benefit most from sharing a skill** with the new task.
5. Signal a new cluster (idx: -1) only when no candidate cluster is a good fit.

[Candidate Clusters]
Each cluster is represented by its list index (idx), item count, and recent preview texts.
{{clusters_json}}

[New Task Intent]
{{memcell_text}}

[Rules]
- Output decision as JSON. Keep "reason" under 50 tokens.
- To assign to an existing cluster: use its idx (non-negative integer).
- To create a new cluster: use idx -1.

Return ONLY valid JSON (no markdown fences, no explanation):
{"idx": <existing_idx or -1>, "reason": "short reason"}
"""

# Old line -> the change that removed it. A bracketed heading became the tag that wraps its block; the two data
# blocks moved to the user message, whose template carries their placeholders inside tags; F made the reply skeleton
# parseable JSON.
_RESTRUCTURED = {
    "[How to decide]": "heading became <how_to_decide>",
    "[Candidate Clusters]": "heading became <candidate_clusters>, in the user message",
    "[New Task Intent]": "heading became <new_task_intent>, in the user message",
    "[Rules]": "heading became <rules>",
    "Return ONLY valid JSON (no markdown fences, no explanation):": "F: the skeleton's stand-in moved into it",
    '{"idx": <existing_idx or -1>, "reason": "short reason"}': "F: skeleton made a parseable JSON object",
}
_REWRITTEN_REPLY_FORMAT = (
    "Return ONLY valid JSON (no markdown fences, no explanation); idx is an existing idx or -1:\n"
    '{"idx": 0, "reason": "short reason"}'
)
_DATA_LINES = {"{{clusters_json}}", "{{memcell_text}}"}


def _lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip()]


def _old_lines() -> list[str]:
    return _lines(_OLD_PROMPT)


def _rule_lines() -> list[str]:
    """The old lines that are neither a heading nor a data placeholder: the rules, the examples and the skeleton."""
    return [line for line in _old_lines() if line not in _RESTRUCTURED and line not in _DATA_LINES]


def _two_clusters() -> list[Cluster]:
    return [
        Cluster(centroid=np.array([1.0, 0.0], dtype=np.float32), last_ts=1, count=3, preview=["fix login bug"]),
        Cluster(centroid=np.array([0.0, 1.0], dtype=np.float32), last_ts=1, count=2, preview=["refactor auth"]),
    ]


def test_every_old_line_is_kept_or_registered() -> None:
    new_lines = set(_lines(CLUSTER_LLM_ASSIGN_SYSTEM_PROMPT)) | set(_lines(CLUSTER_LLM_ASSIGN_USER_PROMPT))
    missing = [line for line in _old_lines() if line not in new_lines and line not in _RESTRUCTURED]
    assert not missing


def test_every_registered_line_is_an_old_line_the_new_prompt_dropped() -> None:
    new_lines = set(_lines(CLUSTER_LLM_ASSIGN_SYSTEM_PROMPT)) | set(_lines(CLUSTER_LLM_ASSIGN_USER_PROMPT))
    assert set(_RESTRUCTURED) <= set(_old_lines())
    assert not set(_RESTRUCTURED) & new_lines


def test_every_old_rule_line_is_in_the_system_message_and_not_in_the_user_template() -> None:
    system = set(_lines(CLUSTER_LLM_ASSIGN_SYSTEM_PROMPT))
    user = set(_lines(CLUSTER_LLM_ASSIGN_USER_PROMPT))
    rules = _rule_lines()
    assert len(rules) > 10
    assert [line for line in rules if line not in system] == []
    assert not user & set(rules)


def test_the_data_placeholders_are_only_in_the_user_template() -> None:
    assert "{{" not in CLUSTER_LLM_ASSIGN_SYSTEM_PROMPT
    assert all(placeholder in CLUSTER_LLM_ASSIGN_USER_PROMPT for placeholder in _DATA_LINES)


def test_the_new_sentence_places_the_data_and_the_old_description_follows_it() -> None:
    assert (
        "The user message holds the data: the candidate clusters in <candidate_clusters> and the new task intent "
        "in <new_task_intent>.\n"
        "Each cluster is represented by its list index (idx), item count, and recent preview texts."
    ) in CLUSTER_LLM_ASSIGN_SYSTEM_PROMPT


def test_f_rewrote_the_reply_format_into_a_sentence_and_a_parseable_skeleton() -> None:
    reply_format = CLUSTER_LLM_ASSIGN_SYSTEM_PROMPT.split("<reply_format>\n", 1)[1].split("\n</reply_format>", 1)[0]

    assert reply_format == _REWRITTEN_REPLY_FORMAT
    assert "<existing_idx or -1>" not in CLUSTER_LLM_ASSIGN_SYSTEM_PROMPT


def test_the_reply_skeleton_parses_and_validates_against_the_reply_model() -> None:
    skeleton = CLUSTER_LLM_ASSIGN_SYSTEM_PROMPT.split("\n</reply_format>", 1)[0].rsplit("\n", 1)[1]

    assert _ClusterAssignmentReply.model_validate(json.loads(skeleton)) == _ClusterAssignmentReply(
        idx=0, reason="short reason"
    )


def test_system_blocks_follow_the_documented_order() -> None:
    positions = [
        CLUSTER_LLM_ASSIGN_SYSTEM_PROMPT.index(tag) for tag in ("<how_to_decide>", "<rules>", "<reply_format>")
    ]
    assert positions == sorted(positions)


def test_user_template_is_only_the_two_tagged_data_blocks() -> None:
    assert CLUSTER_LLM_ASSIGN_USER_PROMPT == (
        "<candidate_clusters>\n{{clusters_json}}\n</candidate_clusters>\n\n"
        "<new_task_intent>\n{{memcell_text}}\n</new_task_intent>"
    )


async def test_the_sent_messages_split_rules_from_data() -> None:
    llm = FakeLLMClient(responses=['{"idx": 0, "reason": "ok"}'])

    await acluster_by_llm(
        Cluster(centroid=np.array([0.7, 0.7], dtype=np.float32), last_ts=1, preview=["investigate token leak"]),
        _two_clusters(),
        llm=llm,
    )

    system, user = llm.calls[0].messages
    assert (system.role, user.role) == ("system", "user")
    assert system.content == CLUSTER_LLM_ASSIGN_SYSTEM_PROMPT
    assert isinstance(user.content, str)
    assert "fix login bug" not in CLUSTER_LLM_ASSIGN_SYSTEM_PROMPT
    assert user.content.startswith("<candidate_clusters>\n[")
    assert user.content.endswith("<new_task_intent>\ninvestigate token leak\n</new_task_intent>")
    assert not set(_rule_lines()) & set(_lines(user.content))


async def test_a_new_cluster_without_preview_sends_an_empty_intent_block() -> None:
    llm = FakeLLMClient(responses=['{"idx": -1, "reason": "new"}'])

    await acluster_by_llm(
        Cluster(centroid=np.array([0.7, 0.7], dtype=np.float32), last_ts=1),
        _two_clusters(),
        llm=llm,
    )

    user = llm.calls[0].messages[1].content
    assert isinstance(user, str)
    assert user.endswith("<new_task_intent>\n\n</new_task_intent>")
