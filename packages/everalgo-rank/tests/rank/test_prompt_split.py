"""The rank prompts' system / user split: every old rule line kept, data only in the user message.

The nine single-message prompts the split replaced are pinned below, verbatim from
``git show c00ebe8:packages/everalgo-rank/src/everalgo/rank/prompts/<lang>/<name>.py`` except for trailing spaces,
which the comparison strips anyway. Every non-blank line of each, after the changes the split made everywhere
(``**`` and inline backticks dropped, a ``### `` heading prefix dropped), is a line of the new system prompt, or is
listed in ``_RESTRUCTURED`` under the structural change that removed it:

    F1  data, its label, or the separator around it moved to the user message
    F2  a Markdown heading with nothing else to say became a tag
    F3  the rerank candidates line points at ``<candidates>`` now that the data follows in the user message
    F5  an output-format line gave way to ``<reply_format>``: a reply sentence plus a parseable skeleton
    F6  a closing cue that introduced the data went with it

No line was rewritten for its content, so there is no C class. The lines the system prompt adds are listed in
``_ADDED``: the reply sentences, the parseable skeleton lines, and the rerank pointer lines.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

import pytest

from everalgo.rank._replies import _MultiQueryReply, _RefinedQueryReply, _RerankReply, _SufficiencyReply
from everalgo.rank.prompts.en.case import CASE_RERANK_PROMPT_EN
from everalgo.rank.prompts.en.episodic import EPISODIC_RERANK_PROMPT_EN
from everalgo.rank.prompts.en.multi_query import MULTI_QUERY_PROMPT_EN, MULTI_QUERY_USER_PROMPT
from everalgo.rank.prompts.en.refined_query import REFINED_QUERY_PROMPT_EN, REFINED_QUERY_USER_PROMPT
from everalgo.rank.prompts.en.skill import SKILL_RERANK_PROMPT_EN
from everalgo.rank.prompts.en.sufficiency import SUFFICIENCY_CHECK_PROMPT_EN, SUFFICIENCY_CHECK_USER_PROMPT
from everalgo.rank.prompts.rerank_user import RERANK_USER_PROMPT
from everalgo.rank.prompts.zh.case import CASE_RERANK_PROMPT_ZH
from everalgo.rank.prompts.zh.episodic import EPISODIC_RERANK_PROMPT_ZH
from everalgo.rank.prompts.zh.skill import SKILL_RERANK_PROMPT_ZH

if TYPE_CHECKING:
    from everalgo.llm import StrictOutputModel

# ``SUFFICIENCY_CHECK_PROMPT_EN`` from ``en/sufficiency.py``.
_OLD_SUFFICIENCY = """You are an expert in information retrieval evaluation. Assess whether the retrieved documents provide sufficient information to answer the user's query.

--------------------------
User Query:
{{query}}

Retrieved Documents:
{{retrieved_docs}}
--------------------------

### Instructions:

1. **Analyze the Query's Needs**
   - **Entities**: Who/What is being asked about?
   - **Attributes**: What specific details (color, time, location, quantity)?
   - **Time**: Does it ask for a specific time (absolute or relative like "last week")?

2. **Evaluate Document Evidence**
   - Check **Content**: Do the documents mention the entities and attributes?
   - Check **Dates**:
     - Use the `Date` field of each document.
     - For relative time queries (e.g., "last week", "yesterday"), verify if document dates fall within that timeframe.
     - If the query asks "When did X happen?", do you have the specific date or just a vague mention?

3. **Judgment Logic**
   - **Sufficient**: You can answer the query *completely* and *precisely* using ONLY the provided documents.
   - **Insufficient**:
     - The specific entity is not found.
     - The entity is found, but the specific attribute (e.g., "price") is missing.
     - The time reference cannot be resolved (e.g., doc says "yesterday" but has no date, or doc date doesn't match query timeframe).
     - Conflicting information without resolution.

### Output Format (strict JSON):
{
  "is_sufficient": true or false,
  "reasoning": "Brief explanation. If insufficient, state WHY (e.g., 'Found X but missing date', 'No mention of Y').",
  "key_information_found": ["Fact 1 (Source: Doc 1)", "Fact 2 (Source: Doc 2)"],
  "missing_information": ["Specific gap 1", "Specific gap 2"]
}

Now evaluate:"""

# ``MULTI_QUERY_PROMPT_EN`` from ``en/multi_query.py``.
_OLD_MULTI_QUERY = """You are an expert at query reformulation for conversational memory retrieval.
Your goal is to generate 2-3 complementary queries to find the MISSING information.

--------------------------
Original Query:
{{original_query}}

Key Information Found:
{{key_info}}

Missing Information:
{{missing_info}}

Retrieved Documents (Context):
{{retrieved_docs}}
--------------------------

### Strategy Selection (Choose based on WHY info is missing):

**[A] Pivot / Entity Association (If entity is missing)**
- If the specific entity is not found, search for related entities or broader categories.
- Example: "manager's feedback" not found → try "performance review", "work evaluation".

**[B] Temporal Calculation (If time is missing/unclear)**
- Use `Date` from Retrieved Documents to anchor relative times.
- Example: doc dated 2024-03-15 mentions "last month" → search for "February 2024".
- Search for the *event* to find its timestamp: "When did the deadline change?"

**[C] Concept Expansion (If vocabulary mismatch)**
- Synonyms: "residence" → "living", "staying at", "moved to".
- General/Specific: "Italian cuisine" ↔ "pasta", "pizza", "restaurant".

**[D] Constraint Relaxation (If too specific)**
- If "quarterly sales report from Q3" fails, try "sales report", "Q3 results".
- Remove one constraint at a time.

### Query Style Requirements (Use DIFFERENT styles):

1. **Keyword Combo** (2-5 words): Key entities only. High recall.
   - e.g., "project deadline", "vacation plans summer"
2. **Natural Question** (5-10 words): Rephrased question.
   - e.g., "When was the meeting scheduled?", "What was discussed about the budget?"
3. **Hypothetical Statement** (HyDE, 5-10 words): A likely sentence in the memory.
   - e.g., "We decided to postpone the launch", "The client requested changes"

### Requirements:
- Generate 2-3 queries.
- **CRITICAL**: Use the strategies above to target the *Missing Information*.
- Keep queries SHORT and SEARCHABLE.

### Output Format (STRICT JSON):
{
  "queries": [
    "Query 1",
    "Query 2",
    "Query 3"
  ],
  "reasoning": "Strategy used for each query (e.g., Q1: Pivot, Q2: Temporal)"
}

Now generate:
"""

# ``REFINED_QUERY_PROMPT_EN`` from ``en/refined_query.py``.
_OLD_REFINED_QUERY = """You are an expert at query reformulation for information retrieval.

**Task**: Generate a refined query that targets the missing information in the retrieved results.

**Original Query**:
{{original_query}}

**Retrieved Documents** (insufficient):
{{retrieved_docs}}

**Missing Information**:
{{missing_info}}

**Instructions**:
1. Keep the core intent of the original query unchanged.
2. Add specific keywords or rephrase to target the missing information.
3. Make the query more specific and focused.
4. The refined query should be a direct question that seeks to extract the missing facts.
5. Do NOT change the query's meaning or make it too broad.
6. Keep it concise (1-2 sentences maximum).

**Examples**:

Example 1:
Original Query: "What happened at the conference?"
Missing Info: ["specific sessions attended", "key takeaways"]
Refined Query: "Which sessions were attended at the conference and what were the main takeaways?"

Example 2:
Original Query: "Tell me about the deadline"
Missing Info: ["deadline date", "deliverables", "responsible team"]
Refined Query: "When is the deadline, what needs to be delivered, and who is responsible?"

Example 3:
Original Query: "The new policy"
Missing Info: ["policy details", "effective date", "affected departments"]
Refined Query: "What are the details of the new policy, when does it take effect, and which departments are affected?"

Now generate the refined query (output only the refined query, no additional text):
Original Query: {{original_query}}
Missing Info: {{missing_info}}

Refined Query:
"""

# ``CASE_RERANK_PROMPT_EN`` from ``en/case.py``.
_OLD_CASE_EN = """Determine whether each agent execution case's task and approach are useful for addressing the user query, preferring cases whose verified successful approach (high `quality_score`) directly applies to the query's problem. Score each case from 0.0 (no transferable experience) to 1.0 (same task pattern with a verified successful approach).

User query:
{{query}}

Candidates (JSON array; each item has `id`, `score`, plus metadata like `task_intent`, `approach`, `quality_score`, `key_insight`):
{{candidates_json}}

Return at most {{top_k}} items, sorted by score descending. Drop candidates whose score would be near 0.

Output strictly the following JSON, no prose:
{"ranked": [{"id": "<candidate_id>", "score": <float in 0..1>}]}
"""

# ``EPISODIC_RERANK_PROMPT_EN`` from ``en/episodic.py``.
_OLD_EPISODIC_EN = """Given the user query and a list of retrieved episodic memories (or atomic facts extracted from them), determine whether each candidate contains information relevant to answering the query. Score each candidate from 0.0 (no relevant information) to 1.0 (directly answers the query).

User query:
{{query}}

Candidates (JSON array; each item has `id`, `score`, an `item_type` of either `episode` or `atomic_fact`, plus metadata like `episode` / `subject` / `summary` / `parent_episode_id`):
{{candidates_json}}

Return at most {{top_k}} items, sorted by score descending. Drop candidates whose score would be near 0.

Output strictly the following JSON, no prose:
{"ranked": [{"id": "<candidate_id>", "score": <float in 0..1>}]}
"""

# ``SKILL_RERANK_PROMPT_EN`` from ``en/skill.py``.
_OLD_SKILL_EN = """Determine whether each agent skill's methodology and domain are applicable to the user query, preferring same-domain skills with directly relevant steps. Score each skill from 0.0 (not applicable at all) to 1.0 (methodology and domain both strongly apply).

Evaluate each skill considering:
- Whether the skill's steps or approach are applicable to the query's problem type
- Whether the skill's target domain (shown in its description, trigger scenarios, and keywords) overlaps with the query's subject matter — same-domain skills should be scored higher

Score bands:
- **0.0**: Completely irrelevant — no applicable methodology or domain connection
- **0.1-0.3**: Weakly related — methodology could loosely apply but domain is different
- **0.4-0.6**: Moderately helpful — useful methodology with partial domain overlap
- **0.7-0.8**: Helpful — applicable approach with good domain alignment
- **0.9-1.0**: Highly relevant — strong fit in both approach and domain

User query:
{{query}}

Candidates (JSON array; each item has `id`, `score`, plus metadata like `name`, `description`, `content`, `confidence`):
{{candidates_json}}

Return at most {{top_k}} items, sorted by score descending. Drop candidates whose score would be near 0.

Output strictly the following JSON, no prose:
{"ranked": [{"id": "<candidate_id>", "score": <float in 0..1>}]}
"""

# ``CASE_RERANK_PROMPT_ZH`` from ``zh/case.py``.
_OLD_CASE_ZH = """判断每个 agent 执行案例（case）的任务和方法是否对回答用户查询有用，优先选择 `quality_score` 高、经过验证成功的方法且直接适用于查询问题的 case。按 0.0（没有可迁移的经验）到 1.0（任务模式相同且方法已被验证成功）给每个 case 打分。

用户查询：
{{query}}

候选列表（JSON 数组；每项含 `id`、`score`，以及元数据如 `task_intent` / `approach` / `quality_score` / `key_insight`）：
{{candidates_json}}

最多返回 {{top_k}} 项，按 score 降序排列。分数接近 0 的候选直接丢弃。

只输出以下 JSON，不要包含任何解释文字：
{"ranked": [{"id": "<候选 id>", "score": <0..1 浮点数>}]}
"""  # noqa: RUF001 — verbatim old prompt, full-width punctuation included

# ``EPISODIC_RERANK_PROMPT_ZH`` from ``zh/episodic.py``.
_OLD_EPISODIC_ZH = """给定用户查询和一组检索到的 episodic 记忆候选（其中可能混入了由 episode 展开出的 atomic_fact 项），判断每个候选是否包含与查询相关的信息。按 0.0（不包含相关信息）到 1.0（直接回答查询）给每个候选打分。

用户查询：
{{query}}

候选列表（JSON 数组；每项含 `id`、`score`、`item_type` 为 `episode` 或 `atomic_fact`，以及元数据如 `episode` / `subject` / `summary` / `parent_episode_id`）：
{{candidates_json}}

最多返回 {{top_k}} 项，按 score 降序排列。分数接近 0 的候选直接丢弃。

只输出以下 JSON，不要包含任何解释文字：
{"ranked": [{"id": "<候选 id>", "score": <0..1 浮点数>}]}
"""  # noqa: RUF001 — verbatim old prompt, full-width punctuation included

# ``SKILL_RERANK_PROMPT_ZH`` from ``zh/skill.py``.
_OLD_SKILL_ZH = """判断每个 agent skill 的方法（methodology）和领域（domain）是否适用于用户查询，优先选择同领域且步骤直接相关的 skill。按 0.0（完全不适用）到 1.0（方法和领域都强适配）给每个 skill 打分。

评估每个 skill 时同时考虑：
- skill 的步骤或方法（steps / approach）是否适用于查询的问题类型
- skill 的目标领域（由 description / 触发场景 / 关键词体现）是否与查询的主题重叠 —— 同领域的 skill 应该打更高分

打分区间：
- **0.0**: 完全不相关 —— 方法不适用、领域也不重叠
- **0.1-0.3**: 弱相关 —— 方法勉强能套，但领域不一样
- **0.4-0.6**: 中等有帮助 —— 方法可用，领域部分重叠
- **0.7-0.8**: 有帮助 —— 方法适用且领域对齐良好
- **0.9-1.0**: 高度相关 —— 方法和领域都强匹配

用户查询：
{{query}}

候选列表（JSON 数组；每项含 `id`、`score`，以及元数据如 `name` / `description` / `content` / `confidence`）：
{{candidates_json}}

最多返回 {{top_k}} 项，按 score 降序排列。分数接近 0 的候选直接丢弃。

只输出以下 JSON，不要包含任何解释文字：
{"ranked": [{"id": "<候选 id>", "score": <0..1 浮点数>}]}
"""  # noqa: RUF001 — verbatim old prompt, full-width punctuation included

_SEPARATOR = "--------------------------"
_REPLY_SENTENCE = "Reply with one JSON object, nothing else."
_CASE_EN_CANDIDATES = (
    "Candidates (JSON array; each item has id, score, plus metadata like task_intent, approach, quality_score, "
    "key_insight):"
)
_EPISODIC_EN_CANDIDATES = (
    "Candidates (JSON array; each item has id, score, an item_type of either episode or atomic_fact, plus metadata "
    "like episode / subject / summary / parent_episode_id):"
)
_SKILL_EN_CANDIDATES = (
    "Candidates (JSON array; each item has id, score, plus metadata like name, description, content, confidence):"
)
_CASE_ZH_CANDIDATES = (
    "候选列表（JSON 数组；每项含 id、score，以及元数据如 task_intent / approach / quality_score / key_insight）："  # noqa: RUF001
)
_EPISODIC_ZH_CANDIDATES = (
    "候选列表（JSON 数组；每项含 id、score、item_type 为 episode 或 atomic_fact，以及元数据如 episode / subject / "  # noqa: RUF001
    "summary / parent_episode_id）："  # noqa: RUF001
)
_SKILL_ZH_CANDIDATES = (
    "候选列表（JSON 数组；每项含 id、score，以及元数据如 name / description / content / confidence）："  # noqa: RUF001
)
_EN_SKELETON = '{"ranked": [{"id": "<candidate_id>", "score": <float in 0..1>}]}'
_ZH_SKELETON = '{"ranked": [{"id": "<候选 id>", "score": <0..1 浮点数>}]}'
_EN_NEW_SKELETON = '{"ranked": [{"id": "<candidate_id>", "score": 0.9}]}'
_ZH_NEW_SKELETON = '{"ranked": [{"id": "<候选 id>", "score": 0.9}]}'


def _pointer(candidates_line: str) -> str:
    """The F3 rewrite of a rerank candidates line: it names ``<candidates>`` and closes as a sentence."""
    if candidates_line.startswith("Candidates ("):
        return candidates_line.replace("Candidates (", "Candidates in <candidates> (", 1)[:-1] + "."
    return candidates_line.replace("候选列表（", "候选列表在 <candidates> 中（", 1)[:-1] + "。"  # noqa: RUF001


def _rerank_registry(candidates_line: str, skeleton: str, *, zh: bool) -> dict[str, str]:
    label = "用户查询：" if zh else "User query:"  # noqa: RUF001
    return {label: "F1", "{{query}}": "F1", "{{candidates_json}}": "F1", candidates_line: "F3", skeleton: "F5"}


_PROMPTS: dict[str, tuple[str, str, str]] = {
    # name: (old single-message prompt, new system prompt, new user prompt)
    "sufficiency": (_OLD_SUFFICIENCY, SUFFICIENCY_CHECK_PROMPT_EN, SUFFICIENCY_CHECK_USER_PROMPT),
    "multi_query": (_OLD_MULTI_QUERY, MULTI_QUERY_PROMPT_EN, MULTI_QUERY_USER_PROMPT),
    "refined_query": (_OLD_REFINED_QUERY, REFINED_QUERY_PROMPT_EN, REFINED_QUERY_USER_PROMPT),
    "case_en": (_OLD_CASE_EN, CASE_RERANK_PROMPT_EN, RERANK_USER_PROMPT),
    "episodic_en": (_OLD_EPISODIC_EN, EPISODIC_RERANK_PROMPT_EN, RERANK_USER_PROMPT),
    "skill_en": (_OLD_SKILL_EN, SKILL_RERANK_PROMPT_EN, RERANK_USER_PROMPT),
    "case_zh": (_OLD_CASE_ZH, CASE_RERANK_PROMPT_ZH, RERANK_USER_PROMPT),
    "episodic_zh": (_OLD_EPISODIC_ZH, EPISODIC_RERANK_PROMPT_ZH, RERANK_USER_PROMPT),
    "skill_zh": (_OLD_SKILL_ZH, SKILL_RERANK_PROMPT_ZH, RERANK_USER_PROMPT),
}

# Old lines, normalized as ``_old_lines`` does, that the split removed on purpose, each with the change that removed it.
_RESTRUCTURED: dict[str, dict[str, str]] = {
    "sufficiency": {
        _SEPARATOR: "F1",
        "User Query:": "F1",
        "{{query}}": "F1",
        "Retrieved Documents:": "F1",
        "{{retrieved_docs}}": "F1",
        "Instructions:": "F2",
        "Output Format (strict JSON):": "F5",
        '"is_sufficient": true or false,': "F5",
        "Now evaluate:": "F6",
    },
    "multi_query": {
        _SEPARATOR: "F1",
        "Original Query:": "F1",
        "{{original_query}}": "F1",
        "Key Information Found:": "F1",
        "{{key_info}}": "F1",
        "Missing Information:": "F1",
        "{{missing_info}}": "F1",
        "Retrieved Documents (Context):": "F1",
        "{{retrieved_docs}}": "F1",
        "Requirements:": "F2",
        "Output Format (STRICT JSON):": "F5",
        "Now generate:": "F6",
    },
    "refined_query": {
        "Original Query:": "F1",
        "{{original_query}}": "F1",
        "Retrieved Documents (insufficient):": "F1",
        "{{retrieved_docs}}": "F1",
        "Missing Information:": "F1",
        "{{missing_info}}": "F1",
        "Original Query: {{original_query}}": "F1",
        "Missing Info: {{missing_info}}": "F1",
        "Instructions:": "F2",
        "Examples:": "F2",
        "Now generate the refined query (output only the refined query, no additional text):": "F5",
        "Refined Query:": "F6",
    },
    "case_en": _rerank_registry(_CASE_EN_CANDIDATES, _EN_SKELETON, zh=False),
    "episodic_en": _rerank_registry(_EPISODIC_EN_CANDIDATES, _EN_SKELETON, zh=False),
    "skill_en": _rerank_registry(_SKILL_EN_CANDIDATES, _EN_SKELETON, zh=False),
    "case_zh": _rerank_registry(_CASE_ZH_CANDIDATES, _ZH_SKELETON, zh=True),
    "episodic_zh": _rerank_registry(_EPISODIC_ZH_CANDIDATES, _ZH_SKELETON, zh=True),
    "skill_zh": _rerank_registry(_SKILL_ZH_CANDIDATES, _ZH_SKELETON, zh=True),
}
_STRUCTURAL_CHANGES = {"F1", "F2", "F3", "F5", "F6"}

# Non-tag lines of the new system prompts that no old line matches.
_ADDED: dict[str, set[str]] = {
    "sufficiency": {f"{_REPLY_SENTENCE} is_sufficient is true or false.", '"is_sufficient": false,'},
    "multi_query": {_REPLY_SENTENCE},
    "refined_query": {
        f"{_REPLY_SENTENCE} text is the refined query only, with no additional text.",
        '{"text": "..."}',
    },
    "case_en": {_pointer(_CASE_EN_CANDIDATES), _EN_NEW_SKELETON},
    "episodic_en": {_pointer(_EPISODIC_EN_CANDIDATES), _EN_NEW_SKELETON},
    "skill_en": {_pointer(_SKILL_EN_CANDIDATES), _EN_NEW_SKELETON},
    "case_zh": {_pointer(_CASE_ZH_CANDIDATES), _ZH_NEW_SKELETON},
    "episodic_zh": {_pointer(_EPISODIC_ZH_CANDIDATES), _ZH_NEW_SKELETON},
    "skill_zh": {_pointer(_SKILL_ZH_CANDIDATES), _ZH_NEW_SKELETON},
}

_AGENTIC = ["sufficiency", "multi_query", "refined_query"]
_RERANK = ["case_en", "episodic_en", "skill_en", "case_zh", "episodic_zh", "skill_zh"]
_REPLY_MODEL: dict[str, type[StrictOutputModel]] = {
    "sufficiency": _SufficiencyReply,
    "multi_query": _MultiQueryReply,
    "refined_query": _RefinedQueryReply,
    **dict.fromkeys(_RERANK, _RerankReply),
}
_USER_SLOTS = {
    "sufficiency": ["query", "retrieved_docs"],
    "multi_query": ["original_query", "key_info", "missing_info", "retrieved_docs"],
    "refined_query": ["original_query", "retrieved_docs", "missing_info"],
    **{name: ["query", "candidates_json"] for name in _RERANK},
}
_USER_TAGS = {
    "sufficiency": ["user_query", "retrieved_documents"],
    "multi_query": ["original_query", "key_information_found", "missing_information", "retrieved_documents"],
    "refined_query": ["original_query", "retrieved_documents", "missing_information"],
    **{name: ["user_query", "candidates"] for name in _RERANK},
}

_SLOT = re.compile(r"\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}")
_TAG_LINE = re.compile(r"^</?[a-z_]+(?: [^>]*)?>$")
_ALL = pytest.mark.parametrize("name", list(_PROMPTS))


def _old_lines(old: str) -> list[str]:
    """The old prompt's non-blank lines with the split's prompt-wide changes applied."""
    lines = (line.strip().replace("**", "").replace("`", "") for line in old.splitlines())
    return [line.removeprefix("### ") for line in lines if line]


def _new_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _block(tag: str, text: str) -> str:
    return text.split(f"<{tag}>\n", 1)[1].split(f"\n</{tag}>", 1)[0]


# ── Every old rule line is kept ─────────────────────────────────────────────────────────────────────


@_ALL
def test_every_kept_old_line_is_a_line_of_the_system_prompt(name: str) -> None:
    old, system, _ = _PROMPTS[name]
    system_lines = set(_new_lines(system))
    kept = [line for line in _old_lines(old) if line not in _RESTRUCTURED[name]]
    missing = [line for line in kept if line not in system_lines]

    assert len(kept) >= 3, "the pinned old prompt yielded too few lines, so little was checked"
    assert not missing


@_ALL
def test_every_restructured_line_is_a_line_of_the_old_prompt(name: str) -> None:
    """An exclusion that matches no old line would hide nothing today and anything after the next edit."""
    assert set(_RESTRUCTURED[name]) <= set(_old_lines(_PROMPTS[name][0]))


@_ALL
def test_every_removal_is_structural(name: str) -> None:
    assert set(_RESTRUCTURED[name].values()) <= _STRUCTURAL_CHANGES


@_ALL
def test_restructured_lines_are_gone_from_the_system_prompt(name: str) -> None:
    system_lines = set(_new_lines(_PROMPTS[name][1]))

    assert not [line for line in _RESTRUCTURED[name] if line in system_lines]


@_ALL
def test_the_system_prompt_adds_only_the_listed_lines(name: str) -> None:
    old, system, _ = _PROMPTS[name]
    added = {line for line in _new_lines(system) if not _TAG_LINE.match(line) and line not in set(_old_lines(old))}

    assert added == _ADDED[name]


@_ALL
def test_role_sentence_opens_the_system_prompt_unchanged(name: str) -> None:
    old, system, _ = _PROMPTS[name]

    assert _new_lines(system)[0] == _old_lines(old)[0]


@pytest.mark.parametrize("name", _RERANK)
def test_rerank_pointer_names_the_user_message_block(name: str) -> None:
    pointer = next(line for line in _ADDED[name] if "<candidates>" in line)

    assert pointer in _PROMPTS[name][1]
    assert "<candidates>" in RERANK_USER_PROMPT


# ── Structure: rules in system, data in user ────────────────────────────────────────────────────────


@_ALL
def test_system_prompt_carries_no_markdown(name: str) -> None:
    system = _PROMPTS[name][1]

    assert "**" not in system
    assert "`" not in system
    assert re.search(r"^#", system, re.MULTILINE) is None


@_ALL
def test_system_prompt_blocks_are_closed_and_end_with_the_reply_format(name: str) -> None:
    system = _PROMPTS[name][1]
    opened = re.findall(r"^<([a-z_]+)>$", system, re.MULTILINE)

    assert opened == re.findall(r"^</([a-z_]+)>$", system, re.MULTILINE)
    assert opened[-1] == "reply_format"
    assert system.endswith("\n</reply_format>")


@pytest.mark.parametrize("name", _AGENTIC)
def test_agentic_system_prompt_names_no_slot(name: str) -> None:
    """The system message is sent as written, so it is the same text on every call."""
    assert _SLOT.findall(_PROMPTS[name][1]) == []


@pytest.mark.parametrize("name", _RERANK)
def test_rerank_system_prompt_names_only_top_k(name: str) -> None:
    assert _SLOT.findall(_PROMPTS[name][1]) == ["top_k"]


@_ALL
def test_user_prompt_carries_only_the_data_slots_in_the_old_order(name: str) -> None:
    user = _PROMPTS[name][2]

    assert _SLOT.findall(user) == _USER_SLOTS[name]
    assert re.findall(r"^<([a-z_]+)", user, re.MULTILINE) == _USER_TAGS[name]
    assert re.sub(r"</?[a-z_]+[^>]*>|\{\{[a-z_]+\}\}", "", user).strip() == ""


@_ALL
def test_every_old_data_slot_reaches_the_user_prompt(name: str) -> None:
    old, _, user = _PROMPTS[name]

    assert set(_SLOT.findall(old)) - {"top_k"} == set(_SLOT.findall(user))


def test_refined_query_documents_keep_the_insufficient_note_as_an_attribute() -> None:
    assert '<retrieved_documents status="insufficient">' in REFINED_QUERY_USER_PROMPT


# ── The reply format ────────────────────────────────────────────────────────────────────────────────


@_ALL
def test_reply_format_skeleton_validates_against_the_reply_model(name: str) -> None:
    """The skeleton is what the model imitates (llm-contract §4.3); it must pass the model requested."""
    sentence, skeleton = _block("reply_format", _PROMPTS[name][1]).split("\n", 1)

    assert "JSON" in sentence
    _REPLY_MODEL[name].model_validate(json.loads(skeleton))


@pytest.mark.parametrize("name", _AGENTIC)
def test_agentic_reply_skeleton_lists_the_fields_in_model_order(name: str) -> None:
    skeleton = _block("reply_format", _PROMPTS[name][1]).split("\n", 1)[1]

    assert list(json.loads(skeleton)) == list(_REPLY_MODEL[name].model_fields)
