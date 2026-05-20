"""LLM-guided agentic retrieval helpers — sufficiency check + query refinement.

Ports check_sufficiency / generate_multi_queries / generate_refined_query from
EverCore's evaluation/src/adapters/evermemos/tools/agentic_utils.py with the
exact same prompts to preserve baseline parity.

The LLM call shape used here differs from EverCore's LLMProvider.generate():
we call ``await llm.chat(messages, response_format={"type": "json_object"})``
and parse ``response.content`` as JSON.  Fallback behaviour mirrors EverCore's
conservative defaults (assume sufficient / return original query on error).
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from benchmarks.common.services import LLMClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates — verbatim ports from EverCore's prompt files:
#   evaluation/src/adapters/evermemos/prompts/sufficiency_check_prompts.py
#   evaluation/src/adapters/evermemos/prompts/multi_query_prompts.py
#   evaluation/src/adapters/evermemos/prompts/refined_query_prompts.py
# ---------------------------------------------------------------------------

_SUFFICIENCY_CHECK_PROMPT = """You are an expert in information retrieval evaluation. Assess whether the retrieved documents provide a complete and temporally sufficient answer to the user's query.
--------------------------
User Query:
{query}

Retrieved Documents:
{retrieved_docs}
--------------------------

### Instructions:

1. **Analyze the Query Structure**
   - Identify key entities AND determine if the query requires temporal reasoning.
   - If the query involves time (e.g., "before", "after", "since", "during", "from X to Y", "how long"), you MUST decompose it into:
       * start_time_needed (if any)
       * end_time_needed (if any)
       * temporal_relation_needed (ordering, duration, interval)

2. **Scan Documents for Coverage**
   - Look for explicit facts addressing *each* required component:
       * required entities
       * start time
       * end time
       * temporal relations (ordering or duration)

3. **Extract Key Information**
   - List specific resolved entities or facts found in the documents.
   - If time expressions exist, normalize them (e.g., "two weeks ago", "before she moved").

4. **Identify Missing Information**
   - For temporal queries:
        * missing start time
        * missing end time
        * missing ordering facts
        * missing duration
   - Use resolved names to be specific (e.g., "Start time of Alice moving", "Whether Bob visited before Alice moved").

5. **Judgment**
   - **Sufficient**: All required components (entities + temporal boundaries + relations) appear explicitly.
   - **Insufficient**: ANY required part is missing.

### Output Format (strict JSON):
{{
  "is_sufficient": true or false,
  "reasoning": "1-2 sentence explanation.",
  "key_information_found": ["List of resolved entities/facts"],
  "missing_information": ["Specific missing components, using resolved entity names"]
}}

Now evaluate:"""

_MULTI_QUERY_GENERATION_PROMPT = """You are an expert at query reformulation for long-term conversational retrieval.
Your goal is to generate multiple complementary search queries that recover BOTH:
- the starting point of a time interval
- the ending point of a time interval
- all temporally-linked events in between

You MUST explicitly expand temporal references (e.g., "last week", "before moving",
"when they first met") into alternative expressions.

--------------------------
Original Query:
{original_query}

Key Information Found:
{key_info}

Missing Information:
{missing_info}

Retrieved Documents:
{retrieved_docs}
--------------------------

### Temporal Reasoning Strategy (MANDATORY)
When the question involves time or order:
1. **Boundary Decomposition**
   Generate queries that separately target:
   - the earliest relevant event ("start boundary")
   - the latest relevant event ("end boundary")

2. **Temporal Expression Expansion**
   Rewrite relative time expressions into multiple equivalent forms:
   - absolute dates (if deducible)
   - session numbers
   - "before/after X"
   - duration phrasing ("two weeks earlier", "shortly after")

3. **Interval Reconstruction**
   Include a declarative query that resembles a hypothetical answer containing BOTH
   the start and end time anchors.

### Standard Query Requirements
1. Generate 2-3 diverse queries.
2. Query 1 MUST be a specific **Question**.
3. Query 2 MUST be a **Declarative Statement or Hypothetical Answer (HyDE)**.
4. Query diversity MUST include different temporal forms (before/after/during).
5. MUST use Key Info to resolve pronouns IF provided.
6. No invented facts.
7. Keep queries < 25 words, same language as original.

### Output Format (STRICT JSON):
{{
  "queries": [
    "Refined query 1",
    "Refined query 2",
    "Refined query 3 (optional)"
  ],
  "reasoning": "Brief explanation of how temporal boundaries and expressions were expanded."
}}

Now generate:
"""

_REFINED_QUERY_PROMPT = """You are an expert at query reformulation for information retrieval.

**Task**: Generate a refined query that targets the missing information in the retrieved results.

**Original Query**:
{original_query}

**Retrieved Documents** (insufficient):
{retrieved_docs}

**Missing Information**:
{missing_info}

**Instructions**:
1. Keep the core intent of the original query unchanged.
2. Add specific keywords or rephrase to target the missing information.
3. Make the query more specific and focused.
4. The refined query should be a direct question that seeks to extract the missing facts.
5. Do NOT change the query's meaning or make it too broad.
6. Keep it concise (1-2 sentences maximum).

**Examples**:

Example 1:
Original Query: "What does Alice like?"
Missing Info: ["Alice's specific interests or hobbies"]
Refined Query: "What are Alice's hobbies and interests?"

Example 2:
Original Query: "Tell me about the meeting"
Missing Info: ["meeting date", "location", "participants"]
Refined Query: "When and where was the meeting held, and who attended?"

Example 3:
Original Query: "Bob's project"
Missing Info: ["project name", "status", "purpose"]
Refined Query: "What is the name, current status, and purpose of Bob's project?"

Now generate the refined query (output only the refined query, no additional text):
Original Query: {original_query}
Missing Info: {missing_info}

Refined Query:
"""


# ---------------------------------------------------------------------------
# Document formatting helpers
# ---------------------------------------------------------------------------


def _format_documents_for_llm(
    results: list[tuple[dict[str, Any], float]],
    max_docs: int,
) -> str:
    """Format retrieval results for LLM consumption using Episode Memory format.

    Mirrors EverCore's ``format_documents_for_llm`` with ``use_episode=True``.

    Args:
        results: Retrieval results as ``[(doc, score), ...]``.
        max_docs: Maximum number of documents to include.

    Returns:
        Multi-line string with one block per document.
    """
    formatted: list[str] = []
    for i, (doc, _score) in enumerate(results[:max_docs], start=1):
        # EverCore stores ``subject`` / ``episode`` at the top level of each doc
        # (``stage1_memcells_extraction.py:495``); our schema nests them under
        # ``episode``. Reading the top-level keys here would yield "N/A" for
        # subject and a dict (not a string) for episode — silently feeding the
        # LLM dict reprs and breaking sufficiency / multi-query judgments.
        episode_dict: dict[str, Any] = doc.get("episode") or {}
        subject: str = episode_dict.get("subject") or "N/A"
        body: str = episode_dict.get("content") or "N/A"
        if len(body) > 500:
            body = body[:500] + "..."
        formatted.append(f"Document {i}:\n  Title: {subject}\n  Content: {body}\n")
    return "\n".join(formatted)


# ---------------------------------------------------------------------------
# Public async helpers
# ---------------------------------------------------------------------------


async def check_sufficiency(
    query: str,
    results: list[tuple[dict[str, Any], float]],
    *,
    llm: LLMClient,
    judge_model: str | None = None,
    max_docs: int = 10,
) -> tuple[bool, str, list[str], list[str], dict[str, int]]:
    """Check whether retrieval results sufficiently answer the query.

    Args:
        query: User query string.
        results: Top retrieval results as ``[(doc, score), ...]``.
        llm: LLMClient instance for the sufficiency judge.
        judge_model: Override model; falls back to the client's default if None.
        max_docs: Maximum documents forwarded to the LLM.

    Returns:
        ``(is_sufficient, reasoning, missing_info, key_information_found, tokens)``
        where ``tokens`` is ``{"prompt_tokens": int, "completion_tokens": int}``.
        On any error the conservative fallback is ``(True, ..., [], [], {})``
        to avoid unnecessary Round 2 on transient failures.
    """
    retrieved_docs = _format_documents_for_llm(results, max_docs)
    prompt = _SUFFICIENCY_CHECK_PROMPT.format(query=query, retrieved_docs=retrieved_docs)
    try:
        # Mirror EverCore ``check_sufficiency`` (tools/agentic_utils.py:206-210):
        # temperature=0 keeps the sufficient/insufficient verdict deterministic
        # so the Round-2 trigger doesn't flip on identical inputs across runs.
        response = await llm.chat(
            [{"role": "user", "content": prompt}],
            model=judge_model,
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=500,  # Mirror locomo-benchmark agentic_utils.py:324
        )
        data: dict[str, Any] = json.loads(response.content)
    except json.JSONDecodeError as exc:
        logger.warning("check_sufficiency: JSON parse error: %s", exc)
        return (True, f"JSON parse error: {exc}", [], [], {})
    except Exception as exc:
        logger.warning("check_sufficiency: LLM call failed: %s", exc)
        return (True, f"Error: {exc}", [], [], {})

    tokens: dict[str, int] = {
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
    }
    return (
        bool(data.get("is_sufficient", True)),
        str(data.get("reasoning", "")),
        list(data.get("missing_information", [])),
        list(data.get("key_information_found", [])),
        tokens,
    )


async def generate_multi_queries(
    original_query: str,
    results: list[tuple[dict[str, Any], float]],
    missing_info: list[str],
    *,
    llm: LLMClient,
    judge_model: str | None = None,
    key_info: list[str] | None = None,
    max_docs: int = 10,
    num_queries: int = 3,
) -> tuple[list[str], str, dict[str, int]]:
    """Generate multiple complementary refined queries for Round 2 retrieval.

    Args:
        original_query: Original user query.
        results: Top retrieval results from Round 1.
        missing_info: Missing information identified by ``check_sufficiency``.
        llm: LLMClient instance.
        judge_model: Override model; falls back to the client's default if None.
        key_info: Key information already found (for better query refinement).
        max_docs: Maximum documents forwarded to the LLM.
        num_queries: Expected number of queries to generate (hint for the LLM).

    Returns:
        ``(queries, strategy_description, tokens)`` — queries list has 1-3
        entries; ``tokens`` is ``{"prompt_tokens": int, "completion_tokens": int}``.
        Falls back to ``([original_query], ..., {})`` on any error.
    """
    retrieved_docs = _format_documents_for_llm(results, max_docs)
    missing_info_str = ", ".join(missing_info) if missing_info else "N/A"
    key_info_str = ", ".join(key_info) if key_info else "N/A"

    prompt = _MULTI_QUERY_GENERATION_PROMPT.format(
        original_query=original_query,
        retrieved_docs=retrieved_docs,
        missing_info=missing_info_str,
        key_info=key_info_str,
    )

    try:
        # Mirror EverCore ``generate_multi_queries`` (tools/agentic_utils.py:401-405):
        # temperature=0.4 to encourage diversity across the 3 generated queries.
        response = await llm.chat(
            [{"role": "user", "content": prompt}],
            model=judge_model,
            response_format={"type": "json_object"},
            temperature=0.4,
            max_tokens=300,  # Mirror locomo-benchmark agentic_utils.py:439
        )
        data = json.loads(response.content)
    except json.JSONDecodeError as exc:
        logger.warning("generate_multi_queries: JSON parse error: %s", exc)
        return ([original_query], f"Parse error: {exc}", {})
    except Exception as exc:
        logger.warning("generate_multi_queries: LLM call failed: %s", exc)
        return ([original_query], f"Error: {exc}", {})

    tokens: dict[str, int] = {
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
    }
    raw_queries: list[Any] = data.get("queries", [])
    strategy: str = str(data.get("reasoning", ""))

    valid: list[str] = [
        q.strip()
        for q in raw_queries
        if isinstance(q, str) and 5 <= len(q.strip()) <= 300 and q.strip().lower() != original_query.lower().strip()
    ][:num_queries]

    if not valid:
        return ([original_query], "Fallback: used original query", tokens)

    return (valid, strategy, tokens)


async def generate_refined_query(
    original_query: str,
    results: list[tuple[dict[str, Any], float]],
    missing_info: list[str],
    *,
    llm: LLMClient,
    judge_model: str | None = None,
    key_info: list[str] | None = None,
    max_docs: int = 10,
) -> tuple[str, dict[str, int]]:
    """Generate a single refined query (legacy single-query mode).

    Args:
        original_query: Original user query.
        results: Top retrieval results from Round 1.
        missing_info: Missing information list.
        llm: LLMClient instance.
        judge_model: Override model; falls back to the client's default if None.
        key_info: Key information found (currently unused in the prompt but kept
            for API parity with ``generate_multi_queries``).
        max_docs: Maximum documents forwarded to the LLM.

    Returns:
        Tuple of (refined query string, tokens dict).  Falls back to
        ``(original_query, {})`` on error or when the LLM output is invalid.
    """
    retrieved_docs = _format_documents_for_llm(results, max_docs)
    missing_info_str = ", ".join(missing_info) if missing_info else "N/A"

    prompt = _REFINED_QUERY_PROMPT.format(
        original_query=original_query,
        retrieved_docs=retrieved_docs,
        missing_info=missing_info_str,
    )

    try:
        # Mirror EverCore ``generate_refined_query`` (tools/agentic_utils.py:274-278):
        # temperature=0.3 for moderate query rewrite creativity.
        response = await llm.chat(
            [{"role": "user", "content": prompt}],
            model=judge_model,
            temperature=0.3,
            max_tokens=150,  # Mirror locomo-benchmark agentic_utils.py:301
        )
        refined = response.content.strip()
    except Exception as exc:
        logger.warning("generate_refined_query: LLM call failed: %s", exc)
        return original_query, {}

    tokens: dict[str, int] = {
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
    }

    # Strip common prefixes the LLM may prepend
    for prefix in ("Refined Query:", "Output:", "Answer:", "Query:"):
        if refined.startswith(prefix):
            refined = refined[len(prefix) :].strip()

    if len(refined) < 5 or len(refined) > 300:
        return original_query, tokens
    if refined.lower() == original_query.lower():
        return original_query, tokens

    return refined, tokens
