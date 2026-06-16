"""Document classification — one LLM call picks a ``CategorySpec.id`` from a closed set.

Pipeline placement: after the doc-level ``summary`` is finalized (post multi-batch
``content_merge``), before ``flatten``. The resulting id is denormalized onto every
emitted ``KnowledgeMemory`` via ``flatten(category_id=...)``.

The taxonomy is caller-owned business data passed in at ingest time. Out-of-set
predictions and parse failures collapse to ``""`` — the downstream retrieval-side
soft boost treats unclassified hits as a non-bucket and is robust to occasional
misses.

NOT exposed in the package ``__all__`` directly — surfaced through ``everalgo.knowledge``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from asgiref.sync import async_to_sync

from everalgo.knowledge._llm_json import parse_llm_json
from everalgo.knowledge.prompts.en.category_classify import CATEGORY_CLASSIFY_PROMPT_EN
from everalgo.llm.types import ChatMessage

if TYPE_CHECKING:
    from everalgo.llm.protocols import LLMClient
    from everalgo.types import CategorySpec

__all__ = ["aclassify_category", "classify_category"]

logger = logging.getLogger(__name__)


def _render_taxonomy(categories: list[CategorySpec]) -> str:
    """Render the taxonomy block as a stable prefix (one entry per line)."""
    return "\n".join(f"- {c.id}: {c.description}" for c in categories)


async def aclassify_category(
    client: LLMClient,
    title: str,
    doc_summary: str,
    categories: list[CategorySpec],
    *,
    prompt: str = CATEGORY_CLASSIFY_PROMPT_EN,
) -> str:
    """Classify a document into one ``CategorySpec`` from the supplied taxonomy.

    One LLM call. The taxonomy block lives in the prompt prefix so it stays stable
    across documents in the same ingest run (prompt caching friendly).

    Args:
        client: Caller-injected ``LLMClient``. Algorithm code never holds a client.
        title: Document title — used as classifier input alongside the summary.
        doc_summary: Finalized document summary (post merge in the multi-batch path).
        categories: Closed taxonomy. An empty list short-circuits to ``""`` without
            issuing any LLM call.
        prompt: Override the classifier prompt template. Must accept ``{taxonomy}``,
            ``{title}``, ``{doc_summary}`` placeholders.

    Returns:
        The chosen ``CategorySpec.id``; ``""`` when categories is empty, the LLM
        response fails to parse, or the predicted id is not in the closed set.
    """
    if not categories:
        return ""

    rendered = prompt.format(
        taxonomy=_render_taxonomy(categories),
        title=title or "Untitled",
        doc_summary=doc_summary,
    )
    response = await client.chat([ChatMessage(role="user", content=rendered)])
    data = parse_llm_json(response.content)
    if data is None:
        logger.warning("classify_category: LLM JSON parse failed; returning empty id")
        return ""

    raw = data.get("category_id", "")
    if not isinstance(raw, str):
        logger.warning("classify_category: non-string category_id %r; returning empty id", raw)
        return ""

    predicted = raw.strip()
    if predicted == "":
        return ""

    valid_ids = {c.id for c in categories}
    if predicted not in valid_ids:
        logger.warning(
            "classify_category: predicted id %r not in taxonomy (size=%d); returning empty id",
            predicted,
            len(valid_ids),
        )
        return ""
    return predicted


classify_category = async_to_sync(aclassify_category)
"""Sync bridge — only callable from non-event-loop contexts."""
