"""KnowledgeExtractor — ParsedContent → list[KnowledgeMemory].

Orchestrates the four-stage knowledge extraction pipeline:

1. **Preprocess + atomize** (``_block_split``): normalize parsed text into
   numbered atom blocks (paragraphs, merged tables, merged lists).
2. **Topic-tree LLM extraction** (``TOPIC_TREE_EXTRACT_PROMPT_EN``): one
   LLM call per token-bounded batch produces a nested ``topics`` JSON
   plus document-level ``summary`` / ``content_labels``.
3. **Cross-batch merge** (``_batch_merge``, multi-batch only): one LLM
   call via ``CONTENT_MERGE_PROMPT_EN`` consolidates per-batch summaries,
   then one LLM call via ``TOPIC_MERGE_PROMPT_EN`` de-duplicates topics
   that straddle a batch boundary. Skipped entirely for single-batch
   documents.
4. **Post-processing** (``_postprocess``): two conditional LLM passes —
   split leaves that still contain multiple sub-headings, and assign any
   blocks no topic claimed.
5. **Tree assembly** (``_topic_build`` → ``_doc_root`` → ``_flatten``):
   materialize a ``TopicClip`` forest, simplify it (collapse trivial
   children + deduplicate parent refs), wrap under a synthetic doc root
   carrying the title + doc-level summary, and DFS-flatten into the
   public ``list[KnowledgeMemory]`` output.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from asgiref.sync import async_to_sync

from everalgo.knowledge._batch_merge import amerge_doc_summary, amerge_topics
from everalgo.knowledge._block_split import (
    format_numbered_paragraphs,
    preprocess_content,
    split_and_batch_content,
    split_content_to_blocks,
)
from everalgo.knowledge._classify import aclassify_category
from everalgo.knowledge._doc_root import build_doc_root
from everalgo.knowledge._flatten import flatten
from everalgo.knowledge._llm_json import parse_llm_json
from everalgo.knowledge._postprocess import apostprocess_topics
from everalgo.knowledge._topic_build import build_topic_clips
from everalgo.knowledge.prompts.en.topic_tree_extract import TOPIC_TREE_EXTRACT_PROMPT_EN
from everalgo.llm.types import ChatMessage

if TYPE_CHECKING:
    from everalgo.llm.protocols import LLMClient
    from everalgo.types import CategorySpec, KnowledgeMemory, ParsedContent

__all__ = ["KnowledgeExtractor"]

logger = logging.getLogger(__name__)


async def _aextract_batch(
    client: LLMClient,
    title: str,
    batch: list[tuple[int, str]],
    prompt_template: str,
) -> dict[str, Any] | None:
    """Single LLM call for one atom batch; returns the parsed JSON dict or None."""
    rendered = prompt_template.format(
        title=title,
        numbered_paragraphs=format_numbered_paragraphs(batch),
        num_paragraphs=len(batch),
        max_id=batch[-1][0] if batch else 0,
    )
    response = await client.chat([ChatMessage(role="user", content=rendered)])
    return parse_llm_json(response.content)


async def _arun_extraction_batches(
    client: LLMClient,
    text: str,
    title: str,
    prompt: str,
) -> tuple[list[tuple[int, str]], list[dict[str, Any]]] | None:
    """Preprocess, atomize, batch, and run per-batch LLM extraction.

    Returns ``(atoms, batch_results)`` or ``None`` when input is empty or
    every batch failed to parse.
    """
    text = preprocess_content(text)
    atoms = split_content_to_blocks(text)
    if not atoms:
        logger.warning("knowledge.aextract: empty input, no atoms")
        return None

    batches = split_and_batch_content(atoms)

    batch_results: list[dict[str, Any]] = []
    for batch in batches:
        data = await _aextract_batch(client, title, batch, prompt)
        if data is None:
            logger.warning("knowledge.aextract: batch JSON parse failed; skipping")
            continue
        batch_results.append(data)

    if not batch_results:
        logger.warning("knowledge.aextract: no batches parsed; returning empty")
        return None

    return atoms, batch_results


def _consolidate_batch_results(
    batch_results: list[dict[str, Any]],
) -> tuple[str, list[str], list[dict[str, Any]]]:
    """Combine per-batch JSON dicts into ``(doc_summary, doc_labels, all_topics)``.

    Pure-compute concatenation only. The single-batch path uses fields
    verbatim. The multi-batch path concatenates topics and union-dedupes
    labels here, while the caller drives the LLM passes in
    ``_batch_merge`` to produce the final document summary and to fold
    duplicate topics across batch boundaries.
    """
    if len(batch_results) == 1:
        only = batch_results[0]
        return (
            only.get("summary", ""),
            list(only.get("content_labels", []) or []),
            list(only.get("topics", []) or []),
        )

    doc_summary = batch_results[0].get("summary", "")
    seen_labels: set[str] = set()
    doc_labels: list[str] = []
    all_topics: list[dict[str, Any]] = []
    for r in batch_results:
        for label in r.get("content_labels", []) or []:
            if label not in seen_labels:
                seen_labels.add(label)
                doc_labels.append(label)
        all_topics.extend(r.get("topics", []) or [])
    return doc_summary, doc_labels, all_topics


class KnowledgeExtractor:
    """Extracts a flattened topic tree from one document.

    Construct with the LLM client (``KnowledgeExtractor(llm=client)``); the
    same instance can extract many documents. Customize the prompt per call
    via ``prompt=``.

    Args:
        llm: Caller-injected ``LLMClient``.
    """

    def __init__(self, *, llm: LLMClient) -> None:
        self._llm = llm

    async def aextract(
        self,
        parsed: ParsedContent,
        *,
        doc_id: str = "",
        title: str = "",
        categories: list[CategorySpec] | None = None,
        category_id: str = "",
        prompt: str = TOPIC_TREE_EXTRACT_PROMPT_EN,
    ) -> list[KnowledgeMemory]:
        """Extract a flattened topic tree from ``parsed``.

        Args:
            parsed: Single-document parser output; ``parsed.text`` carries the
                normalized content to segment.
            doc_id: Identifier copied into every emitted ``KnowledgeMemory.doc_id``.
                ``ParsedContent`` no longer carries an id, so the caller supplies
                it here.
            title: Document title used as the synthetic doc-root topic; falls back
                to ``"Untitled"`` when empty.
            categories: Optional classification taxonomy. When non-empty and
                ``category_id`` is not pre-supplied, the extractor runs a single
                LLM classification call against the finalized doc summary and
                denormalizes the chosen id onto every emitted node. The taxonomy
                is caller-owned business data; EverAlgo ships no default.
            category_id: Pre-known category id. Non-empty values short-circuit
                the classifier and are used verbatim. ``""`` (default) means
                "ask the classifier if ``categories`` is also provided".
            prompt: Override the topic-tree extraction prompt template.

        Returns:
            ``list[KnowledgeMemory]`` — index 0 is the synthetic doc-root
            (``depth=0``, ``parent_index=None``); subsequent entries are
            DFS-ordered descendants. Returns an empty list when input is empty
            or every batch failed to parse.
        """
        client = self._llm
        title = title or "Untitled"

        result = await _arun_extraction_batches(client, parsed.text, title, prompt)
        if result is None:
            return []
        atoms, batch_results = result

        doc_summary, doc_labels, all_topics = _consolidate_batch_results(batch_results)

        if len(batch_results) > 1:
            doc_summary = await amerge_doc_summary(client, title, batch_results)
            all_topics = await amerge_topics(client, title, all_topics)

        if not category_id and categories:
            category_id = await aclassify_category(client, title, doc_summary, categories)

        all_topics = await apostprocess_topics(client, title, all_topics, atoms)
        forest = build_topic_clips(all_topics, atoms)
        root = build_doc_root(
            forest,
            doc_title=title,
            doc_summary=doc_summary,
            doc_content_labels=doc_labels,
        )
        return flatten(root, doc_id=doc_id, category_id=category_id)

    extract = async_to_sync(aextract)
    """Sync bridge — only callable from non-event-loop contexts."""
