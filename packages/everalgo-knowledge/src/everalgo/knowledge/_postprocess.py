"""LLM-driven post-processing of the extracted topic tree.

Stage 4 of the KnowledgeExtractor pipeline — *two* conditional LLM passes:

1. ``asplit_unsplit_leaves`` (uses ``TOPIC_SPLIT_PROMPT_EN``): leaves with
   multiple sub-headings that the initial extraction left intact get
   inspected by the LLM, which may decide to re-split them into a
   parent / children hierarchy.

2. ``aassign_uncovered_blocks`` (uses ``UNCOVERED_ASSIGN_PROMPT_EN``):
   non-separator, non-empty atoms that no topic claimed are attached to
   the most relevant existing topic.

Each pass is a no-op when its detector returns empty (short / fully-
covered documents bypass the LLM entirely). Detectors are pure compute;
appliers are async.

NOT exposed in the package ``__all__`` — internal to the knowledge extractor.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from asgiref.sync import async_to_sync

from everalgo.knowledge._llm_json import parse_llm_json
from everalgo.knowledge._topic_build import collect_all_block_refs, parse_segment_ranges
from everalgo.knowledge.prompts.en.topic_split import TOPIC_SPLIT_PROMPT_EN
from everalgo.knowledge.prompts.en.uncovered_assign import UNCOVERED_ASSIGN_PROMPT_EN
from everalgo.llm.types import ChatMessage

if TYPE_CHECKING:
    from everalgo.llm.protocols import LLMClient

__all__ = [
    "POSTPROCESS_MIN_AVG_CHARS",
    "POSTPROCESS_MIN_TOTAL_CHARS",
    "aassign_uncovered_blocks",
    "apostprocess_topics",
    "asplit_unsplit_leaves",
    "assign_uncovered_blocks",
    "detect_uncovered_blocks",
    "detect_unsplit_leaves",
    "postprocess_topics",
    "split_unsplit_leaves",
]

logger = logging.getLogger(__name__)

# Minimum char thresholds for triggering a split pass on a leaf.
POSTPROCESS_MIN_TOTAL_CHARS = 2000
POSTPROCESS_MIN_AVG_CHARS = 500

_HEADING_RE = re.compile(r"^#{2,5}\s+")
_SEPARATOR_RE = re.compile(r"^[-*=]{3,}$")


# ── pure-compute detectors ───────────────────────────────────────────


def _scan_unsplit_leaves(
    topics: list[dict[str, Any]],
    atom_map: dict[int, str],
    path: list[int],
    results: list[dict[str, Any]],
) -> None:
    """DFS for leaf topics that need splitting based on heading count + size."""
    for i, t in enumerate(topics):
        current_path = [*path, i]
        children = t.get("children", [])
        if children:
            _scan_unsplit_leaves(children, atom_map, current_path, results)
            continue

        refs = parse_segment_ranges(t.get("block_refs", ""))
        if not refs:
            continue

        heading_ids = [r for r in refs if r in atom_map and _HEADING_RE.match(atom_map[r].strip())]
        if len(heading_ids) < 2:
            continue

        total_chars = sum(len(atom_map.get(r, "")) for r in refs)
        avg_chars = total_chars / len(heading_ids)
        if total_chars >= POSTPROCESS_MIN_TOTAL_CHARS and avg_chars >= POSTPROCESS_MIN_AVG_CHARS:
            results.append(
                {
                    "topic_name": t.get("topic", ""),
                    "path": current_path,
                    "block_ids": refs,
                    "heading_ids": heading_ids,
                    "total_chars": total_chars,
                },
            )


def detect_unsplit_leaves(
    topics: list[dict[str, Any]],
    atom_map: dict[int, str],
) -> list[dict[str, Any]]:
    """Find leaf topics with >= 2 sub-headings and enough content to warrant splitting."""
    results: list[dict[str, Any]] = []
    _scan_unsplit_leaves(topics, atom_map, [], results)
    return results


def detect_uncovered_blocks(
    topics: list[dict[str, Any]],
    atom_map: dict[int, str],
) -> list[dict[str, Any]]:
    """Find non-separator, non-empty atoms that no topic claimed."""
    covered = collect_all_block_refs(topics)
    uncovered: list[dict[str, Any]] = []
    for bid, text in sorted(atom_map.items()):
        if bid in covered:
            continue
        stripped = text.strip()
        if not stripped or _SEPARATOR_RE.match(stripped):
            continue
        uncovered.append({"block_id": bid, "text": text})
    return uncovered


# ── tree navigation helpers ──────────────────────────────────────────


def _replace_topic_at_path(
    topics: list[dict[str, Any]],
    path: list[int],
    replacement: dict[str, Any],
) -> None:
    """Replace the topic at ``path`` with a split result, preserving identity fields.

    The original topic name, summary, and content_labels are kept; only
    block_refs and children come from the replacement.
    """
    container = topics
    for idx in path[:-1]:
        if idx >= len(container):
            return
        container = container[idx].get("children", [])
    final_idx = path[-1]
    if final_idx >= len(container):
        return
    original = container[final_idx]
    container[final_idx] = {
        "topic": original.get("topic", ""),
        "summary": original.get("summary", ""),
        "block_refs": replacement.get("block_refs", ""),
        "content_labels": original.get("content_labels", []),
        "children": replacement.get("children", []),
    }


def _get_topic_at_path(
    topics: list[dict[str, Any]],
    path: list[int],
) -> dict[str, Any] | None:
    """Navigate the nested tree by index path; returns ``None`` if out of range."""
    container = topics
    for idx in path[:-1]:
        if idx >= len(container):
            return None
        container = container[idx].get("children", [])
    final_idx = path[-1]
    return container[final_idx] if final_idx < len(container) else None


def _flatten_topics_for_assign(
    topics: list[dict[str, Any]],
    path: list[int],
    result: list[dict[str, Any]],
) -> None:
    """Build a flat list of topics with display strings for the assign-prompt."""
    for i, t in enumerate(topics):
        current_path = [*path, i]
        topic_name = t.get("topic", "")
        refs = parse_segment_ranges(t.get("block_refs", ""))
        children = t.get("children", [])

        if refs or not children:
            display = topic_name
            if refs:
                display += f" (blocks {min(refs)}-{max(refs)})"
            result.append({"display": display, "path": current_path})

        if children:
            _flatten_topics_for_assign(children, current_path, result)


# ── LLM passes ───────────────────────────────────────────────────────


async def _llm_text(llm: LLMClient, prompt: str) -> str:
    """Single-user-message chat shorthand returning the assistant text."""
    response = await llm.chat([ChatMessage(role="user", content=prompt)])
    return response.content


def _render_split_sections(
    unsplit: list[dict[str, Any]],
    atom_map: dict[int, str],
) -> str:
    sections: list[str] = []
    for u in unsplit:
        blocks_text = "\n".join(f"{bid}: {atom_map[bid][:200]}" for bid in u["block_ids"] if bid in atom_map)
        headings_text = "\n".join(f"  [{hid}] {atom_map[hid][:80]}" for hid in u["heading_ids"] if hid in atom_map)
        sections.append(
            f"--- Topic: {u['topic_name']} ({len(u['block_ids'])} blocks, "
            f"{u['total_chars']} chars) ---\n"
            f"Blocks:\n{blocks_text}\n\n"
            f"Sub-headings found:\n{headings_text}",
        )
    return "\n\n".join(sections)


async def asplit_unsplit_leaves(
    llm: LLMClient,
    title: str,
    topics_data: list[dict[str, Any]],
    unsplit: list[dict[str, Any]],
    atom_map: dict[int, str],
    *,
    prompt: str = TOPIC_SPLIT_PROMPT_EN,
) -> list[dict[str, Any]]:
    """Ask the LLM whether to split each leaf and apply the decisions in place."""
    rendered = prompt.format(title=title, topics_section=_render_split_sections(unsplit, atom_map))
    text = await _llm_text(llm, rendered)
    data = parse_llm_json(text)
    if data is None or "results" not in data:
        logger.warning("post-process split: unparsable LLM response")
        return topics_data

    split_map: dict[str, dict[str, Any]] = {}
    for r in data["results"]:
        if isinstance(r, dict) and r.get("should_split") and r.get("children"):
            split_map[r.get("original_topic", "")] = r

    for u in unsplit:
        replacement = split_map.get(u["topic_name"])
        if replacement is not None:
            _replace_topic_at_path(topics_data, u["path"], replacement)

    return topics_data


split_unsplit_leaves = async_to_sync(asplit_unsplit_leaves)
"""Sync bridge — only callable from non-event-loop contexts."""


async def aassign_uncovered_blocks(
    llm: LLMClient,
    title: str,
    topics_data: list[dict[str, Any]],
    uncovered: list[dict[str, Any]],
    atom_map: dict[int, str],
    *,
    prompt: str = UNCOVERED_ASSIGN_PROMPT_EN,
) -> list[dict[str, Any]]:
    """Ask the LLM to attach orphan atoms to existing topics; append to ``block_refs``."""
    flat_topics: list[dict[str, Any]] = []
    _flatten_topics_for_assign(topics_data, [], flat_topics)

    topics_text = "\n".join(f"  {i}: {t['display']}" for i, t in enumerate(flat_topics))
    uncovered_text = "\n".join(f"  block {u['block_id']}: {u['text'][:300]}" for u in uncovered)

    rendered = prompt.format(title=title, topics_text=topics_text, uncovered_text=uncovered_text)
    text = await _llm_text(llm, rendered)
    data = parse_llm_json(text)
    if data is None or "assignments" not in data:
        logger.warning("post-process assign: unparsable LLM response")
        return topics_data

    for assignment in data["assignments"]:
        if not isinstance(assignment, dict):
            continue
        block_id = assignment.get("block_id")
        topic_idx = assignment.get("topic_index")
        if not isinstance(block_id, int) or not isinstance(topic_idx, int):
            continue
        if topic_idx < 0 or topic_idx >= len(flat_topics):
            continue

        target = _get_topic_at_path(topics_data, flat_topics[topic_idx]["path"])
        if target is None:
            continue

        existing = target.get("block_refs", "")
        target["block_refs"] = f"{existing},{block_id}" if existing else str(block_id)

    return topics_data


assign_uncovered_blocks = async_to_sync(aassign_uncovered_blocks)
"""Sync bridge — only callable from non-event-loop contexts."""


async def apostprocess_topics(
    llm: LLMClient,
    title: str,
    topics_data: list[dict[str, Any]],
    atoms: list[tuple[int, str]],
    *,
    split_prompt: str = TOPIC_SPLIT_PROMPT_EN,
    assign_prompt: str = UNCOVERED_ASSIGN_PROMPT_EN,
) -> list[dict[str, Any]]:
    """Run both LLM passes if their detectors find anything to fix.

    Returns ``topics_data`` (possibly mutated in place). Each pass is
    skipped when its detector returns empty.
    """
    atom_map = dict(atoms)

    unsplit = detect_unsplit_leaves(topics_data, atom_map)
    if unsplit:
        topics_data = await asplit_unsplit_leaves(
            llm,
            title,
            topics_data,
            unsplit,
            atom_map,
            prompt=split_prompt,
        )

    uncovered = detect_uncovered_blocks(topics_data, atom_map)
    if uncovered:
        topics_data = await aassign_uncovered_blocks(
            llm,
            title,
            topics_data,
            uncovered,
            atom_map,
            prompt=assign_prompt,
        )

    return topics_data


postprocess_topics = async_to_sync(apostprocess_topics)
"""Sync bridge — only callable from non-event-loop contexts."""
