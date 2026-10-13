"""Full-field reply texts for the knowledge reply models, so a scripted test reply passes strict validation."""

from __future__ import annotations

import json
from typing import Any


def topic(name: str, refs: str, *children: dict[str, Any], labels: tuple[str, ...] = ()) -> dict[str, Any]:
    """One ``TopicNode`` as a dict; its summary is derived from ``name``."""
    return {
        "topic": name,
        "summary": f"{name} summary",
        "block_refs": refs,
        "content_labels": list(labels),
        "children": list(children),
    }


def topic_tree_reply(summary: str, *topics: dict[str, Any], labels: tuple[str, ...] = ()) -> str:
    """A ``TopicTreeReply`` text holding ``topics``."""
    return json.dumps(
        {
            "language": "English",
            "summary": summary,
            "subject": "Subject",
            "keywords": ["keyword"],
            "content_labels": list(labels),
            "topics": list(topics),
        }
    )


def content_merge_reply(summary: str) -> str:
    """A ``ContentMergeReply`` text with ``summary``."""
    return json.dumps({"language": "English", "summary": summary, "subject": "Subject", "keywords": ["keyword"]})


def assign_reply(*pairs: tuple[int, int]) -> str:
    """An ``UncoveredAssignReply`` text, one assignment per ``(block_id, topic_index)``."""
    return json.dumps(
        {"assignments": [{"block_id": block, "topic_index": index, "reason": "near"} for block, index in pairs]}
    )
