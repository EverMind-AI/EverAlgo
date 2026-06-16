"""Helper for loading the verbatim memsys_enterprise fixture JSON files.

The fixtures ship with extra business fields (resource_type, source_type,
participants, ...) that EverAlgo does not consume; this loader keeps only
the two fields ``KnowledgeExtractor`` needs.
"""

from __future__ import annotations

import json
from pathlib import Path

from everalgo.types import ParsedContent

_FIXTURE_DIR = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> tuple[ParsedContent, str, str]:
    """Read a fixture JSON file and project it onto ``ParsedContent``.

    Args:
        name: Fixture stem (e.g. ``"idx_multi_topic"``); the ``.json`` suffix
            is added automatically.

    Returns:
        ``(parsed, doc_id, title)`` — ``ParsedContent`` no longer carries
        id / title, so the fixture stem and JSON title are returned alongside
        for the caller to pass into ``KnowledgeExtractor.aextract``.
    """
    path = _FIXTURE_DIR / f"{name}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    parsed = ParsedContent(text=raw.get("content", ""), mime="text/markdown")
    return parsed, name, raw.get("title", "")
