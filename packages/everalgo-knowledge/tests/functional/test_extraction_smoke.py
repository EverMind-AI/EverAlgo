"""End-to-end smoke tests against a real LLM.

Two representative fixtures from memsys_enterprise drive the full
``KnowledgeExtractor.aextract`` pipeline against a live endpoint. Assertions
are *structural* — we cannot stably check exact topic names against a
non-deterministic LLM, but we can require:

* the root node is present and well-formed
* every emitted ``KnowledgeMemory`` has a non-empty ``topic_path``
* parent-child indices are internally consistent
* at least one descendant node was extracted (i.e. the LLM did not just
  emit an empty topics list)

Skipped automatically when the three ``LLM_*`` env vars are absent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from everalgo.knowledge import KnowledgeExtractor

from ._loader import load_fixture

if TYPE_CHECKING:
    from everalgo.llm.protocols import LLMClient
    from everalgo.types import KnowledgeMemory


pytestmark = pytest.mark.integration


def _assert_well_formed(memories: list[KnowledgeMemory]) -> None:
    assert len(memories) >= 2, "expected at least a root + 1 topic"

    root = memories[0]
    assert root.topic_index == 0
    assert root.depth == 0
    assert root.parent_index is None
    assert root.topic, "root topic should carry the document title"
    assert root.topic_path == root.topic

    # All non-root nodes point at a valid parent_index that lives earlier in the list
    # and the parent's children_index must mention them.
    by_index = {km.topic_index: km for km in memories}
    for km in memories[1:]:
        assert km.topic_path, "every node should have a topic_path"
        assert km.parent_index is not None
        assert km.parent_index < km.topic_index
        parent = by_index[km.parent_index]
        assert km.topic_index in parent.children_index


async def test_extraction_smoke_multi_topic(real_llm: LLMClient) -> None:
    parsed, doc_id, title = load_fixture("idx_multi_topic")
    memories = await KnowledgeExtractor(llm=real_llm).aextract(parsed, doc_id=doc_id, title=title)
    _assert_well_formed(memories)
    # idx_multi_topic explicitly covers Engineering / Marketing / Finance — we
    # expect the LLM to surface at least 2 distinct leaf topics, but allow the
    # model freedom to introduce an intermediate "Executive Overview" wrapper
    # or any other grouping it deems appropriate.
    leaves = [km for km in memories if not km.children_index]
    assert len(leaves) >= 2


async def test_extraction_smoke_confluence(real_llm: LLMClient) -> None:
    parsed, doc_id, title = load_fixture("document_confluence")
    memories = await KnowledgeExtractor(llm=real_llm).aextract(parsed, doc_id=doc_id, title=title)
    _assert_well_formed(memories)
