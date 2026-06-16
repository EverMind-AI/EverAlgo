"""End-to-end smoke tests for ``aclassify_category`` against a real LLM.

Drives the full ``KnowledgeExtractor.aextract`` pipeline with a ~300-token
realistic document plus a 5-entry ``CategorySpec`` taxonomy. Assertions are
tolerance-based (accept any of the human-plausible categories) so the test
does not flake on the LLM's normal latitude between e.g. ``how-to`` and
``concept-explainer``.

Skipped automatically when the three ``LLM_*`` env vars are absent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from everalgo.knowledge import KnowledgeExtractor, aclassify_category
from everalgo.types import CategorySpec, ParsedContent

if TYPE_CHECKING:
    from everalgo.llm.protocols import LLMClient


pytestmark = pytest.mark.integration


_TAXONOMY: list[CategorySpec] = [
    CategorySpec(
        id="how-to",
        description=(
            "Step-by-step tutorials, walkthroughs, and runbooks. The reader's "
            "intent is to perform a concrete task by following instructions. "
            "Examples: 'how to set up X', 'configuring Y', 'step-by-step Z'."
        ),
    ),
    CategorySpec(
        id="reference",
        description=(
            "API references, configuration option tables, command-line flag "
            "listings, and structured lookup material. The reader knows what they "
            "want and is looking it up. Examples: GUC parameter lists, function "
            "signatures, flag glossaries."
        ),
    ),
    CategorySpec(
        id="news",
        description=(
            "Time-bound announcements: release notes, security advisories, "
            "conference schedules, deprecation notices. The reader wants to know "
            "what changed recently."
        ),
    ),
    CategorySpec(
        id="opinion",
        description=(
            "Argumentative or persuasive prose: comparisons, recommendations, "
            "trade-off analyses, opinion pieces. The author is making a case "
            "rather than describing facts."
        ),
    ),
    CategorySpec(
        id="concept-explainer",
        description=(
            "Conceptual explanations of how something works: definitions, "
            "theory, mental models. The reader's intent is to understand rather "
            "than to act."
        ),
    ),
]


_HOW_TO_DOC = """\
How to configure Postgres transaction isolation for high-concurrency workloads.

This guide walks you through the four isolation levels Postgres supports and
how to choose between them.

Step 1: Decide which isolation level you need. Read Committed is the default
and works for most OLTP workloads. Repeatable Read prevents non-repeatable
reads. Serializable provides full SSI guarantees at the cost of some
serialization-failure retries.

Step 2: Set the isolation level. You can do this per-transaction with
``BEGIN ISOLATION LEVEL ...`` or globally via the ``default_transaction_isolation``
GUC parameter. We recommend setting it per-transaction unless every transaction
in your application needs the same level.

Step 3: Handle serialization failures. Under Serializable isolation, your
application must retry transactions that fail with SQLSTATE 40001. Wrap your
transaction code in a retry loop with exponential backoff, typically with
3-5 attempts before surfacing the error to the user.

Step 4: Verify under load. Use pgbench with a custom script to simulate your
workload pattern and monitor pg_stat_database for serialization failure
counts. Tune your retry budget based on what you observe.
"""


_NEWS_DOC = """\
Postgres 17 Release Notes — Highlights

PostgreSQL 17 was released today with significant improvements to vacuum
performance, logical replication, and observability. This release also
deprecates several legacy compatibility options scheduled for removal in
PostgreSQL 19.

Key changes:
- Faster vacuum on large tables via a new memory-aware strategy.
- Logical replication now supports failover slots out of the box.
- New pg_stat_io view exposes per-backend I/O accounting.
- Deprecated: the ``stats_temp_directory`` GUC is removed entirely.
- Deprecated: the legacy ``allow_system_table_mods`` flag now logs a warning.

Upgrade guidance: pg_upgrade from 16.x is supported in-place. From 15.x and
earlier, consult the version-skip notes in the official migration guide.
"""


@pytest.mark.parametrize(
    ("doc_text", "title", "acceptable_ids"),
    [
        (_HOW_TO_DOC, "Configuring Postgres Transaction Isolation", {"how-to", "concept-explainer"}),
        (_NEWS_DOC, "Postgres 17 Release Notes", {"news"}),
    ],
)
async def test_classify_real_document_lands_in_expected_class(
    real_llm: LLMClient,
    doc_text: str,
    title: str,
    acceptable_ids: set[str],
) -> None:
    """Verify ``category_id`` lands in a human-plausible class and is denormalized.

    Runs the full extractor with the taxonomy and asserts (a) the chosen
    ``category_id`` is one of the acceptable answers (or the empty fallback) and
    (b) every node carries the same id.
    """
    parsed = ParsedContent(text=doc_text, mime="text/plain")
    memories = await KnowledgeExtractor(llm=real_llm).aextract(
        parsed,
        doc_id="smoke",
        title=title,
        categories=_TAXONOMY,
    )

    assert len(memories) >= 2, "expected root + at least one topic"
    chosen = memories[0].category_id
    assert chosen in acceptable_ids or chosen == "", (
        f"title={title!r}: expected one of {acceptable_ids} (or empty fallback), got {chosen!r}"
    )
    # Denormalization invariant: every node carries the same id.
    assert {m.category_id for m in memories} == {chosen}


async def test_classify_off_topic_input_stays_safe(real_llm: LLMClient) -> None:
    """Gibberish input must not crash; it should land in some closed-set id or fall back to ``""``."""
    valid_ids = {c.id for c in _TAXONOMY} | {""}
    result = await aclassify_category(
        real_llm,
        title="random gibberish",
        doc_summary="qwerty asdf zxcv 1234 random words with no coherent topic.",
        categories=_TAXONOMY,
    )
    assert result in valid_ids


async def test_classify_pre_supplied_category_id_overrides_classifier(
    real_llm: LLMClient,
) -> None:
    """Passing ``category_id=`` to ``aextract`` should skip the classifier entirely."""
    parsed = ParsedContent(text=_HOW_TO_DOC, mime="text/plain")
    memories = await KnowledgeExtractor(llm=real_llm).aextract(
        parsed,
        doc_id="smoke-override",
        title="Forced reference",
        categories=_TAXONOMY,
        category_id="reference",  # caller-known override
    )
    assert {m.category_id for m in memories} == {"reference"}
