"""Smoke tests for retrieval protocol surface — types instantiable, frozen, importable."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from everalgo.rank import AgenticDecision, RerankFn, RetrieveFn


def test_agentic_decision_default_round1() -> None:
    d = AgenticDecision(is_multi_round=False)
    assert d.is_multi_round is False
    assert d.is_sufficient is None
    assert d.refined_queries == []
    assert d.query_strategy is None


def test_agentic_decision_round2_multi_query() -> None:
    d = AgenticDecision(
        is_multi_round=True,
        is_sufficient=False,
        reasoning="missing temporal anchor",
        missing_info=["start date"],
        refined_queries=["alt1", "alt2"],
        query_strategy="multi_query",
    )
    assert d.is_multi_round is True
    assert d.query_strategy == "multi_query"
    assert len(d.refined_queries) == 2


def test_agentic_decision_frozen() -> None:
    d = AgenticDecision(is_multi_round=False)
    with pytest.raises(ValidationError):
        d.is_multi_round = True  # type: ignore[misc]


def test_protocols_are_callable_aliases() -> None:
    # RetrieveFn / RerankFn are callable type aliases; this test only ensures they exist and are usable as annotations.
    fn: RetrieveFn  # noqa: F842  # pyright: ignore[reportUnusedVariable]
    rn: RerankFn  # noqa: F842  # pyright: ignore[reportUnusedVariable]
