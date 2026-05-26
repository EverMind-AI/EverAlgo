"""Behaviour tests for :func:`everalgo.clustering.cluster_by_llm`."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from everalgo.clustering import Cluster, cluster_by_llm
from everalgo.llm.errors import LLMError
from everalgo.llm.types import ChatMessage, ChatResponse
from everalgo.testing.fake_llm import FakeLLMClient

_BASE_TS_MS = 1_700_000_000_000


def _c(vec: list[float], ts: int = _BASE_TS_MS, count: int = 1, preview: list[str] | None = None) -> Cluster:
    return Cluster(centroid=np.array(vec, dtype=np.float32), last_ts=ts, count=count, preview=preview or [])


def _two_clusters() -> list[Cluster]:
    return [
        _c([1.0, 0.0], count=3),
        _c([0.0, 1.0], count=2),
    ]


async def test_empty_existing_skips_llm_and_returns_none() -> None:
    new_c = _c([1.0, 0.0])
    llm = FakeLLMClient(responses=[])

    result = await cluster_by_llm(new_c, [], llm=llm)

    assert result is None
    assert llm.call_count == 0


async def test_fast_path_top1_above_skip_threshold_skips_llm() -> None:
    existing = _two_clusters()
    # cosine([0.999, 0.001], [1,0]) ≈ 0.9999 > 0.85 skip threshold.
    new_c = _c([0.999, 0.001])
    llm = FakeLLMClient(responses=[])

    result = await cluster_by_llm(new_c, existing, llm=llm)

    assert result is not None
    assert result.count == 4
    assert llm.call_count == 0


async def test_llm_called_when_top1_below_skip_threshold() -> None:
    existing = _two_clusters()
    # cosine([0.7, 0.7], [1,0]) ≈ 0.707 — below 0.85 skip.
    new_c = _c([0.7, 0.7], preview=["ambiguous task"])
    llm = FakeLLMClient(responses=['{"idx": 1, "reason": "matches case bucket"}'])

    result = await cluster_by_llm(new_c, existing, llm=llm)

    assert llm.call_count == 1
    assert result is not None
    assert result.count == 3  # LLM's choice (existing[1], count=2) + new


async def test_llm_returns_minus_one_creates_new() -> None:
    existing = _two_clusters()
    new_c = _c([0.7, 0.7], preview=["novel task"])
    llm = FakeLLMClient(responses=['{"idx": -1, "reason": "new domain"}'])

    result = await cluster_by_llm(new_c, existing, llm=llm)

    assert result is None


async def test_llm_returns_out_of_range_idx_creates_new() -> None:
    existing = _two_clusters()
    new_c = _c([0.7, 0.7])
    llm = FakeLLMClient(responses=['{"idx": 999, "reason": "new domain"}'])

    result = await cluster_by_llm(new_c, existing, llm=llm)

    assert result is None


async def test_bad_llm_response_raises_value_error() -> None:
    """Non-JSON response from LLM → LLMError propagates (no retry, no fallback)."""
    existing = _two_clusters()
    new_c = _c([0.7, 0.7])
    llm = FakeLLMClient(responses=["not json"])

    with pytest.raises(LLMError):
        await cluster_by_llm(new_c, existing, llm=llm)

    assert llm.call_count == 1


async def test_llm_missing_idx_field_raises_value_error() -> None:
    """Valid JSON but missing 'idx' field → LLMError propagates (Pydantic validation)."""
    existing = _two_clusters()
    new_c = _c([0.7, 0.7])
    llm = FakeLLMClient(responses=['{"reason": "x"}'])

    with pytest.raises(LLMError):
        await cluster_by_llm(new_c, existing, llm=llm)

    assert llm.call_count == 1


async def test_llm_clusters_json_contains_idx_count_preview() -> None:
    existing = [
        _c([1.0, 0.0], count=3, preview=["fix login bug"]),
        _c([0.0, 1.0], count=2, preview=["refactor auth"]),
    ]
    new_c = _c([0.7, 0.7], preview=["investigate token leak"])
    captured: dict[str, str] = {}

    def handler(messages: list[ChatMessage], **_kwargs: Any) -> ChatResponse:
        assert isinstance(messages[0].content, str)  # narrow for test
        captured["prompt"] = messages[0].content
        return ChatResponse(
            content='{"idx": 0, "reason": "test"}',
            model="fake",
            usage=None,
            finish_reason="stop",
            raw=None,
        )

    llm = FakeLLMClient(handler=handler)

    await cluster_by_llm(new_c, existing, llm=llm)

    rendered = captured["prompt"]
    assert "investigate token leak" in rendered
    assert "fix login bug" in rendered
    assert "refactor auth" in rendered
    assert '"count": 3' in rendered
    assert '"idx": 0' in rendered


async def test_caller_supplied_prompt_overrides_default() -> None:
    existing = _two_clusters()
    new_c = _c([0.7, 0.7], preview=["hello"])
    custom = "OVERRIDE memcell={memcell_text} clusters={clusters_json}"
    captured: dict[str, str] = {}

    def handler(messages: list[ChatMessage], **_kwargs: Any) -> ChatResponse:
        assert isinstance(messages[0].content, str)  # narrow for test
        captured["prompt"] = messages[0].content
        return ChatResponse(
            content='{"idx": 0, "reason": "ok"}',
            model="fake",
            usage=None,
            finish_reason="stop",
            raw=None,
        )

    llm = FakeLLMClient(handler=handler)

    await cluster_by_llm(new_c, existing, llm=llm, prompt=custom)

    assert captured["prompt"].startswith("OVERRIDE memcell=hello")


async def test_preview_empty_when_new_cluster_has_no_preview() -> None:
    """When new_cluster.preview is empty, query_text is '' and prompt still renders."""
    existing = _two_clusters()
    new_c = _c([0.7, 0.7])  # no preview
    captured: dict[str, str] = {}

    def handler(messages: list[ChatMessage], **_kwargs: Any) -> ChatResponse:
        assert isinstance(messages[0].content, str)  # narrow for test
        captured["prompt"] = messages[0].content
        return ChatResponse(
            content='{"idx": 0, "reason": "ok"}', model="fake", usage=None, finish_reason="stop", raw=None
        )

    llm = FakeLLMClient(handler=handler)
    await cluster_by_llm(new_c, existing, llm=llm)

    assert "idx" in captured["prompt"]  # prompt rendered without error


async def test_merge_appends_members_via_llm_path() -> None:
    existing = [
        Cluster(
            centroid=np.array([1.0, 0.0], dtype=np.float32),
            last_ts=_BASE_TS_MS,
            count=2,
            members=["a", "b"],
        )
    ]
    new_c = Cluster(
        centroid=np.array([0.999, 0.001], dtype=np.float32),
        last_ts=_BASE_TS_MS + 1,
        members=["c"],
    )
    # cosine ~0.9999 >= 0.85 skip threshold → fast-path merge, no LLM call.
    llm = FakeLLMClient(responses=[])

    result = await cluster_by_llm(new_c, existing, llm=llm)

    assert result is not None
    assert result.members == ["a", "b", "c"]
    assert llm.call_count == 0


async def test_none_path_preserves_new_members_llm() -> None:
    # Empty existing → None; caller keeps new_c with its members.
    new_c = Cluster(
        centroid=np.array([1.0, 0.0], dtype=np.float32),
        last_ts=_BASE_TS_MS,
        members=["entity_y"],
    )
    llm = FakeLLMClient(responses=[])

    result = await cluster_by_llm(new_c, [], llm=llm)

    assert result is None
    assert new_c.members == ["entity_y"]
