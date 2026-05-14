"""Behaviour tests for :func:`everalgo.clustering.cluster_by_llm`."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from everalgo.clustering import ClusterConfig, ClusterState, cluster_by_llm
from everalgo.llm.types import ChatMessage, ChatResponse
from everalgo.testing.fake_llm import FakeLLMClient

_BASE_TS_MS = 1_700_000_000_000


def _seed_with_two_clusters() -> ClusterState:
    return ClusterState(
        centroids={
            "cluster_000": np.array([1.0, 0.0], dtype=np.float32),
            "cluster_001": np.array([0.0, 1.0], dtype=np.float32),
        },
        counts={"cluster_000": 3, "cluster_001": 2},
        last_ts={"cluster_000": _BASE_TS_MS, "cluster_001": _BASE_TS_MS},
        next_idx=2,
    )


async def test_empty_state_skips_llm_and_creates_first_cluster() -> None:
    state = ClusterState.empty()
    vector = np.array([1.0, 0.0], dtype=np.float32)
    # FakeLLMClient with no scripted responses — if cluster_by_llm tries to call it, it raises.
    llm = FakeLLMClient(responses=[])

    cid, new_state = await cluster_by_llm(
        vector,
        _BASE_TS_MS,
        "any text",
        state,
        config=ClusterConfig(),
        llm=llm,
        cluster_previews={},
    )

    assert cid == "cluster_000"
    assert new_state.next_idx == 1
    assert llm.call_count == 0


async def test_fast_path_top1_above_skip_threshold_skips_llm() -> None:
    seed = _seed_with_two_clusters()
    # cosine([0.999, 0.001], [1,0]) ≈ 0.9999 > 0.85 skip threshold.
    vector = np.array([0.999, 0.001], dtype=np.float32)
    llm = FakeLLMClient(responses=[])

    cid, _new = await cluster_by_llm(
        vector,
        _BASE_TS_MS,
        "match top1",
        seed,
        config=ClusterConfig(),
        llm=llm,
        cluster_previews={},
    )

    assert cid == "cluster_000"
    assert llm.call_count == 0


async def test_llm_called_when_top1_below_skip_threshold() -> None:
    seed = _seed_with_two_clusters()
    # cosine([0.7, 0.7], [1,0]) ≈ 0.707 — below 0.85 skip, above 0.65 fallback threshold.
    vector = np.array([0.7, 0.7], dtype=np.float32)
    llm = FakeLLMClient(responses=['{"cluster_id": "cluster_001", "reason": "matches case bucket"}'])

    cid, _new = await cluster_by_llm(
        vector,
        _BASE_TS_MS,
        "ambiguous task",
        seed,
        config=ClusterConfig(),
        llm=llm,
        cluster_previews={"cluster_000": ["prev"], "cluster_001": ["x"]},
    )

    assert llm.call_count == 1
    assert cid == "cluster_001"  # LLM's choice wins over geometric top-1.


async def test_llm_returns_unknown_cluster_id_creates_new() -> None:
    seed = _seed_with_two_clusters()
    vector = np.array([0.7, 0.7], dtype=np.float32)
    # LLM picks a cluster that doesn't exist in state → treat as "create new".
    llm = FakeLLMClient(responses=['{"cluster_id": "cluster_999", "reason": "new domain"}'])

    cid, new_state = await cluster_by_llm(
        vector,
        _BASE_TS_MS,
        "novel task",
        seed,
        config=ClusterConfig(),
        llm=llm,
        cluster_previews={},
    )

    assert cid == "cluster_002"  # next_idx was 2 → freshly minted.
    assert new_state.next_idx == 3


async def test_llm_fallback_after_all_retries_fail_assigns_top1_when_above_threshold() -> None:
    seed = _seed_with_two_clusters()
    vector = np.array([0.7, 0.7], dtype=np.float32)  # top-1 sim ~0.707, above 0.65 threshold.
    # 3 bad responses → retries exhausted → fallback path.
    llm = FakeLLMClient(responses=["not json", "still not", "garbage"])

    cid, _new = await cluster_by_llm(
        vector,
        _BASE_TS_MS,
        "anything",
        seed,
        config=ClusterConfig(),
        llm=llm,
        cluster_previews={},
    )

    assert llm.call_count == 3
    assert cid == "cluster_000"  # Fell back to geometric top-1.


async def test_llm_fallback_below_threshold_creates_new() -> None:
    seed = _seed_with_two_clusters()
    # cos([0.5, -1], [1, 0]) = 0.5 / sqrt(1.25) ≈ 0.447 → below 0.65 fallback threshold.
    vector = np.array([0.5, -1.0], dtype=np.float32)
    llm = FakeLLMClient(responses=["bad", "bad", "bad"])

    cid, new_state = await cluster_by_llm(
        vector,
        _BASE_TS_MS,
        "off-distribution",
        seed,
        config=ClusterConfig(),
        llm=llm,
        cluster_previews={},
    )

    assert cid == "cluster_002"
    assert new_state.next_idx == 3


async def test_llm_missing_cluster_id_field_counts_as_retry_failure() -> None:
    seed = _seed_with_two_clusters()
    vector = np.array([0.7, 0.7], dtype=np.float32)
    # All three responses are valid JSON but missing "cluster_id" → retries exhausted → fallback.
    llm = FakeLLMClient(responses=['{"reason": "x"}', '{"reason": "y"}', '{"reason": "z"}'])

    cid, _new = await cluster_by_llm(
        vector,
        _BASE_TS_MS,
        "schema-broken",
        seed,
        config=ClusterConfig(),
        llm=llm,
        cluster_previews={},
    )

    assert llm.call_count == 3
    assert cid == "cluster_000"  # Geometric fallback to top-1.


async def test_llm_clusters_json_contains_previews_and_counts() -> None:
    seed = _seed_with_two_clusters()
    vector = np.array([0.7, 0.7], dtype=np.float32)
    captured: dict[str, str] = {}

    def handler(messages: list[ChatMessage], **_kwargs: Any) -> ChatResponse:
        captured["prompt"] = messages[0].content
        return ChatResponse(
            content='{"cluster_id": "cluster_000", "reason": "test"}',
            model="fake",
            usage=None,
            finish_reason="stop",
            raw=None,
        )

    llm = FakeLLMClient(handler=handler)
    previews = {"cluster_000": ["fix login bug"], "cluster_001": ["refactor auth"]}

    await cluster_by_llm(
        vector,
        _BASE_TS_MS,
        "investigate token leak",
        seed,
        config=ClusterConfig(),
        llm=llm,
        cluster_previews=previews,
    )

    rendered = captured["prompt"]
    assert "investigate token leak" in rendered
    assert "fix login bug" in rendered
    assert "refactor auth" in rendered
    assert '"item_count": 3' in rendered
    # next_new_id should reflect state.next_idx = 2 → "002".
    assert "cluster_002" in rendered


async def test_caller_supplied_prompt_overrides_default() -> None:
    seed = _seed_with_two_clusters()
    vector = np.array([0.7, 0.7], dtype=np.float32)
    custom = "OVERRIDE memcell={memcell_text} clusters={clusters_json} new={next_new_id}"
    captured: dict[str, str] = {}

    def handler(messages: list[ChatMessage], **_kwargs: Any) -> ChatResponse:
        captured["prompt"] = messages[0].content
        return ChatResponse(
            content='{"cluster_id": "cluster_000", "reason": "ok"}',
            model="fake",
            usage=None,
            finish_reason="stop",
            raw=None,
        )

    llm = FakeLLMClient(handler=handler)

    await cluster_by_llm(
        vector,
        _BASE_TS_MS,
        "hello",
        seed,
        config=ClusterConfig(),
        llm=llm,
        cluster_previews={},
        prompt=custom,
    )

    assert captured["prompt"].startswith("OVERRIDE memcell=hello")


@pytest.mark.parametrize(
    "raw",
    [
        '```json\n{"cluster_id": "cluster_000", "reason": "fence-wrapped"}\n```',
    ],
)
async def test_llm_response_with_json_fence_is_parsed(raw: str) -> None:
    seed = _seed_with_two_clusters()
    vector = np.array([0.7, 0.7], dtype=np.float32)
    llm = FakeLLMClient(responses=[raw])

    cid, _new = await cluster_by_llm(
        vector,
        _BASE_TS_MS,
        "fenced response",
        seed,
        config=ClusterConfig(),
        llm=llm,
        cluster_previews={},
    )

    # Direct json.loads will fail; the ```json``` fence fallback kicks in.
    assert cid == "cluster_000"
    assert llm.call_count == 1
