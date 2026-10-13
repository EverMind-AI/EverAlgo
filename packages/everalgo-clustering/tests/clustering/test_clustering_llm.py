"""Behaviour tests for :func:`everalgo.clustering.acluster_by_llm`."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from everalgo.clustering import Cluster, acluster_by_llm, cluster_by_llm
from everalgo.clustering._replies import _ClusterAssignmentReply
from everalgo.clustering.prompts.en.cluster import CLUSTER_LLM_ASSIGN_SYSTEM_PROMPT
from everalgo.llm import LLMOutputError
from everalgo.testing import FakeLLMClient

if TYPE_CHECKING:
    from everalgo.llm import LLMMessage

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

    result = await acluster_by_llm(new_c, [], llm=llm)

    assert result is None
    assert llm.call_count == 0


async def test_fast_path_top1_above_skip_threshold_skips_llm() -> None:
    existing = _two_clusters()
    # cosine([0.999, 0.001], [1,0]) ≈ 0.9999 > 0.85 skip threshold.
    new_c = _c([0.999, 0.001])
    llm = FakeLLMClient(responses=[])

    result = await acluster_by_llm(new_c, existing, llm=llm)

    assert result is not None
    assert result.count == 4
    assert llm.call_count == 0


async def test_llm_called_when_top1_below_skip_threshold() -> None:
    existing = _two_clusters()
    # cosine([0.7, 0.7], [1,0]) ≈ 0.707 — below 0.85 skip.
    new_c = _c([0.7, 0.7], preview=["ambiguous task"])
    llm = FakeLLMClient(responses=['{"idx": 1, "reason": "matches case bucket"}'])

    result = await acluster_by_llm(new_c, existing, llm=llm)

    assert llm.call_count == 1
    assert result is not None
    assert result.count == 3  # LLM's choice (existing[1], count=2) + new


async def test_the_call_asks_for_the_assignment_reply_model() -> None:
    llm = FakeLLMClient(responses=['{"idx": -1, "reason": "new domain"}'])

    await acluster_by_llm(_c([0.7, 0.7]), _two_clusters(), llm=llm)

    assert llm.calls[0].response_format is _ClusterAssignmentReply


async def test_llm_returns_minus_one_creates_new() -> None:
    existing = _two_clusters()
    new_c = _c([0.7, 0.7], preview=["novel task"])
    llm = FakeLLMClient(responses=['{"idx": -1, "reason": "new domain"}'])

    result = await acluster_by_llm(new_c, existing, llm=llm)

    assert result is None


async def test_llm_returns_out_of_range_idx_creates_new() -> None:
    existing = _two_clusters()
    new_c = _c([0.7, 0.7])
    llm = FakeLLMClient(responses=['{"idx": 999, "reason": "new domain"}'])

    result = await acluster_by_llm(new_c, existing, llm=llm)

    assert result is None


@pytest.mark.parametrize(
    ("reply", "fault"),
    [
        ("not json", "<root>"),
        ('{"reason": "x"}', "idx: Field required"),
        ('{"idx": 0}', "reason: Field required"),
        ('{"idx": 0, "reason": "x", "confidence": 0.9}', "confidence: Extra inputs are not permitted"),
        ('{"idx": "0", "reason": "x"}', "idx"),
        ('```json\n{"idx": 0, "reason": "x"}\n```', "<root>"),
    ],
)
async def test_a_bad_reply_raises_llm_output_error_without_retry(reply: str, fault: str) -> None:
    llm = FakeLLMClient(responses=[reply])

    with pytest.raises(LLMOutputError, match="cluster assignment reply is not a valid _ClusterAssignmentReply") as exc:
        await acluster_by_llm(_c([0.7, 0.7]), _two_clusters(), llm=llm)

    assert fault in str(exc.value)
    assert reply not in str(exc.value)
    assert llm.call_count == 1


async def test_a_client_failure_propagates_unchanged() -> None:
    llm = FakeLLMClient(responses=[ConnectionError("upstream down")])

    with pytest.raises(ConnectionError, match="upstream down"):
        await acluster_by_llm(_c([0.7, 0.7]), _two_clusters(), llm=llm)

    assert llm.call_count == 1


async def test_llm_clusters_json_contains_idx_count_preview() -> None:
    existing = [
        _c([1.0, 0.0], count=3, preview=["fix login bug"]),
        _c([0.0, 1.0], count=2, preview=["refactor auth"]),
    ]
    new_c = _c([0.7, 0.7], preview=["investigate token leak"])
    captured: dict[str, list[LLMMessage]] = {}

    def handler(messages: list[LLMMessage], **_kwargs: Any) -> str:
        captured["messages"] = messages
        return '{"idx": 0, "reason": "test"}'

    llm = FakeLLMClient(handler=handler)

    await acluster_by_llm(new_c, existing, llm=llm)

    user = captured["messages"][1].content
    assert isinstance(user, str)
    assert "investigate token leak" in user
    assert "fix login bug" in user
    assert "refactor auth" in user
    assert '"count": 3' in user
    assert '"idx": 0' in user


async def _sent(**kwargs: Any) -> tuple[str, str]:
    """The system and user messages one LLM-path call sends."""
    llm = FakeLLMClient(responses=['{"idx": 0, "reason": "ok"}'])
    await acluster_by_llm(_c([0.7, 0.7], preview=["hello"]), _two_clusters(), llm=llm, **kwargs)
    system, user = llm.calls[0].messages
    assert (system.role, user.role) == ("system", "user")
    return str(system.content), str(user.content)


async def test_custom_instructions_close_the_system_message_verbatim() -> None:
    """The last block, after ``<reply_format>``; never rendered as a template, so a slot name stays literal."""
    custom = "Keep {{memcell_text}} literal.\nGroup  billing tasks together."

    system, user = await _sent(custom_instructions=custom)
    _, default_user = await _sent()

    assert system == f"{CLUSTER_LLM_ASSIGN_SYSTEM_PROMPT}\n\n<custom_instructions>\n{custom}\n</custom_instructions>"
    assert system.endswith("\n</custom_instructions>")
    assert system.index("</reply_format>") < system.index("<custom_instructions>")
    assert user == default_user
    assert user.endswith("<new_task_intent>\nhello\n</new_task_intent>")


@pytest.mark.parametrize("custom_instructions", [None, "", "  \n\t"], ids=["none", "empty", "blank"])
async def test_no_or_blank_custom_instructions_send_the_bundled_rules_byte_for_byte(
    custom_instructions: str | None,
) -> None:
    system, _ = await _sent(custom_instructions=custom_instructions)

    assert system == CLUSTER_LLM_ASSIGN_SYSTEM_PROMPT
    assert "<custom_instructions>" not in system


def test_custom_instructions_is_keyword_only_and_prompt_is_gone() -> None:
    parameters = inspect.signature(acluster_by_llm).parameters

    assert "prompt" not in parameters
    assert parameters["custom_instructions"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["custom_instructions"].default is None


async def test_passing_prompt_raises_type_error() -> None:
    with pytest.raises(TypeError, match="prompt"):
        await _sent(prompt="OVERRIDE rules: pick the closest cluster.")


def test_sync_bridge_forwards_custom_instructions_and_rejects_prompt() -> None:
    llm = FakeLLMClient(responses=['{"idx": 0, "reason": "ok"}'])

    cluster_by_llm(_c([0.7, 0.7]), _two_clusters(), llm=llm, custom_instructions="Group billing tasks together.")

    system = str(llm.calls[0].messages[0].content)
    assert system.endswith("<custom_instructions>\nGroup billing tasks together.\n</custom_instructions>")
    with pytest.raises(TypeError, match="prompt"):
        # The removed keyword, on purpose: a type checker rejects it too.
        cluster_by_llm(_c([0.7, 0.7]), _two_clusters(), llm=llm, prompt="rules")  # type: ignore[call-arg]


async def test_preview_empty_when_new_cluster_has_no_preview() -> None:
    """When new_cluster.preview is empty, query_text is '' and the prompt still renders."""
    existing = _two_clusters()
    new_c = _c([0.7, 0.7])  # no preview
    llm = FakeLLMClient(responses=['{"idx": 0, "reason": "ok"}'])

    await acluster_by_llm(new_c, existing, llm=llm)

    user = llm.calls[0].messages[1].content
    assert isinstance(user, str)
    assert '"idx": 0' in user  # prompt rendered without error


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

    result = await acluster_by_llm(new_c, existing, llm=llm)

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

    result = await acluster_by_llm(new_c, [], llm=llm)

    assert result is None
    assert new_c.members == ["entity_y"]
