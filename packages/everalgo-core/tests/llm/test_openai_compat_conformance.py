"""Run the ``LLMClient`` conformance suite against the reference ``OpenAICompatClient``.

The harness captures requests at the httpx transport through respx and builds each ``EffectiveRequest`` from the
JSON body the client sent, never from the arguments ``achat`` received.
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from typing import TYPE_CHECKING, Any

import httpx
import pytest
import respx

from everalgo.llm.openai_compat import OpenAICompatClient
from everalgo.llm.types import LLMMessage
from everalgo.testing import NOT_SENT, OPTIONAL_CHECKS, REQUIRED_CHECKS, EffectiveRequest, UpstreamStub

if TYPE_CHECKING:
    from collections.abc import Iterator

    from everalgo.testing.conformance import ConformanceCheck

_BASE_URL = "https://conformance.invalid/v1"
_MODEL = "conformance-model"


class OpenAICompatHarness:
    """``LLMClientTestHarness`` for ``OpenAICompatClient`` over a respx-mocked transport.

    Args:
        router: Active respx router the harness installs its upstream route on.
    """

    def __init__(self, router: respx.MockRouter) -> None:
        self._queue: deque[UpstreamStub] = deque()
        self._captured: list[httpx.Request] = []
        router.post("/chat/completions").mock(side_effect=self._respond)

    def build(self, **defaults: Any) -> OpenAICompatClient:
        return OpenAICompatClient(model=_MODEL, api_key="sk-test", base_url=_BASE_URL, **defaults)

    def queue_upstream(self, *responses: UpstreamStub) -> None:
        self._queue.extend(responses)

    def sent_requests(self) -> list[EffectiveRequest]:
        return [_effective_request(request) for request in self._captured]

    async def _respond(self, request: httpx.Request) -> httpx.Response:
        self._captured.append(request)
        if not self._queue:
            # 400 rather than 5xx: the client resends on 5xx, and an exhausted script should fail at once.
            return httpx.Response(400, json={"error": {"message": "conformance upstream script exhausted"}})
        stub = self._queue.popleft()
        if stub.hang:
            await asyncio.Event().wait()
        return httpx.Response(200, json=_completion(stub))


def _effective_request(request: httpx.Request) -> EffectiveRequest:
    body: dict[str, Any] = json.loads(request.content)
    return EffectiveRequest(
        messages=[LLMMessage.model_validate(message) for message in body["messages"]],
        temperature=body.get("temperature", NOT_SENT),
        max_tokens=body.get("max_tokens", NOT_SENT),
    )


def _completion(stub: UpstreamStub) -> dict[str, object]:
    return {
        "id": "chatcmpl-conformance",
        "object": "chat.completion",
        "created": 1700000000,
        "model": _MODEL,
        "choices": [
            {
                "index": 0,
                "finish_reason": stub.finish_reason,
                "message": {"role": "assistant", "content": stub.content, "refusal": stub.refusal},
            }
        ],
    }


@pytest.fixture
def harness() -> Iterator[OpenAICompatHarness]:
    with respx.mock(base_url=_BASE_URL, assert_all_called=False) as router:
        yield OpenAICompatHarness(router)


def _check_id(check: ConformanceCheck) -> str:
    return getattr(check, "__name__", repr(check))


@pytest.mark.parametrize("check", REQUIRED_CHECKS, ids=_check_id)
async def test_reference_client_passes_every_required_check(
    harness: OpenAICompatHarness, check: ConformanceCheck
) -> None:
    await check(harness)


@pytest.mark.parametrize("check", OPTIONAL_CHECKS, ids=_check_id)
async def test_reference_client_passes_every_optional_check(
    harness: OpenAICompatHarness, check: ConformanceCheck
) -> None:
    """The reference client follows the recommendations too, so it is a complete example to copy."""
    await check(harness)
