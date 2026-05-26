"""User-scenario boundary detection facade.

Thin wrapper around the low-level ``everalgo.boundary.detect_boundaries`` primitive. Accepts
``list[ChatMessage]`` (user-scenario path; agent trajectories with tool calls go through
``everalgo.agent_memory.AgentBoundaryDetector`` instead).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from asgiref.sync import async_to_sync

from everalgo.boundary import detect_boundaries

if TYPE_CHECKING:
    from everalgo.boundary import DetectionResult
    from everalgo.llm.protocols import LLMClient
    from everalgo.types import ChatMessage


class BoundaryDetector:
    """Boundary detection on a list of ``ChatMessage``. Native-async with sync bridge.

    Calls ``everalgo.boundary.detect_boundaries`` directly — no extra state, no extra logic.
    Exists so user-scenario callers have a class-style facade consistent with the rest of the
    EverAlgo facade surface.

    The LLM client is bound to the instance at construction time.
    """

    def __init__(self, *, llm: LLMClient) -> None:
        self._llm = llm

    async def adetect(
        self,
        messages: list[ChatMessage],
        *,
        is_final: bool = False,
        prompt: str | None = None,
    ) -> DetectionResult:
        """Detect conversation boundaries in a list of ``ChatMessage``.

        Args:
            messages: Ordered list of chat messages to split into :class:`~everalgo.types.MemCell` slices.
            is_final: When ``True``, treat the message stream as complete — the tail is flushed into the
                last cell rather than held back as a pending partial window.
            prompt: Optional prompt template override passed through to ``detect_boundaries``.

        Returns:
            Named tuple ``(cells, tail)`` — ``cells`` contains completed :class:`~everalgo.types.MemCell`
            slices; ``tail`` carries any unconfirmed trailing messages.
        """
        return await detect_boundaries(messages, llm=self._llm, is_final=is_final, prompt=prompt)

    detect = async_to_sync(adetect)
    """Sync bridge — only callable from non-event-loop contexts."""
