"""Chat-style MemCell extractor — slice a message stream by topic boundaries."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from asgiref.sync import async_to_sync

if TYPE_CHECKING:
    from everalgo.llm.protocols import LLMClient
    from everalgo.types import MemCell, Message


class DetectionOutput(NamedTuple):
    """Result of chat boundary detection."""

    cells: list[MemCell]
    tail_start: int


class ChatMemCellExtractor:
    """Detect MemCell boundaries in a chat-style message stream."""

    async def adetect(
        self,
        messages: list[Message],
        *,
        llm: LLMClient | None = None,
        prompt: str | None = None,
        is_final: bool = False,
        hard_token_limit: int = 65536,
        hard_msg_limit: int = 500,
    ) -> DetectionOutput:
        """Slice ``messages`` into MemCells; return cells + trailing segment index.

        Parameters
        ----------
        messages : list[Message]
            Ordered chat messages.
        llm : LLMClient or None, optional
            Per-call LLM override; ``None`` falls back through ``use(...)`` / ``configure(...)``.
        prompt : str or None, optional
            Per-call batch boundary prompt override.
        is_final : bool, optional
            When ``True``, the trailing segment is folded into the last MemCell and
            ``tail_start == len(messages)``.
        hard_token_limit : int, optional
            Force-split threshold by token count.
        hard_msg_limit : int, optional
            Force-split threshold by message count.

        Returns
        -------
        DetectionOutput

        Raises
        ------
        LLMNotConfiguredError
            No LLM resolvable through the 3-layer chain.
        RuntimeError
            All 5 retries exhausted on JSON parse / schema failure.
        LLMError
            Provider-side failure not absorbed by retry.
        """
        raise NotImplementedError("stub")

    detect = async_to_sync(adetect)
    """Sync bridge — only callable from non-event-loop contexts."""
