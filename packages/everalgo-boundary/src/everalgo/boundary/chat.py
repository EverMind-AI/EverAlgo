"""Chat-style MemCell extractor — slice a message stream by topic boundaries."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from asgiref.sync import async_to_sync

if TYPE_CHECKING:
    from everalgo.llm.protocols import LLMClient
    from everalgo.types import MemCell, Message


class DetectionOutput(NamedTuple):
    """Result of chat boundary detection.

    Subclasses ``tuple``, so both forms are supported and interchangeable:

    - Positional unpacking: ``cells, tail = await extractor.adetect(msgs)``
    - Named access: ``out.cells`` / ``out.tail``
    - Index access: ``out[0]`` / ``out[1]``

    ``tail`` is the trailing segment the LLM did not confidently close. The
    caller is expected to persist it and prepend it to fresh messages on the
    next ``adetect`` call. When ``is_final=True`` is passed, ``tail`` is
    guaranteed to be ``[]``.
    """

    cells: list[MemCell]
    tail: list[Message]


class ChatMemCellExtractor:
    """Detect MemCell boundaries in a chat-style message stream.

    Stateless callable class — no ``__init__``, no instance state. Thread / async safe; instances are
    interchangeable. Customize per call via ``llm=`` / ``prompt=`` / ``is_final=`` / ``hard_token_limit=`` /
    ``hard_msg_limit=`` arguments.

    Pipeline phases (real implementation TBD):

    1. **Input validation** — empty ``messages`` short-circuits to empty output.
    2. **Default resolution** — ``llm`` / ``prompt`` resolved from the registry when not supplied.
    3. **Force-split loop** — repeatedly slice off the head when the running token total exceeds
       ``hard_token_limit`` or the message count exceeds ``hard_msg_limit``; runs *before* the LLM call
       to keep prompts inside the model context window.
    4. **LLM batch detection** — one LLM call returns all topic boundaries for the residual stream;
       retried up to 5 times on JSON parse / schema failure, then raises ``RuntimeError``.
    5. **Boundary slicing + is_final reduction** — slice the residual stream by the returned boundaries;
       if ``is_final=True`` the unclosed trailing segment is absorbed into the last MemCell.
    """

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
        """Async main implementation: slice a chat message stream into MemCells.

        Parameters
        ----------
        messages : list[Message]
            Ordered chat messages (user / assistant turns).
        llm : LLMClient or None, optional
            Per-call LLM override. Falls back through scoped (``use(...)``) and default
            (``configure(...)``); raises ``LLMNotConfiguredError`` if all None.
        prompt : str or None, optional
            Per-call batch boundary prompt override. Defaults to ``CHAT_BOUNDARY_DETECT_PROMPT_EN``.
            Must contain a ``{messages}`` placeholder.
        is_final : bool, optional
            ``True`` forces the trailing segment into the last MemCell (``tail`` guaranteed ``[]``).
            ``False`` returns the unclosed trailing segment as ``tail`` for the caller to persist.
        hard_token_limit : int, optional
            Force-split threshold by token count. Default ``65536``.
        hard_msg_limit : int, optional
            Force-split threshold by message count. Default ``500``.

        Returns
        -------
        DetectionOutput
            ``(cells, tail)`` named tuple. ``is_final=True`` guarantees ``tail == []``.

        Raises
        ------
        LLMNotConfiguredError
            No LLM resolvable through the 3-layer chain.
        RuntimeError
            All 5 retries exhausted on JSON parse / schema failure.
        LLMError
            Any provider-side failure not absorbed by retry.
        """
        raise NotImplementedError("stub")

    detect = async_to_sync(adetect)
    """Sync bridge — only callable from non-event-loop contexts."""
