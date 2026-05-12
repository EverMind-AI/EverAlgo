"""Foresight extractor — derive Foresight memories from a single MemCell."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from asgiref.sync import async_to_sync

import everalgo.llm
from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.prompts import render_prompt
from everalgo.types import Foresight, MemCell
from everalgo.user_memory.prompts.en.foresight import FORESIGHT_EXTRACT_PROMPT_EN

if TYPE_CHECKING:
    from everalgo.llm.protocols import LLMClient


class ForesightExtractor:
    """Extract foresights (anticipated commitments) from a single MemCell.

    Stateless callable class — no ``__init__``, no instance state. Mirrors :class:`EpisodeExtractor` shape;
    the operator is unconditional (runs for any MemCell) and emits zero or more :class:`Foresight` entries.

    Customize per call via ``llm=`` and ``prompt=`` arguments.
    """

    async def aextract(
        self,
        memcell: MemCell,
        *,
        llm: LLMClient | None = None,
        prompt: str | None = None,
    ) -> list[Foresight]:
        """Async main implementation: ask the LLM to extract Foresights.

        Parameters
        ----------
        memcell : MemCell
            Source MemCell (boundary output).
        llm : LLMClient or None, optional
            Per-call LLM override; falls back through the 3-layer chain (scoped via ``use(...)`` and global
            ``configure(...)``); raises :class:`LLMNotConfiguredError` if all None.
        prompt : str or None, optional
            Per-call prompt override; defaults to ``FORESIGHT_EXTRACT_PROMPT_EN``.

        Returns
        -------
        list[Foresight]
            Possibly empty list when no foresights are present in the conversation.

        Raises
        ------
        LLMNotConfiguredError
            Same as the rest of the user-memory operators — no LLM resolvable through the 3-layer chain.
        LLMError
            Any provider-side failure.
        """
        client = everalgo.llm.resolve(llm)
        rendered = render_prompt(
            FORESIGHT_EXTRACT_PROMPT_EN,
            prompt,
            memcell_text=_render_memcell_text(memcell),
            timestamp=memcell.timestamp,
        )
        response = await client.chat(
            messages=[LLMChatMessage(role="user", content=rendered)],
            response_format={"type": "json_object"},
        )
        return _build_foresights_from_llm_response(response.content, memcell)

    extract = async_to_sync(aextract)
    """Sync bridge — only callable from non-event-loop contexts."""


# Module-level helper functions.


def _render_memcell_text(memcell: MemCell) -> str:
    """Render a MemCell as a prompt-friendly conversation transcript."""
    return "\n".join(f"[{m.role.value}] {m.content}" for m in memcell.messages)


def _build_foresights_from_llm_response(raw: str, memcell: MemCell) -> list[Foresight]:
    """Parse LLM JSON and build Foresight list.

    ``parent_id`` and ``parent_type`` are auto-filled from the source MemCell — the LLM is instructed not to
    emit them (see prompts/en/foresight.py).
    """
    parsed = json.loads(raw)
    foresights: list[Foresight] = []
    for fs_dict in parsed.get("foresights", []):
        fs_dict.setdefault("parent_type", "memcell")
        fs_dict.setdefault("parent_id", memcell.id)
        foresights.append(Foresight.model_validate(fs_dict))
    return foresights
