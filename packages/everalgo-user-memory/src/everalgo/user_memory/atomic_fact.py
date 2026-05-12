"""AtomicFact extractor — derive AtomicFact memories from a single MemCell."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from asgiref.sync import async_to_sync

import everalgo.llm
from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.prompts import render_prompt
from everalgo.types import AtomicFact, MemCell
from everalgo.user_memory.prompts.en.atomic_fact import ATOMIC_FACT_EXTRACT_PROMPT_EN

if TYPE_CHECKING:
    from everalgo.llm.protocols import LLMClient


class AtomicFactExtractor:
    """Extract atomic facts (single verifiable assertions) from a single MemCell.

    Stateless callable class — no ``__init__``, no instance state. Mirrors :class:`EpisodeExtractor` shape; the
    operator is unconditional (runs for any MemCell) and emits zero or more :class:`AtomicFact` entries.

    Customize per call via ``llm=`` and ``prompt=`` arguments.
    """

    async def aextract(
        self,
        memcell: MemCell,
        *,
        llm: LLMClient | None = None,
        prompt: str | None = None,
    ) -> list[AtomicFact]:
        """Async main implementation: ask the LLM to extract AtomicFacts.

        Parameters
        ----------
        memcell : MemCell
            Source MemCell (boundary output).
        llm : LLMClient or None, optional
            Per-call LLM override; falls back through the 3-layer chain.
        prompt : str or None, optional
            Per-call prompt override; defaults to ``ATOMIC_FACT_EXTRACT_PROMPT_EN``.

        Returns
        -------
        list[AtomicFact]
            Possibly empty list when no atomic facts are present in the conversation.

        Raises
        ------
        LLMNotConfiguredError
            No LLM resolvable through the 3-layer chain.
        LLMError
            Any provider-side failure.
        """
        client = everalgo.llm.resolve(llm)
        rendered = render_prompt(
            ATOMIC_FACT_EXTRACT_PROMPT_EN,
            prompt,
            memcell_text=_render_memcell_text(memcell),
            timestamp=memcell.timestamp,
        )
        response = await client.chat(
            messages=[LLMChatMessage(role="user", content=rendered)],
            response_format={"type": "json_object"},
        )
        return _build_atomic_facts_from_llm_response(response.content, memcell)

    extract = async_to_sync(aextract)
    """Sync bridge — only callable from non-event-loop contexts."""


# Module-level helper functions.


def _render_memcell_text(memcell: MemCell) -> str:
    """Render a MemCell as a prompt-friendly conversation transcript."""
    return "\n".join(f"[{m.role.value}] {m.content}" for m in memcell.messages)


def _build_atomic_facts_from_llm_response(raw: str, memcell: MemCell) -> list[AtomicFact]:
    """Parse LLM JSON and build AtomicFact list.

    ``parent_id`` and ``parent_type`` are auto-filled from the source MemCell — the LLM is instructed not to
    emit them (see prompts/en/atomic_fact.py).
    """
    parsed = json.loads(raw)
    facts: list[AtomicFact] = []
    for af_dict in parsed.get("atomic_facts", []):
        af_dict.setdefault("parent_type", "memcell")
        af_dict.setdefault("parent_id", memcell.id)
        facts.append(AtomicFact.model_validate(af_dict))
    return facts
