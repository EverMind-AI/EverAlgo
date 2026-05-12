"""Profile extractor — synthesize a long-term user profile from a MemCell cluster."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from asgiref.sync import async_to_sync

import everalgo.llm
from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.prompts import render_prompt
from everalgo.types import MemCell, Profile
from everalgo.user_memory.prompts.en.profile import PROFILE_EXTRACT_PROMPT_EN

if TYPE_CHECKING:
    from everalgo.llm.protocols import LLMClient


class ProfileExtractor:
    """Synthesize a long-term user profile from a MemCell + prior cluster context.

    Stateless callable class — no ``__init__``, no instance state. Unlike :class:`EpisodeExtractor` /
    :class:`ForesightExtractor` / :class:`AtomicFactExtractor`, returns a **single** :class:`Profile` rather
    than a list: a profile is a user-level aggregate, not a per-event extraction.

    Profile semantics in v0.x are a **single-shot LLM snapshot** based on (current MemCell + cluster history).
    Multi-stage profile pipelines (part1 / part2 / evidence_completion / merger) are a planned future minor
    bump; see ``local/plans/boundary-user-memory-execution-plan.md`` §1.1 and docs.md §6.3.

    Customize per call via ``llm=`` and ``prompt=`` arguments.
    """

    async def aextract(
        self,
        memcell: MemCell,
        *,
        cluster_episodes: list[MemCell],
        llm: LLMClient | None = None,
        prompt: str | None = None,
    ) -> Profile:
        """Async main implementation: ask the LLM for a profile snapshot.

        Parameters
        ----------
        memcell : MemCell
            The most recent MemCell that triggers profile re-synthesis.
        cluster_episodes : list[MemCell]
            Prior MemCells from the same user (caller pre-fetches the cluster). Can be empty; the prompt
            instructs the LLM to acknowledge limited evidence when so.
        llm : LLMClient or None, optional
            Per-call LLM override; falls back through the 3-layer chain.
        prompt : str or None, optional
            Per-call prompt override; defaults to ``PROFILE_EXTRACT_PROMPT_EN``.

        Returns
        -------
        Profile
            Single Profile snapshot. Any LLM-emitted optional fields
            (interests / habits / ...) are preserved via ``extra="allow"``.

        Raises
        ------
        LLMNotConfiguredError
            No LLM resolvable through the 3-layer chain.
        LLMError
            Any provider-side failure.
        """
        client = everalgo.llm.resolve(llm)
        rendered = render_prompt(
            PROFILE_EXTRACT_PROMPT_EN,
            prompt,
            current_memcell_text=_render_memcell_text(memcell),
            cluster_summaries=_render_cluster(cluster_episodes),
            timestamp=memcell.timestamp,
        )
        response = await client.chat(
            messages=[LLMChatMessage(role="user", content=rendered)],
            response_format={"type": "json_object"},
        )
        return _build_profile_from_llm_response(response.content, memcell)

    extract = async_to_sync(aextract)
    """Sync bridge — only callable from non-event-loop contexts."""


# Module-level helper functions.


def _render_memcell_text(memcell: MemCell) -> str:
    """Render a MemCell as a prompt-friendly conversation transcript."""
    return "\n".join(f"[{m.role.value}] {m.content}" for m in memcell.messages)


def _render_cluster(cluster_episodes: list[MemCell]) -> str:
    """Render the cluster as a compact chronological summary block.

    Empty cluster yields an explicit marker so the LLM can acknowledge limited evidence rather than
    hallucinating context. Each cell prints its timestamp and a one-line preview of the first user message.
    """
    if not cluster_episodes:
        return "(no prior MemCells in the cluster)"
    lines: list[str] = []
    for mc in cluster_episodes:
        preview = mc.messages[0].content[:120] if mc.messages else "(empty)"
        lines.append(f"- ts={mc.timestamp} id={mc.id}: {preview}")
    return "\n".join(lines)


def _build_profile_from_llm_response(raw: str, memcell: MemCell) -> Profile:
    """Parse LLM JSON and build a Profile.

    The LLM is instructed to emit ``id`` / ``owner_id`` / ``summary`` / ``timestamp`` directly at the top
    level. We fall back to ``memcell.timestamp`` if the LLM omits ``timestamp``.
    """
    parsed = json.loads(raw)
    parsed.setdefault("timestamp", memcell.timestamp)
    return Profile.model_validate(parsed)
