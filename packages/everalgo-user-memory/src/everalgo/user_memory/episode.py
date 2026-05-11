"""Episode extractor — derive Episode memories from a single MemCell."""

from __future__ import annotations

import json

from asgiref.sync import async_to_sync

import everalgo.llm
from everalgo.llm.protocols import LLMClient
from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.types import Episode, MemCell
from everalgo.user_memory.prompts.en.episode import EPISODE_EXTRACT_PROMPT_EN


class EpisodeExtractor:
    """Extract Episode memories from a single MemCell.

    Stateless callable class — no ``__init__``, no instance state.
    Per design.md line 687 / line 697: this is the unconditional
    EPISODE-path operator (runs for any MemCell).

    Customize per call via ``llm=`` and ``prompt=`` arguments.
    """

    async def aextract(
        self,
        memcell: MemCell,
        *,
        llm: LLMClient | None = None,
        prompt: str | None = None,
    ) -> list[Episode]:
        """Async main implementation: ask LLM to extract Episodes.

        Args:
            memcell: Source MemCell (boundary output).
            llm: Per-call LLM override (sub-project 2.5 fallback chain).
            prompt: Per-call prompt override; defaults to
                ``EPISODE_EXTRACT_PROMPT_EN``.

        Returns:
            list[Episode] — typically 1 Episode per MemCell, but the LLM
            may emit multiple if it detects sub-events.

        Raises:
            LLMNotConfiguredError, LLMError: same as boundary.
        """
        client = everalgo.llm.resolve(llm)
        rendered = (prompt or EPISODE_EXTRACT_PROMPT_EN).format(
            memcell_text=_render_memcell_text(memcell),
            timestamp=memcell.timestamp,
        )
        response = await client.chat(
            messages=[LLMChatMessage(role="user", content=rendered)],
            response_format={"type": "json_object"},
        )
        return _build_episodes_from_llm_response(response.content, memcell)

    extract = async_to_sync(aextract)
    """Sync bridge — only callable from non-event-loop contexts."""


# Module-level helper functions.


def _render_memcell_text(memcell: MemCell) -> str:
    """Render a MemCell as a prompt-friendly conversation transcript."""
    return "\n".join(f"[{m.role.value}] {m.content}" for m in memcell.messages)


def _build_episodes_from_llm_response(raw: str, memcell: MemCell) -> list[Episode]:
    """Parse LLM JSON and build Episode list.

    parent_id and parent_type are auto-filled from the source memcell
    (LLM is instructed not to emit them; see prompts/en/episode.py).
    """
    parsed = json.loads(raw)
    episodes: list[Episode] = []
    for ep_dict in parsed.get("episodes", []):
        ep_dict.setdefault("parent_type", "memcell")
        ep_dict.setdefault("parent_id", memcell.id)
        episodes.append(Episode.model_validate(ep_dict))
    return episodes
