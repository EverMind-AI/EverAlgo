"""Merge N chronologically-ordered episodes into one accurate narrative."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from asgiref.sync import async_to_sync
from pydantic import BaseModel, Field

from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.prompts import render_prompt
from everalgo.types import Episode
from everalgo.user_memory.episode import _format_episode_prefix
from everalgo.user_memory.prompts.en.reflect import REFLECT_EPISODE_PROMPT, REFLECT_EPISODE_UPDATE_PROMPT

if TYPE_CHECKING:
    from collections.abc import Sequence

    from everalgo.llm.protocols import LLMClient

logger = logging.getLogger(__name__)


def _validate_inputs(episodes: list[Episode], *, min_count: int) -> None:
    """Fail fast: >= *min_count* episodes, sorted by timestamp ascending."""
    if len(episodes) < min_count:
        msg = f"areflect requires at least {min_count} episode(s), got {len(episodes)}"
        raise ValueError(msg)
    for i in range(1, len(episodes)):
        if episodes[i].timestamp < episodes[i - 1].timestamp:
            msg = (
                f"episodes must be sorted by timestamp ascending, "
                f"but episode[{i - 1}].timestamp={episodes[i - 1].timestamp} "
                f"> episode[{i}].timestamp={episodes[i].timestamp}"
            )
            raise ValueError(msg)


def _render_timeline(episodes: list[Episode]) -> str:
    """Render numbered chronological timeline for LLM prompt.

    No bracketed timestamp: ``ep.episode`` already opens with a code-built, format-stable UTC
    prefix (see ``episode.py::_build_episode``). Rendering ``ep.timestamp`` here too would add a
    second, disagreeing time source — a different instant (closing time vs. the body prefix's
    span-start time) in a conflicting 12-hour format — with no basis for the LLM to prefer one.
    """
    lines: list[str] = []
    for i, ep in enumerate(episodes, 1):
        lines.append(f"{i}. {ep.episode}")
    return "\n".join(lines)


class _ReflectOutput(BaseModel):
    """Structured Output schema for LLM response."""

    content: str = Field(description="The merged narrative text")
    title: str = Field(default="", description="Brief topic title for the merged narrative")


class EpisodeReflector:
    """Merge N chronologically-ordered episodes into one accurate narrative.

    Two modes, mirroring ProfileExtractor:
      - INIT  (old_episode=None): full merge from all episodes.
      - UPDATE (old_episode given): update existing narrative with new episodes.

    Args:
        llm: LLM client satisfying the ``LLMClient`` protocol.
    """

    def __init__(self, *, llm: LLMClient) -> None:
        self._llm = llm

    async def areflect(
        self,
        episodes: Sequence[Episode],
        *,
        old_episode: Episode | None = None,
        prompt: str | None = None,
    ) -> Episode:
        """Merge episodes into one narrative.

        Args:
            episodes: Source episodes sorted by timestamp ascending.
                INIT mode: must contain >= 2 items.
                UPDATE mode: must contain >= 1 item (new episodes only).
            old_episode: Existing merged episode. None -> INIT mode. Episode -> UPDATE mode.
            prompt: Prompt override; None uses bundled default for the selected mode.

        Returns:
            Merged Episode. owner_id=None, timestamp=episodes[-1].timestamp.

        Raises:
            ValueError: Too few episodes, unsorted, or unparseable LLM output.
            LLMError: Network or provider failure.
        """
        if old_episode is None:
            return await self._init_merge(episodes, prompt=prompt)
        return await self._update_merge(old_episode, episodes, prompt=prompt)

    reflect = async_to_sync(areflect)

    async def _init_merge(self, episodes: Sequence[Episode], *, prompt: str | None) -> Episode:
        materialized = list(episodes)
        _validate_inputs(materialized, min_count=2)
        timeline = _render_timeline(materialized)
        rendered = render_prompt(REFLECT_EPISODE_PROMPT, prompt, timeline=timeline)
        return await self._call_llm(rendered, materialized)

    async def _update_merge(self, old_episode: Episode, episodes: Sequence[Episode], *, prompt: str | None) -> Episode:
        materialized = list(episodes)
        _validate_inputs(materialized, min_count=1)
        timeline = _render_timeline(materialized)
        rendered = render_prompt(
            REFLECT_EPISODE_UPDATE_PROMPT, prompt, old_episode=old_episode.episode, new_episodes=timeline
        )
        return await self._call_llm(rendered, materialized)

    async def _call_llm(self, rendered: str, episodes: list[Episode]) -> Episode:
        response = await self._llm.chat(
            messages=[LLMChatMessage(role="user", content=rendered)],
            response_format=_ReflectOutput,
        )
        if response.parsed is None:
            raise ValueError("LLM returned no parsed structured output")
        output: _ReflectOutput = response.parsed  # type: ignore[assignment]
        # Mirrors _build_episode's start/end split: prefix = span start (episodes[0]), the
        # `timestamp` field = span end (episodes[-1]). Episode exposes no `items`, so
        # episodes[0].timestamp is the closest available start anchor, and it is monotonically
        # correct because `episodes` is validated chronological in _validate_inputs.
        prefix = _format_episode_prefix(episodes[0].timestamp)
        return Episode(
            owner_id=None,
            episode=f"{prefix} — {output.content}",
            subject=output.title,
            timestamp=episodes[-1].timestamp,
        )
