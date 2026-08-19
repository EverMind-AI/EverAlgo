"""Merge N chronologically-ordered episodes into one accurate narrative."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from asgiref.sync import async_to_sync
from pydantic import BaseModel, Field

from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.prompts import render_prompt
from everalgo.types import Episode
from everalgo.user_memory._language import (
    EXISTING_NARRATIVE_LANGUAGE_RULE,
    MERGED_EPISODES_LANGUAGE_RULE,
    OutputLanguage,
    build_language_rule,
)
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

    No bracketed timestamp: ``ep.episode`` already carries the times the extractor's prompt made the
    model write, in a pinned 24-hour ``UTC`` format. Rendering ``ep.timestamp`` here too would add a
    second, disagreeing time source — the span's closing time rather than the events' own — with no
    basis for the LLM to prefer one.
    """
    lines: list[str] = []
    for i, ep in enumerate(episodes, 1):
        lines.append(f"{i}. {ep.episode}")
    return "\n".join(lines)


class _ReflectOutput(BaseModel):
    """Structured Output schema for LLM response.

    ``summary`` is declared after ``content`` deliberately: Structured Output generates fields in schema
    order, so a ``summary`` declared first would be written before the narrative it previews.
    """

    content: str = Field(description="The merged narrative text")
    summary: str = Field(description="Display preview of the merged narrative, under 50 words")
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
        output_language: OutputLanguage | str | None = None,
    ) -> Episode:
        """Merge episodes into one narrative.

        Args:
            episodes: Source episodes sorted by timestamp ascending.
                INIT mode: must contain >= 2 items.
                UPDATE mode: must contain >= 1 item (new episodes only).
            old_episode: Existing merged episode. None -> INIT mode. Episode -> UPDATE mode.
            prompt: Prompt override; None uses bundled default for the selected mode.
            output_language: Language to write the merged narrative in, as an :class:`OutputLanguage` member
                or equivalent string in any casing. ``None`` makes each mode inherit the language of its
                input — the episodes being merged, or the narrative being updated. That inheritance is
                weaker than it looks: merging episodes written in different languages leaves the model to
                pick one, and an update inherits whatever the narrative already says, so a language that
                went wrong once stays wrong. Name a language when the episodes may disagree, or to correct a
                narrative already in the wrong one.

        Returns:
            Merged Episode. owner_id=None, timestamp=episodes[-1].timestamp.

        Raises:
            ValueError: Too few episodes, unsorted, unparseable LLM output, or ``output_language`` names no
                supported language.
            LLMError: Network or provider failure.
        """
        if old_episode is None:
            return await self._init_merge(episodes, prompt=prompt, output_language=output_language)
        return await self._update_merge(old_episode, episodes, prompt=prompt, output_language=output_language)

    reflect = async_to_sync(areflect)

    async def _init_merge(
        self,
        episodes: Sequence[Episode],
        *,
        prompt: str | None,
        output_language: OutputLanguage | str | None,
    ) -> Episode:
        materialized = list(episodes)
        _validate_inputs(materialized, min_count=2)
        timeline = _render_timeline(materialized)
        rendered = render_prompt(
            REFLECT_EPISODE_PROMPT,
            prompt,
            timeline=timeline,
            language_rule=build_language_rule(output_language, fallback=MERGED_EPISODES_LANGUAGE_RULE),
        )
        return await self._call_llm(rendered, materialized)

    async def _update_merge(
        self,
        old_episode: Episode,
        episodes: Sequence[Episode],
        *,
        prompt: str | None,
        output_language: OutputLanguage | str | None,
    ) -> Episode:
        materialized = list(episodes)
        _validate_inputs(materialized, min_count=1)
        timeline = _render_timeline(materialized)
        rendered = render_prompt(
            REFLECT_EPISODE_UPDATE_PROMPT,
            prompt,
            old_episode=old_episode.episode,
            new_episodes=timeline,
            language_rule=build_language_rule(output_language, fallback=EXISTING_NARRATIVE_LANGUAGE_RULE),
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
        return Episode(
            owner_id=None,
            episode=output.content,
            subject=output.title,
            summary=output.summary,
            timestamp=episodes[-1].timestamp,
        )
