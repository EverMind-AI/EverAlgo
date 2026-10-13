"""Merge N chronologically-ordered episodes into one accurate narrative."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from asgiref.sync import async_to_sync

from everalgo.llm import LLMMessage
from everalgo.types import Episode
from everalgo.user_memory._language import (
    EXISTING_NARRATIVE_LANGUAGE_RULE,
    MERGED_EPISODES_LANGUAGE_RULE,
    OutputLanguage,
    build_language_rule,
)
from everalgo.user_memory._render import nonblank
from everalgo.user_memory._replies import _call_llm_for_reflect

# Re-exported by name: hosts check their own prompt text against its fields through this module.
from everalgo.user_memory._replies import _ReflectOutput as _ReflectOutput
from everalgo.user_memory.prompts.en.reflect import (
    REFLECT_EPISODE_SYSTEM_PROMPT,
    REFLECT_EPISODE_UPDATE_SYSTEM_PROMPT,
    REFLECT_EPISODE_UPDATE_USER_PROMPT,
    REFLECT_EPISODE_USER_PROMPT,
    assemble_system_prompt,
)
from everalgo.utils import render_prompt

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


def _render_messages(rules: str, *, custom_instructions: str | None, language_rule: str, user: str) -> list[LLMMessage]:
    """The ``[system, user]`` pair: the mode's rules and the call-level blocks, then the data.

    Args:
        rules: The mode's static system rules.
        custom_instructions: The caller's instructions. Placed verbatim in ``<custom_instructions>``, never rendered as
            a template; None or a blank string leaves the block out.
        language_rule: The ``<output_language>`` text.
        user: The rendered user message.
    """
    system = assemble_system_prompt(
        rules, custom_instructions=nonblank(custom_instructions), language_rule=language_rule
    )
    return [LLMMessage(role="system", content=system), LLMMessage(role="user", content=user)]


class EpisodeReflector:
    """Merge N chronologically-ordered episodes into one accurate narrative: one LLM call, no retry.

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
        custom_instructions: str | None = None,
        output_language: OutputLanguage | str | None = None,
    ) -> Episode:
        """Merge episodes into one narrative.

        The model gets a system message of rules and a user message holding only the episodes (and, in UPDATE
        mode, the narrative being updated). Exceptions raised by the LLM client propagate unchanged.

        Args:
            episodes: Source episodes sorted by timestamp ascending.
                INIT mode: must contain >= 2 items.
                UPDATE mode: must contain >= 1 item (new episodes only).
            old_episode: Existing merged episode. None -> INIT mode. Episode -> UPDATE mode.
            custom_instructions: The caller's instructions, placed verbatim in the selected mode's system message as a
                ``<custom_instructions>`` block, closing the message after ``<output_language>``. Never
                rendered as a template, so braces stay literal; None or a blank string leaves the block out.
            output_language: Language to write the merged narrative in, as an :class:`OutputLanguage` member
                or equivalent string in any casing. ``None`` makes each mode inherit the language of its
                input — the episodes being merged, or the narrative being updated. That inheritance is
                weaker than it looks: merging episodes written in different languages leaves the model to
                pick one, and an update inherits whatever the narrative already says, so a language that
                went wrong once stays wrong. Name a language when the episodes may disagree, or to correct a
                narrative already in the wrong one.

        Returns:
            Merged Episode: ``episode`` from the reply's ``content``, ``summary`` from its ``summary``, ``subject``
            from its ``title``; owner_id=None, timestamp=episodes[-1].timestamp.

        Raises:
            ValueError: Too few episodes, unsorted, or ``output_language`` names no supported language.
            LLMOutputError: If the reply is not a valid reflect output.
        """
        if old_episode is None:
            return await self._init_merge(
                episodes, custom_instructions=custom_instructions, output_language=output_language
            )
        return await self._update_merge(
            old_episode, episodes, custom_instructions=custom_instructions, output_language=output_language
        )

    reflect = async_to_sync(areflect)

    async def _init_merge(
        self,
        episodes: Sequence[Episode],
        *,
        custom_instructions: str | None,
        output_language: OutputLanguage | str | None,
    ) -> Episode:
        materialized = list(episodes)
        _validate_inputs(materialized, min_count=2)
        messages = _render_messages(
            REFLECT_EPISODE_SYSTEM_PROMPT,
            custom_instructions=custom_instructions,
            language_rule=build_language_rule(output_language, fallback=MERGED_EPISODES_LANGUAGE_RULE),
            user=render_prompt(REFLECT_EPISODE_USER_PROMPT, None, timeline=_render_timeline(materialized)),
        )
        reply = await _call_llm_for_reflect(self._llm, messages)
        return _build_episode(reply, timestamp=materialized[-1].timestamp)

    async def _update_merge(
        self,
        old_episode: Episode,
        episodes: Sequence[Episode],
        *,
        custom_instructions: str | None,
        output_language: OutputLanguage | str | None,
    ) -> Episode:
        materialized = list(episodes)
        _validate_inputs(materialized, min_count=1)
        messages = _render_messages(
            REFLECT_EPISODE_UPDATE_SYSTEM_PROMPT,
            custom_instructions=custom_instructions,
            language_rule=build_language_rule(output_language, fallback=EXISTING_NARRATIVE_LANGUAGE_RULE),
            user=render_prompt(
                REFLECT_EPISODE_UPDATE_USER_PROMPT,
                None,
                old_episode=old_episode.episode,
                new_episodes=_render_timeline(materialized),
            ),
        )
        reply = await _call_llm_for_reflect(self._llm, messages)
        return _build_episode(reply, timestamp=materialized[-1].timestamp)


def _build_episode(reply: _ReflectOutput, *, timestamp: int) -> Episode:
    """The merged :class:`Episode`, bound to no owner, carrying the reply's three fields verbatim."""
    return Episode(
        owner_id=None,
        episode=reply.content,
        subject=reply.title,
        summary=reply.summary,
        timestamp=timestamp,
    )
