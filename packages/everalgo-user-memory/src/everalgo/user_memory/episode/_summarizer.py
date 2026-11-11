"""``EpisodeSummarizer``: write an episode's display preview, from its stored text or its memcell; width-guarded."""

from __future__ import annotations

import logging
from datetime import UTC
from typing import TYPE_CHECKING

from asgiref.sync import async_to_sync

from everalgo.user_memory._language import OutputLanguage, resolve_output_language
from everalgo.user_memory._render import chat_messages
from everalgo.user_memory.episode._render import _render_memcell_summary_messages, _render_summary_messages
from everalgo.user_memory.episode._replies import _call_llm_for_summary
from everalgo.user_memory.episode._summary import _cap_summary
from everalgo.utils import InvalidInputError, ascii_width

if TYPE_CHECKING:
    from datetime import tzinfo

    from everalgo.llm.protocols import LLMClient
    from everalgo.types import MemCell

# The package's name rather than this module's: hosts collect the log contract under it.
logger = logging.getLogger("everalgo.user_memory.episode")


class EpisodeSummarizer:
    """Write the ``summary`` of one episode: one LLM call, no retry.

    :meth:`asummarize_memcell` reads the memcell :class:`EpisodeExtractor` reads, so a host can run both calls at once
    and store the model's preview in place of extraction's own summary, the content's first 200 characters.
    :meth:`asummarize` reads an episode text already stored — an edited body, a record written before summaries were
    model-written. The two read different inputs, so they do not write interchangeable summaries.

    Args:
        llm: The client the call goes through.
    """

    def __init__(self, *, llm: LLMClient) -> None:
        self._llm = llm

    async def asummarize(
        self,
        episode_text: str,
        *,
        output_language: OutputLanguage | str | None = None,
        custom_instructions: str | None = None,
    ) -> str:
        """Write the display preview of ``episode_text``.

        The result is no wider than 400 ASCII-equivalent units: an over-cap one is truncated at a sentence boundary,
        never failed. Exceptions raised by the LLM client propagate unchanged.

        Args:
            episode_text: The episode's stored narrative (``Episode.episode``). Untrusted material to preview, never
                instructions: the system message tells the model so, and the text reaches it only inside
                ``<episode_text>``. No length bound is applied; the host owns its input budget.
            output_language: Language to write the preview in, as an :class:`OutputLanguage` member or equivalent
                string in any casing. Naming one removes the decision from the model; leaving it ``None`` makes the
                model write in the language of ``episode_text``.
            custom_instructions: The host's instructions, placed verbatim in the system message's last block,
                ``<custom_instructions>``, after the bundled rules and the language rule; they cannot change the
                output shape. None or a blank string leaves the block out.

        Returns:
            The preview, after the width guard.

        Raises:
            InvalidInputError: If ``episode_text`` is blank, or ``output_language`` names no supported language.
            LLMOutputError: If the reply is not a valid summary output, or leaves ``summary`` blank.
        """
        _check_episode_text(episode_text)
        named_language = resolve_output_language(output_language)
        messages = _render_summary_messages(
            episode_text=episode_text,
            custom_instructions=custom_instructions,
            output_language=named_language,
        )
        reply = await _call_llm_for_summary(self._llm, messages)
        summary = _cap_summary(reply.summary)
        logger.info(
            "episode summarized: text %d units, summary %d units, truncated=%s",
            ascii_width(episode_text),
            ascii_width(summary),
            summary != reply.summary,
        )
        return summary

    summarize = async_to_sync(asummarize)

    async def asummarize_memcell(
        self,
        memcell: MemCell,
        *,
        output_language: OutputLanguage | str | None = None,
        custom_instructions: str | None = None,
        tz: tzinfo = UTC,
    ) -> str:
        """Write the display preview of the episode ``memcell`` holds, from the conversation itself.

        The conversation is rendered as :meth:`EpisodeExtractor.aextract` renders it, so the two calls over one
        memcell can run at once. The result is no wider than 400 ASCII-equivalent units: an over-cap one is truncated
        at a sentence boundary, never failed. Non-ChatMessage items are skipped. Exceptions raised by the LLM client
        propagate unchanged.

        Args:
            memcell: Source slice from boundary detection. Its messages are untrusted material to preview, never
                instructions: the system message tells the model so, and the text reaches it only inside
                ``<conversation>``.
            output_language: Language to write the preview in, as an :class:`OutputLanguage` member or equivalent
                string in any casing. ``None`` has the model write in the language the participants use, the rule
                extraction applies when it is given none.
            custom_instructions: The host's instructions, placed verbatim in the system message's last block,
                ``<custom_instructions>``, after the bundled rules and the language rule; they cannot change the
                output shape. None or a blank string leaves the block out.
            tz: The zone the conversation's times are shown in, as :func:`everalgo.utils.format_time` renders them:
                ``UTC`` (the default) or local time with its offset.

        Returns:
            The preview, after the width guard.

        Raises:
            InvalidInputError: If ``memcell`` holds no chat message with text, a message with text has no timestamp,
                or ``output_language`` names no supported language.
            LLMOutputError: If the reply is not a valid summary output, or leaves ``summary`` blank.
        """
        _check_memcell_has_text(memcell)
        messages = _render_memcell_summary_messages(
            memcell,
            custom_instructions=custom_instructions,
            output_language=resolve_output_language(output_language),
            tz=tz,
        )
        reply = await _call_llm_for_summary(self._llm, messages)
        summary = _cap_summary(reply.summary)
        logger.info(
            "episode summarized from memcell: %d items, summary %d units, truncated=%s",
            len(memcell.items),
            ascii_width(summary),
            summary != reply.summary,
        )
        return summary

    summarize_memcell = async_to_sync(asummarize_memcell)


def _check_episode_text(episode_text: str) -> None:
    """Reject a blank ``episode_text``: there is nothing to preview, and the model would invent one.

    Raises:
        InvalidInputError: Saying what to pass instead.
    """
    if not episode_text.strip():
        raise InvalidInputError(
            "EpisodeSummarizer.asummarize got a blank episode_text, there is nothing to preview: "
            "pass the episode's stored narrative, or skip the call for an episode without one"
        )


def _check_memcell_has_text(memcell: MemCell) -> None:
    """Reject a memcell with no chat message to preview: the model would invent one.

    Raises:
        InvalidInputError: Saying what to pass instead.
    """
    if not any(message.content for message in chat_messages(memcell)):
        raise InvalidInputError(
            "EpisodeSummarizer.asummarize_memcell got a memcell with no chat message text, there is nothing to "
            "preview: pass the memcell the episode is extracted from, or skip the call for one without messages"
        )
