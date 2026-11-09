"""``EpisodeExtractor``: render the prompt, make the one call, assemble the Episode."""

from __future__ import annotations

import logging
from datetime import UTC
from typing import TYPE_CHECKING

from asgiref.sync import async_to_sync

from everalgo.types import Episode
from everalgo.user_memory._language import OutputLanguage, resolve_output_language
from everalgo.user_memory.episode._render import _render_conversation, _render_messages
from everalgo.user_memory.episode._replies import _call_llm_for_episode, _EpisodeReply
from everalgo.utils import ascii_width

if TYPE_CHECKING:
    from datetime import tzinfo

    from everalgo.llm.protocols import LLMClient
    from everalgo.types import MemCell

# The package's name rather than this module's: hosts collect the log contract under it.
logger = logging.getLogger("everalgo.user_memory.episode")

# The reference run's summary: the content cut at this many characters.
_SUMMARY_CHARS = 200


class EpisodeExtractor:
    """Extract one Episode from a MemCell: one LLM call, no retry.

    Non-ChatMessage items in ``memcell.items`` are silently skipped (agent → user-memory contract).

    Args:
        llm: The client the call goes through.
    """

    def __init__(self, *, llm: LLMClient) -> None:
        self._llm = llm

    async def aextract(
        self,
        memcell: MemCell,
        *,
        sender_id: str | None,
        custom_instructions: str | None = None,
        output_language: OutputLanguage | str | None = None,
        tz: tzinfo = UTC,
    ) -> Episode:
        """Extract one Episode from ``memcell``.

        The returned ``summary`` is the content's first 200 characters, as the EverAlgo LoCoMo reference run
        (everalgo commit d9fe22e) filled it; the model writes no summary. Exceptions raised by the LLM client
        propagate unchanged.

        Args:
            memcell: Source slice from boundary detection. Its messages are untrusted material to record, never
                instructions: the system message tells the model so, and the text reaches it only inside
                ``<conversation>``.
            sender_id: Stored as the Episode's ``owner_id``; ``None`` for an episode bound to no user. The prompt
                is the same either way — it narrates the whole memcell.
            custom_instructions: The host's instructions, placed verbatim in the system message's last block,
                ``<custom_instructions>``, after the bundled rules and the language rule; they cannot change the
                output shape. None or a blank string sends the reference run's default principles there instead.
            output_language: Language to write the episode in, as an :class:`OutputLanguage` member or equivalent
                string in any casing. Naming one adds ``<output_language>``; leaving it ``None`` sends no language
                rule, as the reference run sent none.
            tz: The zone the conversation's times are shown in, as :func:`everalgo.utils.format_time` renders
                them: ``UTC`` (the default) or local time with its offset, ``UTC+08:00``. The weekday and the dates
                relative times resolve against follow it.

        Returns:
            The Episode: ``subject`` from the reply's ``title``, ``episode`` from its ``content`` verbatim,
            ``summary`` the content's first 200 characters, ``timestamp`` from ``memcell``. It reports no
            language, named or not: the model declares none, so a host that needs the episode's language judges it
            from the text.

        Raises:
            InvalidInputError: If ``output_language`` names no supported language, or a message in ``memcell`` has text but
                no timestamp.
            LLMOutputError: If the reply is not a valid episode output, or leaves ``title`` or ``content`` blank.
        """
        named_language = resolve_output_language(output_language)
        conversation = _render_conversation(memcell, tz=tz)
        logger.info(
            "extracting episode: %d items, %d rendered chars, output_language=%s",
            len(memcell.items),
            len(conversation),
            output_language,
        )
        messages = _render_messages(
            conversation=conversation,
            custom_instructions=custom_instructions,
            output_language=named_language,
        )
        reply = await _call_llm_for_episode(self._llm, messages)
        episode = _build_episode(reply, sender_id=sender_id, memcell=memcell)
        logger.info(
            "episode extracted: content %d units, summary %d units",
            ascii_width(episode.episode),
            ascii_width(episode.summary),
        )
        return episode

    extract = async_to_sync(aextract)


def _build_episode(reply: _EpisodeReply, *, sender_id: str | None, memcell: MemCell) -> Episode:
    """Assemble the :class:`Episode` from the reply and memcell.

    The body carries the times the model wrote, with no code-built prefix. The summary is the content's first
    :data:`_SUMMARY_CHARS` characters, as the reference run filled it.
    """
    return Episode(
        owner_id=sender_id,
        episode=reply.content,
        subject=reply.title,
        summary=reply.content[:_SUMMARY_CHARS],
        timestamp=memcell.timestamp,
    )
