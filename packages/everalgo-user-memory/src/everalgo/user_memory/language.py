"""``LanguageDetector``: name the primary language of a chat record, for the extractors to write in.

A host calls it on the window boundary detection sees, alongside the detector, then passes the result to the
extractors as ``output_language``: a named language is followed far more reliably than one a writing model has to
work out for itself (``prompts/en/_language.py``). One model call, no retry. The record is the window's chat messages,
user and assistant, in order; tool items are skipped. Each message is shown as ``[role] text`` with no sender name, a
long one cut to three windows at its head, middle and tail. The reply is one label, Chinese, English, Japanese or
Other; Other, every other language, comes back as English.

Log lines, on ``everalgo.user_memory.language``; none carries a prompt or model text:

    ``detecting language: <n> messages, <n> rendered chars`` (INFO)
        Once per call that asks the model: the chat messages with text, and the record's length as rendered.
    ``language detected: <label>`` (INFO)
        Once per call that gets a valid reply; ``<label>`` is the reply's, ``Other`` included.
"""

from __future__ import annotations

__all__ = ["LanguageDetector"]

import logging
from typing import TYPE_CHECKING, Literal

from asgiref.sync import async_to_sync

from everalgo.llm import LLMMessage, StrictOutputModel, parse_reply
from everalgo.types import ChatMessage
from everalgo.user_memory._language import OutputLanguage
from everalgo.user_memory.prompts.en.language import LANGUAGE_USER_PROMPT, assemble_system_prompt
from everalgo.utils import InvalidInputError, render_prompt

if TYPE_CHECKING:
    from collections.abc import Sequence

    from everalgo.llm.protocols import LLMClient
    from everalgo.types import ConversationItem

logger = logging.getLogger(__name__)

_OTHER = "Other"
#: What Other comes back as: the language written for everybody outside Chinese and Japanese.
_OTHER_LANGUAGE = OutputLanguage.ENGLISH
#: Windows a long message is cut to. Two, head and tail, showed only the wrapper of a template whose human-written
#: instructions sat in its middle; a third window in the middle fixed those and changed no other measured record.
_WINDOWS = 3


class _LanguageReply(StrictOutputModel):
    """Language reply: the ``{"language"}`` skeleton of the prompt's reply line, one of its labels."""

    language: Literal["Chinese", "English", "Japanese", "Other"]


class LanguageDetector:
    """Name the primary language of a chat record: one LLM call, no retry. Native-async with a sync bridge.

    Args:
        llm: The client the call goes through.
    """

    def __init__(self, *, llm: LLMClient) -> None:
        self._llm = llm

    async def adetect(
        self,
        items: Sequence[ConversationItem],
        *,
        custom_instructions: str | None = None,
        user_chars: int = 200,
        assistant_chars: int = 100,
    ) -> OutputLanguage | None:
        """Name the language ``items`` are carried on in.

        Args:
            items: Ordered conversation items, typically the window passed to boundary detection. Only
                ``ChatMessage`` items with text are read; tool items are skipped. Their text is material to judge,
                never instructions: it reaches the model only inside ``<chat_record>``.
            custom_instructions: The caller's instructions, placed verbatim in the system message's last block,
                ``<custom_instructions>``; they add to the rules and cannot change the reply's shape. ``None`` or a
                blank string leaves the block out.
            user_chars: A user message longer than three times this is shown as three windows of ``user_chars``
                characters, at its head, middle and tail, joined by ``" … "``.
            assistant_chars: The same cut for an assistant message.

        Returns:
            Chinese, Japanese or English as the model names it; English too for a language outside those three.
            ``None`` only when ``items`` hold no chat message with text, in which case no model is called.

        Raises:
            InvalidInputError: If ``user_chars`` or ``assistant_chars`` is below 1.
            LLMOutputError: If the reply is not a valid language reply. Exceptions raised by the LLM client propagate
                unchanged.
        """
        keep = _cuts(user_chars=user_chars, assistant_chars=assistant_chars)
        messages = [item for item in items if isinstance(item, ChatMessage) and item.content.strip()]
        if not messages:
            return None
        record = _render_chat_record(messages, keep=keep)
        logger.info("detecting language: %d messages, %d rendered chars", len(messages), len(record))
        reply = await _call_llm_for_language(
            self._llm, _render_messages(record, custom_instructions=custom_instructions)
        )
        logger.info("language detected: %s", reply.language)
        return _OTHER_LANGUAGE if reply.language == _OTHER else OutputLanguage(reply.language)

    detect = async_to_sync(adetect)
    """Sync bridge — only callable from non-event-loop contexts."""


def _cuts(*, user_chars: int, assistant_chars: int) -> dict[str, int]:
    """The characters each window of a role's long message keeps.

    Raises:
        InvalidInputError: If either is below 1: a window of 0 characters would show nothing of the message.
    """
    cuts = {"user": user_chars, "assistant": assistant_chars}
    for role, chars in cuts.items():
        if chars < 1:
            raise InvalidInputError(f"{role}_chars must be at least 1, got {chars}")
    return cuts


def _render_chat_record(messages: Sequence[ChatMessage], *, keep: dict[str, int]) -> str:
    """The record's body: one ``[role] text`` block per message, a blank line apart, each long text cut to windows.

    Sender names are left out on purpose: a name says nothing about the language its owner writes in, and shown, it
    pulls the model towards the language the name looks like.
    """
    return "\n\n".join(
        f"[{message.role}] {_excerpts(message.content.strip(), keep[message.role])}" for message in messages
    )


def _excerpts(text: str, keep: int) -> str:
    """``text`` itself when it is at most ``_WINDOWS * keep`` characters, else evenly spaced windows joined by `` … ``.

    The windows are ``keep`` characters each, the first at the head and the last at the tail, the rest spread evenly
    between them.
    """
    if len(text) <= _WINDOWS * keep:
        return text
    step = (len(text) - keep) / (_WINDOWS - 1)
    return " … ".join(text[round(i * step) : round(i * step) + keep] for i in range(_WINDOWS))


def _render_messages(record: str, *, custom_instructions: str | None) -> list[LLMMessage]:
    """The ``[system, user]`` pair: the rules and the caller's block, then the record in ``<chat_record>``."""
    system = assemble_system_prompt(custom_instructions=custom_instructions)
    user = render_prompt(LANGUAGE_USER_PROMPT, None, chat_record=record)
    return [LLMMessage(role="system", content=system), LLMMessage(role="user", content=user)]


async def _call_llm_for_language(llm: LLMClient, messages: list[LLMMessage]) -> _LanguageReply:
    """Request a language reply and return it validated (``docs/concepts/llm-contract.md`` §4, §5).

    Raises:
        LLMOutputError: If the reply is not a valid language reply.
    """
    text = await llm.achat(messages=messages, response_format=_LanguageReply)
    return parse_reply(_LanguageReply, text, operator="language")
