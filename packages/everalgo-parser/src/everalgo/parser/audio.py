"""Audio parser — ASR via multimodal LLM.

Sends the audio bytes as a single ``image_url`` data URI (OpenRouter / Gemini accepts
audio MIMEs through the same content-part shape) and the LLM transcribes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from asgiref.sync import async_to_sync

from everalgo.llm import ChatMessage, LLMClient, TextPart, image_url_part_from_bytes
from everalgo.parser.prompts.en import PROMPT_FOR_AUDIO
from everalgo.types import Modality, ParsedContent

if TYPE_CHECKING:
    from everalgo.types import RawFile

__all__ = ["aparse", "parse"]

logger = logging.getLogger(__name__)


_MIME_MAP: dict[str, str] = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "m4a": "audio/mp4",
    "amr": "audio/amr",
    "aiff": "audio/aiff",
    "aac": "audio/aac",
    "ogg": "audio/ogg",
    "flac": "audio/flac",
}


async def aparse(raw_file: RawFile, *, llm: LLMClient) -> ParsedContent:
    """Parse an audio ``RawFile`` into ``ParsedContent`` (transcription)."""
    if not raw_file.content:
        raise ValueError("RawFile.content is empty; cannot parse audio")

    extension = raw_file.extension.lower().lstrip(".")
    mime = _MIME_MAP.get(extension)
    if mime is None:
        raise ValueError(
            f"audio.aparse: unsupported audio extension {raw_file.extension!r}. Supported: {sorted(_MIME_MAP)}"
        )

    message = ChatMessage(
        role="user",
        content=[
            TextPart(text=PROMPT_FOR_AUDIO),
            image_url_part_from_bytes(raw_file.content, mime),
        ],
    )
    logger.debug("parsing audio extension=%s bytes=%d", extension, len(raw_file.content))
    response = await llm.chat([message])
    return ParsedContent(
        text=response.content,
        modality=Modality.AUDIO,
        mime=raw_file.mime or mime,
        metadata={
            "model": response.model,
            "finish_reason": response.finish_reason,
        },
    )


parse = async_to_sync(aparse)
"""Sync bridge — see ``document.parse`` for details."""
