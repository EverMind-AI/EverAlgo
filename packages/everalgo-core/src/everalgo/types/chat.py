"""Chat conversation wire types — ``ChatMessage`` is the common currency across chat and agent pipelines."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from everalgo.types.content import ContentBlock


class ChatMessage(BaseModel):
    """Single chat-style conversation message.

    ``kind="text"`` is the discriminator for ``ConversationItem``. ``content`` is plain ``str`` (shorthand)
    or ``list[ContentBlock]`` for multimodal payloads.
    """

    kind: Literal["text"] = "text"
    id: str
    role: Literal["user", "assistant"]
    content: str | list[ContentBlock]
    timestamp: int  # Unix epoch milliseconds
    sender_id: str
    sender_name: str | None = Field(
        default=None,
        description="Human-readable speaker label rendered into LLM prompts",
    )

    model_config = ConfigDict(extra="ignore")
