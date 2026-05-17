"""Multimodal content blocks inside a ChatMessage."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class TextContent(BaseModel):
    """Plain-text content block.

    Mirrors OpenAI / Anthropic native multimodal content shape.
    """

    type: Literal["text"] = "text"
    text: str

    model_config = ConfigDict(extra="ignore")


# Future: ImageContent / AudioContent / ... will be added as union members here.
ContentBlock = Annotated[TextContent, Field(discriminator="type")]
"""Discriminated union of all supported multimodal content block variants."""
