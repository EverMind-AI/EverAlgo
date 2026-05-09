"""Multimodal parser output contract — schema TBD."""

from pydantic import BaseModel, Field


class ParsedContent(BaseModel):
    """Normalized output of any parser submodule (image / audio / doc / ...).

    Stub — schema fields TBD (T1).
    """

    id: str = Field(default="", description="TBD (T1 review)")
    text: str = Field(default="", description="Extracted text (TBD)")
    mime: str = Field(default="", description="Source MIME type")
