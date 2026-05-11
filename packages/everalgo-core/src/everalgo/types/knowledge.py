"""Knowledge memory contract — schema TBD."""

from pydantic import BaseModel, Field


class KnowledgeMemory(BaseModel):
    """File-based knowledge memory unit.

    Stub — schema fields TBD (T1).
    """

    id: str = Field(default="", description="TBD (T1 review)")
    content: str = Field(default="", description="Knowledge content (TBD)")
