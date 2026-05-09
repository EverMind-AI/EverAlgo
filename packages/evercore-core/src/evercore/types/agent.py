"""Agent-side memory data contracts — schema TBD."""

from pydantic import BaseModel, Field


class AgentCase(BaseModel):
    """Single agent execution case (one task instance).

    Stub — schema fields TBD (T1).
    """

    id: str = Field(default="", description="TBD (T1 review)")
    task_intent: str = Field(default="", description="Task intent description (TBD)")


class AgentSkill(BaseModel):
    """Reusable agent skill aggregated from cluster of cases.

    Stub — schema fields TBD (T1).
    """

    id: str = Field(default="", description="TBD (T1 review)")
    name: str = Field(default="", description="Skill name (TBD)")
