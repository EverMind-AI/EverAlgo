"""User-side memory types — minimal set for the EPISODE path."""

from pydantic import BaseModel, ConfigDict, Field


class Episode(BaseModel):
    """User-side episodic memory — a structured 'what happened' trace.

    Cross-link: agent paths also produce Episode (mem_memorize.py:870-885 in opensource at release/20260403,
    plus design.md §2.4 line 697: "episode 永远跑"). ``owner_id`` always points to the user, even when the
    source MemCell is an agent conversation; the agent is a participant, not the owner.

    Secondary fields (subject / summary / keywords / location / start_time / end_time / sender_ids /
    original_data) are intentionally omitted from the minimal type. ``extra="allow"`` keeps any LLM-emitted
    secondary fields accessible on the model instance until a future minor bump promotes them to first-class
    fields.
    """

    id: str
    owner_id: str
    episode: str
    timestamp: int  # Unix epoch milliseconds
    parent_type: str = "memcell"
    parent_id: str

    model_config = ConfigDict(extra="allow")


class Foresight(BaseModel):
    """User-side foresight memory — anticipated future event / commitment.

    Stub — schema fields TBD (T1).
    """

    id: str = Field(default="", description="TBD (T1 review)")
    text: str = Field(default="", description="TBD (T1 review)")


class AtomicFact(BaseModel):
    """User-side atomic fact memory — single verifiable fact.

    Stub — schema fields TBD (T1).
    """

    id: str = Field(default="", description="TBD (T1 review)")
    text: str = Field(default="", description="TBD (T1 review)")


class Profile(BaseModel):
    """User-side profile memory — incrementally edited user profile.

    Stub — schema fields TBD (T1).
    """

    id: str = Field(default="", description="TBD (T1 review)")
    summary: str = Field(default="", description="TBD (T1 review)")
