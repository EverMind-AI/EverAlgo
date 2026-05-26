"""User-side memory types — minimal set for the EPISODE path."""

from pydantic import BaseModel, ConfigDict


class Episode(BaseModel):
    """User-side episodic memory — a structured 'what happened' trace.

    ``owner_id`` is the user this episode belongs to, or ``None`` for whole-memcell (generic) episodes
    that do not bind to any user. LLM output maps ``title`` → ``subject`` and ``content`` → ``episode``.
    ``extra="allow"`` surfaces any LLM-emitted secondary fields without a schema bump.
    """

    owner_id: str | None
    episode: str
    subject: str = ""
    timestamp: int  # Unix epoch milliseconds

    model_config = ConfigDict(extra="allow")


class Foresight(BaseModel):
    """User-side foresight memory — anticipated future event or commitment.

    LLM output maps ``content`` → ``foresight``. Date fields are ISO ``YYYY-MM-DD`` strings.
    ``extra="allow"`` surfaces any LLM-emitted secondary fields without a schema bump.
    """

    owner_id: str
    foresight: str
    evidence: str
    timestamp: int  # Unix epoch milliseconds
    start_time: str | None = None  # YYYY-MM-DD
    end_time: str | None = None  # YYYY-MM-DD
    duration_days: int | None = None

    model_config = ConfigDict(extra="allow")


class AtomicFact(BaseModel):
    """User-side atomic fact — a single verifiable assertion lifted from a conversation.

    ``owner_id`` is the user this fact belongs to, or ``None`` for whole-memcell (generic) facts that do not bind to any user.
    Secondary LLM fields (confidence, sources, topic, …) land via ``extra="allow"`` without a schema bump.
    """

    owner_id: str | None
    content: str
    timestamp: int  # Unix epoch milliseconds

    model_config = ConfigDict(extra="allow")


class Profile(BaseModel):
    """User-level aggregate of long-term traits — not anchored to a single MemCell.

    Structured trait fields (interests, habits, preferences, …) land via ``extra="allow"`` without a schema
    bump.
    """

    owner_id: str
    summary: str
    timestamp: int  # Unix epoch milliseconds

    model_config = ConfigDict(extra="allow")
