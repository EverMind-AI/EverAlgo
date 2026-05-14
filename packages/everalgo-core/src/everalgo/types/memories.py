"""User-side memory types — minimal set for the EPISODE path."""

from pydantic import BaseModel, ConfigDict


class Episode(BaseModel):
    """User-side episodic memory — a structured 'what happened' trace.

    Cross-link: agent paths also produce Episode (mem_memorize.py:870-885 in opensource at release/20260403,
    plus design.md §2.4 line 697: "episode 永远跑"). ``owner_id`` always points to the user, even when the
    source MemCell is an agent conversation; the agent is a participant, not the owner.

    Core fields + ``subject`` (= opensource ``title``) — the LLM emits ``{title, content}`` per
    ``episode_mem_prompts.py`` and we map title → subject, content → episode. ``extra="allow"`` keeps any
    LLM-emitted secondary fields accessible on the model instance.
    """

    id: str
    owner_id: str
    episode: str
    subject: str = ""
    timestamp: int  # Unix epoch milliseconds
    parent_type: str = "memcell"
    parent_id: str

    model_config = ConfigDict(extra="allow")


class Foresight(BaseModel):
    """User-side foresight memory — anticipated future event or commitment.

    Mirrors opensource ``foresight_prompts.py`` output shape: the LLM emits a list of
    ``{content, evidence, start_time, end_time, duration_days}`` items and we map content → foresight.
    Date fields are stored as ISO ``YYYY-MM-DD`` strings (matching opensource's date normalisation).
    ``extra="allow"`` keeps any LLM-emitted secondary fields accessible on the model instance.
    """

    id: str
    owner_id: str
    foresight: str
    evidence: str
    timestamp: int  # Unix epoch milliseconds
    start_time: str | None = None  # YYYY-MM-DD per opensource date normalisation
    end_time: str | None = None  # YYYY-MM-DD
    duration_days: int | None = None
    parent_type: str = "memcell"
    parent_id: str

    model_config = ConfigDict(extra="allow")


class AtomicFact(BaseModel):
    """User-side atomic fact memory — a single verifiable assertion lifted from a conversation.

    Mirrors :class:`Episode` shape: each fact is anchored to a source MemCell. The minimal field set is
    ``id`` / ``owner_id`` / ``fact`` / ``timestamp`` / ``parent_type`` / ``parent_id``; secondary fields
    (confidence / sources / topic / ...) are intentionally omitted and remain accessible via
    ``extra="allow"`` if the LLM emits them.
    """

    id: str
    owner_id: str
    fact: str
    timestamp: int  # Unix epoch milliseconds
    parent_type: str = "memcell"
    parent_id: str

    model_config = ConfigDict(extra="allow")


class Profile(BaseModel):
    """User-side profile memory — long-term user traits derived from a cluster of conversations.

    Unlike :class:`Episode` / :class:`Foresight` / :class:`AtomicFact`, ``Profile`` is a **user-level
    aggregate**, not anchored to a single source MemCell — no ``parent_id`` / ``parent_type``. The minimal
    field set is ``id`` / ``owner_id`` / ``summary`` / ``timestamp``; any structured trait fields
    (interests / habits / preferences / hard_skills / ...) the LLM emits land in the model via
    ``extra="allow"`` and are accessible without a schema bump.
    """

    id: str
    owner_id: str
    summary: str
    timestamp: int  # Unix epoch milliseconds

    model_config = ConfigDict(extra="allow")
