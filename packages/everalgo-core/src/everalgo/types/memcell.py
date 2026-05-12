"""Conversation message types — minimal field set for the EPISODE path.

Reference: design.md §1.2 (boundary + extract phases).
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class MessageRole(StrEnum):
    """Conversation role taxonomy.

    The EPISODE path consumes user/assistant messages only. ``tool`` and ``system`` roles are intentionally
    omitted from the minimal type set and may be added later via a SemVer minor bump (extending an enum with
    new members is a backward-compatible change).
    """

    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    """Single conversation message.

    Minimal field set for the EPISODE path: ``role`` + ``content`` + ``timestamp``. Other fields (sender_id,
    tool_calls, ...) are out of scope for sub-project 1; ``extra="ignore"`` silently drops them so that
    opensource payloads (which carry sender_id / message_id / tool_calls / ...) round-trip cleanly.
    """

    role: MessageRole
    content: str
    timestamp: int  # Unix epoch milliseconds

    model_config = ConfigDict(extra="ignore")


class MemCell(BaseModel):
    """Boundary extractor output — a coherent slice of conversation.

    Minimal field set for the EPISODE path: ``id`` (data lineage anchor
    referenced by Episode.parent_id), ``messages`` (LLM prompt context),
    ``timestamp`` (Episode.timestamp default). Boundary metadata
    (source_type / sender_ids / start_idx / token_count / boundary_reason)
    is added later when the boundary subpackage lands.

    ``extra="ignore"`` keeps opensource MemCell payloads (which carry source_type / user_id_list / group_id /
    participants) deserialisable without errors.
    """

    id: str
    messages: list[Message]
    timestamp: int  # Unix epoch milliseconds

    model_config = ConfigDict(extra="ignore")
