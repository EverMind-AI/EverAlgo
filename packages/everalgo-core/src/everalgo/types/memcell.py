"""Conversation message + MemCell types — aligned with new-release opensource.

Reference: ``opensource/evermemos-opensource/src/api_specs/memory_models.py::MessageSenderRole`` and
``opensource/evermemos-opensource/src/api_specs/memory_types.py::MemCell`` plus
``opensource/evermemos-opensource/src/api_specs/dtos/memory.py::MessageItem``.

Design notes
------------
- ``MessageRole`` mirrors opensource ``MessageSenderRole`` (USER / ASSISTANT / TOOL).
- ``Message`` mirrors opensource ``MessageItem`` field names (``sender_id`` / ``sender_name``); the
  legacy ``speaker_name`` alias has been removed.
- ``MemCell`` mirrors opensource ``MemCell`` field set (``user_id_list`` / ``original_data`` /
  ``event_id`` / ``group_id`` / ``participants`` / ``sender_ids`` / ``type``) so persistence layers can
  cross-serialise without translation.
- ``MemCell.messages`` is a ``@property`` that reconstructs typed :class:`Message` instances from
  ``original_data`` for downstream extractor ergonomics; it is **not** a stored field and does not appear
  in ``model_dump()`` output (matching opensource ``MemCell.to_dict``).
"""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MessageRole(StrEnum):
    """Conversation role taxonomy — mirrors opensource ``MessageSenderRole``."""

    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class RawDataType(StrEnum):
    """Source format of the raw payload behind a :class:`MemCell` — mirrors opensource ``RawDataType``.

    Only ``CONVERSATION`` is exercised by the chat boundary algorithm. Other values
    (``AGENTCONVERSATION``, ``EMAIL``, ``NOTE``, etc.) are reserved for downstream extractors.
    """

    CONVERSATION = "conversation"
    AGENTCONVERSATION = "agent_conversation"


class Message(BaseModel):
    """Single conversation message — mirrors opensource ``MessageItem`` field shape.

    Core fields (``role`` / ``content`` / ``timestamp``) drive prompt rendering. Identity fields
    (``sender_id`` / ``sender_name``) are populated upstream by caller-side enrichment (opensource
    ``GroupAddRequest.validate_sender_id_required`` enforces ``sender_id`` non-null for group adds at the
    API boundary). ``refer_list`` holds @mention payload (opensource ``MessageItem`` carries this for some
    sources but the new-release boundary detector no longer reads it).

    ``extra="ignore"`` keeps opensource raw payloads (``message_id`` / ``tool_calls`` / ``parsed_*`` / etc.)
    deserialisable without errors.
    """

    role: MessageRole
    content: str
    timestamp: int  # Unix epoch milliseconds
    sender_id: str | None = Field(default=None, description="Stable speaker identity (opensource: sender_id)")
    sender_name: str | None = Field(
        default=None,
        description="Human-readable speaker label for prompt rendering (opensource: sender_name)",
    )
    refer_list: list[dict[str, Any]] = Field(
        default_factory=list,
        description="@mention payload (opensource: refer_list)",
    )

    model_config = ConfigDict(extra="ignore")


class MemCell(BaseModel):
    """Boundary extractor output — a coherent slice of conversation.

    Schema mirrors opensource ``MemCell`` dataclass:
        ``user_id_list`` — participants of the entire group / session (caller-provided context).
        ``original_data`` — list of ``{"message": <msg_dict>}`` items wrapping the raw conversation
        messages (opensource ``_build_original_data_items`` line 181-202). Use the :attr:`messages`
        property for typed access.
        ``timestamp`` — last-message timestamp of the slice (unix epoch milliseconds; opensource uses
        ``datetime`` but EverAlgo retains the ms-int convention for JSON friendliness).
        ``event_id`` — assigned by the persistence layer when saving; algorithm leaves this ``None``.
        ``group_id`` — opaque group identifier (caller-provided).
        ``participants`` / ``sender_ids`` — distinct sender IDs from ``role == "user"`` messages.
        Currently the same value (opensource comment line 152-155: ``participants`` will diverge once
        display-name resolution is available).
        ``type`` — :class:`RawDataType` enum value.

    The :attr:`messages` ``@property`` reconstructs typed :class:`Message` instances from
    ``original_data`` for downstream extractor ergonomics; it is not stored and does not appear in
    ``model_dump()`` output.

    ``extra="ignore"`` keeps opensource MemCell payloads with unmodelled keys deserialisable.
    """

    user_id_list: list[str] = Field(default_factory=list)
    original_data: list[dict[str, Any]] = Field(
        default_factory=list,
        description='List of {"message": msg_dict} items wrapping raw conversation messages.',
    )
    timestamp: int  # Unix epoch milliseconds
    event_id: str | None = Field(default=None, description="Assigned by persistence layer (opensource: event_id)")
    group_id: str | None = None
    participants: list[str] | None = Field(
        default=None,
        description='Distinct sender_ids from role=="user" messages',
    )
    sender_ids: list[str] | None = Field(
        default=None,
        description="Same set as participants (opensource parity)",
    )
    type: RawDataType | None = Field(default=None, description="RawDataType enum value")

    model_config = ConfigDict(extra="ignore")

    @property
    def messages(self) -> list[Message]:
        """Typed view over ``original_data`` — recomputed on each access.

        Reconstructs :class:`Message` instances from the ``{"message": msg_dict}`` wrappers. Use this for
        downstream extractor iteration when typed access is more ergonomic than raw dict access.
        """
        return [Message.model_validate(item["message"]) for item in self.original_data]
