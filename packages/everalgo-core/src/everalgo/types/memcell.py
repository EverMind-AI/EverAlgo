"""Conversation message + MemCell types — aligned with new-release opensource + agent-trace path.

Reference: ``opensource/evermemos-opensource/src/api_specs/memory_models.py::MessageSenderRole`` and
``opensource/evermemos-opensource/src/api_specs/memory_types.py::MemCell`` plus
``opensource/evermemos-opensource/src/api_specs/dtos/memory.py::MessageItem``.

Design notes
------------
- ``MessageRole`` mirrors opensource ``MessageSenderRole`` (USER / ASSISTANT / TOOL).
- ``Message`` mirrors opensource ``MessageItem`` field names (``sender_id`` / ``sender_name``); the
  legacy ``speaker_name`` alias has been removed. It also carries the OpenAI agent-trace fields
  ``tool_calls`` / ``tool_call_id`` so :class:`agent_memory.case.AgentCaseExtractor` can consume
  ``MemCell.messages`` directly without a separate envelope.
- ``MemCell`` mirrors opensource ``MemCell`` field set (``user_id_list`` / ``original_data`` /
  ``event_id`` / ``group_id`` / ``participants`` / ``sender_ids`` / ``type``) so persistence layers can
  cross-serialise without translation.
- ``MemCell.messages`` is a ``@property`` that reconstructs typed :class:`Message` instances from
  ``original_data`` for downstream extractor ergonomics; it is **not** a stored field and does not appear
  in ``model_dump()`` output (matching opensource ``MemCell.to_dict``).
- System prompts are intentionally excluded from the role enum — they are upstream framing, not memory
  content, and boundary extractors strip system prompts before emitting :class:`MemCell` instances.
"""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MessageRole(StrEnum):
    """Conversation role taxonomy — mirrors opensource ``MessageSenderRole``.

    USER / ASSISTANT cover the chat path; TOOL is added for the agent-trace path so AgentCaseExtractor can
    read tool execution results verbatim.
    """

    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class RawDataType(StrEnum):
    """Source format of the raw payload behind a :class:`MemCell` — mirrors opensource ``RawDataType``.

    Only ``CONVERSATION`` is exercised by the chat boundary algorithm. ``AGENTCONVERSATION`` is consumed
    by :class:`agent_memory.case.AgentCaseExtractor`. Other values (``EMAIL``, ``NOTE``, ...) are reserved
    for downstream extractors.
    """

    CONVERSATION = "conversation"
    AGENTCONVERSATION = "agent_conversation"


class ToolCall(BaseModel):
    """OpenAI chat-completion tool call (function-calling).

    Attached to an assistant :class:`Message` when the agent decides to invoke one or more tools. The
    ``function.arguments`` field is a JSON-encoded string (per OpenAI's wire format) — the algorithm
    layer parses it on demand rather than upfront.
    """

    id: str
    type: str = "function"
    function: dict[str, Any]
    """OpenAI ``function`` payload: ``{"name": str, "arguments": str}`` where ``arguments`` is JSON text."""

    model_config = ConfigDict(extra="ignore")


class Message(BaseModel):
    """Single conversation message — mirrors opensource ``MessageItem`` + OpenAI agent-trace fields.

    Core fields (``role`` / ``content`` / ``timestamp``) drive prompt rendering. Identity fields
    (``sender_id`` / ``sender_name``) are populated upstream by caller-side enrichment (opensource
    ``GroupAddRequest.validate_sender_id_required`` enforces ``sender_id`` non-null for group adds at the
    API boundary). ``refer_list`` holds @mention payload (opensource ``MessageItem`` carries this for some
    sources but the new-release boundary detector no longer reads it).

    Agent-trace fields (``tool_calls`` / ``tool_call_id``) mirror the OpenAI Chat Completions wire format
    so :class:`agent_memory.case.AgentCaseExtractor` can read the full execution trajectory verbatim.
    ``content`` is ``str | None`` because OpenAI emits ``content=null`` on assistant messages that carry
    only ``tool_calls``.

    ``extra="ignore"`` keeps opensource raw payloads (``message_id`` / ``parsed_*`` / etc.) deserialisable.
    """

    role: MessageRole
    content: str | None = None
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
    tool_calls: list[ToolCall] | None = None
    """Assistant function-call invocations (OpenAI ``message.tool_calls``); ``None`` on non-assistant or
    plain-response messages."""

    tool_call_id: str | None = None
    """ID of the originating assistant tool call (set on ``role="tool"`` messages to thread the response
    back to its invocation; OpenAI ``message.tool_call_id``)."""

    model_config = ConfigDict(extra="ignore")


class MemCell(BaseModel):
    """Boundary extractor output — a coherent slice of conversation or agent trajectory.

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
