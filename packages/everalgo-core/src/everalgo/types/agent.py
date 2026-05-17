"""Agent trajectory wire types (``ToolCallRequest``, ``ToolCallResult``) and agent-memory output types."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Trajectory wire types — agent-specific items in a ConversationItem union
# ---------------------------------------------------------------------------


class ToolCallFunction(BaseModel):
    """Function descriptor inside a single :class:`ToolCall` (OpenAI Chat Completion shape)."""

    name: str
    arguments: str  # Raw JSON string per OpenAI spec; algorithm library does not pre-parse.


class ToolCall(BaseModel):
    """A single tool invocation declared by the assistant within :class:`ToolCallRequest`."""

    id: str  # Unique identifier — referenced back by ToolCallResult.tool_call_id.
    type: Literal["function"] = "function"
    function: ToolCallFunction


class ToolCallRequest(BaseModel):
    """Assistant-emitted request to invoke one or more tools.

    A single turn may issue several tool calls in parallel; ``tool_calls`` is the ordered list.
    ``content`` carries the optional reasoning text the assistant produced alongside the calls.
    """

    kind: Literal["tool_call"] = "tool_call"
    tool_calls: list[ToolCall]
    timestamp: int  # Unix epoch milliseconds
    content: str | None = None
    sender_id: str = Field(description="Assistant identity")
    sender_name: str | None = Field(default=None, description="Assistant display name")

    model_config = ConfigDict(extra="ignore")


class ToolCallResult(BaseModel):
    """Result returned by tool execution; ``tool_call_id`` points back to the originating :class:`ToolCall`."""

    kind: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    content: str
    timestamp: int  # Unix epoch milliseconds

    model_config = ConfigDict(extra="ignore")


# ---------------------------------------------------------------------------
# Agent-memory output types — produced by everalgo-agent-memory extractors
# ---------------------------------------------------------------------------


class AgentCase(BaseModel):
    """One agent-trajectory experience distilled from a single MemCell — at most one case per cell.

    ``quality_score`` ∈ [0.0, 1.0] gates success-vs-failure prompt selection in ``AgentSkillExtractor``.
    Caller-managed fields (``vector``, ``user_id``, …) are accepted via ``extra="allow"``.
    """

    id: str
    timestamp: int  # Unix epoch milliseconds, mirrors source MemCell
    task_intent: str
    approach: str = ""
    quality_score: float = 0.5
    key_insight: str = ""

    model_config = ConfigDict(extra="allow")


class AgentSkill(BaseModel):
    """Reusable skill aggregated from clustered ``AgentCase`` instances.

    ``confidence`` ∈ [0.0, 1.0] gates soft-retire; ``maturity_score`` ∈ [0.0, 1.0] is LLM readiness.
    Embedding / caller-managed fields persist via ``extra="allow"``.

    ``cluster_id`` is caller-stamped after extraction (algo emits empty); caller fills it from its own cluster identity mapping.
    """

    id: str
    cluster_id: str = ""

    name: str = ""
    description: str = ""
    content: str = ""
    confidence: float = 0.0
    maturity_score: float = 0.6  # populated only when skip_maturity_scoring=False is passed to aextract

    source_case_ids: list[str] = []

    model_config = ConfigDict(extra="allow")
