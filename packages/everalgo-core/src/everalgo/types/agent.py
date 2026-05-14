"""Agent-side memory data contracts — AgentCase / AgentSkill.

Field set is the algorithm-required subset of the opensource ``api_specs/memory_types.py`` shapes (see
``packages/everalgo-agent-memory/src/everalgo/agent_memory/DESIGN.md`` §6 for the mapping). Caller-managed
metadata (``user_id`` / ``group_id`` / ``sender_ids`` / ``vector`` etc.) flow through ``extra="allow"`` so
evermem can attach owner info without forcing a schema bump.
"""

from pydantic import BaseModel, ConfigDict


class AgentCase(BaseModel):
    """One agent-trajectory experience distilled from a single MemCell.

    Each MemCell of agent-conversation type yields at most one AgentCase; multi-turn problem-solving for a
    single task is synthesized into a single record. ``task_intent`` is the retrieval anchor (head-truncated
    to ~300 tokens after extraction); ``approach`` is the natural-language numbered plan with inline
    decisions, results and lessons; ``quality_score`` ∈ [0.0, 1.0] gates the success-vs-failure prompt
    selection downstream in :class:`AgentSkillExtractor`; ``key_insight`` is the optional pivotal-strategy
    quote when the LLM identifies one.

    Caller-managed fields (``vector`` / ``vector_model`` / ``user_id`` / ``group_id`` / ``sender_ids`` / ...)
    are accepted via ``extra="allow"`` — EverAlgo never sets nor reads them; evermem persists them.
    """

    id: str
    timestamp: int  # Unix epoch milliseconds, mirrors MemCell/Episode
    parent_type: str = "memcell"
    parent_id: str

    task_intent: str
    approach: str = ""
    quality_score: float = 0.5
    key_insight: str = ""

    model_config = ConfigDict(extra="allow")


class AgentSkill(BaseModel):
    """Reusable skill aggregated from clustered :class:`AgentCase` instances.

    Skills live under a specific ``cluster_id`` (produced by :func:`everalgo.clustering.cluster_by_llm`).
    ``name`` + ``description`` form the retrieval anchor (caller embeds before persisting); ``content`` is
    the SOP markdown body (≤ 5000 tokens budgeted for downstream prompts). ``confidence`` ∈ [0.0, 1.0] is
    the LLM's belief in the SOP's reliability; values below :attr:`SkillConfig.retire_confidence` signal a
    soft-retire (see agent_memory DESIGN.md §5.2 list-encoding contract). ``maturity_score`` ∈ [0.0, 1.0]
    is the LLM-evaluated readiness score (4-dimension scoring, /20 normalized).

    ``source_case_ids`` traces which :class:`AgentCase` instances contributed to this skill. Embedding
    fields (``vector`` / ``vector_model``) are not part of the schema — they are caller-managed and persist
    outside EverAlgo via ``extra="allow"`` if evermem wants to attach them to model instances.
    """

    id: str
    cluster_id: str

    name: str = ""
    description: str = ""
    content: str = ""
    confidence: float = 0.0
    maturity_score: float = 0.6

    source_case_ids: list[str] = []

    model_config = ConfigDict(extra="allow")
