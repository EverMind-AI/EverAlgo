"""Dataset Protocol and shared value types for benchmark datasets."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class Message(BaseModel):
    """A single message in a conversation."""

    model_config = ConfigDict(frozen=True)

    id: str
    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: int  # milliseconds since epoch
    sender_id: str
    sender_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Conversation(BaseModel):
    """A multi-turn conversation."""

    model_config = ConfigDict(frozen=True)

    id: str
    speakers: tuple[str, ...]
    messages: tuple[Message, ...]


class QAPair(BaseModel):
    """A question-answer pair attached to a conversation."""

    model_config = ConfigDict(frozen=True)

    question_id: str
    conv_id: str
    question: str
    golden_answer: str
    category: str  # str unified across datasets (LoCoMo uses 1..5)
    metadata: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class Dataset(Protocol):
    """A benchmark dataset definition. New datasets implement this protocol."""

    name: str

    def load_conversations(self) -> Iterable[Conversation]:
        """Load all conversations."""
        ...

    def load_qa_pairs(self, conv_id: str) -> Iterable[QAPair]:
        """Load all QA pairs for one conversation."""
        ...

    def category_label(self, category: str) -> str:
        """Map category code to human-readable label."""
        ...

    def filter_categories(self) -> set[str]:
        """Categories to exclude from scoring (e.g. adversarial)."""
        ...

    def judge_prompt(self) -> str:
        """LLM-judge prompt template for this dataset."""
        ...

    def answer_prompt(self) -> str:
        """Answer-generation prompt template for this dataset."""
        ...


@dataclass
class EpisodeMemoryRecord:
    """Caller-side extension of algo Episode — adds embedding for downstream indexing.

    Algo's Episode is a pure algorithm type with no embedding. This dataclass is the
    benchmarks-layer superset: everything from Stage 1 extraction plus the embedding
    vector computed immediately after episode generation. Atomic facts are tracked
    separately in :class:`AtomicFactRecord`.
    """

    mc_id: str
    timestamp: int
    subject: str
    summary: str
    content: str  # LLM-generated episode narrative (episode["content"])
    embedding: list[float] | None  # Computed right after episode extraction; None on empty episode


@dataclass
class AtomicFactRecord:
    """Benchmarks-layer container for atomic facts extracted from one MemCell's episode.

    Separates atomic-fact state from episode state so each can be embedded and
    indexed independently. ``fact_embeddings`` is aligned with ``atomic_fact``
    (same length); both are empty lists when the episode body is empty.
    """

    time: str  # LLM-output human-readable time label (e.g. "March 9, 2024"); may be empty
    atomic_fact: list[str]  # Atomic-fact sentences from AtomicFactExtractor
    fact_embeddings: list[list[float]]  # One embedding vector per fact, aligned with atomic_fact
    timestamp: int  # mc.timestamp — segment end time in Unix ms


__all__ = [
    "AtomicFactRecord",
    "Conversation",
    "Dataset",
    "EpisodeMemoryRecord",
    "Message",
    "QAPair",
]
