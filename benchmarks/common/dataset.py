"""Dataset Protocol and shared value types for benchmark datasets."""

from collections.abc import Iterable
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


__all__ = [
    "Conversation",
    "Dataset",
    "Message",
    "QAPair",
]
