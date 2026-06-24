"""Shared types for benchmark pipeline stages."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from everalgo.llm.errors import LLMError

# Retryable exception tuple shared by Extract and Enrich stages.
_RETRYABLE_EXC = (json.JSONDecodeError, ValueError, LLMError)

if TYPE_CHECKING:
    from pathlib import Path

    from benchmarks.common.config import BenchmarkConfig
    from benchmarks.common.dataset import Dataset
    from benchmarks.common.services import Services


@dataclass(frozen=True)
class StageContext:
    """Immutable per-stage execution context.

    A stage receives this bundle; it provides everything needed to read the
    previous stage's output, write the current stage's output, and call
    external services or LLMs.
    """

    config: BenchmarkConfig
    services: Services
    dataset: Dataset
    input_dir: Path
    output_dir: Path
    smoke: bool = False
    smoke_conv_limit: int = 1
    smoke_qa_limit: int = 10
    smoke_msg_limit: int = 50
    conv_indices: list[int] | None = None


@dataclass
class StageStats:
    """Per-stage execution metrics. Mutable so stages can accumulate."""

    stage_name: str
    duration_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    http_calls: int = 0
    success: int = 0
    failed: int = 0
    extra: dict[str, Any] = field(default_factory=dict)  # type: ignore[arg-type]

    def combine(self, other: StageStats) -> StageStats:
        """Merge stats from a sibling task into a single aggregate.

        Both must share the same ``stage_name``.
        """
        if self.stage_name != other.stage_name:
            raise ValueError(f"cannot combine stats from different stages: {self.stage_name!r} vs {other.stage_name!r}")
        return StageStats(
            stage_name=self.stage_name,
            duration_seconds=self.duration_seconds + other.duration_seconds,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            http_calls=self.http_calls + other.http_calls,
            success=self.success + other.success,
            failed=self.failed + other.failed,
            extra={**self.extra, **other.extra},
        )
