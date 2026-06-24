"""Tests for stage shared types."""

from pathlib import Path

import pytest

from benchmarks.common.config import BenchmarkConfig
from benchmarks.common.stages.types import StageContext, StageStats


def test_stage_stats_defaults():
    """A fresh StageStats has zero metrics."""
    s = StageStats(stage_name="extract")
    assert s.stage_name == "extract"
    assert s.duration_seconds == 0.0
    assert s.prompt_tokens == 0
    assert s.completion_tokens == 0
    assert s.http_calls == 0
    assert s.success == 0
    assert s.failed == 0
    assert s.extra == {}


def test_stage_stats_combine_aggregates_counts():
    """Combining two stats sums numeric fields and merges extras."""
    a = StageStats(stage_name="x", duration_seconds=1.0, prompt_tokens=10, success=2)
    b = StageStats(stage_name="x", duration_seconds=2.0, prompt_tokens=20, success=3)
    c = a.combine(b)
    assert c.stage_name == "x"
    assert c.duration_seconds == 3.0
    assert c.prompt_tokens == 30
    assert c.success == 5


def test_stage_stats_combine_rejects_mismatched_names():
    """combine() must refuse to merge stats from different stages."""
    a = StageStats(stage_name="extract")
    b = StageStats(stage_name="index")
    with pytest.raises(ValueError, match="cannot combine stats"):
        a.combine(b)


def test_stage_context_is_frozen():
    """StageContext must be immutable."""
    # Construct with minimal mock objects — dataset/services can be None for this test
    # since we only test immutability of the dataclass itself.
    ctx = StageContext(
        config=BenchmarkConfig(),
        services=None,  # type: ignore[arg-type]
        dataset=None,  # type: ignore[arg-type]
        input_dir=Path("/tmp/in"),
        output_dir=Path("/tmp/out"),
    )
    with pytest.raises((AttributeError, TypeError)):  # FrozenInstanceError
        ctx.smoke = True  # type: ignore[misc]


def test_stage_context_smoke_defaults():
    """Smoke defaults to False; smoke limits have sensible defaults."""
    ctx = StageContext(
        config=BenchmarkConfig(),
        services=None,  # type: ignore[arg-type]
        dataset=None,  # type: ignore[arg-type]
        input_dir=Path("/tmp/in"),
        output_dir=Path("/tmp/out"),
    )
    assert ctx.smoke is False
    assert ctx.smoke_conv_limit == 1
    assert ctx.smoke_qa_limit == 10
