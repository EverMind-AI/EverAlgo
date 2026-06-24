"""Tests for the pipeline runner."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from pytest import MonkeyPatch

from benchmarks.common.config import BenchmarkConfig
from benchmarks.common.runner import PipelineRequest, run_pipeline


def test_pipeline_request_construction(tmp_path: Path) -> None:
    """Smoke test: PipelineRequest with all fields."""
    req = PipelineRequest(
        dataset_name="locomo",
        run_name="test",
        config=BenchmarkConfig(),
        stages=[1, 2, 3, 4, 5, 6, 7],
        smoke=True,
        data_path=tmp_path / "data.json",
        output_dir=tmp_path / "results",
    )
    assert req.stages == [1, 2, 3, 4, 5, 6, 7]
    assert req.smoke is True


def test_pipeline_request_is_frozen() -> None:
    """PipelineRequest should be immutable."""
    req = PipelineRequest(
        dataset_name="locomo",
        run_name="x",
        config=BenchmarkConfig(),
        stages=[1],
    )
    with pytest.raises((TypeError, AttributeError)):  # FrozenInstanceError from dataclass
        req.run_name = "y"  # type: ignore[misc]


def test_run_pipeline_is_async() -> None:
    assert inspect.iscoroutinefunction(run_pipeline)


@pytest.mark.asyncio
async def test_run_pipeline_rejects_unknown_dataset(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Unknown dataset name -> ValueError before any stage runs."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test")
    req = PipelineRequest(
        dataset_name="nonexistent",
        run_name="test",
        config=BenchmarkConfig(),
        stages=[1],
        data_path=tmp_path / "no_such_file.json",
        output_dir=tmp_path / "results",
    )
    with pytest.raises(ValueError, match="Unknown dataset"):
        await run_pipeline(req)


@pytest.mark.asyncio
async def test_run_pipeline_rejects_missing_data_path(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Required data_path must be provided."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test")
    req = PipelineRequest(
        dataset_name="locomo",
        run_name="test",
        config=BenchmarkConfig(),
        stages=[1],
        data_path=None,
        output_dir=tmp_path / "results",
    )
    with pytest.raises(ValueError, match="data_path"):
        await run_pipeline(req)


@pytest.mark.asyncio
async def test_run_pipeline_subset_stages(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Run only stage 5 (evaluate): expect to fail loading because no stage 4 output exists.

    This verifies the runner respects the ``stages`` subset -- it should NOT try
    to run stages 1-4 just because they are listed in _STAGE_RUNNERS.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test")

    # Provide a tiny fixture file at the path
    fixture = Path(__file__).parent / "fixtures" / "locomo_mini.json"
    output_root = tmp_path / "results"
    output_root.mkdir()

    req = PipelineRequest(
        dataset_name="locomo",
        run_name="test",
        config=BenchmarkConfig(),
        stages=[7],
        data_path=fixture,
        output_dir=output_root,
    )
    # Stage 7 reads answers.json from the previous stage output (which doesn't exist).
    # This raises FileNotFoundError, proving stages 1-6 are NOT being run.
    with pytest.raises(FileNotFoundError):
        await run_pipeline(req)
