"""Tests for reproducibility metadata collection."""

from benchmarks.common.config import BenchmarkConfig
from benchmarks.common.profile import collect_config_dump, collect_package_versions


def test_collect_package_versions_includes_everalgo_packages():
    versions = collect_package_versions()
    # All 4 everalgo packages exercised by the LoCoMo benchmark
    for pkg in ("everalgo-core", "everalgo-boundary", "everalgo-user-memory", "everalgo-rank"):
        assert pkg in versions
        # Either a real version or "unknown" (if not installed)
        assert isinstance(versions[pkg], str)
        assert versions[pkg]  # non-empty


def test_collect_config_dump_returns_dict():
    cfg = BenchmarkConfig()
    dump = collect_config_dump(cfg)
    # Spot-check key fields — llm_model was split into extract_model + answer_model
    assert dump["extract_model"] == "openai/gpt-4.1-mini"
    assert dump["answer_model"] == "openai/gpt-4.1-mini"
    assert dump["judge_runs"] == 3
    assert dump["retrieval_mode"] == "agentic"


def test_collect_config_dump_handles_override():
    cfg = BenchmarkConfig(llm_temperature=0.5)
    dump = collect_config_dump(cfg)
    assert dump["llm_temperature"] == 0.5
