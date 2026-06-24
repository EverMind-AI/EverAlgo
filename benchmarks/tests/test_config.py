"""Tests for BenchmarkConfig TOML loading and field changes."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

import pytest

from benchmarks.common.config import BenchmarkConfig

if TYPE_CHECKING:
    from pathlib import Path


class TestConfigDefaults:
    def test_extract_model_default(self):
        cfg = BenchmarkConfig()
        assert cfg.extract_model == "openai/gpt-4.1-mini"

    def test_answer_model_default(self):
        cfg = BenchmarkConfig()
        assert cfg.answer_model == "openai/gpt-4.1-mini"

    def test_enable_reflection_default_false(self):
        cfg = BenchmarkConfig()
        assert cfg.enable_reflection is False

    def test_session_filter_default_none(self):
        cfg = BenchmarkConfig()
        assert cfg.session_filter is None

    def test_no_enable_clustering_field(self):
        """enable_clustering removed — clustering is always on."""
        cfg = BenchmarkConfig()
        assert not hasattr(cfg, "enable_clustering")

    def test_no_llm_model_field(self):
        """llm_model split into extract_model + answer_model."""
        cfg = BenchmarkConfig()
        assert not hasattr(cfg, "llm_model")


class TestFromToml:
    def test_load_overrides(self, tmp_path: Path):
        toml_content = textwrap.dedent("""\
            extract_model = "openai/gpt-4.1"
            answer_model = "openai/gpt-4o"
            enable_reflection = true

            [session_filter]
            "5" = [1, 2, 12]
            "2" = [3, 8]
        """)
        cfg_dir = tmp_path / "benchmarks" / "configs"
        cfg_dir.mkdir(parents=True)
        (cfg_dir / "test.toml").write_text(toml_content)

        cfg = BenchmarkConfig.from_toml("test", config_dir=cfg_dir)
        assert cfg.extract_model == "openai/gpt-4.1"
        assert cfg.answer_model == "openai/gpt-4o"
        assert cfg.enable_reflection is True
        assert cfg.session_filter == {5: [1, 2, 12], 2: [3, 8]}

    def test_load_defaults_preserved(self, tmp_path: Path):
        toml_content = "enable_reflection = true\n"
        cfg_dir = tmp_path / "benchmarks" / "configs"
        cfg_dir.mkdir(parents=True)
        (cfg_dir / "minimal.toml").write_text(toml_content)

        cfg = BenchmarkConfig.from_toml("minimal", config_dir=cfg_dir)
        assert cfg.enable_reflection is True
        assert cfg.extract_model == "openai/gpt-4.1-mini"  # default preserved

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            BenchmarkConfig.from_toml("nonexistent", config_dir=tmp_path)

    def test_session_filter_keys_coerced_to_int(self, tmp_path: Path):
        toml_content = '[session_filter]\n"5" = [1, 2]\n'
        cfg_dir = tmp_path / "benchmarks" / "configs"
        cfg_dir.mkdir(parents=True)
        (cfg_dir / "sf.toml").write_text(toml_content)

        cfg = BenchmarkConfig.from_toml("sf", config_dir=cfg_dir)
        assert cfg.session_filter is not None
        assert isinstance(next(iter(cfg.session_filter.keys())), int)
