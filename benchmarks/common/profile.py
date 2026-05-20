"""Reproducibility metadata — package versions + config dump."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from benchmarks.common.config import BenchmarkConfig


_EVERALGO_PACKAGES = (
    "everalgo-core",
    "everalgo-boundary",
    "everalgo-user-memory",
    "everalgo-rank",
)


def collect_package_versions() -> dict[str, str]:
    """Return installed version of every everalgo package exercised by the benchmark.

    Returns ``"unknown"`` for any package not present in the active env.
    """
    versions: dict[str, str] = {}
    for pkg in _EVERALGO_PACKAGES:
        try:
            versions[pkg] = pkg_version(pkg)
        except PackageNotFoundError:
            versions[pkg] = "unknown"
    return versions


def collect_config_dump(cfg: BenchmarkConfig) -> dict[str, object]:
    """Serialize the config to a dict for the report."""
    return cfg.model_dump()
