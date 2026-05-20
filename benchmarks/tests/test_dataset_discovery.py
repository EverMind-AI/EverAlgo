"""Tests for dataset auto-discovery."""

from benchmarks.datasets import discover_datasets


def test_discover_locomo():
    registry = discover_datasets()
    assert "locomo" in registry
    locomo_factory = registry["locomo"]
    assert callable(locomo_factory)


def test_unknown_dataset_not_in_registry():
    registry = discover_datasets()
    assert "nonexistent" not in registry
