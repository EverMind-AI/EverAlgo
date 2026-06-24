"""Tests for the benchmark CLI argument parser."""

import pytest

from benchmarks.cli import parse_args


def test_required_dataset_argument():
    """Without --dataset, argparse must exit."""
    with pytest.raises(SystemExit):
        parse_args([])


def test_minimal_invocation():
    """--dataset alone with all defaults."""
    args = parse_args(["--dataset", "locomo"])
    assert args.dataset == "locomo"
    assert args.smoke is False
    assert args.run_name == "default"
    assert args.stages == [1, 2, 3, 4, 5, 6, 7]


def test_all_options_parsed():
    """Verify CLI exposes every option from the plan."""
    args = parse_args(
        [
            "--dataset",
            "locomo",
            "--run-name",
            "v1",
            "--smoke",
            "--stages",
            "3",
            "4",
            "5",
        ]
    )
    assert args.run_name == "v1"
    assert args.smoke is True
    assert args.stages == [3, 4, 5]
