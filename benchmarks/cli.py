"""CLI entry point for the EverAlgo benchmark suite.

T7 establishes the argparse surface. The runner wiring lands in T21.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse benchmark CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="benchmarks",
        description="EverAlgo end-to-end benchmark runner.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name (auto-discovered from benchmarks/datasets/). E.g. locomo.",
    )
    parser.add_argument(
        "--run-name",
        default="default",
        help="Run identifier. Results land in results/{dataset}-{run_name}/.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=[1, 2, 3, 4, 5],
        help="Stage numbers to run (1=extract, 2=index, 3=search, 4=answer, 5=evaluate).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke mode: 3 convs x 10 qa each. Under 5 minutes.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Override the dataset's default data path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override default output dir (results/{dataset}-{run_name}).",
    )
    return parser.parse_args(argv)


def main() -> int:
    """Entry point — load env, build pipeline request, run."""
    import asyncio

    from dotenv import load_dotenv

    from benchmarks.common.config import BenchmarkConfig
    from benchmarks.common.runner import PipelineRequest, run_pipeline

    # Load .env from benchmarks/ if it exists
    load_dotenv(Path("benchmarks/.env"))

    args = parse_args()

    # Default data paths per dataset (extend as new datasets land)
    default_data_paths = {
        "locomo": Path("benchmarks/datasets/locomo/data/locomo10.json"),
    }
    data_path = args.data_path or default_data_paths.get(args.dataset)
    if data_path is None:
        print(f"Error: no default data path for dataset {args.dataset!r}. Pass --data-path explicitly.")
        return 2
    if not data_path.exists():
        print(f"Error: data file not found at {data_path}. Run the dataset download script or pass --data-path.")
        return 2

    req = PipelineRequest(
        dataset_name=args.dataset,
        run_name=args.run_name,
        config=BenchmarkConfig(),
        stages=list(args.stages),
        smoke=args.smoke,
        data_path=data_path,
        output_dir=args.output_dir,
    )
    result = asyncio.run(run_pipeline(req))
    print(f"Done. Results: {result['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
