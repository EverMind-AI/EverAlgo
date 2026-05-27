"""CLI entry point for the EverAlgo benchmark suite.

T7 establishes the argparse surface. The runner wiring lands in T21.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING, override

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
        help="Smoke mode: 1 conversation x 10 QA (~10 questions). Quick end-to-end sanity check, not a scored run.",
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


class _NoTracebackFormatter(logging.Formatter):
    """Stderr-only formatter: render the message line and drop ``exc_info`` traceback.

    A 152-frame retry storm with full tracebacks is unreadable on stderr; we still
    want the tracebacks for post-mortem in ``run.log`` (see file handler below).
    Override ``format`` entirely instead of returning ``""`` from
    ``formatException``, because ``logging.Formatter`` caches the result on
    ``record.exc_text`` and would leak the empty cache into other handlers that
    share the same ``LogRecord``.
    """

    @override
    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()
        if self.usesTime():
            record.asctime = self.formatTime(record, self.datefmt)
        return self.formatMessage(record)


def _setup_logging(output_dir: Path) -> None:
    """Configure root logger to emit WARNING+ to stderr (message-only) and to ``run.log`` (full traceback).

    Captures all relevant scenarios:
    - LLM JSON parse retry failures  (everalgo.llm.parse)
    - Clustering skip / missing embeddings  (benchmarks.common.stages.extract)
    - Cluster index build failures  (benchmarks.common.stages.index)
    - Provider-layer HTTP errors  (everalgo.llm.providers.*)
    - Any other WARNING/ERROR in the benchmarks or everalgo namespaces
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fmt_text = "%(asctime)s %(levelname)-8s %(name)s  %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    stderr_handler = logging.StreamHandler()
    stderr_handler.setFormatter(_NoTracebackFormatter(fmt=fmt_text, datefmt=datefmt))

    log_file = output_dir / "run.log"
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(fmt=fmt_text, datefmt=datefmt))

    root = logging.getLogger()
    root.setLevel(logging.WARNING)
    root.addHandler(stderr_handler)
    root.addHandler(file_handler)


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

    output_dir = args.output_dir or (Path("benchmarks/results") / f"{args.dataset}-{args.run_name}")
    _setup_logging(output_dir)

    req = PipelineRequest(
        dataset_name=args.dataset,
        run_name=args.run_name,
        config=BenchmarkConfig(),
        stages=list(args.stages),
        smoke=args.smoke,
        data_path=data_path,
        output_dir=output_dir,
    )
    result = asyncio.run(run_pipeline(req))
    print(f"Done. Results: {result['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
