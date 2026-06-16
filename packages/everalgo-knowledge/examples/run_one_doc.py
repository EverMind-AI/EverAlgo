r"""Run KnowledgeExtractor end-to-end against a real LLM and dump the result.

Standalone CLI walkthrough — not a test. Uses the same env-var convention as
the functional test suite:

    LLM_API_KEY     OpenAI-compatible key
    LLM_BASE_URL    e.g. https://openrouter.ai/api/v1
    LLM_MODEL       e.g. anthropic/claude-sonnet-4-6

Usage::

    uv run python packages/everalgo-knowledge/examples/run_one_doc.py \
        packages/everalgo-knowledge/tests/functional/fixtures/idx_multi_topic.json

The script reads the same memsys_enterprise fixture JSON shape the functional
tests use (``title`` + ``content``); any other top-level fields are ignored.

Output destination
------------------
By default the resulting ``list[KnowledgeMemory]`` JSON is written to
``packages/everalgo-knowledge/tests/functional/outputs/<fixture_stem>.out.json``.
That directory is gitignored — every run is allowed to differ since the LLM
is non-deterministic.

Pass ``-o -`` (or ``--output -``) to stream to stdout instead, or
``-o /some/path.json`` to write somewhere else.

Pass ``--html`` to additionally emit a sibling ``.out.html`` (renderer in
``_viz.py``: vanilla HTML + CSS + minimal inline JS, open in any browser).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from everalgo.llm.protocols import LLMClient

# Allow sibling-module import (_viz) when this script is invoked directly.
# examples/ is intentionally not a Python package — it is a folder of standalone
# walkthrough scripts. Python adds the script's directory to sys.path[0] when
# running ``python path/to/script.py``, but we make it explicit so that tools
# like mypy / pyright pick up the path too.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _viz import render_html  # Local sibling — requires sys.path manipulation above.
from pydantic import SecretStr

from everalgo.knowledge import KnowledgeExtractor
from everalgo.knowledge._block_split import preprocess_content, split_content_to_blocks
from everalgo.llm import build_client
from everalgo.llm.config import LLMConfig
from everalgo.types import ParsedContent

_REQUIRED_ENV_VARS = ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL")

# Default output directory relative to this script (package-rooted, not CWD-rooted)
# so the script works the same regardless of where you launch it from.
_DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "tests" / "functional" / "outputs"


def _build_client_from_env() -> LLMClient:
    missing = [n for n in _REQUIRED_ENV_VARS if not os.environ.get(n)]
    if missing:
        sys.stderr.write(
            f"error: missing env vars: {', '.join(missing)}\n"
            f"set LLM_API_KEY / LLM_BASE_URL / LLM_MODEL to point at any "
            f"OpenAI-compatible endpoint (OpenRouter / OpenAI / vLLM / ...)\n",
        )
        sys.exit(2)
    return build_client(
        LLMConfig(
            api_key=SecretStr(os.environ["LLM_API_KEY"]),
            base_url=os.environ["LLM_BASE_URL"],
            model=os.environ["LLM_MODEL"],
        ),
    )


def _load_parsed(path: Path) -> tuple[ParsedContent, str, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    parsed = ParsedContent(text=raw.get("content", ""), mime="text/markdown")
    return parsed, path.stem, raw.get("title", "")


def _resolve_output_path(arg: str | None, fixture_path: Path) -> Path | None:
    """Map the ``-o`` argument to a concrete destination.

    Returns ``None`` for stdout. Default (``arg is None``) lands the file in
    the package's ``tests/functional/outputs/`` directory keyed on the fixture
    stem.
    """
    if arg == "-":
        return None
    if arg is None:
        return _DEFAULT_OUTPUT_DIR / f"{fixture_path.stem}.out.json"
    return Path(arg)


async def _run(fixture_path: Path, out_path: Path | None, *, emit_html: bool) -> None:
    parsed, doc_id, title = _load_parsed(fixture_path)
    sys.stderr.write(f"loaded fixture: {fixture_path} (title={title!r}, chars={len(parsed.text)})\n")
    client = _build_client_from_env()
    model = os.environ["LLM_MODEL"]
    sys.stderr.write(f"calling LLM at {os.environ['LLM_BASE_URL']} with model {model}\n")

    memories = await KnowledgeExtractor(llm=client).aextract(parsed, doc_id=doc_id, title=title)

    sys.stderr.write(f"extracted {len(memories)} KnowledgeMemory nodes\n")
    payload = [km.model_dump() for km in memories]
    rendered_json = json.dumps(payload, ensure_ascii=False, indent=2)

    if out_path is None:
        sys.stdout.write(rendered_json)
        sys.stdout.write("\n")
        if emit_html:
            sys.stderr.write("warning: --html ignored when writing to stdout (-o -)\n")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered_json + "\n", encoding="utf-8")
    sys.stderr.write(f"wrote {out_path}\n")

    if emit_html:
        html_path = (
            out_path.with_suffix(".html")
            if out_path.suffix == ".json"
            else out_path.with_suffix(
                out_path.suffix + ".html",
            )
        )
        # Re-run the deterministic block-split pass so the viz can show the
        # original atom text inside each square (extractor's KnowledgeMemory
        # only carries block_refs, not the source atoms).
        atoms = split_content_to_blocks(preprocess_content(parsed.text))
        rendered_html = render_html(memories, atoms, source_label=str(fixture_path), model=model)
        html_path.write_text(rendered_html, encoding="utf-8")
        sys.stderr.write(f"wrote {html_path}\n")


def main() -> None:
    """CLI entry point: parse args and dispatch to the async runner."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "fixture",
        type=Path,
        help="Path to a fixture JSON file with top-level `title` and `content` fields.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help=("Output destination. Default: tests/functional/outputs/<fixture-stem>.out.json. Pass `-` for stdout."),
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Also emit a sibling HTML file (vanilla HTML + CSS + minimal inline JS). Ignored with -o -.",
    )
    args = parser.parse_args()
    if not args.fixture.is_file():
        sys.stderr.write(f"error: fixture not found: {args.fixture}\n")
        sys.exit(2)
    out_path = _resolve_output_path(args.output, args.fixture)
    asyncio.run(_run(args.fixture, out_path, emit_html=args.html))


if __name__ == "__main__":
    main()
