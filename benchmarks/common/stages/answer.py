"""Stage 4 — Answer generation.

For each retrieved question: build context from top-K event_ids (matching
Stage 1's serialized memcells), call the LLM with the dataset's answer_prompt,
extract the FINAL ANSWER section. Output: answers.json.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING, Any

from benchmarks.common.stages.types import StageStats
from everalgo.llm.parse import extract_final_answer as _algo_extract_final_answer
from everalgo.llm.types import ChatMessage

if TYPE_CHECKING:
    from pathlib import Path

    from benchmarks.common.stages.types import StageContext


logger = logging.getLogger(__name__)


_CONTEXT_TEMPLATE = """Episodes memories for conversation between {speaker_a} and {speaker_b}:

    {episodes}
"""

_ANSWER_RETRIES = 5


def _extract_final_answer(raw: str) -> str:
    """LoCoMo-specific marker fallback chain (3 markers in priority order).

    Delegates per-marker extraction to ``everalgo.llm.parse.extract_final_answer`` which uses
    ``rsplit`` to take the LAST occurrence (handles marker appearing in reasoning prose before the
    actual answer). Supported markers in priority order:
      1. ``## STEP 7: FINAL ANSWER`` (prompt STEP 7 section header)
      2. ``FINAL ANSWER:`` (colon-suffixed)
      3. ``FINAL ANSWER`` (bare — leading colon stripped if present)
    """
    result = raw.strip()
    for marker in ("## STEP 7: FINAL ANSWER", "FINAL ANSWER:", "FINAL ANSWER"):
        if marker in result:
            extracted = _algo_extract_final_answer(result, marker=marker)
            # Bare "FINAL ANSWER" may have a leading ":" — strip it
            if marker == "FINAL ANSWER" and extracted.startswith(":"):
                extracted = extracted[1:].strip()
            return extracted
    return result


def _build_context(
    selected_memcells: list[dict[str, Any]],
    speakers: tuple[str, str] | None = None,
) -> str:
    r"""Build the context string from pre-selected memcells.

    Each memcell renders as ``{subject}: {content}\n---`` and entries are joined by ``\n\n``. The
    ``\n---`` suffix plus double-newline separator give the LLM clear per-memory block boundaries for
    STEP 1 (RELEVANT MEMORIES EXTRACTION) in the answer prompt. ``subject``/``content`` are read from
    the nested EverAlgo schema; the raw-ms ``timestamp`` field is intentionally omitted (LLMs cannot
    parse ms epochs and any temporal cues are already inside ``content`` in natural language).
    """
    speaker_a, speaker_b = speakers if speakers else ("A", "B")
    episode_lines = [
        f"{mc.get('episode', {}).get('subject', 'N/A')}: {mc.get('episode', {}).get('content', 'N/A')}\n---"
        for mc in selected_memcells
    ]
    return _CONTEXT_TEMPLATE.format(
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        episodes="\n\n".join(episode_lines),
    )


def _load_memcell_map(stage1_dir: Path, ctx: StageContext) -> dict[str, dict[str, dict[str, Any]]]:
    """Load all memcells_conv_*.json files into a session-scoped map.

    Returns:
        ``{conv_id: {mc_id: mc_dict}}`` keyed by dataset conv_id and
        session-local memcell ``id`` strings (e.g. ``"0"``, ``"1"``).
    """
    map_per_session: dict[str, dict[str, dict[str, Any]]] = {}
    sessions = list(ctx.dataset.load_conversations())
    for idx, conv in enumerate(sessions):
        path = stage1_dir / f"memcells_conv_{idx}.json"
        if not path.exists():
            continue
        try:
            memcells: list[dict[str, Any]] = json.loads(path.read_text())
            map_per_session[conv.id] = {mc["id"]: mc for mc in memcells if "id" in mc}
        except (json.JSONDecodeError, OSError, KeyError):
            continue  # tolerate corrupted/empty files
    return map_per_session


async def _answer_one_question(
    item: dict[str, Any],
    memcells_map: dict[str, dict[str, dict[str, Any]]],
    speakers_lookup: dict[str, tuple[str, str]],
    ctx: StageContext,
) -> dict[str, Any]:
    """Generate an answer for a single question item."""
    qa = item["original_qa"]
    conv_id = qa.get("conv_id", item.get("conversation_id", ""))
    speakers = speakers_lookup.get(conv_id)
    session_memcells = memcells_map.get(conv_id, {})
    mc_ids = item.get("members", [])
    selected = [session_memcells[mc_id] for mc_id in mc_ids[: ctx.config.response_top_k] if mc_id in session_memcells]
    context = _build_context(selected, speakers)
    prompt = ctx.dataset.answer_prompt().format(context=context, question=qa["question"])
    # try/except inside the retry loop so both empty completions (OpenRouter timeout / streaming
    # truncation) and exceptions (network blips, transient 429 / 503) trigger another attempt.
    # Fail-loud on exhaustion.
    answer = ""
    response = None
    for attempt in range(_ANSWER_RETRIES):
        try:
            response = await ctx.services.llm.chat(
                [ChatMessage(role="user", content=prompt)],
                temperature=0.0,
                max_tokens=32768,
            )
            answer = _extract_final_answer(response.content)
            if answer:
                break
        except Exception:
            if attempt == _ANSWER_RETRIES - 1:
                raise
            logger.warning(
                "answer attempt %d/%d failed for question_id=%s; retrying",
                attempt + 1,
                _ANSWER_RETRIES,
                qa["question_id"],
                exc_info=True,
            )
            await asyncio.sleep(1.0 * (2**attempt))
            continue
        # Empty-answer path: retry with backoff.
        if attempt < _ANSWER_RETRIES - 1:
            await asyncio.sleep(1.0 * (2**attempt))
    if not answer or response is None:
        raise RuntimeError(f"answer empty after {_ANSWER_RETRIES} retries (question_id={qa['question_id']})")
    pt = (response.usage.prompt_tokens or 0) if response.usage is not None else 0
    ct = (response.usage.completion_tokens or 0) if response.usage is not None else 0
    return {
        "question_id": qa["question_id"],
        "question": qa["question"],
        "answer": answer,
        "golden_answer": qa["golden_answer"],
        "category": qa["category"],
        "conversation_id": conv_id,
        "formatted_context": context,
        "raw_response": response.content,
        "prompt_tokens": pt,
        "completion_tokens": ct,
    }


def _flatten_search_results(
    search_results: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Flatten per-conversation search results into a single list."""
    items: list[dict[str, Any]] = []
    for conv_id, conv_items in search_results.items():
        for item in conv_items:
            item.setdefault("conversation_id", conv_id)
            items.append(item)
    return items


def _build_speakers_lookup(ctx: StageContext) -> dict[str, tuple[str, str]]:
    """Build conv_id → (speaker_a, speaker_b) lookup from the dataset."""
    lookup: dict[str, tuple[str, str]] = {}
    for conv in ctx.dataset.load_conversations():
        if len(conv.speakers) >= 2:
            lookup[conv.id] = (conv.speakers[0], conv.speakers[1])
    return lookup


async def run_answer_stage(ctx: StageContext) -> StageStats:
    """Stage 4 — generate an answer for every retrieved question."""
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    stats = StageStats(stage_name="answer")
    started = time.monotonic()

    search_results: dict[str, list[dict[str, Any]]] = json.loads((ctx.input_dir / "search_results.json").read_text())

    stage1_dir = ctx.input_dir.parent / "stage1_extract"
    memcells_map = _load_memcell_map(stage1_dir, ctx)
    speakers_lookup = _build_speakers_lookup(ctx)

    all_items = _flatten_search_results(search_results)
    if ctx.smoke:
        all_items = all_items[: ctx.smoke_conv_limit * ctx.smoke_qa_limit]

    sem = asyncio.Semaphore(ctx.config.max_concurrent_qa)

    async def _process(item: dict[str, Any]) -> dict[str, Any]:
        async with sem:
            result = await _answer_one_question(item, memcells_map, speakers_lookup, ctx)
            stats.prompt_tokens += result["prompt_tokens"]
            stats.completion_tokens += result["completion_tokens"]
            return result

    from benchmarks.common._progress import gather_with_progress

    results = await gather_with_progress(
        *(_process(item) for item in all_items),
        desc="answer",
        unit="q",
    )

    # Fail-loud: any per-question exception aborts the stage, so every item in
    # ``results`` is a successful answer.
    stats.success = len(results)
    stats.failed = 0
    stats.duration_seconds = time.monotonic() - started

    (ctx.output_dir / "answers.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    return stats
