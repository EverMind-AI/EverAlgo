"""Stage 4 — Answer generation.

For each retrieved question: build context from top-K event_ids (matching
Stage 1's serialized memcells), call the LLM with the dataset's answer_prompt,
extract the FINAL ANSWER section. Output: answers.json.
"""

from __future__ import annotations

import asyncio
import json
import time
import traceback
from typing import TYPE_CHECKING, Any

from benchmarks.common.stages.types import StageStats

if TYPE_CHECKING:
    from pathlib import Path

    from benchmarks.common.stages.types import StageContext


_CONTEXT_TEMPLATE = "Episodes memories for conversation between {speaker_a} and {speaker_b}:\n\n{episodes}"

# Mirror locomo-benchmark stage4_response.py: max_retries=5.
_ANSWER_RETRIES = 5


def _extract_final_answer(raw: str) -> str:
    """Extract the final answer from an LLM response using rsplit on the last marker.

    Mirrors locomo-benchmark ``stage4_response.py:154-170``: uses ``rsplit`` to
    take the LAST occurrence of each marker (handles cases where "FINAL ANSWER"
    appears in reasoning text before the actual answer section). Supports three
    marker formats in priority order:
      1. ``## STEP 7: FINAL ANSWER`` (prompt STEP 7 section header)
      2. ``FINAL ANSWER:`` (standard colon-suffixed format)
      3. ``FINAL ANSWER`` (bare, colon stripped if present)
    No truncation at ``##`` headings or blank lines — the new prompt's STEP 7
    IS the final answer (single section), so trailing markdown does not appear.
    """
    result = raw.strip()
    if "## STEP 7: FINAL ANSWER" in result:
        result = result.rsplit("## STEP 7: FINAL ANSWER", 1)[1].strip()
    elif "FINAL ANSWER:" in result:
        result = result.rsplit("FINAL ANSWER:", 1)[1].strip()
    elif "FINAL ANSWER" in result:
        candidate = result.rsplit("FINAL ANSWER", 1)[1].strip()
        if candidate.startswith(":"):
            candidate = candidate[1:].strip()
        result = candidate
    return result


def _build_context(
    selected_memcells: list[dict[str, Any]],
    speakers: tuple[str, str] | None = None,
) -> str:
    r"""Build the context string from pre-selected memcells.

    Mirrors EverCore main ``stage4_response.py:94-100``: each memcell renders as
    ``{subject}: {content}\n---`` and entries are joined by ``\n\n``. The
    ``\n---`` suffix plus double-newline separator give the LLM clear per-memory
    block boundaries for STEP 1 (RELEVANT MEMORIES EXTRACTION) in the answer
    prompt. ``subject``/``content`` are read from the nested EverAlgo schema; the
    raw-ms ``timestamp`` field is intentionally omitted (LLMs cannot parse ms
    epochs and any temporal cues are already inside ``content`` in natural
    language).
    """
    speaker_a, speaker_b = speakers if speakers else ("A", "B")
    episode_lines = [
        f"{mc.get('episode', {}).get('subject', '')}: {mc.get('episode', {}).get('content', '')}\n---"
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
    mc_ids = item.get("memcell_ids", [])
    selected = [session_memcells[mc_id] for mc_id in mc_ids[: ctx.config.response_top_k] if mc_id in session_memcells]
    context = _build_context(selected, speakers)
    prompt = ctx.dataset.answer_prompt().format(context=context, question=qa["question"])
    # Mirror EverCore: answer stage uses temperature=0 (``stage4_response.py:131``).
    # Without this, ctx.services.llm.chat would fall back to BenchmarkConfig.llm_temperature=0.3
    # and introduce non-determinism that hurts factual QA accuracy.
    # Retry on empty answer mirrors EverCore ``stage4_response.py:129-147``: the LLM
    # occasionally returns an empty completion (OpenRouter timeout / streaming truncation)
    # and a retry usually recovers it.
    answer = ""
    response = None
    for _attempt in range(_ANSWER_RETRIES):
        response = await ctx.services.llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=32768,  # Mirror locomo-benchmark stage4_response.py:142
        )
        answer = _extract_final_answer(response.content)
        if answer:
            break
    assert response is not None  # loop runs ≥1 iteration
    return {
        "question_id": qa["question_id"],
        "question": qa["question"],
        "answer": answer,
        "golden_answer": qa["golden_answer"],
        "category": qa["category"],
        "conversation_id": conv_id,
        "formatted_context": context,
        "raw_response": response.content,
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
    }


def _make_error_item(item: dict[str, Any]) -> dict[str, Any]:
    """Build a placeholder output item for a failed question."""
    qa = item.get("original_qa", {})
    return {
        "question_id": qa.get("question_id", ""),
        "question": qa.get("question", ""),
        "answer": "Error: failed to generate answer",
        "golden_answer": qa.get("golden_answer", ""),
        "category": qa.get("category", ""),
        "conversation_id": item.get("conversation_id", ""),
        "formatted_context": "",
        "raw_response": "",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "error": True,
        "traceback": traceback.format_exc(),
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
            try:
                result = await _answer_one_question(item, memcells_map, speakers_lookup, ctx)
            except Exception:  # broad catch is intentional — isolate per-question failures
                return _make_error_item(item)
            else:
                stats.prompt_tokens += result["prompt_tokens"]
                stats.completion_tokens += result["completion_tokens"]
                return result

    from tqdm.asyncio import tqdm as async_tqdm  # type: ignore[import-untyped]

    results: list[dict[str, Any]] = await async_tqdm.gather(  # type: ignore[attr-defined]
        *(_process(item) for item in all_items),
        desc="answer",
        unit="q",
        dynamic_ncols=True,
    )

    stats.success = sum(1 for r in results if not r.get("error"))
    stats.failed = sum(1 for r in results if r.get("error"))
    stats.duration_seconds = time.monotonic() - started

    (ctx.output_dir / "answers.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    return stats
