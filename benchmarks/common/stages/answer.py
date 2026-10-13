"""Stage 6 — Answer generation.

For each retrieved question: build context from top-K event_ids (matching
Stage 3's enriched episodes), call the LLM with the dataset's answer_prompt,
extract the FINAL ANSWER section. Output: answers.json.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING, Any

from benchmarks.common.metrics import estimate_chat_tokens
from benchmarks.common.stages.types import StageStats
from everalgo.llm import LLMMessage, StrictOutputModel, parse_reply
from everalgo.utils import render_prompt

if TYPE_CHECKING:
    from pathlib import Path

    from benchmarks.common.stages.types import StageContext


logger = logging.getLogger(__name__)


_CONTEXT_TEMPLATE = """Episodes memories for conversation between {{speaker_a}} and {{speaker_b}}:

    {{episodes}}
"""


class _AnswerReply(StrictOutputModel):
    """The answer model's reply; the contract has no free-text tier, so the free-form reasoning rides in ``text``."""

    text: str


def _extract_final_answer(raw: str) -> str:
    """LoCoMo-specific marker fallback chain (3 markers in priority order).

    Takes the text after the LAST occurrence of the first marker found, since a marker may also appear in
    reasoning prose before the actual answer. Supported markers in priority order:
      1. ``## STEP 7: FINAL ANSWER`` (prompt STEP 7 section header)
      2. ``FINAL ANSWER:`` (colon-suffixed)
      3. ``FINAL ANSWER`` (bare — leading colon stripped if present)
    """
    result = raw.strip()
    for marker in ("## STEP 7: FINAL ANSWER", "FINAL ANSWER:", "FINAL ANSWER"):
        if marker in result:
            extracted = result.rsplit(marker, 1)[1].strip()
            # Bare "FINAL ANSWER" may have a leading ":" — strip it
            if marker == "FINAL ANSWER" and extracted.startswith(":"):
                extracted = extracted[1:].strip()
            return extracted
    return result


def _build_context(
    selected_episodes: list[dict[str, Any]],
    speakers: tuple[str, str] | None = None,
) -> str:
    r"""Build the context string from pre-selected episodes.

    Each episode renders as ``{subject}: {episode_text}\n---`` and entries are joined by ``\n\n``. The
    ``\n---`` suffix plus double-newline separator give the LLM clear per-memory block boundaries for
    STEP 1 (RELEVANT MEMORIES EXTRACTION) in the answer prompt. ``subject`` and ``episode`` are flat
    fields on the episode dict (entity-split model). The raw-ms ``timestamp`` field is intentionally
    omitted (LLMs cannot parse ms epochs and any temporal cues are already inside the episode text).
    """
    speaker_a, speaker_b = speakers if speakers else ("A", "B")
    episode_lines = [f"{ep.get('subject', 'N/A')}: {ep.get('episode', 'N/A')}\n---" for ep in selected_episodes]
    return render_prompt(
        _CONTEXT_TEMPLATE,
        None,
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        episodes="\n\n".join(episode_lines),
    )


def _load_episode_map(stage1_dir: Path, ctx: StageContext) -> dict[str, dict[str, dict[str, Any]]]:
    """Load all episodes_conv_*.json files into a session-scoped map.

    Returns:
        ``{conv_id: {episode_id: episode_dict}}`` keyed by dataset conv_id and
        episode ``id`` strings (e.g. ``"0"``, ``"1"``).
    """
    map_per_session: dict[str, dict[str, dict[str, Any]]] = {}
    sessions = list(ctx.dataset.load_conversations())
    for idx, conv in enumerate(sessions):
        path = stage1_dir / f"episodes_conv_{idx}.json"
        if not path.exists():
            continue
        try:
            episodes: list[dict[str, Any]] = json.loads(path.read_text())
            map_per_session[conv.id] = {ep["id"]: ep for ep in episodes if "id" in ep}
        except (json.JSONDecodeError, OSError, KeyError):
            continue  # tolerate corrupted/empty files
    return map_per_session


async def _retry_llm_answer(
    messages: list[LLMMessage],
    ctx: StageContext,
    question_id: str,
) -> tuple[str, str]:
    """Call LLM with retry, extracting the final answer. Fail-loud on exhaustion.

    Returns:
        Tuple of ``(answer_text, raw_text)``, ``raw_text`` being the reply's ``text`` before marker extraction.

    Raises:
        LLMOutputError: If the last attempt's reply is not a valid ``_AnswerReply``.
        RuntimeError: If every attempt yields an empty final answer.
    """
    for attempt in range(ctx.config.llm_max_retries):
        try:
            reply = await ctx.services.llm.achat(
                messages,
                response_format=_AnswerReply,
                temperature=0.0,
                max_tokens=32768,
            )
            raw = parse_reply(_AnswerReply, reply, operator="answer").text
            answer = _extract_final_answer(raw)
            if answer:
                return answer, raw
        except Exception:
            if attempt == ctx.config.llm_max_retries - 1:
                raise
            logger.warning(
                "answer attempt %d/%d failed for question_id=%s; retrying",
                attempt + 1,
                ctx.config.llm_max_retries,
                question_id,
                exc_info=True,
            )
            await asyncio.sleep(1.0 * (2**attempt))
            continue
        if attempt < ctx.config.llm_max_retries - 1:
            await asyncio.sleep(1.0 * (2**attempt))
    raise RuntimeError(f"answer empty after {ctx.config.llm_max_retries} retries (question_id={question_id})")


async def _answer_one_question(
    item: dict[str, Any],
    episodes_map: dict[str, dict[str, dict[str, Any]]],
    speakers_lookup: dict[str, tuple[str, str]],
    ctx: StageContext,
) -> dict[str, Any]:
    """Generate an answer for a single question item."""
    qa = item["original_qa"]
    conv_id = qa.get("conv_id", item.get("conversation_id", ""))
    speakers = speakers_lookup.get(conv_id)
    session_episodes = episodes_map.get(conv_id, {})
    ep_ids = item.get("members", [])
    selected = [session_episodes[ep_id] for ep_id in ep_ids[: ctx.config.response_top_k] if ep_id in session_episodes]
    context = _build_context(selected, speakers)
    prompt = render_prompt(ctx.dataset.answer_prompt(), None, context=context, question=qa["question"])

    messages = [LLMMessage(role="user", content=prompt)]
    answer, raw = await _retry_llm_answer(messages, ctx, qa["question_id"])
    prompt_tokens, completion_tokens = estimate_chat_tokens(messages, raw)
    return {
        "question_id": qa["question_id"],
        "question": qa["question"],
        "answer": answer,
        "golden_answer": qa["golden_answer"],
        "category": qa["category"],
        "conversation_id": conv_id,
        "formatted_context": context,
        "raw_response": raw,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
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
    """Stage 6 — generate an answer for every retrieved question."""
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    stats = StageStats(stage_name="answer")
    started = time.monotonic()

    search_results: dict[str, list[dict[str, Any]]] = json.loads((ctx.input_dir / "search_results.json").read_text())

    # Read episodes from stage3_enrich (which always carries the latest episodes —
    # either original from stage1 or reflected from stage2, passed through by enrich).
    enrich_dir = ctx.input_dir.parent / "stage3_enrich"
    episodes_map = _load_episode_map(enrich_dir, ctx)
    speakers_lookup = _build_speakers_lookup(ctx)

    all_items = _flatten_search_results(search_results)
    if ctx.smoke:
        all_items = all_items[: ctx.smoke_conv_limit * ctx.smoke_qa_limit]

    sem = asyncio.Semaphore(ctx.config.max_concurrent_qa)

    async def _process(item: dict[str, Any]) -> dict[str, Any]:
        async with sem:
            result = await _answer_one_question(item, episodes_map, speakers_lookup, ctx)
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
