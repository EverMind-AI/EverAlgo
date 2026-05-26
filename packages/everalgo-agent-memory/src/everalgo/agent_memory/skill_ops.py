"""Operation atoms for :class:`AgentSkillExtractor` — add, update, maturity scoring, and formatting helpers."""

from __future__ import annotations

import json
import logging
import re
import uuid
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any

from everalgo.agent_memory._text import json_default, truncate_text
from everalgo.agent_memory.prompts.skill_maturity import AGENT_SKILL_MATURITY_SCORE_PROMPT
from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.prompts import render_prompt
from everalgo.types import AgentCase, AgentSkill

if TYPE_CHECKING:
    from collections.abc import Sequence

    from everalgo.agent_memory.skill import _SkillCfg
    from everalgo.llm.protocols import LLMClient

logger = logging.getLogger(__name__)


__all__ = [
    "_apply_add",
    "_apply_update",
    "_content_change_ratio",
    "_evaluate_maturity",
    "_format_cases",
    "_format_existing_skills",
    "_is_hypothesis_promotion",
    "_is_skill_content_sufficient",
]


def _op_data(op: dict[str, Any]) -> dict[str, Any]:
    """Pull the ``data`` field off an LLM-emitted op dict, defaulting to ``{}``."""
    raw = op.get("data")
    if isinstance(raw, dict):
        return raw  # type: ignore[return-value]
    return {}


# ── Formatting helpers ──────────────────────────────────────────────────────────────────────────────


def _format_cases(case_records: Sequence[AgentCase]) -> str:
    """Serialize AgentCase instances to a JSON string for the LLM prompt."""
    formatted: list[dict[str, Any]] = []
    for rec in case_records:
        entry: dict[str, Any] = {
            "timestamp": rec.timestamp,
            "task_intent": rec.task_intent or "",
            "approach": rec.approach or "",
            "quality_score": rec.quality_score,
        }
        if rec.key_insight:
            entry["key_insight"] = rec.key_insight
        formatted.append(entry)
    return json.dumps(formatted, ensure_ascii=False, indent=2, default=json_default)


def _summarise_case_for_prompt(case_record: AgentCase, max_approach_tokens: int = 200) -> dict[str, Any]:
    """Compact case summary dict for inclusion in the skill prompt."""
    entry: dict[str, Any] = {
        "task_intent": case_record.task_intent or "",
        "quality_score": case_record.quality_score,
    }
    if case_record.key_insight:
        entry["key_insight"] = case_record.key_insight
    if case_record.approach:
        entry["approach"] = truncate_text(case_record.approach, max_approach_tokens, suffix="... [omitted]")
    return entry


def _format_existing_skills(
    existing_records: Sequence[AgentSkill],
    supporting_cases: Sequence[AgentCase],
    *,
    max_description_tokens: int,
    max_content_tokens: int,
    max_support_cases: int = 3,
    max_approach_tokens: int = 200,
) -> str:
    """Format existing skills with index numbers and supporting case summaries for the LLM prompt."""
    if not existing_records:
        return "(empty — no existing skills)"

    case_map: dict[str, AgentCase] = {}
    for case_rec in supporting_cases:
        cid = str(case_rec.id or "")
        if cid:
            case_map[cid] = case_rec

    lines: list[str] = []
    for idx, skill_rec in enumerate(existing_records):
        item: dict[str, Any] = {
            "index": idx,
            "name": skill_rec.name or "",
            "description": truncate_text(skill_rec.description or "", max_description_tokens, suffix="... [omitted]"),
            "content": truncate_text(skill_rec.content or "", max_content_tokens, suffix="... [omitted]"),
            "confidence": skill_rec.confidence,
        }

        if case_map:
            source_ids = list(skill_rec.source_case_ids or [])
            matched_ids = [sid for sid in source_ids if str(sid) in case_map]
            if matched_ids:
                recent_ids = matched_ids[-max_support_cases:]
                item["supporting_case_count"] = len(matched_ids)
                item["supporting_cases"] = [
                    _summarise_case_for_prompt(case_map[str(sid)], max_approach_tokens=max_approach_tokens)
                    for sid in recent_ids
                ]

        lines.append(json.dumps(item, ensure_ascii=False, default=json_default))
    return "[\n" + ",\n".join(lines) + "\n]"


# ── Content validation helpers ──────────────────────────────────────────────────────────────────────


def _is_skill_content_sufficient(content: str, min_lines: int = 5, min_length: int = 50) -> bool:
    """Return ``True`` iff ``content`` meets the minimum line count and character length."""
    if not content:
        return False
    stripped = content.strip()
    if len(stripped) < min_length:
        return False
    non_empty_lines = [line for line in stripped.splitlines() if line.strip()]
    return len(non_empty_lines) >= min_lines


def _content_change_ratio(old: str, new: str) -> float:
    """Return ``1 - SequenceMatcher.ratio()`` as a 0.0-1.0 change fraction."""
    if not old and not new:
        return 0.0
    if not old or not new:
        return 1.0
    ratio = SequenceMatcher(None, old, new).ratio()
    return round(1.0 - ratio, 4)


_HYPOTHESIS_HEADER = re.compile(r"^##\s+Potential Steps", re.MULTILINE)
_VERIFIED_HEADER = re.compile(r"^##\s+Steps", re.MULTILINE)


def _is_hypothesis_promotion(old_content: str, new_content: str) -> bool:
    """Detect a ``## Potential Steps`` → ``## Steps`` header promotion."""
    old_has_potential = bool(_HYPOTHESIS_HEADER.search(old_content or ""))
    new_has_steps = bool(_VERIFIED_HEADER.search(new_content or ""))
    new_has_potential = bool(_HYPOTHESIS_HEADER.search(new_content or ""))
    return old_has_potential and new_has_steps and not new_has_potential


# ── LLM callsite — brace-balanced JSON extraction + 5-retry ────────────────────────────────────────


async def _call_llm_for_maturity(llm: LLMClient, rendered: str) -> dict[str, Any]:
    """Call LLM for maturity scoring and return validated dict with dimension scores.

    MaturityScoreResponse is nominally flat (completeness/executability/evidence/clarity/reason),
    but uses brace-balanced extraction for safety in case ``reason`` contains nested braces.

    Raises:
        ValueError: If no JSON found or required dimension keys are missing.
    """
    response = await llm.chat(messages=[LLMChatMessage(role="user", content=rendered)])
    text = response.content
    json_str = _extract_json_object(text)
    data: dict[str, Any] = json.loads(json_str)
    missing = [d for d in _MATURITY_DIMENSIONS if d not in data]
    if missing:
        raise ValueError(f"Maturity response missing dimension keys {missing!r}: {data!r}")
    return data


def _extract_json_object(text: str) -> str:
    """First balanced {{...}} block in text (brace-balanced parser for nested/complex JSON)."""
    start = text.find("{")
    if start < 0:
        raise ValueError(f"No JSON object found in maturity LLM response: {text[:200]!r}")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ValueError(f"Unbalanced JSON in maturity LLM response: {text[:200]!r}")


# ── Maturity scoring ────────────────────────────────────────────────────────────────────────────────


_MATURITY_DIMENSIONS = ("completeness", "executability", "evidence", "clarity")


async def _evaluate_maturity(
    *,
    name: str,
    description: str,
    content: str,
    confidence: float,
    client: LLMClient,
    cfg: _SkillCfg,
    prompt_maturity: str | None = None,
) -> float:
    """4-dimension LLM scoring (completeness / executability / evidence / clarity) normalised to [0,1] via /20.

    Short-circuits to ``1.0`` when ``cfg.skip_maturity_scoring`` is ``True`` (default — current retrieval
    doesn't consult maturity_score; kept for forward compatibility).

    Raises:
        LLMError: Propagated from the underlying LLM client call.
        json.JSONDecodeError: If the LLM response is not valid JSON.
        ValueError: If the response is not a dict or is missing required dimension fields, or if
            dimension values cannot be converted to float.
    """
    if cfg.skip_maturity_scoring:
        logger.info("maturity scoring skipped by config, returning 1.0")
        return 1.0

    rendered = render_prompt(
        AGENT_SKILL_MATURITY_SCORE_PROMPT,
        prompt_maturity,
        name=name or "",
        description=description or "",
        content=content or "",
        confidence=confidence,
    )
    maturity_data = await _call_llm_for_maturity(client, rendered)
    raw_total = sum(int(maturity_data.get(d, 0)) for d in _MATURITY_DIMENSIONS)
    score: float = max(0.0, min(1.0, raw_total / 20.0))
    logger.info(
        "maturity evaluation: name=%r, raw=%.1f, score=%.2f, threshold=%.2f, ready=%s",
        name,
        raw_total,
        score,
        cfg.maturity_threshold,
        score >= cfg.maturity_threshold,
    )
    return score


# ── add / update operation atoms ────────────────────────────────────────────────────────────────────


async def _apply_add(
    op: dict[str, Any],
    source_case_ids: Sequence[str],
    *,
    client: LLMClient,
    cfg: _SkillCfg,
    prompt_maturity: str | None = None,
) -> AgentSkill | None:
    """Apply an ``add`` op: validate content / name / description, clamp confidence, run maturity scoring.

    Returns a fresh :class:`AgentSkill` or ``None`` when validation fails.
    cluster_id is left empty; caller stamps it after extraction.
    """
    data = _op_data(op)
    content = str(data.get("content") or "")
    name = str(data.get("name") or "")
    description = str(data.get("description") or "")

    if not content:
        logger.warning("add op empty content, skipping")
        return None
    if not _is_skill_content_sufficient(content):
        logger.warning(
            "add op insufficient content (len=%d), skipping content=%r",
            len(content),
            content[:100],
        )
        return None
    if not name and not description:
        logger.warning("add op missing both name and description, skipping")
        return None

    description = truncate_text(description, cfg.max_description_tokens, suffix="...")
    try:
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
    except (TypeError, ValueError):
        confidence = 0.5

    score = await _evaluate_maturity(
        name=name,
        description=description,
        content=content,
        confidence=confidence,
        client=client,
        cfg=cfg,
        prompt_maturity=prompt_maturity,
    )

    return AgentSkill(
        id=uuid.uuid4().hex,
        name=name,
        description=description,
        content=content,
        confidence=confidence,
        maturity_score=score,
        source_case_ids=list(source_case_ids),
    )


async def _apply_update(  # noqa: C901
    op: dict[str, Any],
    existing_list: Sequence[AgentSkill],
    source_case_ids: Sequence[str],
    source_quality: float,
    *,
    client: LLMClient,
    cfg: _SkillCfg,
    processed_indices: set[int],
    prompt_maturity: str | None = None,
) -> AgentSkill | None:
    """Apply an ``update`` op; return a modified :class:`AgentSkill` or ``None`` when nothing changed.

    ``confidence < cfg.retire_confidence`` signals retire; ``confidence >=`` signals upsert.
    Returns ``None`` for invalid / duplicate index, insufficient new content, or no changed fields.
    """
    # ── index parse + duplicate guard ─────────────────────────────────────────
    try:
        index = int(op.get("index", -1))
    except (TypeError, ValueError):
        logger.warning("update index not int: %r", op.get("index"))
        return None
    if index < 0 or index >= len(existing_list):
        logger.warning("update index %d out of range [0, %d)", index, len(existing_list))
        return None
    if index in processed_indices:
        logger.warning("duplicate update on index %d, skipping", index)
        return None
    processed_indices.add(index)

    prior = existing_list[index]
    data = _op_data(op)

    # ── extract new field values ──────────────────────────────────────────────
    new_name = str(data.get("name") or "")
    raw_new_description = str(data.get("description") or "")
    new_description = (
        truncate_text(raw_new_description, cfg.max_description_tokens, suffix="...") if raw_new_description else ""
    )
    new_content = str(data.get("content") or "")
    new_confidence_raw = data.get("confidence")

    if new_content and not _is_skill_content_sufficient(new_content):
        logger.warning(
            "update[%d] insufficient content (len=%d), skipping",
            index,
            len(new_content),
        )
        return None

    # ── confidence pre-parse: invalid input is treated as "no confidence change" ─
    # Opensource :580-585 doesn't add to `updates` dict on parse failure, then the
    # `if not updates` guard (:595-599) skips the op. We mirror that semantics here
    # by folding the confidence parse into the no-fields-to-change guard below.
    confidence_changed = False
    parsed_confidence: float | None = None
    if new_confidence_raw is not None:
        try:
            parsed_confidence = max(0.0, min(1.0, float(new_confidence_raw)))
            confidence_changed = True
        except (TypeError, ValueError):
            logger.warning(
                "update[%d] confidence parse failed (got %r), treating as no confidence change",
                index,
                new_confidence_raw,
            )

    # ── change flags ──────────────────────────────────────────────────────────
    name_changed = bool(new_name) and new_name != (prior.name or "")
    desc_changed = bool(new_description) and new_description != (prior.description or "")
    real_content_changed = bool(new_content) and new_content != (prior.content or "")

    if not name_changed and not desc_changed and not real_content_changed and not confidence_changed:
        logger.warning("update[%d] has no fields to change, skipping", index)
        return None

    # ── effective values ──────────────────────────────────────────────────────
    eff_name = new_name if new_name else (prior.name or "")
    eff_description = new_description if new_description else (prior.description or "")
    eff_content = new_content if new_content else (prior.content or "")
    # confidence_changed ⇔ parsed_confidence is not None; the explicit None-check below also narrows the
    # type to float for pyright.
    eff_confidence: float = parsed_confidence if parsed_confidence is not None else prior.confidence

    # ── source_case_ids append (dedup) ────────────────────────────────────────
    existing_ids = list(prior.source_case_ids or [])
    for cid in source_case_ids:
        if cid and cid not in existing_ids:
            existing_ids.append(cid)

    # ── retire branch (DESIGN.md §5.2: confidence < retire_confidence) ────────
    if eff_confidence < cfg.retire_confidence:
        logger.warning(
            "retire skill[%d] (id=%s, name=%r): confidence=%.2f < %.2f",
            index,
            prior.id,
            prior.name or "",
            eff_confidence,
            cfg.retire_confidence,
        )
        return prior.model_copy(
            update={
                "confidence": eff_confidence,
                "source_case_ids": existing_ids,
            }
        )

    # ── maturity re-eval decision (three-band) ────────────────────────────────
    new_maturity = prior.maturity_score
    content_changed = real_content_changed or name_changed or desc_changed
    if content_changed:
        change_ratio = _content_change_ratio(prior.content or "", new_content or prior.content or "")
        hyp_promo = _is_hypothesis_promotion(prior.content or "", new_content or "")

        if change_ratio < cfg.maturity_trivial_change_ratio and not hyp_promo:
            logger.info(
                "skipping maturity re-eval for skill[%d]: trivial change_ratio=%.2f < %.2f",
                index,
                change_ratio,
                cfg.maturity_trivial_change_ratio,
            )
        elif change_ratio >= cfg.maturity_reeval_change_ratio or hyp_promo:
            reason = "hypothesis promotion" if hyp_promo else f"major change ratio={change_ratio:.2f}"
            logger.info("%s for skill[%d], re-evaluating maturity", reason, index)
            new_maturity = await _evaluate_maturity(
                name=eff_name,
                description=eff_description,
                content=eff_content,
                confidence=eff_confidence,
                client=client,
                cfg=cfg,
                prompt_maturity=prompt_maturity,
            )
        else:
            # moderate band 0.2 ~ 0.4
            old_score = prior.maturity_score or 0.0
            old_confidence = prior.confidence or 0.0
            confidence_dropping = eff_confidence < old_confidence

            if old_score >= cfg.maturity_threshold and (not confidence_dropping or eff_confidence >= 0.5):
                logger.info(
                    "skipping maturity re-eval for skill[%d]: already mature (%.2f >= %.2f), "
                    "confidence=%.2f (dropping=%s), change_ratio=%.2f",
                    index,
                    old_score,
                    cfg.maturity_threshold,
                    eff_confidence,
                    confidence_dropping,
                    change_ratio,
                )
            elif old_score < cfg.maturity_threshold and source_quality < 0.3:
                logger.info(
                    "skipping maturity re-eval for skill[%d]: immature (%.2f) + low source quality "
                    "(%.2f < 0.3), change_ratio=%.2f",
                    index,
                    old_score,
                    source_quality,
                    change_ratio,
                )
            else:
                logger.info(
                    "moderate change for skill[%d]: score=%.2f, conf_drop=%s, source_quality=%.2f, "
                    "re-evaluating maturity",
                    index,
                    old_score,
                    confidence_dropping,
                    source_quality,
                )
                new_maturity = await _evaluate_maturity(
                    name=eff_name,
                    description=eff_description,
                    content=eff_content,
                    confidence=eff_confidence,
                    client=client,
                    cfg=cfg,
                    prompt_maturity=prompt_maturity,
                )

    # ── update branch: emit AgentSkill (caller diffs name/description against prior to decide reembed) ──
    updated = prior.model_copy(
        update={
            "name": eff_name,
            "description": eff_description,
            "content": eff_content,
            "confidence": eff_confidence,
            "maturity_score": new_maturity,
            "source_case_ids": existing_ids,
        }
    )
    logger.info(
        "update skill[%d]: id=%s, fields_changed=name=%s/desc=%s/content=%s/conf=%s",
        index,
        prior.id,
        name_changed,
        desc_changed,
        real_content_changed,
        new_confidence_raw is not None,
    )
    return updated
