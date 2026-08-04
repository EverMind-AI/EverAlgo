"""Synthesize a user Profile from a chronological sequence of MemCells."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from asgiref.sync import async_to_sync

from everalgo.llm.format import format_message_timestamp
from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.prompts import render_prompt
from everalgo.types import MemCell, Profile
from everalgo.user_memory._render import chat_messages, render_content
from everalgo.user_memory.prompts.en.profile import (
    PROFILE_COMPACT_PROMPT,
    PROFILE_INITIAL_EXTRACTION_PROMPT,
    PROFILE_UPDATE_PROMPT,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from everalgo.llm.protocols import LLMClient

logger = logging.getLogger(__name__)


_PROFILE_MAX_ITEMS = 30
_PROFILE_COMPACT_THRESHOLD = int(_PROFILE_MAX_ITEMS * 1.5)


class ProfileExtractor:
    """Synthesize a Profile from a chronologically ordered sequence of MemCells.

    Non-ChatMessage items are silently skipped (agent → user-memory contract).
    ``memcells`` must be ordered chronologically; the last element is the most recent and
    its timestamp becomes ``Profile.timestamp``.
    """

    def __init__(self, *, llm: LLMClient) -> None:
        self._llm = llm

    async def aextract(
        self,
        memcells: Sequence[MemCell],
        *,
        sender_id: str,
        old_profile: Profile | None = None,
        prompt: str | None = None,
    ) -> Profile:
        """Extract one Profile for ``sender_id`` from ``memcells``.

        Two extraction modes:
          INIT   (old_profile is None): full extraction from memcells.
          UPDATE (old_profile present): LLM emits add/update/delete ops on top of old_profile;
                                         when post-merge explicit_info+implicit_traits item count
                                         exceeds the internal compact threshold, a second LLM pass
                                         runs to re-summarise (compact strategy, caller-transparent).
        Returns the final Profile in both modes.

        Args:
            memcells: Must be non-empty and ordered chronologically; last element is the most recent.
            sender_id: Must be one of the memcells' human (``role == "user"``) senders; not inferred.
                An assistant sender_id is rejected — an assistant is never a valid Profile owner.
                Injected into the prompt as the target user so extraction is scoped to this speaker
                in multi-party conversations.
            old_profile: Existing profile for UPDATE mode; None triggers INIT mode.
            prompt: Prompt override; None uses the bundled default for the selected mode.

        Raises:
            ValueError: If ``memcells`` is empty, ``sender_id`` is not a user speaker in ``memcells``,
                or the LLM response is malformed.
            LLMError: From the LLM call.
            json.JSONDecodeError: On unparseable response.
        """
        if not memcells:
            raise ValueError("memcells must contain at least one MemCell")

        user_senders = _user_senders(memcells)
        if sender_id not in user_senders:
            raise ValueError(
                f"sender_id {sender_id!r} is not a user speaker in the provided memcells; "
                f"available user senders: {sorted(user_senders)!r}"
            )

        if old_profile is None:
            return await self._init_extract(memcells, sender_id=sender_id, prompt=prompt)
        return await self._update_extract(memcells, sender_id=sender_id, old_profile=old_profile, prompt=prompt)

    extract = async_to_sync(aextract)

    # ------------------------------------------------------------------
    # Private: INIT path
    # ------------------------------------------------------------------

    async def _init_extract(
        self,
        memcells: Sequence[MemCell],
        *,
        sender_id: str,
        prompt: str | None,
    ) -> Profile:
        conversation_text = _render_conversation(memcells)
        rendered = render_prompt(
            PROFILE_INITIAL_EXTRACTION_PROMPT, prompt, conversation_text=conversation_text, target_user=sender_id
        )

        data = await _call_llm_for_profile_init(self._llm, rendered)
        explicit_info = _dedupe(data["explicit_info"], source="init")
        implicit_traits = _dedupe(data["implicit_traits"], source="init")
        summary = _build_summary(explicit_info, implicit_traits)
        return Profile.model_validate(
            {
                "owner_id": sender_id,
                "summary": summary,
                "timestamp": memcells[-1].timestamp,
                "explicit_info": explicit_info,
                "implicit_traits": implicit_traits,
            }
        )

    # ------------------------------------------------------------------
    # Private: UPDATE path
    # ------------------------------------------------------------------

    async def _update_extract(
        self,
        memcells: Sequence[MemCell],
        *,
        sender_id: str,
        old_profile: Profile,
        prompt: str | None,
    ) -> Profile:
        current_profile_text = _render_profile_for_update(old_profile)
        conversation_text = _render_conversation(memcells)
        rendered = render_prompt(
            PROFILE_UPDATE_PROMPT,
            prompt,
            current_profile=current_profile_text,
            conversations=conversation_text,
            target_user=sender_id,
        )

        data = await _call_llm_for_profile_update(self._llm, rendered)
        ops_payload = data["operations"]
        merged_profile = _apply_ops(old_profile, ops_payload, timestamp=memcells[-1].timestamp)

        explicit_info: list[Any] = list(getattr(merged_profile, "explicit_info", []) or [])
        implicit_traits: list[Any] = list(getattr(merged_profile, "implicit_traits", []) or [])
        total_items = len(explicit_info) + len(implicit_traits)
        if total_items > _PROFILE_COMPACT_THRESHOLD:
            return await self._compact(merged_profile)
        return merged_profile

    async def _compact(self, profile: Profile) -> Profile:
        explicit_info: list[Any] = list(getattr(profile, "explicit_info", []) or [])
        implicit_traits: list[Any] = list(getattr(profile, "implicit_traits", []) or [])
        total_items = len(explicit_info) + len(implicit_traits)
        profile_text = json.dumps(
            {"explicit_info": explicit_info, "implicit_traits": implicit_traits},
            ensure_ascii=False,
            indent=2,
        )
        rendered = render_prompt(
            PROFILE_COMPACT_PROMPT,
            None,
            total_items=total_items,
            max_items=_PROFILE_MAX_ITEMS,
            profile_text=profile_text,
        )

        data = await _call_llm_for_profile_compact(self._llm, rendered)
        new_explicit = _dedupe(data["explicit_info"], source="compact")
        new_implicit = _dedupe(data["implicit_traits"], source="compact")
        summary = _build_summary(new_explicit, new_implicit)
        return Profile.model_validate(
            {
                "owner_id": profile.owner_id,
                "summary": summary,
                "timestamp": profile.timestamp,
                "explicit_info": new_explicit,
                "implicit_traits": new_implicit,
            }
        )


# ---------------------------------------------------------------------------
# LLM callsites — brace-balanced JSON extraction + 5-retry (mirror b150b32).
# ---------------------------------------------------------------------------


async def _call_llm_for_profile_init(llm: LLMClient, rendered: str) -> dict[str, Any]:
    """Call LLM for profile initial extraction; return validated dict with explicit_info + implicit_traits lists."""
    response = await llm.chat(messages=[LLMChatMessage(role="user", content=rendered)])
    text = response.content
    json_str = _extract_json_object(text)
    data: dict[str, Any] = json.loads(json_str)
    if "explicit_info" not in data or "implicit_traits" not in data:
        raise ValueError(f"Profile init response missing required keys: {list(data.keys())!r}")
    if not isinstance(data["explicit_info"], list) or not isinstance(data["implicit_traits"], list):
        raise ValueError(f"explicit_info and implicit_traits must be lists: {data!r}")  # noqa: TRY004
    return data


async def _call_llm_for_profile_update(llm: LLMClient, rendered: str) -> dict[str, Any]:
    """Call LLM for profile update; return validated dict with operations list."""
    response = await llm.chat(messages=[LLMChatMessage(role="user", content=rendered)])
    text = response.content
    json_str = _extract_json_object(text)
    data: dict[str, Any] = json.loads(json_str)
    if "operations" not in data:
        raise ValueError(f"Profile update response missing 'operations' key: {list(data.keys())!r}")
    if not isinstance(data["operations"], list):
        raise ValueError(f"operations must be a list: {data!r}")  # noqa: TRY004
    return data


async def _call_llm_for_profile_compact(llm: LLMClient, rendered: str) -> dict[str, Any]:
    """Call LLM for profile compact; return validated dict with explicit_info + implicit_traits lists."""
    response = await llm.chat(messages=[LLMChatMessage(role="user", content=rendered)])
    text = response.content
    json_str = _extract_json_object(text)
    data: dict[str, Any] = json.loads(json_str)
    if "explicit_info" not in data or "implicit_traits" not in data:
        raise ValueError(f"Profile compact response missing required keys: {list(data.keys())!r}")
    if not isinstance(data["explicit_info"], list) or not isinstance(data["implicit_traits"], list):
        raise ValueError(f"explicit_info and implicit_traits must be lists: {data!r}")  # noqa: TRY004
    return data


def _extract_json_object(text: str) -> str:
    """First balanced {{...}} block in text (brace-balanced parser for nested JSON)."""
    start = text.find("{")
    if start < 0:
        raise ValueError(f"No JSON object found in profile LLM response: {text[:200]!r}")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ValueError(f"Unbalanced JSON in profile LLM response: {text[:200]!r}")


# ---------------------------------------------------------------------------
# Module-level helpers.
# ---------------------------------------------------------------------------


def _user_senders(memcells: Sequence[MemCell]) -> set[str]:
    """Return sender_ids of human (``role == "user"``) chat messages across all memcells.

    Assistant turns are intentionally excluded: an assistant is never a valid Profile owner.
    """
    return {m.sender_id for cell in memcells for m in chat_messages(cell) if m.role == "user"}


def _render_conversation(memcells: Sequence[MemCell]) -> str:
    """Render ChatMessage items as ``[ISO-ts] speaker(user_id:xxx): content`` lines; tool items are skipped."""
    lines: list[str] = []
    for cell in memcells:
        for m in chat_messages(cell):
            text = render_content(m.content)
            if not text:
                continue
            speaker = m.sender_name or m.sender_id
            user_id = m.sender_id or ""
            time_str = format_message_timestamp(m.timestamp)
            lines.append(f"[{time_str}] {speaker}(user_id:{user_id}): {text}")
    if not lines:
        lines.append("(no prior MemCells in the cluster)")
    return "\n".join(lines)


def _render_profile_for_update(profile: Profile) -> str:
    """Render existing profile fields as indexed JSON for the UPDATE prompt."""
    explicit_info: list[Any] = list(getattr(profile, "explicit_info", []) or [])
    implicit_traits: list[Any] = list(getattr(profile, "implicit_traits", []) or [])
    parts: list[str] = ["=== explicit_info ==="]
    for i, item in enumerate(explicit_info):
        parts.append(f"[{i}] {json.dumps(item, ensure_ascii=False)}")
    parts.append("=== implicit_traits ===")
    for i, item in enumerate(implicit_traits):
        parts.append(f"[{i}] {json.dumps(item, ensure_ascii=False)}")
    return "\n".join(parts)


@dataclass
class _BucketOps:
    """One bucket's ops after validation, keyed by the index numbering the LLM was shown."""

    updates: dict[int, dict[str, Any]] = field(default_factory=dict)
    deletes: set[int] = field(default_factory=set)
    adds: list[dict[str, Any]] = field(default_factory=list)


def _apply_ops(old_profile: Profile, ops: list[dict[str, Any]], *, timestamp: int) -> Profile:
    """Apply add/update/delete ops to old_profile and return a new Profile.

    Collection is separated from application so that every ``index`` resolves against ``old_profile``
    exactly as ``_render_profile_for_update`` numbered it for the LLM. Applying ops one at a time to a
    mutating list would shift the indices of every op that follows a delete, and would let an ``add``
    widen the bound an out-of-range index is checked against. Ops that fail validation are dropped
    with a warning instead of landing on a neighbouring item.

    Args:
        old_profile: Profile the ops were generated against; supplies the index numbering.
        ops: Raw ``operations`` payload from the LLM, unvalidated.
        timestamp: Timestamp for the merged Profile, normally the newest MemCell's.

    Returns:
        A new Profile; ``old_profile`` is left untouched.
    """
    buckets = {
        "explicit_info": list(getattr(old_profile, "explicit_info", []) or []),
        "implicit_traits": list(getattr(old_profile, "implicit_traits", []) or []),
    }
    collected = _collect_ops(ops, buckets)
    explicit_info = _apply_bucket_ops(buckets["explicit_info"], collected["explicit_info"])
    implicit_traits = _apply_bucket_ops(buckets["implicit_traits"], collected["implicit_traits"])

    return Profile.model_validate(
        {
            "owner_id": old_profile.owner_id,
            "summary": _build_summary(explicit_info, implicit_traits),
            "timestamp": timestamp,
            "explicit_info": explicit_info,
            "implicit_traits": implicit_traits,
        }
    )


def _collect_ops(ops: list[dict[str, Any]], buckets: dict[str, list[Any]]) -> dict[str, _BucketOps]:
    """Group ops by bucket, validating each against ``buckets`` without modifying it.

    ``buckets`` doubles as the bucket-name whitelist, so an unrecognised ``type`` cannot fall through
    to whichever bucket an ``if/else`` happened to leave as the default.
    """
    collected = {name: _BucketOps() for name in buckets}
    for op in ops:
        action = op.get("action")
        if action == "none":  # documented no-op: the conversation carried no user info
            continue
        op_type = op.get("type")
        if op_type not in collected:
            _log_rejected_op(op, "unknown type")
        elif action == "add":
            _collect_add(op, collected[op_type])
        elif action in ("update", "delete"):
            _collect_indexed(op, collected[op_type], items=buckets[op_type])
        else:
            _log_rejected_op(op, "unknown action")

    for slot in collected.values():
        _drop_updates_superseded_by_delete(slot)
    return collected


def _collect_add(op: dict[str, Any], slot: _BucketOps) -> None:
    """Queue an add, rejecting a payload that is unusable or has no identity to deduplicate on."""
    data = op.get("data")
    if not isinstance(data, dict):
        _log_rejected_op(op, "data is not an object")
        return
    item = cast("dict[str, Any]", data)
    if _identity_key(item) is None:
        _log_rejected_op(op, "missing description")
        return
    slot.adds.append(item)


def _collect_indexed(op: dict[str, Any], slot: _BucketOps, *, items: list[Any]) -> None:
    """Queue an update or delete whose index falls inside the original snapshot."""
    idx = op.get("index")
    # bool is an int subclass, so a JSON `true` would otherwise address item 1.
    if not isinstance(idx, int) or isinstance(idx, bool) or not 0 <= idx < len(items):
        _log_rejected_op(op, "index out of range")
        return
    if op.get("action") == "delete":
        slot.deletes.add(idx)
        return
    data = op.get("data")
    if not isinstance(data, dict):
        _log_rejected_op(op, "data is not an object")
        return
    if not isinstance(items[idx], dict):
        _log_rejected_op(op, "target item is not an object")
        return
    # An update is a partial merge, so two of them on one index accumulate rather than replace.
    slot.updates[idx] = {**slot.updates.get(idx, {}), **cast("dict[str, Any]", data)}


def _drop_updates_superseded_by_delete(slot: _BucketOps) -> None:
    """Resolve an index that is both updated and deleted in favour of the delete.

    ``delete`` answers to an explicit negation, an expiry, or a contradiction, so keeping a rewritten
    copy of something the user has disowned is worse than losing one correction.
    """
    for idx in sorted(slot.updates.keys() & slot.deletes):  # sorted so the warnings come out in a fixed order
        del slot.updates[idx]
        logger.warning("profile update op rejected: action=update index=%d reason=superseded by delete", idx)


def _apply_bucket_ops(items: list[Any], bucket_ops: _BucketOps) -> list[Any]:
    """Apply one bucket's ops: patch by original index, drop deletes, then append the new items.

    Deletes are removed in a single filtering pass rather than by repeated ``pop``, which is what kept
    the surviving indices aligned with the numbering every op was validated against. Every index here
    was validated during collection, including that the item it addresses is an object.
    """
    for idx, patch in bucket_ops.updates.items():
        items[idx] = {**cast("dict[str, Any]", items[idx]), **patch}
    kept = [item for i, item in enumerate(items) if i not in bucket_ops.deletes]
    kept.extend(bucket_ops.adds)
    return _dedupe(kept, source="update")


def _dedupe(items: list[Any], *, source: str) -> list[Any]:
    """Drop items whose identity has already appeared, keeping the earliest.

    Applied to the whole bucket rather than only to newly added items, which covers three ways a
    duplicate arises: an ``add`` of something already stored, an ``update`` that rewrites one item
    into a copy of another, and duplicates already sitting in a profile written before this check
    existed. That last one matters because ``_compact`` is the only other backstop and does not run
    until the item count passes ``_PROFILE_COMPACT_THRESHOLD``, so an update is a polluted profile's
    one route back. Discarding an exact duplicate loses no information; an item with no identity is
    kept untouched, since it cannot be compared and dropping it would lose data.

    Args:
        items: Bucket contents in order; the earliest occurrence of each identity survives.
        source: Which path is deduplicating, for the warning only.

    Returns:
        A new list preserving the original order.
    """
    kept: list[Any] = []
    seen: set[str] = set()
    for item in items:
        key = _identity_key(cast("dict[str, Any]", item)) if isinstance(item, dict) else None
        if key is None:
            kept.append(item)
            continue
        if key in seen:
            logger.warning("profile duplicate item dropped: source=%s", source)
            continue
        seen.add(key)
        kept.append(item)
    return kept


def _identity_key(item: dict[str, Any]) -> str | None:
    """Semantic identity of a profile item, or None when it carries none.

    Identity is the ``description`` alone, which is the item's content and the field both buckets
    share. The label (``category`` for ``explicit_info``, ``trait`` for ``implicit_traits``) stays out
    because the LLM writes it freely — the same fact reaches ``diet`` one round and ``preferences`` the
    next — and a label that drifts would be a way around the very dedup this guards. ``evidence`` /
    ``basis`` stay out as provenance: the same fact quoted from a second conversation is still the
    same fact.
    """
    description = item.get("description")
    if not isinstance(description, str):
        return None
    # An empty description is no identity: it would match every other item that is missing one.
    return _normalize(description) or None


def _normalize(text: str) -> str:
    """Fold whitespace runs and case, so a restated item still matches the copy already stored."""
    return " ".join(text.split()).casefold()


def _log_rejected_op(op: dict[str, Any], reason: str) -> None:
    """Report a dropped op. ``data`` is never logged — it holds LLM-authored profile content."""
    logger.warning(
        "profile update op rejected: action=%s type=%s index=%s reason=%s",
        op.get("action"),
        op.get("type"),
        op.get("index"),
        reason,
    )


def _build_summary(explicit_info: list[Any], implicit_traits: list[Any]) -> str:
    """Return the first ``description`` from explicit_info, then implicit_traits; sentinel ``"(no summary)"`` if both empty."""
    for item in explicit_info:
        if not isinstance(item, dict):
            continue
        desc = item.get("description")  # type: ignore[reportUnknownVariableType,reportUnknownMemberType]
        if isinstance(desc, str) and desc.strip():
            return desc.strip()
    for item in implicit_traits:
        if not isinstance(item, dict):
            continue
        desc = item.get("description") or item.get("trait")  # type: ignore[reportUnknownVariableType,reportUnknownMemberType]
        if isinstance(desc, str) and desc.strip():
            return desc.strip()
    return "(no summary)"
