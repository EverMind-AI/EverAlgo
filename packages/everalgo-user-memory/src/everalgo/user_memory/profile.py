"""Synthesize a user Profile from a chronological sequence of MemCells."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from asgiref.sync import async_to_sync
from pydantic import BaseModel, Field

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


# ------------------------------------------------------------------
# Pydantic schemas for structured outputs (json_schema mode)
# ------------------------------------------------------------------


class _ExplicitInfoItem(BaseModel):
    """Single explicit_info item from profile extraction."""

    category: str
    description: str
    evidence: str


class _ImplicitTraitItem(BaseModel):
    """Single implicit_traits item from profile extraction."""

    trait: str
    description: str
    basis: str
    evidence: str


class _ProfileInitialExtractionResponse(BaseModel):
    """Response schema for PROFILE_INITIAL_EXTRACTION_PROMPT."""

    explicit_info: list[_ExplicitInfoItem]
    implicit_traits: list[_ImplicitTraitItem]


class _ProfileOperation(BaseModel):
    """Single operation for profile update."""

    action: str
    type: str | None = Field(default=None)
    index: int | None = Field(default=None)
    data: dict[str, Any] | None = Field(default=None)
    reason: str | None = Field(default=None)


class _ProfileUpdateResponse(BaseModel):
    """Response schema for PROFILE_UPDATE_PROMPT."""

    operations: list[_ProfileOperation]
    update_note: str


class _ProfileCompactResponse(BaseModel):
    """Response schema for PROFILE_COMPACT_PROMPT."""

    explicit_info: list[_ExplicitInfoItem]
    implicit_traits: list[_ImplicitTraitItem]
    compact_note: str


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
            sender_id: Must be one of the memcells' chat senders; not inferred.
            old_profile: Existing profile for UPDATE mode; None triggers INIT mode.
            prompt: Prompt override; None uses the bundled default for the selected mode.

        Raises:
            ValueError: If ``memcells`` is empty or the LLM response is malformed.
            LLMError: From the LLM call.
            json.JSONDecodeError: On unparseable response.
        """
        if not memcells:
            raise ValueError("memcells must contain at least one MemCell")

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
        rendered = render_prompt(PROFILE_INITIAL_EXTRACTION_PROMPT, prompt, conversation_text=conversation_text)

        response = await self._llm.chat(
            messages=[LLMChatMessage(role="user", content=rendered)],
            response_format=_ProfileInitialExtractionResponse,
        )
        parsed = response.parsed
        if not isinstance(parsed, _ProfileInitialExtractionResponse):
            raise TypeError(f"LLM returned unexpected parsed type: {type(parsed)}")

        explicit_info = [item.model_dump() for item in parsed.explicit_info]
        implicit_traits = [item.model_dump() for item in parsed.implicit_traits]
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
        )

        response = await self._llm.chat(
            messages=[LLMChatMessage(role="user", content=rendered)],
            response_format=_ProfileUpdateResponse,
        )
        parsed = response.parsed
        if not isinstance(parsed, _ProfileUpdateResponse):
            raise TypeError(f"LLM returned unexpected parsed type: {type(parsed)}")
        ops_payload = [op.model_dump(exclude_none=True) for op in parsed.operations]
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

        response = await self._llm.chat(
            messages=[LLMChatMessage(role="user", content=rendered)],
            response_format=_ProfileCompactResponse,
        )
        parsed = response.parsed
        if not isinstance(parsed, _ProfileCompactResponse):
            raise TypeError(f"LLM returned unexpected parsed type: {type(parsed)}")
        new_explicit = [item.model_dump() for item in parsed.explicit_info]
        new_implicit = [item.model_dump() for item in parsed.implicit_traits]
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
# Module-level helpers.
# ---------------------------------------------------------------------------


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


def _to_list(value: object) -> list[Any]:
    """Coerce to list[Any]; returns [] for non-list values."""
    return value if isinstance(value, list) else []


def _apply_ops(old_profile: Profile, ops: list[dict[str, Any]], *, timestamp: int) -> Profile:
    """Apply add/update/delete ops to old_profile and return a new Profile."""
    explicit_info: list[Any] = list(getattr(old_profile, "explicit_info", []) or [])
    implicit_traits: list[Any] = list(getattr(old_profile, "implicit_traits", []) or [])

    for op in ops:
        action = op.get("action")
        if action == "none":
            continue
        op_type = op.get("type")
        target = explicit_info if op_type == "explicit_info" else implicit_traits

        if action == "add":
            data = op.get("data")
            if isinstance(data, dict):
                target.append(data)
        elif action == "update":
            idx = op.get("index")
            data = op.get("data")
            if isinstance(idx, int) and isinstance(data, dict) and 0 <= idx < len(target):
                merged = dict(target[idx])
                merged.update(data)
                target[idx] = merged
        elif action == "delete":
            idx = op.get("index")
            if isinstance(idx, int) and 0 <= idx < len(target):
                target.pop(idx)

    summary = _build_summary(explicit_info, implicit_traits)
    return Profile.model_validate(
        {
            "owner_id": old_profile.owner_id,
            "summary": summary,
            "timestamp": timestamp,
            "explicit_info": explicit_info,
            "implicit_traits": implicit_traits,
        }
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
