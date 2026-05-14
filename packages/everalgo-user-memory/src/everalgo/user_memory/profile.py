"""Profile extractor — aligned with new-release single-call initial extraction.

Interface preserves the existing ``aextract(...) -> Profile`` shape (single Profile return). Internally
runs the new-release initial-extraction prompt with one LLM call and parses
``{explicit_info, implicit_traits}``.

Algorithm details (mirrors opensource ``profile_memory/extractor.py`` new-release initial path):
    - One LLM call against :data:`PROFILE_INITIAL_EXTRACTION_PROMPT` with the rendered conversation text
      (current MemCell + optional prior cluster MemCells concatenated chronologically).
    - 5-retry on JSON parse / schema failure; on exhaustion returns a minimal fallback Profile so the
      pipeline can continue.
    - Parsed ``explicit_info`` / ``implicit_traits`` are preserved on the Profile via ``extra="allow"``;
      ``summary`` is synthesised from the first explicit_info description (falls back to a sentinel).
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from asgiref.sync import async_to_sync

import everalgo.llm
from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.prompts import render_prompt
from everalgo.types import MemCell, Profile
from everalgo.user_memory.prompts.en.profile import PROFILE_INITIAL_EXTRACTION_PROMPT

if TYPE_CHECKING:
    from everalgo.llm.protocols import LLMClient

logger = logging.getLogger(__name__)

_MAX_LLM_RETRIES = 5
"""Retry budget on JSON parse / schema failure (matches opensource retry pattern)."""


class ProfileExtractor:
    """Synthesize a user profile from a MemCell + optional prior cluster.

    Stateless callable class. Runs the new-release single-call initial extraction; returns one Profile
    whose ``summary`` is derived from the first ``explicit_info`` entry and whose ``explicit_info`` /
    ``implicit_traits`` lists are preserved via Profile's ``extra="allow"``.

    Customize per call via ``prompt=`` override and ``cluster_episodes=`` context passthrough.
    """

    async def aextract(
        self,
        memcell: MemCell,
        *,
        cluster_episodes: list[MemCell] | None = None,
        llm: LLMClient | None = None,
        prompt: str | None = None,
    ) -> Profile:
        """Run a single LLM extraction; return a Profile (or a fallback on exhaustion)."""
        client = everalgo.llm.resolve(llm)
        conversation_text = _render_conversation(memcell, cluster_episodes or [])
        rendered = render_prompt(PROFILE_INITIAL_EXTRACTION_PROMPT, prompt, conversation_text=conversation_text)

        parsed = await _llm_call_with_retry(client, rendered)
        if parsed is None:
            return _fallback_profile(memcell)

        explicit_info = parsed.get("explicit_info") or []
        implicit_traits = parsed.get("implicit_traits") or []
        if not isinstance(explicit_info, list):
            explicit_info = []
        if not isinstance(implicit_traits, list):
            implicit_traits = []

        owner_id = _derive_owner_id(memcell)
        summary = _build_summary(explicit_info, implicit_traits)
        return Profile.model_validate(
            {
                "id": f"pf_{uuid.uuid4().hex[:12]}",
                "owner_id": owner_id,
                "summary": summary,
                "timestamp": memcell.timestamp,
                "explicit_info": explicit_info,
                "implicit_traits": implicit_traits,
            }
        )

    extract = async_to_sync(aextract)
    """Sync bridge — only callable from non-event-loop contexts."""


# Module-level helpers.


def _render_conversation(memcell: MemCell, cluster_episodes: list[MemCell]) -> str:
    """Render messages as ``[timestamp][event_id] speaker(user_id:xxx): content`` lines.

    Cluster MemCells (if any) are rendered chronologically before the current MemCell so the LLM sees
    historical context first.
    """
    cells = [*cluster_episodes, memcell]
    lines: list[str] = []
    for cell in cells:
        for m in cell.messages:
            if not m.content:
                continue
            speaker = m.sender_name or m.role.value
            user_id = m.sender_id or ""
            time_str = datetime.fromtimestamp(m.timestamp / 1000, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
            cell_ref = cell.event_id or ""
            lines.append(f"[{time_str}][{cell_ref}] {speaker}(user_id:{user_id}): {m.content}")
    if not lines:
        lines.append("(no prior MemCells in the cluster)")
    return "\n".join(lines)


async def _llm_call_with_retry(client: LLMClient, rendered_prompt: str) -> dict[str, Any] | None:
    """Call LLM with 5-retry; parse ``{explicit_info, implicit_traits}`` envelope. Return None on exhaustion."""
    for attempt in range(_MAX_LLM_RETRIES):
        try:
            response = await client.chat(
                messages=[LLMChatMessage(role="user", content=rendered_prompt)],
                response_format={"type": "json_object"},
            )
            return _parse_profile_payload(response.content)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.warning("profile retry %d/%d: %s", attempt + 1, _MAX_LLM_RETRIES, e)
            if attempt == _MAX_LLM_RETRIES - 1:
                logger.exception("profile extraction exhausted %d retries", _MAX_LLM_RETRIES)
                return None
            continue
    return None


def _parse_profile_payload(raw: str) -> dict[str, Any]:
    """Parse the new-release ``{explicit_info, implicit_traits}`` payload.

    Raises :class:`ValueError` on shape mismatch (caller catches and retries).
    """
    parsed: object = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("LLM response is not a JSON object")  # noqa: TRY004 — uniform retry semantics
    data = cast("dict[str, Any]", parsed)
    if "explicit_info" not in data and "implicit_traits" not in data:
        raise ValueError("LLM response missing both explicit_info and implicit_traits")
    return data


def _build_summary(explicit_info: list[Any], implicit_traits: list[Any]) -> str:
    """Synthesise a one-line summary from the first available description/trait.

    Prefers the first ``explicit_info[].description``; falls back to the first
    ``implicit_traits[].description``; sentinel ``"(no summary)"`` when both are empty.
    """
    for item in explicit_info:
        if not isinstance(item, dict):
            continue
        desc = item.get("description")
        if isinstance(desc, str) and desc.strip():
            return desc.strip()
    for item in implicit_traits:
        if not isinstance(item, dict):
            continue
        desc = item.get("description") or item.get("trait")
        if isinstance(desc, str) and desc.strip():
            return desc.strip()
    return "(no summary)"


def _fallback_profile(memcell: MemCell) -> Profile:
    """Return a minimal Profile when the LLM call exhausts retries."""
    return Profile.model_validate(
        {
            "id": f"pf_{uuid.uuid4().hex[:12]}",
            "owner_id": _derive_owner_id(memcell),
            "summary": "(no summary)",
            "timestamp": memcell.timestamp,
            "explicit_info": [],
            "implicit_traits": [],
        }
    )


def _derive_owner_id(memcell: MemCell) -> str:
    if memcell.participants:
        return memcell.participants[0]
    for m in memcell.messages:
        if m.sender_id:
            return m.sender_id
    return "u_default"
