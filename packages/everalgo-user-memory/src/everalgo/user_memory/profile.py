"""Synthesize a user Profile from chronological MemCells or dated Episodes, keeping the profile's timeline."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from asgiref.sync import async_to_sync

from everalgo.llm.format import format_message_timestamp
from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.prompts import render_prompt
from everalgo.types import Episode, MemCell, Profile
from everalgo.user_memory._language import (
    COMPACTED_PROFILE_LANGUAGE_RULE,
    EPISODE_PROFILE_INIT_LANGUAGE_RULE,
    EXISTING_PROFILE_LANGUAGE_RULE,
    PROFILE_INIT_LANGUAGE_RULE,
    OutputLanguage,
    build_language_rule,
)
from everalgo.user_memory._render import chat_messages, render_content
from everalgo.user_memory._width import ascii_width as _ascii_width
from everalgo.user_memory.prompts.en.profile import (
    PROFILE_COMPACT_PROMPT,
    PROFILE_INITIAL_EXTRACTION_PROMPT,
    PROFILE_REGROUP_PROMPT,
    PROFILE_UPDATE_PROMPT,
)
from everalgo.user_memory.prompts.en.profile_from_episode_texts import (
    HISTORICAL_PASS_RULE,
    PROFILE_COMPACT_FROM_EPISODE_TEXTS_PROMPT,
    PROFILE_INITIAL_FROM_EPISODE_TEXTS_PROMPT,
    PROFILE_REGROUP_FROM_EPISODE_TEXTS_PROMPT,
    PROFILE_UPDATE_FROM_EPISODE_TEXTS_PROMPT,
)

if TYPE_CHECKING:
    from everalgo.llm.protocols import LLMClient

logger = logging.getLogger(__name__)


# Which field names a bucket's grouping dimension. v2: the label is a GROUP key shared by many
# atomic items, not a unique key — used to canonicalise spelling, never to merge content.
_LABEL_FIELDS = {"explicit_info": "category", "implicit_traits": "trait"}
# v2 caps. Items are atomic (one fact each), so the same portrait spans more items than v1's
# one-blob-per-dimension packing; 60 total covers v1's 30 dimensions at two facts each. The
# per-category cap bounds a single dimension flooding the portrait.
#
# Length is measured in ASCII-EQUIVALENT WIDTH UNITS (East Asian Wide/Fullwidth chars count 2,
# everything else 1), so ONE threshold holds across languages with no per-language enumeration:
# 200 units = ~200 English chars = ~100 CJK chars, and Japanese/Korean inherit it for free.
# The backstop is an ENGINEERING guard against runaway concatenation only -- the style rule
# ("one or two short sentences") lives in the prompt. 250 units: measured compliant-item p95 is
# 183-198 units (en corpus), so 250 catches the runaway tail without policing compliant items;
# 200 would sit ON the style budget and thrash regroup on legitimate English items.
_PROFILE_MAX_ITEMS = 60
_PROFILE_MAX_PER_CATEGORY = 8
_ITEM_WIDTH_BACKSTOP = 250
# Full-profile compaction fires ONLY on the total cap; per-label breaches (count or width) get a
# group-scoped regroup instead — see _overcrowded_labels and _regroup. Compaction remains the only
# path that removes never-was-a-portrait items (UPDATE emits such deletes in ~1 run out of 10).
#
# Per-item observation date (epoch ms): the date of the source that last established the item's
# description. Written by this module only — the model is shown it as a date but any value it echoes
# back is discarded — so that a narrative observed BEFORE an item was established cannot rewrite or
# delete it (see _collect_indexed). Without it a backfilled older Episode arriving after a newer one
# would overwrite the newer state, because the UPDATE prompt otherwise treats every new narrative as
# the latest word.
_OBSERVED_AT = "observed_at"


class ProfileExtractor:
    """Synthesize one owner-scoped Profile from chronological memory inputs.

    ``aextract`` reads chronologically ordered MemCells and silently skips non-ChatMessage items.
    ``aextract_from_episodes`` reads dated generic or reflected Episodes in any order, skips items that do not
    reference the resolved owner, raises only when none reference that owner, and keeps the profile's timeline
    itself: an Episode observed before the stored profile's newest date can add to the profile but never
    rewrites or deletes what a later Episode established.
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
        output_language: OutputLanguage | str | None = None,
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
            output_language: Language to write the profile in, as an :class:`OutputLanguage` member or
                equivalent string in any casing. Honoured by all three modes; pass the same language on every
                call and a profile's language is fixed by the caller for its whole life, with no inference
                anywhere. Left ``None`` each mode falls back to its own judgement: INIT follows the
                participants, which measured 18% wrong over four models against 0% when the language is
                named, while UPDATE and COMPACT inherit the language the profile is already written in —
                safe against a later conversation in another language splitting a profile in half, but
                unrecoverable if the language it inherits is already wrong. Naming a language on an update
                is also how such a profile gets corrected. See ``prompts/en/_language.py`` for the
                measurements.

        Raises:
            ValueError: If ``memcells`` is empty, ``sender_id`` is not a user speaker in ``memcells``,
                the LLM response is malformed, or ``output_language`` names no supported language.
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

        mode = "INIT" if old_profile is None else "UPDATE"
        logger.info(
            "extracting profile: mode=%s, %d memcells, existing explicit=%d implicit=%d, output_language=%s",
            mode,
            len(memcells),
            len(getattr(old_profile, "explicit_info", []) or []) if old_profile else 0,
            len(getattr(old_profile, "implicit_traits", []) or []) if old_profile else 0,
            output_language,
        )
        if old_profile is None:
            result = await self._init_extract(
                memcells, sender_id=sender_id, prompt=prompt, output_language=output_language
            )
        else:
            result = await self._update_extract(
                memcells,
                sender_id=sender_id,
                old_profile=old_profile,
                prompt=prompt,
                output_language=output_language,
            )
        logger.info(
            "profile extracted: mode=%s -> explicit=%d implicit=%d",
            mode,
            len(getattr(result, "explicit_info", []) or []),
            len(getattr(result, "implicit_traits", []) or []),
        )
        return result

    extract = async_to_sync(aextract)

    async def aextract_from_episodes(
        self,
        episodes: Sequence[Episode],
        *,
        owner_id: str,
        owner_name: str | None = None,
        old_profile: Profile | None = None,
        categories: Sequence[str] | None = None,
        prompt: str | None = None,
        output_language: OutputLanguage | str | None = None,
    ) -> Profile:
        """Extract one owner-scoped Profile from dated Episode narratives.

        Each Episode's ``timestamp`` is the observation date of what it narrates. The batch is ordered by that
        date, every narrative is shown to the model under it, and in UPDATE mode the batch is split around the
        stored ``Profile.timestamp``: Episodes observed before it are older knowledge that may add facts and
        evidence but never rewrite or delete what a later Episode established; Episodes observed after it update
        normally. Callers therefore pass whatever Episodes are new to them, in any order — the profile's timeline
        is kept here, not upstream.

        Args:
            episodes: Non-empty Episodes with non-blank narratives. Items that do not contain the resolved target
                reference are excluded from extraction; at least one item must contain it.
            owner_id: Authoritative owner identifier written to ``Profile.owner_id``.
            owner_name: Optional display name used to select Episode narratives about the owner. A blank or
                missing name falls back to ``owner_id``.
            old_profile: Existing profile for UPDATE mode; None triggers INIT mode.
            categories: Complete current category snapshot for ``explicit_info``. Values are stripped, blank
                values are ignored, and exact duplicates are removed in first-seen order. None or an empty
                result lets the model create concise categories as needed. This does not constrain
                ``implicit_traits.trait``.
            prompt: Prompt override for INIT or UPDATE; None uses the matching bundled Episode prompt.
            output_language: Language to write the profile in. None makes INIT inherit the Episode language and
                UPDATE preserve the existing Profile language.

        Returns:
            The extracted or updated Profile. ``Profile.timestamp`` is the newest observation date seen so far and
            never moves backwards; every item carries ``observed_at``, the observation date of the narrative that
            last established its description.

        Raises:
            TypeError: If an item of ``episodes`` is not an :class:`Episode`, or ``categories`` is not a sequence
                of strings or contains a non-string value.
            ValueError: If owner or Episode validation fails, ``old_profile`` belongs to another owner, the LLM
                response is malformed, or ``output_language`` names no supported language.
            LLMError: From the LLM call.
            json.JSONDecodeError: On an unparseable response.
        """
        target_user = _resolve_episode_target(owner_id, owner_name)
        selected = _select_episodes(episodes, target_user=target_user)
        available_categories = _render_available_categories(_normalize_available_categories(categories))
        if old_profile is not None and old_profile.owner_id != owner_id:
            raise ValueError(f"old_profile.owner_id {old_profile.owner_id!r} does not match owner_id {owner_id!r}")

        mode = "INIT" if old_profile is None else "UPDATE"
        logger.info(
            "extracting profile from Episodes: mode=%s, %d/%d matching episodes observed %s..%s, "
            "existing explicit=%d implicit=%d, output_language=%s",
            mode,
            len(selected),
            len(episodes),
            format_message_timestamp(selected[0].timestamp),
            format_message_timestamp(selected[-1].timestamp),
            len(getattr(old_profile, "explicit_info", []) or []) if old_profile else 0,
            len(getattr(old_profile, "implicit_traits", []) or []) if old_profile else 0,
            output_language,
        )
        if old_profile is None:
            result = await self._init_extract_from_episodes(
                selected,
                owner_id=owner_id,
                target_user=target_user,
                available_categories=available_categories,
                prompt=prompt,
                output_language=output_language,
            )
        else:
            result = await self._update_extract_from_episodes(
                selected,
                owner_id=owner_id,
                target_user=target_user,
                old_profile=old_profile,
                available_categories=available_categories,
                prompt=prompt,
                output_language=output_language,
            )
        logger.info(
            "profile extracted from Episodes: mode=%s -> explicit=%d implicit=%d",
            mode,
            len(getattr(result, "explicit_info", []) or []),
            len(getattr(result, "implicit_traits", []) or []),
        )
        return result

    extract_from_episodes = async_to_sync(aextract_from_episodes)

    # ------------------------------------------------------------------
    # Private: INIT path
    # ------------------------------------------------------------------

    async def _init_extract(
        self,
        memcells: Sequence[MemCell],
        *,
        sender_id: str,
        prompt: str | None,
        output_language: OutputLanguage | str | None,
    ) -> Profile:
        conversation_text = _render_conversation(memcells)
        rendered = render_prompt(
            PROFILE_INITIAL_EXTRACTION_PROMPT,
            prompt,
            conversation_text=conversation_text,
            target_user=_sender_display_name(memcells, sender_id),
            target_user_id=sender_id,
            language_rule=build_language_rule(output_language, fallback=PROFILE_INIT_LANGUAGE_RULE),
        )

        data = await _call_llm_for_profile_init(self._llm, rendered)
        observed_at = memcells[-1].timestamp
        explicit_info = _stamp_observed_at(_dedupe(data["explicit_info"], source="init"), observed_at)
        implicit_traits = _stamp_observed_at(_dedupe(data["implicit_traits"], source="init"), observed_at)
        summary = _build_summary(explicit_info, implicit_traits)
        return Profile.model_validate(
            {
                "owner_id": sender_id,
                "summary": summary,
                "timestamp": observed_at,
                "explicit_info": explicit_info,
                "implicit_traits": implicit_traits,
            }
        )

    async def _init_extract_from_episodes(
        self,
        episodes: Sequence[Episode],
        *,
        owner_id: str,
        target_user: str,
        available_categories: str,
        prompt: str | None,
        output_language: OutputLanguage | str | None,
    ) -> Profile:
        rendered = render_prompt(
            PROFILE_INITIAL_FROM_EPISODE_TEXTS_PROMPT,
            prompt,
            episode_texts=_render_episodes(episodes),
            target_user=target_user,
            available_categories=available_categories,
            language_rule=build_language_rule(output_language, fallback=EPISODE_PROFILE_INIT_LANGUAGE_RULE),
        )

        data = await _call_llm_for_profile_init(self._llm, rendered)
        observed_at = episodes[-1].timestamp
        explicit_info = _stamp_observed_at(_dedupe(data["explicit_info"], source="episode-init"), observed_at)
        implicit_traits = _stamp_observed_at(_dedupe(data["implicit_traits"], source="episode-init"), observed_at)
        return Profile.model_validate(
            {
                "owner_id": owner_id,
                "summary": _build_summary(explicit_info, implicit_traits),
                "timestamp": observed_at,
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
        output_language: OutputLanguage | str | None,
    ) -> Profile:
        current_profile_text = _render_profile_for_update(old_profile)
        conversation_text = _render_conversation(memcells)
        rendered = render_prompt(
            PROFILE_UPDATE_PROMPT,
            prompt,
            current_profile=current_profile_text,
            conversations=conversation_text,
            target_user=_sender_display_name(memcells, sender_id),
            target_user_id=sender_id,
            language_rule=build_language_rule(output_language, fallback=EXISTING_PROFILE_LANGUAGE_RULE),
        )

        data = await _call_llm_for_profile_update(self._llm, rendered)
        ops_payload = data["operations"]
        merged_profile = _apply_ops(old_profile, ops_payload, timestamp=memcells[-1].timestamp)

        return await self._maintain_updated_profile(
            merged_profile,
            operation_count=len(ops_payload),
            display_name=_sender_display_name(memcells, sender_id),
            sender_id=sender_id,
            output_language=output_language,
            compact_prompt=PROFILE_COMPACT_PROMPT,
            regroup_prompt=PROFILE_REGROUP_PROMPT,
            available_categories=None,
        )

    async def _update_extract_from_episodes(
        self,
        episodes: Sequence[Episode],
        *,
        owner_id: str,
        target_user: str,
        old_profile: Profile,
        available_categories: str,
        prompt: str | None,
        output_language: OutputLanguage | str | None,
    ) -> Profile:
        """Run UPDATE in up to two passes: Episodes older than the stored profile first, then the rest.

        A pass carries ONE observation date — its newest narrative — and ``_collect_indexed`` judges every
        rewrite against that date. Mixing an older and a newer narrative in one pass would let the newer one's
        date vouch for a rewrite the older one actually motivated, so the split is what gives the guard its
        meaning. Real-time ingestion never has an older half; only backfill and out-of-order delivery do.
        """
        historical = [episode for episode in episodes if episode.timestamp < old_profile.timestamp]
        current = list(episodes[len(historical) :])
        if historical:
            logger.info(
                "profile update split around stored timestamp %s: %d historical, %d current narratives",
                format_message_timestamp(old_profile.timestamp),
                len(historical),
                len(current),
            )
        profile = old_profile
        for batch, is_historical in ((historical, True), (current, False)):
            if batch:
                profile = await self._update_pass_from_episodes(
                    batch,
                    owner_id=owner_id,
                    target_user=target_user,
                    old_profile=profile,
                    available_categories=available_categories,
                    prompt=prompt,
                    output_language=output_language,
                    historical=is_historical,
                )
        return profile

    async def _update_pass_from_episodes(
        self,
        episodes: Sequence[Episode],
        *,
        owner_id: str,
        target_user: str,
        old_profile: Profile,
        available_categories: str,
        prompt: str | None,
        output_language: OutputLanguage | str | None,
        historical: bool = False,
    ) -> Profile:
        # The historical pass knows every narrative predates the stored profile, so it can withdraw the general
        # permission to add "a fact not yet on file" for dimensions that already hold a value: an earlier value of a
        # settled dimension is not a new fact. The current pass renders an empty rule slot.
        rendered = render_prompt(
            PROFILE_UPDATE_FROM_EPISODE_TEXTS_PROMPT,
            prompt,
            current_profile=_render_profile_for_episode_update(old_profile),
            episode_texts=_render_episodes(episodes),
            target_user=target_user,
            available_categories=available_categories,
            language_rule=build_language_rule(output_language, fallback=EXISTING_PROFILE_LANGUAGE_RULE),
            pass_rule=HISTORICAL_PASS_RULE if historical else "",
        )

        data = await _call_llm_for_profile_update(self._llm, rendered)
        operations = data["operations"]
        merged_profile = _apply_ops(old_profile, operations, timestamp=episodes[-1].timestamp)
        return await self._maintain_updated_profile(
            merged_profile,
            operation_count=len(operations),
            display_name=target_user,
            sender_id=owner_id,
            output_language=output_language,
            compact_prompt=PROFILE_COMPACT_FROM_EPISODE_TEXTS_PROMPT,
            regroup_prompt=PROFILE_REGROUP_FROM_EPISODE_TEXTS_PROMPT,
            available_categories=available_categories,
        )

    async def _maintain_updated_profile(
        self,
        merged_profile: Profile,
        *,
        operation_count: int,
        display_name: str,
        sender_id: str,
        output_language: OutputLanguage | str | None,
        compact_prompt: str,
        regroup_prompt: str,
        available_categories: str | None = None,
    ) -> Profile:
        """Apply the shared post-UPDATE cap maintenance with source-specific prompts."""
        explicit_info: list[Any] = list(getattr(merged_profile, "explicit_info", []) or [])
        implicit_traits: list[Any] = list(getattr(merged_profile, "implicit_traits", []) or [])
        logger.info(
            "profile update applied: %d ops -> explicit=%d implicit=%d",
            operation_count,
            len(explicit_info),
            len(implicit_traits),
        )
        # Full-profile compaction ONLY on the total cap: rewriting everything to fix one
        # overcrowded group deletes valid items in groups that were fine (measured: a
        # noise round dropped 6 of 13 gold facts). Group-level breaches — too many items
        # under one label, or a runaway-length item — get a REGROUP pass scoped to that
        # one group, whose legal moves are split/rename/merge-restatements, not deletion.
        if len(explicit_info) + len(implicit_traits) > _PROFILE_MAX_ITEMS:
            logger.warning(
                "profile total cap breached (%d > %d items), running full compact",
                len(explicit_info) + len(implicit_traits),
                _PROFILE_MAX_ITEMS,
            )
            return await self._compact(
                merged_profile,
                display_name=display_name,
                sender_id=sender_id,
                output_language=output_language,
                prompt_template=compact_prompt,
                available_categories=available_categories,
            )
        result = merged_profile
        for bucket, label in _overcrowded_labels(explicit_info, implicit_traits):
            result = await self._regroup(
                result,
                bucket=bucket,
                label=label,
                display_name=display_name,
                sender_id=sender_id,
                output_language=output_language,
                prompt_template=regroup_prompt,
                available_categories=available_categories,
            )
        return result

    async def _regroup(
        self,
        profile: Profile,
        *,
        bucket: str,
        label: str,
        display_name: str,
        sender_id: str,
        prompt_template: str,
        available_categories: str | None = None,
        output_language: OutputLanguage | str | None = None,
    ) -> Profile:
        """Reorganise ONE overcrowded group in place; every other item passes through untouched.

        The group's items are re-filed under dimension-true labels (split/rename), restatements
        of one fact merge, and only never-was-a-portrait items may drop. A model response that
        still breaches the cap is accepted with a warning — regroup never loops.
        """
        wanted = _normalize(label)
        label_field = _LABEL_FIELDS[bucket]
        buckets: dict[str, list[Any]] = {
            "explicit_info": list(getattr(profile, "explicit_info", []) or []),
            "implicit_traits": list(getattr(profile, "implicit_traits", []) or []),
        }
        group: list[Any] = [
            it
            for it in buckets[bucket]
            if (lbl := _item_label(it, label_field)) is not None and _normalize(lbl) == wanted
        ]
        rest: list[Any] = [it for it in buckets[bucket] if it not in group]
        other_labels = sorted({lbl for it in rest if (lbl := _item_label(it, label_field)) is not None})
        logger.warning(
            "profile group over cap, regrouping: bucket=%s label=%s group_items=%d (cap %d/label, %d-unit width backstop)",
            bucket,
            label,
            len(group),
            _PROFILE_MAX_PER_CATEGORY,
            _ITEM_WIDTH_BACKSTOP,
        )
        rendered = render_prompt(
            prompt_template,
            None,
            label=label,
            label_field=label_field,
            count=len(group),
            max_per_category=_PROFILE_MAX_PER_CATEGORY,
            other_labels=", ".join(other_labels) if other_labels else "(none)",
            items_text=json.dumps(_without_observed_at(group), ensure_ascii=False, indent=2),
            target_user=display_name,
            target_user_id=sender_id,
            available_categories=available_categories or "[]",
            language_rule=build_language_rule(output_language, fallback=COMPACTED_PROFILE_LANGUAGE_RULE),
        )
        data = await _call_llm_for_profile_regroup(self._llm, rendered)
        returned: list[Any] = list(data["items"])
        kept: list[Any] = [it for it in returned if isinstance(it, dict) and _identity_key(cast("dict[str, Any]", it))]
        dropped_malformed = len(returned) - len(kept)
        if dropped_malformed:
            logger.warning("profile regroup dropped %d malformed item(s) label=%s", dropped_malformed, label)
        # Same as compaction: a regrouped item has no traceable origin, so it takes the profile's own date.
        new_bucket = _dedupe([*rest, *_stamp_observed_at(kept, profile.timestamp)], source="regroup")
        still = sum(
            1 for it in new_bucket if (lbl := _item_label(it, label_field)) is not None and _normalize(lbl) == wanted
        )
        if still > _PROFILE_MAX_PER_CATEGORY:
            logger.warning("profile regroup left label over cap: label=%s count=%d", label, still)
        buckets[bucket] = new_bucket
        logger.info(
            "profile regroup done: label=%s group %d -> %d items",
            label,
            len(group),
            len(kept),
        )
        return Profile.model_validate(
            {
                "owner_id": profile.owner_id,
                "summary": _build_summary(buckets["explicit_info"], buckets["implicit_traits"]),
                "timestamp": profile.timestamp,
                "explicit_info": buckets["explicit_info"],
                "implicit_traits": buckets["implicit_traits"],
            }
        )

    async def _compact(
        self,
        profile: Profile,
        *,
        display_name: str,
        sender_id: str,
        prompt_template: str,
        available_categories: str | None = None,
        output_language: OutputLanguage | str | None = None,
    ) -> Profile:
        explicit_info: list[Any] = list(getattr(profile, "explicit_info", []) or [])
        implicit_traits: list[Any] = list(getattr(profile, "implicit_traits", []) or [])
        total_items = len(explicit_info) + len(implicit_traits)
        profile_text = json.dumps(
            {
                "explicit_info": _without_observed_at(explicit_info),
                "implicit_traits": _without_observed_at(implicit_traits),
            },
            ensure_ascii=False,
            indent=2,
        )
        rendered = render_prompt(
            prompt_template,
            None,
            total_items=total_items,
            max_items=_PROFILE_MAX_ITEMS,
            max_per_category=_PROFILE_MAX_PER_CATEGORY,
            profile_text=profile_text,
            target_user=display_name,
            target_user_id=sender_id,
            available_categories=available_categories or "[]",
            language_rule=build_language_rule(output_language, fallback=COMPACTED_PROFILE_LANGUAGE_RULE),
        )

        data = await _call_llm_for_profile_compact(self._llm, rendered)
        # Rewritten items cannot be traced back to the ones they came from, so they take the profile's own
        # date: the newest known, which keeps them protected from older narratives.
        new_explicit = _stamp_observed_at(_dedupe(data["explicit_info"], source="compact"), profile.timestamp)
        new_implicit = _stamp_observed_at(_dedupe(data["implicit_traits"], source="compact"), profile.timestamp)
        logger.info(
            "profile compacted: %d -> %d items",
            total_items,
            len(new_explicit) + len(new_implicit),
        )
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
# LLM callsites — brace-balanced JSON extraction, one attempt each (no retry; see OpenAICompatClient).
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


async def _call_llm_for_profile_regroup(llm: LLMClient, rendered: str) -> dict[str, Any]:
    """Call LLM for a single-group regroup; return validated dict with an items list."""
    response = await llm.chat(messages=[LLMChatMessage(role="user", content=rendered)])
    json_str = _extract_json_object(response.content)
    data: dict[str, Any] = json.loads(json_str)
    if "items" not in data:
        raise ValueError(f"Profile regroup response missing 'items' key: {list(data.keys())!r}")
    if not isinstance(data["items"], list):
        raise ValueError(f"items must be a list: {data!r}")  # noqa: TRY004
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


def _resolve_episode_target(owner_id: str, owner_name: str | None) -> str:
    """Resolve the one literal owner reference used to select Episode narratives."""
    if not owner_id.strip():
        raise ValueError("owner_id must be a non-blank string")
    return owner_name.strip() if owner_name and owner_name.strip() else owner_id.strip()


def _select_episodes(episodes: Sequence[object], *, target_user: str) -> tuple[Episode, ...]:
    """Validate the batch, keep the Episodes whose narrative references the target user, order them by date."""
    if isinstance(episodes, (str, bytes)) or not episodes:
        raise ValueError("episodes must be a non-empty sequence of Episode")

    normalized_target = _normalize(target_user)
    selected: list[Episode] = []
    for index, episode in enumerate(episodes):
        if not isinstance(episode, Episode):
            raise TypeError(f"episodes[{index}] must be an Episode")
        if not episode.episode.strip():
            raise ValueError(f"episodes[{index}].episode must be a non-blank string")
        if normalized_target in _normalize(episode.episode):
            selected.append(episode)

    if not selected:
        raise ValueError(f"no episodes reference target user {target_user!r}")
    return tuple(sorted(selected, key=lambda episode: episode.timestamp))


def _normalize_available_categories(categories: object) -> tuple[str, ...]:
    """Validate and normalize a caller-provided category snapshot without changing its order."""
    if categories is None:
        return ()
    if isinstance(categories, (str, bytes)) or not isinstance(categories, Sequence):
        raise TypeError("categories must be a sequence of strings or None")

    normalized: list[str] = []
    seen: set[str] = set()
    category_values = cast("Sequence[object]", categories)
    for index, category in enumerate(category_values):
        if not isinstance(category, str):
            raise TypeError(f"categories[{index}] must be a string")
        stripped = category.strip()
        if not stripped or stripped in seen:
            continue
        seen.add(stripped)
        normalized.append(stripped)
    return tuple(normalized)


def _render_available_categories(categories: Sequence[str]) -> str:
    """Render the normalized category snapshot as deterministic JSON for prompt injection."""
    return json.dumps(list(categories), ensure_ascii=False, indent=2)


def _render_episodes(episodes: Sequence[Episode]) -> str:
    """Join narratives oldest to newest, each under its observation date; no synthetic identifiers or numbering."""
    return "\n\n---\n\n".join(
        f"[{format_message_timestamp(episode.timestamp)} UTC]\n{episode.episode.strip()}" for episode in episodes
    )


def _user_senders(memcells: Sequence[MemCell]) -> set[str]:
    """Return sender_ids of human (``role == "user"``) chat messages across all memcells.

    Assistant turns are intentionally excluded: an assistant is never a valid Profile owner.
    """
    return {m.sender_id for cell in memcells for m in chat_messages(cell) if m.role == "user"}


def _sender_display_name(memcells: Sequence[MemCell], sender_id: str) -> str:
    """Human-readable label for ``sender_id``; the id itself when the conversation carries no name.

    Only the label is affected — ``sender_id`` stays the locator the prompt matches speakers by, because
    ``ChatMessage.sender_name`` is optional and not guaranteed unique, and single-subject attribution is
    the strongest requirement these prompts carry.
    """
    for cell in memcells:
        for m in chat_messages(cell):
            if m.sender_id == sender_id and m.sender_name and m.sender_name.strip():
                return m.sender_name.strip()
    return sender_id


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
    """Render the existing profile for the UPDATE prompt: label inventory first, then indexed JSON."""
    explicit_info, implicit_traits = _dated_buckets(profile)
    return "\n".join(
        (_render_label_inventory(explicit_info, implicit_traits), _render_indexed_items(explicit_info, implicit_traits))
    )


def _render_profile_for_episode_update(profile: Profile) -> str:
    """Render indexed items without treating stored category names as preferred classification choices."""
    return _render_indexed_items(*_dated_buckets(profile))


def _dated_buckets(profile: Profile) -> tuple[list[Any], list[Any]]:
    """The profile's two buckets with every item dated, exactly as ``_apply_ops`` will judge them.

    Rendering and merging must agree: an item the model sees without a date is one it cannot reason about, and
    an item the guard dates differently from what the model saw is a rule the model could not have followed.
    """
    return (
        _fill_observed_at(list(getattr(profile, "explicit_info", []) or []), profile.timestamp),
        _fill_observed_at(list(getattr(profile, "implicit_traits", []) or []), profile.timestamp),
    )


def _render_label_inventory(explicit_info: list[Any], implicit_traits: list[Any]) -> str:
    """List the category / trait labels already in use, deduplicated, in first-seen order.

    Listed on their own rather than left implicit in the JSON dump below: deciding add-versus-update is a
    comparison against the labels, and asking for it inside a wall of JSON is what let near-synonyms
    accumulate — measured at 8 explicit items from 8 conversations, 86% of them one-off actions restated as
    standing capabilities, with one category repeated 7 times.
    """
    return "\n".join(
        (
            "=== categories already in use (reuse one when the new fact belongs to that dimension) ===",
            _render_labels(explicit_info, key="category"),
            "=== trait labels already in use (one disposition = one trait; add evidence to one of these rather than naming a synonym) ===",
            _render_labels(implicit_traits, key="trait"),
        )
    )


def _render_labels(items: list[Any], *, key: str) -> str:
    """Comma-joined distinct non-empty ``key`` values, or a sentinel when the bucket carries none."""
    seen: dict[str, None] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        label = item.get(key)
        if isinstance(label, str) and label.strip():
            seen.setdefault(label.strip(), None)
    return ", ".join(seen) if seen else "(none yet)"


def _render_indexed_items(explicit_info: list[Any], implicit_traits: list[Any]) -> str:
    """Render both buckets as ``[i] {json}`` lines — the numbering every op index resolves against."""
    parts: list[str] = ["=== explicit_info ==="]
    for i, item in enumerate(explicit_info):
        parts.append(f"[{i}] {_render_item(item)}")
    parts.append("=== implicit_traits ===")
    for i, item in enumerate(implicit_traits):
        parts.append(f"[{i}] {_render_item(item)}")
    return "\n".join(parts)


def _render_item(item: Any) -> str:
    """One item as JSON, with ``observed_at`` shown as the date the model can compare against narrative dates."""
    if isinstance(item, dict):
        item = cast("dict[str, Any]", item)
        observed_at = item.get(_OBSERVED_AT)
        if _is_timestamp(observed_at):
            item = {**item, _OBSERVED_AT: f"{format_message_timestamp(observed_at)} UTC"}
    return json.dumps(item, ensure_ascii=False)


def _is_timestamp(value: object) -> bool:
    """True for an epoch-ms int; ``bool`` is an int subclass and is excluded."""
    return isinstance(value, int) and not isinstance(value, bool)


def _stamp_observed_at(items: list[Any], observed_at: int) -> list[Any]:
    """Copies of the dict items with ``observed_at`` SET to ``observed_at``, whatever the model wrote there."""
    return [
        {**cast("dict[str, Any]", item), _OBSERVED_AT: observed_at} if isinstance(item, dict) else item
        for item in items
    ]


def _fill_observed_at(items: list[Any], observed_at: int) -> list[Any]:
    """Copies of the dict items with a missing or malformed ``observed_at`` set to ``observed_at``.

    Profiles written before the field existed carry none; their items take the profile's own date, the newest
    knowledge they can be assumed to reflect.
    """
    return [
        item
        if not isinstance(item, dict) or _is_timestamp(cast("dict[str, Any]", item).get(_OBSERVED_AT))
        else {**cast("dict[str, Any]", item), _OBSERVED_AT: observed_at}
        for item in items
    ]


def _without_observed_at(items: list[Any]) -> list[Any]:
    """Copies of the dict items with ``observed_at`` removed, for prompts that rewrite items rather than date them."""
    return [
        {k: v for k, v in cast("dict[str, Any]", item).items() if k != _OBSERVED_AT} if isinstance(item, dict) else item
        for item in items
    ]


def _is_stale_for(item: Any, observed_at: int) -> bool:
    """True when ``observed_at`` is earlier than the date that established ``item``: older knowledge, no rewrites."""
    if not isinstance(item, dict):
        return False
    established = cast("dict[str, Any]", item).get(_OBSERVED_AT)
    return isinstance(established, int) and not isinstance(established, bool) and observed_at < established


def _changes_description(item: dict[str, Any], patch: dict[str, Any]) -> bool:
    """True when ``patch`` states a different fact than ``item``; grounding-only patches are not rewrites."""
    proposed = _identity_key(patch)
    return proposed is not None and proposed != _identity_key(item)


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

    ``timestamp`` is the observation date of the source the ops came from. It stamps every added item and every
    rewritten description, and it is what an update or delete is judged against: an item established by a
    LATER source is not rewritten or deleted on the strength of this one. The merged profile's own timestamp
    never moves backwards.

    Args:
        old_profile: Profile the ops were generated against; supplies the index numbering.
        ops: Raw ``operations`` payload from the LLM, unvalidated.
        timestamp: Observation date (epoch ms) of the source batch, normally its newest MemCell's or Episode's.

    Returns:
        A new Profile; ``old_profile`` is left untouched.
    """
    explicit_items, implicit_items = _dated_buckets(old_profile)
    buckets = {"explicit_info": explicit_items, "implicit_traits": implicit_items}
    collected = _collect_ops(ops, buckets, observed_at=timestamp)
    explicit_info = _apply_bucket_ops(buckets["explicit_info"], collected["explicit_info"], observed_at=timestamp)
    implicit_traits = _apply_bucket_ops(buckets["implicit_traits"], collected["implicit_traits"], observed_at=timestamp)

    return Profile.model_validate(
        {
            "owner_id": old_profile.owner_id,
            "summary": _build_summary(explicit_info, implicit_traits),
            "timestamp": max(old_profile.timestamp, timestamp),
            "explicit_info": explicit_info,
            "implicit_traits": implicit_traits,
        }
    )


def _collect_ops(
    ops: list[dict[str, Any]], buckets: dict[str, list[Any]], *, observed_at: int
) -> dict[str, _BucketOps]:
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
            _collect_add(op, collected[op_type], items=buckets[op_type], label_field=_LABEL_FIELDS[op_type])
        elif action in ("update", "delete"):
            _collect_indexed(op, collected[op_type], items=buckets[op_type], observed_at=observed_at)
        else:
            _log_rejected_op(op, "unknown action")

    for slot in collected.values():
        _drop_updates_superseded_by_delete(slot)
    return collected


def _collect_add(op: dict[str, Any], slot: _BucketOps, *, items: list[Any], label_field: str) -> None:
    """Queue an add, canonicalising its label to the spelling already on file.

    v2: a label is a GROUP key — several atomic items under one category is the intended shape, so an
    add whose label is already in use is appended as a sibling, never folded into the owner's prose
    (v1's fold is what let items concatenate without bound). Only the label's spelling is normalised,
    so one dimension cannot split into casing/whitespace variants of its own name. Near-synonym labels
    are the prompt's job (label inventory) with compaction as the backstop. Fact-level duplicates are
    still caught by ``_dedupe`` (description identity).
    """
    data = op.get("data")
    if not isinstance(data, dict):
        _log_rejected_op(op, "data is not an object")
        return
    item = cast("dict[str, Any]", data)
    item.pop(_OBSERVED_AT, None)  # code-owned; stamped in _apply_bucket_ops
    if _identity_key(item) is None:
        _log_rejected_op(op, "missing description")
        return

    owner = _index_owning_label(items, item.get(label_field), label_field=label_field)
    if owner is not None:
        item[label_field] = cast("dict[str, Any]", items[owner])[label_field]
    slot.adds.append(item)


def _index_owning_label(items: list[Any], label: object, *, label_field: str) -> int | None:
    """Index of the item already using ``label``, comparing the way ``_normalize`` does; None if free."""
    if not isinstance(label, str) or not label.strip():
        return None
    wanted = _normalize(label)
    for idx, existing in enumerate(items):
        if not isinstance(existing, dict):
            continue
        current = existing.get(label_field)
        if isinstance(current, str) and _normalize(current) == wanted:
            return idx
    return None


def _overcrowded_labels(explicit_info: list[Any], implicit_traits: list[Any]) -> list[tuple[str, str]]:
    """``(bucket, label)`` pairs needing a regroup pass.

    Flagged when a label holds too many items, or one of its items is past the width backstop
    (runaway concatenation — a split candidate, not a style gate). The label returned is the first-seen original spelling, for prompt display; matching is by
    ``_normalize``. Order is deterministic: explicit_info first, then first-seen label order.
    """
    flagged: list[tuple[str, str]] = []
    for bucket, items in (("explicit_info", explicit_info), ("implicit_traits", implicit_traits)):
        label_field = _LABEL_FIELDS[bucket]
        counts: dict[str, int] = {}
        first_spelling: dict[str, str] = {}
        oversized: dict[str, bool] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            item = cast("dict[str, Any]", item)
            label = item.get(label_field)
            if not isinstance(label, str) or not label.strip():
                continue
            key = _normalize(label)
            first_spelling.setdefault(key, label.strip())
            counts[key] = counts.get(key, 0) + 1
            description = item.get("description")
            if isinstance(description, str) and _ascii_width(description) > _ITEM_WIDTH_BACKSTOP:
                oversized[key] = True
        for key, spelling in first_spelling.items():
            if counts[key] > _PROFILE_MAX_PER_CATEGORY or oversized.get(key):
                flagged.append((bucket, spelling))
    return flagged


def _collect_indexed(op: dict[str, Any], slot: _BucketOps, *, items: list[Any], observed_at: int) -> None:
    """Queue an update or delete whose index falls inside the original snapshot.

    ``observed_at`` is the source batch's date. Against an item established later than that, a delete or a
    description rewrite is stale evidence and is dropped — the later source already settled the current state
    — while a grounding-only update still lands, since more evidence for a fact is never wrong.
    """
    idx = op.get("index")
    if not _is_timestamp(idx) or not 0 <= cast("int", idx) < len(items):
        _log_rejected_op(op, "index out of range")
        return
    idx = cast("int", idx)
    stale = _is_stale_for(items[idx], observed_at)
    if op.get("action") == "delete":
        if stale:
            _log_rejected_op(op, "stale evidence")
            return
        slot.deletes.add(idx)
        return
    data = op.get("data")
    if not isinstance(data, dict):
        _log_rejected_op(op, "data is not an object")
        return
    if not isinstance(items[idx], dict):
        _log_rejected_op(op, "target item is not an object")
        return
    patch = cast("dict[str, Any]", data)
    patch.pop(_OBSERVED_AT, None)  # code-owned; re-stamped in _apply_bucket_ops when the description changes
    if stale and _changes_description(cast("dict[str, Any]", items[idx]), patch):
        _log_rejected_op(op, "stale evidence")
        return
    # An update is a partial merge, so two of them on one index accumulate rather than replace.
    slot.updates[idx] = {**slot.updates.get(idx, {}), **patch}


def _drop_updates_superseded_by_delete(slot: _BucketOps) -> None:
    """Resolve an index that is both updated and deleted in favour of the delete.

    ``delete`` answers to an explicit negation, an expiry, or a contradiction, so keeping a rewritten
    copy of something the user has disowned is worse than losing one correction.
    """
    for idx in sorted(slot.updates.keys() & slot.deletes):  # sorted so the warnings come out in a fixed order
        del slot.updates[idx]
        logger.warning("profile update op rejected: action=update index=%d reason=superseded by delete", idx)


def _apply_bucket_ops(items: list[Any], bucket_ops: _BucketOps, *, observed_at: int) -> list[Any]:
    """Apply one bucket's ops: patch by original index, drop deletes, then append the new items.

    Deletes are removed in a single filtering pass rather than by repeated ``pop``, which is what kept
    the surviving indices aligned with the numbering every op was validated against. Every index here
    was validated during collection, including that the item it addresses is an object. A rewritten
    description and every added item take ``observed_at``; a grounding-only patch keeps the item's date.
    """
    for idx, patch in bucket_ops.updates.items():
        current = cast("dict[str, Any]", items[idx])
        merged = {**current, **patch}
        if _changes_description(current, patch):
            merged[_OBSERVED_AT] = observed_at
        items[idx] = merged
    kept = [item for i, item in enumerate(items) if i not in bucket_ops.deletes]
    kept.extend(_stamp_observed_at(bucket_ops.adds, observed_at))
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


def _item_label(item: Any, label_field: str) -> str | None:
    """The item's stripped label string, or None when it is not a dict or carries no usable label."""
    if not isinstance(item, dict):
        return None
    label = cast("dict[str, Any]", item).get(label_field)
    if isinstance(label, str) and label.strip():
        return label.strip()
    return None


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
