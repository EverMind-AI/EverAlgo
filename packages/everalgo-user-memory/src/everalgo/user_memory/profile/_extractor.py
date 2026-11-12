"""``ProfileExtractor``: the two entry paths and the stages each runs, wired from the sibling modules."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC
from typing import TYPE_CHECKING, Any

from asgiref.sync import async_to_sync

from everalgo.llm import LLMOutputError
from everalgo.user_memory._language import OutputLanguage, resolve_output_language
from everalgo.user_memory.profile._compact_ops import apply_compact_ops
from everalgo.user_memory.profile._do_not_record import (
    DO_NOT_RECORD_MAX_ENTRIES,
    apply_merge_ops,
    enter_matters,
    load_matters,
    mergeable,
    withdraw_matters,
)
from everalgo.user_memory.profile._items import (
    _ITEM_WIDTH_BACKSTOP,
    _LABEL_FIELDS,
    _PROFILE_MAX_ITEMS,
    _PROFILE_MAX_PER_CATEGORY,
    _dated_buckets,
    _in_group,
    _Item,
    _item_label,
    _normalize,
)
from everalgo.user_memory.profile._merge import (
    _apply_ops,
    _dedupe,
    _identity_key,
    _overcrowded_labels,
    _profile_from_buckets,
    _stored_items,
)
from everalgo.user_memory.profile._pending import (
    _expire_pending,
    check_init_items,
    check_update_ops,
    unchecked_init_items,
    unchecked_update_ops,
)
from everalgo.user_memory.profile._render import (
    _normalize_available_categories,
    _render_available_categories,
    _render_bucket,
    _render_conversation,
    _render_do_not_record,
    _render_episodes,
    _render_group,
    _render_pending,
    _render_profile_for_compact,
    _render_profile_for_episode_update,
    _render_profile_for_update,
    _resolve_episode_target,
    _select_episodes,
    _sender_display_name,
    _user_senders,
)
from everalgo.user_memory.profile._replies import (
    _AnyCompactOutput,
    _AnyRegroupOutput,
    _call_llm_for_do_not_record_merge,
    _call_llm_for_profile_compact,
    _call_llm_for_profile_init,
    _call_llm_for_profile_regroup,
    _call_llm_for_profile_update,
    _CompactOpsOutput,
    _RegroupOpsOutput,
)
from everalgo.user_memory.profile._stages import (
    _EPISODE_COMPACT,
    _EPISODE_INIT,
    _EPISODE_REGROUP,
    _EPISODE_UPDATE,
    _MEMCELL_COMPACT,
    _MEMCELL_INIT,
    _MEMCELL_REGROUP,
    _MEMCELL_UPDATE,
    _StagePrompts,
    render_do_not_record_merge,
)
from everalgo.user_memory.prompts.en.profile_from_episode_texts import REGROUP_REPLY_SKELETONS
from everalgo.utils import InvalidInputError

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import tzinfo

    from everalgo.llm.protocols import LLMClient
    from everalgo.types import Episode, MemCell, Profile

# The package's name rather than this module's: hosts collect the log contract under it.
logger = logging.getLogger("everalgo.user_memory.profile")


@dataclass(frozen=True)
class _EpisodeLists:
    """The two lists an Episode call carries beside the profile, changed in place by its stages.

    ``aextract_from_episodes`` copies them from the stored profile's extra fields (empty for INIT), hands them to
    every stage it runs, and writes them back onto the returned Profile. The MemCell path has neither.

    Attributes:
        do_not_record: The matters this person asked not to be recorded, each ``{"topic", "observed_at"}`` plus the
            parts since allowed (``_do_not_record``); INIT and every UPDATE pass add the matters their reply names,
            and an UPDATE pass also takes off the ones withdrawn.
        pending: The pattern entries waiting for a later batch to repeat them (``_pending``).
    """

    do_not_record: list[dict[str, Any]] = field(default_factory=list)
    pending: list[dict[str, Any]] = field(default_factory=list)


class ProfileExtractor:
    """Synthesize one owner-scoped Profile from chronological memory inputs.

    ``aextract`` reads chronologically ordered MemCells and silently skips non-ChatMessage items.
    ``aextract_from_episodes`` reads dated generic or reflected Episodes in any order, skips items that do not
    reference the resolved owner, raises only when none reference that owner, and keeps the profile's timeline
    itself: an Episode observed before the stored profile's newest date can add to the profile but never
    rewrites or deletes what a later Episode established.

    Unless ``new_traits`` is set, no new ``implicit_traits`` item is created on either path: INIT keeps none of the
    traits its reply holds and UPDATE refuses an add into ``implicit_traits``, while a stored trait can still be
    updated or deleted. On both paths an item whose description carries an identification or account number is
    dropped and such a number in its evidence or basis is masked (``_merge._scrub``).

    Args:
        llm: The client every stage calls.
        item_checks: Run the Episode path's item checks (``_pending``): an item whose quote is not found in the
            batch is dropped and an unrepeated pattern item waits on the pending list. Off, every item is written as
            the reply has it and nothing is held, while ``confirm`` is still read as an add and the reply-only fields
            are still dropped; the prompts are the same either way.
        new_traits: Let INIT keep the reply's traits and UPDATE add to ``implicit_traits``. Off by default: a trait
            is the model's inference, and in long-run evaluation inferred traits accounted for about half of the
            severe errors (items that give a wrong picture of the person); what a person says about their own values,
            situation and habits is stated directly, so the prompts file it under ``explicit_info`` instead.
    """

    def __init__(self, *, llm: LLMClient, item_checks: bool = True, new_traits: bool = False) -> None:
        self._llm = llm
        self._item_checks = item_checks
        self._new_traits = new_traits

    async def aextract(
        self,
        memcells: Sequence[MemCell],
        *,
        sender_id: str,
        old_profile: Profile | None = None,
        custom_instructions: str | None = None,
        output_language: OutputLanguage | str | None = None,
        tz: tzinfo = UTC,
    ) -> Profile:
        """Extract one Profile for ``sender_id`` from ``memcells``.

        With no ``old_profile`` this is INIT, a full extraction. Otherwise it is UPDATE: the model emits
        add / update / delete ops on ``old_profile``, and a merged profile past an item cap gets one COMPACT or
        REGROUP pass, transparent to the caller. Exceptions raised by the LLM client propagate unchanged.

        Args:
            memcells: Non-empty and chronological; the last element is the most recent.
            sender_id: One of the memcells' human (``role == "user"``) senders, never an assistant. The prompt
                scopes extraction to this speaker.
            old_profile: Existing profile for UPDATE; None for INIT.
            custom_instructions: The host's instructions for every call this makes, added verbatim in an
                【Additional Requirements】 section after the bundled rules and before the data. They cannot replace
                the rules or the output shape. None or a blank string adds nothing.
            output_language: Language to write the profile in, as an :class:`OutputLanguage` member or its value
                in any casing. None lets INIT follow the participants and UPDATE and COMPACT keep the language the
                profile is written in.
            tz: The zone the times shown to the model are in, as :func:`everalgo.utils.format_time` renders them:
                ``UTC`` (the default) or local time with its offset, ``UTC+08:00``.

        Returns:
            The extracted or updated Profile.

        Raises:
            InvalidInputError: If ``memcells`` is empty, ``sender_id`` is not a user speaker in ``memcells``,
                ``output_language`` names no supported language, or a ChatMessage with text has no timestamp (0).
            LLMOutputError: If a reply is not a valid INIT, UPDATE, COMPACT or REGROUP output.
        """
        if not memcells:
            raise InvalidInputError(
                "memcells is empty, there is nothing to extract a profile from: "
                "pass at least one MemCell, or skip the call when a conversation has none"
            )

        user_senders = _user_senders(memcells)
        if sender_id not in user_senders:
            raise InvalidInputError(
                f"sender_id {sender_id!r} is not a user speaker in memcells (user senders: {sorted(user_senders)!r}): "
                "pass the sender_id of a human speaker (role 'user') who appears in the memcells"
            )

        language = resolve_output_language(output_language)
        mode = "INIT" if old_profile is None else "UPDATE"
        logger.info(
            "extracting profile: mode=%s, %d memcells, existing explicit=%d implicit=%d, output_language=%s, "
            "custom_instructions=%d chars",
            mode,
            len(memcells),
            len(old_profile.explicit_info) if old_profile else 0,
            len(old_profile.implicit_traits) if old_profile else 0,
            language,
            len(custom_instructions or ""),
        )
        if old_profile is None:
            result = await self._init_extract(
                memcells,
                sender_id=sender_id,
                custom_instructions=custom_instructions,
                output_language=language,
                tz=tz,
            )
        else:
            result = await self._update_extract(
                memcells,
                sender_id=sender_id,
                old_profile=old_profile,
                custom_instructions=custom_instructions,
                output_language=language,
                tz=tz,
            )
        logger.info(
            "profile extracted: mode=%s -> explicit=%d implicit=%d",
            mode,
            len(result.explicit_info),
            len(result.implicit_traits),
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
        custom_instructions: str | None = None,
        output_language: OutputLanguage | str | None = None,
        tz: tzinfo = UTC,
    ) -> Profile:
        """Extract one owner-scoped Profile from dated Episode narratives.

        Each Episode's ``timestamp`` is the observation date of what it narrates. The batch is ordered by that
        date, and in UPDATE mode split around the stored ``Profile.timestamp``: Episodes observed before it may
        add facts and evidence but never rewrite or delete what a later Episode established; the rest update
        normally. Callers pass whatever Episodes are new to them, in any order. Exceptions raised by the LLM client
        propagate unchanged.

        The profile keeps a do-not-record list: the matters this person asked not to be recorded, as INIT's
        ``do_not_record`` field and UPDATE's ``forget`` operations name them, each kept only when the program finds
        the quoted request in the batch's narratives. Every UPDATE shows the list to the model, whose rules keep any
        item about a listed matter out of the profile. A matter leaves the list, whole or in part, when an UPDATE's
        ``allow`` operation quotes a withdrawal from a narrative later than the request; it never expires. A list
        that a call leaves longer than before and past 30 entries gets one extra call that removes redundant entries
        (``_do_not_record``); no entry is ever dropped for the cap.

        Every item the model writes carries a quote the program finds in the batch's narratives, or it is dropped; an
        item claiming a standing pattern (a habit, preference, way of working, trait) is written at once only when its
        quote shows regularity, and otherwise waits on the profile's pending list until a later batch shows it again
        (``_pending``). Neither check calls the model again.

        Args:
            episodes: Non-empty Episodes with non-blank narratives. Items that do not contain the resolved target
                reference are excluded; at least one must contain it.
            owner_id: Authoritative owner identifier written to ``Profile.owner_id``.
            owner_name: Display name used to select narratives about the owner; blank or missing falls back to
                ``owner_id``.
            old_profile: Existing profile for UPDATE; None for INIT. Its ``do_not_record`` extra field, when present,
                is the do-not-record list so far, and its ``pending`` extra field the pending pattern entries; the
                host persists both with the profile, replacing the stored field with the returned one each time, and
                passes them back unchanged.
            categories: Complete current category snapshot for ``explicit_info``, stripped, blanks ignored and
                exact duplicates removed in first-seen order. None or empty lets the model coin categories. It
                does not constrain ``implicit_traits.trait``.
            custom_instructions: The host's instructions for every call this makes, added verbatim as an
                ``<additional_requirements>`` block closing the system message. They cannot replace the rules or the
                output shape. None or a blank string adds nothing.
            output_language: Language to write the profile in. None makes INIT inherit the Episode language and
                UPDATE preserve the existing Profile language.
            tz: The zone the times shown to the model are in, as :func:`everalgo.utils.format_time` renders them:
                ``UTC`` (the default) or local time with its offset, ``UTC+08:00``.

        Returns:
            The extracted or updated Profile. ``Profile.timestamp`` is the newest observation date seen so far and
            never moves backwards; every item carries ``observed_at``, the observation date of the narrative that
            last established its description. When the do-not-record list is not empty, or ``old_profile`` carried
            one, it is the extra field ``do_not_record``: a list of ``{"topic", "observed_at"}``, oldest first, an
            entry with parts since allowed also carrying ``"except"``, a list of the same shape. When the pending
            list is not empty, or ``old_profile`` carried one, it is the extra field ``pending``: a list of held items,
            each its item fields plus ``bucket`` and ``observed_at``, oldest first, at most 30 once an UPDATE has run.

        Raises:
            TypeError: If an item of ``episodes`` is not an :class:`Episode`, or ``categories`` is not a sequence
                of strings or contains a non-string value.
            InvalidInputError: If owner or Episode validation fails, ``old_profile`` belongs to another owner, or
                ``output_language`` names no supported language.
            LLMOutputError: If a reply is not a valid INIT, UPDATE, COMPACT or REGROUP output.
        """
        target_user = _resolve_episode_target(owner_id, owner_name)
        selected = _select_episodes(episodes, target_user=target_user)
        category_snapshot = _normalize_available_categories(categories)
        available_categories = _render_available_categories(category_snapshot)
        if old_profile is not None and old_profile.owner_id != owner_id:
            raise InvalidInputError(
                f"old_profile.owner_id {old_profile.owner_id!r} does not match owner_id {owner_id!r}: "
                "pass the stored profile of the same owner, or None to start a new one"
            )
        language = resolve_output_language(output_language)

        mode = "INIT" if old_profile is None else "UPDATE"
        stored_extra = (old_profile.model_extra or {}) if old_profile else {}
        # Copies, do_not_record in the current shape: the stages change them in place, and the caller's stored
        # profile stays as it was.
        lists = _EpisodeLists(
            do_not_record=load_matters(stored_extra.get("do_not_record")),
            pending=[dict(entry) for entry in stored_extra.get("pending") or []],
        )
        matters_before = len(lists.do_not_record)
        logger.info(
            "extracting profile from Episodes: mode=%s, %d/%d matching episodes observed at ms %d..%d, "
            "existing explicit=%d implicit=%d, output_language=%s, categories=%d, custom_instructions=%d chars",
            mode,
            len(selected),
            len(episodes),
            selected[0].timestamp,
            selected[-1].timestamp,
            len(old_profile.explicit_info) if old_profile else 0,
            len(old_profile.implicit_traits) if old_profile else 0,
            language,
            len(category_snapshot),
            len(custom_instructions or ""),
        )
        if old_profile is None:
            result = await self._init_extract_from_episodes(
                selected,
                owner_id=owner_id,
                target_user=target_user,
                available_categories=available_categories,
                custom_instructions=custom_instructions,
                output_language=language,
                tz=tz,
                lists=lists,
            )
        else:
            result = await self._update_extract_from_episodes(
                selected,
                owner_id=owner_id,
                target_user=target_user,
                old_profile=old_profile,
                available_categories=available_categories,
                custom_instructions=custom_instructions,
                output_language=language,
                tz=tz,
                lists=lists,
            )
        # Only a list this call left longer is merged: one that no safe merge shortened costs no call on every batch.
        if len(lists.do_not_record) > max(matters_before, DO_NOT_RECORD_MAX_ENTRIES):
            await self._merge_do_not_record(lists.do_not_record, target_user=target_user)
        # The stages build the Profile from its two buckets alone, so the lists are put back on it here. Only the
        # fields that were set are carried over, so an item field never set does not come back as an explicit None.
        extra: dict[str, Any] = {}
        # Both are written back even when emptied, so a host that keeps the stored field does not keep a withdrawn
        # matter or a confirmed entry.
        if lists.do_not_record or "do_not_record" in stored_extra:
            extra["do_not_record"] = lists.do_not_record
        if lists.pending or "pending" in stored_extra:
            extra["pending"] = lists.pending
        if extra:
            result = type(result).model_validate({**result.model_dump(exclude_unset=True), **extra})
        logger.info(
            "profile extracted from Episodes: mode=%s -> explicit=%d implicit=%d",
            mode,
            len(result.explicit_info),
            len(result.implicit_traits),
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
        custom_instructions: str | None,
        output_language: OutputLanguage | str | None,
        tz: tzinfo,
    ) -> Profile:
        messages = _MEMCELL_INIT.render(
            custom_instructions=custom_instructions,
            conversation_text=_render_conversation(memcells, tz=tz),
            target_user=_sender_display_name(memcells, sender_id),
            target_user_id=sender_id,
            language_rule=_MEMCELL_INIT.language_rule(output_language),
        )

        data = await _call_llm_for_profile_init(self._llm, messages, reply=_MEMCELL_INIT.reply)
        observed_at = memcells[-1].timestamp
        # Without new traits the reply's traits are not kept (the MemCell prompt still asks for them).
        traits = data["implicit_traits"] if self._new_traits else []
        return _profile_from_buckets(
            owner_id=sender_id,
            timestamp=observed_at,
            explicit_info=_dedupe(_stored_items("explicit_info", data["explicit_info"], observed_at), source="init"),
            implicit_traits=_dedupe(_stored_items("implicit_traits", traits, observed_at), source="init"),
        )

    async def _init_extract_from_episodes(
        self,
        episodes: Sequence[Episode],
        *,
        owner_id: str,
        target_user: str,
        available_categories: str,
        custom_instructions: str | None,
        output_language: OutputLanguage | str | None,
        tz: tzinfo,
        lists: _EpisodeLists,
    ) -> Profile:
        """Build the first profile from ``episodes``, every item dated by the newest of them.

        The reply's items pass the item checks first (``_pending.check_init_items``): an item whose quote is not
        found is dropped and an unrepeated pattern item goes onto ``lists.pending``. The reply's ``do_not_record``
        matters join ``lists.do_not_record`` when their quote is found (``_do_not_record.enter_matters``), each dated
        by the narrative holding it; without new traits the reply's ``implicit_traits`` are not kept.
        """
        messages = _EPISODE_INIT.render(
            custom_instructions=custom_instructions,
            episode_texts=_render_episodes(episodes, tz=tz),
            target_user=target_user,
            available_categories=available_categories,
            language_rule=_EPISODE_INIT.language_rule(output_language),
        )

        data = await _call_llm_for_profile_init(self._llm, messages, reply=_EPISODE_INIT.reply)
        data = check_init_items(data, episodes, lists.pending) if self._item_checks else unchecked_init_items(data)
        observed_at = episodes[-1].timestamp
        enter_matters(
            lists.do_not_record,
            data.get("do_not_record") or [],
            episodes,
            source="init",
            item_checks=self._item_checks,
            pending=lists.pending,
        )
        traits = data["implicit_traits"] if self._new_traits else []
        return _profile_from_buckets(
            owner_id=owner_id,
            timestamp=observed_at,
            explicit_info=_dedupe(
                _stored_items("explicit_info", data["explicit_info"], observed_at), source="episode-init"
            ),
            implicit_traits=_dedupe(_stored_items("implicit_traits", traits, observed_at), source="episode-init"),
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
        custom_instructions: str | None,
        output_language: OutputLanguage | str | None,
        tz: tzinfo,
    ) -> Profile:
        messages = _MEMCELL_UPDATE.render(
            custom_instructions=custom_instructions,
            stored_profile=_render_profile_for_update(old_profile, tz=tz),
            conversations=_render_conversation(memcells, tz=tz),
            target_user=_sender_display_name(memcells, sender_id),
            target_user_id=sender_id,
            language_rule=_MEMCELL_UPDATE.language_rule(output_language),
        )

        data = await _call_llm_for_profile_update(self._llm, messages, reply=_MEMCELL_UPDATE.reply)
        merged_profile = _apply_ops(
            old_profile,
            data["operations"],
            timestamp=memcells[-1].timestamp,
            pass_name="single",
            new_traits=self._new_traits,
        )

        return await self._maintain_updated_profile(
            merged_profile,
            display_name=_sender_display_name(memcells, sender_id),
            sender_id=sender_id,
            output_language=output_language,
            custom_instructions=custom_instructions,
            compact_prompts=_MEMCELL_COMPACT,
            regroup_prompts=_MEMCELL_REGROUP,
            available_categories=None,
            tz=tz,
        )

    async def _update_extract_from_episodes(
        self,
        episodes: Sequence[Episode],
        *,
        owner_id: str,
        target_user: str,
        old_profile: Profile,
        available_categories: str,
        custom_instructions: str | None,
        output_language: OutputLanguage | str | None,
        tz: tzinfo,
        lists: _EpisodeLists,
    ) -> Profile:
        """Run UPDATE in up to two passes: Episodes older than the stored profile first, then the rest.

        Each pass carries one observation date, its newest narrative, which ``_collect_indexed`` judges every
        rewrite against.
        """
        historical = [episode for episode in episodes if episode.timestamp < old_profile.timestamp]
        current = list(episodes[len(historical) :])
        if historical:
            logger.info(
                "profile update split around stored timestamp ms %d: %d historical, %d current narratives",
                old_profile.timestamp,
                len(historical),
                len(current),
            )
        profile = old_profile
        for pass_name, batch in (("historical", historical), ("current", current)):
            if batch:
                profile = await self._update_pass_from_episodes(
                    batch,
                    pass_name=pass_name,
                    owner_id=owner_id,
                    target_user=target_user,
                    old_profile=profile,
                    available_categories=available_categories,
                    custom_instructions=custom_instructions,
                    output_language=output_language,
                    tz=tz,
                    lists=lists,
                )
        return profile

    async def _update_pass_from_episodes(
        self,
        episodes: Sequence[Episode],
        *,
        pass_name: str,
        owner_id: str,
        target_user: str,
        old_profile: Profile,
        available_categories: str,
        custom_instructions: str | None,
        output_language: OutputLanguage | str | None,
        tz: tzinfo,
        lists: _EpisodeLists,
    ) -> Profile:
        """Run one UPDATE pass over ``episodes``, then COMPACT and REGROUP the merged profile as its caps require.

        The model sees ``lists.do_not_record`` as ``<do_not_record>`` and ``lists.pending``, expired for this pass,
        as ``<pending_patterns>``. The reply's operations pass the item checks (``_pending.check_update_ops``), which
        turn ``confirm`` into ``add`` and hold or release pattern items. A ``forget`` operation adds its matter to the
        list and drops the pending entries about it (``_do_not_record.enter_matters``); an ``allow`` takes a listed
        matter, or a part of one, off the list (``_do_not_record.withdraw_matters``). Both are taken out before the
        other operations are merged, since they name no stored item.
        """
        _expire_pending(lists.pending, episodes[-1].timestamp)
        messages = _EPISODE_UPDATE.render(
            custom_instructions=custom_instructions,
            do_not_record=_render_do_not_record(lists.do_not_record, tz=tz),
            pending=_render_pending(lists.pending, tz=tz),
            stored_profile=_render_profile_for_episode_update(old_profile, tz=tz),
            episode_texts=_render_episodes(episodes, tz=tz),
            target_user=target_user,
            available_categories=available_categories,
            language_rule=_EPISODE_UPDATE.language_rule(output_language),
        )

        data = await _call_llm_for_profile_update(self._llm, messages, reply=_EPISODE_UPDATE.reply)
        stored = dict(zip(("explicit_info", "implicit_traits"), _dated_buckets(old_profile), strict=True))
        # The lists as the model was shown them: the indices of its reply resolve against these.
        shown_pending = list(lists.pending)
        shown_matters = len(lists.do_not_record)
        operations = (
            check_update_ops(data["operations"], episodes, stored, lists.pending, tz=tz)
            if self._item_checks
            else unchecked_update_ops(data["operations"])
        )
        enter_matters(
            lists.do_not_record,
            [op for op in operations if op.get("action") == "forget"],
            episodes,
            source="update",
            item_checks=self._item_checks,
            pending=lists.pending,
            shown_pending=shown_pending,
        )
        # After the requests: a request repeated later than a withdrawal in this batch then refuses it.
        withdraw_matters(
            lists.do_not_record,
            [op for op in operations if op.get("action") == "allow"],
            episodes,
            item_checks=self._item_checks,
            shown=shown_matters,
        )
        merged_profile = _apply_ops(
            old_profile,
            [op for op in operations if op.get("action") not in ("forget", "allow")],
            timestamp=episodes[-1].timestamp,
            pass_name=pass_name,
            new_traits=self._new_traits,
        )
        return await self._maintain_updated_profile(
            merged_profile,
            display_name=target_user,
            sender_id=owner_id,
            output_language=output_language,
            custom_instructions=custom_instructions,
            compact_prompts=_EPISODE_COMPACT,
            regroup_prompts=_EPISODE_REGROUP,
            available_categories=available_categories,
            tz=tz,
        )

    async def _merge_do_not_record(self, do_not_record: list[dict[str, Any]], *, target_user: str) -> None:
        """Ask the model which entries of a do-not-record list past its cap are redundant, and remove those, in place.

        The model is shown the entries a merge may name (``_do_not_record.mergeable``) and replies with merges that
        each keep one entry word for word (``_do_not_record.apply_merge_ops``). With fewer than two such entries no
        call is made. A reply that is not a valid merge output leaves the list as it is: nothing is dropped for the
        cap, and a list still past it afterwards is logged.
        """
        shown = mergeable(do_not_record)
        operations: list[dict[str, Any]] = []
        if len(shown) > 1:
            messages = render_do_not_record_merge(
                do_not_record=_render_do_not_record(do_not_record, only=shown, dated=False),
                count=len(do_not_record),
                cap=DO_NOT_RECORD_MAX_ENTRIES,
                target_user=target_user,
            )
            try:
                operations = await _call_llm_for_do_not_record_merge(self._llm, messages)
            except LLMOutputError:
                logger.warning("profile do-not-record merge reply invalid: matters=%d", len(do_not_record))
        apply_merge_ops(do_not_record, operations)

    async def _maintain_updated_profile[CompactT: _AnyCompactOutput, RegroupT: _AnyRegroupOutput](
        self,
        merged_profile: Profile,
        *,
        display_name: str,
        sender_id: str,
        output_language: OutputLanguage | str | None,
        custom_instructions: str | None,
        compact_prompts: _StagePrompts[CompactT],
        regroup_prompts: _StagePrompts[RegroupT],
        available_categories: str | None = None,
        tz: tzinfo = UTC,
    ) -> Profile:
        """COMPACT the whole profile past the total cap, then REGROUP each group still past a per-label cap.

        The per-label check reads COMPACT's result when COMPACT ran, so a group COMPACT left over its cap is
        regrouped in the same call instead of waiting for the next UPDATE.
        """
        profile = merged_profile
        total_items = len(profile.explicit_info) + len(profile.implicit_traits)
        if total_items > _PROFILE_MAX_ITEMS:
            logger.warning(
                "profile total cap breached (%d > %d items), running full compact",
                total_items,
                _PROFILE_MAX_ITEMS,
            )
            profile = await self._compact(
                profile,
                display_name=display_name,
                sender_id=sender_id,
                output_language=output_language,
                custom_instructions=custom_instructions,
                prompts=compact_prompts,
                available_categories=available_categories,
                tz=tz,
            )
        return await self._regroup_overcrowded(
            profile,
            display_name=display_name,
            sender_id=sender_id,
            output_language=output_language,
            custom_instructions=custom_instructions,
            prompts=regroup_prompts,
            available_categories=available_categories,
            tz=tz,
        )

    async def _regroup_overcrowded[ReplyT: _AnyRegroupOutput](
        self,
        profile: Profile,
        *,
        display_name: str,
        sender_id: str,
        prompts: _StagePrompts[ReplyT],
        custom_instructions: str | None,
        available_categories: str | None = None,
        output_language: OutputLanguage | str | None = None,
        tz: tzinfo = UTC,
    ) -> Profile:
        """REGROUP, once each, every group past its per-label cap or holding an item past the width backstop."""
        result = profile
        for bucket, label in _overcrowded_labels(profile.explicit_info, profile.implicit_traits):
            result = await self._regroup(
                result,
                bucket=bucket,
                label=label,
                display_name=display_name,
                sender_id=sender_id,
                output_language=output_language,
                custom_instructions=custom_instructions,
                prompts=prompts,
                available_categories=available_categories,
                tz=tz,
            )
        return result

    async def _regroup[ReplyT: _AnyRegroupOutput](
        self,
        profile: Profile,
        *,
        bucket: str,
        label: str,
        display_name: str,
        sender_id: str,
        prompts: _StagePrompts[ReplyT],
        custom_instructions: str | None,
        available_categories: str | None = None,
        output_language: OutputLanguage | str | None = None,
        tz: tzinfo = UTC,
    ) -> Profile:
        """Reorganise one overcrowded group in place; every other item passes through untouched.

        The Episode path shows the group as indexed items and its reply is operations on them
        (:func:`apply_compact_ops`): an item no operation names keeps its host keys and ``observed_at``. The MemCell
        path's reply is the group rewritten, every item dated by the profile; a rewrite that still breaches the cap
        is accepted with a warning. Either way regroup never loops.
        """
        wanted = _normalize(label)
        buckets: dict[str, Sequence[_Item]] = {
            "explicit_info": profile.explicit_info,
            "implicit_traits": profile.implicit_traits,
        }
        group = [item for item in buckets[bucket] if _in_group(item, wanted)]
        rest = [item for item in buckets[bucket] if not _in_group(item, wanted)]
        other_labels = sorted({lbl for item in rest if (lbl := _item_label(item)) is not None})
        logger.warning(
            "profile group over cap, regrouping: bucket=%s label_len=%d group_items=%d "
            "(cap %d/label, %d-unit width backstop)",
            bucket,
            len(label),
            len(group),
            _PROFILE_MAX_PER_CATEGORY,
            _ITEM_WIDTH_BACKSTOP,
        )
        messages = prompts.render(
            custom_instructions=custom_instructions,
            bucket=bucket,
            reply_skeleton=REGROUP_REPLY_SKELETONS[bucket],
            label=label,
            label_field=_LABEL_FIELDS[bucket],
            count=len(group),
            max_per_category=_PROFILE_MAX_PER_CATEGORY,
            other_labels=", ".join(other_labels) if other_labels else "(none)",
            # An operations reply names items by index, so the group is shown indexed; a rewrite needs no index.
            items_text=_render_bucket(bucket, group, tz=tz)
            if prompts.reply is _RegroupOpsOutput
            else _render_group(group),
            target_user=display_name,
            target_user_id=sender_id,
            available_categories=available_categories or "[]",
            language_rule=prompts.language_rule(output_language),
        )
        data = await _call_llm_for_profile_regroup(self._llm, messages, reply=prompts.reply, bucket=bucket)
        if "operations" in data:
            # An item stored undated takes the profile's date, as the merge and the renderers date it.
            dated_group = [
                item if item.observed_at is not None else item.model_copy(update={"observed_at": profile.timestamp})
                for item in group
            ]
            new_group = apply_compact_ops({bucket: dated_group}, data["operations"], source="regroup")[bucket]
            buckets[bucket] = _dedupe([*rest, *new_group], source="regroup")
            still = sum(1 for item in buckets[bucket] if _in_group(item, wanted))
            if still > _PROFILE_MAX_PER_CATEGORY:
                logger.warning(
                    "profile regroup left label over cap: bucket=%s label_len=%d count=%d", bucket, len(label), still
                )
            logger.info(
                "profile regroup ops applied: bucket=%s label_len=%d group_before=%d group_after=%d",
                bucket,
                len(label),
                len(group),
                len(new_group),
            )
            return _profile_from_buckets(
                owner_id=profile.owner_id,
                timestamp=profile.timestamp,
                explicit_info=buckets["explicit_info"],
                implicit_traits=buckets["implicit_traits"],
            )
        returned: list[dict[str, Any]] = data["items"]
        kept = [item for item in returned if _identity_key(item["description"]) is not None]
        dropped_malformed = len(returned) - len(kept)
        if dropped_malformed:
            logger.warning(
                "profile regroup dropped %d malformed item(s) bucket=%s label_len=%d",
                dropped_malformed,
                bucket,
                len(label),
            )
        # A regrouped item has no traceable origin, so it takes the profile's own date.
        buckets[bucket] = _dedupe([*rest, *_stored_items(bucket, kept, profile.timestamp)], source="regroup")
        still = sum(1 for item in buckets[bucket] if _in_group(item, wanted))
        if still > _PROFILE_MAX_PER_CATEGORY:
            logger.warning(
                "profile regroup left label over cap: bucket=%s label_len=%d count=%d", bucket, len(label), still
            )
        logger.info(
            "profile regroup done: bucket=%s label_len=%d group %d -> %d items",
            bucket,
            len(label),
            len(group),
            len(kept),
        )
        return _profile_from_buckets(
            owner_id=profile.owner_id,
            timestamp=profile.timestamp,
            explicit_info=buckets["explicit_info"],
            implicit_traits=buckets["implicit_traits"],
        )

    async def _compact[ReplyT: _AnyCompactOutput](
        self,
        profile: Profile,
        *,
        display_name: str,
        sender_id: str,
        prompts: _StagePrompts[ReplyT],
        custom_instructions: str | None,
        available_categories: str | None = None,
        output_language: OutputLanguage | str | None = None,
        tz: tzinfo = UTC,
    ) -> Profile:
        """Bring the whole profile back inside its caps in one call.

        The Episode path shows both buckets as indexed items, as UPDATE does, and its reply is operations on them
        (:func:`apply_compact_ops`): an item no operation names keeps its host keys and ``observed_at``. The MemCell
        path's reply is the profile rewritten, every item dated by the profile.
        """
        total_items = len(profile.explicit_info) + len(profile.implicit_traits)
        messages = prompts.render(
            custom_instructions=custom_instructions,
            total_items=total_items,
            max_items=_PROFILE_MAX_ITEMS,
            max_per_category=_PROFILE_MAX_PER_CATEGORY,
            # An operations reply names items by index, so the profile is shown indexed; a rewrite needs no index.
            profile_text=_render_profile_for_episode_update(profile, tz=tz)
            if prompts.reply is _CompactOpsOutput
            else _render_profile_for_compact(profile),
            target_user=display_name,
            target_user_id=sender_id,
            available_categories=available_categories or "[]",
            language_rule=prompts.language_rule(output_language),
        )

        data = await _call_llm_for_profile_compact(self._llm, messages, reply=prompts.reply)
        if "operations" in data:
            # The same dated buckets the profile was rendered from, so every index resolves to the item shown.
            explicit_items, implicit_items = _dated_buckets(profile)
            new = apply_compact_ops(
                {"explicit_info": explicit_items, "implicit_traits": implicit_items},
                data["operations"],
                source="compact",
            )
            logger.info(
                "profile compact ops applied: items_before=%d items_after=%d",
                total_items,
                len(new["explicit_info"]) + len(new["implicit_traits"]),
            )
            return _profile_from_buckets(
                owner_id=profile.owner_id,
                timestamp=profile.timestamp,
                explicit_info=new["explicit_info"],
                implicit_traits=new["implicit_traits"],
            )
        # A rewritten item cannot be traced to the ones it came from, so it takes the profile's own date.
        new_explicit = _dedupe(
            _stored_items("explicit_info", data["explicit_info"], profile.timestamp), source="compact"
        )
        new_implicit = _dedupe(
            _stored_items("implicit_traits", data["implicit_traits"], profile.timestamp), source="compact"
        )
        logger.info(
            "profile compacted: %d -> %d items",
            total_items,
            len(new_explicit) + len(new_implicit),
        )
        return _profile_from_buckets(
            owner_id=profile.owner_id,
            timestamp=profile.timestamp,
            explicit_info=new_explicit,
            implicit_traits=new_implicit,
        )
