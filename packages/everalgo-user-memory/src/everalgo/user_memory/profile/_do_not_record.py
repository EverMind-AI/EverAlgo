"""The do-not-record list of the Episode path: how a matter enters it, leaves it, and how the list is kept in its cap.

The list is the profile's extra field ``do_not_record``: the matters this person asked not to be recorded, oldest
first. Each entry is ``{"topic", "observed_at"}`` — the matter in a few words and the date of the narrative that holds
the request — plus ``"except"``, a list of ``{"topic", "observed_at"}``, when this person has since allowed parts of
it. No grounding is kept: an entry read from a profile stored with ``evidence`` loses it (:func:`load_matters`).

Entering. INIT's ``do_not_record`` entries and UPDATE's ``forget`` operations name matters, each with ``quote``: the
words of the request copied from one narrative of the batch. A matter enters only when the program finds its quote
(``_pending._find_quote``), and is dated by the narrative holding it. A topic already listed is not added again; its
date moves to the newer request. Pending entries about an entered matter are dropped (:func:`enter_matters`).

Leaving. An UPDATE ``allow`` operation names a listed matter by its index and carries the quote of the withdrawal.
It is accepted only when its quote is found in a narrative later than the entry's date, so an older narrative arriving
late never lifts a newer request. A whole withdrawal removes the entry and leaves no trace; one naming a ``part`` adds
that part to the entry's ``except`` and keeps the rest listed (:func:`withdraw_matters`). Entries never expire.

The cap. Past :data:`DO_NOT_RECORD_MAX_ENTRIES` entries the extractor asks the model, in a call of its own, which
entries are redundant: the same matter as another entry in other words, or wholly inside a wider entry that is itself
listed. A merge keeps one entry word for word and removes the redundant ones (:func:`apply_merge_ops`); no entry is
reworded and none is written, so the list can never keep out more than this person asked. An entry no safe merge
removes stays, however long the list: nothing is dropped for the cap.

With the item checks off (``ProfileExtractor(item_checks=False)``) no quote is looked for: a matter enters as the
reply names it and a withdrawal counts as made at the batch's newest narrative, still refused unless that is later than
the request.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from everalgo.user_memory.profile._merge import _mask_account_numbers
from everalgo.user_memory.profile._pending import _find_quote, _fold, _quotes, _size

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from everalgo.types import Episode

# The package's name rather than this module's: hosts collect the log contract under it.
logger = logging.getLogger("everalgo.user_memory.profile")

# Past this many entries the list is checked for redundant ones (the pending list's cap is the same number). A matter
# is a few words, so thirty of them stay a small block of the UPDATE prompt; people ask for a handful at most, and the
# cap is there for the profile that somehow gathers far more.
DO_NOT_RECORD_MAX_ENTRIES = 30

_EXCEPT = "except"
# A topic shorter than this (Han-character units, ``_pending._size``) is not looked for in pending descriptions: one
# or two letters would be found in entries that have nothing to do with it.
_MIN_TOPIC_SIZE = 2
# The two ways an entry is redundant beside another; a merge of any other kind is refused.
_MERGE_RELATIONS = ("same", "inside")


def _key(topic: str) -> str:
    """What two topics are compared by: the folded text (``_pending._fold``), or the stripped text when that is empty."""
    return _fold(topic) or topic.strip()


def _as_index(value: Any, size: int) -> int | None:
    """``value`` as an index below ``size``: an int, or a string of digits as models now and then write one."""
    if isinstance(value, str) and value.strip().isdecimal():
        value = int(value.strip())
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if 0 <= value < size else None


def load_matters(stored: Any) -> list[dict[str, Any]]:
    """The stored ``do_not_record`` field as the list of this call: a copy, each entry in the current shape.

    An entry keeps its topic, its date (0 when it has none) and its exceptions; any other key, the ``evidence`` of a
    profile stored before this shape among them, is left behind. Account numbers in a topic are masked, as they are
    when a matter enters, so one stored before topics were masked is masked from this call on. An entry without a
    topic is skipped.
    """
    matters: list[dict[str, Any]] = []
    for entry in stored or []:
        if not isinstance(entry, dict) or not isinstance(entry.get("topic"), str) or not entry["topic"].strip():
            continue
        matter: dict[str, Any] = {"topic": _mask_account_numbers(entry["topic"]), "observed_at": _date(entry)}
        allowed = [
            {"topic": _mask_account_numbers(part["topic"]), "observed_at": _date(part)}
            for part in entry.get(_EXCEPT) or []
            if isinstance(part, dict) and isinstance(part.get("topic"), str) and part["topic"].strip()
        ]
        if allowed:
            matter[_EXCEPT] = allowed
        matters.append(matter)
    return matters


def _date(entry: Mapping[str, Any]) -> int:
    value = entry.get("observed_at")
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _quote_time(
    op: Mapping[str, Any], episodes: Sequence[Episode], folded: Sequence[str], *, item_checks: bool
) -> int | None:
    """The date of the narrative holding one of ``op``'s quotes; None when none is found.

    With the item checks off no quote is looked for and the date is the batch's newest narrative.
    """
    if not item_checks:
        return episodes[-1].timestamp
    for quote in _quotes(op):
        found = _find_quote(quote, folded)
        if found is not None:
            return episodes[found].timestamp
    return None


def enter_matters(
    matters: list[dict[str, Any]],
    named: Iterable[Mapping[str, Any]],
    episodes: Sequence[Episode],
    *,
    source: str,
    item_checks: bool,
    pending: list[dict[str, Any]] | None = None,
    shown_pending: Sequence[dict[str, Any]] = (),
) -> int:
    """Add the matters a reply names to the do-not-record list in place and return how many entries were appended.

    A matter whose quote is not found in the batch is refused. Account numbers in a topic are masked as in an item's
    grounding (``_merge._mask_account_numbers``), and topics are compared folded. A topic already listed, or named
    earlier in the same reply, is not added again: the entry's date moves to the newer request. A topic that an entry
    lists as an exception takes that exception away when the request is newer than it, and is left out when it is
    older. Any other matter is appended, dated by the narrative holding its quote.

    Pending entries about a matter that entered or was already listed are dropped: those the reply names in the
    matter's ``pending`` list, by their index in ``shown_pending``, and those whose description holds the topic.

    Args:
        matters: The list of the call in progress.
        named: The reply's matters, each with ``topic`` and ``quote``: INIT's ``do_not_record`` entries or UPDATE's
            ``forget`` operations.
        episodes: The batch the reply was written from.
        source: ``init`` or ``update``, naming the stage in the log line.
        item_checks: Whether the quotes are looked for (``ProfileExtractor(item_checks=...)``).
        pending: The call's pending list, changed in place; None leaves it alone.
        shown_pending: The pending entries as the model was shown them, the list a ``pending`` index resolves in.
    """
    named = list(named)
    if not named:
        return 0
    folded = [_fold(episode.episode) for episode in episodes]
    added = refreshed = refused = 0
    listed: list[str] = []
    drop: list[dict[str, Any]] = []
    for matter in named:
        topic = _mask_account_numbers(str(matter.get("topic") or "")).strip()
        when = _quote_time(matter, episodes, folded, item_checks=item_checks) if topic else None
        if when is None:
            refused += 1
            continue
        key = _key(topic)
        outcome = _enter(matters, topic, key, when)
        if outcome == "superseded":
            refused += 1
            continue
        added += outcome == "added"
        refreshed += outcome == "refreshed"
        listed.append(key)
        drop.extend(_named_pending(matter, shown_pending))
    dropped = _drop_pending(pending, drop, listed) if pending is not None else 0
    logger.info(
        "profile do-not-record updated: source=%s named=%d added=%d refreshed=%d refused=%d pending_dropped=%d "
        "matters=%d",
        source,
        len(named),
        added,
        refreshed,
        refused,
        dropped,
        len(matters),
    )
    return added


def _enter(matters: list[dict[str, Any]], topic: str, key: str, when: int) -> str:
    """Put one matter on the list: ``added``, ``refreshed`` (already covered, dates moved) or ``superseded``."""
    for entry in matters:
        if _key(entry["topic"]) == key:
            entry["observed_at"] = max(entry["observed_at"], when)
            return "refreshed"
    for entry in matters:
        for part in entry.get(_EXCEPT) or []:
            if _key(part["topic"]) != key:
                continue
            if when <= part["observed_at"]:
                # The allowance is the newer word of the two: the older request does not come back.
                return "superseded"
            remaining = [other for other in entry[_EXCEPT] if other is not part]
            if remaining:
                entry[_EXCEPT] = remaining
            else:
                del entry[_EXCEPT]
            # The entry now stands for this request too, so a withdrawal must be later than it.
            entry["observed_at"] = max(entry["observed_at"], when)
            return "refreshed"
    matters.append({"topic": topic, "observed_at": when})
    return "added"


def _named_pending(matter: Mapping[str, Any], shown_pending: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The shown pending entries a ``forget`` names in ``pending``: one index or a list of them, bad ones ignored."""
    named = matter.get("pending")
    values = named if isinstance(named, list) else [named]
    indices = (_as_index(value, len(shown_pending)) for value in values)
    return [shown_pending[index] for index in indices if index is not None]


def _drop_pending(pending: list[dict[str, Any]], named: Sequence[dict[str, Any]], topics: Sequence[str]) -> int:
    """Drop, in place, the pending entries in ``named`` and those whose description holds one of ``topics``.

    A topic shorter than :data:`_MIN_TOPIC_SIZE` is not looked for.
    """
    named_ids = {id(entry) for entry in named}
    topics = [topic for topic in topics if _size(topic) >= _MIN_TOPIC_SIZE]
    kept = [
        entry
        for entry in pending
        if id(entry) not in named_ids and not any(topic in _fold(entry.get("description") or "") for topic in topics)
    ]
    dropped = len(pending) - len(kept)
    pending[:] = kept
    return dropped


def withdraw_matters(
    matters: list[dict[str, Any]],
    ops: Iterable[Mapping[str, Any]],
    episodes: Sequence[Episode],
    *,
    item_checks: bool,
    shown: int,
) -> None:
    """Apply a reply's ``allow`` operations to the do-not-record list in place.

    Each names a matter by its index in the list as the model was shown it. One is refused when the index names no
    shown entry, when its quote is not found in the batch, or when the narrative holding the quote is not later than
    the entry's date. An accepted one removes the entry; with a ``part`` that is not the whole topic it adds the part
    to the entry's exceptions instead, dated by the withdrawal, and the rest stays listed.

    Run after :func:`enter_matters` of the same reply, which only appends and moves dates: a request the same batch
    repeats later than the withdrawal has then moved the entry's date past it, and the withdrawal is refused.

    Args:
        matters: The list of the call in progress; its first ``shown`` entries are the ones the model was shown.
        ops: The reply's ``allow`` operations.
        episodes: The pass's batch.
        item_checks: Whether the quotes are looked for (``ProfileExtractor(item_checks=...)``).
        shown: How many entries the model of this pass was shown.
    """
    ops = list(ops)
    if not ops:
        return
    folded = [_fold(episode.episode) for episode in episodes]
    removed: set[int] = set()
    whole = part_count = refused = 0
    for op in ops:
        index = _as_index(op.get("matter"), min(shown, len(matters)))
        when = _quote_time(op, episodes, folded, item_checks=item_checks) if index is not None else None
        if index is None or when is None or when <= matters[index]["observed_at"]:
            refused += 1
            continue
        entry = matters[index]
        raw = op.get("part")
        part = _mask_account_numbers(raw).strip() if isinstance(raw, str) else ""
        if not part or _key(part) == _key(entry["topic"]):
            whole += index not in removed
            removed.add(index)
            continue
        allowed = entry.setdefault(_EXCEPT, [])
        same = next((other for other in allowed if _key(other["topic"]) == _key(part)), None)
        if same is None:
            allowed.append({"topic": part, "observed_at": when})
        else:
            same["observed_at"] = max(same["observed_at"], when)
        part_count += 1
    if removed:
        matters[:] = [entry for index, entry in enumerate(matters) if index not in removed]
    logger.info(
        "profile do-not-record withdrawn: named=%d whole=%d part=%d refused=%d matters=%d",
        len(ops),
        whole,
        part_count,
        refused,
        len(matters),
    )


def mergeable(matters: Sequence[Mapping[str, Any]]) -> list[int]:
    """The indices of the entries a merge may name: those without exceptions.

    An entry with exceptions keeps out less than its topic says. Kept, it would not cover a narrower entry that falls
    inside an exception; dropped, its exceptions would be lost and the wider entry would keep out what this person
    allowed. Such entries take no part in a merge and are not shown to the model.
    """
    return [index for index, entry in enumerate(matters) if not entry.get(_EXCEPT)]


def apply_merge_ops(matters: list[dict[str, Any]], ops: Iterable[Mapping[str, Any]]) -> None:
    """Apply a merge reply to the do-not-record list in place: redundant entries removed, every other one untouched.

    Each operation names the entry to ``keep``, the entries to ``drop`` as redundant beside it and their
    ``relation``: ``same`` (the same matter in other words) or ``inside`` (wholly inside the kept, wider entry). The
    kept entry stays word for word and takes the newest date among itself and the dropped ones, so that a withdrawal
    must still be later than every request the entry now stands for. An operation is refused whole when its relation
    is neither of the two, an index names no mergeable entry (:func:`mergeable`), ``drop`` is empty or holds ``keep``,
    or an index already appeared in an earlier operation of the reply.

    Args:
        matters: The list of the call in progress, as shown to the model.
        ops: The reply's operations, each ``{"keep": int, "drop": [int, ...], "relation": "same" | "inside"}``.
    """
    allowed = set(mergeable(matters))
    used: set[int] = set()
    removed: set[int] = set()
    applied = refused = 0
    for op in ops:
        named = [op.get("keep"), *(op.get("drop") or [])]
        indices = [index for index in named if isinstance(index, int) and not isinstance(index, bool)]
        if (
            op.get("relation") not in _MERGE_RELATIONS
            or len(indices) < 2
            or len(indices) != len(named)
            or len(set(indices)) != len(indices)
            or not set(indices) <= allowed
            or used & set(indices)
        ):
            refused += 1
            continue
        used |= set(indices)
        removed |= set(indices[1:])
        matters[indices[0]]["observed_at"] = max(matters[index]["observed_at"] for index in indices)
        applied += 1
    before = len(matters)
    if removed:
        matters[:] = [entry for index, entry in enumerate(matters) if index not in removed]
    logger.warning(
        "profile do-not-record merged: matters_before=%d merges=%d refused=%d removed=%d matters=%d cap=%d",
        before,
        applied,
        refused,
        len(removed),
        len(matters),
        DO_NOT_RECORD_MAX_ENTRIES,
    )
    if len(matters) > DO_NOT_RECORD_MAX_ENTRIES:
        logger.warning("profile do-not-record over cap: matters=%d cap=%d", len(matters), DO_NOT_RECORD_MAX_ENTRIES)
