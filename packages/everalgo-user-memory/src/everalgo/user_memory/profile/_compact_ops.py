"""Applying a COMPACT or REGROUP reply that is expressed as operations on indexed items.

The Episode path's COMPACT and REGROUP used to reply with the whole profile (or group) rewritten, and a rewritten
item cannot be traced to the stored one it came from: every item lost its host keys (ids, provenance) and took the
profile's own date, untouched items included. Operations name the items they change by the index the model was
shown, so everything else stays as stored.

Every item no operation names passes through as the same object, so its host keys and its ``observed_at`` survive.
A merge keeps the first named item's host keys and the newest ``observed_at`` of the items it merges; a split's
first piece that builds keeps the item's host keys and every piece keeps its date, the later pieces arriving without
host keys; an update keeps both; a move keeps the trait's date only, so the host gives the new explicit_info item an
id of that bucket. Every item an operation writes passes ``_merge._scrub``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from everalgo.user_memory.profile._items import _OBSERVED_AT, _Item
from everalgo.user_memory.profile._merge import _ITEM_MODELS, _dedupe, _identity_key, _scrub

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

# The package's name rather than this module's: hosts collect the log contract under it.
logger = logging.getLogger("everalgo.user_memory.profile")

# Keys the model writes; every other key on a stored item is the host's or the system's and rides through.
_MODEL_KEYS = frozenset({"evidence", "basis", "description", "category", "trait"})
# The keys an item of each bucket may carry: an operation whose item has another bucket's key lands nowhere.
_BUCKET_KEYS = {
    "explicit_info": frozenset({"evidence", "description", "category"}),
    "implicit_traits": frozenset({"basis", "description", "trait"}),
}


def _host_keys(item: _Item) -> dict[str, Any]:
    """The stored item's keys the model does not write: host extras and ``observed_at``."""
    return {k: v for k, v in item.model_dump(exclude_unset=True).items() if k not in _MODEL_KEYS}


def _build(bucket: str, data: Mapping[str, Any], keys: Mapping[str, Any]) -> _Item | None:
    """Build a stored item of ``bucket`` from the model's ``data`` and the stored ``keys`` it carries over.

    Args:
        bucket: The bucket the item lands in.
        data: The item as the operation writes it.
        keys: Host keys and ``observed_at`` taken from the stored item(s) the operation names.

    Returns:
        The item, or None when ``data`` has no description, holds a key ``bucket`` does not, or is dropped by
        ``_scrub`` for an account number in its description.
    """
    if _identity_key(data.get("description")) is None or not set(data) <= _BUCKET_KEYS[bucket]:
        return None
    scrubbed = _scrub(data)
    if scrubbed is None:
        return None
    return _ITEM_MODELS[bucket].model_validate({**scrubbed, **keys})


def apply_compact_ops(  # noqa: C901  — one branch per operation kind and refusal reason, algorithm-intrinsic
    buckets: Mapping[str, Sequence[_Item]], ops: list[dict[str, Any]], *, source: str
) -> dict[str, list[_Item]]:
    """Apply a COMPACT or REGROUP reply's operations to ``buckets`` and return the new buckets.

    ``merge`` replaces the items at ``indices`` with ``data`` in the first one's place; ``split`` replaces the item
    at ``index`` with the first of ``items`` and appends the rest; ``move`` turns the trait at ``index`` into the
    explicit_info item ``data``, appended to explicit_info with the trait's date and none of its host keys;
    ``update`` replaces the item at ``index`` with ``data``; ``delete`` removes it. Each index may be named by one
    operation only. An operation is refused, alone and with one warning counting every refusal, when its ``type`` is
    not one of ``buckets``, an index is out of range or already named by an earlier operation, or the item it writes
    cannot be built (see :func:`_build`); a split whose other pieces build keeps them.

    Args:
        buckets: The stored items of each bucket the reply may touch, each list in the order and with the dates it
            was rendered with, so every index resolves to the item the model saw.
        ops: The reply's operations, dumped with their ``type`` key.
        source: ``compact`` or ``regroup``, naming the stage in the log lines.

    Returns:
        Each bucket of ``buckets``: the kept and rewritten items in their original order, then the appended ones,
        deduplicated.
    """
    result: dict[str, list[_Item | None]] = {name: list(items) for name, items in buckets.items()}
    appended: dict[str, list[_Item]] = {name: [] for name in buckets}
    used: dict[str, set[int]] = {name: set() for name in buckets}
    refused = 0

    def _claim(bucket: str, indices: list[int]) -> bool:
        """Mark ``indices`` of ``bucket`` as named; False when the bucket is unknown or an index is unusable."""
        if bucket not in result or not indices:
            return False
        if any(not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < len(buckets[bucket]) for i in indices):
            return False
        if len(set(indices)) != len(indices) or used[bucket] & set(indices):
            return False
        used[bucket].update(indices)
        return True

    for op in ops:
        # The reply's raw values, checked by ``_claim`` before any of them is used as a bucket name or an index.
        action: Any = op.get("action")
        bucket: Any = op.get("type")
        index: Any = op.get("index")
        if action == "merge":
            indices = list(op.get("indices") or [])
            if len(indices) < 2 or not _claim(bucket, indices):
                refused += 1
                continue
            items = [buckets[bucket][i] for i in indices]
            # The merged item stands for all of them: the first one's host keys, the newest of their dates.
            keys = _host_keys(items[0])
            dates = [it.observed_at for it in items if it.observed_at is not None]
            if dates:
                keys[_OBSERVED_AT] = max(dates)
            merged = _build(bucket, op.get("data") or {}, keys)
            if merged is None:
                used[bucket].difference_update(indices)
                refused += 1
                continue
            result[bucket][indices[0]] = merged
            for i in indices[1:]:
                result[bucket][i] = None
        elif action == "split":
            pieces = list(op.get("items") or [])
            if not pieces or not _claim(bucket, [index]):
                refused += 1
                continue
            origin = buckets[bucket][index]
            keys = _host_keys(origin)
            date = {_OBSERVED_AT: origin.observed_at} if origin.observed_at is not None else {}
            # The first piece that builds carries on the item itself, host keys included, even when an earlier piece
            # was refused; the others are new items that keep only its date.
            built: list[_Item] = []
            for piece in pieces:
                piece_item = _build(bucket, piece, date if built else keys)
                if piece_item is not None:
                    built.append(piece_item)
            if not built:
                used[bucket].discard(index)
                refused += 1
                continue
            result[bucket][index] = built[0]
            appended[bucket].extend(built[1:])
        elif action == "move":
            if bucket != "implicit_traits" or "explicit_info" not in result or not _claim(bucket, [index]):
                refused += 1
                continue
            # The trait becomes a new explicit_info item, appended to that bucket, that keeps the trait's date and none
            # of its host keys: a host's ids are minted per bucket (an implicit_traits id cannot address an
            # explicit_info item), so the host gives the moved item a fresh one, as it did after a rewrite COMPACT.
            origin = buckets[bucket][index]
            date = {_OBSERVED_AT: origin.observed_at} if origin.observed_at is not None else {}
            moved = _build("explicit_info", op.get("data") or {}, date)
            if moved is None:
                used[bucket].discard(index)
                refused += 1
                continue
            result[bucket][index] = None
            appended["explicit_info"].append(moved)
        elif action == "update":
            if not _claim(bucket, [index]):
                refused += 1
                continue
            updated = _build(bucket, op.get("data") or {}, _host_keys(buckets[bucket][index]))
            if updated is None:
                used[bucket].discard(index)
                refused += 1
                continue
            result[bucket][index] = updated
        elif action == "delete":
            if not _claim(bucket, [index]):
                refused += 1
                continue
            result[bucket][index] = None
        else:  # an action this stage does not take
            refused += 1
    if refused:
        logger.warning("profile %s ops refused: count=%d", source, refused)
    return {
        name: _dedupe([it for it in items if it is not None] + appended[name], source=source)
        for name, items in result.items()
    }
