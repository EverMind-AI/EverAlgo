"""Synthesize a user Profile from chronological MemCells or dated Episodes, keeping the profile's timeline.

Design rationale and measurements: ``docs/concepts/profile-design-notes.md``. Modules, each importing only those
below it: ``_extractor`` (:class:`ProfileExtractor`) wires ``_stages`` (prompt templates per stage, over
``_replies``: reply models and LLM callsites), ``_render`` (what the model is shown), ``_compact_ops`` (COMPACT and
REGROUP operations, over ``_merge``), ``_pending`` (the Episode item checks: a quote for every item, pattern items
held on the ``pending`` list until repeated, over ``_merge``) and ``_merge`` (UPDATE ops, the account-number scrub,
dedupe, the stored profile); ``_render`` and ``_merge`` share only ``_items``.

Log lines:
    The lines below are a contract: each starts with ``profile <event>:`` followed by space-separated
    ``key=value`` pairs, and none carries a prompt, a model output or an item's content — only counts, dates
    (epoch ms), lengths and enum values. ``reason`` is the one value with spaces; it runs to the next
    `` key=`` or the end of the line. Every line is logged on ``everalgo.user_memory.profile``, whichever
    module writes it.

    ``profile update applied: pass= observed_at_ms= add= update= delete= none= rejected= rewrote_description=
    history_appended= grounding_only= canonicalised= profile_timestamp_ms=<old>-><new> explicit= implicit=`` (INFO)
        Once per UPDATE pass. ``pass`` is ``historical`` or ``current`` for the two halves of an Episode batch
        split around the stored timestamp, ``single`` for the MemCell path, which never splits.
        ``observed_at_ms`` is the date the pass's ops are judged against. ``add`` / ``update`` / ``delete`` /
        ``none`` count the reply's ops by action; ``rejected`` counts the ops refused, where the updates to an
        index the same reply deletes count once. Of the updates that land — several on one index merge into one —
        ``rewrote_description`` restate the fact, ``history_appended`` come from a pass older than their item and
        add earlier history in a parenthesis after its description (the item keeps its ``observed_at``), and
        ``grounding_only`` only add evidence or basis.
        ``canonicalised`` counts adds whose label was respelled to one already on file. ``explicit`` /
        ``implicit`` are the merged bucket sizes, before any COMPACT or REGROUP.
    ``profile update op rejected: action= type= index= reason=`` (WARNING)
        One per refused op. A stale rewrite or delete appends its criterion:
        ``reason=stale evidence item_observed_at_ms= batch_observed_at_ms=``.
    ``profile add label canonicalised: bucket= from_len= to_len=`` (DEBUG)
        One per respelled add label; lengths only, the label being model-written text.
    ``profile duplicates dropped: source= count=`` (WARNING)
        One per deduplicated list that lost items; ``source`` is ``init``, ``episode-init``, ``update``,
        ``compact`` or ``regroup``.
    ``profile item dropped: reason=description carries an account number`` (WARNING)
        One per model-written item left out because its description carries an identification or account number;
        an UPDATE ``update`` dropped this way also logs ``profile update op rejected: ... reason=description carries
        an account number``. An UPDATE add into implicit_traits is rejected with ``reason=new traits disabled``.
    ``profile update ops dropped: count= reason=item shaped for the other bucket`` (WARNING)
        Once per UPDATE reply holding adds or updates whose item is shaped for the other bucket; the reply's other
        operations still apply. A ``confirm`` whose item is shaped for the other bucket logs
        ``profile update ops dropped: count=1 reason=confirm item shaped for the other bucket``, once per such op.
    ``profile item checks: stage= quote_dropped= held= confirmed= pending=`` (INFO)
        Once per Episode INIT reply and UPDATE pass while the checks are on: ``stage`` is ``init`` or ``update``,
        ``quote_dropped`` the items dropped for a quote not found, ``held`` the pattern items put on the pending
        list, ``confirmed`` the entries written and taken off it, ``pending`` the list's size after it.
    ``profile pending expired: removed= pending=`` (INFO)
        Once per UPDATE pass that drops entries for age (over 120 days) or past the 30 newest, before the list is
        shown.
    ``profile compact ops applied: items_before= items_after=`` (INFO)
        Once per Episode-path COMPACT: the profile's item count before and after its operations.
    ``profile regroup ops applied: bucket= label_len= group_before= group_after=`` (INFO)
        Once per Episode-path REGROUP: the group's item count before and after its operations. A group still over
        its cap afterwards also logs ``profile regroup left label over cap: bucket= label_len= count=`` (WARNING),
        as a MemCell-path rewrite does.
    ``profile compact ops refused: count=`` / ``profile regroup ops refused: count=`` (WARNING)
        Once per Episode-path COMPACT or REGROUP reply with refused operations.
    ``profile do-not-record updated: source= named= added= matters=`` (INFO)
        Once per Episode INIT or UPDATE reply that names matters not to record: ``source`` is ``init`` or
        ``update``, ``named`` the matters the reply names, ``added`` those new to the list, ``matters`` the list's
        size after it.

    The Episode-path entry line, ``extracting profile from Episodes: ...``, carries ``categories=``, the size of
    the caller's category snapshot after normalisation. Both entry lines log ``output_language`` once it is
    validated, as the :class:`OutputLanguage` member's value or ``None``. The REGROUP lines name the group by
    ``bucket`` and ``label_len``, never by the label's text.
"""

from __future__ import annotations

__all__ = ["ProfileExtractor"]

from everalgo.user_memory.profile._extractor import ProfileExtractor
