"""Episode-path item checks on the model's one reply: a quote for every item, and pattern items held until repeated.

Quote check. Every item a reply writes — each INIT item, each UPDATE add or ``confirm``, and each update that changes
a description — carries ``quote``: words copied from one narrative of the batch. The program looks for the quote in
the batch's narratives (:func:`_find_quote`); an item whose quote is missing or not found is dropped, and a dropped
update leaves the stored item as it was.

Pending patterns. An item that claims a standing pattern of this person (:func:`_is_pattern`: every
``implicit_traits`` item, a habit, hobby or emotion category, or ``"pattern": true`` from the model) is written at
once only when one of its found quotes shows regularity (:func:`_shows_regularity`: every week, usually, always,
fixed ...). Any other pattern item waits on the profile's extra field ``pending``. Every later UPDATE shows the entries to the
model as ``<pending_patterns>``; when a batch shows the same pattern again the model replies with a ``confirm``
operation naming the entry (an add pointing at one with ``"pending"`` or restating one nearly word for word counts
too), the item is written and the entry leaves the list. Entries older than :data:`PENDING_MAX_AGE_MS` are dropped,
and at most :data:`PENDING_MAX_ENTRIES` are kept, the newest.

Each pending entry is the held item's model-facing fields (``evidence``, ``description``, ``category`` or
``basis``, ``description``, ``trait``) plus ``bucket`` and ``observed_at``, the newest narrative of the batch that
held it. Entries pass :func:`_merge._scrub` like stored items: one whose description carries an account number is
not kept, and such a number in its grounding is masked.

Languages. The checks work on narratives in any language. Text is compared in NFKC form, casefolded, with punctuation
and spacing taken out, and lengths are counted in Han-character units (:func:`_size`). Regularity is read with a
word list for a Chinese quote (:data:`_REGULAR_ZH`) and for an English one (:data:`_REGULAR_EN`). The model also
copies the words that show it out of its quote into ``regular`` and the program checks that they are there: an English
quote the English list misses, or a quote in any other language, is read this way; a Chinese one never is.

Neither check calls the model again. The reply-only fields ``pattern``, ``quote``, ``regular`` and ``pending`` never
reach a stored item (``_merge._REPLY_ONLY_FIELDS``).
"""

from __future__ import annotations

import difflib
import logging
import re
import unicodedata
from datetime import UTC
from typing import TYPE_CHECKING, Any

from everalgo.user_memory.profile._merge import _REPLY_ONLY_FIELDS, _identity_key, _scrub
from everalgo.utils import format_date

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from datetime import tzinfo

    from everalgo.types import Episode
    from everalgo.user_memory.profile._items import _Item

# The package's name rather than this module's: hosts collect the log contract under it.
logger = logging.getLogger("everalgo.user_memory.profile")

PENDING_MAX_ENTRIES = 30
PENDING_MAX_AGE_MS = 120 * 86_400_000
# Character-bigram overlap from which a proposed pattern restates a pending entry.
_SAME_PATTERN = 0.6
# A quote shorter than this, once punctuation and spacing are taken out, is too short to find (Han-character units:
# six Chinese characters, about eighteen letters).
_MIN_QUOTE_CHARS = 6
# A quote counts as found when one narrative holds a common run covering this share of it.
_NEAR_QUOTE = 0.9
# Pieces of a quote split by an ellipsis are each looked for when at least this long (Han-character units).
_MIN_PIECE_CHARS = 4
# The model's regular words count when at least this long (Han-character units).
_MIN_REGULAR_SIZE = 1
_ELLIPSIS = re.compile(r"…+|\.{3,}|⋯+")

# Categories whose items are patterns whatever the model's flag says, matched as substrings of the lowercased label.
_PATTERN_CATEGORIES = ("行为习惯", "兴趣爱好", "情绪特征", "habit", "hobby", "interest", "emotion")

# Words in a Chinese quote that show the pattern is regular or standing: every day / week / time, a count of times,
# fixed, without fail, often, always, usually, for years, from now on, by default ... A weekday name alone (this
# Saturday), 一次 (once) and 非常 (very) do not count; a run of weekdays with a time of day (一三五早上) does. 以后 counts
# only as "from now on": after 从今 / 从此 / 今, or followed within a few characters by 都 / 就 / 一律 / 每 ... (以后回复都用
# 中文), not as "later" or "after" (以后再说, 毕业以后). A stated span followed by 都 (最近三个月都在练) counts; the span
# alone (最近三个月去过一次) does not. Matched on the quote in NFKC form, so a comma or semicolon in either width ends a
# 一周…次 span, a 以后 lookahead and a span's reach to its 都.
_REGULAR_ZH = re.compile(
    r"每[天日周月年晚早个]|每次|每逢|天天|一周[^。\uff0c\uff1b,;]{0,6}次|[两二三四五六七八九十几]+次|固定|雷打不动|照例|照常|"
    r"(?<![非正日异反照无经])常|经常|总是|总会|一直|习惯|一般|平时|平常|通常|向来|多年|长期|坚持|从不|从来|今后|默认|一律|"
    r"(?:从今|从此|今)以后|以后(?=[^。\uff0c\uff1b,;]{0,4}?(?:都|就|一律|统一|每|一直|总|全|别|不要|不再|尽量))|"
    r"(?:最近|近|这|过去)(?:[一二两三四五六七八九十几半]|\d)+个?多?(?:天|周|星期|礼拜|月|年)(?:来|以来)?[^。\uff0c\uff1b,;]{0,3}?都|"
    r"[一二三四五六日](?:[、和]?[一二三四五六日])+(?:早|晚|上|下|都)"
)
_EN_COUNT = (
    r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|several|a\s+few|few|"
    r"(?:a\s+)?couple\s+of|many)"
)
_EN_UNIT = (
    r"(?:days?|mornings?|afternoons?|evenings?|nights?|weeks?|weekends?|weekdays?|workdays?|months?|years?|times?|"
    r"summers?|winters?|springs?|autumns?|falls?|holidays?|mondays?|tuesdays?|wednesdays?|thursdays?|fridays?|"
    r"saturdays?|sundays?)"
)
# Words in an English quote that show the pattern is regular or standing, the counterpart of the Chinese list: every
# day / week / other Sunday, daily, a count a week, twice, a weekday in the plural (on Wednesdays), usually, always,
# often, most of the time, never, habit, routine, for years, ever since, from now on, by default, set in stone ...
# A weekday alone (this Saturday), once, every so often, whenever possible and "three times better" do not count;
# "used to" does only as "is used to". Matched on the quote in NFKC form, casefolded, with spacing collapsed.
_REGULAR_EN = re.compile(
    # every day / each week / every other Sunday / every two weeks / every single time
    r"\b(?:every|each)\s+(?:other\s+|single\s+|" + _EN_COUNT + r"\s+)?" + _EN_UNIT + r"\b|\beveryday\b|"
    r"\b(?:daily|nightly|weekly|fortnightly|biweekly|monthly|quarterly|yearly|annually)\b|"
    # once a week, three times a month, five days a week, two hours a day
    r"\b(?:once|twice|thrice|" + _EN_COUNT + r"\s+(?:times|days|nights|mornings|evenings|hours?|minutes?|km|"
    r"kilometers?|kilometres?|miles?|laps?|cups?|sessions?))\s+(?:a|an|per|each|every)\s+"
    r"(?:day|night|week|weekend|month|year)\b|\bper\s+(?:day|night|week|month|year)\b|"
    # twice / three times (a count of times; once does not count)
    r"\btwice\b|\b(?:two|three|four|five|six|seven|eight|nine|ten|several|many|countless|\d+)\s+times\b"
    r"(?!\s+(?:as|more|less|better|worse|faster|slower|bigger|larger|higher|lower|longer|smaller|cheaper|over))|"
    # on Mondays, Sunday evenings, on weekends, on workdays (a weekday alone, this Saturday, does not count)
    r"\b(?:mondays|tuesdays|wednesdays|thursdays|fridays|saturdays|sundays)\b|\bon\s+(?:weekends|weekdays|workdays)\b|"
    r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+(?:mornings|afternoons|evenings|nights)\b|"
    r"\bmost\s+(?:days|mornings|afternoons|evenings|nights|weekends|weeks|weekdays|workdays)\b|"
    r"\bmost\s+of\s+the\s+time\b|"
    # usually, always, often, never ...
    r"\b(?:usually|normally|typically|generally|always|often|frequently|regularly|routinely|habitually|customarily|"
    r"constantly|consistently|invariably|religiously|never)\b(?<!every\sso\soften)|"
    r"\bwhenever\b(?!\s+(?:possible|needed|necessary))|\btend(?:s|ed)?\s+to\b|\bas\s+(?:a\s+rule|usual|always|ever)\b|"
    r"\bwithout\s+(?:fail|exception)\b|\bno\s+matter\s+what\b|\ball\s+the\s+time\b|\bon\s+a\s+regular\s+basis\b|"
    r"\bmakes?\s+a\s+point\s+of\b|\bset\s+in\s+stone\b|"
    # habit, routine, used to it, sticks to it, keeps it up
    r"\b(?:habit|habits|habitual|routine|routines|ritual|rituals)\b|"
    r"\b(?:is|am|are|was|were|be|been|get|gets|got|gotten|become|becomes|became)\s+used\s+to\b|"
    r"\b(?:stick|sticks|stuck|sticking)\s+(?:to|with)\b|\b(?:keeps?|kept|keeping)\s+(?:it\s+)?up\b|"
    r"\binsists?\s+on\b|"
    # for years, for almost a year, long-term, all along, ever since
    r"\bfor\s+(?:(?:many|several|a\s+few|over|more\s+than|nearly|almost|about|some|\d+|\w+)\s+){0,3}"
    r"(?:years|decades)\b|\bfor\s+(?:almost|nearly|over|more\s+than)\s+a\s+year\b|\bfor\s+a\s+long\s+time\b|"
    r"\byears\s+of\b|\blong[-\s]?(?:term|standing|time)\b|\ball\s+along\b|\bever\s+since\b|"
    r"\bsince\s+(?:childhood|college|university|high\s+school|school|he\s+was|she\s+was|they\s+were|i\s+was)\b|"
    # from now on, by default
    r"\bfrom\s+now\s+on\b|\bgoing\s+forward\b|\bhenceforth\b|\bin\s+(?:the\s+)?future\b|\bfrom\s+then\s+on\b|"
    r"\bby\s+default\b|\bfixed\s+(?:time|times|schedule|day|days|hour|hours|routine)\b"
)
# Common English words, one of which marks an alphabetic quote as English (:func:`_is_english`).
_EN_WORDS = frozenset(
    (
        *("the", "a", "an", "and", "or", "but", "not", "no", "of", "to", "in", "on", "at", "for", "with", "from"),
        *("by", "about", "after", "before", "than", "then", "when", "while", "if", "so", "as", "up", "out", "off"),
        *("he", "she", "his", "her", "him", "they", "their", "them", "it", "its", "i", "my", "me", "you", "your"),
        *("we", "our", "that", "this", "these", "those", "what", "who", "which", "all", "every", "each", "more"),
        *("is", "was", "were", "are", "be", "been", "has", "had", "have", "do", "does", "did", "will", "would"),
        *("can", "could", "should", "must", "may", "might", "very", "just", "only", "still", "also", "again"),
    )
)
_DATE = re.compile(r"(20\d\d)[-/.年](\d{1,2})[-/.月](\d{1,2})")


def _is_pattern(bucket: str, item: Mapping[str, Any]) -> bool:
    """True when ``item`` claims a standing pattern: a trait, a habit-like category, or the model's ``pattern`` flag.

    The model leaves out of patterns the one item summing up how this person wants the assistant to work and what
    they say matters most to them in life; the prompt's ``<item_checks>`` block says so.
    """
    if bucket == "implicit_traits":
        return True
    category = (item.get("category") or "").lower()
    if any(label in category for label in _PATTERN_CATEGORIES):
        return True
    return item.get("pattern") is True


def _without_reply_fields(item: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in item.items() if k not in _REPLY_ONLY_FIELDS}


def _grounding(item: Mapping[str, Any]) -> str:
    return item.get("evidence") or item.get("basis") or ""


def _bigrams(text: str) -> set[str]:
    folded = _fold(text)
    return {folded[i : i + 2] for i in range(len(folded) - 1)} or {folded}


def _overlap(a: str, b: str) -> float:
    """Shared character bigrams of ``a`` and ``b`` over the smaller set."""
    x, y = _bigrams(a), _bigrams(b)
    return len(x & y) / (min(len(x), len(y)) or 1)


def _day(ms: int, *, tz: tzinfo) -> str:
    return format_date(ms, tz=tz)


def _dates_in(text: str) -> set[str]:
    """The ``YYYY-MM-DD`` dates written in ``text`` (2026-03-02, 2026/3/2, 2026年3月2日)."""
    return {f"{y}-{int(m):02d}-{int(d):02d}" for y, m, d in _DATE.findall(text or "")}


def _batch_days(episodes: Sequence[Episode], *, tz: tzinfo) -> set[str]:
    return {_day(episode.timestamp, tz=tz) for episode in episodes}


# ---------------------------------------------------------------- quotes


def _fold(text: str) -> str:
    """``text`` in NFKC form, casefolded, with everything but letters and digits taken out."""
    return "".join(ch for ch in unicodedata.normalize("NFKC", text or "").casefold() if ch.isalnum())


def _is_wide(ch: str) -> bool:
    """True for a character of a script written without spaces, one character a syllable or more (Han, kana, Hangul)."""
    o = ord(ch)
    return (
        0x3040 <= o <= 0x30FF
        or 0x3400 <= o <= 0x4DBF
        or 0x4E00 <= o <= 0x9FFF
        or 0xF900 <= o <= 0xFAFF
        or 0xAC00 <= o <= 0xD7AF
        or 0x1100 <= o <= 0x11FF
        or 0x3130 <= o <= 0x318F
        or o >= 0x20000
    )


def _is_han(ch: str) -> bool:
    """True for a Han character."""
    o = ord(ch)
    return 0x3400 <= o <= 0x4DBF or 0x4E00 <= o <= 0x9FFF or 0xF900 <= o <= 0xFAFF or o >= 0x20000


def _size(folded: str) -> float:
    """The length of folded text in Han-character units: a wide character counts one, any other letter or digit a third.

    Six Chinese characters and about eighteen letters of an alphabetic script carry roughly the same words.
    """
    wide = sum(1 for ch in folded if _is_wide(ch))
    return wide + (len(folded) - wide) / 3


def _quotes(item: Mapping[str, Any]) -> list[str]:
    """The item's quotes: ``quote`` as one string or a list of strings, blanks left out."""
    quote = item.get("quote")
    if isinstance(quote, str):
        quote = [quote]
    if not isinstance(quote, list):
        return []
    return [q for q in quote if isinstance(q, str) and q.strip()]


def _find_quote(quote: str, folded_narratives: Sequence[str]) -> int | None:
    """Index of the narrative holding ``quote``, else None.

    Punctuation, spacing, case and character width are ignored (:func:`_fold`). A quote is found when a narrative
    holds it whole, holds each of its pieces around an ellipsis, or holds a common run covering 90% of it. A quote
    shorter than six Han-character units (:func:`_size`: six Chinese characters, about eighteen letters) is never found.
    """
    folded = _fold(quote)
    if _size(folded) < _MIN_QUOTE_CHARS:
        return None
    for i, narrative in enumerate(folded_narratives):
        if folded in narrative:
            return i
    pieces = [_fold(piece) for piece in _ELLIPSIS.split(quote) if _size(_fold(piece)) >= _MIN_PIECE_CHARS]
    if len(pieces) > 1:
        for i, narrative in enumerate(folded_narratives):
            if all(piece in narrative for piece in pieces):
                return i
    for i, narrative in enumerate(folded_narratives):
        match = difflib.SequenceMatcher(None, folded, narrative, autojunk=False).find_longest_match(
            0, len(folded), 0, len(narrative)
        )
        if match.size >= _NEAR_QUOTE * len(folded):
            return i
    return None


def _quote_found(item: Mapping[str, Any], folded_narratives: Sequence[str]) -> bool:
    return any(_find_quote(q, folded_narratives) is not None for q in _quotes(item))


def _mostly_han(folded: str) -> bool:
    """True for Chinese: folded text without kana or Hangul, half of it or more Han characters (:func:`_size` units)."""
    if any(_is_wide(ch) and not _is_han(ch) for ch in folded):
        return False
    han = sum(1 for ch in folded if _is_han(ch))
    return han > 0 and han >= _size(folded) / 2


def _is_english(text: str) -> bool:
    """True for English: no kana or Hangul, nine in ten other letters or more ASCII, and one common English word.

    Han characters (a Chinese name in an English sentence) are left out of the count; a quote mostly in Han characters
    is read as Chinese before this (:func:`_mostly_han`).
    """
    norm = unicodedata.normalize("NFKC", text or "").casefold()
    if any(_is_wide(ch) and not _is_han(ch) for ch in norm):
        return False
    letters = [ch for ch in norm if ch.isalpha() and not _is_han(ch)]
    if not letters:
        return False
    if sum(1 for ch in letters if ch.isascii()) < 0.9 * len(letters):
        return False
    return any(word in _EN_WORDS for word in re.findall(r"[a-z]+", norm))


def _shows_regularity(item: Mapping[str, Any], folded_narratives: Sequence[str]) -> bool:
    """True when one of the item's found quotes shows the pattern is regular or standing.

    A quote written mostly in Han characters (Chinese) is read with the Chinese word list (:data:`_REGULAR_ZH`) only,
    as evaluated: the model's marks are looser in Chinese (it takes 基本, "mostly", for regular). An English quote
    (:func:`_is_english`) shows regularity when the English list (:data:`_REGULAR_EN`) finds a word in it, else as a
    quote in any other language does: through the model, whose ``regular`` words, copied out of its quote (still
    using, jeden Morgen, 毎週 ...), count when they stand in that found quote; words shorter than one Han character or
    three letters (:func:`_size`) do not. The list alone held back habits only the model saw (still using the same old
    camera, took turns) and wrote no fewer wrong items, so English keeps the model as the fallback.
    """
    words = item.get("regular")
    folded_words = _fold(words) if isinstance(words, str) else ""
    for q in _quotes(item):
        if _find_quote(q, folded_narratives) is None:
            continue
        folded_quote = _fold(q)
        if _mostly_han(folded_quote):
            if _REGULAR_ZH.search(unicodedata.normalize("NFKC", q)):
                return True
        elif _english_shows_regularity(q) or (
            _size(folded_words) >= _MIN_REGULAR_SIZE and folded_words in folded_quote
        ):
            return True
    return False


def _english_shows_regularity(quote: str) -> bool:
    """True for an English quote (:func:`_is_english`) in which the English word list finds a word."""
    spaced = " ".join(unicodedata.normalize("NFKC", quote).casefold().split())
    return _is_english(quote) and _REGULAR_EN.search(spaced) is not None


# ---------------------------------------------------------------- the pending list


def _expire_pending(pending: list[dict[str, Any]], now: int) -> None:
    """Drop, in place, entries older than :data:`PENDING_MAX_AGE_MS` at ``now``, then all but the newest 30.

    Run before the list is shown to the model, so the indices the model names are the ones kept.
    """
    before = len(pending)
    kept = [entry for entry in pending if now - entry["observed_at"] <= PENDING_MAX_AGE_MS]
    pending[:] = kept[-PENDING_MAX_ENTRIES:]
    if len(pending) != before:
        logger.info("profile pending expired: removed=%d pending=%d", before - len(pending), len(pending))


def _hold(bucket: str, item: Mapping[str, Any], observed_at: int, pending: list[dict[str, Any]]) -> bool:
    """Put ``item`` on the pending list unless an entry already states it; True when it was added."""
    description = item.get("description") or ""
    if any(_overlap(description, entry["description"]) >= _SAME_PATTERN for entry in pending):
        return False
    fields = _scrub(_without_reply_fields(item))
    if fields is None:
        return False
    pending.append({**fields, "bucket": bucket, "observed_at": observed_at})
    return True


def _match_pending(
    item: Mapping[str, Any], pending: Sequence[Mapping[str, Any]], episodes: Sequence[Episode], *, tz: tzinfo
) -> int | None:
    """The pending entry a proposed pattern item repeats, else None.

    The model's own pointer (``"pending": k``) counts when the item's grounding names a date of this batch or no date
    at all. Otherwise a proposal whose description shares most bigrams with an entry's repeats it, unless its
    grounding names only dates from outside this batch (then it copies the entry and shows no new occasion).
    """
    grounding = _grounding(item)
    dates = _dates_in(grounding)
    days = _batch_days(episodes, tz=tz)
    k = item.get("pending")
    if isinstance(k, int) and not isinstance(k, bool) and 0 <= k < len(pending) and (not dates or dates & days):
        return k
    if not pending or (dates and not dates & days):
        return None
    description = item.get("description") or ""
    best = max(range(len(pending)), key=lambda i: _overlap(description, pending[i]["description"]))
    return best if _overlap(description, pending[best]["description"]) >= _SAME_PATTERN else None


# ---------------------------------------------------------------- the two stages


def check_init_items(
    data: dict[str, Any], episodes: Sequence[Episode], pending: list[dict[str, Any]]
) -> dict[str, Any]:
    """An Episode INIT reply with its items checked: unquoted items dropped, unrepeated pattern items held.

    Args:
        data: The validated INIT reply.
        episodes: The batch the reply was written from; the newest dates the held entries.
        pending: The call's pending list, appended to in place.

    Returns:
        A copy of ``data`` whose ``explicit_info`` and ``implicit_traits`` hold the items to write.
    """
    folded = [_fold(episode.episode) for episode in episodes]
    observed_at = episodes[-1].timestamp
    out = dict(data)
    dropped = held = 0
    for bucket in ("explicit_info", "implicit_traits"):
        kept = []
        for item in data.get(bucket) or []:
            if not _quote_found(item, folded):
                dropped += 1
                continue
            if _is_pattern(bucket, item) and not _shows_regularity(item, folded):
                held += _hold(bucket, item, observed_at, pending)
                continue
            kept.append(_without_reply_fields(item))
        out[bucket] = kept
    _log("init", dropped=dropped, held=held, confirmed=0, pending=pending)
    return out


def unchecked_init_items(data: dict[str, Any]) -> dict[str, Any]:
    """An Episode INIT reply with the checks off: every item kept as written, its reply-only fields dropped."""
    out = dict(data)
    for bucket in ("explicit_info", "implicit_traits"):
        out[bucket] = [_without_reply_fields(item) for item in data.get(bucket) or []]
    return out


def check_update_ops(
    ops: Sequence[Mapping[str, Any]],
    episodes: Sequence[Episode],
    stored: Mapping[str, Sequence[_Item]],
    pending: list[dict[str, Any]],
    *,
    tz: tzinfo = UTC,
) -> list[dict[str, Any]]:
    """An Episode UPDATE reply's operations with their items checked, ``confirm`` turned into ``add``.

    A ``confirm`` becomes an add of its item into its ``type`` (``explicit_info`` when it names none) pointing at its
    entry. An add or a description-changing update whose quote is not found is dropped. A pattern add that repeats a
    pending entry (:func:`_match_pending`) is written and the entry removed; a second add for the same entry is
    dropped; any other pattern add is written only when its quote shows regularity, else held. Delete, forget and
    none pass through.

    Args:
        ops: The validated reply's operations.
        episodes: The pass's batch; the newest dates the held entries.
        stored: The pass's stored items by bucket, as the prompt indexed them.
        pending: The call's pending list, already expired for this pass and shown to the model, changed in place.
        tz: The zone the prompt showed the narratives' times in; a grounding's dates are read as days in it.

    Returns:
        The operations to merge; ``[{"action": "none"}]`` when none is left.
    """
    folded = [_fold(episode.episode) for episode in episodes]
    observed_at = episodes[-1].timestamp
    out: list[dict[str, Any]] = []
    confirmed: set[int] = set()
    dropped = held = 0
    for reply_op in ops:
        op = _as_add(reply_op) if reply_op.get("action") == "confirm" else reply_op
        if op is None:
            continue
        action = op.get("action")
        if action not in ("add", "update"):
            out.append(dict(op))
            continue
        bucket: Any = op.get("type")
        item = op.get("data") or {}
        if _needs_quote(op, stored) and not _quote_found(item, folded):
            dropped += 1
            continue
        if action == "add" and _is_pattern(bucket, item):
            verdict = _pattern_add(item, pending, episodes, folded, confirmed, tz=tz)
            if verdict == "hold":
                held += _hold(bucket, item, observed_at, pending)
            if verdict != "write":
                continue
        out.append({**op, "data": _without_reply_fields(item)})
    if confirmed:
        pending[:] = [entry for i, entry in enumerate(pending) if i not in confirmed]
    _log("update", dropped=dropped, held=held, confirmed=len(confirmed), pending=pending)
    return out or [{"action": "none"}]


def unchecked_update_ops(ops: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The operations with the checks off: ``confirm`` read as an add, reply-only fields dropped, nothing held."""
    out = []
    for reply_op in ops:
        op = _as_add(reply_op) if reply_op.get("action") == "confirm" else dict(reply_op)
        if op is not None and op.get("action") in ("add", "update"):
            op = {**op, "data": _without_reply_fields(op.get("data") or {})}
        if op is not None:
            out.append(op)
    return out or [{"action": "none"}]


def _pattern_add(
    item: Mapping[str, Any],
    pending: Sequence[Mapping[str, Any]],
    episodes: Sequence[Episode],
    folded: Sequence[str],
    confirmed: set[int],
    *,
    tz: tzinfo,
) -> str:
    """What becomes of a pattern add: ``write``, ``hold`` or ``repeat`` (a second proposal for an entry written).

    One that repeats a pending entry is written and its entry recorded in ``confirmed``; any other is written when
    its quote shows regularity, else held.
    """
    k = _match_pending(item, pending, episodes, tz=tz)
    if k is not None and k in confirmed:
        return "repeat"
    if k is not None:
        confirmed.add(k)
        return "write"
    return "write" if _shows_regularity(item, folded) else "hold"


def _as_add(op: Mapping[str, Any]) -> dict[str, Any] | None:
    """A ``confirm`` as the add it stands for, pointing at its entry; None, with a warning, when misshaped."""
    data = {**(op.get("data") or {}), "pending": op.get("pending"), "pattern": True}
    bucket = op.get("type") or "explicit_info"
    if ("category" in data) != (bucket == "explicit_info"):
        logger.warning("profile update ops dropped: count=1 reason=confirm item shaped for the other bucket")
        return None
    return {"action": "add", "type": bucket, "data": data}


def _needs_quote(op: Mapping[str, Any], stored: Mapping[str, Sequence[_Item]]) -> bool:
    """True for an add, and for an update that changes its stored item's description (or names no stored item)."""
    if op.get("action") == "add":
        return True
    bucket: Any = op.get("type")
    items = stored.get(bucket) or []
    index = op.get("index")
    old = items[index] if isinstance(index, int) and 0 <= index < len(items) else None
    return old is None or _identity_key((op.get("data") or {}).get("description")) != _identity_key(old.description)


def _log(stage: str, *, dropped: int, held: int, confirmed: int, pending: list[dict[str, Any]]) -> None:
    logger.info(
        "profile item checks: stage=%s quote_dropped=%d held=%d confirmed=%d pending=%d",
        stage,
        dropped,
        held,
        confirmed,
        len(pending),
    )
