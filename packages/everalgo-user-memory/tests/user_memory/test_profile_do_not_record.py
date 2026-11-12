"""The do-not-record list: matters this person asked not to be recorded, kept on the Episode-path profile.

INIT lists them in its reply's ``do_not_record``; UPDATE names one with a ``forget`` operation, each with the quoted
request. A matter whose quote the program finds is kept on the returned Profile as the extra field ``do_not_record``
(``{"topic", "observed_at"}``, plus ``"except"`` for the parts since allowed), the host passes it back with the stored
profile, and every later UPDATE shows the entries to the model in ``<do_not_record>``. An ``allow`` operation
quoting a withdrawal later than the request takes a matter off the list, whole or in part. Past its cap the list gets
one merge call that removes redundant entries and rewords none. Each topic enters once, account numbers in a topic are
masked, a topic's closing tags are defused, and pending entries about an entered matter are dropped.

The item checks are off here (``ProfileExtractor(item_checks=False)``) unless a test is about the quotes: with them off
no quote is looked for and a request or withdrawal is dated by the batch's newest narrative.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import timedelta, timezone
from typing import Any
from xml.etree import ElementTree

import pytest

from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import ChatMessage, Episode, MemCell, Profile
from everalgo.user_memory import ProfileExtractor
from everalgo.user_memory.profile import _do_not_record
from everalgo.user_memory.profile._replies import _EpisodeInitOutput, _MatterMergeOutput, _UpdateOutput
from everalgo.user_memory.prompts.en.profile_from_episode_texts import (
    PROFILE_DO_NOT_RECORD_MERGE_SYSTEM_PROMPT,
    PROFILE_UPDATE_FROM_EPISODE_TEXTS_SYSTEM_PROMPT,
)
from everalgo.utils import format_time

_TS = 1_780_000_000_000
_DAY = 86_400_000
_PROFILE_LOGGER = "everalgo.user_memory.profile"
_MARKER = "MATTER-MARKER-3c9d"
_ID_CARD = "42050219591120321X"
_BANK_CARD = "6222 0212 3456 7890"
_LOG_PAIR = re.compile(r"(\w+)=(.*?)(?= \w+=|$)")
_CAP = _do_not_record.DO_NOT_RECORD_MAX_ENTRIES


def _log_fields(caplog: pytest.LogCaptureFixture, prefix: str) -> list[dict[str, str]]:
    """The ``key=value`` pairs of every captured line opening with ``prefix``, as the package docstring defines them."""
    return [
        dict(_LOG_PAIR.findall(record.getMessage().removeprefix(prefix).strip()))
        for record in caplog.records
        if record.getMessage().startswith(prefix)
    ]


def _episode(text: str, timestamp: int) -> Episode:
    return Episode(owner_id=None, episode=f"Mara {text}", summary="s", timestamp=timestamp)


def _init_reply(*matters: dict[str, Any]) -> str:
    return json.dumps(
        {
            "explicit_info": [
                {"evidence": "said she lives in Ghent", "description": "Lives in Ghent.", "category": "Identity"},
                {"evidence": "said her mother is ill", "description": "Mother has diabetes.", "category": "Family"},
            ],
            "implicit_traits": [],
            "do_not_record": list(matters),
        }
    )


def _operations(*operations: dict[str, Any]) -> str:
    return json.dumps({"operations": list(operations)})


def _forget(topic: str, quote: str = "asked not to remember it", **more: Any) -> dict[str, Any]:
    return {"action": "forget", "topic": topic, "quote": quote, **more}


def _allow(matter: Any, quote: str = "said it may be recorded now", **more: Any) -> dict[str, Any]:
    return {"action": "allow", "matter": matter, "quote": quote, **more}


def _merges(*operations: dict[str, Any]) -> str:
    return json.dumps({"operations": list(operations)})


def _do_not_record_of(profile: Profile) -> list[dict[str, Any]] | None:
    return (profile.model_extra or {}).get("do_not_record")


def _stored(*matters: dict[str, Any], timestamp: int = _TS, **extra: Any) -> Profile:
    fields: dict[str, Any] = {"owner_id": "u1", "summary": "s", "timestamp": timestamp, **extra}
    if matters:
        fields["do_not_record"] = list(matters)
    return Profile.model_validate(fields)


def _matter(topic: str, observed_at: int = _TS, *, allowed: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """A stored entry; ``allowed`` is its ``except`` list, the parts this person has since allowed."""
    entry: dict[str, Any] = {"topic": topic, "observed_at": observed_at}
    if allowed is not None:
        entry["except"] = allowed
    return entry


def _shown(index: int, topic: str, since: int = _TS) -> str:
    """One entry as ``<do_not_record>`` shows it to UPDATE: its index, the time of the request, the topic."""
    return f'<matter index="{index}" since="{format_time(since)}">{topic}</matter>'


def _matters_shown(fake: FakeLLMClient, call: int) -> str:
    """The ``<do_not_record>`` block of an UPDATE call's user message."""
    user = str(fake.calls[call].messages[1].content)
    return user.split("<do_not_record>\n", 1)[1].split("\n</do_not_record>", 1)[0]


async def _update(fake: FakeLLMClient, stored: Profile, *episodes: Episode) -> Profile:
    """One UPDATE with the item checks off."""
    return await ProfileExtractor(llm=fake, item_checks=False).aextract_from_episodes(
        list(episodes), owner_id="u1", owner_name="Mara", old_profile=stored
    )


async def _checked_update(fake: FakeLLMClient, stored: Profile, *episodes: Episode) -> Profile:
    """One UPDATE with the item checks on, so every quote is looked for."""
    return await ProfileExtractor(llm=fake).aextract_from_episodes(
        list(episodes), owner_id="u1", owner_name="Mara", old_profile=stored
    )


# ==========================================================================
# Entering the list
# ==========================================================================


async def test_init_keeps_the_matters_its_reply_lists_dated_by_the_newest_narrative() -> None:
    matter = {"topic": "her mother's health", "quote": "asked not to note it"}
    fake = FakeLLMClient(responses=[_init_reply(matter, dict(matter))])

    profile = await ProfileExtractor(item_checks=False, llm=fake).aextract_from_episodes(
        [_episode("talked about home.", _TS), _episode("asked not to note her mother's health.", _TS + _DAY)],
        owner_id="u1",
        owner_name="Mara",
    )

    # One entry per topic, and nothing kept but the topic and its date.
    assert _do_not_record_of(profile) == [_matter("her mother's health", _TS + _DAY)]
    system = str(fake.calls[0].messages[0].content)
    assert '"do_not_record": [{"topic": "...", "quote": "...", "evidence": null}]' in system


async def test_an_empty_list_adds_no_field() -> None:
    profile = await ProfileExtractor(
        item_checks=False, llm=FakeLLMClient(responses=[_init_reply()])
    ).aextract_from_episodes([_episode("talked about home.", _TS)], owner_id="u1", owner_name="Mara")

    assert "do_not_record" not in (profile.model_extra or {})
    assert "do_not_record" not in profile.model_dump()


async def test_a_forget_operation_is_kept_on_the_profile_and_shown_in_the_next_update() -> None:
    fake = FakeLLMClient(
        responses=[
            _init_reply(),
            _operations(
                _forget("her mother's health"),
                {"action": "delete", "type": "explicit_info", "index": 1, "reason": "asked not to record"},
            ),
            _operations({"action": "none"}),
        ]
    )
    extractor = ProfileExtractor(item_checks=False, llm=fake)

    first = await extractor.aextract_from_episodes(
        [_episode("talked about home.", _TS)], owner_id="u1", owner_name="Mara"
    )
    second = await extractor.aextract_from_episodes(
        [_episode("asked not to remember her mother's health.", _TS + _DAY)],
        owner_id="u1",
        owner_name="Mara",
        old_profile=first,
    )
    # The host stores the profile and passes it back: the list rides along as an extra field.
    stored = Profile.model_validate(json.loads(second.model_dump_json()))
    third = await extractor.aextract_from_episodes(
        [_episode("mentioned a hospital visit.", _TS + 2 * _DAY)], owner_id="u1", owner_name="Mara", old_profile=stored
    )

    assert _matters_shown(fake, 1) == "(none)"
    assert [item.description for item in second.explicit_info] == ["Lives in Ghent."]
    expected = [_matter("her mother's health", _TS + _DAY)]
    assert _do_not_record_of(second) == expected
    assert _matters_shown(fake, 2) == _shown(0, "her mother's health", _TS + _DAY)
    assert _do_not_record_of(third) == expected


async def test_a_matter_already_listed_is_not_added_again_and_takes_the_newer_date() -> None:
    stored = _stored(
        # As a profile stored before this shape holds it: the evidence is left behind when the list is read.
        {"topic": "her salary", "evidence": "said keep it out", "observed_at": _TS},
        explicit_info=[{"evidence": "e", "description": "Lives in Ghent.", "category": "Identity"}],
    )
    fake = FakeLLMClient(responses=[_operations(_forget("Her  salary.", "said it again"), _forget("her diet"))])

    profile = await _update(fake, stored, _episode("asked again to keep her salary and her diet out.", _TS + _DAY))

    assert _matters_shown(fake, 0) == _shown(0, "her salary")
    # The same topic in another casing and spacing is the listed one; a withdrawal must now be later than this request.
    assert _do_not_record_of(profile) == [_matter("her salary", _TS + _DAY), _matter("her diet", _TS + _DAY)]
    assert _do_not_record_of(stored) == [{"topic": "her salary", "evidence": "said keep it out", "observed_at": _TS}]


async def test_a_matter_named_in_the_older_pass_is_shown_to_the_newer_pass() -> None:
    stored = _stored(timestamp=_TS + 5 * _DAY)
    fake = FakeLLMClient(responses=[_operations(_forget("her divorce")), _operations({"action": "none"})])

    profile = await _update(
        fake, stored, _episode("asked not to note her divorce.", _TS), _episode("talked about work.", _TS + 9 * _DAY)
    )

    assert fake.call_count == 2  # historical pass, then current pass
    assert _matters_shown(fake, 0) == "(none)"
    assert _matters_shown(fake, 1) == _shown(0, "her divorce")
    assert _do_not_record_of(profile) == [_matter("her divorce", _TS)]


async def test_a_forget_operation_is_not_merged_and_no_log_line_carries_the_matter(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake = FakeLLMClient(responses=[_operations(_forget(f"her {_MARKER}", f"said {_MARKER}"))])

    with caplog.at_level(logging.DEBUG, logger=_PROFILE_LOGGER):
        await _update(fake, _stored(), _episode("asked to forget something.", _TS + _DAY))

    # The line follows the package's log contract: an event, then key=value pairs of counts.
    assert _log_fields(caplog, "profile do-not-record updated:") == [
        {
            "source": "update",
            "named": "1",
            "added": "1",
            "refreshed": "0",
            "refused": "0",
            "pending_dropped": "0",
            "matters": "1",
        }
    ]
    assert "profile update op rejected" not in caplog.text
    assert "rejected=0" in caplog.text
    assert _MARKER not in caplog.text


async def test_the_memcell_path_keeps_no_list_and_refuses_a_forget_operation(caplog: pytest.LogCaptureFixture) -> None:
    """Only the Episode prompts ask for ``forget``; on the MemCell path one parses and is refused by the merge."""
    message = ChatMessage(id="m1", role="user", content="hi", timestamp=_TS, sender_id="u_mara", sender_name="Mara")
    stored = Profile.model_validate({"owner_id": "u_mara", "summary": "s", "timestamp": _TS})
    forget = {"action": "forget", "topic": "her diet", "evidence": "asked not to remember it"}
    reply = json.dumps({"operations": [forget], "update_note": "n"})

    with caplog.at_level(logging.WARNING, logger=_PROFILE_LOGGER):
        profile = await ProfileExtractor(item_checks=False, llm=FakeLLMClient(responses=[reply])).aextract(
            [MemCell(items=[message], timestamp=_TS)], sender_id="u_mara", old_profile=stored
        )

    assert _do_not_record_of(profile) is None
    assert "action=forget type=None index=None reason=unknown type" in caplog.text


async def test_a_topic_named_twice_in_one_reply_is_added_once(caplog: pytest.LogCaptureFixture) -> None:
    """Within one UPDATE reply, as across replies, a topic enters the list once; so it does within one INIT reply."""
    fake = FakeLLMClient(
        responses=[
            _operations(_forget("her diet"), _forget("her diet", "said it again"), _forget("her salary")),
            _init_reply({"topic": "her divorce", "quote": "first"}, {"topic": "her divorce", "quote": "second"}),
        ]
    )
    extractor = ProfileExtractor(item_checks=False, llm=fake)

    with caplog.at_level(logging.INFO, logger=_PROFILE_LOGGER):
        updated = await extractor.aextract_from_episodes(
            [_episode("asked twice to keep her diet out.", _TS + _DAY)],
            owner_id="u1",
            owner_name="Mara",
            old_profile=_stored(),
        )
        initial = await extractor.aextract_from_episodes(
            [_episode("asked to keep her divorce out.", _TS)], owner_id="u2", owner_name="Mara"
        )

    assert [entry["topic"] for entry in _do_not_record_of(updated) or []] == ["her diet", "her salary"]
    assert [entry["topic"] for entry in _do_not_record_of(initial) or []] == ["her divorce"]
    counts = [
        {key: line[key] for key in ("source", "named", "added", "refreshed", "matters")}
        for line in _log_fields(caplog, "profile do-not-record updated:")
    ]
    assert counts == [
        {"source": "update", "named": "3", "added": "2", "refreshed": "1", "matters": "2"},
        {"source": "init", "named": "2", "added": "1", "refreshed": "1", "matters": "1"},
    ]


async def test_account_numbers_in_a_topic_are_masked_like_item_grounding() -> None:
    """The topic is kept, masked; a masked topic already listed is not added again."""
    fake = FakeLLMClient(
        responses=[
            _init_reply({"topic": f"her ID card {_ID_CARD}", "quote": f"said never keep ID {_ID_CARD}"}),
            _operations(
                _forget(f"her card {_BANK_CARD}", f"said forget card {_BANK_CARD}"),
                _forget("her card 6222 0212 3456 7891", "said it again"),
            ),
            _operations({"action": "none"}),
        ]
    )
    extractor = ProfileExtractor(item_checks=False, llm=fake)

    initial = await extractor.aextract_from_episodes(
        [_episode("asked to keep her ID out.", _TS)], owner_id="u1", owner_name="Mara"
    )
    updated = await extractor.aextract_from_episodes(
        [_episode("asked to keep her card out.", _TS + _DAY)], owner_id="u1", owner_name="Mara", old_profile=initial
    )
    await extractor.aextract_from_episodes(
        [_episode("paid by card.", _TS + 2 * _DAY)], owner_id="u1", owner_name="Mara", old_profile=updated
    )

    assert _do_not_record_of(updated) == [_matter("her ID card ***", _TS), _matter("her card ***", _TS + _DAY)]
    assert _matters_shown(fake, 2) == (_shown(0, "her ID card ***") + "\n" + _shown(1, "her card ***", _TS + _DAY))
    assert _ID_CARD not in updated.model_dump_json()


async def test_reattaching_the_list_leaves_unset_item_fields_unset() -> None:
    """A stored trait without basis keeps no basis key, and the next UPDATE does not show it as null."""
    stored = _stored(implicit_traits=[{"description": "Plans ahead.", "trait": "Planner"}])
    fake = FakeLLMClient(responses=[_operations(_forget("her diet")), _operations({"action": "none"})])
    extractor = ProfileExtractor(item_checks=False, llm=fake)

    updated = await extractor.aextract_from_episodes(
        [_episode("asked to keep her diet out.", _TS + _DAY)], owner_id="u1", owner_name="Mara", old_profile=stored
    )
    await extractor.aextract_from_episodes(
        [_episode("talked about work.", _TS + 2 * _DAY)], owner_id="u1", owner_name="Mara", old_profile=updated
    )

    assert _do_not_record_of(updated) is not None
    assert updated.implicit_traits[0].model_dump(exclude_unset=True) == {
        "description": "Plans ahead.",
        "trait": "Planner",
        "observed_at": _TS,
    }
    assert "null" not in str(fake.calls[1].messages[1].content)


async def test_a_failed_call_leaves_the_callers_stored_lists_as_they_were() -> None:
    """The stages change copies of the stored lists, so a call that fails midway changes nothing the caller holds."""

    class _ClientError(Exception):
        pass

    matters = [{"topic": "her divorce", "evidence": "asked not to record it", "observed_at": _TS}]
    entries = [
        {
            "description": "Swims.",
            "category": "hobby",
            "evidence": "swam",
            "bucket": "explicit_info",
            "observed_at": _TS,
        }
    ]
    stored = Profile.model_validate(
        {"owner_id": "u1", "summary": "s", "timestamp": _TS, "do_not_record": matters, "pending": entries}
    )
    before = stored.model_dump()
    fake = FakeLLMClient(responses=[_ClientError("upstream down")])

    with pytest.raises(_ClientError):
        await _update(fake, stored, _episode("talked about work.", _TS + _DAY))

    assert stored.model_dump() == before


async def test_a_closing_tag_in_a_topic_is_defused_as_in_items_and_narratives() -> None:
    """Only the ``<`` turns fullwidth, so neither <matter>, <except> nor <do_not_record> closes early."""
    stored = _stored(
        _matter(
            "her </matter></DO_NOT_RECORD> notes",
            allowed=[_matter("her </except></matter> list", _TS)],
        )
    )
    fake = FakeLLMClient(responses=[_operations({"action": "none"})])

    await _update(fake, stored, _episode("typed </do_not_record> into the chat.", _TS + _DAY))

    user = str(fake.calls[0].messages[1].content)
    assert _matters_shown(fake, 0) == (
        _shown(
            0,
            "her \uff1c/matter>\uff1c/DO_NOT_RECORD> notes<except>her \uff1c/except>\uff1c/matter> list</except>",
        )
    )
    assert "typed \uff1c/do_not_record> into the chat." in user
    root = ElementTree.fromstring(f"<root>{user}</root>")
    blocks = ["target_person", "do_not_record", "stored_profile", "episode_narratives", "pending_patterns"]
    assert [element.tag for element in root] == blocks


async def test_the_request_time_is_shown_in_the_calls_time_zone() -> None:
    eight = timezone(timedelta(hours=8))
    fake = FakeLLMClient(responses=[_operations({"action": "none"})])

    await ProfileExtractor(llm=fake, item_checks=False).aextract_from_episodes(
        [_episode("talked about work.", _TS + _DAY)],
        owner_id="u1",
        owner_name="Mara",
        old_profile=_stored(_matter("her salary")),
        tz=eight,
    )

    assert _matters_shown(fake, 0) == f'<matter index="0" since="{format_time(_TS, tz=eight)}">her salary</matter>'
    assert "UTC+08:00" in _matters_shown(fake, 0)


async def test_a_stored_topic_is_masked_when_read_so_a_legacy_unmasked_one_is_not_added_again() -> None:
    """A profile stored before topics were masked may hold a number in a topic; it is masked from this call on."""
    legacy = {"topic": f"her card {_BANK_CARD}", "evidence": "said keep it out", "observed_at": _TS}
    fake = FakeLLMClient(responses=[_operations(_forget(f"her card {_BANK_CARD}", "said it again"))])

    profile = await _update(fake, _stored(legacy), _episode("asked again to keep her card out.", _TS + _DAY))

    assert _do_not_record_of(profile) == [_matter("her card ***", _TS + _DAY)]
    assert _BANK_CARD not in str(fake.calls[0].messages[1].content)


def test_a_stored_list_of_any_earlier_shape_is_read_into_the_current_one() -> None:
    stored = [
        {"topic": "her salary", "evidence": "said keep it out", "observed_at": _TS},
        {"topic": "her diet"},  # no date: any dated withdrawal is later
        {"topic": "  ", "observed_at": _TS},
        {"evidence": "no topic"},
        "her divorce",
        {"topic": "her finances", "observed_at": _TS, "except": [{"topic": "her mortgage"}, {"topic": ""}, 3]},
    ]

    assert _do_not_record.load_matters(stored) == [
        _matter("her salary"),
        _matter("her diet", 0),
        _matter("her finances", allowed=[_matter("her mortgage", 0)]),
    ]
    assert _do_not_record.load_matters(None) == []
    assert stored[0] == {"topic": "her salary", "evidence": "said keep it out", "observed_at": _TS}


async def test_a_matter_whose_quote_is_not_in_the_narratives_does_not_enter(caplog: pytest.LogCaptureFixture) -> None:
    """The request is checked as an item's quote is: found in the batch, or the matter is refused."""
    fake = FakeLLMClient(
        responses=[
            _init_reply(
                {"topic": "her salary", "quote": "asked never to record her salary"},
                {"topic": "her diet", "quote": "asked that her diet be kept out for good"},
                {"topic": "her divorce"},
                {"topic": "her debts", "evidence": "asked never to record her salary"},
            ),
            _operations(
                _forget("her illness", "wants her illness never written down"),
                _forget("her weight", "said her weight is private from now on"),
                {"action": "forget", "topic": "her age"},
            ),
        ]
    )
    extractor = ProfileExtractor(llm=fake)

    with caplog.at_level(logging.INFO, logger=_PROFILE_LOGGER):
        initial = await extractor.aextract_from_episodes(
            [_episode("asked never to record her salary.", _TS)], owner_id="u1", owner_name="Mara"
        )
        updated = await extractor.aextract_from_episodes(
            [_episode("wants her illness never written down, she said.", _TS + _DAY)],
            owner_id="u1",
            owner_name="Mara",
            old_profile=initial,
        )

    # A missing quote counts as not found, and the earlier reply key "evidence" is accepted but is no quote.
    assert _do_not_record_of(initial) == [_matter("her salary", _TS)]
    assert _do_not_record_of(updated) == [_matter("her salary", _TS), _matter("her illness", _TS + _DAY)]
    assert [line["refused"] for line in _log_fields(caplog, "profile do-not-record updated:")] == ["3", "2"]


async def test_a_matter_is_dated_by_the_narrative_holding_its_quote() -> None:
    fake = FakeLLMClient(responses=[_operations(_forget("her salary", "asked never to record her salary"))])

    profile = await _checked_update(
        fake,
        _stored(),
        _episode("asked never to record her salary.", _TS + _DAY),
        _episode("talked about a trip to the coast.", _TS + 3 * _DAY),
    )

    assert _do_not_record_of(profile) == [_matter("her salary", _TS + _DAY)]


async def test_entering_a_matter_drops_the_pending_entries_about_it() -> None:
    """Those the forget names by index and those whose description holds the topic; any other entry stays."""
    held = [
        {"evidence": "e", "description": "Weighs herself on Sundays.", "category": "Habits"},
        {"evidence": "e", "description": "Checks her blood pressure at night.", "category": "Habits"},
        {"evidence": "e", "description": "Swims before work.", "category": "Habits"},
        {"evidence": "e", "description": "Reads on the train.", "category": "Habits"},
    ]
    pending = [{**entry, "bucket": "explicit_info", "observed_at": _TS} for entry in held]
    fake = FakeLLMClient(
        responses=[_operations(_forget("her blood pressure", pending=[0, "9", None]), _forget("her diet", pending=3))]
    )

    profile = await _update(fake, _stored(pending=pending), _episode("asked to keep two things out.", _TS + _DAY))

    kept = [entry["description"] for entry in (profile.model_extra or {})["pending"]]
    assert kept == ["Swims before work."]
    assert [entry["topic"] for entry in _do_not_record_of(profile) or []] == ["her blood pressure", "her diet"]


# ==========================================================================
# Leaving the list
# ==========================================================================


async def test_a_withdrawn_matter_leaves_the_list_and_the_emptied_list_is_returned(
    caplog: pytest.LogCaptureFixture,
) -> None:
    stored = _stored(_matter("her salary"), _matter("her diet"))
    add = {
        "action": "add",
        "type": "explicit_info",
        "data": {"evidence": "said what she earns", "description": "Earns 4,000 a month.", "category": "Work"},
    }
    fake = FakeLLMClient(responses=[_operations(_allow(0), add), _operations(_allow("0"))])
    extractor = ProfileExtractor(item_checks=False, llm=fake)

    with caplog.at_level(logging.INFO, logger=_PROFILE_LOGGER):
        second = await extractor.aextract_from_episodes(
            [_episode("said her salary may be recorded now.", _TS + _DAY)],
            owner_id="u1",
            owner_name="Mara",
            old_profile=stored,
        )
        third = await extractor.aextract_from_episodes(
            [_episode("asked that her diet be remembered.", _TS + 2 * _DAY)],
            owner_id="u1",
            owner_name="Mara",
            old_profile=Profile.model_validate(json.loads(second.model_dump_json())),
        )

    # The entry is gone without a trace, and the item about the matter is written by the same reply.
    assert _do_not_record_of(second) == [_matter("her diet")]
    assert [item.description for item in second.explicit_info] == ["Earns 4,000 a month."]
    # An index written as a string of digits is read; the emptied list comes back as [] for the host to store.
    assert _do_not_record_of(third) == []
    assert json.loads(third.model_dump_json())["do_not_record"] == []
    assert _log_fields(caplog, "profile do-not-record withdrawn:") == [
        {"named": "1", "whole": "1", "part": "0", "refused": "0", "matters": "1"},
        {"named": "1", "whole": "1", "part": "0", "refused": "0", "matters": "0"},
    ]
    assert "salary" not in "".join(record.getMessage() for record in caplog.records)


async def test_a_withdrawal_counts_only_when_its_quote_is_found_later_than_the_request(
    caplog: pytest.LogCaptureFixture,
) -> None:
    stored = _stored(
        _matter("her salary", _TS + 5 * _DAY),
        _matter("her diet", _TS),
        _matter("her divorce", _TS),
        timestamp=_TS + 5 * _DAY,
    )
    fake = FakeLLMClient(
        responses=[
            # Historical pass: the narrative is older than the request for her salary.
            _operations(_allow(0, "said her salary may be recorded now")),
            _operations(
                _allow(1, "said her diet may be recorded now"),
                _allow(2, "said her divorce may be recorded now"),
                {"action": "allow", "matter": 1},
                _allow(7, "said her diet may be recorded now"),
                {"action": "allow", "matter": True, "quote": "said her diet may be recorded now"},
            ),
        ]
    )

    with caplog.at_level(logging.INFO, logger=_PROFILE_LOGGER):
        profile = await _checked_update(
            fake,
            stored,
            _episode("said her salary may be recorded now.", _TS + 2 * _DAY),
            _episode("said her diet may be recorded now.", _TS + 9 * _DAY),
        )

    # Refused: a withdrawal older than the request, a quote not in the batch, no quote, and indices naming no entry.
    assert _do_not_record_of(profile) == [_matter("her salary", _TS + 5 * _DAY), _matter("her divorce", _TS)]
    assert [(line["whole"], line["refused"]) for line in _log_fields(caplog, "profile do-not-record withdrawn:")] == [
        ("0", "1"),
        ("1", "4"),
    ]


async def test_a_withdrawal_dated_like_the_request_is_refused() -> None:
    stored = _stored(_matter("her salary", _TS + _DAY))
    fake = FakeLLMClient(responses=[_operations(_allow(0))])

    profile = await _update(fake, stored, _episode("mentioned her salary.", _TS + _DAY))

    assert _do_not_record_of(profile) == [_matter("her salary", _TS + _DAY)]


async def test_a_request_repeated_after_the_withdrawal_in_the_same_batch_keeps_the_matter() -> None:
    """Requests are read before withdrawals, so the newer of the two decides whichever order the reply has them in."""
    stored = _stored(_matter("her salary", _TS))
    fake = FakeLLMClient(
        responses=[
            _operations(
                _allow(0, "said her salary may be recorded now"),
                _forget("her salary", "asked again never to record her salary"),
            ),
            _operations(
                _forget("her salary", "asked again never to record her salary"),
                _allow(0, "said her salary may be recorded now"),
            ),
        ]
    )
    allowed = _episode("said her salary may be recorded now.", _TS + _DAY)

    kept = await _checked_update(
        fake, stored, allowed, _episode("asked again never to record her salary.", _TS + 2 * _DAY)
    )
    lifted = await _checked_update(
        fake, stored, _episode("asked again never to record her salary.", _TS + _DAY // 2), allowed
    )

    assert _do_not_record_of(kept) == [_matter("her salary", _TS + 2 * _DAY)]
    assert _do_not_record_of(lifted) == []


async def test_a_matter_entered_by_the_same_reply_cannot_be_withdrawn_by_it() -> None:
    """An allow names an entry the model was shown; the index of one the reply itself adds names nothing."""
    fake = FakeLLMClient(responses=[_operations(_forget("her diet"), _allow(1), _allow(0))])

    profile = await _update(fake, _stored(_matter("her salary")), _episode("spoke of both.", _TS + _DAY))

    assert _do_not_record_of(profile) == [_matter("her diet", _TS + _DAY)]


async def test_a_part_allowed_again_becomes_an_exception_and_the_rest_stays_listed() -> None:
    stored = _stored(_matter("her finances"), _matter("her diet"))
    fake = FakeLLMClient(
        responses=[
            _operations(_allow(0, part="her mortgage")),
            _operations(_allow(0, part="Her Mortgage"), _allow(0, part="her rent"), _allow(1, part="her diet")),
            _operations({"action": "none"}),
        ]
    )
    extractor = ProfileExtractor(item_checks=False, llm=fake)

    second = await extractor.aextract_from_episodes(
        [_episode("said her mortgage may be noted now.", _TS + _DAY)],
        owner_id="u1",
        owner_name="Mara",
        old_profile=stored,
    )
    third = await extractor.aextract_from_episodes(
        [_episode("said her rent may be noted too.", _TS + 2 * _DAY)],
        owner_id="u1",
        owner_name="Mara",
        old_profile=Profile.model_validate(json.loads(second.model_dump_json())),
    )
    await extractor.aextract_from_episodes(
        [_episode("talked about work.", _TS + 3 * _DAY)], owner_id="u1", owner_name="Mara", old_profile=third
    )

    assert _do_not_record_of(second) == [
        _matter("her finances", allowed=[_matter("her mortgage", _TS + _DAY)]),
        _matter("her diet"),
    ]
    # The same part again only moves its date; a part that is the whole topic withdraws the whole matter.
    assert _do_not_record_of(third) == [
        _matter(
            "her finances",
            allowed=[_matter("her mortgage", _TS + 2 * _DAY), _matter("her rent", _TS + 2 * _DAY)],
        )
    ]
    assert _matters_shown(fake, 2) == (_shown(0, "her finances<except>her mortgage</except><except>her rent</except>"))


async def test_a_new_request_for_an_allowed_part_takes_the_exception_back_only_when_newer() -> None:
    entry = _matter(
        "her finances",
        _TS,
        allowed=[_matter("her mortgage", _TS + 5 * _DAY), _matter("her rent", _TS + 5 * _DAY)],
    )
    stored = _stored(entry, timestamp=_TS + 5 * _DAY)
    fake = FakeLLMClient(
        responses=[
            _operations(_forget("her mortgage")),  # historical pass: older than the allowance
            _operations(_forget("her rent")),
        ]
    )

    profile = await _update(
        fake,
        stored,
        _episode("asked to keep her mortgage out.", _TS + 2 * _DAY),
        _episode("asked to keep her rent out after all.", _TS + 8 * _DAY),
    )

    # The older request does not come back; the newer one removes the exception, and no second entry is added because
    # the wider one covers it again. The entry's date moves, so that a withdrawal must be later than this request.
    assert _do_not_record_of(profile) == [
        _matter("her finances", _TS + 8 * _DAY, allowed=[_matter("her mortgage", _TS + 5 * _DAY)])
    ]


def test_the_update_rules_say_how_a_request_stands_and_how_it_is_withdrawn_without_contradiction() -> None:
    system = PROFILE_UPDATE_FROM_EPISODE_TEXTS_SYSTEM_PROMPT

    assert "A matter listed in <do_not_record> stays out until this person withdraws the request" in system
    # The rule this replaced kept a listed matter out "however it is mentioned again" with no way back.
    assert "never enters, in any wording and however it is mentioned again" not in system
    assert "A request for this one time or for now" in system
    assert "by asking themselves that the same matter be remembered, noted or kept in mind" in system
    assert "Asking the assistant to remember a listed matter is a withdrawal, not a mention of it." in system
    assert "an allow is the opposite of a forget" in system
    assert "Only their own request counts" in system
    assert '{"action": "forget", "topic": "...", "quote": "...", "pending": null, "evidence": null}' in system
    assert '{"action": "allow", "matter": 0, "part": null, "quote": "..."}' in system


def test_the_reply_models_take_the_new_operations_and_the_earlier_evidence_key() -> None:
    update = _UpdateOutput.model_validate_json(
        _operations(
            _forget("her diet", pending=[0]),
            {"action": "forget", "topic": "her diet", "evidence": "as replies carried it before"},
            _allow(0, part="her rent"),
        )
    )
    init = _EpisodeInitOutput.model_validate_json(
        _init_reply({"topic": "t", "quote": "q"}, {"topic": "t", "evidence": "e"}, {"topic": "t"})
    )

    assert [op.action for op in update.operations] == ["forget", "forget", "allow"]
    assert [entry.quote for entry in init.do_not_record] == ["q", None, None]


# ==========================================================================
# The cap: merging redundant entries
# ==========================================================================


def _full_list(count: int = _CAP) -> list[dict[str, Any]]:
    return [_matter(f"matter number {index}", _TS + index) for index in range(count)]


async def test_a_list_grown_past_its_cap_gets_one_merge_call_that_only_removes_redundant_entries(
    caplog: pytest.LogCaptureFixture,
) -> None:
    listed = _full_list()
    listed[3] = _matter("her finances", _TS + 3)
    listed[7] = _matter("her salary", _TS + 7)
    listed[9] = _matter("what she earns", _TS + 9, allowed=[_matter("her bonus", _TS + 10)])
    fake = FakeLLMClient(
        responses=[
            _operations(_forget("her pay")),
            _merges({"keep": 3, "drop": [7, 30], "relation": "inside"}),
            _operations({"action": "none"}),
        ]
    )
    extractor = ProfileExtractor(item_checks=False, llm=fake)

    with caplog.at_level(logging.INFO, logger=_PROFILE_LOGGER):
        profile = await extractor.aextract_from_episodes(
            [_episode("asked to keep her pay out.", _TS + _DAY)],
            owner_id="u1",
            owner_name="Mara",
            old_profile=_stored(*listed),
            custom_instructions="Write tersely.",
        )
        await extractor.aextract_from_episodes(
            [_episode("talked about work.", _TS + 2 * _DAY)], owner_id="u1", owner_name="Mara", old_profile=profile
        )

    merge_call = fake.calls[1]
    system, user = (str(message.content) for message in merge_call.messages)
    assert merge_call.response_format is _MatterMergeOutput
    assert system == PROFILE_DO_NOT_RECORD_MERGE_SYSTEM_PROMPT  # no profile blocks, no additional requirements
    assert user.startswith(f'<target_person>Mara</target_person>\n\n<do_not_record count="{_CAP + 1}" cap="{_CAP}">\n')
    shown = re.findall(r'<matter index="(\d+)">', user)
    # Every entry keeps its place in the list as its index; the one with an exception is not offered for merging.
    assert shown == [str(index) for index in range(_CAP + 1) if index != 9]
    assert "what she earns" not in user
    assert "since=" not in user  # no date matters to a merge

    result = _do_not_record_of(profile) or []
    assert len(result) == _CAP - 1
    topics = [entry["topic"] for entry in result]
    assert "her salary" not in topics and "her pay" not in topics
    # The kept entry is word for word the one this person asked for, dated by the newest request it now stands for.
    assert result[3] == _matter("her finances", _TS + _DAY)
    assert [entry for entry in result if entry["topic"] != "her finances"] == [
        entry for index, entry in enumerate(listed) if index not in (3, 7)
    ]
    assert _log_fields(caplog, "profile do-not-record merged:") == [
        {
            "matters_before": str(_CAP + 1),
            "merges": "1",
            "refused": "0",
            "removed": "2",
            "matters": str(_CAP - 1),
            "cap": str(_CAP),
        }
    ]
    assert "profile do-not-record over cap" not in caplog.text
    assert fake.call_count == 3  # the later update of a list inside its cap makes no merge call


async def test_a_list_past_its_cap_with_nothing_redundant_keeps_every_entry_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake = FakeLLMClient(
        responses=[_operations(_forget("her pay")), _merges(), _operations({"action": "none"})],
    )
    extractor = ProfileExtractor(item_checks=False, llm=fake)

    with caplog.at_level(logging.WARNING, logger=_PROFILE_LOGGER):
        profile = await extractor.aextract_from_episodes(
            [_episode("asked to keep her pay out.", _TS + _DAY)],
            owner_id="u1",
            owner_name="Mara",
            old_profile=_stored(*_full_list()),
        )
        again = await extractor.aextract_from_episodes(
            [_episode("talked about work.", _TS + 2 * _DAY)], owner_id="u1", owner_name="Mara", old_profile=profile
        )

    assert _do_not_record_of(profile) == [*_full_list(), _matter("her pay", _TS + _DAY)]
    assert _log_fields(caplog, "profile do-not-record over cap:") == [{"matters": str(_CAP + 1), "cap": str(_CAP)}]
    # A list that did not grow is not offered again, so a long list costs no extra call on every batch.
    assert fake.call_count == 3
    assert _do_not_record_of(again) == _do_not_record_of(profile)
    assert len(_matters_shown(fake, 2).splitlines()) == _CAP + 1  # and the whole list is still shown to UPDATE


async def test_a_list_inside_its_cap_gets_no_merge_call() -> None:
    fake = FakeLLMClient(responses=[_operations(_forget("her pay"))])

    profile = await _update(fake, _stored(*_full_list(_CAP - 1)), _episode("asked to keep her pay out.", _TS + _DAY))

    assert fake.call_count == 1
    assert len(_do_not_record_of(profile) or []) == _CAP


async def test_an_invalid_merge_reply_leaves_the_list_as_it_is(caplog: pytest.LogCaptureFixture) -> None:
    fake = FakeLLMClient(responses=[_operations(_forget("her pay")), json.dumps({"operations": [{"keep": "x"}]})])

    with caplog.at_level(logging.WARNING, logger=_PROFILE_LOGGER):
        profile = await _update(fake, _stored(*_full_list()), _episode("asked to keep her pay out.", _TS + _DAY))

    assert len(_do_not_record_of(profile) or []) == _CAP + 1
    assert "profile do-not-record merge reply invalid: matters=31" in caplog.text
    assert "profile do-not-record over cap: matters=31 cap=30" in caplog.text


async def test_no_merge_call_is_made_when_fewer_than_two_entries_could_merge() -> None:
    listed = [_matter(f"matter {index}", _TS, allowed=[_matter("a part", _TS)]) for index in range(_CAP)]
    fake = FakeLLMClient(responses=[_operations(_forget("her pay"))])

    profile = await _update(fake, _stored(*listed), _episode("asked to keep her pay out.", _TS + _DAY))

    assert fake.call_count == 1
    assert len(_do_not_record_of(profile) or []) == _CAP + 1


def test_a_merge_the_program_cannot_hold_to_its_rule_is_refused_alone(caplog: pytest.LogCaptureFixture) -> None:
    matters = [
        _matter("her finances", _TS + 1),
        _matter("her salary", _TS + 9),
        _matter("what she earns", _TS + 2),
        _matter("her diet", _TS + 3, allowed=[_matter("her allergies", _TS + 4)]),
        _matter("what she eats", _TS + 5),
        _matter("her divorce", _TS + 6),
        _matter("her marriage", _TS + 7),
    ]
    before = json.loads(json.dumps(matters))

    with caplog.at_level(logging.WARNING, logger=_PROFILE_LOGGER):
        _do_not_record.apply_merge_ops(
            matters,
            [
                {"keep": 0, "drop": [1, 2], "relation": "inside"},
                {"keep": 5, "drop": [6], "relation": "related"},  # neither of the two relations
                {"keep": 4, "drop": [3], "relation": "same"},  # an entry with an exception takes no part
                {"keep": 3, "drop": [4], "relation": "inside"},
                {"keep": 5, "drop": [0], "relation": "inside"},  # an index an earlier merge used
                {"keep": 5, "drop": [], "relation": "same"},
                {"keep": 5, "drop": [5], "relation": "same"},
                {"keep": 5, "drop": [6, 6], "relation": "same"},
                {"keep": 5, "drop": [12], "relation": "same"},
                {"keep": True, "drop": [6], "relation": "same"},
                {"keep": 5, "drop": ["6"], "relation": "same"},
            ],
        )

    assert matters == [_matter("her finances", _TS + 9), *before[3:]]
    assert [(line["merges"], line["refused"]) for line in _log_fields(caplog, "profile do-not-record merged:")] == [
        ("1", "10")
    ]


def test_the_merge_rules_let_the_model_name_entries_only() -> None:
    system = PROFILE_DO_NOT_RECORD_MERGE_SYSTEM_PROMPT

    assert "{{" not in system  # nothing per-request: the list travels in the user message
    assert "You cannot reword an entry or write a new one" in system
    assert "a wider entry is never dropped for a narrower one" in system
    assert "Finding nothing to merge is a normal answer" in system
    assert '{"keep": 2, "drop": [5, 9], "relation": "inside"}' in system
    for skeleton in re.findall(r"^\{\n.*?^\}", system, flags=re.MULTILINE | re.DOTALL):
        _MatterMergeOutput.model_validate_json(skeleton)
