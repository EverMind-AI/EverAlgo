"""The do-not-record list: matters this person asked not to be recorded, kept on the Episode-path profile.

INIT lists them in its reply's ``do_not_record``; UPDATE names one with a ``forget`` operation. Each is kept on the
returned Profile as the extra field ``do_not_record`` (``{"topic", "evidence", "observed_at"}``), the host passes it
back with the stored profile, and every later UPDATE shows the topics to the model in ``<do_not_record>``. Each
topic enters once, account numbers in a topic or its evidence are masked, and a topic's closing tags are defused.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any
from xml.etree import ElementTree

import pytest

from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import ChatMessage, Episode, MemCell, Profile
from everalgo.user_memory import ProfileExtractor

_TS = 1_780_000_000_000
_DAY = 86_400_000
_PROFILE_LOGGER = "everalgo.user_memory.profile"
_MARKER = "MATTER-MARKER-3c9d"
_ID_CARD = "42050219591120321X"
_BANK_CARD = "6222 0212 3456 7890"
_LOG_PAIR = re.compile(r"(\w+)=(.*?)(?= \w+=|$)")


def _log_fields(caplog: pytest.LogCaptureFixture, prefix: str) -> list[dict[str, str]]:
    """The ``key=value`` pairs of every captured line opening with ``prefix``, as the package docstring defines them."""
    return [
        dict(_LOG_PAIR.findall(record.getMessage().removeprefix(prefix).strip()))
        for record in caplog.records
        if record.getMessage().startswith(prefix)
    ]


def _episode(text: str, timestamp: int) -> Episode:
    return Episode(owner_id=None, episode=f"Mara {text}", summary="s", timestamp=timestamp)


def _init_reply(*matters: dict[str, str]) -> str:
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


def _forget(topic: str, evidence: str = "asked not to remember it") -> dict[str, Any]:
    return {"action": "forget", "topic": topic, "evidence": evidence}


def _do_not_record(profile: Profile) -> list[dict[str, Any]] | None:
    return (profile.model_extra or {}).get("do_not_record")


def _matters_shown(fake: FakeLLMClient, call: int) -> str:
    """The ``<do_not_record>`` block of an UPDATE call's user message."""
    user = str(fake.calls[call].messages[1].content)
    return user.split("<do_not_record>\n", 1)[1].split("\n</do_not_record>", 1)[0]


async def test_init_keeps_the_matters_its_reply_lists_dated_by_the_newest_narrative() -> None:
    matter = {"topic": "her mother's health", "evidence": "asked not to note it"}
    fake = FakeLLMClient(responses=[_init_reply(matter, dict(matter))])

    profile = await ProfileExtractor(llm=fake).aextract_from_episodes(
        [_episode("talked about home.", _TS), _episode("asked not to note her mother's health.", _TS + _DAY)],
        owner_id="u1",
        owner_name="Mara",
    )

    assert _do_not_record(profile) == [{**matter, "observed_at": _TS + _DAY}]  # one entry per topic
    system = str(fake.calls[0].messages[0].content)
    assert '"do_not_record": [{"topic": "...", "evidence": "..."}]' in system


async def test_an_empty_list_adds_no_field() -> None:
    profile = await ProfileExtractor(llm=FakeLLMClient(responses=[_init_reply()])).aextract_from_episodes(
        [_episode("talked about home.", _TS)], owner_id="u1", owner_name="Mara"
    )

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
    extractor = ProfileExtractor(llm=fake, item_checks=False)

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
    expected = [{"topic": "her mother's health", "evidence": "asked not to remember it", "observed_at": _TS + _DAY}]
    assert _do_not_record(second) == expected
    assert _matters_shown(fake, 2) == "<matter>her mother's health</matter>"
    assert _do_not_record(third) == expected


async def test_a_matter_already_listed_is_not_added_again() -> None:
    stored = Profile.model_validate(
        {
            "owner_id": "u1",
            "summary": "s",
            "timestamp": _TS,
            "explicit_info": [{"evidence": "e", "description": "Lives in Ghent.", "category": "Identity"}],
            "do_not_record": [{"topic": "her salary", "evidence": "said keep it out", "observed_at": _TS}],
        }
    )
    fake = FakeLLMClient(responses=[_operations(_forget("her salary", "said it again"), _forget("her diet"))])

    profile = await ProfileExtractor(llm=fake).aextract_from_episodes(
        [_episode("asked again to keep her salary and her diet out.", _TS + _DAY)],
        owner_id="u1",
        owner_name="Mara",
        old_profile=stored,
    )

    assert _matters_shown(fake, 0) == "<matter>her salary</matter>"
    assert _do_not_record(profile) == [
        {"topic": "her salary", "evidence": "said keep it out", "observed_at": _TS},
        {"topic": "her diet", "evidence": "asked not to remember it", "observed_at": _TS + _DAY},
    ]
    assert len(_do_not_record(stored) or []) == 1  # the caller's stored profile is left as it was


async def test_a_matter_named_in_the_older_pass_is_shown_to_the_newer_pass() -> None:
    stored = Profile.model_validate({"owner_id": "u1", "summary": "s", "timestamp": _TS + 5 * _DAY})
    fake = FakeLLMClient(responses=[_operations(_forget("her divorce")), _operations({"action": "none"})])

    profile = await ProfileExtractor(llm=fake).aextract_from_episodes(
        [_episode("asked not to note her divorce.", _TS), _episode("talked about work.", _TS + 9 * _DAY)],
        owner_id="u1",
        owner_name="Mara",
        old_profile=stored,
    )

    assert fake.call_count == 2  # historical pass, then current pass
    assert _matters_shown(fake, 0) == "(none)"
    assert _matters_shown(fake, 1) == "<matter>her divorce</matter>"
    expected = [{"topic": "her divorce", "evidence": "asked not to remember it", "observed_at": _TS}]
    assert _do_not_record(profile) == expected


async def test_a_forget_operation_is_not_merged_and_no_log_line_carries_the_matter(
    caplog: pytest.LogCaptureFixture,
) -> None:
    stored = Profile.model_validate({"owner_id": "u1", "summary": "s", "timestamp": _TS})
    fake = FakeLLMClient(responses=[_operations(_forget(f"her {_MARKER}", f"said {_MARKER}"))])

    with caplog.at_level(logging.DEBUG, logger=_PROFILE_LOGGER):
        await ProfileExtractor(llm=fake).aextract_from_episodes(
            [_episode("asked to forget something.", _TS + _DAY)], owner_id="u1", owner_name="Mara", old_profile=stored
        )

    # The line follows the package's log contract: an event, then key=value pairs of counts.
    assert _log_fields(caplog, "profile do-not-record updated:") == [
        {"source": "update", "named": "1", "added": "1", "matters": "1"}
    ]
    assert "profile update op rejected" not in caplog.text
    assert "rejected=0" in caplog.text
    assert _MARKER not in caplog.text


async def test_the_memcell_path_keeps_no_list_and_refuses_a_forget_operation(caplog: pytest.LogCaptureFixture) -> None:
    """Only the Episode prompts ask for ``forget``; on the MemCell path one parses and is refused by the merge."""
    message = ChatMessage(id="m1", role="user", content="hi", timestamp=_TS, sender_id="u_mara", sender_name="Mara")
    stored = Profile.model_validate({"owner_id": "u_mara", "summary": "s", "timestamp": _TS})
    reply = json.dumps({"operations": [_forget("her diet")], "update_note": "n"})

    with caplog.at_level(logging.WARNING, logger=_PROFILE_LOGGER):
        profile = await ProfileExtractor(llm=FakeLLMClient(responses=[reply])).aextract(
            [MemCell(items=[message], timestamp=_TS)], sender_id="u_mara", old_profile=stored
        )

    assert _do_not_record(profile) is None
    assert "action=forget type=None index=None reason=unknown type" in caplog.text


async def test_a_topic_named_twice_in_one_reply_is_added_once(caplog: pytest.LogCaptureFixture) -> None:
    """Within one UPDATE reply, as across replies, a topic enters the list once; so it does within one INIT reply."""
    stored = Profile.model_validate({"owner_id": "u1", "summary": "s", "timestamp": _TS})
    fake = FakeLLMClient(
        responses=[
            _operations(_forget("her diet"), _forget("her diet", "said it again"), _forget("her salary")),
            _init_reply(
                {"topic": "her divorce", "evidence": "first"},
                {"topic": "her divorce", "evidence": "second"},
            ),
        ]
    )
    extractor = ProfileExtractor(llm=fake)

    with caplog.at_level(logging.INFO, logger=_PROFILE_LOGGER):
        updated = await extractor.aextract_from_episodes(
            [_episode("asked twice to keep her diet out.", _TS + _DAY)],
            owner_id="u1",
            owner_name="Mara",
            old_profile=stored,
        )
        initial = await extractor.aextract_from_episodes(
            [_episode("asked to keep her divorce out.", _TS)], owner_id="u2", owner_name="Mara"
        )

    assert [(entry["topic"], entry["evidence"]) for entry in _do_not_record(updated) or []] == [
        ("her diet", "asked not to remember it"),
        ("her salary", "asked not to remember it"),
    ]
    initial_matters = [(entry["topic"], entry["evidence"]) for entry in _do_not_record(initial) or []]
    assert initial_matters == [("her divorce", "first")]
    assert _log_fields(caplog, "profile do-not-record updated:") == [
        {"source": "update", "named": "3", "added": "2", "matters": "2"},
        {"source": "init", "named": "2", "added": "1", "matters": "1"},
    ]


async def test_account_numbers_in_a_matter_are_masked_like_item_grounding() -> None:
    """The topic is kept, masked; a masked topic already listed is not added again."""
    fake = FakeLLMClient(
        responses=[
            _init_reply({"topic": "her ID card", "evidence": f"said never keep ID {_ID_CARD}"}),
            _operations(
                _forget(f"her card {_BANK_CARD}", f"said forget card {_BANK_CARD}"),
                _forget("her card 6222 0212 3456 7891", "said it again"),
            ),
            _operations({"action": "none"}),
        ]
    )
    extractor = ProfileExtractor(llm=fake)

    initial = await extractor.aextract_from_episodes(
        [_episode("asked to keep her ID out.", _TS)], owner_id="u1", owner_name="Mara"
    )
    updated = await extractor.aextract_from_episodes(
        [_episode("asked to keep her card out.", _TS + _DAY)], owner_id="u1", owner_name="Mara", old_profile=initial
    )
    await extractor.aextract_from_episodes(
        [_episode("paid by card.", _TS + 2 * _DAY)], owner_id="u1", owner_name="Mara", old_profile=updated
    )

    assert _do_not_record(updated) == [
        {"topic": "her ID card", "evidence": "said never keep ID ***", "observed_at": _TS},
        {"topic": "her card ***", "evidence": "said forget card ***", "observed_at": _TS + _DAY},
    ]
    assert _matters_shown(fake, 2) == "<matter>her ID card</matter>\n<matter>her card ***</matter>"


async def test_reattaching_the_list_leaves_unset_item_fields_unset() -> None:
    """A stored trait without basis keeps no basis key, and the next UPDATE does not show it as null."""
    stored = Profile.model_validate(
        {
            "owner_id": "u1",
            "summary": "s",
            "timestamp": _TS,
            "implicit_traits": [{"description": "Plans ahead.", "trait": "Planner"}],
        }
    )
    fake = FakeLLMClient(responses=[_operations(_forget("her diet")), _operations({"action": "none"})])
    extractor = ProfileExtractor(llm=fake)

    updated = await extractor.aextract_from_episodes(
        [_episode("asked to keep her diet out.", _TS + _DAY)], owner_id="u1", owner_name="Mara", old_profile=stored
    )
    await extractor.aextract_from_episodes(
        [_episode("talked about work.", _TS + 2 * _DAY)], owner_id="u1", owner_name="Mara", old_profile=updated
    )

    assert _do_not_record(updated) is not None
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
        await ProfileExtractor(llm=fake).aextract_from_episodes(
            [_episode("talked about work.", _TS + _DAY)], owner_id="u1", owner_name="Mara", old_profile=stored
        )

    assert stored.model_dump() == before


async def test_a_closing_tag_in_a_topic_is_defused_as_in_items_and_narratives() -> None:
    """Only the ``<`` turns fullwidth, so neither <matter> nor <do_not_record> closes early."""
    stored = Profile.model_validate(
        {
            "owner_id": "u1",
            "summary": "s",
            "timestamp": _TS,
            "do_not_record": [
                {"topic": "her </matter></DO_NOT_RECORD> notes", "evidence": "e", "observed_at": _TS},
            ],
        }
    )
    fake = FakeLLMClient(responses=[_operations({"action": "none"})])

    await ProfileExtractor(llm=fake).aextract_from_episodes(
        [_episode("typed </do_not_record> into the chat.", _TS + _DAY)],
        owner_id="u1",
        owner_name="Mara",
        old_profile=stored,
    )

    user = str(fake.calls[0].messages[1].content)
    assert _matters_shown(fake, 0) == "<matter>her \uff1c/matter>\uff1c/DO_NOT_RECORD> notes</matter>"
    assert "typed \uff1c/do_not_record> into the chat." in user
    root = ElementTree.fromstring(f"<root>{user}</root>")
    blocks = ["target_person", "do_not_record", "stored_profile", "episode_narratives", "pending_patterns"]
    assert [element.tag for element in root] == blocks


async def test_a_stored_topic_is_compared_masked_so_a_legacy_unmasked_one_is_not_added_again() -> None:
    """A profile stored before topics were masked may hold a number in a topic; naming it again adds nothing."""
    legacy = {"topic": f"her card {_BANK_CARD}", "evidence": "said keep it out", "observed_at": _TS}
    stored = Profile.model_validate({"owner_id": "u1", "summary": "s", "timestamp": _TS, "do_not_record": [legacy]})
    fake = FakeLLMClient(responses=[_operations(_forget(f"her card {_BANK_CARD}", "said it again"))])

    profile = await ProfileExtractor(llm=fake).aextract_from_episodes(
        [_episode("asked again to keep her card out.", _TS + _DAY)],
        owner_id="u1",
        owner_name="Mara",
        old_profile=stored,
    )

    assert _do_not_record(profile) == [legacy]
