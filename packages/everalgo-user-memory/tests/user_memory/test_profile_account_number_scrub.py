"""Identification and account numbers never reach a stored item.

Twelve or more digits (single spaces allowed between them, an ID card's trailing X included): an item whose
description carries one is dropped, and one in its evidence or basis is masked as ``***``. An eleven-digit mobile
number is not such a number. The rule holds wherever a model writes an item: INIT, an UPDATE add or update, and a
COMPACT or REGROUP operation.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import pytest

from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import ChatMessage, Episode, MemCell, Profile
from everalgo.user_memory import ProfileExtractor
from everalgo.user_memory.profile._items import _PROFILE_MAX_ITEMS
from everalgo.user_memory.profile._merge import _scrub

_TS = 1_780_000_000_000
_LATER = _TS + 86_400_000
_PROFILE_LOGGER = "everalgo.user_memory.profile"
_ID_CARD = "42050219591120321X"
_BANK_CARD = "6222 0212 3456 7890"
_INSURANCE = "420500358812"
_PHONE = "13812345678"
_LOG_PAIR = re.compile(r"(\w+)=(.*?)(?= \w+=|$)")


def _log_fields(caplog: pytest.LogCaptureFixture, prefix: str) -> list[dict[str, str]]:
    """The ``key=value`` pairs of every captured line opening with ``prefix``, as the package docstring defines them."""
    return [
        dict(_LOG_PAIR.findall(record.getMessage().removeprefix(prefix).strip()))
        for record in caplog.records
        if record.getMessage().startswith(prefix)
    ]


@pytest.mark.parametrize(
    "number",
    [_ID_CARD, _ID_CARD.lower(), _BANK_CARD, _INSURANCE, "6222021234567890123"],
    ids=["id-card-x", "id-card-lowercase-x", "spaced-bank-card", "twelve-digits", "nineteen-digits"],
)
def test_a_number_in_the_grounding_is_masked_and_one_in_the_description_drops_the_item(number: str) -> None:
    kept = _scrub({"evidence": f"her number is {number}, she said", "description": "Has health insurance."})

    assert kept == {"evidence": "her number is ***, she said", "description": "Has health insurance."}
    assert _scrub({"evidence": "e", "description": f"Card {number}."}) is None
    assert _scrub({"basis": f"wrote {number} twice", "description": "d", "trait": "t"}) == {
        "basis": "wrote *** twice",
        "description": "d",
        "trait": "t",
    }


@pytest.mark.parametrize(
    "text",
    [_PHONE, "138 1234 5678", "2026-06-01 and 2026-06-02", "12345678901", "call 010-6512-3456", "1,234,567,890,123"],
    ids=["mobile", "spaced-mobile", "dates", "eleven-digits", "hyphenated-landline", "comma-grouped-amount"],
)
def test_shorter_or_separated_digits_are_left_as_they_are(text: str) -> None:
    fields = {"evidence": f"said {text}", "description": f"Reachable at {text}."}

    assert _scrub(fields) == fields


def test_scrub_returns_a_copy_and_leaves_its_input_alone() -> None:
    fields = {"evidence": f"ID {_ID_CARD}", "description": "d"}

    _scrub(fields)

    assert fields == {"evidence": f"ID {_ID_CARD}", "description": "d"}


def _episode(text: str, timestamp: int = _TS) -> Episode:
    return Episode(owner_id=None, episode=text, summary=text, timestamp=timestamp)


async def test_episode_init_drops_and_masks_and_logs_no_number(caplog: pytest.LogCaptureFixture) -> None:
    reply = {
        "explicit_info": [
            {"evidence": f"Mara gave her ID {_ID_CARD}", "description": "Lives in Ghent.", "category": "Identity"},
            {"evidence": "e", "description": f"Bank card {_BANK_CARD}.", "category": "Finance"},
            {"evidence": f"Mara gave her mobile {_PHONE}", "description": "Prefers calls.", "category": "Contact"},
        ],
        "implicit_traits": [],
        "do_not_record": [],
    }
    fake = FakeLLMClient(responses=[json.dumps(reply)])

    with caplog.at_level(logging.DEBUG, logger=_PROFILE_LOGGER):
        profile = await ProfileExtractor(llm=fake, item_checks=False).aextract_from_episodes(
            [_episode("Mara filled in a form.")], owner_id="u1", owner_name="Mara"
        )

    items = [item.model_dump(exclude_unset=True) for item in profile.explicit_info]
    assert [item["description"] for item in items] == ["Lives in Ghent.", "Prefers calls."]
    assert items[0]["evidence"] == "Mara gave her ID ***"
    assert items[1]["evidence"] == f"Mara gave her mobile {_PHONE}"
    # One line per dropped item, following the package's log contract: an event, then key=value pairs.
    assert _log_fields(caplog, "profile item dropped:") == [{"reason": "description carries an account number"}]
    for number in (_ID_CARD, _BANK_CARD, "6222"):
        assert number not in caplog.text


async def test_memcell_init_drops_and_masks_too() -> None:
    reply = {
        "explicit_info": [
            {"evidence": f"Alice gave her ID {_ID_CARD}", "description": "Lives in Ghent.", "category": "Identity"},
            {"evidence": "e", "description": f"Insurance number {_INSURANCE}.", "category": "Health"},
        ],
        "implicit_traits": [],
    }
    message = ChatMessage(id="m1", role="user", content="hi", timestamp=_TS, sender_id="u_alice", sender_name="Alice")
    fake = FakeLLMClient(responses=[json.dumps(reply)])

    profile = await ProfileExtractor(llm=fake).aextract([MemCell(items=[message], timestamp=_TS)], sender_id="u_alice")

    assert [(item.description, item.evidence) for item in profile.explicit_info] == [
        ("Lives in Ghent.", "Alice gave her ID ***")
    ]


def _stored() -> Profile:
    return Profile.model_validate(
        {
            "owner_id": "u1",
            "summary": "s",
            "timestamp": _TS,
            "explicit_info": [
                {"evidence": "said so", "description": "Has health insurance.", "category": "Health"},
                {"evidence": "said so", "description": "Lives in Ghent.", "category": "Identity"},
            ],
        }
    )


async def test_update_drops_a_numbered_add_refuses_a_numbered_rewrite_and_masks_grounding(
    caplog: pytest.LogCaptureFixture,
) -> None:
    reply = {
        "operations": [
            {
                "action": "add",
                "type": "explicit_info",
                "data": {"evidence": "e", "description": f"Passport {_INSURANCE}.", "category": "Identity"},
            },
            {
                "action": "update",
                "type": "explicit_info",
                "index": 0,
                "data": {"evidence": "e", "description": f"Insurance number {_INSURANCE}.", "category": "Health"},
            },
            {
                "action": "update",
                "type": "explicit_info",
                "index": 1,
                "data": {
                    "evidence": f"said so; gave ID {_ID_CARD}",
                    "description": "Lives in Ghent.",
                    "category": "Identity",
                },
            },
        ]
    }
    fake = FakeLLMClient(responses=[json.dumps(reply)])

    with caplog.at_level(logging.WARNING, logger=_PROFILE_LOGGER):
        profile = await ProfileExtractor(llm=fake, item_checks=False).aextract_from_episodes(
            [_episode("Mara gave her numbers.", _LATER)], owner_id="u1", owner_name="Mara", old_profile=_stored()
        )

    items = [item.model_dump(exclude_unset=True) for item in profile.explicit_info]
    assert [item["description"] for item in items] == ["Has health insurance.", "Lives in Ghent."]
    assert items[0]["evidence"] == "said so"  # the refused rewrite left the stored item as it was
    assert items[1]["evidence"] == "said so; gave ID ***"
    assert "reason=description carries an account number" in caplog.text
    assert _INSURANCE not in caplog.text


async def test_a_compact_operation_is_held_to_the_same_rule() -> None:
    stored = Profile.model_validate(
        {
            "owner_id": "u1",
            "summary": "s",
            "timestamp": _TS,
            "explicit_info": [
                {"evidence": f"said fact {i}", "description": f"Fact {i}.", "category": f"Topic {i}"}
                for i in range(_PROFILE_MAX_ITEMS + 1)
            ],
        }
    )
    merged: dict[str, Any] = {"evidence": f"fact 0 and ID {_ID_CARD}", "description": "Facts 0 and 1.", "category": "T"}
    compact = {
        "operations": [
            {"action": "merge", "type": "explicit_info", "indices": [0, 1], "data": merged},
            {
                "action": "merge",
                "type": "explicit_info",
                "indices": [2, 3],
                "data": {"evidence": "e", "description": f"Card {_BANK_CARD}.", "category": "T"},
            },
        ]
    }
    fake = FakeLLMClient(responses=[json.dumps({"operations": [{"action": "none"}]}), json.dumps(compact)])

    profile = await ProfileExtractor(llm=fake).aextract_from_episodes(
        [_episode("Mara said nothing new.", _LATER)], owner_id="u1", owner_name="Mara", old_profile=stored
    )

    items = {item.description: item for item in profile.explicit_info}
    assert items["Facts 0 and 1."].evidence == "fact 0 and ID ***"
    assert "Fact 2." in items  # the refused merge left both of its items as stored
    assert "Fact 3." in items
    assert not any("6222" in description for description in items)
