"""No new implicit_traits item is created; a stored trait can still be updated or deleted.

INIT keeps none of the traits its reply holds and an UPDATE add into implicit_traits is refused, on both paths, unless
the extractor is built with ``new_traits=True``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import ChatMessage, Episode, MemCell, Profile
from everalgo.user_memory import ProfileExtractor

_TS = 1_780_000_000_000
_LATER = _TS + 86_400_000
_PROFILE_LOGGER = "everalgo.user_memory.profile"
_EXPLICIT = {"evidence": "said she runs a bookshop", "description": "Runs a bookshop.", "category": "Identity"}
_TRAIT = {"basis": "asked twice to check the numbers", "description": "Checks numbers first.", "trait": "Careful"}


def _episode(timestamp: int = _TS) -> Episode:
    return Episode(owner_id=None, episode="Mara talked about her bookshop.", summary="s", timestamp=timestamp)


def _memcell() -> MemCell:
    message = ChatMessage(
        id="m1", role="user", content="I run a bookshop.", timestamp=_TS, sender_id="u_mara", sender_name="Mara"
    )
    return MemCell(items=[message], timestamp=_TS)


async def _init(path: str, reply: dict[str, Any], *, new_traits: bool = False) -> Profile:
    fake = FakeLLMClient(responses=[json.dumps(reply)])
    extractor = ProfileExtractor(llm=fake, item_checks=False, new_traits=new_traits)
    if path == "memcell":
        return await extractor.aextract([_memcell()], sender_id="u_mara")
    return await extractor.aextract_from_episodes([_episode()], owner_id="u_mara", owner_name="Mara")


def _init_reply(path: str) -> dict[str, Any]:
    reply: dict[str, Any] = {"explicit_info": [_EXPLICIT], "implicit_traits": [_TRAIT]}
    return reply if path == "memcell" else {**reply, "do_not_record": []}


@pytest.mark.parametrize("path", ["memcell", "episode"])
async def test_init_keeps_the_facts_and_none_of_the_traits(path: str) -> None:
    profile = await _init(path, _init_reply(path))

    assert [item.description for item in profile.explicit_info] == ["Runs a bookshop."]
    assert profile.implicit_traits == []


@pytest.mark.parametrize("path", ["memcell", "episode"])
async def test_init_keeps_traits_when_new_traits_are_allowed(path: str) -> None:
    profile = await _init(path, _init_reply(path), new_traits=True)

    assert [item.description for item in profile.implicit_traits] == ["Checks numbers first."]


def _stored() -> Profile:
    return Profile.model_validate(
        {
            "owner_id": "u_mara",
            "summary": "s",
            "timestamp": _TS,
            "explicit_info": [_EXPLICIT],
            "implicit_traits": [
                _TRAIT,
                {"basis": "b", "description": "Plans ahead.", "trait": "Planner"},
            ],
        }
    )


async def _update(path: str, *operations: dict[str, Any]) -> Profile:
    reply: dict[str, Any] = {"operations": list(operations)}
    if path == "memcell":
        reply["update_note"] = "applied"
    fake = FakeLLMClient(responses=[json.dumps(reply)])
    extractor = ProfileExtractor(llm=fake, item_checks=False)
    if path == "memcell":
        return await extractor.aextract([_memcell()], sender_id="u_mara", old_profile=_stored())
    return await extractor.aextract_from_episodes(
        [_episode(_LATER)], owner_id="u_mara", owner_name="Mara", old_profile=_stored()
    )


@pytest.mark.parametrize("path", ["memcell", "episode"])
async def test_update_refuses_an_added_trait_and_applies_the_rest(path: str, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger=_PROFILE_LOGGER):
        profile = await _update(
            path,
            {"action": "add", "type": "implicit_traits", "data": {"basis": "b", "description": "New.", "trait": "T"}},
            {
                "action": "add",
                "type": "explicit_info",
                "data": {"evidence": "e", "description": "Lives in Porto.", "category": "Identity"},
            },
        )

    assert [item.description for item in profile.explicit_info] == ["Runs a bookshop.", "Lives in Porto."]
    assert [item.description for item in profile.implicit_traits] == ["Checks numbers first.", "Plans ahead."]
    assert "action=add type=implicit_traits index=None reason=new traits disabled" in caplog.text


@pytest.mark.parametrize("path", ["memcell", "episode"])
async def test_a_stored_trait_can_still_be_updated_or_deleted(path: str) -> None:
    profile = await _update(
        path,
        {
            "action": "update",
            "type": "implicit_traits",
            "index": 0,
            "data": {**_TRAIT, "description": "Checks every number before deciding."},
        },
        {"action": "delete", "type": "implicit_traits", "index": 1, "reason": "contradicted"},
    )

    assert [(item.trait, item.description) for item in profile.implicit_traits] == [
        ("Careful", "Checks every number before deciding.")
    ]
