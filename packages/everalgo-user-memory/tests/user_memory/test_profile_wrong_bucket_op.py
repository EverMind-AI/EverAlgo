"""An UPDATE add or update whose item is shaped for the other bucket is dropped; the reply's other operations apply.

It used to fail the whole reply with LLMOutputError, which lost every other operation of the batch. The structural
model accepts either item shape in an operation's ``data``, so the check runs after parsing, in the UPDATE callsite.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import pytest

from everalgo.llm import LLMOutputError
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import Episode, Profile
from everalgo.user_memory import ProfileExtractor
from everalgo.user_memory.profile._replies import _call_llm_for_profile_update, _UpdateOutput

_TS = 1_780_000_000_000
_LATER = _TS + 86_400_000
_PROFILE_LOGGER = "everalgo.user_memory.profile"
_EXPLICIT = {"evidence": "said she lives in Ghent", "description": "Lives in Ghent.", "category": "Identity"}
_TRAIT = {"basis": "checked the numbers twice", "description": "Checks numbers first.", "trait": "Careful"}
_LOG_PAIR = re.compile(r"(\w+)=(.*?)(?= \w+=|$)")


def _log_fields(caplog: pytest.LogCaptureFixture, prefix: str) -> list[dict[str, str]]:
    """The ``key=value`` pairs of every captured line opening with ``prefix``, as the package docstring defines them."""
    return [
        dict(_LOG_PAIR.findall(record.getMessage().removeprefix(prefix).strip()))
        for record in caplog.records
        if record.getMessage().startswith(prefix)
    ]


def _stored() -> Profile:
    return Profile.model_validate(
        {
            "owner_id": "u1",
            "summary": "s",
            "timestamp": _TS,
            "explicit_info": [_EXPLICIT, {"evidence": "e", "description": "Works as a nurse.", "category": "Work"}],
            "implicit_traits": [_TRAIT],
        }
    )


async def _update(*operations: dict[str, Any]) -> Profile:
    fake = FakeLLMClient(responses=[json.dumps({"operations": list(operations)})])
    return await ProfileExtractor(llm=fake, item_checks=False).aextract_from_episodes(
        [Episode(owner_id=None, episode="Mara moved to Porto.", summary="s", timestamp=_LATER)],
        owner_id="u1",
        owner_name="Mara",
        old_profile=_stored(),
    )


async def test_an_update_into_traits_carrying_a_fact_is_dropped_and_the_add_applies(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The case seen on real data: an update into implicit_traits that carries an explicit_info item."""
    with caplog.at_level(logging.WARNING, logger=_PROFILE_LOGGER):
        profile = await _update(
            {"action": "update", "type": "implicit_traits", "index": 0, "data": {**_EXPLICIT, "description": "x."}},
            {
                "action": "add",
                "type": "explicit_info",
                "data": {"evidence": "e", "description": "Married to Lin.", "category": "Family"},
            },
        )

    assert [item.description for item in profile.explicit_info] == [
        "Lives in Ghent.",
        "Works as a nurse.",
        "Married to Lin.",
    ]
    assert [item.model_dump(exclude_unset=True) for item in profile.implicit_traits] == [{**_TRAIT, "observed_at": _TS}]
    # The warning follows the package's log contract: an event, then key=value pairs, reason last.
    assert _log_fields(caplog, "profile update ops dropped:") == [
        {"count": "1", "reason": "item shaped for the other bucket"}
    ]


async def test_an_add_into_facts_carrying_a_trait_is_dropped_and_the_delete_applies() -> None:
    profile = await _update(
        {"action": "add", "type": "explicit_info", "data": {**_TRAIT, "description": "Plans ahead."}},
        {"action": "delete", "type": "explicit_info", "index": 1, "reason": "changed jobs"},
    )

    assert [item.description for item in profile.explicit_info] == ["Lives in Ghent."]
    assert [item.description for item in profile.implicit_traits] == ["Checks numbers first."]


async def test_the_callsite_returns_the_reply_without_the_dropped_operations() -> None:
    kept = {"action": "delete", "type": "implicit_traits", "index": 0, "reason": "contradicted"}
    reply = json.dumps(
        {
            "operations": [
                {"action": "add", "type": "implicit_traits", "data": _EXPLICIT},
                kept,
                {"action": "update", "type": "explicit_info", "index": 0, "data": _TRAIT},
            ]
        }
    )

    data = await _call_llm_for_profile_update(FakeLLMClient(responses=[reply]), [], reply=_UpdateOutput)

    assert data == {"operations": [kept]}


async def test_a_reply_that_fails_its_model_still_raises() -> None:
    """Only the bucket check turned lenient; an item with a key neither shape has still fails the reply."""
    reply = json.dumps(
        {"operations": [{"action": "add", "type": "explicit_info", "data": {**_EXPLICIT, "trait": "T"}}]}
    )

    with pytest.raises(LLMOutputError):
        await _call_llm_for_profile_update(FakeLLMClient(responses=[reply]), [], reply=_UpdateOutput)
