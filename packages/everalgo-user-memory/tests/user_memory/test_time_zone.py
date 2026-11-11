"""``tz=`` on every user-memory entry point that shows the model a time: the zone reaches the prompt.

10:00 UTC on Sunday 2024-03-10 is 00:00 on Monday 2024-03-11 in UTC+14, so a time shown in that zone moves its date,
clock and weekday and carries the offset; UTC (the default) renders as before. The episode prompt's own coverage is in
``test_episode_prompt_split.py``.
"""

from __future__ import annotations

import json
from datetime import UTC, timedelta, timezone

import pytest

from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import ChatMessage, Episode, ExplicitInfoItem, MemCell, Profile
from everalgo.user_memory import AtomicFactExtractor, ForesightExtractor, ProfileExtractor
from everalgo.user_memory.profile._render import _render_bucket, _render_indexed_items, _render_pending

_SUNDAY_10_00_UTC = 1_710_064_800_000  # 2024-03-10T10:00:00Z
_UTC_PLUS_14 = timezone(timedelta(hours=14))
_LOCAL = "2024-03-11 00:00 UTC+14:00 (Monday)"
_LOCAL_NO_WEEKDAY = "2024-03-11 00:00 UTC+14:00"


def _memcell() -> MemCell:
    message = ChatMessage(
        id="m1",
        role="user",
        content="I will fly to Paris next week.",
        timestamp=_SUNDAY_10_00_UTC,
        sender_id="u_alice",
        sender_name="Alice",
    )
    return MemCell(items=[message], timestamp=_SUNDAY_10_00_UTC)


def _prompt(fake: FakeLLMClient) -> str:
    return "\n".join(str(message.content) for message in fake.calls[0].messages)


async def test_atomic_facts_from_a_memcell_show_the_time_and_every_line_in_the_zone() -> None:
    fake = FakeLLMClient(responses=[json.dumps({"atomic_facts": {"atomic_fact": ["Alice will fly to Paris."]}})])

    await AtomicFactExtractor(llm=fake).aextract(_memcell(), sender_id="u_alice", tz=_UTC_PLUS_14)

    prompt = _prompt(fake)
    assert f"[{_LOCAL}] Alice: I will fly to Paris next week." in prompt
    assert prompt.count(_LOCAL) == 2  # the <time> anchor and the one chat line


async def test_atomic_facts_from_text_show_the_time_in_the_zone() -> None:
    fake = FakeLLMClient(responses=[json.dumps({"atomic_facts": {"atomic_fact": ["Alice will fly to Paris."]}})])

    await AtomicFactExtractor(llm=fake).aextract_from_text(
        "Alice will fly to Paris next week.", timestamp=_SUNDAY_10_00_UTC, tz=_UTC_PLUS_14
    )

    assert f"<time>\n{_LOCAL}\n</time>" in str(fake.calls[0].messages[1].content)


async def test_foresight_shows_lines_in_the_zone_and_dates_a_missing_start_in_its_calendar() -> None:
    reply = {
        "foresights": [
            {
                "content": "Alice will be in Paris.",
                "evidence": "Alice said she will fly to Paris next week.",
                "start_time": None,
                "end_time": None,
                "duration_days": None,
            }
        ]
    }
    fake = FakeLLMClient(responses=[json.dumps(reply)])

    foresights = await ForesightExtractor(llm=fake).aextract(_memcell(), sender_id="u_alice", tz=_UTC_PLUS_14)

    assert f"[{_LOCAL}] Alice: I will fly to Paris next week." in _prompt(fake)
    assert foresights[0].start_time == "2024-03-11"  # the first message's date in UTC+14, not 2024-03-10


async def test_profile_from_memcells_shows_lines_in_the_zone() -> None:
    fake = FakeLLMClient(responses=[json.dumps({"explicit_info": [], "implicit_traits": []})])

    await ProfileExtractor(llm=fake).aextract([_memcell()], sender_id="u_alice", tz=_UTC_PLUS_14)

    assert f"[{_LOCAL}] Alice(user_id:u_alice): I will fly to Paris next week." in _prompt(fake)


async def test_profile_from_episodes_shows_narrative_times_in_the_zone() -> None:
    fake = FakeLLMClient(responses=[json.dumps({"explicit_info": [], "implicit_traits": [], "do_not_record": []})])
    episode = Episode(
        owner_id="u_alice", episode="Alice plans a trip to Paris.", summary="", timestamp=_SUNDAY_10_00_UTC
    )

    await ProfileExtractor(llm=fake).aextract_from_episodes(
        [episode], owner_id="u_alice", owner_name="Alice", tz=_UTC_PLUS_14
    )

    assert f'<narrative time="{_LOCAL_NO_WEEKDAY}">Alice plans a trip to Paris.</narrative>' in _prompt(fake)


@pytest.mark.parametrize(("tz", "written"), [(_UTC_PLUS_14, ["Swims weekly."]), (UTC, [])])
async def test_a_confirm_dated_by_the_shown_day_counts_in_the_zone(tz: timezone, written: list[str]) -> None:
    """The model dates its grounding by the day it was shown: 2024-03-11 is this batch's day in UTC+14 only."""
    pending = [{"description": "Swims.", "category": "hobby", "evidence": "swam", "observed_at": _SUNDAY_10_00_UTC - 1}]
    old = Profile.model_validate(
        {"owner_id": "u_alice", "summary": "s", "timestamp": _SUNDAY_10_00_UTC - 1, "pending": pending}
    )
    confirm = {
        "action": "confirm",
        "type": "explicit_info",
        "pending": 0,
        "data": {
            "evidence": "2024-03-11: swam again",
            "description": "Swims weekly.",
            "category": "hobby",
            "quote": "Alice swam again this week",
            "pattern": True,
        },
    }
    fake = FakeLLMClient(responses=[json.dumps({"operations": [confirm]})])
    episode = Episode(
        owner_id="u_alice", episode="Alice swam again this week.", summary="", timestamp=_SUNDAY_10_00_UTC
    )

    profile = await ProfileExtractor(llm=fake).aextract_from_episodes(
        [episode], owner_id="u_alice", owner_name="Alice", old_profile=old, tz=tz
    )

    assert [i.description for i in profile.explicit_info] == written
    assert (profile.model_extra or {})["pending"] == ([] if written else pending)


def test_pending_entries_show_their_first_seen_day_in_the_zone() -> None:
    entry = {"description": "Swims.", "category": "hobby", "evidence": "", "observed_at": _SUNDAY_10_00_UTC}

    assert 'first_seen="2024-03-11"' in _render_pending([entry], tz=_UTC_PLUS_14)
    assert 'first_seen="2024-03-10"' in _render_pending([entry])


def test_stored_items_show_their_observation_time_in_the_zone() -> None:
    """Both UPDATE renderings date each stored item in the zone passed."""
    item = ExplicitInfoItem(
        description="Works mainly in Python.", category="work", evidence="", observed_at=_SUNDAY_10_00_UTC
    )

    assert f'"observed_at": "{_LOCAL_NO_WEEKDAY}"' in _render_indexed_items([item], [], tz=_UTC_PLUS_14)
    assert f'"observed_at": "{_LOCAL_NO_WEEKDAY}"' in _render_bucket("explicit_info", [item], tz=_UTC_PLUS_14)
