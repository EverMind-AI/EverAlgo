"""Episode-path COMPACT and REGROUP reply with operations on indexed items.

Every item no operation names comes back as the stored object: its host keys (ids, provenance) and its
``observed_at`` survive. A merge keeps the first named item's host keys and the newest date of the items it merges;
a split's first piece keeps the item's host keys and every piece its date; an update keeps both; a move keeps only
the trait's date. An operation naming an index twice, out of range, on a bucket the stage was not given, or carrying
the other bucket's item is refused alone, with a warning. Duplicates the operations leave are removed.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import pytest

from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import Episode, ExplicitInfoItem, Profile
from everalgo.user_memory import ProfileExtractor
from everalgo.user_memory.profile._compact_ops import apply_compact_ops
from everalgo.user_memory.profile._items import _PROFILE_MAX_ITEMS, _PROFILE_MAX_PER_CATEGORY

_DAY = 86_400_000
_T0 = 1_780_000_000_000
_STORED_AT = _T0 + 100 * _DAY
_BATCH_AT = _T0 + 200 * _DAY
_PROFILE_LOGGER = "everalgo.user_memory.profile"
_LOG_PAIR = re.compile(r"(\w+)=(.*?)(?= \w+=|$)")


def _log_fields(caplog: pytest.LogCaptureFixture, prefix: str) -> list[dict[str, str]]:
    """The ``key=value`` pairs of every captured line opening with ``prefix``, as the package docstring defines them."""
    return [
        dict(_LOG_PAIR.findall(record.getMessage().removeprefix(prefix).strip()))
        for record in caplog.records
        if record.getMessage().startswith(prefix)
    ]


def _explicit(index: int, category: str | None = None) -> dict[str, Any]:
    return {
        "item_id": f"e{index}",
        "evidence": f"said fact {index}",
        "description": f"Fact {index}.",
        "category": category or f"Topic {index}",
        "observed_at": _T0 + index * _DAY,
    }


def _trait(index: int, trait: str = "Careful") -> dict[str, Any]:
    return {
        "item_id": f"t{index}",
        "basis": f"signal {index}",
        "description": f"Pattern {index}.",
        "trait": trait,
        "observed_at": _T0 + index * _DAY,
    }


def _profile(explicit_info: list[dict[str, Any]], implicit_traits: list[dict[str, Any]]) -> Profile:
    return Profile.model_validate(
        {
            "owner_id": "u1",
            "summary": "s",
            "timestamp": _STORED_AT,
            "explicit_info": explicit_info,
            "implicit_traits": implicit_traits,
        }
    )


def _episode(text: str = "Mara said nothing new.") -> Episode:
    return Episode(owner_id="u1", episode=text, subject="s", summary="", timestamp=_BATCH_AT)


async def _update(old: Profile, *replies: dict[str, Any]) -> tuple[FakeLLMClient, Profile]:
    fake = FakeLLMClient(responses=[json.dumps(reply) for reply in replies])
    profile = await ProfileExtractor(llm=fake).aextract_from_episodes(
        [_episode()], owner_id="u1", owner_name="Mara", old_profile=old
    )
    return fake, profile


def _by_description(items: list[Any]) -> dict[str, dict[str, Any]]:
    return {item.description: item.model_dump(exclude_unset=True) for item in items}


def _over_the_total_cap() -> Profile:
    """Sixty explicit items, each under its own category, and one trait: one over the total cap."""
    explicit = [_explicit(index) for index in range(_PROFILE_MAX_ITEMS)]
    assert len(explicit) + 1 > _PROFILE_MAX_ITEMS
    return _profile(explicit, [_trait(0)])


async def test_compact_operations_keep_ids_and_dates_of_what_they_do_not_name() -> None:
    compact = {
        "operations": [
            {
                "action": "merge",
                "type": "explicit_info",
                "indices": [0, 3],
                "data": {
                    "evidence": "said fact 0; said fact 3",
                    "description": "Facts 0 and 3.",
                    "category": "Topic 0",
                },
            },
            {"action": "delete", "type": "explicit_info", "index": 1, "reason": "trivial"},
            {
                "action": "split",
                "type": "explicit_info",
                "index": 2,
                "items": [
                    {"evidence": "said fact 2", "description": "Fact 2a.", "category": "Topic 2"},
                    {"evidence": "said fact 2", "description": "Fact 2b.", "category": "Topic 2"},
                ],
            },
            {
                "action": "update",
                "type": "explicit_info",
                "index": 4,
                "data": {"evidence": "said fact 4", "description": "Fact 4.", "category": "Renamed"},
            },
            {
                "action": "move",
                "type": "implicit_traits",
                "index": 0,
                "data": {"evidence": "signal 0", "description": "Pattern 0.", "category": "Habits"},
            },
        ]
    }

    fake, profile = await _update(_over_the_total_cap(), {"operations": [{"action": "none"}]}, compact)

    assert fake.call_count == 2  # UPDATE, COMPACT
    _, compact_user = fake.calls[1].messages
    assert '<item index="0">' in str(compact_user.content)
    items = _by_description(profile.explicit_info)
    # merge: the first named item's host keys, the newest date of the two.
    assert items["Facts 0 and 3."]["item_id"] == "e0"
    assert items["Facts 0 and 3."]["observed_at"] == _T0 + 3 * _DAY
    assert "Fact 1." not in items
    assert "Fact 3." not in items
    # split: the first piece carries on the item, every piece keeps its date.
    assert items["Fact 2a."]["item_id"] == "e2"
    assert "item_id" not in items["Fact 2b."]
    assert items["Fact 2a."]["observed_at"] == items["Fact 2b."]["observed_at"] == _T0 + 2 * _DAY
    # update: re-filed, host keys and date kept.
    assert (items["Fact 4."]["category"], items["Fact 4."]["item_id"]) == ("Renamed", "e4")
    assert items["Fact 4."]["observed_at"] == _T0 + 4 * _DAY
    # move: the trait becomes a new explicit_info item with the trait's date and none of its host keys. A host mints
    # ids per bucket (the cloud's implicit_traits ids cannot address an explicit_info item), so the moved item must
    # arrive without the trait's id and get a fresh one, as it did after a rewrite COMPACT.
    assert profile.implicit_traits == []
    assert "item_id" not in items["Pattern 0."]
    assert items["Pattern 0."]["observed_at"] == _T0
    assert items["Pattern 0."]["evidence"] == "signal 0"
    # Every item no operation named is the stored item.
    for index in range(5, _PROFILE_MAX_ITEMS):
        assert items[f"Fact {index}."] == _explicit(index)


async def test_compact_refuses_an_operation_reusing_an_index_or_out_of_range_and_applies_the_rest(
    caplog: pytest.LogCaptureFixture,
) -> None:
    compact = {
        "operations": [
            {"action": "delete", "type": "explicit_info", "index": 1, "reason": "trivial"},
            {
                "action": "merge",
                "type": "explicit_info",
                "indices": [1, 2],
                "data": {"evidence": "e", "description": "Merged.", "category": "Topic 1"},
            },
            {"action": "delete", "type": "explicit_info", "index": 99, "reason": "trivial"},
            {
                "action": "move",
                "type": "explicit_info",
                "index": 5,
                "data": {"evidence": "said fact 5", "description": "Fact 5.", "category": "Topic 5"},
            },
        ]
    }

    with caplog.at_level(logging.WARNING, logger=_PROFILE_LOGGER):
        _, profile = await _update(_over_the_total_cap(), {"operations": [{"action": "none"}]}, compact)

    descriptions = [item.description for item in profile.explicit_info]
    assert "Fact 1." not in descriptions
    assert "Fact 2." in descriptions
    assert "Merged." not in descriptions
    assert "Fact 5." in descriptions  # a move takes a trait only
    assert _log_fields(caplog, "profile compact ops refused:") == [{"count": "3"}]


async def test_a_merge_carrying_the_other_bucket_item_is_refused_and_its_items_stay() -> None:
    compact = {
        "operations": [
            {
                "action": "merge",
                "type": "explicit_info",
                "indices": [0, 1],
                "data": {"basis": "b", "description": "Merged.", "trait": "Careful"},
            },
            {"action": "delete", "type": "explicit_info", "index": 2, "reason": "trivial"},
        ]
    }

    _, profile = await _update(_over_the_total_cap(), {"operations": [{"action": "none"}]}, compact)

    items = _by_description(profile.explicit_info)
    assert items["Fact 0."] == _explicit(0)
    assert items["Fact 1."] == _explicit(1)
    assert "Fact 2." not in items
    assert "Merged." not in items


async def test_regroup_operations_keep_ids_and_dates_of_the_group() -> None:
    crowded = [_explicit(index, "Work") for index in range(_PROFILE_MAX_PER_CATEGORY + 1)]
    regroup = {
        "operations": [
            {
                "action": "update",
                "type": "explicit_info",
                "index": 4,
                "data": {"evidence": "said fact 4", "description": "Fact 4.", "category": "Career"},
            }
        ]
    }

    fake, profile = await _update(_profile(crowded, []), {"operations": [{"action": "none"}]}, regroup)

    assert fake.call_count == 2  # UPDATE, REGROUP
    _, regroup_user = fake.calls[1].messages
    assert '<group bucket="explicit_info" label_field="category" label="Work" count="9">' in str(regroup_user.content)
    items = _by_description(profile.explicit_info)
    assert len(items) == len(crowded)
    assert items["Fact 4."] == {**_explicit(4, "Work"), "category": "Career"}
    for index in (0, 1, 2, 3, 5, 6, 7, 8):
        assert items[f"Fact {index}."] == _explicit(index, "Work")


async def test_regroup_of_a_trait_group_merges_into_the_first_named_trait() -> None:
    traits = [_trait(index) for index in range(_PROFILE_MAX_PER_CATEGORY + 1)]
    regroup = {
        "operations": [
            {
                "action": "merge",
                "type": "implicit_traits",
                "indices": [2, 6],
                "data": {"basis": "signal 2; signal 6", "description": "Patterns 2 and 6.", "trait": "Careful"},
            }
        ]
    }

    _, profile = await _update(_profile([], traits), {"operations": [{"action": "none"}]}, regroup)

    items = _by_description(profile.implicit_traits)
    assert len(items) == len(traits) - 1
    assert items["Patterns 2 and 6."]["item_id"] == "t2"
    assert items["Patterns 2 and 6."]["observed_at"] == _T0 + 6 * _DAY
    assert items["Pattern 0."] == _trait(0)


def test_apply_compact_ops_dates_an_undated_item_only_through_the_buckets_it_is_given() -> None:
    """The extractor passes dated buckets; the function itself writes no date onto an item it does not name."""
    undated = ExplicitInfoItem.model_validate({"description": "Undated.", "category": "c", "evidence": "e"})
    named = ExplicitInfoItem.model_validate(
        {"description": "Named.", "category": "c", "evidence": "e", "observed_at": _T0, "item_id": "n"}
    )
    ops = [
        {
            "action": "update",
            "type": "explicit_info",
            "index": 1,
            "data": {"evidence": "e2", "description": "Named.", "category": "d"},
        }
    ]

    result = apply_compact_ops({"explicit_info": [undated, named]}, ops, source="compact")["explicit_info"]

    assert result[0] is undated
    assert result[1].model_dump(exclude_unset=True) == {
        "description": "Named.",
        "category": "d",
        "evidence": "e2",
        "observed_at": _T0,
        "item_id": "n",
    }


async def test_compact_and_regroup_log_lines_follow_the_key_value_contract(caplog: pytest.LogCaptureFixture) -> None:
    """Each line is an event, then key=value pairs of counts and lengths, as the package docstring lists them."""
    explicit = [_explicit(index, "Work") for index in range(_PROFILE_MAX_PER_CATEGORY + 1)]
    explicit += [_explicit(index) for index in range(len(explicit), _PROFILE_MAX_ITEMS + 1)]
    compact = {"operations": [{"action": "delete", "type": "explicit_info", "index": 20, "reason": "trivial"}]}
    regroup = {"operations": [{"action": "delete", "type": "explicit_info", "index": 0, "reason": "trivial"}]}

    with caplog.at_level(logging.INFO, logger=_PROFILE_LOGGER):
        await _update(_profile(explicit, []), {"operations": [{"action": "none"}]}, compact, regroup)

    assert _log_fields(caplog, "profile compact ops applied:") == [{"items_before": "61", "items_after": "60"}]
    assert _log_fields(caplog, "profile regroup ops applied:") == [
        {"bucket": "explicit_info", "label_len": "4", "group_before": "9", "group_after": "8"}
    ]
    assert _log_fields(caplog, "profile regroup left label over cap:") == []


async def test_a_regroup_that_leaves_the_group_over_its_cap_is_warned_about(caplog: pytest.LogCaptureFixture) -> None:
    """As with a MemCell-path rewrite, a group still over its cap is accepted with a warning; regroup never loops."""
    crowded = [_explicit(index, "Work") for index in range(_PROFILE_MAX_PER_CATEGORY + 1)]

    with caplog.at_level(logging.WARNING, logger=_PROFILE_LOGGER):
        fake, profile = await _update(_profile(crowded, []), {"operations": [{"action": "none"}]}, {"operations": []})

    assert fake.call_count == 2  # UPDATE, one REGROUP
    assert len(profile.explicit_info) == len(crowded)
    assert _log_fields(caplog, "profile regroup left label over cap:") == [
        {"bucket": "explicit_info", "label_len": "4", "count": "9"}
    ]


def test_a_split_whose_first_piece_is_refused_keeps_the_host_keys_on_the_first_piece_that_builds() -> None:
    stored = ExplicitInfoItem.model_validate(_explicit(2))
    ops = [
        {
            "action": "split",
            "type": "explicit_info",
            "index": 0,
            "items": [
                {"basis": "b", "description": "Shaped as a trait.", "trait": "T"},
                {"evidence": "said fact 2", "description": "Fact 2a.", "category": "Topic 2"},
                {"evidence": "said fact 2", "description": "Fact 2b.", "category": "Topic 2"},
            ],
        }
    ]

    first, second = apply_compact_ops({"explicit_info": [stored]}, ops, source="compact")["explicit_info"]

    assert first.model_dump(exclude_unset=True) == {
        "evidence": "said fact 2",
        "description": "Fact 2a.",
        "category": "Topic 2",
        "item_id": "e2",
        "observed_at": _T0 + 2 * _DAY,
    }
    assert second.model_dump(exclude_unset=True) == {
        "evidence": "said fact 2",
        "description": "Fact 2b.",
        "category": "Topic 2",
        "observed_at": _T0 + 2 * _DAY,
    }


def _crowded_group() -> Profile:
    """Nine explicit_info items under "Work", one over the per-label cap, and one stored trait."""
    return _profile([_explicit(index, "Work") for index in range(_PROFILE_MAX_PER_CATEGORY + 1)], [_trait(0)])


def _merge_op(bucket: str, indices: list[int], description: str) -> dict[str, Any]:
    data = (
        {"evidence": "e", "description": description, "category": "Work"}
        if bucket == "explicit_info"
        else {"basis": "b", "description": description, "trait": "Careful"}
    )
    return {"action": "merge", "type": bucket, "indices": indices, "data": data}


@pytest.mark.parametrize("stage", ["compact", "regroup"])
async def test_a_merge_naming_one_index_twice_is_refused_and_the_item_stays(
    stage: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Applied, it would put the merged item at index 3 and then remove index 3 as a merged-away duplicate."""
    old = _over_the_total_cap() if stage == "compact" else _crowded_group()
    stored = _by_description(old.explicit_info)["Fact 3."]
    reply = {"operations": [_merge_op("explicit_info", [3, 3], "Merged.")]}

    with caplog.at_level(logging.WARNING, logger=_PROFILE_LOGGER):
        fake, profile = await _update(old, {"operations": [{"action": "none"}]}, reply)

    assert fake.call_count == 2
    items = _by_description(profile.explicit_info)
    assert items["Fact 3."] == stored
    assert "Merged." not in items
    assert _log_fields(caplog, f"profile {stage} ops refused:") == [{"count": "1"}]


async def test_a_regroup_operation_on_the_other_bucket_is_refused_and_the_rest_apply(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """REGROUP is given its group's bucket only; an operation naming the other one is refused, not a KeyError."""
    reply = {
        "operations": [
            {"action": "delete", "type": "implicit_traits", "index": 0, "reason": "trivial"},
            {
                "action": "update",
                "type": "implicit_traits",
                "index": 0,
                "data": {"basis": "b", "description": "Changed.", "trait": "Careful"},
            },
            _merge_op("implicit_traits", [0, 1], "Merged trait."),
            {
                "action": "split",
                "type": "implicit_traits",
                "index": 0,
                "items": [{"basis": "b", "description": "Piece.", "trait": "Careful"}],
            },
            {"action": "delete", "type": "explicit_info", "index": 8, "reason": "trivial"},
        ]
    }

    with caplog.at_level(logging.WARNING, logger=_PROFILE_LOGGER):
        _, profile = await _update(_crowded_group(), {"operations": [{"action": "none"}]}, reply)

    assert [item.model_dump(exclude_unset=True) for item in profile.implicit_traits] == [_trait(0)]
    assert sorted(_by_description(profile.explicit_info)) == [f"Fact {index}." for index in range(8)]
    assert _log_fields(caplog, "profile regroup ops refused:") == [{"count": "4"}]


async def test_duplicates_left_by_compact_operations_are_removed(caplog: pytest.LogCaptureFixture) -> None:
    """An update or a split piece restating another item's fact leaves two copies; the dedupe step keeps the first."""
    compact = {
        "operations": [
            {
                "action": "update",
                "type": "explicit_info",
                "index": 1,
                "data": {"evidence": "said fact 1", "description": "fact   0.", "category": "Topic 1"},
            },
            {
                "action": "split",
                "type": "explicit_info",
                "index": 2,
                "items": [
                    {"evidence": "said fact 2", "description": "Fact 2.", "category": "Topic 2"},
                    {"evidence": "said fact 2", "description": "FACT 5.", "category": "Topic 2"},
                ],
            },
        ]
    }

    with caplog.at_level(logging.WARNING, logger=_PROFILE_LOGGER):
        fake, profile = await _update(_over_the_total_cap(), {"operations": [{"action": "none"}]}, compact)

    assert fake.call_count == 2  # UPDATE, COMPACT; no group is over its cap
    descriptions = [" ".join(item.description.split()).casefold() for item in profile.explicit_info]
    assert len(descriptions) == len(set(descriptions))
    items = _by_description(profile.explicit_info)
    assert items["Fact 0."] == _explicit(0)
    assert items["Fact 5."] == _explicit(5)
    assert "fact   0." not in items
    assert "FACT 5." not in items
    assert _log_fields(caplog, "profile duplicates dropped:") == [{"source": "compact", "count": "2"}]
