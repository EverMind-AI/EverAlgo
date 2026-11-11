"""The Episode item checks: a quote found in the narratives for every item, pattern items held until repeated.

Every item the model writes carries a quote found in the narratives, and pattern items wait on the profile's
``pending`` list until a later batch shows them again (``profile._pending``).

INIT and UPDATE replies carry ``pattern``, ``quote`` and ``regular`` on each item; UPDATE may ``confirm`` a pending
entry. The list
is the extra field ``pending`` on the returned Profile, which the host passes back like ``do_not_record``; every
UPDATE shows it as ``<pending_patterns>``.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import pytest

from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import Episode, Profile
from everalgo.user_memory import ProfileExtractor
from everalgo.user_memory.profile import _pending

_TS = 1_780_000_000_000  # 2026-05-28 UTC
_DAY = 86_400_000
_PROFILE_LOGGER = "everalgo.user_memory.profile"
_LOG_PAIR = re.compile(r"(\w+)=(.*?)(?= \w+=|$)")
_MARKER = "CONTENT-MARKER-51ab"
_ID_CARD = "42050219591120321X"


def _log_fields(caplog: pytest.LogCaptureFixture, prefix: str) -> list[dict[str, str]]:
    return [
        dict(_LOG_PAIR.findall(record.getMessage().removeprefix(prefix).strip()))
        for record in caplog.records
        if record.getMessage().startswith(prefix)
    ]


def _episode(text: str, timestamp: int) -> Episode:
    return Episode(owner_id=None, episode=f"Mara {text}", summary="s", timestamp=timestamp)


def _item(
    description: str,
    quote: str | None,
    *,
    category: str = "Preferences",
    pattern: bool | None = None,
    evidence: str = "said so",
    **extra: Any,
) -> dict[str, Any]:
    item: dict[str, Any] = {"evidence": evidence, "description": description, "category": category, **extra}
    if pattern is not None:
        item["pattern"] = pattern
    if quote is not None:
        item["quote"] = quote
    return item


def _init(*explicit: dict[str, Any], traits: tuple[dict[str, Any], ...] = ()) -> str:
    return json.dumps({"explicit_info": list(explicit), "implicit_traits": list(traits), "do_not_record": []})


def _operations(*operations: dict[str, Any]) -> str:
    return json.dumps({"operations": list(operations)})


def _pending_of(profile: Profile) -> list[dict[str, Any]] | None:
    return (profile.model_extra or {}).get("pending")


def _shown(fake: FakeLLMClient, call: int) -> str:
    """The ``<pending_patterns>`` block of an UPDATE call's user message."""
    user = str(fake.calls[call].messages[1].content)
    return user.split("<pending_patterns>\n", 1)[1].split("\n</pending_patterns>", 1)[0]


def _entry(
    description: str, observed_at: int, *, category: str = "Preferences", evidence: str = "said so"
) -> dict[str, Any]:
    return {
        "evidence": evidence,
        "description": description,
        "category": category,
        "bucket": "explicit_info",
        "observed_at": observed_at,
    }


def _stored(*, pending: list[dict[str, Any]] | None = None, explicit: list[dict[str, Any]] | None = None) -> Profile:
    data: dict[str, Any] = {"owner_id": "u1", "summary": "s", "timestamp": _TS, "explicit_info": explicit or []}
    if pending is not None:
        data["pending"] = pending
    return Profile.model_validate(data)


async def _extract(
    fake: FakeLLMClient, episodes: list[Episode], old: Profile | None = None, *, item_checks: bool = True
) -> Profile:
    return await ProfileExtractor(llm=fake, item_checks=item_checks).aextract_from_episodes(
        episodes, owner_id="u1", owner_name="Mara", old_profile=old
    )


# ---------------------------------------------------------------- INIT


async def test_init_drops_items_whose_quote_is_missing_short_or_not_in_the_narratives() -> None:
    fake = FakeLLMClient(
        responses=[
            _init(
                _item("Teaches violin in Ghent.", "说她在根特教小提琴", category="Identity", pattern=False),
                _item("Has a cat.", None, category="Family", pattern=False),
                _item("Lives by the sea.", "海边", category="Identity", pattern=False),
                _item("Loves jazz.", "she loves jazz and blues", category="Hobbies"),
            )
        ]
    )

    profile = await _extract(fake, [_episode("说她在根特教小提琴。", _TS)])

    assert [item.description for item in profile.explicit_info] == ["Teaches violin in Ghent."]
    assert _pending_of(profile) is None


async def test_a_quote_is_found_across_punctuation_case_an_ellipsis_or_a_near_match() -> None:
    narrative = "说她在根特教小提琴、每周二晚上还去合唱团排练。她也说 Prefers Tea, not coffee."
    fake = FakeLLMClient(
        responses=[
            _init(
                _item("Teaches violin.", "说她在根特 教小提琴", category="Identity", pattern=False),
                _item("Sings in a choir.", "她在根特……合唱团排练", category="Identity", pattern=False),
                _item("Prefers tea.", "prefers tea not coffee", category="Identity", pattern=False),
                _item("Rehearses on Tuesdays.", "每周二晚上还去合唱团排练了", category="Identity", pattern=False),
            )
        ]
    )

    profile = await _extract(fake, [_episode(narrative, _TS)])

    assert len(profile.explicit_info) == 4


async def test_init_writes_a_regular_pattern_at_once_and_holds_the_others() -> None:
    fake = FakeLLMClient(
        responses=[
            _init(
                _item("Swims every Wednesday.", "她说每周三晚上都去游泳", category="行为习惯"),
                _item("Prefers coffee without sugar.", "说她喜欢喝不加糖的咖啡", pattern=True),
                _item("Works late.", "她昨天熬夜到两点赶稿", category="Work style", pattern=True),
                _item("Lives in Ghent.", "说她住在根特", category="Identity", pattern=False),
                traits=({"basis": "b", "description": "Driven.", "trait": "Driven", "quote": "她昨天熬夜到两点赶稿"},),
            )
        ]
    )

    profile = await _extract(
        fake,
        [
            _episode("她说每周三晚上都去游泳。说她住在根特。", _TS),
            _episode("说她喜欢喝不加糖的咖啡。她昨天熬夜到两点赶稿。", _TS + _DAY),
        ],
    )

    assert [item.description for item in profile.explicit_info] == ["Swims every Wednesday.", "Lives in Ghent."]
    assert profile.implicit_traits == []
    assert _pending_of(profile) == [
        _entry("Prefers coffee without sugar.", _TS + _DAY),
        _entry("Works late.", _TS + _DAY, category="Work style"),
        {
            "basis": "b",
            "description": "Driven.",
            "trait": "Driven",
            "bucket": "implicit_traits",
            "observed_at": _TS + _DAY,
        },
    ]
    for item in profile.explicit_info:
        assert not {"pattern", "quote", "pending", "regular"} & set(item.model_dump(exclude_unset=True))


async def test_a_held_item_is_scrubbed_like_a_stored_one() -> None:
    fake = FakeLLMClient(
        responses=[
            _init(
                _item(f"Pays with card {_ID_CARD}.", "她昨天熬夜到两点赶稿", pattern=True),
                _item("Works late.", "她昨天熬夜到两点赶稿", pattern=True, evidence=f"ID {_ID_CARD} on the form"),
            )
        ]
    )

    profile = await _extract(fake, [_episode("她昨天熬夜到两点赶稿。", _TS)])

    assert _pending_of(profile) == [_entry("Works late.", _TS, evidence="ID *** on the form")]


@pytest.mark.parametrize(
    ("quote", "expected"),
    [
        ("她每周三晚上都去游泳", True),
        ("一三五早上雷打不动去游泳", True),
        ("一周跑三次步", True),
        ("平时都很早睡", True),
        ("以后回复都用中文", True),
        ("以后就叫我小王", True),
        ("从今以后早睡", True),
        ("最近三个月都在练瑜伽", True),
        ("过去两年都在周末爬山", True),
        ("这半年来都早起跑步", True),
        ("以后再说吧", False),
        ("毕业以后去了北京", False),
        ("以后有空再聊", False),
        ("最近三个月去过一次健身房", False),
        ("最近三个月\uff0c都没空", False),
        ("这周末去游泳", False),
        ("周三去游了一次泳", False),
        ("她非常喜欢这家店", False),
        ("又熬夜到三点", False),
        ("一周忙\uff0c次日歇", False),
        ("一周忙,次日歇", False),
    ],
)
def test_the_words_that_show_regularity_in_chinese(quote: str, expected: object) -> None:
    assert bool(_pending._REGULAR_ZH.search(quote)) is expected


# ---------------------------------------------------------------- languages


@pytest.mark.parametrize(
    ("quote", "expected"),
    [
        ("she swims every Wednesday evening", True),
        ("goes for a run every other morning", True),
        ("calls her parents on Sunday evenings", True),
        ("runs three times a week along the river", True),
        ("runs five km a day", True),
        ("has played the cello for over ten years", True),
        ("From now on, reply in English", True),
        ("usually takes the night train", True),
        ("it has become a habit to write a few lines", True),
        ("cooks dinner at home about five days a week", True),
        ("gets up at 6:00 on workdays", True),
        ("works from home most of the time", True),
        ("kept up the twenty-minute walk after dinner", True),
        ("the Sunday call was set in stone", True),
        ("this Saturday she went swimming", False),
        ("went swimming once on Wednesday", False),
        ("every so often she bakes bread", False),
        ("review meetings should avoid Saturday whenever possible", False),
        ("three times faster than the old build", False),
        ("she used to smoke", False),
        ("stayed up until three again", False),
    ],
)
def test_the_words_that_show_regularity_in_english(quote: str, expected: object) -> None:
    assert bool(_pending._REGULAR_EN.search(" ".join(quote.casefold().split()))) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Mara swims every Wednesday evening", True),
        ("林晓 swims every Wednesday evening", True),
        ("José runs with his dog", True),
        ("Sie geht jeden Morgen joggen im Park", False),
        ("毎週水曜日にプールで泳いでいる", False),
        ("매일 아침 he swims", False),
        ("should avoid Saturdays whenever possible", True),
        ("2024-05-01 12:00", False),
    ],
)
def test_a_quote_counts_as_english_when_alphabetic_with_a_common_english_word(text: str, expected: object) -> None:
    assert _pending._is_english(text) is expected


async def test_an_english_quote_is_written_on_the_english_word_list_else_on_the_models_regular_words() -> None:
    narrative = (
        "Mara said she swims every Wednesday evening at the pool. She stayed up until two finishing the draft. "
        "She went for a run along the river this morning. On Sundays she calls her parents. "
        "She is still using the same old Fujifilm camera."
    )
    fake = FakeLLMClient(
        responses=[
            _init(
                _item("Swims every Wednesday.", "swims every Wednesday evening at the pool", pattern=True, regular=""),
                _item("Calls her parents weekly.", "On Sundays she calls her parents", pattern=True, regular="ev"),
                _item("Shoots film.", "still using the same old Fujifilm camera", pattern=True, regular="still using"),
                _item("Works late.", "stayed up until two finishing the draft", pattern=True, regular=""),
                _item("Runs daily.", "went for a run along the river this morning", pattern=True, regular="every day"),
            )
        ]
    )

    profile = await _extract(fake, [_episode(narrative, _TS)])

    assert [item.description for item in profile.explicit_info] == [
        "Swims every Wednesday.",
        "Calls her parents weekly.",
        "Shoots film.",
    ]
    assert [entry["description"] for entry in _pending_of(profile) or []] == ["Works late.", "Runs daily."]


async def test_in_any_other_language_the_models_regular_words_decide_and_must_stand_in_the_quote() -> None:
    narrative = (
        "毎週水曜日にプールで泳いでいる。Sie geht jeden Morgen joggen im Park. Sie hat gestern bis zwei gearbeitet. "
        "Elle nage le mercredi à la piscine."
    )
    fake = FakeLLMClient(
        responses=[
            _init(
                _item("Swims weekly (ja).", "毎週水曜日にプールで泳いでいる", pattern=True, regular="毎週水曜日"),
                _item("Jogs every morning.", "geht jeden Morgen joggen im Park", pattern=True, regular="jeden Morgen"),
                _item("Works late.", "Sie hat gestern bis zwei gearbeitet", pattern=True, regular=""),
                _item(
                    "Swims on Wednesdays.", "Elle nage le mercredi à la piscine", pattern=True, regular="chaque semaine"
                ),
                _item("Likes the pool.", "Elle nage le mercredi à la piscine", pattern=True, regular="le"),
            )
        ]
    )

    profile = await _extract(fake, [_episode(narrative, _TS)])

    assert [item.description for item in profile.explicit_info] == ["Swims weekly (ja).", "Jogs every morning."]
    assert [entry["description"] for entry in _pending_of(profile) or []] == [
        "Works late.",
        "Swims on Wednesdays.",
        "Likes the pool.",
    ]
    for item in profile.explicit_info:
        assert "regular" not in item.model_dump(exclude_unset=True)
    assert all("regular" not in entry for entry in _pending_of(profile) or [])


async def test_a_chinese_quote_is_read_with_the_word_list_whatever_the_model_marks() -> None:
    fake = FakeLLMClient(
        responses=[
            _init(
                _item("Works late.", "她昨天熬夜到两点赶稿", pattern=True, regular="昨天熬夜"),
                _item("Swims every Wednesday.", "她说每周三晚上都去游泳", pattern=True, regular=""),
            )
        ]
    )

    profile = await _extract(fake, [_episode("她昨天熬夜到两点赶稿。她说每周三晚上都去游泳。", _TS)])

    assert [item.description for item in profile.explicit_info] == ["Swims every Wednesday."]
    assert [entry["description"] for entry in _pending_of(profile) or []] == ["Works late."]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("她每周三晚上都去游泳", True),
        ("我每周都跟 team 开 weekly sync", True),
        ("林晓 swims every Wednesday evening", False),
        ("payment API 在 staging 又 timeout 了", False),
        ("毎日散歩する", False),
        ("매일 아침 수영을 한다", False),
    ],
)
def test_a_quote_counts_as_chinese_when_mostly_han_without_kana_or_hangul(text: str, expected: object) -> None:
    assert _pending._mostly_han(_pending._fold(text)) is expected


@pytest.mark.parametrize(
    ("quote", "narrative", "found"),
    [
        ("i'll take the night train to porto", "Mara: “I\u2019ll take the Night Train to Porto,” she said.", True),
        ("她用gpt写周报每周五", "她用ＧＰＴ写周报\uff0c每周五交。", True),
        ("loves jazz", "Mara loves jazz.", False),
        ("loves jazz and the blues", "Mara loves jazz and the blues.", True),
        ("喜欢爵士乐", "她喜欢爵士乐。", False),
        ("喜欢爵士乐和蓝调", "她喜欢爵士乐和蓝调。", True),
    ],
)
def test_quotes_match_across_width_case_and_quote_marks_and_length_counts_letters_as_a_third(
    quote: str, narrative: str, found: object
) -> None:
    assert (_pending._find_quote(quote, [_pending._fold(narrative)]) is not None) is found


@pytest.mark.parametrize(
    ("text", "masked"),
    [
        ("card 4111-1111-1111-1111 on file", True),
        ("card 4111 1111 1111 1111 on file", True),
        (
            "ID \uff11\uff11\uff10\uff11\uff10\uff11\uff11\uff19\uff19\uff10\uff10\uff13\uff10\uff17\uff11\uff12\uff13\uff38",
            True,
        ),
        ("call 138-0013-8000", False),
        ("2026-03-02 and 2026-03-09", False),
    ],
)
def test_account_numbers_with_hyphens_or_fullwidth_digits_are_caught(text: str, masked: object) -> None:
    from everalgo.user_memory.profile._merge import _ACCOUNT_NUMBER

    assert bool(_ACCOUNT_NUMBER.search(text)) is masked


async def test_both_stages_tell_the_model_not_to_gender_this_person_and_to_mark_regular_words() -> None:
    fake = FakeLLMClient(responses=[_init(), _operations({"action": "none"})])

    first = await _extract(fake, [_episode("said hello.", _TS)])
    await _extract(fake, [_episode("said hello again.", _TS + _DAY)], Profile.model_validate(first.model_dump()))

    for call in fake.calls:
        system = str(call.messages[0].content)
        block = system.split("<item_checks>", 1)[1].split("</item_checks>", 1)[0]
        assert "no description refers to this person with a gendered word" in block
        assert '"regular"' in block


# ---------------------------------------------------------------- UPDATE


async def test_update_shows_the_list_and_a_confirm_writes_the_entry_and_takes_it_off() -> None:
    stored = _stored(pending=[_entry("Prefers coffee without sugar.", _TS), _entry("Works late.", _TS)])
    fake = FakeLLMClient(
        responses=[
            _operations(
                {
                    "action": "confirm",
                    "type": "explicit_info",
                    "pending": 1,
                    "data": _item("Often works late.", "又熬夜到三点改稿子", category="Work style", pattern=True),
                },
            )
        ]
    )

    profile = await _extract(fake, [_episode("又熬夜到三点改稿子。", _TS + 10 * _DAY)], stored)

    assert _shown(fake, 0) == (
        '<entry index="0" first_seen="2026-05-28">{"description": "Prefers coffee without sugar.", '
        '"category": "Preferences", "evidence": "said so"}</entry>\n'
        '<entry index="1" first_seen="2026-05-28">{"description": "Works late.", "category": "Preferences", '
        '"evidence": "said so"}</entry>'
    )
    assert [item.description for item in profile.explicit_info] == ["Often works late."]
    assert _pending_of(profile) == [_entry("Prefers coffee without sugar.", _TS)]


async def test_an_empty_list_is_shown_as_none_and_written_back_when_the_stored_one_empties() -> None:
    stored = _stored(pending=[_entry("Works late.", _TS)])
    fake = FakeLLMClient(
        responses=[
            _operations(
                {"action": "confirm", "pending": 0, "data": _item("Works late.", "又熬夜到三点改稿子", pattern=True)}
            ),
            _operations({"action": "none"}),
        ]
    )

    first = await _extract(fake, [_episode("又熬夜到三点改稿子。", _TS + _DAY)], stored)
    second = await _extract(
        fake, [_episode("说了点别的。", _TS + 2 * _DAY)], Profile.model_validate(first.model_dump())
    )

    assert _pending_of(first) == []
    assert [item.description for item in first.explicit_info] == ["Works late."]
    assert _shown(fake, 1) == "(none)"
    assert _pending_of(second) == []


async def test_a_restated_entry_is_written_once_and_a_second_proposal_dropped() -> None:
    stored = _stored(pending=[_entry("Prefers coffee without sugar.", _TS)])
    fake = FakeLLMClient(
        responses=[
            _operations(
                {
                    "action": "add",
                    "type": "explicit_info",
                    "data": _item("Prefers coffee without sugar.", "咖啡还是要不加糖的", pattern=True),
                },
                {
                    "action": "add",
                    "type": "explicit_info",
                    "data": _item("Prefers coffee without any sugar.", "咖啡还是要不加糖的", pattern=True),
                },
            )
        ]
    )

    profile = await _extract(fake, [_episode("她说咖啡还是要不加糖的。", _TS + _DAY)], stored)

    assert [item.description for item in profile.explicit_info] == ["Prefers coffee without sugar."]
    assert _pending_of(profile) == []


async def test_a_pointer_whose_evidence_shows_no_occasion_of_this_batch_is_not_a_repeat() -> None:
    stored = _stored(pending=[_entry("Works late.", _TS)])
    fake = FakeLLMClient(
        responses=[
            _operations(
                {
                    "action": "add",
                    "type": "explicit_info",
                    "data": _item(
                        "Works late.",
                        "又熬夜到三点改稿子",
                        pattern=True,
                        pending=0,
                        evidence="2026-05-28: stayed up until two",
                    ),
                },
            )
        ]
    )

    profile = await _extract(fake, [_episode("又熬夜到三点改稿子。", _TS + 3 * _DAY)], stored)

    assert profile.explicit_info == []
    assert _pending_of(profile) == [_entry("Works late.", _TS)]  # not held a second time either


async def test_a_new_pattern_is_held_and_a_regular_one_written() -> None:
    fake = FakeLLMClient(
        responses=[
            _operations(
                {
                    "action": "add",
                    "type": "explicit_info",
                    "data": _item("Likes rain sounds while editing.", "她说改稿时喜欢听雨声", pattern=True),
                },
                {
                    "action": "add",
                    "type": "explicit_info",
                    "data": _item("Cycles to work every day.", "她每天骑车上班", category="行为习惯"),
                },
            )
        ]
    )

    profile = await _extract(fake, [_episode("她说改稿时喜欢听雨声。她每天骑车上班。", _TS + _DAY)], _stored())

    assert [item.description for item in profile.explicit_info] == ["Cycles to work every day."]
    assert _pending_of(profile) == [_entry("Likes rain sounds while editing.", _TS + _DAY)]


async def test_an_update_needs_a_quote_only_when_it_changes_the_description() -> None:
    stored = _stored(
        explicit=[
            {"evidence": "e0", "description": "Teaches violin in Ghent.", "category": "Identity", "observed_at": _TS},
            {"evidence": "e1", "description": "Lives in Ghent.", "category": "Identity", "observed_at": _TS},
        ]
    )
    fake = FakeLLMClient(
        responses=[
            _operations(
                {
                    "action": "update",
                    "type": "explicit_info",
                    "index": 0,
                    "data": _item("Plays in the Bruges orchestra.", None, category="Identity"),
                },
                {
                    "action": "update",
                    "type": "explicit_info",
                    "index": 1,
                    "data": _item("Lives in Ghent.", None, category="Identity", evidence="e1; said again"),
                },
            )
        ]
    )

    profile = await _extract(fake, [_episode("她换了新工作、在布鲁日乐团。", _TS + _DAY)], stored)

    assert [(item.description, item.evidence) for item in profile.explicit_info] == [
        ("Teaches violin in Ghent.", "e0"),
        ("Lives in Ghent.", "e1; said again"),
    ]


async def test_a_confirm_shaped_for_the_other_bucket_is_dropped_alone(caplog: pytest.LogCaptureFixture) -> None:
    stored = _stored(pending=[_entry("Works late.", _TS)])
    fake = FakeLLMClient(
        responses=[
            _operations(
                {
                    "action": "confirm",
                    "type": "explicit_info",
                    "pending": 0,
                    "data": {
                        "basis": "b",
                        "description": "Works late.",
                        "trait": "Driven",
                        "quote": "又熬夜到三点改稿子",
                    },
                },
                {
                    "action": "add",
                    "type": "explicit_info",
                    "data": _item("Lives in Bruges.", "她搬到了布鲁日", category="Identity", pattern=False),
                },
            )
        ]
    )

    with caplog.at_level(logging.WARNING, logger=_PROFILE_LOGGER):
        profile = await _extract(fake, [_episode("又熬夜到三点改稿子。她搬到了布鲁日。", _TS + _DAY)], stored)

    assert [item.description for item in profile.explicit_info] == ["Lives in Bruges."]
    assert _pending_of(profile) == [_entry("Works late.", _TS)]
    assert _log_fields(caplog, "profile update ops dropped:") == [
        {"count": "1", "reason": "confirm item shaped for the other bucket"}
    ]


async def test_old_entries_expire_before_the_list_is_shown_so_indices_match() -> None:
    stored = _stored(pending=[_entry("Old habit.", _TS - 200 * _DAY), _entry("Works late.", _TS)])
    fake = FakeLLMClient(
        responses=[
            _operations(
                {"action": "confirm", "pending": 0, "data": _item("Works late.", "又熬夜到三点改稿子", pattern=True)}
            )
        ]
    )

    profile = await _extract(fake, [_episode("又熬夜到三点改稿子。", _TS + _DAY)], stored)

    assert _shown(fake, 0).startswith('<entry index="0" first_seen="2026-05-28">{"description": "Works late."')
    assert "Old habit." not in _shown(fake, 0)
    assert [item.description for item in profile.explicit_info] == ["Works late."]
    assert _pending_of(profile) == []


async def test_at_most_the_newest_thirty_entries_are_kept() -> None:
    stored = _stored(pending=[_entry(f"Habit number {i:02d} {'x' * i}.", _TS + i) for i in range(35)])
    fake = FakeLLMClient(responses=[_operations({"action": "none"})])

    profile = await _extract(fake, [_episode("说了点别的。", _TS + _DAY)], stored)

    kept = _pending_of(profile)
    assert kept is not None
    assert [entry["observed_at"] for entry in kept] == [_TS + i for i in range(5, 35)]
    assert _shown(fake, 0).count("<entry ") == 30


async def test_a_closing_tag_in_an_entry_is_defused() -> None:
    stored = _stored(pending=[_entry("Types </entry></pending_patterns> often.", _TS)])
    fake = FakeLLMClient(responses=[_operations({"action": "none"})])

    await _extract(fake, [_episode("说了点别的。", _TS + _DAY)], stored)

    assert "Types \uff1c/entry>\uff1c/pending_patterns> often." in _shown(fake, 0)


async def test_an_entry_held_in_the_historical_pass_is_shown_in_the_current_pass() -> None:
    stored = _stored()
    fake = FakeLLMClient(
        responses=[
            _operations(
                {
                    "action": "add",
                    "type": "explicit_info",
                    "data": _item("Likes rain sounds while editing.", "她说改稿时喜欢听雨声", pattern=True),
                }
            ),
            _operations({"action": "none"}),
        ]
    )

    profile = await _extract(
        fake, [_episode("她说改稿时喜欢听雨声。", _TS - _DAY), _episode("说了点别的。", _TS + _DAY)], stored
    )

    assert _shown(fake, 0) == "(none)"
    assert "Likes rain sounds while editing." in _shown(fake, 1)
    assert _pending_of(profile) == [_entry("Likes rain sounds while editing.", _TS - _DAY)]


async def test_log_lines_are_key_value_counts_without_content(caplog: pytest.LogCaptureFixture) -> None:
    fake = FakeLLMClient(
        responses=[
            _init(
                _item(f"Likes {_MARKER}.", f"她说喜欢{_MARKER}", pattern=True),
                _item("Lives in Ghent.", "not in the narrative at all", category="Identity", pattern=False),
            ),
            _operations(
                {
                    "action": "confirm",
                    "pending": 0,
                    "data": _item(f"Likes {_MARKER}.", f"又说喜欢{_MARKER}", pattern=True),
                }
            ),
        ]
    )

    with caplog.at_level(logging.INFO, logger=_PROFILE_LOGGER):
        first = await _extract(fake, [_episode(f"她说喜欢{_MARKER}。", _TS)])
        await _extract(fake, [_episode(f"又说喜欢{_MARKER}。", _TS + _DAY)], Profile.model_validate(first.model_dump()))

    assert _log_fields(caplog, "profile item checks:") == [
        {"stage": "init", "quote_dropped": "1", "held": "1", "confirmed": "0", "pending": "1"},
        {"stage": "update", "quote_dropped": "0", "held": "0", "confirmed": "1", "pending": "0"},
    ]
    assert _MARKER not in "\n".join(
        record.getMessage() for record in caplog.records if record.getMessage().startswith("profile item checks")
    )


async def test_with_the_checks_off_every_item_is_written_and_nothing_held() -> None:
    fake = FakeLLMClient(
        responses=[
            _init(
                _item("Prefers coffee without sugar.", "not in the narrative", pattern=True),
                _item("Has a cat.", None, category="Family", pattern=False),
            )
        ]
    )

    profile = await _extract(fake, [_episode("说了点别的。", _TS)], item_checks=False)

    assert [item.description for item in profile.explicit_info] == ["Prefers coffee without sugar.", "Has a cat."]
    assert _pending_of(profile) is None


async def test_the_memcell_path_takes_no_reply_only_fields() -> None:
    """Only the Episode INIT and UPDATE items carry pattern / quote; the MemCell and maintenance replies stay plain."""
    from everalgo.user_memory.profile._replies import (
        _CompactOpsOutput,
        _MemCellUpdateOutput,
        _ProfileItems,
        _RegroupOpsOutput,
    )

    for model in (_ProfileItems, _MemCellUpdateOutput, _CompactOpsOutput, _RegroupOpsOutput):
        assert "quote" not in json.dumps(model.model_json_schema()), model.__name__
