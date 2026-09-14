"""Tests for Profile extraction from dated generic or reflected Episodes."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

import pytest

from everalgo.llm.format import format_message_timestamp
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import Episode, Profile
from everalgo.user_memory import OutputLanguage, ProfileExtractor
from everalgo.user_memory.prompts.en.profile import _ITEM_RULES as _MEMCELL_ITEM_RULES
from everalgo.user_memory.prompts.en.profile import _PORTRAIT
from everalgo.user_memory.prompts.en.profile_from_episode_texts import (
    _CATEGORY_RULES,
    _EPISODE_PRIORITY_RULES,
    _SOURCE_RULES,
    PROFILE_COMPACT_FROM_EPISODE_TEXTS_PROMPT,
    PROFILE_INITIAL_FROM_EPISODE_TEXTS_PROMPT,
    PROFILE_REGROUP_FROM_EPISODE_TEXTS_PROMPT,
    PROFILE_UPDATE_FROM_EPISODE_TEXTS_PROMPT,
)
from everalgo.user_memory.prompts.en.profile_from_episode_texts import (
    _ITEM_RULES as _EPISODE_ITEM_RULES,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

_OWNER_ID = "user-123"
_OWNER_NAME = "Alice"
_TIMESTAMP = 1_700_000_010_000
_EARLIER = _TIMESTAMP - 86_400_000
_LATER = _TIMESTAMP + 86_400_000
_AVAILABLE_CATEGORIES = ("Technical context", "Communication preferences")


def _episode(text: str, *, timestamp: int = _TIMESTAMP) -> Episode:
    return Episode(owner_id=None, episode=text, summary=text, timestamp=timestamp)


def _dated(text: str, *, timestamp: int = _TIMESTAMP) -> str:
    """The narrative exactly as the prompt renders it: under its observation date."""
    return f"[{format_message_timestamp(timestamp)} UTC]\n{text}"


def _profile_payload(*, description: str = "Works mainly in Python.", evidence: str = "Alice works in Python.") -> str:
    return json.dumps(
        {
            "explicit_info": [
                {"category": "Technical stack", "description": description, "evidence": evidence},
            ],
            "implicit_traits": [],
        }
    )


def _operations(*operations: dict[str, Any]) -> str:
    return json.dumps({"operations": list(operations)})


def _add(description: str, *, category: str = "Location", evidence: str = "Narrative evidence.") -> dict[str, Any]:
    return {
        "action": "add",
        "type": "explicit_info",
        "data": {"category": category, "description": description, "evidence": evidence},
    }


def _update(index: int, **data: str) -> dict[str, Any]:
    return {"action": "update", "type": "explicit_info", "index": index, "data": data}


def _delete(index: int) -> dict[str, Any]:
    return {"action": "delete", "type": "explicit_info", "index": index, "reason": "contradicted"}


def _old_profile(
    *,
    owner_id: str = _OWNER_ID,
    explicit_info: list[dict[str, Any]] | None = None,
    timestamp: int = _TIMESTAMP - 1,
) -> Profile:
    items = explicit_info or [
        {"category": "Technical stack", "description": "Works mainly in Python.", "evidence": "Alice uses Python."}
    ]
    return Profile.model_validate(
        {
            "owner_id": owner_id,
            "summary": items[0]["description"],
            "timestamp": timestamp,
            "explicit_info": items,
            "implicit_traits": [],
        }
    )


def _explicit_items(profile: Profile) -> list[dict[str, Any]]:
    return cast("list[dict[str, Any]]", profile.model_dump()["explicit_info"])


def _prompt(fake: FakeLLMClient, call: int = 0) -> str:
    return cast("str", fake.calls[call].messages[0].content)


def _rendered_category_snapshot(prompt: str) -> str:
    """Extract the injected category section from a rendered Episode Profile prompt."""
    return prompt.split("【Available Categories】\n", maxsplit=1)[1].split("\n\n【", maxsplit=1)[0]


def _expected_category_snapshot(categories: Sequence[str] = _AVAILABLE_CATEGORIES) -> str:
    """Return the stable JSON representation expected in every Episode Profile stage."""
    return json.dumps(list(categories), ensure_ascii=False, indent=2)


async def test_init_uses_owner_name_and_preserves_authoritative_owner_fields() -> None:
    fake = FakeLLMClient(responses=[_profile_payload(evidence="Alice selected Python for the service.")])

    profile = await ProfileExtractor(llm=fake).aextract_from_episodes(
        [_episode("Alice selected Python for the service.")],
        owner_id=_OWNER_ID,
        owner_name="  Alice  ",
    )

    assert profile.owner_id == _OWNER_ID
    assert profile.timestamp == _TIMESTAMP
    assert profile.summary == "Works mainly in Python."
    assert fake.call_count == 1


async def test_init_falls_back_to_owner_id_when_name_is_missing_or_blank() -> None:
    for owner_name in (None, "   "):
        fake = FakeLLMClient(responses=[_profile_payload(evidence=f"{_OWNER_ID} selected Python.")])

        profile = await ProfileExtractor(llm=fake).aextract_from_episodes(
            [_episode(f"{_OWNER_ID} selected Python.")],
            owner_id=_OWNER_ID,
            owner_name=owner_name,
        )

        assert profile.owner_id == _OWNER_ID
        assert fake.call_count == 1


@pytest.mark.parametrize(
    ("episodes", "error"),
    [
        ([], "non-empty sequence"),
        (cast("Sequence[Episode]", "Alice selected Python."), "non-empty sequence"),
        ([_episode("   ")], r"episodes\[0\]\.episode must be a non-blank string"),
        (
            [_episode("Bob selected Python."), _episode("The team selected Ruff.")],
            "no episodes reference target user 'Alice'",
        ),
    ],
)
async def test_invalid_episode_batches_fail_before_llm(episodes: Sequence[Episode], error: str) -> None:
    fake = FakeLLMClient(responses=[_profile_payload()])

    with pytest.raises(ValueError, match=error):
        await ProfileExtractor(llm=fake).aextract_from_episodes(episodes, owner_id=_OWNER_ID, owner_name=_OWNER_NAME)

    assert fake.call_count == 0


async def test_non_episode_items_fail_before_llm() -> None:
    fake = FakeLLMClient(responses=[_profile_payload()])

    with pytest.raises(TypeError, match=r"episodes\[0\] must be an Episode"):
        await ProfileExtractor(llm=fake).aextract_from_episodes(
            cast("Sequence[Episode]", ["Alice selected Python."]),
            owner_id=_OWNER_ID,
            owner_name=_OWNER_NAME,
        )

    assert fake.call_count == 0


async def test_blank_owner_id_fails_before_llm() -> None:
    fake = FakeLLMClient(responses=[_profile_payload()])

    with pytest.raises(ValueError, match="owner_id must be a non-blank string"):
        await ProfileExtractor(llm=fake).aextract_from_episodes(
            [_episode("Alice selected Python.")],
            owner_id="   ",
            owner_name=_OWNER_NAME,
        )

    assert fake.call_count == 0


async def test_owner_name_takes_precedence_over_owner_id_during_validation() -> None:
    fake = FakeLLMClient(responses=[_profile_payload()])

    with pytest.raises(ValueError, match="no episodes reference target user 'Alice'"):
        await ProfileExtractor(llm=fake).aextract_from_episodes(
            [_episode(f"{_OWNER_ID} selected Python.")],
            owner_id=_OWNER_ID,
            owner_name=_OWNER_NAME,
        )

    assert fake.call_count == 0


async def test_init_skips_episodes_that_do_not_reference_the_resolved_target() -> None:
    fake = FakeLLMClient(responses=[_profile_payload()])

    await ProfileExtractor(llm=fake).aextract_from_episodes(
        [
            _episode("Alice selected Python.", timestamp=_EARLIER),
            _episode("UNRELATED_EPISODE selected Ruff."),
            _episode("Alice adopted uv.", timestamp=_LATER),
        ],
        owner_id=_OWNER_ID,
        owner_name=_OWNER_NAME,
    )

    prompt = _prompt(fake)
    assert "Alice selected Python." in prompt
    assert "Alice adopted uv." in prompt
    assert prompt.index("Alice selected Python.") < prompt.index("Alice adopted uv.")
    assert "UNRELATED_EPISODE selected Ruff." not in prompt


async def test_update_skips_episodes_that_do_not_reference_the_resolved_target() -> None:
    fake = FakeLLMClient(responses=[_operations({"action": "none"})])

    await ProfileExtractor(llm=fake).aextract_from_episodes(
        [_episode("UNRELATED_EPISODE selected Ruff."), _episode("Alice adopted uv.")],
        owner_id=_OWNER_ID,
        owner_name=_OWNER_NAME,
        old_profile=_old_profile(),
    )

    prompt = _prompt(fake)
    assert "Alice adopted uv." in prompt
    assert "UNRELATED_EPISODE selected Ruff." not in prompt


async def test_existing_profile_owner_must_match_before_llm() -> None:
    fake = FakeLLMClient(responses=[_operations({"action": "none"})])

    with pytest.raises(ValueError, match="does not match owner_id"):
        await ProfileExtractor(llm=fake).aextract_from_episodes(
            [_episode("Alice selected Python.")],
            owner_id=_OWNER_ID,
            owner_name=_OWNER_NAME,
            old_profile=_old_profile(owner_id="someone-else"),
        )

    assert fake.call_count == 0


async def test_unsupported_output_language_fails_before_llm() -> None:
    fake = FakeLLMClient(responses=[_profile_payload()])

    with pytest.raises(ValueError, match="unsupported output_language"):
        await ProfileExtractor(llm=fake).aextract_from_episodes(
            [_episode("Alice selected Python.")],
            owner_id=_OWNER_ID,
            owner_name=_OWNER_NAME,
            output_language="Klingon",
        )

    assert fake.call_count == 0


@pytest.mark.parametrize("categories", [None, []])
async def test_none_or_empty_categories_allow_extraction_with_an_empty_snapshot(
    categories: Sequence[str] | None,
) -> None:
    fake = FakeLLMClient(responses=[_profile_payload()])

    await ProfileExtractor(llm=fake).aextract_from_episodes(
        [_episode("Alice selected Python.")],
        owner_id=_OWNER_ID,
        owner_name=_OWNER_NAME,
        categories=categories,
    )

    assert _rendered_category_snapshot(_prompt(fake)) == "[]"


async def test_categories_are_stripped_deduplicated_and_rendered_in_first_seen_order() -> None:
    fake = FakeLLMClient(responses=[_profile_payload()])

    await ProfileExtractor(llm=fake).aextract_from_episodes(
        [_episode("Alice selected Python.")],
        owner_id=_OWNER_ID,
        owner_name=_OWNER_NAME,
        categories=["  Technical context  ", "", "Technical context", "  ", "Communication preferences"],
    )

    assert _rendered_category_snapshot(_prompt(fake)) == _expected_category_snapshot()


async def test_category_snapshot_is_not_an_output_whitelist_and_does_not_filter_traits() -> None:
    payload = json.dumps(
        {
            "explicit_info": [
                {
                    "category": "Specialized tooling",
                    "description": "Uses an uncommon build tool.",
                    "evidence": "Alice consistently uses the tool.",
                }
            ],
            "implicit_traits": [
                {
                    "trait": "Methodical",
                    "description": "Approaches decisions methodically.",
                    "basis": "Alice independently compared constraints in two Episodes.",
                }
            ],
        }
    )
    fake = FakeLLMClient(responses=[payload])

    profile = await ProfileExtractor(llm=fake).aextract_from_episodes(
        [_episode("Alice consistently uses an uncommon build tool.")],
        owner_id=_OWNER_ID,
        owner_name=_OWNER_NAME,
        categories=["Communication preferences"],
    )

    profile_data = profile.model_dump()
    explicit_info = cast("list[dict[str, Any]]", profile_data["explicit_info"])
    implicit_traits = cast("list[dict[str, Any]]", profile_data["implicit_traits"])
    assert explicit_info[0]["category"] == "Specialized tooling"
    assert implicit_traits[0]["trait"] == "Methodical"


@pytest.mark.parametrize(
    ("categories", "error"),
    [
        (cast("Any", "Technical context"), "categories must be a sequence of strings or None"),
        (cast("Any", 42), "categories must be a sequence of strings or None"),
        (cast("Any", ["Technical context", None]), r"categories\[1\] must be a string"),
    ],
)
async def test_invalid_categories_fail_before_llm(categories: Sequence[str], error: str) -> None:
    fake = FakeLLMClient(responses=[_profile_payload()])

    with pytest.raises(TypeError, match=error):
        await ProfileExtractor(llm=fake).aextract_from_episodes(
            [_episode("Alice selected Python.")],
            owner_id=_OWNER_ID,
            owner_name=_OWNER_NAME,
            categories=categories,
        )

    assert fake.call_count == 0


async def test_init_prompt_receives_target_and_dated_unnumbered_narratives_oldest_first() -> None:
    marker_one = "Alice chose marker-one-tooling."
    marker_two = "Alice documented marker-two-preferences."
    fake = FakeLLMClient(responses=[_profile_payload()])

    await ProfileExtractor(llm=fake).aextract_from_episodes(
        [_episode(marker_two, timestamp=_LATER), _episode(marker_one, timestamp=_EARLIER)],
        owner_id="internal-owner-id",
        owner_name=_OWNER_NAME,
        categories=_AVAILABLE_CATEGORIES,
    )

    prompt = _prompt(fake)
    assert "build a profile of Alice" in prompt
    assert "internal-owner-id" not in prompt
    assert prompt.count(marker_one) == 1
    assert prompt.count(marker_two) == 1
    assert f"{_dated(marker_one, timestamp=_EARLIER)}\n\n---\n\n{_dated(marker_two, timestamp=_LATER)}" in prompt
    assert "[0]" not in prompt
    assert "SAME language EPISODE_TEXT itself is written in" in prompt
    assert _rendered_category_snapshot(prompt) == _expected_category_snapshot()


async def test_explicit_output_language_and_custom_prompt_are_rendered() -> None:
    fake = FakeLLMClient(responses=[_profile_payload()])

    await ProfileExtractor(llm=fake).aextract_from_episodes(
        [_episode("Alice selected Python.")],
        owner_id=_OWNER_ID,
        owner_name=_OWNER_NAME,
        categories=_AVAILABLE_CATEGORIES,
        prompt=(
            "TARGET={target_user}; LANGUAGE={language_rule}; CATEGORIES={available_categories}; "
            "EPISODES={episode_texts}"
        ),
        output_language=OutputLanguage.CHINESE,
    )

    prompt = _prompt(fake)
    assert prompt.startswith("TARGET=Alice")
    assert "Write ALL output fields in Chinese" in prompt
    assert prompt.endswith(f"EPISODES={_dated('Alice selected Python.')}")
    assert f"CATEGORIES={_expected_category_snapshot()}" in prompt


async def test_init_items_are_stamped_with_the_newest_episode_date() -> None:
    fake = FakeLLMClient(responses=[_profile_payload()])

    profile = await ProfileExtractor(llm=fake).aextract_from_episodes(
        [_episode("Alice selected Python.", timestamp=_EARLIER), _episode("Alice adopted uv.", timestamp=_LATER)],
        owner_id=_OWNER_ID,
        owner_name=_OWNER_NAME,
    )

    assert profile.timestamp == _LATER
    assert [item["observed_at"] for item in _explicit_items(profile)] == [_LATER]


async def test_update_applies_operations_and_dates_every_item() -> None:
    fake = FakeLLMClient(
        responses=[_operations(_add("Prefers concise answers.", category="Communication", evidence="Alice asked."))]
    )

    profile = await ProfileExtractor(llm=fake).aextract_from_episodes(
        [_episode("Alice requested concise answers.")],
        owner_id=_OWNER_ID,
        owner_name=_OWNER_NAME,
        old_profile=_old_profile(),
        categories=_AVAILABLE_CATEGORIES,
    )

    items = _explicit_items(profile)
    assert profile.owner_id == _OWNER_ID
    assert profile.timestamp == _TIMESTAMP
    assert [item["description"] for item in items] == ["Works mainly in Python.", "Prefers concise answers."]
    # A stored item written before the field existed takes the profile's own date; the new one takes the batch's.
    assert [item["observed_at"] for item in items] == [_TIMESTAMP - 1, _TIMESTAMP]
    assert fake.call_count == 1
    update_prompt = _prompt(fake)
    assert _rendered_category_snapshot(update_prompt) == _expected_category_snapshot()
    assert "categories already in use" not in update_prompt
    # The model is shown the same date the merge will judge the legacy item by.
    assert f'"observed_at": "{format_message_timestamp(_TIMESTAMP - 1)} UTC"' in update_prompt


async def test_stored_items_show_observed_at_as_a_date_the_model_can_compare() -> None:
    fake = FakeLLMClient(responses=[_operations({"action": "none"})])
    stored = {
        "category": "Location",
        "description": "Lives in Shanghai.",
        "evidence": "Alice moved.",
        "observed_at": _TIMESTAMP,
    }

    await ProfileExtractor(llm=fake).aextract_from_episodes(
        [_episode("Alice adopted uv.", timestamp=_LATER)],
        owner_id=_OWNER_ID,
        owner_name=_OWNER_NAME,
        old_profile=_old_profile(explicit_info=[stored], timestamp=_TIMESTAMP),
    )

    prompt = _prompt(fake)
    assert f'"observed_at": "{format_message_timestamp(_TIMESTAMP)} UTC"' in prompt
    assert str(_TIMESTAMP) not in prompt


async def test_newer_episode_rewrites_a_description_and_restamps_it() -> None:
    stored = {
        "category": "Location",
        "description": "Lives in Beijing.",
        "evidence": "Alice lives there.",
        "observed_at": _TIMESTAMP,
    }
    fake = FakeLLMClient(
        responses=[
            _operations(
                _update(0, category="Location", description="Lives in Shanghai.", evidence="Alice moved to Shanghai.")
            )
        ]
    )

    profile = await ProfileExtractor(llm=fake).aextract_from_episodes(
        [_episode("Alice moved to Shanghai.", timestamp=_LATER)],
        owner_id=_OWNER_ID,
        owner_name=_OWNER_NAME,
        old_profile=_old_profile(explicit_info=[stored], timestamp=_TIMESTAMP),
    )

    (item,) = _explicit_items(profile)
    assert item["description"] == "Lives in Shanghai."
    assert item["observed_at"] == _LATER
    assert profile.timestamp == _LATER


async def test_grounding_only_update_keeps_the_item_date() -> None:
    stored = {
        "category": "Location",
        "description": "Lives in Shanghai.",
        "evidence": "Alice moved.",
        "observed_at": _TIMESTAMP,
    }
    fake = FakeLLMClient(responses=[_operations(_update(0, evidence="Alice moved. Alice commutes in Shanghai."))])

    profile = await ProfileExtractor(llm=fake).aextract_from_episodes(
        [_episode("Alice commutes in Shanghai.", timestamp=_LATER)],
        owner_id=_OWNER_ID,
        owner_name=_OWNER_NAME,
        old_profile=_old_profile(explicit_info=[stored], timestamp=_TIMESTAMP),
    )

    (item,) = _explicit_items(profile)
    assert item["evidence"] == "Alice moved. Alice commutes in Shanghai."
    assert item["observed_at"] == _TIMESTAMP


async def test_older_episode_adds_and_grounds_but_never_rewrites_or_deletes_newer_state() -> None:
    """A backfilled Episode observed before the stored state must not turn the profile back in time."""
    lives = {
        "category": "Location",
        "description": "Lives in Shanghai.",
        "evidence": "Alice moved.",
        "observed_at": _TIMESTAMP,
    }
    stack = {
        "category": "Stack",
        "description": "Works mainly in Python.",
        "evidence": "Alice uses Python.",
        "observed_at": _TIMESTAMP,
    }
    fake = FakeLLMClient(
        responses=[
            _operations(
                _update(0, category="Location", description="Lives in Beijing.", evidence="Alice lived in Beijing."),
                _delete(1),
                _update(1, evidence="Alice uses Python. Alice wrote a Python script."),
                _add("Grew up in Chengdu.", evidence="Alice grew up in Chengdu."),
            )
        ]
    )

    profile = await ProfileExtractor(llm=fake).aextract_from_episodes(
        [_episode("Alice lived in Beijing, grew up in Chengdu and wrote a Python script.", timestamp=_EARLIER)],
        owner_id=_OWNER_ID,
        owner_name=_OWNER_NAME,
        old_profile=_old_profile(explicit_info=[lives, stack], timestamp=_TIMESTAMP),
    )

    items = _explicit_items(profile)
    assert [item["description"] for item in items] == [
        "Lives in Shanghai.",
        "Works mainly in Python.",
        "Grew up in Chengdu.",
    ]
    assert items[1]["evidence"] == "Alice uses Python. Alice wrote a Python script."
    assert [item["observed_at"] for item in items] == [_TIMESTAMP, _TIMESTAMP, _EARLIER]
    assert profile.timestamp == _TIMESTAMP


async def test_legacy_items_without_a_date_are_protected_by_the_profile_date() -> None:
    fake = FakeLLMClient(
        responses=[_operations(_update(0, category="Stack", description="Works mainly in Rust.", evidence="Alice."))]
    )

    profile = await ProfileExtractor(llm=fake).aextract_from_episodes(
        [_episode("Alice wrote Rust.", timestamp=_EARLIER)],
        owner_id=_OWNER_ID,
        owner_name=_OWNER_NAME,
        old_profile=_old_profile(timestamp=_TIMESTAMP),
    )

    (item,) = _explicit_items(profile)
    assert item["description"] == "Works mainly in Python."
    assert item["observed_at"] == _TIMESTAMP


async def test_mixed_batch_runs_the_historical_pass_before_the_current_pass() -> None:
    lives = {
        "category": "Location",
        "description": "Lives in Shanghai.",
        "evidence": "Alice moved.",
        "observed_at": _TIMESTAMP,
    }
    fake = FakeLLMClient(
        responses=[
            _operations(_add("Grew up in Chengdu.", evidence="Alice grew up in Chengdu.")),
            _operations(_update(0, category="Location", description="Lives in Hangzhou.", evidence="Alice relocated.")),
        ]
    )

    profile = await ProfileExtractor(llm=fake).aextract_from_episodes(
        [
            _episode("Alice relocated to Hangzhou.", timestamp=_LATER),
            _episode("Alice grew up in Chengdu.", timestamp=_EARLIER),
        ],
        owner_id=_OWNER_ID,
        owner_name=_OWNER_NAME,
        old_profile=_old_profile(explicit_info=[lives], timestamp=_TIMESTAMP),
    )

    assert fake.call_count == 2
    historical_prompt, current_prompt = _prompt(fake, 0), _prompt(fake, 1)
    older, newer = (
        _dated("Alice grew up in Chengdu.", timestamp=_EARLIER),
        _dated("Alice relocated to Hangzhou.", timestamp=_LATER),
    )
    assert older in historical_prompt
    assert newer not in historical_prompt
    assert newer in current_prompt
    assert older not in current_prompt
    # The current pass sees the profile as the historical pass left it, dates included.
    assert "Grew up in Chengdu." in current_prompt
    assert f'"observed_at": "{format_message_timestamp(_EARLIER)} UTC"' in current_prompt
    items = _explicit_items(profile)
    assert [item["description"] for item in items] == ["Lives in Hangzhou.", "Grew up in Chengdu."]
    assert [item["observed_at"] for item in items] == [_LATER, _EARLIER]
    assert profile.timestamp == _LATER


async def test_model_written_observed_at_is_discarded() -> None:
    fake = FakeLLMClient(
        responses=[
            _operations(
                {
                    "action": "add",
                    "type": "explicit_info",
                    "data": {
                        "category": "Location",
                        "description": "Grew up in Chengdu.",
                        "evidence": "Alice grew up in Chengdu.",
                        "observed_at": "2001-01-01 00:00:00 UTC",
                    },
                },
                {
                    "action": "update",
                    "type": "explicit_info",
                    "index": 0,
                    "data": {"evidence": "Alice uses Python daily.", "observed_at": "2001-01-01 00:00:00 UTC"},
                },
            )
        ]
    )

    profile = await ProfileExtractor(llm=fake).aextract_from_episodes(
        [_episode("Alice grew up in Chengdu and uses Python daily.")],
        owner_id=_OWNER_ID,
        owner_name=_OWNER_NAME,
        old_profile=_old_profile(),
    )

    assert [item["observed_at"] for item in _explicit_items(profile)] == [_TIMESTAMP - 1, _TIMESTAMP]


async def test_episode_update_uses_episode_specific_compact_prompt() -> None:
    old_items = [
        {
            "category": f"dimension-{index}",
            "description": f"Stored fact {index}.",
            "evidence": "Narrative evidence.",
            "observed_at": _EARLIER,
        }
        for index in range(60)
    ]
    fake = FakeLLMClient(
        responses=[
            _operations(_add("Prefers Ruff.", category="new", evidence="Alice chose Ruff.")),
            _profile_payload(evidence="Alice chose Ruff."),
        ]
    )

    profile = await ProfileExtractor(llm=fake).aextract_from_episodes(
        [_episode("Alice chose Ruff.")],
        owner_id=_OWNER_ID,
        owner_name=_OWNER_NAME,
        old_profile=_old_profile(explicit_info=old_items),
        categories=_AVAILABLE_CATEGORIES,
    )

    compact_prompt = _prompt(fake, 1)
    assert fake.call_count == 2
    assert "Episode-narrative form" in compact_prompt
    assert "never turn it into a direct user quotation" in compact_prompt
    assert "observed_at" not in compact_prompt.split("【Stored Profile】", maxsplit=1)[1]
    assert all(_rendered_category_snapshot(_prompt(fake, call)) == _expected_category_snapshot() for call in range(2))
    assert profile.owner_id == _OWNER_ID
    assert profile.timestamp == _TIMESTAMP
    # Rewritten items have no traceable origin, so they take the profile's own date.
    assert [item["observed_at"] for item in _explicit_items(profile)] == [_TIMESTAMP]


async def test_episode_update_uses_episode_specific_regroup_prompt() -> None:
    old_items = [
        {
            "category": "Environment",
            "description": f"Environment fact {index}.",
            "evidence": "Narrative evidence.",
            "observed_at": _EARLIER,
        }
        for index in range(8)
    ]
    regroup = json.dumps(
        {"items": [{"category": "Environment", "description": "Uses Ruff.", "evidence": "Alice selected Ruff."}]}
    )
    fake = FakeLLMClient(
        responses=[_operations(_add("Uses Ruff.", category="Environment", evidence="Alice selected Ruff.")), regroup]
    )

    profile = await ProfileExtractor(llm=fake).aextract_from_episodes(
        [_episode("Alice selected Ruff.")],
        owner_id=_OWNER_ID,
        owner_name=_OWNER_NAME,
        old_profile=_old_profile(explicit_info=old_items),
        categories=_AVAILABLE_CATEGORIES,
    )

    regroup_prompt = _prompt(fake, 1)
    assert fake.call_count == 2
    assert "THIS GROUP ONLY" in regroup_prompt
    assert "Episode-narrative evidence excerpts" in regroup_prompt
    assert "observed_at" not in regroup_prompt.split("【Items】", maxsplit=1)[1]
    assert all(_rendered_category_snapshot(_prompt(fake, call)) == _expected_category_snapshot() for call in range(2))
    assert [item["observed_at"] for item in _explicit_items(profile)] == [_TIMESTAMP]


async def test_same_generic_episode_is_extracted_separately_for_each_owner() -> None:
    generic_episode = _episode("Alice selected Python while Bob selected Rust.")
    fake = FakeLLMClient(
        responses=[
            _profile_payload(description="Works mainly in Python.", evidence="Alice selected Python."),
            _profile_payload(description="Works mainly in Rust.", evidence="Bob selected Rust."),
        ]
    )
    extractor = ProfileExtractor(llm=fake)

    alice = await extractor.aextract_from_episodes([generic_episode], owner_id="alice-id", owner_name="Alice")
    bob = await extractor.aextract_from_episodes([generic_episode], owner_id="bob-id", owner_name="Bob")

    assert alice.owner_id == "alice-id"
    assert bob.owner_id == "bob-id"
    assert alice.summary == "Works mainly in Python."
    assert bob.summary == "Works mainly in Rust."
    assert fake.call_count == 2


def test_sync_bridge_extracts_profile() -> None:
    fake = FakeLLMClient(responses=[_profile_payload()])

    profile = ProfileExtractor(llm=fake).extract_from_episodes(
        [_episode("Alice selected Python.")],
        owner_id=_OWNER_ID,
        owner_name=_OWNER_NAME,
        categories=_AVAILABLE_CATEGORIES,
    )

    assert profile.owner_id == _OWNER_ID
    assert fake.call_count == 1
    assert _rendered_category_snapshot(_prompt(fake)) == _expected_category_snapshot()


def test_episode_prompts_preserve_shared_rules_and_isolate_source_specific_contracts() -> None:
    prompts = (
        PROFILE_INITIAL_FROM_EPISODE_TEXTS_PROMPT,
        PROFILE_UPDATE_FROM_EPISODE_TEXTS_PROMPT,
        PROFILE_COMPACT_FROM_EPISODE_TEXTS_PROMPT,
        PROFILE_REGROUP_FROM_EPISODE_TEXTS_PROMPT,
    )

    assert _EPISODE_ITEM_RULES.split("\n\n")[:1] == _MEMCELL_ITEM_RULES.split("\n\n")[:1]
    for prompt in prompts:
        assert _PORTRAIT in prompt
        assert _EPISODE_PRIORITY_RULES in prompt
        assert _CATEGORY_RULES in prompt
        assert prompt.index(_PORTRAIT) < prompt.index(_EPISODE_PRIORITY_RULES)
        assert prompt.count("{language_rule}") == 2
        assert prompt.count("{available_categories}") == 1
        assert "subjectless declarative" in prompt
        assert "never imperative" in prompt or 'imperative "Use Python"' in prompt
        assert "direct user quotation" in prompt
        assert '"evidence" and "basis" are JSON strings, never arrays' in prompt
        assert "faithful paraphrase" in prompt
        assert "invent" in prompt
        assert _rendered_category_snapshot(prompt) == "{available_categories}"


def test_episode_prompts_prioritise_fidelity_and_independent_classification() -> None:
    prompts = (
        PROFILE_INITIAL_FROM_EPISODE_TEXTS_PROMPT,
        PROFILE_UPDATE_FROM_EPISODE_TEXTS_PROMPT,
        PROFILE_COMPACT_FROM_EPISODE_TEXTS_PROMPT,
        PROFILE_REGROUP_FROM_EPISODE_TEXTS_PROMPT,
    )

    for prompt in prompts:
        assert "factual correctness and category accuracy are co-equal" in prompt
        assert "They outrank recall, category count, category reuse" in prompt
        assert "clearly supported by the source and attributable to the profile owner" in prompt
        assert "assistant statements or facts about other participants" in prompt
        assert "one-off actions, passing or short-term states, expiring concrete plans, standalone questions" in prompt
        assert "continuing explanatory value for the long-term portrait" in prompt
        assert "at least two mutually independent, consistent signals" in prompt
        assert "across different Episode narratives" in prompt
        assert "A stored explicit_info item counts as one such signal" in prompt
        assert "leave implicit_traits empty or delete the stored trait" in prompt
        assert "Do not generate an implicit trait merely because the input is detailed" in prompt
        assert "choose the most accurate matching category from the 【Available Categories】 section" in prompt
        assert "Never sacrifice category accuracy merely to reuse a listed category" in prompt
        assert "If the list is empty or no listed category accurately fits" in prompt
        assert "The list is not a whitelist" in prompt
        assert "It does not constrain implicit_traits.trait" in prompt


def test_episode_prompts_explain_dates_and_the_update_prompt_forbids_older_rewrites() -> None:
    assert "[YYYY-MM-DD HH:MM:SS UTC]" in _SOURCE_RULES
    assert "the narratives are given oldest first" in _SOURCE_RULES
    for prompt in (PROFILE_INITIAL_FROM_EPISODE_TEXTS_PROMPT, PROFILE_UPDATE_FROM_EPISODE_TEXTS_PROMPT):
        assert _SOURCE_RULES in prompt
    assert "**Older narratives never rewrite newer state.**" in PROFILE_UPDATE_FROM_EPISODE_TEXTS_PROMPT
    assert "never write observed_at yourself" in PROFILE_UPDATE_FROM_EPISODE_TEXTS_PROMPT
    for prompt in (PROFILE_COMPACT_FROM_EPISODE_TEXTS_PROMPT, PROFILE_REGROUP_FROM_EPISODE_TEXTS_PROMPT):
        assert "observed_at" not in prompt


def test_episode_prompts_do_not_encode_category_sources_or_stored_category_lifecycle() -> None:
    prompts = (
        PROFILE_INITIAL_FROM_EPISODE_TEXTS_PROMPT,
        PROFILE_UPDATE_FROM_EPISODE_TEXTS_PROMPT,
        PROFILE_COMPACT_FROM_EPISODE_TEXTS_PROMPT,
        PROFILE_REGROUP_FROM_EPISODE_TEXTS_PROMPT,
    )
    forbidden_phrases = (
        "built-in category",
        "custom category",
        "categories already in use",
        "existing category",
        "old category",
        "previous category",
        "other names in use",
        "current organisation",
        "prior json content",
        "organisation stable",
        "preserve the profile",
    )

    for prompt in prompts:
        folded = prompt.casefold()
        assert all(phrase not in folded for phrase in forbidden_phrases)


def test_episode_update_example_assigns_category_on_explicit_info_update() -> None:
    expected_update = (
        '{{"action": "update", "type": "explicit_info", "index": 0, '
        '"data": {{"category": "...", "description": "...", "evidence": "..."}}}}'
    )

    assert expected_update in PROFILE_UPDATE_FROM_EPISODE_TEXTS_PROMPT
    assert '"data": {{"description": "...", "evidence": "..."}}' not in PROFILE_UPDATE_FROM_EPISODE_TEXTS_PROMPT


def test_episode_compact_corrects_bucket_without_inventing_grounding() -> None:
    """Full-profile compaction can rehome stored content but cannot manufacture it."""
    assert "**Move anything misfiled.**" in PROFILE_COMPACT_FROM_EPISODE_TEXTS_PROMPT
    assert "do not leave a copy behind in implicit_traits" in PROFILE_COMPACT_FROM_EPISODE_TEXTS_PROMPT
    assert "NEVER by inventing a claim or grounding" in PROFILE_COMPACT_FROM_EPISODE_TEXTS_PROMPT


def test_episode_regroup_preserves_bucket_and_grounding_contracts() -> None:
    """Single-bucket regroup cannot manufacture content or move an item to the unseen bucket."""
    assert "Never change an item's bucket" in PROFILE_REGROUP_FROM_EPISODE_TEXTS_PROMPT
    assert "shown items support" in PROFILE_REGROUP_FROM_EPISODE_TEXTS_PROMPT


async def test_historical_pass_carries_its_own_rule_and_the_current_pass_does_not() -> None:
    """A batch straddling the stored timestamp renders the historical-pass rule only for the older half."""
    from everalgo.user_memory.prompts.en.profile_from_episode_texts import HISTORICAL_PASS_RULE

    fake = FakeLLMClient(responses=[json.dumps({"operations": []}), json.dumps({"operations": []})])
    old = Profile.model_validate(
        {
            "owner_id": _OWNER_ID,
            "summary": "",
            "timestamp": 1_800_000_000_000,
            "explicit_info": [],
            "implicit_traits": [],
        }
    )
    await ProfileExtractor(llm=fake).aextract_from_episodes(
        [
            _episode("Alice moved.", timestamp=1_900_000_000_000),
            _episode("Alice studied.", timestamp=1_700_000_000_000),
        ],
        owner_id=_OWNER_ID,
        owner_name=_OWNER_NAME,
        old_profile=old,
    )
    assert fake.call_count == 2
    first, second = _prompt(fake, 0), _prompt(fake, 1)
    assert HISTORICAL_PASS_RULE in first and "{pass_rule}" not in first
    assert HISTORICAL_PASS_RULE not in second and "{pass_rule}" not in second
