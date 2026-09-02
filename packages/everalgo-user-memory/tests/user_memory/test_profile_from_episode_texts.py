"""Tests for Profile extraction from generic or reflected Episode narrative texts."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

import pytest

from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import Profile
from everalgo.user_memory import OutputLanguage, ProfileExtractor
from everalgo.user_memory.prompts.en.profile import _ITEM_RULES as _MEMCELL_ITEM_RULES
from everalgo.user_memory.prompts.en.profile import _PORTRAIT
from everalgo.user_memory.prompts.en.profile_from_episode_texts import (
    _CATEGORY_RULES,
    _EPISODE_PRIORITY_RULES,
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
_AVAILABLE_CATEGORIES = ("Technical context", "Communication preferences")


def _profile_payload(*, description: str = "Works mainly in Python.", evidence: str = "Alice works in Python.") -> str:
    return json.dumps(
        {
            "explicit_info": [
                {"category": "Technical stack", "description": description, "evidence": evidence},
            ],
            "implicit_traits": [],
        }
    )


def _old_profile(
    *,
    owner_id: str = _OWNER_ID,
    explicit_info: list[dict[str, Any]] | None = None,
) -> Profile:
    items = explicit_info or [
        {"category": "Technical stack", "description": "Works mainly in Python.", "evidence": "Alice uses Python."}
    ]
    return Profile.model_validate(
        {
            "owner_id": owner_id,
            "summary": items[0]["description"],
            "timestamp": _TIMESTAMP - 1,
            "explicit_info": items,
            "implicit_traits": [],
        }
    )


def _rendered_category_snapshot(prompt: str) -> str:
    """Extract the injected category section from a rendered Episode Profile prompt."""
    return prompt.split("【Available Categories】\n", maxsplit=1)[1].split("\n\n【", maxsplit=1)[0]


def _expected_category_snapshot(categories: Sequence[str] = _AVAILABLE_CATEGORIES) -> str:
    """Return the stable JSON representation expected in every Episode Profile stage."""
    return json.dumps(list(categories), ensure_ascii=False, indent=2)


async def test_init_uses_owner_name_and_preserves_authoritative_owner_fields() -> None:
    fake = FakeLLMClient(responses=[_profile_payload(evidence="Alice selected Python for the service.")])

    profile = await ProfileExtractor(llm=fake).aextract_from_episode_texts(
        ["Alice selected Python for the service."],
        owner_id=_OWNER_ID,
        owner_name="  Alice  ",
        timestamp=_TIMESTAMP,
    )

    assert profile.owner_id == _OWNER_ID
    assert profile.timestamp == _TIMESTAMP
    assert profile.summary == "Works mainly in Python."
    assert fake.call_count == 1


async def test_init_falls_back_to_owner_id_when_name_is_missing_or_blank() -> None:
    for owner_name in (None, "   "):
        fake = FakeLLMClient(responses=[_profile_payload(evidence=f"{_OWNER_ID} selected Python.")])

        profile = await ProfileExtractor(llm=fake).aextract_from_episode_texts(
            [f"{_OWNER_ID} selected Python."],
            owner_id=_OWNER_ID,
            owner_name=owner_name,
            timestamp=_TIMESTAMP,
        )

        assert profile.owner_id == _OWNER_ID
        assert fake.call_count == 1


@pytest.mark.parametrize(
    ("episode_texts", "error"),
    [
        ([], "non-empty sequence"),
        (cast("Sequence[str]", "Alice selected Python."), "non-empty sequence"),
        (["   "], r"episode_texts\[0\] must be a non-blank string"),
        (cast("Sequence[str]", [42]), r"episode_texts\[0\] must be a non-blank string"),
        (["Bob selected Python."], r"episode_texts\[0\] does not reference target user 'Alice'"),
    ],
)
async def test_invalid_episode_batches_fail_before_llm(
    episode_texts: Sequence[str],
    error: str,
) -> None:
    fake = FakeLLMClient(responses=[_profile_payload()])

    with pytest.raises(ValueError, match=error):
        await ProfileExtractor(llm=fake).aextract_from_episode_texts(
            episode_texts,
            owner_id=_OWNER_ID,
            owner_name=_OWNER_NAME,
            timestamp=_TIMESTAMP,
        )

    assert fake.call_count == 0


async def test_blank_owner_id_fails_before_llm() -> None:
    fake = FakeLLMClient(responses=[_profile_payload()])

    with pytest.raises(ValueError, match="owner_id must be a non-blank string"):
        await ProfileExtractor(llm=fake).aextract_from_episode_texts(
            ["Alice selected Python."],
            owner_id="   ",
            owner_name=_OWNER_NAME,
            timestamp=_TIMESTAMP,
        )

    assert fake.call_count == 0


async def test_owner_name_takes_precedence_over_owner_id_during_validation() -> None:
    fake = FakeLLMClient(responses=[_profile_payload()])

    with pytest.raises(ValueError, match=r"episode_texts\[0\].*'Alice'"):
        await ProfileExtractor(llm=fake).aextract_from_episode_texts(
            [f"{_OWNER_ID} selected Python."],
            owner_id=_OWNER_ID,
            owner_name=_OWNER_NAME,
            timestamp=_TIMESTAMP,
        )

    assert fake.call_count == 0


async def test_every_episode_must_reference_the_resolved_target() -> None:
    fake = FakeLLMClient(responses=[_profile_payload()])

    with pytest.raises(ValueError, match=r"episode_texts\[1\].*'Alice'"):
        await ProfileExtractor(llm=fake).aextract_from_episode_texts(
            ["Alice selected Python.", "The team selected Ruff."],
            owner_id=_OWNER_ID,
            owner_name=_OWNER_NAME,
            timestamp=_TIMESTAMP,
        )

    assert fake.call_count == 0


async def test_existing_profile_owner_must_match_before_llm() -> None:
    fake = FakeLLMClient(responses=[json.dumps({"operations": [{"action": "none"}]})])

    with pytest.raises(ValueError, match="does not match owner_id"):
        await ProfileExtractor(llm=fake).aextract_from_episode_texts(
            ["Alice selected Python."],
            owner_id=_OWNER_ID,
            owner_name=_OWNER_NAME,
            timestamp=_TIMESTAMP,
            old_profile=_old_profile(owner_id="someone-else"),
        )

    assert fake.call_count == 0


async def test_unsupported_output_language_fails_before_llm() -> None:
    fake = FakeLLMClient(responses=[_profile_payload()])

    with pytest.raises(ValueError, match="unsupported output_language"):
        await ProfileExtractor(llm=fake).aextract_from_episode_texts(
            ["Alice selected Python."],
            owner_id=_OWNER_ID,
            owner_name=_OWNER_NAME,
            timestamp=_TIMESTAMP,
            output_language="Klingon",
        )

    assert fake.call_count == 0


@pytest.mark.parametrize("categories", [None, []])
async def test_none_or_empty_categories_allow_extraction_with_an_empty_snapshot(
    categories: Sequence[str] | None,
) -> None:
    fake = FakeLLMClient(responses=[_profile_payload()])

    await ProfileExtractor(llm=fake).aextract_from_episode_texts(
        ["Alice selected Python."],
        owner_id=_OWNER_ID,
        owner_name=_OWNER_NAME,
        timestamp=_TIMESTAMP,
        categories=categories,
    )

    prompt = cast("str", fake.calls[0].messages[0].content)
    assert _rendered_category_snapshot(prompt) == "[]"


async def test_categories_are_stripped_deduplicated_and_rendered_in_first_seen_order() -> None:
    fake = FakeLLMClient(responses=[_profile_payload()])

    await ProfileExtractor(llm=fake).aextract_from_episode_texts(
        ["Alice selected Python."],
        owner_id=_OWNER_ID,
        owner_name=_OWNER_NAME,
        timestamp=_TIMESTAMP,
        categories=["  Technical context  ", "", "Technical context", "  ", "Communication preferences"],
    )

    prompt = cast("str", fake.calls[0].messages[0].content)
    assert _rendered_category_snapshot(prompt) == _expected_category_snapshot()


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

    profile = await ProfileExtractor(llm=fake).aextract_from_episode_texts(
        ["Alice consistently uses an uncommon build tool."],
        owner_id=_OWNER_ID,
        owner_name=_OWNER_NAME,
        timestamp=_TIMESTAMP,
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
        await ProfileExtractor(llm=fake).aextract_from_episode_texts(
            ["Alice selected Python."],
            owner_id=_OWNER_ID,
            owner_name=_OWNER_NAME,
            timestamp=_TIMESTAMP,
            categories=categories,
        )

    assert fake.call_count == 0


async def test_init_prompt_receives_target_and_unnumbered_episode_narratives() -> None:
    marker_one = "Alice chose marker-one-tooling."
    marker_two = "Alice documented marker-two-preferences."
    fake = FakeLLMClient(responses=[_profile_payload()])

    await ProfileExtractor(llm=fake).aextract_from_episode_texts(
        [marker_one, marker_two],
        owner_id="internal-owner-id",
        owner_name=_OWNER_NAME,
        timestamp=_TIMESTAMP,
        categories=_AVAILABLE_CATEGORIES,
    )

    prompt = cast("str", fake.calls[0].messages[0].content)
    assert "build a profile of Alice" in prompt
    assert "internal-owner-id" not in prompt
    assert prompt.count(marker_one) == 1
    assert prompt.count(marker_two) == 1
    assert f"{marker_one}\n\n---\n\n{marker_two}" in prompt
    assert "[0]" not in prompt
    assert "SAME language EPISODE_TEXT itself is written in" in prompt
    assert _rendered_category_snapshot(prompt) == _expected_category_snapshot()


async def test_explicit_output_language_and_custom_prompt_are_rendered() -> None:
    fake = FakeLLMClient(responses=[_profile_payload()])

    await ProfileExtractor(llm=fake).aextract_from_episode_texts(
        ["Alice selected Python."],
        owner_id=_OWNER_ID,
        owner_name=_OWNER_NAME,
        timestamp=_TIMESTAMP,
        categories=_AVAILABLE_CATEGORIES,
        prompt=(
            "TARGET={target_user}; LANGUAGE={language_rule}; CATEGORIES={available_categories}; "
            "EPISODES={episode_texts}"
        ),
        output_language=OutputLanguage.CHINESE,
    )

    prompt = cast("str", fake.calls[0].messages[0].content)
    assert prompt.startswith("TARGET=Alice")
    assert "Write ALL output fields in Chinese" in prompt
    assert prompt.endswith("EPISODES=Alice selected Python.")
    assert f"CATEGORIES={_expected_category_snapshot()}" in prompt


async def test_update_applies_operations_and_uses_explicit_timestamp() -> None:
    operations = json.dumps(
        {
            "operations": [
                {
                    "action": "add",
                    "type": "explicit_info",
                    "data": {
                        "category": "Communication",
                        "description": "Prefers concise answers.",
                        "evidence": "Alice requested concise answers.",
                    },
                }
            ]
        }
    )
    fake = FakeLLMClient(responses=[operations])

    profile = await ProfileExtractor(llm=fake).aextract_from_episode_texts(
        ["Alice requested concise answers."],
        owner_id=_OWNER_ID,
        owner_name=_OWNER_NAME,
        timestamp=_TIMESTAMP,
        old_profile=_old_profile(),
        categories=_AVAILABLE_CATEGORIES,
    )

    descriptions = [item["description"] for item in profile.explicit_info]  # type: ignore[attr-defined]
    assert profile.owner_id == _OWNER_ID
    assert profile.timestamp == _TIMESTAMP
    assert descriptions == ["Works mainly in Python.", "Prefers concise answers."]
    assert fake.call_count == 1
    update_prompt = cast("str", fake.calls[0].messages[0].content)
    assert _rendered_category_snapshot(update_prompt) == _expected_category_snapshot()
    assert "categories already in use" not in update_prompt


async def test_episode_update_uses_episode_specific_compact_prompt() -> None:
    old_items = [
        {"category": f"dimension-{index}", "description": f"Stored fact {index}.", "evidence": "Narrative evidence."}
        for index in range(60)
    ]
    update = json.dumps(
        {
            "operations": [
                {
                    "action": "add",
                    "type": "explicit_info",
                    "data": {"category": "new", "description": "Prefers Ruff.", "evidence": "Alice chose Ruff."},
                }
            ]
        }
    )
    fake = FakeLLMClient(responses=[update, _profile_payload(evidence="Alice chose Ruff.")])

    profile = await ProfileExtractor(llm=fake).aextract_from_episode_texts(
        ["Alice chose Ruff."],
        owner_id=_OWNER_ID,
        owner_name=_OWNER_NAME,
        timestamp=_TIMESTAMP,
        old_profile=_old_profile(explicit_info=old_items),
        categories=_AVAILABLE_CATEGORIES,
    )

    compact_prompt = cast("str", fake.calls[1].messages[0].content)
    assert fake.call_count == 2
    assert "Episode-narrative form" in compact_prompt
    assert "never turn it into a direct user quotation" in compact_prompt
    assert all(
        _rendered_category_snapshot(cast("str", call.messages[0].content)) == _expected_category_snapshot()
        for call in fake.calls
    )
    assert profile.owner_id == _OWNER_ID
    assert profile.timestamp == _TIMESTAMP


async def test_episode_update_uses_episode_specific_regroup_prompt() -> None:
    old_items = [
        {"category": "Environment", "description": f"Environment fact {index}.", "evidence": "Narrative evidence."}
        for index in range(8)
    ]
    update = json.dumps(
        {
            "operations": [
                {
                    "action": "add",
                    "type": "explicit_info",
                    "data": {
                        "category": "Environment",
                        "description": "Uses Ruff.",
                        "evidence": "Alice selected Ruff.",
                    },
                }
            ]
        }
    )
    regroup = json.dumps(
        {"items": [{"category": "Environment", "description": "Uses Ruff.", "evidence": "Alice selected Ruff."}]}
    )
    fake = FakeLLMClient(responses=[update, regroup])

    await ProfileExtractor(llm=fake).aextract_from_episode_texts(
        ["Alice selected Ruff."],
        owner_id=_OWNER_ID,
        owner_name=_OWNER_NAME,
        timestamp=_TIMESTAMP,
        old_profile=_old_profile(explicit_info=old_items),
        categories=_AVAILABLE_CATEGORIES,
    )

    regroup_prompt = cast("str", fake.calls[1].messages[0].content)
    assert fake.call_count == 2
    assert "THIS GROUP ONLY" in regroup_prompt
    assert "Episode-narrative evidence excerpts" in regroup_prompt
    assert all(
        _rendered_category_snapshot(cast("str", call.messages[0].content)) == _expected_category_snapshot()
        for call in fake.calls
    )


async def test_same_generic_episode_is_extracted_separately_for_each_owner() -> None:
    generic_episode = "Alice selected Python while Bob selected Rust."
    fake = FakeLLMClient(
        responses=[
            _profile_payload(description="Works mainly in Python.", evidence="Alice selected Python."),
            _profile_payload(description="Works mainly in Rust.", evidence="Bob selected Rust."),
        ]
    )
    extractor = ProfileExtractor(llm=fake)

    alice = await extractor.aextract_from_episode_texts(
        [generic_episode], owner_id="alice-id", owner_name="Alice", timestamp=_TIMESTAMP
    )
    bob = await extractor.aextract_from_episode_texts(
        [generic_episode], owner_id="bob-id", owner_name="Bob", timestamp=_TIMESTAMP
    )

    assert alice.owner_id == "alice-id"
    assert bob.owner_id == "bob-id"
    assert alice.summary == "Works mainly in Python."
    assert bob.summary == "Works mainly in Rust."
    assert fake.call_count == 2


def test_sync_bridge_extracts_profile() -> None:
    fake = FakeLLMClient(responses=[_profile_payload()])

    profile = ProfileExtractor(llm=fake).extract_from_episode_texts(
        ["Alice selected Python."],
        owner_id=_OWNER_ID,
        owner_name=_OWNER_NAME,
        timestamp=_TIMESTAMP,
        categories=_AVAILABLE_CATEGORIES,
    )

    assert profile.owner_id == _OWNER_ID
    assert fake.call_count == 1
    prompt = cast("str", fake.calls[0].messages[0].content)
    assert _rendered_category_snapshot(prompt) == _expected_category_snapshot()


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
        assert "leave implicit_traits empty or delete the stored trait" in prompt
        assert "Do not generate an implicit trait merely because the input is detailed" in prompt
        assert "choose the most accurate matching category from the 【Available Categories】 section" in prompt
        assert "Never sacrifice category accuracy merely to reuse a listed category" in prompt
        assert "If the list is empty or no listed category accurately fits" in prompt
        assert "The list is not a whitelist" in prompt
        assert "It does not constrain implicit_traits.trait" in prompt


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
