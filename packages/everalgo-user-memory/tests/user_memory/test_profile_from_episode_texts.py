"""Tests for Profile extraction from dated generic or reflected Episodes."""

from __future__ import annotations

import inspect
import json
import logging
import re
from collections import Counter
from datetime import UTC, datetime
from itertools import pairwise
from typing import TYPE_CHECKING, Any, cast
from xml.etree import ElementTree

import pytest

from everalgo.llm import LLMOutputError
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import Episode, ExplicitInfoItem, ImplicitTraitItem, Profile
from everalgo.user_memory import OutputLanguage, ProfileExtractor
from everalgo.user_memory._language import (
    COMPACTED_PROFILE_LANGUAGE_RULE,
    EPISODE_PROFILE_INIT_LANGUAGE_RULE,
    EXISTING_PROFILE_LANGUAGE_RULE,
    REGROUPED_PROFILE_LANGUAGE_RULE,
    build_language_rule,
)
from everalgo.user_memory.profile._items import _PROFILE_MAX_ITEMS, _PROFILE_MAX_PER_CATEGORY
from everalgo.user_memory.profile._render import _render_episodes, _render_item, _render_profile_for_episode_update
from everalgo.user_memory.profile._replies import (
    _call_llm_for_profile_init,
    _call_llm_for_profile_update,
    _ExplicitItem,
    _ImplicitTrait,
    _ProfileItems,
    _UpdateOutput,
)
from everalgo.user_memory.prompts.en.profile import _PORTRAIT as _MEMCELL_PORTRAIT
from everalgo.user_memory.prompts.en.profile_from_episode_texts import (
    _EXAMPLE_FACTS_AND_A_PATTERN,
    _EXAMPLE_NAMED_LANGUAGE_SAME_SUBJECT,
    _EXAMPLE_OPERATIONS_ONLY,
    _EXAMPLES_INIT,
    _EXAMPLES_UPDATE,
    _HOW_TO_COMPACT,
    _HOW_TO_REGROUP,
    _HOW_TO_UPDATE,
    _JSON_OUTPUT_RULE,
    _LEAVE_UNCERTAIN_OUT,
    _NEVER_IN_THE_PROFILE_LIST,
    _STORED_CLAIM_IS_REMOVED,
    _WHAT_THE_PROFILE_HOLDS,
    PROFILE_COMPACT_FROM_EPISODE_TEXTS_SYSTEM_PROMPT,
    PROFILE_COMPACT_FROM_EPISODE_TEXTS_USER_PROMPT,
    PROFILE_INITIAL_FROM_EPISODE_TEXTS_SYSTEM_PROMPT,
    PROFILE_INITIAL_FROM_EPISODE_TEXTS_USER_PROMPT,
    PROFILE_REGROUP_FROM_EPISODE_TEXTS_SYSTEM_PROMPT,
    PROFILE_REGROUP_FROM_EPISODE_TEXTS_USER_PROMPT,
    PROFILE_UPDATE_FROM_EPISODE_TEXTS_SYSTEM_PROMPT,
    PROFILE_UPDATE_FROM_EPISODE_TEXTS_USER_PROMPT,
    REGROUP_REPLY_SKELETONS,
    additional_requirements_block,
)
from everalgo.utils import format_utc_time, render_prompt

if TYPE_CHECKING:
    from collections.abc import Sequence

_OWNER_ID = "user-123"
_OWNER_NAME = "Alice"
_TIMESTAMP = 1_700_000_010_000
_EARLIER = _TIMESTAMP - 86_400_000
_LATER = _TIMESTAMP + 86_400_000
#: 2024-04-10T14:00:59.999Z: one millisecond short of the next minute, so rounding would show 14:01.
_WEDNESDAY_14_00_59 = 1_712_757_659_999
_AVAILABLE_CATEGORIES = ("Technical context", "Communication preferences")

_SYSTEM_PROMPTS = (
    PROFILE_INITIAL_FROM_EPISODE_TEXTS_SYSTEM_PROMPT,
    PROFILE_UPDATE_FROM_EPISODE_TEXTS_SYSTEM_PROMPT,
    PROFILE_COMPACT_FROM_EPISODE_TEXTS_SYSTEM_PROMPT,
    PROFILE_REGROUP_FROM_EPISODE_TEXTS_SYSTEM_PROMPT,
)
_USER_PROMPTS = (
    PROFILE_INITIAL_FROM_EPISODE_TEXTS_USER_PROMPT,
    PROFILE_UPDATE_FROM_EPISODE_TEXTS_USER_PROMPT,
    PROFILE_COMPACT_FROM_EPISODE_TEXTS_USER_PROMPT,
    PROFILE_REGROUP_FROM_EPISODE_TEXTS_USER_PROMPT,
)
_STAGE_IDS = ("init", "update", "compact", "regroup")
#: Every stage's role sentence opens with these words; the stage's verb follows at once.
_ROLE_PREFIX = "You are a user-profile analyst. "
#: Everything that differs between two requests. The system message must carry none of it, so that it is
#: the same text for every owner and every request.
_DATA_PLACEHOLDERS = (
    "{{target_user}}",
    "{{stored_profile}}",
    "{{episode_texts}}",
    "{{profile_text}}",
    "{{items_text}}",
    "{{total_items}}",
    "{{count}}",
    "{{label}}",
    "{{label_field}}",
    "{{bucket}}",
)
#: The rule texts of the system message; none may reach a user message.
_RULE_BLOCKS = (
    _WHAT_THE_PROFILE_HOLDS,
    _NEVER_IN_THE_PROFILE_LIST,
    _EXAMPLES_INIT,
    _EXAMPLES_UPDATE,
    _HOW_TO_UPDATE,
    _HOW_TO_COMPACT,
    _HOW_TO_REGROUP,
)
#: Every field any stage renders, so one rendering covers all four.
_RENDER_FIELDS = {
    "language_rule": "",
    "target_user": "Bob",
    "available_categories": "- A",
    "episode_texts": "EP",
    "stored_profile": "P",
    "profile_text": "P",
    "total_items": 61,
    "max_items": 60,
    "max_per_category": 8,
    "count": 9,
    "bucket": "explicit_info",
    "reply_skeleton": REGROUP_REPLY_SKELETONS["explicit_info"],
    "label_field": "category",
    "label": "L",
    "items_text": "IT",
}
#: A block delimiter stands alone on its line, so a rule that mentions a tag inline ("the
#: <never_in_the_profile> list") is prose and is not counted as a second delimiter.
_BLOCK_OPEN = re.compile(r'^<([a-z_]+)(?: [a-z_]+="[^"]*")*>$', re.MULTILINE)
_BLOCK_CLOSE = re.compile(r"^</([a-z_]+)>$", re.MULTILINE)
#: The blocks each system message opens: what the profile holds, what never enters it, worked examples (INIT and
#: UPDATE only), the stage's own way of working (INIT has none), the reply format, then the blocks that vary
#: between requests, least-variable first. A worked example's own elements nest inside <examples>.
_SYSTEM_BLOCK_ORDER: dict[str, tuple[str, ...]] = {
    "init": (
        "what_the_profile_holds",
        "never_in_the_profile",
        "examples",
        "reply_format",
        "output_language",
        "available_categories",
    ),
    "update": (
        "what_the_profile_holds",
        "never_in_the_profile",
        "examples",
        "how_to_update",
        "reply_format",
        "output_language",
        "available_categories",
    ),
    "compact": (
        "what_the_profile_holds",
        "never_in_the_profile",
        "how_to_compact",
        "reply_format",
        "output_language",
        "available_categories",
    ),
    "regroup": (
        "what_the_profile_holds",
        "never_in_the_profile",
        "how_to_regroup",
        "reply_format",
        "output_language",
        "available_categories",
    ),
}
#: Both passes of an UPDATE straddling the stored timestamp send the same system message.
_SYSTEM_BLOCK_ORDER["update-historical"] = _SYSTEM_BLOCK_ORDER["update"]
#: The data blocks each user message opens. <target_person> sits inline on one line, so it opens no block.
_USER_BLOCK_ORDER: dict[str, tuple[str, ...]] = {
    "init": ("episode_narratives",),
    "update": ("stored_profile", "episode_narratives"),
    "compact": ("stored_profile",),
    "regroup": ("group",),
}
#: The CJK brackets the retired headings were written with, spelled as escapes so this file is
#: itself free of them and can be swept with the prompts.
_RETIRED_BRACKETS = ("\u3010", "\u3011")
#: The bracketed section names these prompts used before the XML tags. A rule still naming a
#: section in prose would send the model looking for a heading that no longer exists.
_RETIRED_SECTION_NAMES = (
    "Target",
    "Stored Profile",
    "Episode Narratives",
    "Group",
    "Items",
    "Available Categories",
    "Work in this order",
    "How an item is written",
    "Never a portrait item",
    "Example",
    "Output",
)
#: Every text the Episode path can put in the language slot: each caller-named language, then the
#: fallback of each stage.
_LANGUAGE_RULES: dict[str, str] = {
    **{language.value: build_language_rule(language) for language in OutputLanguage},
    "init-fallback": EPISODE_PROFILE_INIT_LANGUAGE_RULE,
    "update-fallback": EXISTING_PROFILE_LANGUAGE_RULE,
    "compacted-fallback": COMPACTED_PROFILE_LANGUAGE_RULE,
    "regrouped-fallback": REGROUPED_PROFILE_LANGUAGE_RULE,
}
_LANGUAGE_BLOCK_OPEN = "\n\n<output_language>\n"


def _top_level_blocks(text: str) -> tuple[list[str], list[str]]:
    """The open and close tags of the blocks at depth zero, each block closed in the order it nests."""
    opened: list[str] = []
    closed: list[str] = []
    stack: list[str] = []
    for match in re.finditer(r'^<(/?)([a-z_]+)(?: [a-z_]+="[^"]*")*>$', text, re.MULTILINE):
        closing, tag = match.groups()
        if not closing:
            if not stack:
                opened.append(tag)
            stack.append(tag)
            continue
        assert stack.pop() == tag, tag
        if not stack:
            closed.append(tag)
    assert not stack, stack
    return opened, closed


def _episode(text: str, *, timestamp: int = _TIMESTAMP) -> Episode:
    return Episode(owner_id=None, episode=text, summary=text, timestamp=timestamp)


def _dated(text: str, *, timestamp: int = _TIMESTAMP) -> str:
    """The narrative exactly as the prompt renders it: an element carrying its observation time."""
    return f'<narrative time="{format_utc_time(timestamp)}">{text}</narrative>'


def _profile_payload(*, description: str = "Works mainly in Python.", evidence: str = "Alice works in Python.") -> str:
    return json.dumps(
        {
            "explicit_info": [
                {"category": "Technical stack", "description": description, "evidence": evidence},
            ],
            "implicit_traits": [],
        }
    )


def _compact_payload() -> str:
    return json.dumps(
        {
            "explicit_info": [
                {
                    "category": "Technical stack",
                    "description": "Works mainly in Python.",
                    "evidence": "Alice chose Ruff.",
                },
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


def _update(index: int, *, category: str, description: str, evidence: str) -> dict[str, Any]:
    """An update restating the whole item, as the UPDATE prompts ask."""
    data = {"category": category, "description": description, "evidence": evidence}
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


def _system_and_user(fake: FakeLLMClient, call: int = 0) -> tuple[str, str]:
    """The two messages of one Episode-path call, after checking that is exactly what was sent."""
    system, user = fake.calls[call].messages
    assert (system.role, user.role) == ("system", "user")
    return cast("str", system.content), cast("str", user.content)


def _system_prompt(fake: FakeLLMClient, call: int = 0) -> str:
    return _system_and_user(fake, call)[0]


def _user_prompt(fake: FakeLLMClient, call: int = 0) -> str:
    return _system_and_user(fake, call)[1]


def _rendered_category_snapshot(prompt: str) -> str:
    """Extract the injected category section from a rendered Episode Profile prompt."""
    body = prompt.split("<available_categories>\n", maxsplit=1)[1]
    return body.split("\n</available_categories>", maxsplit=1)[0]


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
        ([_episode("   ")], r"episodes\[0\]\.episode is blank"),
        (
            [_episode("Bob selected Python."), _episode("The team selected Ruff.")],
            "no episode narrative mentions target user 'Alice'",
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

    with pytest.raises(ValueError, match="owner_id is blank"):
        await ProfileExtractor(llm=fake).aextract_from_episodes(
            [_episode("Alice selected Python.")],
            owner_id="   ",
            owner_name=_OWNER_NAME,
        )

    assert fake.call_count == 0


async def test_owner_name_takes_precedence_over_owner_id_during_validation() -> None:
    fake = FakeLLMClient(responses=[_profile_payload()])

    with pytest.raises(ValueError, match="no episode narrative mentions target user 'Alice'"):
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

    prompt = _user_prompt(fake)
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

    prompt = _user_prompt(fake)
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

    with pytest.raises(ValueError, match="names no supported language"):
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

    assert _rendered_category_snapshot(_system_prompt(fake)) == "[]"


async def test_categories_are_stripped_deduplicated_and_rendered_in_first_seen_order() -> None:
    fake = FakeLLMClient(responses=[_profile_payload()])

    await ProfileExtractor(llm=fake).aextract_from_episodes(
        [_episode("Alice selected Python.")],
        owner_id=_OWNER_ID,
        owner_name=_OWNER_NAME,
        categories=["  Technical context  ", "", "Technical context", "  ", "Communication preferences"],
    )

    assert _rendered_category_snapshot(_system_prompt(fake)) == _expected_category_snapshot()


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

    system, user = _system_and_user(fake)
    assert user.startswith(f"<target_person>{_OWNER_NAME}</target_person>\n\n<episode_narratives>\n")
    assert (
        "Build the profile of the target person from the Episode narratives. The target person is named in "
        "<target_person>. The Episode narratives are in <episode_narratives>."
    ) in system
    assert "internal-owner-id" not in system + user
    assert user.count(marker_one) == 1
    assert user.count(marker_two) == 1
    assert f"{_dated(marker_one, timestamp=_EARLIER)}\n{_dated(marker_two, timestamp=_LATER)}" in user
    assert "[0]" not in user
    assert "---" not in user
    assert EPISODE_PROFILE_INIT_LANGUAGE_RULE in system
    assert _rendered_category_snapshot(system) == _expected_category_snapshot()


async def test_explicit_output_language_and_additional_requirements_are_both_rendered() -> None:
    """The caller's requirements are added to the bundled rules; they replace none of them, nor the data message."""
    fake = FakeLLMClient(responses=[_profile_payload()])

    await ProfileExtractor(llm=fake).aextract_from_episodes(
        [_episode("Alice selected Python.")],
        owner_id=_OWNER_ID,
        owner_name=_OWNER_NAME,
        categories=_AVAILABLE_CATEGORIES,
        custom_instructions="Prefer the categories as listed.",
        output_language=OutputLanguage.CHINESE,
    )

    system, user = _system_and_user(fake)
    assert system.startswith(_ROLE_PREFIX)
    assert build_language_rule(OutputLanguage.CHINESE) in system
    assert _rendered_category_snapshot(system) == _expected_category_snapshot()
    assert system.endswith("Prefer the categories as listed.\n</additional_requirements>")
    assert user == (
        f"<target_person>Alice</target_person>\n\n<episode_narratives>\n{_dated('Alice selected Python.')}\n</episode_narratives>"
    )


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
    system, user = _system_and_user(fake)
    assert _rendered_category_snapshot(system) == _expected_category_snapshot()
    assert "categories already in use" not in system
    # The model is shown the same date the merge will judge the legacy item by.
    assert f'"observed_at": "{format_utc_time(_TIMESTAMP - 1)}"' in user


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

    prompt = _user_prompt(fake)
    assert f'"observed_at": "{format_utc_time(_TIMESTAMP)}"' in prompt
    assert str(_TIMESTAMP) not in prompt


async def test_narrative_time_attribute_is_the_utc_minute_seconds_truncated() -> None:
    fake = FakeLLMClient(responses=[_profile_payload()])

    await ProfileExtractor(llm=fake).aextract_from_episodes(
        [_episode("Alice selected Python.", timestamp=_WEDNESDAY_14_00_59)],
        owner_id=_OWNER_ID,
        owner_name=_OWNER_NAME,
        categories=_AVAILABLE_CATEGORIES,
    )

    user = _user_prompt(fake)
    assert '<episode_narratives>\n<narrative time="2024-04-10 14:00 UTC">Alice selected Python.</narrative>\n' in user
    assert "[2024-04-10 14:00 UTC]" not in user


async def test_stored_observed_at_is_shown_as_the_utc_minute_seconds_truncated() -> None:
    fake = FakeLLMClient(responses=[_operations({"action": "none"})])
    stored = {
        "category": "Location",
        "description": "Lives in Shanghai.",
        "evidence": "Alice moved.",
        "observed_at": _WEDNESDAY_14_00_59,
    }

    await ProfileExtractor(llm=fake).aextract_from_episodes(
        [_episode("Alice adopted uv.", timestamp=_WEDNESDAY_14_00_59 + 86_400_000)],
        owner_id=_OWNER_ID,
        owner_name=_OWNER_NAME,
        old_profile=_old_profile(explicit_info=[stored], timestamp=_WEDNESDAY_14_00_59),
    )

    assert '"observed_at": "2024-04-10 14:00 UTC"' in _user_prompt(fake)


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
    fake = FakeLLMClient(
        responses=[
            _operations(
                _update(
                    0,
                    category="Location",
                    description="Lives in Shanghai.",
                    evidence="Alice moved. Alice commutes in Shanghai.",
                )
            )
        ]
    )

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
                _update(
                    1,
                    category="Stack",
                    description="Works mainly in Python.",
                    evidence="Alice uses Python. Alice wrote a Python script.",
                ),
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


_CHENGDU = {
    "category": "Location",
    "description": "Lives in Chengdu",
    "evidence": "Alice said she lives in Chengdu.",
    "observed_at": _TIMESTAMP,
}
_CHONGQING_EVIDENCE = "Alice said she lived in Chongqing; later she said she lives in Chengdu."
_UPDATE_COUNTERS = ("rejected", "rewrote_description", "history_appended", "grounding_only")


@pytest.mark.parametrize(
    ("description", "counter"),
    [
        pytest.param("Lives in Chengdu (earlier in Chongqing)", "history_appended", id="history-appended"),
        pytest.param("Lives in Chengdu", "grounding_only", id="grounding-only"),
        pytest.param("Lives in Chongqing (moved to Chengdu later)", "rejected", id="current-value-replaced"),
    ],
)
async def test_older_episode_may_add_history_after_the_current_value_but_never_replace_it(
    description: str, counter: str, caplog: pytest.LogCaptureFixture
) -> None:
    """An older batch's update lands only if it keeps the stored description whole; the item's date never moves."""
    fake = FakeLLMClient(
        responses=[_operations(_update(0, category="Location", description=description, evidence=_CHONGQING_EVIDENCE))]
    )

    with caplog.at_level(logging.INFO, logger=_PROFILE_LOGGER):
        profile = await ProfileExtractor(llm=fake).aextract_from_episodes(
            [_episode("Alice lived in Chongqing.", timestamp=_EARLIER)],
            owner_id=_OWNER_ID,
            owner_name=_OWNER_NAME,
            old_profile=_old_profile(explicit_info=[_CHENGDU], timestamp=_TIMESTAMP),
        )

    (item,) = _explicit_items(profile)
    landed = (description, _CHONGQING_EVIDENCE)
    expected = (_CHENGDU["description"], _CHENGDU["evidence"]) if counter == "rejected" else landed
    assert (item["description"], item["evidence"]) == expected
    # The later narrative still holds the current value, so it still dates the item.
    assert item["observed_at"] == _TIMESTAMP
    assert profile.timestamp == _TIMESTAMP
    (applied,) = _log_fields(caplog, "profile update applied:")
    assert {key: applied[key] for key in _UPDATE_COUNTERS} == {
        key: str(int(key == counter)) for key in _UPDATE_COUNTERS
    }


@pytest.mark.parametrize(
    ("stored", "proposed", "counter"),
    [
        pytest.param(
            "住在成都", "住在成都\uff08此前住在重庆\uff0c后换城市到成都\uff09", "history_appended", id="zh-parenthesis"
        ),
        pytest.param(
            "Lives in Chengdu.", "Lives in Chengdu (earlier in Chongqing).", "history_appended", id="en-stop-moved"
        ),
        pytest.param(
            "住在成都\u3002", "住在成都\uff08此前住在重庆\uff09\u3002", "history_appended", id="zh-stop-moved"
        ),
        pytest.param("Lives in Chengdu", "No longer lives in Chengdu", "rejected", id="en-negation-before"),
        pytest.param("住在成都", "不再住在成都", "rejected", id="zh-negation-before"),
        pytest.param("养狗", "曾经养狗\uff0c现在不养了", "rejected", id="contradiction-around"),
        pytest.param("住在成都", "住在成都\uff0c此前住在重庆", "rejected", id="history-without-parenthesis"),
    ],
)
async def test_older_episode_history_is_the_stored_value_then_a_parenthesis(
    stored: str, proposed: str, counter: str, caplog: pytest.LogCaptureFixture
) -> None:
    """The stored words open the new description, their closing stop aside, and only a parenthesis follows them."""
    item = {"category": "Location", "description": stored, "evidence": "Alice said so.", "observed_at": _TIMESTAMP}
    fake = FakeLLMClient(
        responses=[_operations(_update(0, category="Location", description=proposed, evidence=_CHONGQING_EVIDENCE))]
    )

    with caplog.at_level(logging.INFO, logger=_PROFILE_LOGGER):
        profile = await ProfileExtractor(llm=fake).aextract_from_episodes(
            [_episode("Alice lived in Chongqing.", timestamp=_EARLIER)],
            owner_id=_OWNER_ID,
            owner_name=_OWNER_NAME,
            old_profile=_old_profile(explicit_info=[item], timestamp=_TIMESTAMP),
        )

    (kept,) = _explicit_items(profile)
    assert kept["description"] == (stored if counter == "rejected" else proposed)
    assert kept["observed_at"] == _TIMESTAMP
    (applied,) = _log_fields(caplog, "profile update applied:")
    assert {key: applied[key] for key in _UPDATE_COUNTERS} == {
        key: str(int(key == counter)) for key in _UPDATE_COUNTERS
    }


async def test_newer_episode_extending_a_description_is_a_rewrite_and_restamps_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Keeping the old words is history only from an older batch; from a newer one it states a new current value."""
    extended = "Lives in Chengdu (in the Gaoxin district)"
    fake = FakeLLMClient(
        responses=[_operations(_update(0, category="Location", description=extended, evidence="Alice moved within."))]
    )

    with caplog.at_level(logging.INFO, logger=_PROFILE_LOGGER):
        profile = await ProfileExtractor(llm=fake).aextract_from_episodes(
            [_episode("Alice moved to the Gaoxin district.", timestamp=_LATER)],
            owner_id=_OWNER_ID,
            owner_name=_OWNER_NAME,
            old_profile=_old_profile(explicit_info=[_CHENGDU], timestamp=_TIMESTAMP),
        )

    (item,) = _explicit_items(profile)
    assert (item["description"], item["observed_at"]) == (extended, _LATER)
    (applied,) = _log_fields(caplog, "profile update applied:")
    assert (applied["rewrote_description"], applied["history_appended"]) == ("1", "0")


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
    historical_prompt, current_prompt = _user_prompt(fake, 0), _user_prompt(fake, 1)
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
    assert f'"observed_at": "{format_utc_time(_EARLIER)}"' in current_prompt
    items = _explicit_items(profile)
    assert [item["description"] for item in items] == ["Lives in Hangzhou.", "Grew up in Chengdu."]
    assert [item["observed_at"] for item in items] == [_LATER, _EARLIER]
    assert profile.timestamp == _LATER


async def test_model_written_observed_at_raises_llm_output_error() -> None:
    """``observed_at`` is code-owned: a reply that writes it fails the item model instead of being cleaned up."""
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
                        "observed_at": "2001-01-01 00:00 UTC",
                    },
                },
                {
                    "action": "update",
                    "type": "explicit_info",
                    "index": 0,
                    "data": {
                        "category": "Technical stack",
                        "description": "Works mainly in Python.",
                        "evidence": "Alice uses Python daily.",
                        "observed_at": "2001-01-01 00:00 UTC",
                    },
                },
            )
        ]
    )

    with pytest.raises(LLMOutputError):
        await ProfileExtractor(llm=fake).aextract_from_episodes(
            [_episode("Alice grew up in Chengdu and uses Python daily.")],
            owner_id=_OWNER_ID,
            owner_name=_OWNER_NAME,
            old_profile=_old_profile(),
        )


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
            _compact_payload(),
        ]
    )

    profile = await ProfileExtractor(llm=fake).aextract_from_episodes(
        [_episode("Alice chose Ruff.")],
        owner_id=_OWNER_ID,
        owner_name=_OWNER_NAME,
        old_profile=_old_profile(explicit_info=old_items),
        categories=_AVAILABLE_CATEGORIES,
    )

    compact_system, compact_user = _system_and_user(fake, 1)
    assert fake.call_count == 2
    # COMPACT points at the one statement of the grounding form instead of restating it.
    assert "Keep every kept item's grounding in the form <what_the_profile_holds> gives." in compact_system
    assert "<how_an_item_is_written>" not in compact_system
    assert "use a direct user quotation only where the source itself contains that quotation" not in compact_system
    assert "Episode-narrative form" not in compact_system
    assert "never turn it into a direct user quotation" not in compact_system
    # The live count belongs to the request, so it travels in the user message; the cap stays in the rules.
    assert compact_user.startswith(
        f'<target_person>{_OWNER_NAME}</target_person>\n\n<stored_profile total_items="61">\n'
    )
    assert "observed_at" not in compact_user
    assert all(
        _rendered_category_snapshot(_system_prompt(fake, call)) == _expected_category_snapshot() for call in range(2)
    )
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
        {
            "items": [{"category": "Environment", "description": "Uses Ruff.", "evidence": "Alice selected Ruff."}],
        }
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

    regroup_system, regroup_user = _system_and_user(fake, 1)
    assert fake.call_count == 2
    assert "Reorganise this group only." in regroup_system
    assert "the grounding in the form <what_the_profile_holds> gives" in regroup_system
    assert "<how_an_item_is_written>" not in regroup_system
    assert "Episode-narrative evidence excerpts" not in regroup_system
    # One element carries the group's attributes, its bucket first, and wraps its items; no separate <items> block.
    assert regroup_user.startswith(
        f"<target_person>{_OWNER_NAME}</target_person>\n\n"
        '<group bucket="explicit_info" label_field="category" label="Environment" count="9">\n['
    )
    assert regroup_user.endswith("]\n</group>")
    assert "items>" not in regroup_user
    assert "observed_at" not in regroup_user
    assert all(
        _rendered_category_snapshot(_system_prompt(fake, call)) == _expected_category_snapshot() for call in range(2)
    )
    assert [item["observed_at"] for item in _explicit_items(profile)] == [_TIMESTAMP]


def _labelled_items(label: str, count: int) -> list[dict[str, Any]]:
    return [
        {"category": label, "description": f"{label} fact {index}.", "evidence": "Narrative evidence."}
        for index in range(count)
    ]


def _distinct_items(count: int) -> list[dict[str, Any]]:
    return [
        {"category": f"dimension-{index}", "description": f"Stored fact {index}.", "evidence": "Narrative evidence."}
        for index in range(count)
    ]


async def _update_past_the_total_cap(
    compacted: list[dict[str, Any]], *later_replies: str
) -> tuple[FakeLLMClient, Profile]:
    """UPDATE adds one item to 69 stored ones, so 70 breach the total cap and COMPACT answers with ``compacted``."""
    stored = [{**item, "observed_at": _EARLIER} for item in _distinct_items(69)]
    assert len(stored) + 1 > _PROFILE_MAX_ITEMS
    fake = FakeLLMClient(
        responses=[
            _operations(_add("Plays chess.", category="new-dimension", evidence="Alice plays chess.")),
            json.dumps({"explicit_info": compacted, "implicit_traits": []}),
            *later_replies,
        ]
    )
    profile = await ProfileExtractor(llm=fake).aextract_from_episodes(
        [_episode("Alice plays chess.")],
        owner_id=_OWNER_ID,
        owner_name=_OWNER_NAME,
        old_profile=_old_profile(explicit_info=stored),
        categories=_AVAILABLE_CATEGORIES,
    )
    return fake, profile


async def test_a_group_compact_leaves_over_its_cap_is_regrouped_in_the_same_call(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """COMPACT meets the total cap but leaves one group at 11; that group gets one REGROUP before the call returns."""
    hobbies = _labelled_items("Hobbies", 11)
    assert len(hobbies) > _PROFILE_MAX_PER_CATEGORY
    regrouped = json.dumps({"items": [*hobbies[:6], *({**item, "category": "Sports"} for item in hobbies[6:])]})

    with caplog.at_level(logging.WARNING, logger=_PROFILE_LOGGER):
        fake, profile = await _update_past_the_total_cap([*hobbies, *_distinct_items(40)], regrouped)

    assert fake.call_count == 3  # UPDATE, COMPACT, one REGROUP
    _, compact_user = _system_and_user(fake, 1)
    assert '<stored_profile total_items="70">' in compact_user
    _, regroup_user = _system_and_user(fake, 2)
    assert '<group bucket="explicit_info" label_field="category" label="Hobbies" count="11">' in regroup_user
    assert "Stored fact 0." not in regroup_user  # the groups COMPACT left within their caps are not shown
    assert "group_items=11" in caplog.text
    labels = Counter(item["category"] for item in _explicit_items(profile))
    assert (labels["Hobbies"], labels["Sports"]) == (6, 5)
    assert max(labels.values()) <= _PROFILE_MAX_PER_CATEGORY
    assert len(_explicit_items(profile)) == 51


async def test_a_compact_result_within_every_group_cap_is_not_regrouped() -> None:
    """A group COMPACT leaves at exactly its cap is within it: COMPACT's result is returned as it came back."""
    at_cap = _labelled_items("Hobbies", _PROFILE_MAX_PER_CATEGORY)

    fake, profile = await _update_past_the_total_cap([*at_cap, *_distinct_items(43)])

    assert fake.call_count == 2  # UPDATE, COMPACT
    labels = Counter(item["category"] for item in _explicit_items(profile))
    assert labels["Hobbies"] == _PROFILE_MAX_PER_CATEGORY
    assert len(_explicit_items(profile)) == 51


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
    assert _rendered_category_snapshot(_system_prompt(fake)) == _expected_category_snapshot()


async def test_episode_stages_send_a_data_free_system_message_and_a_target_led_user_message() -> None:
    """The rules go in a system message that never sees the request; the person and the data go in the user message."""
    owner = "Zaphod"
    narrative = "Zaphod chose marker-one-tooling."
    fake = FakeLLMClient(responses=[_profile_payload(), _operations({"action": "none"})])
    extractor = ProfileExtractor(llm=fake)

    await extractor.aextract_from_episodes([_episode(narrative)], owner_id=_OWNER_ID, owner_name=owner)
    await extractor.aextract_from_episodes(
        [_episode(narrative)],
        owner_id=_OWNER_ID,
        owner_name=owner,
        old_profile=_old_profile(),
    )

    assert fake.call_count == 2
    for call in range(2):
        system, user = _system_and_user(fake, call)
        assert owner not in system
        assert narrative not in system
        assert user.startswith(f"<target_person>{owner}</target_person>\n\n")
        assert narrative in user


async def test_system_message_is_identical_across_owners() -> None:
    """Two owners, same categories: the system message is byte-identical, only the user message differs."""
    fake = FakeLLMClient(responses=[_profile_payload(), _profile_payload()])
    extractor = ProfileExtractor(llm=fake)

    for owner in ("Zaphod", "Trillian"):
        await extractor.aextract_from_episodes(
            [_episode(f"{owner} selected Python.")],
            owner_id=f"{owner.lower()}-id",
            owner_name=owner,
            categories=_AVAILABLE_CATEGORIES,
        )

    assert _system_prompt(fake, 0) == _system_prompt(fake, 1)
    assert _user_prompt(fake, 0) != _user_prompt(fake, 1)


def test_episode_system_prompts_share_one_block_per_rule_in_working_order() -> None:
    """Every stage's system prompt carries the shared rules once each, in working order.

    What a profile holds, what never enters one, then the stage's own way of working and its reply format. Each rule
    has one home, so nothing later in the prompt can quietly override something earlier.
    """
    for prompt in _SYSTEM_PROMPTS:
        assert prompt.index(_WHAT_THE_PROFILE_HOLDS) < prompt.index(_NEVER_IN_THE_PROFILE_LIST)
        assert prompt.index(_NEVER_IN_THE_PROFILE_LIST) < prompt.index("<reply_format>")
        assert prompt.count(_WHAT_THE_PROFILE_HOLDS) == 1
        assert prompt.count(_NEVER_IN_THE_PROFILE_LIST) == 1
        assert prompt.count(_LEAVE_UNCERTAIN_OUT) == 1
        assert prompt.startswith(_ROLE_PREFIX)
        assert prompt.count("{{language_rule}}") == 1
        assert prompt.count("{{available_categories}}") == 1
        assert _rendered_category_snapshot(prompt) == "{{available_categories}}"
    # This path owns its definition: the MemCell block is not spliced in and cannot be overridden.
    assert _WHAT_THE_PROFILE_HOLDS != _MEMCELL_PORTRAIT
    for prompt in _SYSTEM_PROMPTS:
        assert _MEMCELL_PORTRAIT not in prompt


@pytest.mark.parametrize(
    ("stage", "prompt", "block"),
    [
        ("update", PROFILE_UPDATE_FROM_EPISODE_TEXTS_SYSTEM_PROMPT, _HOW_TO_UPDATE),
        ("compact", PROFILE_COMPACT_FROM_EPISODE_TEXTS_SYSTEM_PROMPT, _HOW_TO_COMPACT),
        ("regroup", PROFILE_REGROUP_FROM_EPISODE_TEXTS_SYSTEM_PROMPT, _HOW_TO_REGROUP),
    ],
    ids=["update", "compact", "regroup"],
)
def test_each_maintenance_stage_carries_only_its_own_way_of_working(stage: str, prompt: str, block: str) -> None:
    assert f"\n<how_to_{stage}>\n{block}\n</how_to_{stage}>\n" in prompt
    for other in {_HOW_TO_UPDATE, _HOW_TO_COMPACT, _HOW_TO_REGROUP} - {block}:
        assert other not in prompt


def test_only_update_rewrites_a_stored_item_in_another_language() -> None:
    """COMPACT and REGROUP rewrite every item they return anyway, and INIT has nothing stored."""
    sentence = (
        "The profile is written in one language, the one <output_language> gives. A stored item written in any other "
        "language is updated: its category or trait, its description and its evidence or basis rewritten in that "
        "language and nothing else changed, even when no narrative mentions it."
    )
    init, update, compact, regroup = _SYSTEM_PROMPTS

    assert update.count(sentence) == 1
    for prompt in (init, compact, regroup):
        assert sentence not in prompt
        assert "A stored item written in any other language" not in prompt


def test_init_carries_no_way_of_working_block() -> None:
    """Role, definition and exclusions already say everything INIT needs."""
    for tag in ("how_to_update", "how_to_compact", "how_to_regroup", "work_in_this_order"):
        assert tag not in PROFILE_INITIAL_FROM_EPISODE_TEXTS_SYSTEM_PROMPT


@pytest.mark.parametrize("prompt", _SYSTEM_PROMPTS[1:], ids=_STAGE_IDS[1:])
def test_stages_that_read_stored_items_remove_a_stored_excluded_claim(prompt: str) -> None:
    """The sentence sits at the end of <never_in_the_profile>, right before the uncertainty rule, once."""
    exclusions = _block(prompt, "never_in_the_profile")

    assert prompt.count(_STORED_CLAIM_IS_REMOVED) == 1
    assert exclusions.endswith(f"\n\n{_STORED_CLAIM_IS_REMOVED} {_LEAVE_UNCERTAIN_OUT}")


def test_init_never_in_the_profile_has_no_stored_item_sentence() -> None:
    """INIT has nothing on file, so the sentence about removing a stored item would point at nothing."""
    exclusions = _block(PROFILE_INITIAL_FROM_EPISODE_TEXTS_SYSTEM_PROMPT, "never_in_the_profile")

    assert _STORED_CLAIM_IS_REMOVED not in PROFILE_INITIAL_FROM_EPISODE_TEXTS_SYSTEM_PROMPT
    assert "A stored item" not in PROFILE_INITIAL_FROM_EPISODE_TEXTS_SYSTEM_PROMPT
    assert exclusions == f"{_NEVER_IN_THE_PROFILE_LIST}\n\n{_LEAVE_UNCERTAIN_OUT}"


def test_never_in_the_profile_lists_six_exclusions_one_per_line() -> None:
    lines = _NEVER_IN_THE_PROFILE_LIST.splitlines()

    assert lines[0] == "Whatever the wording and whatever evidence is attached, these never become items:"
    assert len(lines) == 7
    assert all(line.startswith("- ") for line in lines[1:])


def test_never_in_the_profile_excludes_what_the_target_person_tells_about_someone_else() -> None:
    """Being told is the fourth exclusion; a fact about another person is its own, even when the owner says it."""
    lines = _NEVER_IN_THE_PROFILE_LIST.splitlines()
    told = (
        "- A single question, and anything the assistant or another participant said: being told something is not "
        "knowing it."
    )
    # The example gives nothing: round 5 turned "Has a brother" into bare items ("has a colleague"). A relationship
    # enters only as a standing feature of the person's own life, carrying nothing about the other person.
    other_person = (
        "- A fact about anyone other than the target person, even one the target person told, and even inside an item "
        'about the relationship. "My brother is an accountant in Shanghai" gives nothing: the accounting and Shanghai '
        'are his, and "Has a brother" on its own is an empty item. A relationship enters only as a standing feature of '
        "this person's own life — a partner, someone they live with — and then with nothing about the other person."
    )

    assert lines[4] == told
    assert lines[5] == other_person


@pytest.mark.parametrize(("stage", "prompt"), list(zip(_STAGE_IDS, _SYSTEM_PROMPTS, strict=True)), ids=_STAGE_IDS)
def test_episode_system_prompts_put_the_static_rules_first_and_the_per_request_blocks_last(
    stage: str, prompt: str
) -> None:
    """Role sentence, the static rule blocks, then the per-request blocks, <available_categories> last.

    Prompt caches match on a prefix, so the blocks that vary between requests close the template, least
    variable first: the language rule (a few fixed texts), the category snapshot (supplied per request).
    """
    order = _SYSTEM_BLOCK_ORDER[stage]
    positions = [prompt.index(f"\n\n<{tag}>\n") for tag in order]
    role = prompt[: positions[0]]
    assert role.startswith(_ROLE_PREFIX)
    assert _BLOCK_OPEN.findall(role) == []
    assert all(earlier < later for earlier, later in pairwise(positions))
    assert order[-3:] == ("reply_format", "output_language", "available_categories")
    assert prompt.endswith("<available_categories>\n{{available_categories}}\n</available_categories>")


@pytest.mark.parametrize("rule", list(_LANGUAGE_RULES.values()), ids=list(_LANGUAGE_RULES))
@pytest.mark.parametrize("prompt", _SYSTEM_PROMPTS, ids=_STAGE_IDS)
def test_episode_language_rule_renders_in_its_own_tag_between_reply_format_and_categories(
    prompt: str, rule: str
) -> None:
    system = render_prompt(prompt, None, **{**_RENDER_FIELDS, "language_rule": rule})

    block = f"{_LANGUAGE_BLOCK_OPEN}{rule}\n</output_language>\n\n<available_categories>\n"
    assert system.count(rule) == 1
    assert block in system
    assert system.index("\n\n<reply_format>\n") < system.index(block)


@pytest.mark.parametrize("prompt", _SYSTEM_PROMPTS, ids=_STAGE_IDS)
def test_episode_system_prefix_before_the_language_rule_is_the_same_for_every_language(prompt: str) -> None:
    """Everything above <output_language> is static, so every language can reuse one cached prefix."""
    assert "{{language_rule}}" not in prompt.split(_LANGUAGE_BLOCK_OPEN, maxsplit=1)[0]
    prefixes = {
        render_prompt(prompt, None, **{**_RENDER_FIELDS, "language_rule": rule}).split(
            _LANGUAGE_BLOCK_OPEN, maxsplit=1
        )[0]
        for rule in _LANGUAGE_RULES.values()
    }
    assert len(prefixes) == 1
    (prefix,) = prefixes
    for rule in _LANGUAGE_RULES.values():
        assert rule not in prefix


async def test_system_messages_of_two_languages_differ_only_after_the_static_prefix() -> None:
    fake = FakeLLMClient(responses=[_profile_payload(), _profile_payload()])
    extractor = ProfileExtractor(llm=fake)

    for language in (OutputLanguage.CHINESE, OutputLanguage.GERMAN):
        await extractor.aextract_from_episodes(
            [_episode("Alice selected Python.")],
            owner_id=_OWNER_ID,
            owner_name=_OWNER_NAME,
            categories=_AVAILABLE_CATEGORIES,
            output_language=language,
        )

    chinese, german = _system_prompt(fake, 0), _system_prompt(fake, 1)
    assert chinese != german
    assert _LANGUAGE_BLOCK_OPEN in chinese
    assert chinese.split(_LANGUAGE_BLOCK_OPEN, maxsplit=1)[0] == german.split(_LANGUAGE_BLOCK_OPEN, maxsplit=1)[0]


@pytest.mark.parametrize("prompt", _SYSTEM_PROMPTS, ids=_STAGE_IDS)
def test_episode_system_prompts_carry_no_request_data(prompt: str) -> None:
    """The system message is the same for every owner and request: no name, no data, no live count."""
    for placeholder in _DATA_PLACEHOLDERS:
        assert placeholder not in prompt
    assert "<target_person>" in prompt  # It tells the model where the person is named — without naming them.


def test_only_regroup_takes_a_reply_skeleton_and_only_inside_its_reply_format() -> None:
    """The skeleton slot is a per-bucket variant of the rules, not request data: two fixed texts, one per bucket."""
    for prompt in _SYSTEM_PROMPTS[:-1]:
        assert "{{reply_skeleton}}" not in prompt
    assert PROFILE_REGROUP_FROM_EPISODE_TEXTS_SYSTEM_PROMPT.count("{{reply_skeleton}}") == 1
    assert _block(PROFILE_REGROUP_FROM_EPISODE_TEXTS_SYSTEM_PROMPT, "reply_format").endswith("\n\n{{reply_skeleton}}")
    assert set(REGROUP_REPLY_SKELETONS) == {"explicit_info", "implicit_traits"}


@pytest.mark.parametrize(
    ("prompt", "placeholders"),
    [
        (PROFILE_INITIAL_FROM_EPISODE_TEXTS_USER_PROMPT, ("{{episode_texts}}",)),
        (PROFILE_UPDATE_FROM_EPISODE_TEXTS_USER_PROMPT, ("{{stored_profile}}", "{{episode_texts}}")),
        (PROFILE_COMPACT_FROM_EPISODE_TEXTS_USER_PROMPT, ("{{total_items}}", "{{profile_text}}")),
        (
            PROFILE_REGROUP_FROM_EPISODE_TEXTS_USER_PROMPT,
            ("{{bucket}}", "{{count}}", "{{label_field}}", "{{label}}", "{{items_text}}"),
        ),
    ],
    ids=("init", "update", "compact", "regroup"),
)
def test_episode_user_prompts_open_with_the_target_and_carry_only_data(
    prompt: str, placeholders: tuple[str, ...]
) -> None:
    assert prompt.startswith("<target_person>{{target_user}}</target_person>\n\n<")
    for placeholder in placeholders:
        assert prompt.count(placeholder) == 1
    for rule_block in _RULE_BLOCKS:
        assert rule_block not in prompt
    for rules_only in (
        "<reply_format>",
        "<how_to_update>",
        "<how_to_compact>",
        "<how_to_regroup>",
        "{{language_rule}}",
        "{{available_categories}}",
        "{{reply_skeleton}}",
    ):
        assert rules_only not in prompt


@pytest.mark.parametrize(("stage", "prompt"), list(zip(_STAGE_IDS, _SYSTEM_PROMPTS, strict=True)), ids=_STAGE_IDS)
def test_episode_system_prompts_delimit_every_block_with_one_ordered_xml_tag(stage: str, prompt: str) -> None:
    """Each block is opened once, closed once, in working order — and no CJK bracket is left."""
    expected = list(_SYSTEM_BLOCK_ORDER[stage])
    assert _top_level_blocks(prompt) == (expected, expected)
    for bracket in _RETIRED_BRACKETS:
        assert bracket not in prompt


@pytest.mark.parametrize(("stage", "prompt"), list(zip(_STAGE_IDS, _USER_PROMPTS, strict=True)), ids=_STAGE_IDS)
def test_episode_user_prompts_carry_only_tagged_data_blocks(stage: str, prompt: str) -> None:
    expected = list(_USER_BLOCK_ORDER[stage])
    assert _BLOCK_OPEN.findall(prompt) == expected
    assert _BLOCK_CLOSE.findall(prompt) == expected
    for bracket in _RETIRED_BRACKETS:
        assert bracket not in prompt
    assert _LEAVE_UNCERTAIN_OUT not in prompt


def test_rendered_episode_user_messages_parse_as_xml() -> None:
    """Wrapped in one root, every rendered user message is well-formed XML, so no tag is mis-nested."""
    for prompt in _USER_PROMPTS:
        rendered = render_prompt(prompt, None, **_RENDER_FIELDS)
        ElementTree.fromstring(f"<root>{rendered}</root>")


@pytest.mark.parametrize("name", _RETIRED_SECTION_NAMES)
def test_episode_prompts_name_blocks_by_tag_and_never_by_the_retired_section_name(name: str) -> None:
    for prompt in _SYSTEM_PROMPTS + _USER_PROMPTS:
        assert name not in prompt


_EPISODE_RULE_CLAUSES = (
    # Definition: what an assistant should know, past major facts included, not a log.
    "A profile is what an assistant should know about this person before the next conversation",
    "Major facts about their past",
    "It is not a log of what happened.",
    "A profile has two lists: explicit_info and implicit_traits.",
    # explicit_info: stated directly by a narrative, and each field's meaning and form.
    "explicit_info holds the facts about this person that a narrative states directly.",
    "A fact counts as stated directly only when a narrative explicitly attributes it to this person.",
    "narrative excerpts or faithful paraphrases, with their dates, in one string, one per occasion the fact was "
    "established or changed; a repeat of the same occasion adds nothing.",
    "A paraphrase keeps the narrative's attribution",
    "invents no wording, date, attribution or identifier",
    "one fact per item, in one or two short sentences without a subject",
    "named after the fact itself, not after the topic of the narrative it appeared in",
    "Prefer a category listed in <available_categories>, and when there is no list or none fits, coin a short",
    # implicit_traits: a pattern across narratives, grounded in named signals.
    "implicit_traits holds the stable patterns that no single narrative states but that show across several.",
    "A signal is a choice or request of theirs that a narrative records, not an operation they carried out",
    "Consulting the assistant is what every user does and is not a signal; what they chose or asked for, and how, "
    "can be.",
    "A fact a narrative states directly stays in explicit_info; neither that fact nor the occasions it was stated is "
    "a signal, and a trait resting on one is that fact restated.",
    "if you cannot name the signals the trait does not belong here",
    "Every signal in a basis is something a narrative shown records, in its own terms; a basis invents no signal, "
    "wording or date to reach a count.",
    "description is what the pattern is: one pattern per item, in one or two short sentences without a subject.",
    # Exclusions — one list, one home.
    "<never_in_the_profile>",
    "Having run the tests once is not being responsible for testing.",
    "A passing state, anything trivial, and what this person is working on right now",
    "A rule that would hold for whoever sat in their seat",
    "a decision taken once for one project",
    "being told something is not knowing it",
    "A fact about anyone other than the target person, even one the target person told",
    "and even inside an item about the relationship.",
    '"My brother is an accountant in Shanghai" gives nothing: the accounting and Shanghai are his, and "Has a brother" '
    "on its own is an empty item.",
    "A relationship enters only as a standing feature of this person's own life — a partner, someone they live with "
    "— and then with nothing about the other person.",
    "A one-off request about format or procedure",
    "When attribution, durability or grounding is uncertain, leave the item out.",
    # Reply format: the schema fixes the keys, not their order.
    "Reply with one JSON object, nothing else. Write each item's fields in the order shown, evidence or basis first.",
)

#: Removed on purpose. _EPISODE_RULE_CLAUSES pins what is present and cannot notice a deleted rule
#: coming back; each of these needs its own guard.
_EPISODE_FORBIDDEN_CLAUSES = (
    # The MemCell trait threshold, which this path used to import and then override in a later block.
    "one clear signal is enough",
    "should not leave this list empty",
    "overrides any general guidance",
    # A meta-instruction ranking rules against each other; it introduced no rule of its own.
    "factual correctness and category accuracy are co-equal",
    "They outrank recall, category count, category reuse",
    # Category guidance that pulled against "prefer a listed category".
    "Never sacrifice category accuracy merely to reuse a listed category",
    "The list is not a whitelist",
    # The first gate's carve-out, which readmitted rules phrased as demands.
    "what may enter is its personal side",
    "they set this rule for you to follow",
    # <what_the_profile_holds> rewrite (2026-09-28).
    "is attributable to the target person",
    "is clearly supported by the narratives",
    "wants every reply to end with a given word",
    "implicit_traits holds inferences",
    "what someone stated needs no inference",
    "A requirement on the work is not discarded, though",
    "Nothing enters it because it occurred",
    "An item belongs in the profile only when",
    "A profile says who this person is",
    "A profile is a **portrait of a person**",
    "It holds standing properties of this person, not a log of what happened.",
    "their identity, what they know and can do",
    "That something happened is not, by itself, a reason to record it.",
    "is still true the next time you meet this person and is clearly supported by its grounding",
    # Prohibitions restated as what to do (2026-09-28).
    "do not generate a trait because",
    "Never chain separate facts",
    "Do not broaden a category",
    "do not change a fact's meaning",
    "Never name the subject",
    "it never widens an item",
    "do not add its earlier value",
    "do not leave a copy behind",
    "never collapse a dimension's items",
    "never changes its claim",
    "Renaming never rewrites",
    "do not assume anything about it",
    "Never change an item's bucket",
    "never invent content",
    "delete those on sight",
    "do NOT merge",
    "still leaves a claim this person never made",
    "Deleting is NOT a way to meet the cap",
    # <what_the_profile_holds> rewrite (2026-10-08): the restatement test, the bucket-by-speaker rule and the
    # two-signal gate are gone; the definition no longer calls a profile a portrait.
    "A profile is a portrait of who this person is",
    "A conversation, however long, can correctly yield nothing at all.",
    "restating it with this person as the subject",
    "this is the restatement test",
    "A requirement placed on the WORK fails the test",
    "It does not enter explicit_info, however firmly or personally it was stated.",
    "what this person **stated** about themselves",
    "One clear statement is enough.",
    "at least two mutually independent, consistent signals from different narratives",
    "A stored explicit_info item counts as one such signal",
    "add a trait only when this gate is met",
    "become ONE implicit_traits item grounded in them",
    "The bucket is decided by whether this person said it",
    # An earlier value said only what it is not; it now names where it goes (2026-10-08, round 2).
    "is not a new fact",
    # A fixed count of excerpts left no room for an earlier value; evidence is one segment per occasion (round 2).
    "at most two",
    "; a paraphrase keeps",
    # Listing another person's attributes read as a closed list; a relationship item still carried them (round 3).
    "another person's job, skills, habits or whereabouts",
    # "Acts on the SAME fact" did not stop an update replacing the item's fact, and "however close the topic"
    # pushed detail on one subject into sibling items; an update now keeps its item on one subject (round 3).
    "Update acts on the SAME fact only",
    "however close the topic",
    "never changes which fact the item is about",
    # A trait description did not say it holds one pattern; models merged two patterns into one trait (round 3).
    "description is what the pattern is, one or two short sentences",
    # <never_in_the_profile> slimmed (2026-10-08).
    "Whatever the wording, whatever evidence is attached:",
    "An action, or an action restated as a capability, duty, skill, familiarity, interest or attitude.",
    "Do not call this person responsible for",
    "A passing state or an expiring plan.",
    "Facts about other participants belong to them.",
    "An instruction to a tool or an assistant",
    "This is about the claim, not the wording, and holds in every language.",
    "An item already on file",
    "never improve recall by guessing",
    # <how_an_item_is_written> folded into <what_the_profile_holds> (2026-09-29).
    "One fact, one item.",
    "A category groups items; it never merges them.",
    "Write descriptions without a subject.",
    '"evidence" and "basis" are JSON strings, never arrays',
    "use a direct user quotation only where the source itself contains that quotation",
    # The numbered steps and the category step (2026-10-08: requirements, not an order).
    "List the candidate facts",
    "Apply the restatement test",
    "Decide the bucket",
    "for every explicit_info item you add or update",
    "first look for a listed category in <available_categories> that holds this fact's dimension",
    "The list does not constrain implicit_traits.trait",
    "Compare with the profile in <stored_profile> and choose the operation",
    "Move anything misfiled.",
    "Keep every item in its bucket.",
    "Splitting is not a way around the caps",
    "they are ceilings, not targets",
    "A conversation made only of operations",
    # The older-batch rule (2026-10-08: folded into <how_to_update>, the historical system message is gone).
    "Every narrative in this batch is older than everything on file.",
    "A dimension that already holds a value was settled by a later narrative",
    # The UPDATE reply format's index sentence and brace rule (2026-10-08).
    "Every item in <stored_profile> carries an index",
    "operations within one response never shift each other's indices",
    "Every operation object closes with exactly two braces",
    # The worked example (2026-09-29).
    "the target person here is called Noor",
    "The Rust detail extends the SAME fact",
    "Works at a healthcare software company.",
    # The note fields (2026-10-08): the code never read them.
    "update_note",
    "compact_note",
    "regroup_note",
    "narratives contain nothing about this person",
    # "on file" became "stored", the word the <stored_profile> tag uses (2026-10-08).
    "on file",
    # <additional_requirements> lead (2026-10-08): the tag says what the block is.
    "The caller's additional requirements.",
    # Round 5 traits leaned on stated facts; forbidding a rephrasing did not forbid using the fact as a signal.
    "rephrasing it as a disposition does not make it a trait",
    # "Same subject" read as any new content on the topic: a goal and a current task were merged into items
    # about a job and a hobby (round 5); an update now keeps its item on one fact.
    "keeps the item on the same subject",
    "Only something about a different subject is an add.",
    # Example 2 taught relationship items that hold nothing but a name; example 3 taught merging a second
    # dimension into an item (round 5).
    "Has a cousin.",
    # The exclusion's own example taught the same bare item (round 6).
    'gives "Has a brother"',
    "ceramics",
)


@pytest.mark.parametrize("clause", _EPISODE_RULE_CLAUSES)
def test_episode_prompts_carry_each_rule(clause: str) -> None:
    for prompt in _SYSTEM_PROMPTS:
        assert clause in prompt


@pytest.mark.parametrize("clause", _EPISODE_FORBIDDEN_CLAUSES)
async def test_episode_prompts_do_not_readmit_removed_rules(clause: str) -> None:
    for prompt in _SYSTEM_PROMPTS + _USER_PROMPTS:
        assert clause not in prompt
    for stage, (system, user) in (await _messages_by_stage(_ADDITIONAL_REQUIREMENTS)).items():
        assert clause not in system, stage
        assert clause not in user, stage


@pytest.mark.parametrize("prompt", _SYSTEM_PROMPTS, ids=_STAGE_IDS)
def test_episode_system_prompts_carry_no_markdown(prompt: str) -> None:
    """One delimiter format per prompt: tags for structure, plain sentences inside; the exclusion list is the one list."""
    rendered = render_prompt(prompt, None, **{**_RENDER_FIELDS, "available_categories": "[]"})

    assert "**" not in rendered
    assert "```" not in rendered
    assert re.search(r"^#", rendered, re.MULTILINE) is None
    assert re.search(r"^\s*\d+\. ", rendered, re.MULTILINE) is None
    list_lines = re.findall(r"^\s*- .*$", rendered, re.MULTILINE)
    assert list_lines == _NEVER_IN_THE_PROFILE_LIST.splitlines()[1:]


def _line_start_json_blocks(text: str) -> list[str]:
    """Every brace-balanced block that opens at the start of a line — the reply skeletons."""
    blocks: list[str] = []
    depth, start = 0, -1
    for i, ch in enumerate(text):
        if ch == "{" and depth == 0 and (i == 0 or text[i - 1] == "\n"):
            start = i
        if start < 0:
            continue
        depth += (ch == "{") - (ch == "}")
        if depth == 0:
            blocks.append(text[start : i + 1])
            start = -1
    return blocks


def test_every_stage_renders_with_no_placeholder_left_behind() -> None:
    """A rendered message carries no ``{{placeholder}}`` — the templates and the render fields agree.

    A template and its call site can drift apart — a renamed slot, a field dropped from the call —
    and the visible symptom would be a literal "{{stored_profile}}" reaching the model. Rendering
    every stage against the full field set is what makes that drift fail here instead: a slot with
    no field raises out of ``render_prompt``, and one that survives substitution is caught below.
    """
    for system, user in zip(_SYSTEM_PROMPTS, _USER_PROMPTS, strict=True):
        for message in (render_prompt(system, None, **_RENDER_FIELDS), render_prompt(user, None, **_RENDER_FIELDS)):
            assert re.search(r"\{\{[A-Za-z_][A-Za-z0-9_]*\}\}", message) is None, message


def test_system_prompts_carry_no_stray_angle_brackets() -> None:
    """Every <...> in a system prompt is a real block tag or a tag the rules reference by name.

    A literal placeholder like "<Name>" would read as an opening tag in a prompt that now
    delimits with tags, and would leave the message unparseable as XML.
    """
    known = {
        "what_the_profile_holds",
        "never_in_the_profile",
        "how_to_update",
        "how_to_compact",
        "how_to_regroup",
        "reply_format",
        "output_language",
        "available_categories",
        "target_person",
        "stored_profile",
        "episode_narratives",
        "group",
        "examples",
        "example",
        "narrative",
        "reply",
        "explicit_info",
        "implicit_traits",
        "item",
    }
    for prompt in _SYSTEM_PROMPTS:
        for tag in re.findall(r"</?([A-Za-z_][A-Za-z0-9_]*)", prompt):
            assert tag in known, tag


def test_episode_prompts_demand_one_bare_json_object_and_every_shown_json_parses() -> None:
    """One sentence states the format; the skeletons prove it by being valid JSON.

    The old skeletons sat inside ```json fences right after "nothing else", which both licensed
    a fenced reply and contradicted the sentence before it.
    """
    for system, user in zip(_SYSTEM_PROMPTS, _USER_PROMPTS, strict=True):
        assert system.count(_JSON_OUTPUT_RULE) == 1
        assert _JSON_OUTPUT_RULE not in user
        for prompt in (system, user):
            assert "```" not in prompt
            assert "Output JSON directly" not in prompt
        blocks = _line_start_json_blocks(render_prompt(system, None, **_RENDER_FIELDS))
        assert blocks, "no JSON shape shown to the model"
        for block in blocks:
            json.loads(block)
        assert not _line_start_json_blocks(render_prompt(user, None, **_RENDER_FIELDS))


@pytest.mark.parametrize(
    ("bucket", "item_model"), [("explicit_info", _ExplicitItem), ("implicit_traits", _ImplicitTrait)]
)
def test_regroup_reply_skeleton_shows_the_group_bucket_item_shape(bucket: str, item_model: type[Any]) -> None:
    """A group's items come back in their bucket's shape, so the skeleton shows exactly that shape, in field order."""
    (item,) = json.loads(REGROUP_REPLY_SKELETONS[bucket])["items"]

    assert list(item) == list(item_model.model_fields)


@pytest.mark.parametrize("prompt", _SYSTEM_PROMPTS, ids=_STAGE_IDS)
def test_every_item_skeleton_lists_grounding_first_and_label_last(prompt: str) -> None:
    for bucket in ("explicit_info", "implicit_traits"):
        system = render_prompt(prompt, None, **{**_RENDER_FIELDS, "reply_skeleton": REGROUP_REPLY_SKELETONS[bucket]})
        for block in _line_start_json_blocks(system):
            for item in _items_in(json.loads(block)):
                assert list(item) in (list(_ExplicitItem.model_fields), list(_ImplicitTrait.model_fields)), item


def _items_in(reply: dict[str, Any]) -> list[dict[str, Any]]:
    """The item objects a reply skeleton shows: bucket lists, regroup items, operation data."""
    items: list[dict[str, Any]] = [*reply.get("explicit_info", []), *reply.get("implicit_traits", [])]
    items += reply.get("items", [])
    items += [op["data"] for op in reply.get("operations", []) if "data" in op]
    return items


def test_output_item_models_declare_grounding_first() -> None:
    """The strict schema lists properties in declaration order: evidence before the claim it supports."""
    assert list(_ExplicitItem.model_fields) == ["evidence", "description", "category"]
    assert list(_ImplicitTrait.model_fields) == ["basis", "description", "trait"]


def test_how_to_update_holds_six_unnumbered_requirements() -> None:
    paragraphs = _HOW_TO_UPDATE.split("\n\n")

    assert len(paragraphs) == 6
    assert paragraphs[0].startswith("add or update is decided by the fact, not the wording.")
    assert paragraphs[-1].startswith("Newer state stands; older narratives only add to it.")
    assert paragraphs[-1].endswith("leave observed_at to the system.")
    # An update names the fields it restates; "every field" would also cover the observed_at stored items show.
    assert "An update restates the whole item — its category or trait, its description and its evidence or basis" in (
        _HOW_TO_UPDATE
    )
    assert "every field" not in _HOW_TO_UPDATE
    # An update keeps its item on one fact; another dimension is an add and an excluded claim goes nowhere (round 5:
    # "same subject" let a goal and a current task ride into the items about a job and a hobby).
    assert paragraphs[1].endswith(
        "An update keeps the item on the same fact: a correction, a sharper wording or a changed value of that fact "
        "goes into the item. A fact of another dimension — a goal beside a job, a habit beside a hobby — is an add, "
        "and what <never_in_the_profile> excludes goes nowhere."
    )
    # The output language reaches stored items no narrative touches: UPDATE replies with operations only, so an
    # untouched item in another language would otherwise bypass the model.
    assert paragraphs[2] == (
        "The profile is written in one language, the one <output_language> gives. A stored item written in any other "
        "language is updated: its category or trait, its description and its evidence or basis rewritten in that "
        "language and nothing else changed, even when no narrative mentions it."
    )
    # The older-batch rule, generalised: an earlier value of a settled dimension is grounding for that item.
    # History may follow the current value in the description; the stale guard checks the current value stays whole.
    assert (
        'must not rewrite the item\'s description or "delete" it. An earlier value of a dimension that already holds '
        "one goes into that item's evidence, never into a new item; the description may add that history in a short "
        "parenthesis after the current value, but the current value stays as written. The system maintains"
    ) in _HOW_TO_UPDATE
    assert "never into a new item. The system" not in _HOW_TO_UPDATE
    assert "is not a new fact" not in _HOW_TO_UPDATE
    assert "Narratives made only of operations contain nothing about who this person is" in _HOW_TO_UPDATE


def test_how_to_compact_puts_the_one_ordering_that_matters_first() -> None:
    paragraphs = _HOW_TO_COMPACT.split("\n\n")

    assert len(paragraphs) == 7
    assert paragraphs[0] == (
        "Delete everything on the <never_in_the_profile> list before merging anything: twelve such items become zero "
        "items, not one."
    )


def test_how_to_regroup_holds_four_unnumbered_requirements() -> None:
    paragraphs = _HOW_TO_REGROUP.split("\n\n")

    assert len(paragraphs) == 4
    assert (
        paragraphs[-1]
        == "Keep every item in the bucket it came from; every claim and grounding comes from the shown items."
    )


@pytest.mark.parametrize(
    "prompt", _SYSTEM_PROMPTS + _USER_PROMPTS, ids=[*_STAGE_IDS, *(f"{s}-user" for s in _STAGE_IDS)]
)
def test_no_template_carries_a_fixed_person_name(prompt: str) -> None:
    """The worked examples have people of their own; no template stands in a name for the target person."""
    assert _OWNER_NAME not in prompt
    assert "Noor" not in prompt


def test_update_prompt_explains_times_and_forbids_older_rewrites() -> None:
    prompt = PROFILE_UPDATE_FROM_EPISODE_TEXTS_SYSTEM_PROMPT

    assert "Every stored item shows observed_at" in prompt
    assert "every narrative shows its own time" in prompt
    assert "A narrative whose time is before an item's observed_at" in prompt
    assert "leave observed_at to the system" in prompt
    assert "dated before" not in prompt
    assert "Older narratives never rewrite newer state" not in prompt
    assert "never write observed_at yourself" not in prompt
    for maintenance in (
        PROFILE_COMPACT_FROM_EPISODE_TEXTS_SYSTEM_PROMPT,
        PROFILE_COMPACT_FROM_EPISODE_TEXTS_USER_PROMPT,
        PROFILE_REGROUP_FROM_EPISODE_TEXTS_SYSTEM_PROMPT,
        PROFILE_REGROUP_FROM_EPISODE_TEXTS_USER_PROMPT,
    ):
        assert "observed_at" not in maintenance


def test_episode_prompts_do_not_encode_category_sources_or_stored_category_lifecycle() -> None:
    prompts = _SYSTEM_PROMPTS + _USER_PROMPTS
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


def test_update_reply_format_says_only_what_index_means() -> None:
    reply_format = _block(PROFILE_UPDATE_FROM_EPISODE_TEXTS_SYSTEM_PROMPT, "reply_format")

    assert reply_format.startswith(
        "In update and delete, index is the item's index attribute in <stored_profile>. "
        "Reply with one JSON object, nothing else. Write each item's fields in the order shown, evidence or basis "
        "first.\n\nNothing to do:\n{"
    )
    assert "\n\nOne or more operations:\n{" in reply_format
    expected_update = (
        '{"action": "update", "type": "explicit_info", "index": 0, '
        '"data": {"evidence": "...", "description": "...", "category": "..."}}'
    )
    assert expected_update in reply_format


def test_init_reply_format_allows_an_empty_profile() -> None:
    reply_format = _block(PROFILE_INITIAL_FROM_EPISODE_TEXTS_SYSTEM_PROMPT, "reply_format")

    assert reply_format.startswith(
        "Both lists may be empty when the narratives carry nothing about this person. "
        "Reply with one JSON object, nothing else. Write each item's fields in the order shown, evidence or basis "
        "first.\n\n"
    )


def test_episode_compact_corrects_bucket_without_inventing_grounding() -> None:
    """Full-profile compaction can rehome stored content but cannot manufacture it."""
    prompt = PROFILE_COMPACT_FROM_EPISODE_TEXTS_SYSTEM_PROMPT

    assert "Move a trait that merely restates a fact a narrative states directly into explicit_info" in prompt
    assert "every claim and grounding you keep comes from the items you were given" in prompt
    assert "do not leave a copy behind" not in prompt
    assert "NEVER by inventing a claim or grounding" not in prompt


def test_episode_regroup_preserves_bucket_and_grounding_contracts() -> None:
    """Single-bucket regroup cannot manufacture content or move an item to the unseen bucket."""
    prompt = PROFILE_REGROUP_FROM_EPISODE_TEXTS_SYSTEM_PROMPT

    assert "Keep every item in the bucket it came from" in prompt
    assert "every claim and grounding comes from the shown items" in prompt
    assert "Never change an item's bucket" not in prompt
    assert "never invent content" not in prompt


# ==========================================================================
# One home per rule — the stage blocks point at the block that states it
# ==========================================================================

#: The grounding form is stated in <what_the_profile_holds>; a stage that needs it names that block.
_GROUNDING_FORM = "one per occasion the fact was established or changed"
#: The canonical "evidence does not rescue an excluded claim" wording, in <never_in_the_profile> only.
_EVIDENCE_DOES_NOT_RESCUE = "whatever evidence"


def _block(system: str, tag: str) -> str:
    return system.split(f"\n<{tag}>\n", maxsplit=1)[1].split(f"\n</{tag}>", maxsplit=1)[0]


@pytest.mark.parametrize("system", _SYSTEM_PROMPTS, ids=_STAGE_IDS)
def test_grounding_form_is_stated_only_in_what_the_profile_holds(system: str) -> None:
    definition = _block(system, "what_the_profile_holds")

    assert definition.count(_GROUNDING_FORM) == 1
    assert _GROUNDING_FORM not in system.replace(definition, "")


@pytest.mark.parametrize("system", _SYSTEM_PROMPTS, ids=_STAGE_IDS)
def test_evidence_does_not_rescue_is_stated_only_in_never_in_the_profile(system: str) -> None:
    exclusions = _block(system, "never_in_the_profile")

    assert "Whatever the wording and whatever evidence is attached, these never become items:" in exclusions
    assert _EVIDENCE_DOES_NOT_RESCUE not in system.replace(exclusions, "")
    assert "whatever evidence they carry" not in system


@pytest.mark.parametrize(("system", "user"), list(zip(_SYSTEM_PROMPTS, _USER_PROMPTS, strict=True)), ids=_STAGE_IDS)
def test_every_tag_a_rule_names_is_given_to_the_model(system: str, user: str) -> None:
    """A rule that points at <x> is only followable if a block or element <x> opens in that request."""
    given = set(re.findall(r"^<([a-z_]+)[ >]", f"{system}\n{user}", re.MULTILINE))
    named = set(re.findall(r"<([a-z_]+)>", system))

    assert named <= given, named - given


def test_the_stage_blocks_point_at_the_blocks_that_state_the_rules() -> None:
    by_stage = dict(zip(_STAGE_IDS, _SYSTEM_PROMPTS, strict=True))
    for stage in ("compact", "regroup"):
        assert "in the form <what_the_profile_holds> gives" in _block(by_stage[stage], f"how_to_{stage}")
        assert "as <what_the_profile_holds> defines" in _block(by_stage[stage], f"how_to_{stage}")
    for stage in ("update", "compact", "regroup"):
        assert "the <never_in_the_profile> list" in _block(by_stage[stage], f"how_to_{stage}")
        assert "\n<never_in_the_profile>\n" in by_stage[stage]


async def _straddling_update_system_messages() -> tuple[str, str]:
    """The system messages of a batch straddling the stored timestamp: the older half's, then the newer half's."""
    fake = FakeLLMClient(responses=[_operations(), _operations()])
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
    return _system_prompt(fake, 0), _system_prompt(fake, 1)


async def test_both_update_passes_send_one_and_the_same_system_message() -> None:
    """The split is kept for the observation date; the older narratives get no rule of their own."""
    historical, current = await _straddling_update_system_messages()

    assert historical == current
    assert "rule_for_this_older_batch" not in historical
    assert "\n\n\n" not in historical


def test_no_episode_template_carries_a_slot_that_is_sometimes_empty() -> None:
    """The retired ``{{pass_rule}}`` slot rendered as nothing for the current batch; no template keeps one."""
    for prompt in _SYSTEM_PROMPTS + _USER_PROMPTS:
        assert "{{pass_rule}}" not in prompt


# ==========================================================================
# Worked examples — INIT shows two, UPDATE one, each in its stage's reply shape; COMPACT and REGROUP none
# ==========================================================================

_INIT_EXAMPLE_BODIES = (_EXAMPLE_OPERATIONS_ONLY, _EXAMPLE_FACTS_AND_A_PATTERN)
_UPDATE_EXAMPLE_BODIES = (_EXAMPLE_NAMED_LANGUAGE_SAME_SUBJECT,)
_REPLY = re.compile(r"\n<reply>\n(.*?)\n</reply>", re.DOTALL)


def _example_replies(examples: str) -> list[str]:
    return _REPLY.findall(examples)


def _utc_ms(text: str) -> int:
    """Epoch milliseconds of a ``YYYY-MM-DD HH:MM`` UTC minute."""
    return int(datetime.strptime(text, "%Y-%m-%d %H:%M").replace(tzinfo=UTC).timestamp() * 1000)


def test_examples_follow_never_in_the_profile_in_init_and_update_only() -> None:
    """Rules first, then the examples that show them at work, then the stage's own blocks."""
    init, update, compact, regroup = _SYSTEM_PROMPTS

    assert f"\n</never_in_the_profile>\n\n{_EXAMPLES_INIT}\n\n<reply_format>\n" in init
    assert f"\n</never_in_the_profile>\n\n{_EXAMPLES_UPDATE}\n\n<how_to_update>\n" in update
    for prompt in (compact, regroup):
        assert "<examples>" not in prompt
        assert "<example>" not in prompt


@pytest.mark.parametrize(
    ("examples", "own", "other", "lead"),
    [
        (
            _EXAMPLES_INIT,
            _INIT_EXAMPLE_BODIES,
            _UPDATE_EXAMPLE_BODIES,
            "Two worked examples. They show what enters the profile and what stays out; copy their judgement, not "
            "their wording.",
        ),
        (
            _EXAMPLES_UPDATE,
            _UPDATE_EXAMPLE_BODIES,
            _INIT_EXAMPLE_BODIES,
            "One worked example. It shows what enters the profile and what stays out; copy its judgement, not its "
            "wording.",
        ),
    ],
    ids=["init", "update"],
)
def test_each_stage_shows_only_its_own_examples_after_one_lead_sentence(
    examples: str, own: tuple[str, ...], other: tuple[str, ...], lead: str
) -> None:
    body = "\n\n".join(f"<example>\n{example}\n</example>" for example in own)

    assert examples == f"<examples>\n{lead}\n\n{body}\n</examples>"
    for example in other:
        assert example not in examples


@pytest.mark.parametrize(
    ("examples", "count", "callsite", "reply"),
    [
        (_EXAMPLES_INIT, 2, _call_llm_for_profile_init, _ProfileItems),
        (_EXAMPLES_UPDATE, 1, _call_llm_for_profile_update, _UpdateOutput),
    ],
    ids=["init", "update"],
)
async def test_every_example_reply_parses_and_passes_its_stage_reply_model(
    examples: str, count: int, callsite: Any, reply: type[Any]
) -> None:
    """Each example's reply is what the stage itself would accept: bare JSON its strict model and checks let through."""
    replies = _example_replies(examples)

    assert len(replies) == count
    for text in replies:
        assert isinstance(json.loads(text), dict)
        validated = await callsite(FakeLLMClient(responses=[text]), [], reply=reply)
        assert validated == json.loads(text)


def test_init_examples_keep_an_empty_profile_and_leave_out_what_never_enters() -> None:
    """Operations and a one-off request give nothing; another person's job, a one-off request and an action stay out.

    The cousin the narrative mentions leaves no item at all, not even a bare relationship; the target person's own
    household does enter, so the example cannot be read as keeping every relationship out.
    """
    operations_only, facts_and_a_pattern = (json.loads(text) for text in _example_replies(_EXAMPLES_INIT))

    assert operations_only == {"explicit_info": [], "implicit_traits": []}
    assert "her cousin, a lawyer, handles the paperwork" in _EXAMPLES_INIT
    by_category = {item["category"]: item["description"] for item in facts_and_a_pattern["explicit_info"]}
    assert by_category == {
        "Identity": "Runs a small bookshop in Lisbon.",
        "Goals": "Wants to open a second shop one day.",
        "Relationships": "Lives with her mother.",
    }
    written = json.dumps(facts_and_a_pattern)
    for left_out in ("cousin", "lawyer", "paperwork", "formal", "advertisement"):
        assert left_out not in written
    (trait,) = facts_and_a_pattern["implicit_traits"]
    assert trait["basis"].count("2026-") == 2  # The pattern names a signal from each of two narratives.


def test_update_example_writes_the_named_language_over_a_stored_one_and_updates_the_same_subject() -> None:
    (reply,) = (json.loads(text) for text in _example_replies(_EXAMPLES_UPDATE))

    (operation,) = reply["operations"]
    assert operation["action"] == "update"
    assert operation["index"] == 0
    assert operation["data"]["category"] == "Identity"
    assert re.search(r"[\u4e00-\u9fff]", json.dumps(operation, ensure_ascii=False)) is None
    assert "caption" not in json.dumps(operation)
    assert _EXAMPLES_UPDATE.count("<output_language>") == 1
    # The new detail is a sharper statement of the stored fact, so the item stays one fact.
    description = operation["data"]["description"]
    assert "flower shop in Kyoto" in description
    assert "Gion" in description
    assert ";" not in description
    assert " also " not in description


@pytest.mark.parametrize(
    ("stage", "bodies"), [("init", _INIT_EXAMPLE_BODIES), ("update", _UPDATE_EXAMPLE_BODIES)], ids=["init", "update"]
)
def test_example_inputs_open_the_blocks_the_stage_user_message_opens(stage: str, bodies: tuple[str, ...]) -> None:
    """An example's input is marked up as the real input is: the same outer tags, in the same order, then its reply."""
    for body in bodies:
        opened, _ = _top_level_blocks(body)
        assert opened == [*_USER_BLOCK_ORDER[stage], "reply"]


def test_examples_show_their_input_as_the_production_renderer_does() -> None:
    """Narratives and the stored profile in an example are what the user message would carry for the same input."""
    stored = Profile.model_validate(
        {
            "owner_id": _OWNER_ID,
            "summary": "",
            "timestamp": _utc_ms("2025-11-03 10:00"),
            "explicit_info": [
                {
                    "evidence": "2025-11-03\uff1a说自己在京都经营一家花店",
                    "description": "在京都经营一家花店\u3002",
                    "category": "身份",
                    "observed_at": _utc_ms("2025-11-03 10:00"),
                }
            ],
            "implicit_traits": [],
        }
    )

    assert f"<stored_profile>\n{_render_profile_for_episode_update(stored)}\n</stored_profile>" in _EXAMPLES_UPDATE
    for examples in (_EXAMPLES_INIT, _EXAMPLES_UPDATE):
        for block in re.findall(r"\n<episode_narratives>\n(.*?)\n</episode_narratives>", examples, re.DOTALL):
            pairs = re.findall(r'^<narrative time="([^"]+) UTC">(.*)</narrative>$', block, re.MULTILINE)
            assert len(pairs) == block.count("\n") + 1
            episodes = [_episode(text, timestamp=_utc_ms(time)) for time, text in pairs]
            assert _render_episodes(episodes) == block


# ==========================================================================
# additional_requirements — the host's requirements, appended after every bundled block
# ==========================================================================

#: Braces included: the text is sent as written, never rendered as a template.
_ADDITIONAL_REQUIREMENTS = "Write every description in the third person. Keep {{braces}} and {single} literal."
_ADDITIONAL_REQUIREMENTS_LEAD = (
    "They apply on top of the rules above; the reply shape in <reply_format> stays as it is. Where they conflict "
    "with the rules above, the rules above hold."
)
_ADDITIONAL_REQUIREMENTS_BLOCK = (
    f"\n\n<additional_requirements>\n{_ADDITIONAL_REQUIREMENTS_LEAD}\n"
    f"{_ADDITIONAL_REQUIREMENTS}\n</additional_requirements>"
)
_ALL_STAGES = ("init", "update", "update-historical", "compact", "regroup")


async def _messages_by_stage(
    custom_instructions: str | None, *, output_language: OutputLanguage | None = None
) -> dict[str, tuple[str, str]]:
    """Drive every Episode-path stage once and return each stage's ``(system, user)`` messages.

    Calls, in order: INIT; an UPDATE straddling the stored timestamp (historical, then current batch); an
    UPDATE that breaches the total cap, so COMPACT follows; an UPDATE that breaches a group cap, so REGROUP follows.
    """
    regroup = json.dumps(
        {"items": [{"evidence": "Alice selected Ruff.", "description": "Uses Ruff.", "category": "Environment"}]}
    )
    fake = FakeLLMClient(
        responses=[
            _profile_payload(),
            _operations(),
            _operations(),
            _operations(_add("Prefers Ruff.", category="new", evidence="Alice chose Ruff.")),
            _compact_payload(),
            _operations(_add("Uses Ruff.", category="Environment", evidence="Alice selected Ruff.")),
            regroup,
        ]
    )
    extractor = ProfileExtractor(llm=fake)
    crowded = [
        {"category": f"dimension-{index}", "description": f"Fact {index}.", "evidence": "E.", "observed_at": _EARLIER}
        for index in range(60)
    ]
    grouped = [
        {"category": "Environment", "description": f"Fact {index}.", "evidence": "E.", "observed_at": _EARLIER}
        for index in range(8)
    ]
    runs: list[tuple[list[Episode], Profile | None]] = [
        ([_episode("Alice selected Python.")], None),
        ([_episode("Alice studied.", timestamp=_EARLIER), _episode("Alice moved.", timestamp=_LATER)], _old_profile()),
        ([_episode("Alice chose Ruff.")], _old_profile(explicit_info=crowded)),
        ([_episode("Alice selected Ruff.")], _old_profile(explicit_info=grouped)),
    ]
    for episodes, old_profile in runs:
        await extractor.aextract_from_episodes(
            episodes,
            owner_id=_OWNER_ID,
            owner_name=_OWNER_NAME,
            old_profile=old_profile,
            categories=_AVAILABLE_CATEGORIES,
            custom_instructions=custom_instructions,
            output_language=output_language,
        )
    assert fake.call_count == 7
    call_of_stage = {"init": 0, "update-historical": 1, "update": 2, "compact": 4, "regroup": 6}
    return {stage: _system_and_user(fake, call) for stage, call in call_of_stage.items()}


@pytest.mark.parametrize("stage", _ALL_STAGES)
async def test_a_named_language_holds_for_the_stored_items_too_in_every_stage(stage: str) -> None:
    """Every Episode stage, those that show stored items included, says the named language covers what is stored."""
    system, user = (await _messages_by_stage(None, output_language=OutputLanguage.GERMAN))[stage]

    assert (
        "<output_language>\nWrite every output field in German, whatever language the input is in, including anything already stored. "
        in system
    )
    assert "German, whatever language the input is in." not in system
    assert "German" not in user


@pytest.mark.parametrize("stage", _ALL_STAGES)
async def test_additional_requirements_close_every_stage_system_message_after_the_categories(stage: str) -> None:
    """One block, last, after <available_categories>, the lead-in on its first line and the text verbatim."""
    system, user = (await _messages_by_stage(_ADDITIONAL_REQUIREMENTS))[stage]

    assert system.endswith(f"\n</available_categories>{_ADDITIONAL_REQUIREMENTS_BLOCK}")
    assert system.count("<additional_requirements>") == 1
    assert _top_level_blocks(system)[0] == [*_SYSTEM_BLOCK_ORDER[stage], "additional_requirements"]
    assert _ADDITIONAL_REQUIREMENTS not in user


@pytest.mark.parametrize("stage", _ALL_STAGES)
async def test_additional_requirements_add_to_the_bundled_blocks_and_replace_none(stage: str) -> None:
    """With requirements, every stage sends exactly what it sends without them, plus the closing block."""
    with_requirements = (await _messages_by_stage(_ADDITIONAL_REQUIREMENTS))[stage]
    without = (await _messages_by_stage(None))[stage]

    assert with_requirements[0] == without[0] + _ADDITIONAL_REQUIREMENTS_BLOCK
    assert with_requirements[1] == without[1]


@pytest.mark.parametrize("custom_instructions", [None, "", "  \n\t "], ids=["none", "empty", "blank"])
async def test_no_additional_requirements_add_no_block_to_any_stage(custom_instructions: str | None) -> None:
    for system, user in (await _messages_by_stage(custom_instructions)).values():
        assert "additional_requirements" not in system + user
        assert _ADDITIONAL_REQUIREMENTS_LEAD not in system + user
        assert system.endswith("\n</available_categories>")


def test_additional_requirements_block_puts_the_lead_in_on_its_first_line() -> None:
    lines = additional_requirements_block("Line one.\nLine two.").splitlines()

    assert lines == [
        "<additional_requirements>",
        _ADDITIONAL_REQUIREMENTS_LEAD,
        "Line one.",
        "Line two.",
        "</additional_requirements>",
    ]


def test_additional_requirements_yield_to_the_bundled_rules() -> None:
    """They may tighten the rules, never loosen them: where they conflict, the bundled rules hold."""
    assert _ADDITIONAL_REQUIREMENTS_LEAD.endswith("Where they conflict with the rules above, the rules above hold.")
    assert "The caller's additional requirements." not in additional_requirements_block("x")


async def test_the_parameter_is_custom_instructions_and_its_text_still_reaches_the_model_as_the_block() -> None:
    """The interface says custom_instructions; the model still reads an <additional_requirements> block."""
    for method in ("aextract", "aextract_from_episodes"):
        parameters = inspect.signature(getattr(ProfileExtractor, method)).parameters
        assert parameters["custom_instructions"].kind is inspect.Parameter.KEYWORD_ONLY, method
        assert parameters["custom_instructions"].default is None, method
        for retired in ("additional_requirements", "custom_prompt", "prompt"):
            assert retired not in parameters, (method, retired)

    for system, user in (await _messages_by_stage(_ADDITIONAL_REQUIREMENTS)).values():
        assert system.endswith(_ADDITIONAL_REQUIREMENTS_BLOCK)
        assert "custom_instructions" not in system + user


@pytest.mark.parametrize("stage", _ALL_STAGES)
async def test_every_stage_opens_with_one_role_stated_directly(stage: str) -> None:
    """One role for every stage, in short sentences: role, task, where each input is, constraints, the rules."""
    system, _ = (await _messages_by_stage(None))[stage]
    role = system.split("\n\n", maxsplit=1)[0]
    sentences = re.split(r"(?<=\.) (?=[A-Z])", role)

    assert system.startswith(_ROLE_PREFIX)
    assert "The target person is named in <target_person>." in role
    assert all(sentence.endswith(".") for sentence in sentences)
    assert sentences[-1] == "Follow the rules below."
    assert "which" not in role
    assert ", the one named in" not in role
    assert "updater" not in system
    assert "editor" not in system
    assert "The user message gives" not in system


@pytest.mark.parametrize(
    ("stage", "task"),
    [
        ("init", "Build the profile of the target person from the Episode narratives."),
        ("update", "Update the stored profile of the target person from the Episode narratives."),
    ],
)
async def test_the_reading_stages_name_their_input_and_action_in_the_second_sentence(stage: str, task: str) -> None:
    system, _ = (await _messages_by_stage(None))[stage]

    assert system.startswith(f"{_ROLE_PREFIX}{task} ")


async def test_update_role_states_the_output_form_third() -> None:
    system, _ = (await _messages_by_stage(None))["update"]
    sentences = re.split(r"(?<=\.) (?=[A-Z])", system.split("\n\n", maxsplit=1)[0])

    assert sentences[2] == "Express the update as operations on the stored profile."
    assert "Update the profile of the target person." not in system


@pytest.mark.parametrize("stage", ["update", "update-historical"])
async def test_update_role_does_not_call_the_narratives_new(stage: str) -> None:
    """The narratives are what the stored profile has not absorbed yet; a historical batch is older than it."""
    system, _ = (await _messages_by_stage(None))[stage]

    assert "the stored profile does not reflect them yet, and they may be older than it" in system
    assert "new Episode narratives" not in system


@pytest.mark.parametrize(
    ("prompt", "slots"),
    [
        (PROFILE_COMPACT_FROM_EPISODE_TEXTS_SYSTEM_PROMPT, ("max_items", "max_per_category")),
        (PROFILE_REGROUP_FROM_EPISODE_TEXTS_SYSTEM_PROMPT, ("max_per_category",)),
    ],
    ids=["compact", "regroup"],
)
def test_cap_numbers_are_stated_in_the_reply_format_block_not_the_role(prompt: str, slots: tuple[str, ...]) -> None:
    """The role only says the profile or group is over its caps; the numbers to meet sit in <reply_format>."""
    caps = {"max_items": 97, "max_per_category": 13}  # distinct from every other number in the prompt
    system = render_prompt(prompt, None, **{**_RENDER_FIELDS, **caps})
    role = system.split("\n\n", maxsplit=1)[0]
    output = system.split("\n<reply_format>\n", maxsplit=1)[1].split("\n</reply_format>", maxsplit=1)[0]

    for slot in slots:
        assert str(caps[slot]) in output
        assert str(caps[slot]) not in role


# ==========================================================================
# Input markup — narratives and stored items are XML elements; nothing describes how to read them
# ==========================================================================

#: The retired <how_to_read_the_narratives> text (BOSS 2026-09-28, deleted 2026-10-08): the markup now says it.
_RETIRED_NARRATIVE_READING = (
    "<episode_narratives> holds one or more Episode narratives.",
    "Each narrative is a third-person retelling of one conversation, not the raw conversation turns.",
    "Narratives are separated by a line holding only ---.",
    "Each narrative opens with the time of that conversation on its own line",
    "[YYYY-MM-DD HH:MM UTC]",
    "Narratives are given oldest first.",
    "In the narratives, the target person appears under the name in <target_person>.",
    # Dropped from that block on earlier revisions (2026-09-28).
    "Each narrative retells one conversation.",
    'Where the rules below say "conversation", they mean these narratives.',
    "A quotation mark is not required.",
    "A narrative refers to the target person by the name in <target_person>; other people may appear as well.",
)
#: Names the blocks carried before. A rule still naming one would point the model at nothing.
_RETIRED_TAG_NAMES = (
    "source_records",
    "portrait_definition",
    "never_a_portrait_item",
    "<output>",
    "</output>",
    "historical_batch",
    "custom_prompt",
    "how_to_read_the_narratives",
    "work_in_this_order",
    "how_an_item_is_written",
    "rule_for_this_older_batch",
)
_PROFILE_DEFINITION = (
    "A profile is what an assistant should know about this person before the next conversation: who they are, "
)


@pytest.mark.parametrize("retired", _RETIRED_TAG_NAMES)
async def test_no_message_carries_a_retired_tag_name(retired: str) -> None:
    """All five system messages, rendered with additional requirements so that block is present too."""
    for stage, (system, user) in (await _messages_by_stage(_ADDITIONAL_REQUIREMENTS)).items():
        assert retired not in system, stage
        assert retired not in user, stage


@pytest.mark.parametrize("sentence", _RETIRED_NARRATIVE_READING)
async def test_the_retired_narrative_reading_is_in_no_message(sentence: str) -> None:
    for stage, (system, user) in (await _messages_by_stage(_ADDITIONAL_REQUIREMENTS)).items():
        assert sentence not in system, stage
        assert sentence not in user, stage


@pytest.mark.parametrize("stage", _ALL_STAGES)
async def test_the_definition_follows_the_role_in_every_stage(stage: str) -> None:
    system, _ = (await _messages_by_stage(None))[stage]
    blocks = system.split("\n\n<", maxsplit=1)

    assert blocks[0].startswith(_ROLE_PREFIX)
    assert blocks[1].startswith(f"what_the_profile_holds>\n{_PROFILE_DEFINITION}")


@pytest.mark.parametrize("stage", _ALL_STAGES)
async def test_user_message_names_the_target_person_under_its_tag(stage: str) -> None:
    _, user = (await _messages_by_stage(None))[stage]

    assert user.startswith(f"<target_person>{_OWNER_NAME}</target_person>\n\n<")
    assert "<target>" not in user
    assert "</target>" not in user


@pytest.mark.parametrize("stage", _ALL_STAGES)
async def test_the_definition_is_stated_once_and_portrait_is_retired(stage: str) -> None:
    """One word for the thing being built: a profile, defined once; the old "portrait" wording is gone."""
    system, user = (await _messages_by_stage(_ADDITIONAL_REQUIREMENTS))[stage]

    assert system.count(_PROFILE_DEFINITION) == 1
    assert "portrait" not in system.casefold()
    assert "portrait" not in user.casefold()


async def test_regroup_user_message_wraps_the_items_in_the_group_element() -> None:
    """The group's attributes, its bucket first, and its items are one element; the separate <items> block is gone."""
    _, user = (await _messages_by_stage(None))["regroup"]
    (group,) = ElementTree.fromstring(f"<root>{user}</root>").iter("group")

    assert list(group.attrib.items()) == [
        ("bucket", "explicit_info"),
        ("label_field", "category"),
        ("label", "Environment"),
        ("count", "9"),
    ]
    assert group.text is not None
    assert [item["description"] for item in json.loads(group.text)][:2] == ["Fact 0.", "Fact 1."]
    assert "items>" not in user


async def _regroup_messages(bucket: str, label: str) -> tuple[str, str]:
    """The REGROUP messages for one over-cap group of ``bucket``."""
    label_field, grounding = ("category", "evidence") if bucket == "explicit_info" else ("trait", "basis")
    crowded = [
        {label_field: label, "description": f"Fact {index}.", grounding: "E.", "observed_at": _EARLIER}
        for index in range(9)
    ]
    old = Profile.model_validate(
        {
            "owner_id": _OWNER_ID,
            "summary": "s",
            "timestamp": _TIMESTAMP - 1,
            "explicit_info": crowded if bucket == "explicit_info" else [],
            "implicit_traits": crowded if bucket == "implicit_traits" else [],
        }
    )
    fake = FakeLLMClient(responses=[_operations({"action": "none"}), json.dumps({"items": []})])
    await ProfileExtractor(llm=fake).aextract_from_episodes(
        [_episode("Alice selected Ruff.")], owner_id=_OWNER_ID, owner_name=_OWNER_NAME, old_profile=old
    )
    return _system_and_user(fake, 1)


@pytest.mark.parametrize("bucket", ["explicit_info", "implicit_traits"])
async def test_regroup_system_shows_the_group_bucket_skeleton_and_names_the_bucket(bucket: str) -> None:
    system, user = await _regroup_messages(bucket, "Crowd")
    other = "implicit_traits" if bucket == "explicit_info" else "explicit_info"

    assert _block(system, "reply_format").endswith(f"\n\n{REGROUP_REPLY_SKELETONS[bucket]}")
    assert REGROUP_REPLY_SKELETONS[other] not in system
    assert f'<group bucket="{bucket}" ' in user


async def test_regroup_system_is_the_same_for_every_group_of_a_bucket() -> None:
    """A per-bucket skeleton keeps the system message cacheable: two groups of one bucket send the same text."""
    first, _ = await _regroup_messages("explicit_info", "Environment")
    second, _ = await _regroup_messages("explicit_info", "Hobbies")
    traits, _ = await _regroup_messages("implicit_traits", "Environment")

    assert first == second
    assert first != traits
    assert first.split("\n<reply_format>\n", maxsplit=1)[0] == traits.split("\n<reply_format>\n", maxsplit=1)[0]


async def test_stored_profile_renders_each_bucket_as_an_element_of_indexed_items() -> None:
    """The index every op names is the item's attribute; no ``=== bucket ===`` header and no ``[0]`` prefix."""
    trait = {"trait": "Methodical", "description": "Decides methodically.", "basis": "B.", "observed_at": _TIMESTAMP}
    old = Profile.model_validate(
        {
            "owner_id": _OWNER_ID,
            "summary": "s",
            "timestamp": _TIMESTAMP,
            "explicit_info": [
                {"category": "Stack", "description": "Uses Python.", "evidence": "E1.", "observed_at": _TIMESTAMP},
                {"category": "Home", "description": "Lives in Porto.", "evidence": "E2.", "observed_at": _TIMESTAMP},
            ],
            "implicit_traits": [trait],
        }
    )
    fake = FakeLLMClient(responses=[_operations({"action": "none"})])
    await ProfileExtractor(llm=fake).aextract_from_episodes(
        [_episode("Alice adopted uv.", timestamp=_LATER)], owner_id=_OWNER_ID, owner_name=_OWNER_NAME, old_profile=old
    )

    user = _user_prompt(fake)
    when = format_utc_time(_TIMESTAMP)
    stored = user.split("<stored_profile>\n", maxsplit=1)[1].split("\n</stored_profile>", maxsplit=1)[0]
    assert stored.splitlines() == [
        "<explicit_info>",
        f'<item index="0">{{"evidence": "E1.", "description": "Uses Python.", "category": "Stack", "observed_at": "{when}"}}</item>',
        f'<item index="1">{{"evidence": "E2.", "description": "Lives in Porto.", "category": "Home", "observed_at": "{when}"}}</item>',
        "</explicit_info>",
        "<implicit_traits>",
        f'<item index="0">{{"basis": "B.", "description": "Decides methodically.", "trait": "Methodical", "observed_at": "{when}"}}</item>',
        "</implicit_traits>",
    ]
    assert "===" not in user
    assert "[0] " not in user
    root = ElementTree.fromstring(f"<root>{user}</root>")
    assert [item.get("index") for item in root.iter("item")] == ["0", "1", "0"]


async def test_an_empty_bucket_renders_as_an_empty_element() -> None:
    fake = FakeLLMClient(responses=[_operations({"action": "none"})])
    await ProfileExtractor(llm=fake).aextract_from_episodes(
        [_episode("Alice adopted uv.")], owner_id=_OWNER_ID, owner_name=_OWNER_NAME, old_profile=_old_profile()
    )

    assert "\n<implicit_traits>\n</implicit_traits>\n</stored_profile>" in _user_prompt(fake)


def test_render_item_puts_known_fields_in_declared_order_and_leaves_host_keys_out() -> None:
    legacy = ExplicitInfoItem.model_validate(
        {"note": "kept", "category": "Stack", "observed_at": _TIMESTAMP, "description": "D.", "evidence": "E."}
    )

    rendered = json.loads(_render_item(legacy))

    assert list(rendered) == ["evidence", "description", "category", "observed_at"]
    assert rendered["observed_at"] == format_utc_time(_TIMESTAMP)


def test_render_item_leaves_a_missing_field_missing() -> None:
    item = ImplicitTraitItem(description="D.", trait="T")

    assert list(json.loads(_render_item(item))) == ["description", "trait"]


_HOST_KEYS = {"item_id": "item-7", "source": "crm-import"}


async def test_every_stage_shows_stored_items_without_host_keys() -> None:
    """A host's own keys are stored, not shown: UPDATE, COMPACT and REGROUP each render items without them."""
    crowded = [
        {"category": f"dimension-{index}", "description": f"Fact {index}.", "evidence": "E.", **_HOST_KEYS}
        for index in range(60)
    ]
    grouped = [
        {"category": "Environment", "description": f"Fact {index}.", "evidence": "E.", **_HOST_KEYS}
        for index in range(8)
    ]
    regroup = json.dumps({"items": [{"evidence": "E.", "description": "Uses Ruff.", "category": "Environment"}]})
    fake = FakeLLMClient(
        responses=[
            _operations(_add("Prefers Ruff.", category="new")),
            _compact_payload(),
            _operations(_add("Uses Ruff.", category="Environment")),
            regroup,
        ]
    )
    extractor = ProfileExtractor(llm=fake)
    for items in (crowded, grouped):
        await extractor.aextract_from_episodes(
            [_episode("Alice chose Ruff.")],
            owner_id=_OWNER_ID,
            owner_name=_OWNER_NAME,
            old_profile=_old_profile(explicit_info=items),
        )

    assert fake.call_count == 4  # UPDATE + COMPACT, then UPDATE + REGROUP
    for call in range(fake.call_count):
        user = _user_prompt(fake, call)
        assert "Fact 1." in user, call
        assert all(f'"{key}"' not in user for key in _HOST_KEYS), call


async def test_an_update_keeps_host_keys_it_never_showed() -> None:
    stored = [{"category": "Technical stack", "description": "Works mainly in Python.", "evidence": "E.", **_HOST_KEYS}]
    fake = FakeLLMClient(
        responses=[
            _operations(
                _update(0, category="Technical stack", description="Works mainly in Python.", evidence="E. Again.")
            )
        ]
    )

    profile = await ProfileExtractor(llm=fake).aextract_from_episodes(
        [_episode("Alice used Python again.")],
        owner_id=_OWNER_ID,
        owner_name=_OWNER_NAME,
        old_profile=_old_profile(explicit_info=stored),
    )

    (item,) = _explicit_items(profile)
    assert item["evidence"] == "E. Again."
    assert {key: item[key] for key in _HOST_KEYS} == _HOST_KEYS


_CLOSING_TAGS = (
    "</narrative>",
    "</episode_narratives>",
    "</item>",
    "</explicit_info>",
    "</implicit_traits>",
    "</stored_profile>",
    "</group>",
)
_FULLWIDTH_LESS_THAN = "\uff1c"


@pytest.mark.parametrize("tag", _CLOSING_TAGS)
async def test_a_closing_tag_in_a_narrative_is_defused_in_any_case(tag: str) -> None:
    """Only the ``<`` turns fullwidth, so the element still closes where it should and the text reads the same."""
    text = f"Alice typed {tag} and {tag.upper()} into the chat."
    fake = FakeLLMClient(responses=[_profile_payload()])

    await ProfileExtractor(llm=fake).aextract_from_episodes(
        [_episode(text)], owner_id=_OWNER_ID, owner_name=_OWNER_NAME
    )

    defused = f"Alice typed {_FULLWIDTH_LESS_THAN}{tag[1:]} and {_FULLWIDTH_LESS_THAN}{tag.upper()[1:]} into the chat."
    assert _dated(defused) in _user_prompt(fake)


@pytest.mark.parametrize("tag", _CLOSING_TAGS)
def test_a_closing_tag_in_a_stored_item_is_defused_in_any_case(tag: str) -> None:
    item = ExplicitInfoItem(evidence=f"Said {tag.title()}.", description=f"Writes {tag}.", category="Docs")

    rendered = json.loads(_render_item(item))

    assert rendered["evidence"] == f"Said {_FULLWIDTH_LESS_THAN}{tag.title()[1:]}."
    assert rendered["description"] == f"Writes {_FULLWIDTH_LESS_THAN}{tag[1:]}."


def test_other_markup_in_a_stored_item_passes_as_written() -> None:
    """Only the prompts' own closing tags are touched; text is otherwise shown exactly as stored."""
    text = "Uses <div>, <item>, </items>, </ item> and &lt;b&gt;."
    item = ExplicitInfoItem(description=text, category="Docs")

    assert json.loads(_render_item(item))["description"] == text


async def test_maintenance_stages_show_items_in_declared_field_order() -> None:
    """COMPACT and REGROUP show the stored JSON too, so it reads in the order the reply is asked to write."""
    messages = await _messages_by_stage(None)
    # COMPACT nests the items one level deeper (inside the bucket lists) than REGROUP (a bare list).
    expected = {
        "compact": '"evidence": "E.",\n      "description": "Fact 0.",\n      "category": "dimension-0"\n',
        "regroup": '"evidence": "E.",\n    "description": "Fact 0.",\n    "category": "Environment"\n',
    }
    for stage, item in expected.items():
        assert item in messages[stage][1], stage


# ==========================================================================
# Language — the first sentence names what to write in; the original-form sentence is shared
# ==========================================================================


@pytest.mark.parametrize("retired", ["personality tag", "tag examples", "EPISODE_TEXT", "CRITICAL", "MUST"])
async def test_no_rendered_system_message_names_what_the_prompt_does_not_contain(retired: str) -> None:
    """Nothing the prompts no longer contain is named, and the language rule no longer shouts.

    v2 items have no personality tags, no stage shows tag examples, and no message has an EPISODE_TEXT.
    """
    for stage, (system, _) in (await _messages_by_stage(None)).items():
        assert retired not in system, stage


async def test_init_fallback_points_at_the_element_that_holds_the_narratives() -> None:
    system, user = (await _messages_by_stage(None))["init"]
    language_rule = system.split(_LANGUAGE_BLOCK_OPEN, maxsplit=1)[1].split("\n</output_language>", maxsplit=1)[0]

    assert language_rule == (
        "Write every output field in the language of the narratives in <episode_narratives>; this choice becomes "
        "the profile's language. Names, user IDs, proper nouns and technical terms keep their original form."
    )
    assert "\n<episode_narratives>\n" in user


@pytest.mark.parametrize(
    ("stage", "first_sentence"),
    [
        (
            "update",
            "Write every output field in the language most of the stored profile in <stored_profile> is written in, "
            "even when the narratives are in another language.",
        ),
        (
            "compact",
            "Write every output field in the language most of the stored profile in <stored_profile> is written in.",
        ),
        ("regroup", "Write every output field in the language most of the items in <group> are written in."),
    ],
)
async def test_each_maintenance_fallback_names_the_element_it_inherits_from(stage: str, first_sentence: str) -> None:
    system, _ = (await _messages_by_stage(None))[stage]
    language_rule = system.split(_LANGUAGE_BLOCK_OPEN, maxsplit=1)[1].split("\n</output_language>", maxsplit=1)[0]

    assert language_rule.startswith(f"{first_sentence} ")


# ==========================================================================
# Log lines — counts, dates, lengths and enums only, never what the model or the input wrote
# ==========================================================================

_PROFILE_LOGGER = "everalgo.user_memory.profile"
_LOG_PAIR = re.compile(r"(\w+)=(.*?)(?= \w+=|$)")


def _log_fields(caplog: pytest.LogCaptureFixture, prefix: str) -> list[dict[str, str]]:
    """The ``key=value`` pairs of every captured line opening with ``prefix``, as the module docstring defines them."""
    return [
        dict(_LOG_PAIR.findall(record.getMessage().removeprefix(prefix).strip()))
        for record in caplog.records
        if record.getMessage().startswith(prefix)
    ]


async def test_each_update_pass_logs_its_ops_its_date_and_the_profile_date_it_leaves(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A straddling batch logs one line per pass; the historical pass's refusals carry the dates that decided them."""
    lives = {"category": "Location", "description": "Lives in Shanghai.", "evidence": "e", "observed_at": _TIMESTAMP}
    stack = {"category": "Tech stack", "description": "Works in Python.", "evidence": "e", "observed_at": _TIMESTAMP}
    historical_reply = _operations(
        _update(0, category="Location", description="Lives in Beijing.", evidence="e"),
        _delete(1),
        _update(1, category="Tech stack", description="Works in Python.", evidence="e. Wrote a script."),
        _add("Uses Ruff.", category="tech   stack"),
        {"action": "none"},
    )
    current_reply = _operations(_update(0, category="Location", description="Lives in Hangzhou.", evidence="e"))
    fake = FakeLLMClient(responses=[historical_reply, current_reply])

    with caplog.at_level(logging.INFO, logger=_PROFILE_LOGGER):
        await ProfileExtractor(llm=fake).aextract_from_episodes(
            [
                _episode("Alice relocated to Hangzhou.", timestamp=_LATER),
                _episode("Alice lived in Beijing and uses Ruff.", timestamp=_EARLIER),
            ],
            owner_id=_OWNER_ID,
            owner_name=_OWNER_NAME,
            old_profile=_old_profile(explicit_info=[lives, stack], timestamp=_TIMESTAMP),
        )

    historical, current = _log_fields(caplog, "profile update applied:")
    assert historical == {
        "pass": "historical",
        "observed_at_ms": str(_EARLIER),
        "add": "1",
        "update": "2",
        "delete": "1",
        "none": "1",
        "rejected": "2",
        "rewrote_description": "0",
        "history_appended": "0",
        "grounding_only": "1",
        "canonicalised": "1",
        "profile_timestamp_ms": f"{_TIMESTAMP}->{_TIMESTAMP}",
        "explicit": "3",
        "implicit": "0",
    }
    assert current == {
        "pass": "current",
        "observed_at_ms": str(_LATER),
        "add": "0",
        "update": "1",
        "delete": "0",
        "none": "0",
        "rejected": "0",
        "rewrote_description": "1",
        "history_appended": "0",
        "grounding_only": "0",
        "canonicalised": "0",
        "profile_timestamp_ms": f"{_TIMESTAMP}->{_LATER}",
        "explicit": "3",
        "implicit": "0",
    }
    stale = {"item_observed_at_ms": str(_TIMESTAMP), "batch_observed_at_ms": str(_EARLIER)}
    assert _log_fields(caplog, "profile update op rejected:") == [
        {"action": "update", "type": "explicit_info", "index": "0", "reason": "stale evidence", **stale},
        {"action": "delete", "type": "explicit_info", "index": "1", "reason": "stale evidence", **stale},
    ]


@pytest.mark.parametrize(
    ("categories", "expected"),
    [(None, "0"), ([], "0"), (["Work", " Work ", "", "Hobbies"], "2")],
)
async def test_entry_line_counts_the_normalised_category_snapshot(
    categories: list[str] | None, expected: str, caplog: pytest.LogCaptureFixture
) -> None:
    """The size of the snapshot the prompts carry, after stripping, blank removal and deduplication; never the names."""
    fake = FakeLLMClient(responses=[_profile_payload()])
    with caplog.at_level(logging.INFO, logger=_PROFILE_LOGGER):
        await ProfileExtractor(llm=fake).aextract_from_episodes(
            [_episode("Alice works in Python.")], owner_id=_OWNER_ID, owner_name=_OWNER_NAME, categories=categories
        )

    (entry,) = [record.getMessage() for record in caplog.records if record.getMessage().startswith("extracting")]
    assert re.findall(r"\bcategories=(\d+),", entry) == [expected]
    assert "Hobbies" not in caplog.text


async def test_episode_init_dedupe_is_one_summary_line(caplog: pytest.LogCaptureFixture) -> None:
    item = {"category": "Stack", "description": "Works in Python.", "evidence": "e"}
    reply = json.dumps({"explicit_info": [item, item, item], "implicit_traits": []})
    with caplog.at_level(logging.WARNING, logger=_PROFILE_LOGGER):
        await ProfileExtractor(llm=FakeLLMClient(responses=[reply])).aextract_from_episodes(
            [_episode("Alice works in Python.")], owner_id=_OWNER_ID, owner_name=_OWNER_NAME
        )

    assert _log_fields(caplog, "profile duplicates dropped:") == [{"source": "episode-init", "count": "2"}]


async def test_no_log_line_carries_narratives_or_profile_content(caplog: pytest.LogCaptureFixture) -> None:
    """INIT then a straddling UPDATE, every line at DEBUG: the narratives and every item field stay out of the log."""
    marker = "EPISODE-MARKER-51f0"
    init_reply = json.dumps(
        {
            "explicit_info": [
                {"category": f"Stack {marker}", "description": f"Works in Python {marker}.", "evidence": marker},
                {"category": f"Stack {marker}", "description": f"Works in Python {marker}.", "evidence": marker},
            ],
            "implicit_traits": [{"trait": marker, "description": f"Careful {marker}.", "basis": marker}],
        }
    )
    update_reply = _operations(
        _update(0, category=f"Stack {marker}", description=f"Works in Rust {marker}.", evidence=marker),
        _delete(0),
        _add(f"Uses Ruff {marker}.", category=f"stack   {marker}", evidence=marker),
    )
    extractor = ProfileExtractor(llm=FakeLLMClient(responses=[init_reply, update_reply]))

    with caplog.at_level(logging.DEBUG, logger=_PROFILE_LOGGER):
        profile = await extractor.aextract_from_episodes(
            [_episode(f"Alice works in Python {marker}.", timestamp=_LATER)],
            owner_id=_OWNER_ID,
            owner_name=_OWNER_NAME,
            categories=[f"Stack {marker}"],
        )
        await extractor.aextract_from_episodes(
            [_episode(f"Alice tried Rust {marker}.", timestamp=_EARLIER)],
            owner_id=_OWNER_ID,
            owner_name=_OWNER_NAME,
            old_profile=profile,
        )

    for prefix in ("profile duplicates dropped:", "profile update op rejected:", "profile add label canonicalised:"):
        assert prefix in caplog.text
    assert marker.casefold() not in caplog.text.casefold()


async def test_episode_entry_logs_the_validated_language_and_refuses_an_unknown_one_unlogged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    marker = "LANGUAGE-MARKER-77b2"
    fake = FakeLLMClient(responses=[_profile_payload()])
    extractor = ProfileExtractor(llm=fake)

    with caplog.at_level(logging.DEBUG, logger=_PROFILE_LOGGER):
        await extractor.aextract_from_episodes(
            [_episode("Alice works in Python.")], owner_id=_OWNER_ID, owner_name=_OWNER_NAME, output_language="GERMAN"
        )
        with pytest.raises(ValueError, match="names no supported language"):
            await extractor.aextract_from_episodes(
                [_episode("Alice works in Python.")],
                owner_id=_OWNER_ID,
                owner_name=_OWNER_NAME,
                output_language=f"Klingon {marker}",
            )

    entries = [r.getMessage() for r in caplog.records if r.getMessage().startswith("extracting profile from Episodes:")]
    assert [re.findall(r"\boutput_language=(\w+),", entry) for entry in entries] == [["German"]]
    assert fake.call_count == 1
    assert marker not in caplog.text
