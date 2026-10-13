"""Profile's typed buckets: item models round-trip host keys and refuse an item with no description."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from everalgo.types import ExplicitInfoItem, ImplicitTraitItem, Profile


def _stored_profile() -> dict[str, Any]:
    return {
        "owner_id": "u1",
        "summary": "Writes Python.",
        "timestamp": 1_700_000_000_000,
        "host_version": 3,
        "explicit_info": [
            {
                "evidence": "Alice writes Python.",
                "description": "Writes Python.",
                "category": "Skills",
                "observed_at": 1_700_000_000_000,
                "source_id": "doc-7",
            }
        ],
        "implicit_traits": [
            {"basis": "chose ruff twice", "description": "Prefers lean tooling.", "trait": "Pragmatic"}
        ],
    }


def test_profile_items_validate_into_their_bucket_models() -> None:
    profile = Profile.model_validate(_stored_profile())

    assert isinstance(profile.explicit_info[0], ExplicitInfoItem)
    assert isinstance(profile.implicit_traits[0], ImplicitTraitItem)
    assert profile.explicit_info[0].category == "Skills"
    assert profile.implicit_traits[0].trait == "Pragmatic"


def test_profile_round_trips_host_keys_at_both_levels() -> None:
    data = _stored_profile()

    profile = Profile.model_validate(data)

    assert profile.explicit_info[0].model_extra == {"source_id": "doc-7"}
    assert profile.model_extra == {"host_version": 3}
    assert profile.model_dump(exclude_unset=True) == data
    assert Profile.model_validate_json(profile.model_dump_json()) == profile


def test_unset_item_fields_stay_out_of_an_exclude_unset_dump() -> None:
    item = ImplicitTraitItem(description="Prefers lean tooling.", trait="Pragmatic")

    assert item.model_dump(exclude_unset=True) == {"description": "Prefers lean tooling.", "trait": "Pragmatic"}
    assert item.model_dump() == {
        "description": "Prefers lean tooling.",
        "trait": "Pragmatic",
        "basis": None,
        "observed_at": None,
    }


def test_buckets_default_to_empty_lists() -> None:
    profile = Profile(owner_id="u1", summary="s", timestamp=1)

    assert profile.explicit_info == []
    assert profile.implicit_traits == []


@pytest.mark.parametrize("bucket", ["explicit_info", "implicit_traits"])
def test_an_item_without_a_description_is_refused(bucket: str) -> None:
    """Description is the item's identity: without it the item can be neither deduplicated nor compared."""
    with pytest.raises(ValidationError, match=rf"{bucket}\.0\.description"):
        Profile.model_validate({"owner_id": "u1", "summary": "s", "timestamp": 1, bucket: [{"category": "c"}]})


@pytest.mark.parametrize("bucket", ["explicit_info", "implicit_traits"])
def test_a_non_object_item_is_refused(bucket: str) -> None:
    with pytest.raises(ValidationError, match=rf"{bucket}\.0"):
        Profile.model_validate({"owner_id": "u1", "summary": "s", "timestamp": 1, bucket: ["not an object"]})
