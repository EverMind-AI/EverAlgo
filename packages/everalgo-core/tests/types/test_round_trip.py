"""Cross-type JSON round-trip parametrized check.

Ensures every public type from ``everalgo.types`` survives ``model_dump_json`` -> ``model_validate_json``
cleanly, including the ``extra='allow'`` path for Episode.
"""

from typing import Any

import pytest

from everalgo.types import ChatMessage, Episode, MemCell


@pytest.mark.parametrize(
    "obj",
    [
        ChatMessage(id="m1", role="user", content="hi", timestamp=1, sender_id="user"),
        ChatMessage(id="m2", role="assistant", content="response", timestamp=2, sender_id="assistant"),
        MemCell(items=[], timestamp=1),
        MemCell(
            items=[ChatMessage(id="m3", role="user", content="hi", timestamp=1, sender_id="user")],
            timestamp=10,
        ),
        Episode(owner_id="u1", episode="Alice asked about Q3.", summary="Alice asked about Q3.", timestamp=1),
        Episode.model_validate(
            {
                "owner_id": "u2",
                "episode": "Bob shared the plan.",
                "timestamp": 2,
                "summary": "shared plan",
                "keywords": ["plan"],
            }
        ),
    ],
    ids=[
        "chatmessage-user",
        "chatmessage-assistant",
        "memcell-empty",
        "memcell-one-message",
        "episode-minimal",
        "episode-with-extras",
    ],
)
def test_model_dump_json_then_validate_json_round_trips(obj: Any) -> None:
    serialised = obj.model_dump_json()
    rebuilt = type(obj).model_validate_json(serialised)  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
    assert rebuilt == obj
