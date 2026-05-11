"""Cross-type JSON round-trip parametrized check.

Ensures every public type from ``everalgo.types`` survives
``model_dump_json`` -> ``model_validate_json`` cleanly, including
the ``extra='allow'`` path for Episode.
"""

from typing import Any

import pytest

from everalgo.types import Episode, MemCell, Message, MessageRole


@pytest.mark.parametrize(
    "obj",
    [
        Message(role=MessageRole.USER, content="hi", timestamp=1),
        Message(role=MessageRole.ASSISTANT, content="response", timestamp=2),
        MemCell(id="m_empty", messages=[], timestamp=1),
        MemCell(
            id="m_one",
            messages=[Message(role=MessageRole.USER, content="hi", timestamp=1)],
            timestamp=10,
        ),
        Episode(
            id="ep1",
            owner_id="u1",
            episode="Alice asked about Q3.",
            timestamp=1,
            parent_id="m1",
        ),
        Episode.model_validate(
            {
                "id": "ep2",
                "owner_id": "u2",
                "episode": "Bob shared the plan.",
                "timestamp": 2,
                "parent_id": "m2",
                "summary": "shared plan",
                "keywords": ["plan"],
            }
        ),
    ],
    ids=[
        "message-user",
        "message-assistant",
        "memcell-empty",
        "memcell-one-message",
        "episode-minimal",
        "episode-with-extras",
    ],
)
def test_model_dump_json_then_validate_json_round_trips(obj: Any) -> None:
    serialised = obj.model_dump_json()
    rebuilt = type(obj).model_validate_json(serialised)
    assert rebuilt == obj
