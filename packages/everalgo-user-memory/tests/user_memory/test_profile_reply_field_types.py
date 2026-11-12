"""The Episode profile replies' optional fields: typed in the strict schema, still tolerant to validation.

Strict structured outputs refuse an untyped value schema, and these fields were plain ``Any``: on rc6 every Episode
INIT and UPDATE call failed with a 400 on OpenAI. Their schema now declares the type the prompt asks for, while
validation still lets a stray value through to the item checks, which drop the one item rather than the reply.
"""

from __future__ import annotations

from typing import Any

from everalgo.llm import StrictOutputModel, to_openai_response_format
from everalgo.user_memory.profile._replies import (
    _AllowOp,
    _EpisodeExplicitItem,
    _EpisodeForgetOp,
    _EpisodeInitOutput,
)


def _properties(model: type[StrictOutputModel], name: str) -> dict[str, Any]:
    """The strict schema's properties of ``name``: a ``$defs`` entry, or ``model`` itself."""
    response_format = to_openai_response_format(model, "json_schema")
    assert response_format is not None
    schema: dict[str, Any] = response_format["json_schema"]["schema"]
    owner: dict[str, Any] = schema.get("$defs", {}).get(name, schema)
    properties: dict[str, Any] = owner["properties"]
    return properties


def test_the_reply_only_fields_declare_the_type_the_prompt_asks_for() -> None:
    item = _properties(_EpisodeInitOutput, "_EpisodeExplicitItem")
    assert item["pattern"] == {"anyOf": [{"type": "boolean"}, {"type": "null"}]}
    assert item["pending"] == {"anyOf": [{"type": "integer"}, {"type": "null"}]}
    assert item["quote"] == {"anyOf": [{"type": "string"}, {"type": "null"}]}
    forget = _properties(_EpisodeForgetOp, "_EpisodeForgetOp")
    assert forget["pending"] == {"anyOf": [{"type": "array", "items": {"type": "integer"}}, {"type": "null"}]}


def test_validation_still_lets_a_stray_value_through_to_the_checks() -> None:
    """A model now and then writes an index as a string or several quotes; the checks handle both."""
    allow = _AllowOp.model_validate({"action": "allow", "matter": "0", "quote": "please record it again"})
    assert allow.matter == "0"
    item = _EpisodeExplicitItem.model_validate(
        {"evidence": "e", "description": "d", "category": "c", "quote": ["one", "two"], "pattern": "yes"}
    )
    assert item.quote == ["one", "two"]


def test_a_left_out_field_is_none() -> None:
    item = _EpisodeExplicitItem.model_validate({"evidence": "e", "description": "d", "category": "c"})
    assert (item.pattern, item.pending, item.quote, item.regular) == (None, None, None, None)
