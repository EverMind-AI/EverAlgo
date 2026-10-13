"""Tests for ``_to_strict_json_schema`` and the ``to_<api>_response_format`` builders on top of it.

Inside the strict subset the result is pinned, key order included, to what openai SDK 2.35.1's
``openai.lib._pydantic.to_strict_json_schema`` returns for the same model; outside it, the function raises where the
SDK emits a schema the API rejects.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

import pytest
from pydantic import BaseModel, ConfigDict, Field, RootModel, WithJsonSchema

from everalgo.llm import (
    StrictOutputModel,
    to_anthropic_response_format,
    to_gemini_response_format,
    to_ollama_response_format,
    to_openai_response_format,
)
from everalgo.llm.schema import _to_strict_json_schema

if TYPE_CHECKING:
    from collections.abc import Callable


class _Flat(BaseModel):
    title: str
    score: int


class _Leaf(BaseModel):
    label: str
    weight: float


class _Nested(BaseModel):
    """Model whose field is a list of sub-models, so the schema carries ``$defs`` and ``$ref``."""

    name: str
    leaves: list[_Leaf]


class _DecoratedRef(BaseModel):
    """Model whose sub-model field carries a description, so pydantic emits a ``$ref`` with siblings."""

    child: _Leaf = Field(description="the child")


class _Optional(BaseModel):
    """Model with ``X | None`` fields, both with and without a default."""

    required_note: str
    note: str | None = None
    tag: str | None = Field(default=None, description="a ref-free nullable with a description")


class _Tree(BaseModel):
    label: str
    children: list[_Tree]


class _StrictLeaf(StrictOutputModel):
    kind: Literal["a", "b"]


class _StrictRoot(StrictOutputModel):
    """A model written the recommended way: ``extra="forbid"`` already yields ``additionalProperties: false``."""

    leaves: list[_StrictLeaf]
    note: str | None


@pytest.mark.parametrize(
    ("model", "sdk_baseline"),
    [
        pytest.param(
            _Flat,
            {
                "properties": {
                    "title": {"title": "Title", "type": "string"},
                    "score": {"title": "Score", "type": "integer"},
                },
                "required": ["title", "score"],
                "title": "_Flat",
                "type": "object",
                "additionalProperties": False,
            },
            id="_Flat",
        ),
        pytest.param(
            _Nested,
            {
                "$defs": {
                    "_Leaf": {
                        "properties": {
                            "label": {"title": "Label", "type": "string"},
                            "weight": {"title": "Weight", "type": "number"},
                        },
                        "required": ["label", "weight"],
                        "title": "_Leaf",
                        "type": "object",
                        "additionalProperties": False,
                    }
                },
                "description": "Model whose field is a list of sub-models, so the schema carries ``$defs`` and ``$ref``.",
                "properties": {
                    "name": {"title": "Name", "type": "string"},
                    "leaves": {"items": {"$ref": "#/$defs/_Leaf"}, "title": "Leaves", "type": "array"},
                },
                "required": ["name", "leaves"],
                "title": "_Nested",
                "type": "object",
                "additionalProperties": False,
            },
            id="_Nested",
        ),
        pytest.param(
            _Optional,
            {
                "description": "Model with ``X | None`` fields, both with and without a default.",
                "properties": {
                    "required_note": {"title": "Required Note", "type": "string"},
                    "note": {"anyOf": [{"type": "string"}, {"type": "null"}], "title": "Note"},
                    "tag": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "description": "a ref-free nullable with a description",
                        "title": "Tag",
                    },
                },
                "required": ["required_note", "note", "tag"],
                "title": "_Optional",
                "type": "object",
                "additionalProperties": False,
            },
            id="_Optional",
        ),
        pytest.param(
            _DecoratedRef,
            {
                "$defs": {
                    "_Leaf": {
                        "properties": {
                            "label": {"title": "Label", "type": "string"},
                            "weight": {"title": "Weight", "type": "number"},
                        },
                        "required": ["label", "weight"],
                        "title": "_Leaf",
                        "type": "object",
                        "additionalProperties": False,
                    }
                },
                "description": "Model whose sub-model field carries a description, so pydantic emits a ``$ref`` with siblings.",
                "properties": {
                    "child": {
                        "description": "the child",
                        "properties": {
                            "label": {"title": "Label", "type": "string"},
                            "weight": {"title": "Weight", "type": "number"},
                        },
                        "required": ["label", "weight"],
                        "title": "_Leaf",
                        "type": "object",
                        "additionalProperties": False,
                    }
                },
                "required": ["child"],
                "title": "_DecoratedRef",
                "type": "object",
                "additionalProperties": False,
            },
            id="_DecoratedRef",
        ),
        pytest.param(
            _Tree,
            {
                "$defs": {
                    "_Tree": {
                        "properties": {
                            "label": {"title": "Label", "type": "string"},
                            "children": {"items": {"$ref": "#/$defs/_Tree"}, "title": "Children", "type": "array"},
                        },
                        "required": ["label", "children"],
                        "title": "_Tree",
                        "type": "object",
                        "additionalProperties": False,
                    }
                },
                "properties": {
                    "label": {"title": "Label", "type": "string"},
                    "children": {"items": {"$ref": "#/$defs/_Tree"}, "title": "Children", "type": "array"},
                },
                "required": ["label", "children"],
                "title": "_Tree",
                "type": "object",
                "additionalProperties": False,
            },
            id="_Tree",
        ),
        pytest.param(
            _StrictRoot,
            {
                "$defs": {
                    "_StrictLeaf": {
                        "additionalProperties": False,
                        "properties": {"kind": {"enum": ["a", "b"], "title": "Kind", "type": "string"}},
                        "required": ["kind"],
                        "title": "_StrictLeaf",
                        "type": "object",
                    }
                },
                "additionalProperties": False,
                "description": 'A model written the recommended way: ``extra="forbid"`` already yields ``additionalProperties: false``.',
                "properties": {
                    "leaves": {"items": {"$ref": "#/$defs/_StrictLeaf"}, "title": "Leaves", "type": "array"},
                    "note": {"anyOf": [{"type": "string"}, {"type": "null"}], "title": "Note"},
                },
                "required": ["leaves", "note"],
                "title": "_StrictRoot",
                "type": "object",
            },
            id="_StrictRoot",
        ),
    ],
)
def test_models_inside_the_subset_match_the_sdk_baseline(model: type[BaseModel], sdk_baseline: dict[str, Any]) -> None:
    schema = _to_strict_json_schema(model)

    assert json.dumps(schema) == json.dumps(sdk_baseline)


def test_returns_the_bare_schema_without_a_request_envelope() -> None:
    schema = _to_strict_json_schema(_Flat)

    assert schema["type"] == "object"
    assert "json_schema" not in schema
    assert "strict" not in schema


def test_every_object_forbids_extra_properties_and_requires_all_properties() -> None:
    """Spot-check the two rewrites the baseline comparison would also catch, on the nested model."""
    schema = _to_strict_json_schema(_Nested)

    assert schema["additionalProperties"] is False
    assert schema["required"] == ["name", "leaves"]
    leaf = schema["$defs"]["_Leaf"]
    assert leaf["additionalProperties"] is False
    assert leaf["required"] == ["label", "weight"]


def test_declared_additional_properties_false_is_kept() -> None:
    schema = _to_strict_json_schema(_StrictRoot)

    assert schema["additionalProperties"] is False
    assert schema["$defs"]["_StrictLeaf"]["additionalProperties"] is False


def test_single_arm_all_of_is_flattened() -> None:
    class _OneArm(BaseModel):
        value: Annotated[int, WithJsonSchema({"allOf": [{"type": "integer"}]})]

    value = _to_strict_json_schema(_OneArm)["properties"]["value"]

    assert "allOf" not in value
    assert value["type"] == "integer"


def test_an_inline_one_of_arm_is_made_strict_like_an_any_of_arm() -> None:
    """``oneOf`` is renamed before the node's children are rewritten, so its arms get the same rewrite."""
    arm = {"type": "object", "properties": {"a": {"type": "string"}}}

    class _InlineArm(BaseModel):
        value: Annotated[int, WithJsonSchema({"oneOf": [arm]})]

    value = _to_strict_json_schema(_InlineArm)["properties"]["value"]

    assert "oneOf" not in value
    assert value["anyOf"] == [{**arm, "additionalProperties": False, "required": ["a"]}]


def test_the_model_is_not_mutated_between_calls() -> None:
    """``model_json_schema`` hands out a fresh dict; rewriting it in place must not leak into a second call."""
    assert _to_strict_json_schema(_Nested) == _to_strict_json_schema(_Nested)


# ---------------------------------------------------------------------------
# Build-time checks: structures strict mode cannot express fail here, not with a 400 at request time
# ---------------------------------------------------------------------------


class _Counts(BaseModel):
    counts: dict[str, int]


class _Blob(BaseModel):
    blob: dict[str, Any]


class _Open(BaseModel):
    model_config = ConfigDict(extra="allow")

    value: int


class _HoldsOpen(BaseModel):
    """The open object is nested, so the error must name the definition it sits in."""

    inner: _Open


class _TwoArms(BaseModel):
    value: Annotated[int, WithJsonSchema({"allOf": [{"type": "integer"}, {"minimum": 0}]})]


class _ListOfDicts(BaseModel):
    rows: list[dict[str, str]]


class _OneOfBesideAnyOf(BaseModel):
    """pydantic never emits this shape; a hand-written schema can, and the two cannot be merged into one ``anyOf``."""

    value: Annotated[int | str, WithJsonSchema({"anyOf": [{"type": "integer"}], "oneOf": [{"type": "string"}]})]


def test_dynamic_key_mapping_raises_with_path_and_alternative() -> None:
    with pytest.raises(ValueError, match=r"_Counts\.counts") as caught:
        _to_strict_json_schema(_Counts)

    assert "dict[str, X]" in str(caught.value)
    assert "list[SubModel]" in str(caught.value)


def test_dynamic_key_mapping_inside_array_items_names_the_item_path() -> None:
    with pytest.raises(ValueError, match=r"_ListOfDicts\.rows\[\]"):
        _to_strict_json_schema(_ListOfDicts)


def test_open_mapping_field_raises() -> None:
    with pytest.raises(ValueError, match=r"_Blob\.blob.*additionalProperties: true"):
        _to_strict_json_schema(_Blob)


def test_extra_allow_model_raises_with_the_fix() -> None:
    with pytest.raises(ValueError, match=r"^_Open: .*additionalProperties: true") as caught:
        _to_strict_json_schema(_Open)

    assert "StrictOutputModel" in str(caught.value)


def test_nested_open_object_names_its_definition() -> None:
    with pytest.raises(ValueError, match=r"^_Open: "):
        _to_strict_json_schema(_HoldsOpen)


def test_multi_arm_all_of_raises() -> None:
    with pytest.raises(ValueError, match=r"_TwoArms\.value: an allOf with 2 arms"):
        _to_strict_json_schema(_TwoArms)


def test_one_of_beside_any_of_raises() -> None:
    with pytest.raises(ValueError, match=r"_OneOfBesideAnyOf\.value: a oneOf beside an anyOf"):
        _to_strict_json_schema(_OneOfBesideAnyOf)


def test_root_model_raises_even_when_it_wraps_an_object() -> None:
    """``RootModel`` is a ``BaseModel`` subclass, so the signature alone does not keep it out."""
    with pytest.raises(TypeError, match="RootModel"):
        _to_strict_json_schema(RootModel[int])
    with pytest.raises(TypeError, match="RootModel"):
        _to_strict_json_schema(RootModel[_Flat])


def test_recursive_model_top_level_is_followed_through_its_ref() -> None:
    """A self-referential model's top level is a bare ``$ref``; the object it points to satisfies the rule."""
    assert "type" not in _Tree.model_json_schema()

    assert _to_strict_json_schema(_Tree)["type"] == "object"


def test_non_object_top_level_raises() -> None:
    class _Scalar(BaseModel):
        model_config = ConfigDict(json_schema_extra={"type": "string"})

        value: int

    with pytest.raises(ValueError, match=r"_Scalar: the top level .* must be an object, got 'string'"):
        _to_strict_json_schema(_Scalar)


# ---------------------------------------------------------------------------
# to_openai_response_format: json_schema / json_object / prompt_only
# ---------------------------------------------------------------------------


def test_openai_json_schema_names_the_model_and_carries_its_strict_schema() -> None:
    assert to_openai_response_format(_Flat, "json_schema") == {
        "type": "json_schema",
        "json_schema": {"name": "_Flat", "schema": _to_strict_json_schema(_Flat), "strict": True},
    }


def test_openai_mode_defaults_to_json_schema() -> None:
    assert to_openai_response_format(_Flat) == to_openai_response_format(_Flat, "json_schema")


def test_openai_json_object_is_json_mode() -> None:
    assert to_openai_response_format(_Flat, "json_object") == {"type": "json_object"}


def test_openai_prompt_only_leaves_the_parameter_out() -> None:
    assert to_openai_response_format(_Flat, "prompt_only") is None


def test_openai_json_schema_refuses_a_model_outside_the_strict_subset() -> None:
    with pytest.raises(ValueError, match=r"_Counts\.counts"):
        to_openai_response_format(_Counts, "json_schema")


class _Add(StrictOutputModel):
    action: Literal["add"]
    text: str


class _Drop(StrictOutputModel):
    action: Literal["drop"]
    index: int


class _Edits(StrictOutputModel):
    """A list of discriminated-union items, and a nullable one whose ``oneOf`` sits inside an ``anyOf`` arm."""

    edits: list[Annotated[_Add | _Drop, Field(discriminator="action")]]
    last: Annotated[_Add | _Drop, Field(discriminator="action")] | None


def test_openai_discriminated_union_becomes_any_of_without_the_discriminator() -> None:
    """Strict mode rejects ``oneOf``; each arm keeps its own ``const``, so ``anyOf`` accepts the same replies."""
    raw = json.dumps(_Edits.model_json_schema())
    assert '"oneOf"' in raw
    assert '"discriminator"' in raw

    result = to_openai_response_format(_Edits)
    assert result is not None
    schema = result["json_schema"]["schema"]
    arms = [{"$ref": "#/$defs/_Add"}, {"$ref": "#/$defs/_Drop"}]

    assert '"oneOf"' not in json.dumps(schema)
    assert '"discriminator"' not in json.dumps(schema)
    assert schema["properties"]["edits"]["items"] == {"anyOf": arms}
    assert schema["properties"]["last"]["anyOf"] == [{"anyOf": arms}, {"type": "null"}]
    assert [schema["$defs"][name]["properties"]["action"] for name in ("_Add", "_Drop")] == [
        {"const": "add", "title": "Action", "type": "string"},
        {"const": "drop", "title": "Action", "type": "string"},
    ]


@pytest.mark.parametrize("mode", ["json_object", "prompt_only"])
def test_openai_other_modes_never_build_a_schema(mode: Literal["json_object", "prompt_only"]) -> None:
    """Only the strict-schema mode needs the model inside the strict subset."""
    assert "json_schema" not in json.dumps(to_openai_response_format(_Counts, mode))


# ---------------------------------------------------------------------------
# to_anthropic / to_gemini / to_ollama _response_format: the format of each API; keywords an API does not take move
# into the description, so the reply is still validated against the full model
# ---------------------------------------------------------------------------


class _Constrained(StrictOutputModel):
    code: Annotated[str, Field(pattern=r"^[A-Z]{3}$", description="Airport code")]
    score: Annotated[int, Field(ge=0, le=10)]
    tags: Annotated[list[str], Field(min_length=2)]
    note: Annotated[str, Field(max_length=20)]
    kind: Literal["a"]


class _Cat(StrictOutputModel):
    pet: Literal["cat"]


class _Dog(StrictOutputModel):
    pet: Literal["dog"]


class _Pet(StrictOutputModel):
    animal: Annotated[_Cat | _Dog, Field(discriminator="pet")]


def _properties(schema: dict[str, Any]) -> dict[str, Any]:
    return cast("dict[str, Any]", schema["properties"])


def test_anthropic_json_schema_carries_the_strict_schema_without_a_name() -> None:
    assert to_anthropic_response_format(_Flat) == {"type": "json_schema", "schema": _to_strict_json_schema(_Flat)}


def test_anthropic_prompt_only_leaves_the_parameter_out() -> None:
    assert to_anthropic_response_format(_Flat, "prompt_only") is None


def test_anthropic_moves_what_it_does_not_take_into_the_description() -> None:
    result = to_anthropic_response_format(_Constrained)
    assert result is not None
    properties = _properties(result["schema"])

    assert properties["code"] == {
        "type": "string",
        "title": "Code",
        "description": "Airport code\n\n{pattern: ^[A-Z]{3}$}",
    }
    assert properties["score"]["description"] == "{maximum: 10, minimum: 0}"
    assert properties["tags"]["description"] == "{minItems: 2}"
    assert properties["note"]["description"] == "{maxLength: 20}"
    moved = {"pattern", "minimum", "maximum", "minItems", "maxLength"}
    assert all(moved.isdisjoint(prop) for prop in properties.values())


def test_anthropic_keeps_a_min_items_of_one_a_listed_format_and_const() -> None:
    class _Kept(StrictOutputModel):
        items: Annotated[list[str], Field(min_length=1)]
        at: Annotated[str, WithJsonSchema({"type": "string", "format": "date-time"})]
        kind: Literal["a"]

    result = to_anthropic_response_format(_Kept)
    assert result is not None
    assert _properties(result["schema"]) == _properties(_to_strict_json_schema(_Kept))


def test_anthropic_moves_an_unlisted_string_format() -> None:
    class _Address(StrictOutputModel):
        host: Annotated[str, WithJsonSchema({"type": "string", "format": "ipvanyaddress"})]

    result = to_anthropic_response_format(_Address)
    assert result is not None
    assert _properties(result["schema"])["host"] == {
        "type": "string",
        "title": "Host",
        "description": "{format: ipvanyaddress}",
    }


def test_anthropic_turns_one_of_into_any_of() -> None:
    result = to_anthropic_response_format(_Pet)
    assert result is not None
    animal = _properties(result["schema"])["animal"]

    assert "oneOf" not in animal
    assert len(animal["anyOf"]) == 2


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda model: to_openai_response_format(model), id="openai"),
        pytest.param(lambda model: to_anthropic_response_format(model), id="anthropic"),
        pytest.param(lambda model: to_gemini_response_format(model), id="gemini"),
        pytest.param(lambda model: to_ollama_response_format(model), id="ollama"),
    ],
)
def test_no_builder_sends_one_of_or_the_discriminator(build: Callable[[type[BaseModel]], object]) -> None:
    """The rewrite happens in the strict schema every builder starts from, not in any one builder.

    Plain substrings, so a keyword a builder moves into a description counts too; ``_Pet`` has no docstring to trip them.
    """
    sent = json.dumps(build(_Pet))

    assert "oneOf" not in sent
    assert "discriminator" not in sent
    assert '"anyOf"' in sent


def test_gemini_json_schema_sets_the_mime_type_and_the_json_schema() -> None:
    assert to_gemini_response_format(_Flat) == {
        "responseMimeType": "application/json",
        "responseJsonSchema": _to_strict_json_schema(_Flat),
    }


def test_gemini_json_object_sets_only_the_mime_type() -> None:
    assert to_gemini_response_format(_Flat, "json_object") == {"responseMimeType": "application/json"}


def test_gemini_prompt_only_adds_no_entries() -> None:
    assert to_gemini_response_format(_Flat, "prompt_only") == {}


def test_gemini_moves_what_it_does_not_list_and_keeps_what_it_does() -> None:
    properties = _properties(to_gemini_response_format(_Constrained)["responseJsonSchema"])

    assert properties["code"]["description"] == "Airport code\n\n{pattern: ^[A-Z]{3}$}"
    assert properties["note"]["description"] == "{maxLength: 20}"
    assert properties["kind"] == {"type": "string", "title": "Kind", "description": "{const: a}"}
    assert (properties["score"]["minimum"], properties["score"]["maximum"]) == (0, 10)
    assert properties["tags"]["minItems"] == 2


def test_ollama_json_schema_is_the_bare_strict_schema() -> None:
    assert to_ollama_response_format(_Flat) == _to_strict_json_schema(_Flat)


def test_ollama_json_object_is_the_json_string() -> None:
    assert to_ollama_response_format(_Flat, "json_object") == "json"


def test_ollama_prompt_only_leaves_the_parameter_out() -> None:
    assert to_ollama_response_format(_Flat, "prompt_only") is None


def test_openai_and_ollama_keep_every_constraint() -> None:
    strict = _to_strict_json_schema(_Constrained)
    openai_format = to_openai_response_format(_Constrained)
    assert openai_format is not None

    assert openai_format["json_schema"]["schema"] == strict
    assert to_ollama_response_format(_Constrained) == strict
