"""Tests for ``everalgo.llm.output``: ``StrictOutputModel`` rejects what the strict schema rejects; ``parse_reply``."""

import pytest
from pydantic import BaseModel, ValidationError

from everalgo.llm import LLMOutputError, StrictOutputModel, describe_errors, parse_reply
from everalgo.llm.schema import _to_strict_json_schema


class _Counter(StrictOutputModel):
    count: int


class _Child(StrictOutputModel):
    name: str


class _Parent(StrictOutputModel):
    child: _Child


class _PlainChild(BaseModel):
    name: str


class _ParentOfPlainChild(StrictOutputModel):
    child: _PlainChild


def test_base_holds_configuration_only() -> None:
    assert StrictOutputModel.model_config == {"strict": True, "extra": "forbid"}
    assert StrictOutputModel.model_fields == {}


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param("{}", id="a missing field is not filled in"),
        pytest.param('{"count": "3"}', id="a string is not coerced into an int"),
        pytest.param('{"count": 3, "extra": true}', id="an extra key is not ignored"),
    ],
)
def test_rejects_what_the_strict_schema_rejects(payload: str) -> None:
    with pytest.raises(ValidationError):
        _Counter.model_validate_json(payload)


def test_accepts_a_conforming_reply() -> None:
    assert _Counter.model_validate_json('{"count": 3}').count == 3


def test_nullable_field_takes_explicit_null_but_not_a_missing_key() -> None:
    class _Note(StrictOutputModel):
        note: str | None

    assert _Note.model_validate_json('{"note": null}').note is None
    with pytest.raises(ValidationError):
        _Note.model_validate_json("{}")


def test_inheriting_sub_model_forbids_extra_keys() -> None:
    with pytest.raises(ValidationError):
        _Parent.model_validate_json('{"child": {"name": "x", "extra": 1}}')


def test_configuration_does_not_propagate_through_field_types() -> None:
    """Why sub-models must inherit the base too: a plain child silently accepts what the schema forbids."""
    parsed = _ParentOfPlainChild.model_validate_json('{"child": {"name": "x", "extra": 1}}')

    assert parsed.child.name == "x"


def test_strict_output_models_lie_inside_the_strict_subset() -> None:
    schema = _to_strict_json_schema(_Parent)

    assert schema["additionalProperties"] is False
    assert schema["$defs"]["_Child"]["additionalProperties"] is False


# ── parse_reply / describe_errors ──────────────────────────────────────────────────────────────────


def test_parse_reply_returns_the_validated_model() -> None:
    assert parse_reply(_Parent, '{"child": {"name": "Ada"}}', operator="demo") == _Parent(child=_Child(name="Ada"))


@pytest.mark.parametrize(
    ("reply", "errors"),
    [
        pytest.param("not json", "<root>: Invalid JSON: expected ident at line 1 column 2", id="not JSON"),
        pytest.param('{"child": {"name": 7}}', "child.name: Input should be a valid string", id="nested location"),
        pytest.param(
            '{"child": {}, "extra": 1}',
            "extra: Extra inputs are not permitted; child.name: Field required",
            id="several errors",
        ),
    ],
)
def test_parse_reply_names_the_operator_the_model_and_each_error(reply: str, errors: str) -> None:
    with pytest.raises(LLMOutputError) as raised:
        parse_reply(_Parent, reply, operator="profile init")

    assert str(raised.value) == f"profile init reply is not a valid _Parent: {errors}"
    assert isinstance(raised.value.__cause__, ValidationError)


def test_parse_reply_never_quotes_the_reply() -> None:
    with pytest.raises(LLMOutputError) as raised:
        parse_reply(_Counter, '{"count": "a private sentence"}', operator="demo")

    assert "a private sentence" not in str(raised.value)
    assert str(raised.value) == "demo reply is not a valid _Counter: count: Input should be a valid integer"


def test_describe_errors_leaves_out_the_input_values() -> None:
    with pytest.raises(ValidationError) as raised:
        _Counter.model_validate_json('{"count": "a private sentence"}')

    assert describe_errors(raised.value) == "count: Input should be a valid integer"
