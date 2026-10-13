"""The knowledge reply models keep the operator-side obligations of ``docs/concepts/llm-contract.md`` §2.2 / §4.3."""

from __future__ import annotations

import inspect

import pytest
from pydantic import BaseModel

from everalgo.knowledge import _replies
from everalgo.llm import StrictOutputModel, to_openai_response_format

_REQUEST_MODELS: list[type[StrictOutputModel]] = [
    _replies.TopicTreeReply,
    _replies.ContentMergeReply,
    _replies.TopicMergeReply,
    _replies.CategoryReply,
    _replies.TopicSplitReply,
    _replies.UncoveredAssignReply,
]


def _declared_models() -> list[type[BaseModel]]:
    """Every model class the module exports."""
    return [
        value
        for name in _replies.__all__
        if inspect.isclass(value := getattr(_replies, name)) and issubclass(value, BaseModel)
    ]


def test_every_exported_model_is_a_request_model_or_reachable_from_one() -> None:
    reachable: set[type[BaseModel]] = set()

    def visit(model: type[BaseModel]) -> None:
        if model in reachable:
            return
        reachable.add(model)
        for info in model.model_fields.values():
            for arg in _type_args(info.annotation):
                visit(arg)

    for model in _REQUEST_MODELS:
        visit(model)

    assert set(_declared_models()) == reachable


def _type_args(annotation: object) -> list[type[BaseModel]]:
    """The model classes nested anywhere in ``annotation`` (``list[X]``, ``X | None`` ...)."""
    if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
        return [annotation]
    return [model for arg in getattr(annotation, "__args__", ()) for model in _type_args(arg)]


@pytest.mark.parametrize("model", _declared_models(), ids=lambda m: m.__name__)
def test_model_inherits_strict_output_model_directly(model: type[BaseModel]) -> None:
    assert model.__bases__ == (StrictOutputModel,)


@pytest.mark.parametrize("model", _declared_models(), ids=lambda m: m.__name__)
def test_model_fields_have_no_defaults(model: type[BaseModel]) -> None:
    assert all(info.is_required() for info in model.model_fields.values())


@pytest.mark.parametrize("model", _declared_models(), ids=lambda m: m.__name__)
def test_model_carries_no_validators(model: type[BaseModel]) -> None:
    decorators = model.__pydantic_decorators__
    assert not (decorators.field_validators or decorators.model_validators or decorators.validators)


@pytest.mark.parametrize("model", _REQUEST_MODELS, ids=lambda m: m.__name__)
def test_request_model_builds_a_strict_schema(model: type[StrictOutputModel]) -> None:
    """``to_openai_response_format`` raises for anything outside the strict subset, a dynamic-key dict included."""
    built = to_openai_response_format(model, "json_schema")

    assert built is not None
    assert built["json_schema"]["strict"] is True
