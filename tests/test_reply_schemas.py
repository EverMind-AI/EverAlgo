"""Every reply model's strict schema gives each value a type, as strict structured outputs require.

``to_openai_response_format`` raises on what the strict subset cannot express, but an untyped value -- a field
annotated ``Any``, rendered ``{}`` -- passes it and is refused upstream with a 400 ("schema must have a 'type'
key"). user-memory 0.9.0rc6 shipped such fields, and its Episode profile path failed on every OpenAI call while
every test passed against doubles. This walks every ``StrictOutputModel`` in the workspace, so a reply model added
to any distribution is held to it.

Lives at the repo-root ``tests/`` because it spans every distribution, like ``test_prompt_conventions.py``.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import TYPE_CHECKING, Any

import pytest

import everalgo
from everalgo.llm import StrictOutputModel, to_openai_response_format

if TYPE_CHECKING:
    from collections.abc import Iterator

# A value schema carries one of these, or an untyped ``{}`` reaches the provider.
_TYPED = frozenset({"type", "anyOf", "$ref", "enum", "const"})


def _reply_models() -> list[type[StrictOutputModel]]:
    """Every ``StrictOutputModel`` subclass defined in the workspace's ``everalgo`` packages."""
    for module in pkgutil.walk_packages(everalgo.__path__, "everalgo."):
        importlib.import_module(module.name)
    found: dict[str, type[StrictOutputModel]] = {}
    pending = list(StrictOutputModel.__subclasses__())
    while pending:
        model = pending.pop()
        pending.extend(model.__subclasses__())
        if model.__module__.startswith("everalgo."):
            found[f"{model.__module__}.{model.__qualname__}"] = model
    return [found[name] for name in sorted(found)]


def _untyped(node: Any, path: str) -> Iterator[str]:
    """The paths of the value schemas under ``node`` that declare no type."""
    if isinstance(node, list):
        for i, item in enumerate(node):
            yield from _untyped(item, f"{path}[{i}]")
        return
    if not isinstance(node, dict):
        return
    for key in ("properties", "$defs"):
        for name, value in node.get(key, {}).items():
            if isinstance(value, dict) and not _TYPED & value.keys():
                yield f"{path}/{key}/{name}"
            yield from _untyped(value, f"{path}/{key}/{name}")
    for key in ("items", "anyOf"):
        if key in node:
            yield from _untyped(node[key], f"{path}/{key}")


_MODELS = _reply_models()


def test_the_walk_finds_the_reply_models() -> None:
    assert len(_MODELS) > 50


@pytest.mark.parametrize("model", _MODELS, ids=lambda model: f"{model.__module__}.{model.__qualname__}")
def test_every_value_in_the_strict_schema_has_a_type(model: type[StrictOutputModel]) -> None:
    response_format = to_openai_response_format(model, "json_schema")
    assert response_format is not None
    assert list(_untyped(response_format["json_schema"]["schema"], model.__qualname__)) == []
