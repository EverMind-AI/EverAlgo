"""Tests for ``everalgo.prompts.render``."""

from __future__ import annotations

import pytest

from everalgo.prompts import render_prompt


def test_uses_default_when_prompt_is_none() -> None:
    result = render_prompt("Hello, {name}!", None, name="world")
    assert result == "Hello, world!"


def test_uses_override_when_prompt_is_provided() -> None:
    result = render_prompt("Hello, {name}!", "Hi {name}", name="world")
    assert result == "Hi world"


def test_empty_string_prompt_falls_back_to_default() -> None:
    """An empty string is falsy in Python, so it should fall back to ``default``.

    This matches ``(prompt or default)`` semantics and matters because some callers may pass ``prompt=""`` to
    mean "no override"; we preserve that behaviour rather than rendering an empty template silently.
    """
    result = render_prompt("Hello, {name}!", "", name="world")
    assert result == "Hello, world!"


def test_missing_placeholder_in_fields_leaves_verbatim() -> None:
    """Missing fields leave the placeholder unchanged — no KeyError."""
    result = render_prompt("Hello, {name}!", None)
    assert result == "Hello, {name}!"


def test_extra_fields_are_silently_ignored() -> None:
    """Extra kwargs that the template does not reference are silently dropped."""
    result = render_prompt("Hello, {name}!", None, name="world", unused="extra")
    assert result == "Hello, world!"


def test_default_and_prompt_are_positional_only() -> None:
    """The ``/`` separator forbids passing ``default=`` or ``prompt=`` as kwargs."""
    with pytest.raises(TypeError):
        render_prompt(default="Hello, {name}!", prompt=None, name="world")  # type: ignore[call-arg]


def test_render_prompt_keeps_literal_braces() -> None:
    """Template containing literal JSON-like braces renders without KeyError."""
    out = render_prompt('Reply as {"key": value} for {name}', None, name="Alice")
    assert out == 'Reply as {"key": value} for Alice'


def test_render_prompt_literal_braces_in_value_are_preserved() -> None:
    """Literal braces inside a substituted value are not re-interpreted as placeholders."""
    out = render_prompt("msg: {messages}", None, messages="hello {} world")
    assert out == "msg: hello {} world"


def test_render_prompt_json_example_in_template() -> None:
    """Full JSON example block in the template body is untouched after substitution."""
    template = 'Output JSON like {"key": "value"} for {name}'
    out = render_prompt(template, None, name="Alice")
    assert out == 'Output JSON like {"key": "value"} for Alice'
