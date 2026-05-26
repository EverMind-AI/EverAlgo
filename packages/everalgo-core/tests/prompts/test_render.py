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


def test_render_prompt_collapses_double_braces() -> None:
    """``{{ }}`` collapses to ``{ }`` — mirrors ``str.format`` brace-escape semantics.

    Prompts authored against evercore (e.g. ``EPISODE_GENERATION_PROMPT``) write JSON examples as
    ``{{...}}`` because evercore uses ``str.format``. Without this collapse the LLM sees raw
    double braces and downstream JSON parsers fail.
    """
    template = 'Reply with {{"label": "CORRECT"}} for question {name}'
    out = render_prompt(template, None, name="Q1")
    assert out == 'Reply with {"label": "CORRECT"} for question Q1'


def test_render_prompt_matches_str_format_for_escaped_template() -> None:
    """Output is byte-for-byte identical to ``str.format`` for any template that uses ``{{ }}``."""
    template = 'Schema example:\n{{\n    "title": "{title}",\n    "items": [{{"id": 1}}]\n}}\nEnd.'
    rendered = render_prompt(template, None, title="Demo")
    expected = template.format(title="Demo")
    assert rendered == expected


def test_render_prompt_nested_double_braces_collapse_once() -> None:
    """``{{{{`` collapses to ``{{`` (one nesting level), matching ``str.format``."""
    template = "{{{{nested}}}}"
    rendered = render_prompt(template, None)
    assert rendered == "{{nested}}"
    assert rendered == template.format()


def test_render_prompt_double_braces_inside_value_are_preserved() -> None:
    """``{{`` inside a substituted value is NOT collapsed — only template-level escapes are.

    Matches ``str.format`` semantics: brace-escape is a template-time syntax, not a value-time
    transform. Once a value is injected, its braces are inert.
    """
    out = render_prompt("envelope: {payload}", None, payload="{{still doubled}}")
    assert out == "envelope: {{still doubled}}"
