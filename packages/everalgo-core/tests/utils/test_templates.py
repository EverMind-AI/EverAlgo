"""Tests for ``everalgo.utils.templates``."""

from __future__ import annotations

import pytest

from everalgo.utils import render_prompt


def test_uses_default_when_prompt_is_none() -> None:
    result = render_prompt("Hello, {{name}}!", None, name="world")
    assert result == "Hello, world!"


def test_uses_override_when_prompt_is_provided() -> None:
    result = render_prompt("Hello, {{name}}!", "Hi {{name}}", name="world")
    assert result == "Hi world"


def test_empty_string_prompt_falls_back_to_default() -> None:
    """An empty string is falsy in Python, so it should fall back to ``default``.

    This matches ``(prompt or default)`` semantics and matters because some callers may pass ``prompt=""`` to
    mean "no override"; we preserve that behaviour rather than rendering an empty template silently.
    """
    result = render_prompt("Hello, {{name}}!", "", name="world")
    assert result == "Hello, world!"


def test_placeholder_without_a_field_raises() -> None:
    """A placeholder nobody supplied a value for is an authoring bug, not a prompt to ship."""
    with pytest.raises(ValueError, match=r"\{\{name\}\} \(no value supplied\)"):
        render_prompt("Hello, {{name}}!", None)


def test_empty_string_field_is_substituted() -> None:
    """``""`` is a legitimate value (an empty ``{{pass_rule}}``, say): it renders, it does not raise."""
    result = render_prompt("Rule: {{pass_rule}}.", None, pass_rule="")
    assert result == "Rule: ."


def test_none_field_named_by_template_raises() -> None:
    """``str(None)`` would write a literal ``None`` into the prompt — never the author's intent."""
    with pytest.raises(ValueError, match=r"\{\{name\}\} \(value is None\)"):
        render_prompt("Hello, {{name}}!", None, name=None)


def test_none_field_not_named_by_template_is_ignored() -> None:
    """The ``None`` check is scoped to placeholders the template actually names."""
    result = render_prompt("Hello, {{name}}!", None, name="world", unused=None)
    assert result == "Hello, world!"


def test_single_brace_placeholder_raises_and_names_the_new_syntax() -> None:
    """``{name}`` is what a placeholder looked like before the migration; it must not render.

    Rendering it would ship the word ``{name}`` to the model where a profile belonged — the
    quietest possible failure — so the renderer refuses and spells out the replacement.
    """
    with pytest.raises(ValueError, match=r"single-brace placeholder syntax.*\{name\} \(write \{\{name\}\}\)"):
        render_prompt("Hello, {name}!", None, name="world")


def test_single_brace_guard_names_every_offender_once() -> None:
    """A half-migrated template usually has several slots; one run has to list them all."""
    with pytest.raises(ValueError) as excinfo:
        render_prompt("{language_rule}\n{conversation}\n{language_rule}", None)
    message = str(excinfo.value)
    assert "{language_rule} (write {{language_rule}})" in message
    assert "{conversation} (write {{conversation}})" in message
    assert message.count("{language_rule} (write") == 1


def test_single_brace_guard_ignores_a_well_formed_placeholder() -> None:
    """``{{name}}`` contains the ``{name}`` shape; reading it as legacy syntax would reject everything."""
    assert render_prompt("Hi {{name}}", None, name="world") == "Hi world"


@pytest.mark.parametrize(
    "template",
    ['{"key": "value"}', "a lone {", "{ spaced }", "{123}", "{}", "{kebab-case}", "{"],
)
def test_non_placeholder_brace_shapes_stay_verbatim(template: str) -> None:
    """Only ``{{identifier}}`` is a placeholder; a brace shape that never was one is literal text."""
    assert render_prompt(template, None) == template


def test_json_example_needs_no_escaping() -> None:
    """The point of the double-brace syntax: a JSON example is written as the model should read it."""
    template = 'Output JSON like {"key": "value", "nested": {"id": 1}} for {{name}}'
    out = render_prompt(template, None, name="Alice")
    assert out == 'Output JSON like {"key": "value", "nested": {"id": 1}} for Alice'


def test_placeholder_shape_inside_a_value_is_not_rescanned() -> None:
    """A ``{{word}}`` inside a substituted value is user data, not a placeholder to fill.

    Every caller injects user content (chat messages, episodes, profiles), so a second pass would
    let anyone typing braces into a conversation reach for another field's text.
    """
    result = render_prompt("msg: {{messages}}", None, messages="use {{placeholder}} here")
    assert result == "msg: use {{placeholder}} here"


def test_single_brace_shape_inside_a_value_does_not_trip_the_guard() -> None:
    """The legacy guard reads the template only — a user writing ``{name}`` must not break a call."""
    result = render_prompt("msg: {{messages}}", None, messages="the config uses {name} here")
    assert result == "msg: the config uses {name} here"


def test_extra_fields_are_silently_ignored() -> None:
    """Extra kwargs that the template does not reference are silently dropped."""
    result = render_prompt("Hello, {{name}}!", None, name="world", unused="extra")
    assert result == "Hello, world!"


def test_default_and_prompt_are_positional_only() -> None:
    """The ``/`` separator forbids passing ``default=`` or ``prompt=`` as kwargs."""
    with pytest.raises(TypeError):
        render_prompt(default="Hello, {{name}}!", prompt=None, name="world")  # type: ignore[call-arg]


def test_literal_braces_in_a_value_are_preserved() -> None:
    """Braces inside a substituted value are inert, whichever shape they take."""
    out = render_prompt("msg: {{messages}}", None, messages="hello {} world {{ and }} tail")
    assert out == "msg: hello {} world {{ and }} tail"


def test_placeholder_in_one_value_is_not_filled_from_another_field() -> None:
    """A user-typed ``{{pass_rule}}`` in narrative text must not pull the real rule body in.

    Every caller passes user content as field values, so a second substitution pass would hand
    users the prompt's structure: typing ``{{pass_rule}}`` into a conversation would splice the
    historical-pass rule into the middle of the narrative data block.
    """
    template = (
        "<episode_narratives>\n{{episode_texts}}\n</episode_narratives>\n"
        "<cats>{{available_categories}}</cats>{{pass_rule}}"
    )
    narrative = "Zaphod said the config uses {{pass_rule}} and asked about {{available_categories}}."
    result = render_prompt(
        template,
        None,
        episode_texts=narrative,
        available_categories='["identity"]',
        pass_rule="HISTORICAL PASS RULE: only pre-cutoff facts.",
    )
    assert result == (
        f"<episode_narratives>\n{narrative}\n</episode_narratives>\n"
        '<cats>["identity"]</cats>HISTORICAL PASS RULE: only pre-cutoff facts.'
    )


def test_render_is_independent_of_field_order() -> None:
    """Single-pass substitution makes the result a function of the field set, not its order."""
    template = "{{alpha}}|{{beta}}"
    forward = render_prompt(template, None, alpha="mentions {{beta}}", beta="mentions {{alpha}}")
    reverse = render_prompt(template, None, beta="mentions {{alpha}}", alpha="mentions {{beta}}")
    assert forward == "mentions {{beta}}|mentions {{alpha}}"
    assert forward == reverse


def test_regex_escapes_in_a_value_are_preserved() -> None:
    r"""Substitution uses ``re.sub``'s callback form, so ``\1`` / ``\g<0>`` stay verbatim.

    The replacement-string form would read them as backreferences — expanding ``\g<0>`` to the
    whole ``{{payload}}`` match and ``\1`` to the placeholder name — mangling any value that
    legitimately contains backslashes.
    """
    value = r"path C:\1 and \g<0> plus \\ tail"
    result = render_prompt("value: {{payload}}", None, payload=value)
    assert result == f"value: {value}"
