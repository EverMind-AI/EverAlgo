"""Render a prompt template with caller-provided fields, falling back to a default.

Centralises the ``(prompt or DEFAULT).format(**fields)`` pattern used by every LLM-calling operator in
EverAlgo, so that future cross-cutting concerns (prompt logging, escape rules, i18n switching,
instrumentation) have a single edit point.
"""

from __future__ import annotations

from typing import Any


def render_prompt(default: str, prompt: str | None, /, **fields: Any) -> str:
    """Render ``prompt`` with ``fields``; if ``prompt`` is None, use ``default``.

    Designed for operators that accept an optional caller-override of the prompt template while still
    shipping a sensible default. ``default`` and ``prompt`` are positional-only so callers cannot accidentally
    swap their order via keyword arguments.

    Parameters
    ----------
    default : str
        Module-level prompt constant shipped with the operator (for example ``CHAT_BOUNDARY_DETECT_PROMPT_EN``).
    prompt : str or None
        Caller override; if ``None``, falls back to ``default``.
    **fields : Any
        Keyword arguments substituted into the template via :py:meth:`str.format`. The template is responsible
        for naming each placeholder; missing placeholders raise :class:`KeyError`.

    Returns
    -------
    str
        The rendered prompt string, ready to send to the LLM.

    Raises
    ------
    KeyError
        If the template references a placeholder not in ``fields``.

    Examples
    --------
    >>> render_prompt("Hello, {name}!", None, name="world")
    'Hello, world!'
    >>> render_prompt("Hello, {name}!", "Hi {name}", name="world")
    'Hi world'
    """
    return (prompt or default).format(**fields)


def render_prompt_replace(default: str, prompt: str | None, /, replacements: dict[str, str]) -> str:
    """Render ``prompt`` with ``replacements`` via :py:meth:`str.replace`; fall back to ``default``.

    Sister of :func:`render_prompt`. Use this when the prompt body contains unescaped ``{`` / ``}`` (e.g.,
    JSON examples in the body that would otherwise break ``str.format``). The caller passes a mapping from
    the **exact** placeholder text (e.g., ``"{conversation}"`` or ``"{{EPISODE_TEXT}}"``) to its substitute
    string — so the same helper handles both opensource ``profile`` conventions (single-brace placeholders
    coexisting with literal JSON braces) and opensource ``event_log`` conventions (double-brace
    placeholders).

    Mirrors opensource ``build_profile_prompt`` in
    ``memory_layer/memory_extractor/profile_memory/conversation.py:410-430``, which chains ``.replace`` calls
    for each placeholder.

    Parameters
    ----------
    default : str
        Module-level prompt constant.
    prompt : str or None
        Caller override; if ``None``, falls back to ``default``.
    replacements : dict[str, str]
        Map from placeholder text (verbatim, including any braces) to replacement value.

    Returns
    -------
    str
        Rendered prompt string.

    Examples
    --------
    >>> render_prompt_replace(
    ...     'Hi {name}, here is JSON: {"key": 1}',
    ...     None,
    ...     {"{name}": "Alice"},
    ... )
    'Hi Alice, here is JSON: {"key": 1}'
    >>> render_prompt_replace(
    ...     "TEXT={{TEXT}}",
    ...     None,
    ...     {"{{TEXT}}": "hello"},
    ... )
    'TEXT=hello'
    """
    template = prompt or default
    for placeholder, value in replacements.items():
        template = template.replace(placeholder, value)
    return template
