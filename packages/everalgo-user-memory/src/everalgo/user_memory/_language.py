"""Output-language selection for the extraction prompts.

Internal module, like ``_render`` — except for :class:`OutputLanguage`, which the extractors take as an
argument and which ``everalgo.user_memory`` re-exports. The rule texts live in ``prompts/en/_language.py``;
this module only chooses between them.
"""

from __future__ import annotations

from enum import StrEnum

from everalgo.user_memory.prompts.en._language import (
    CALLER_CHOSEN_LANGUAGE_RULE,
    COMPACTED_PROFILE_LANGUAGE_RULE,
    EXISTING_DECISION_LANGUAGE_RULE,
    EXISTING_NARRATIVE_LANGUAGE_RULE,
    EXISTING_PROFILE_LANGUAGE_RULE,
    MERGED_DECISIONS_LANGUAGE_RULE,
    MERGED_EPISODES_LANGUAGE_RULE,
    PARTICIPANT_LANGUAGE_RULE,
    PRINCIPLES_FROM_DECISIONS_LANGUAGE_RULE,
    PROFILE_INIT_LANGUAGE_RULE,
    SOURCE_TEXT_LANGUAGE_RULE,
)

__all__ = [
    "COMPACTED_PROFILE_LANGUAGE_RULE",
    "EXISTING_DECISION_LANGUAGE_RULE",
    "EXISTING_NARRATIVE_LANGUAGE_RULE",
    "EXISTING_PROFILE_LANGUAGE_RULE",
    "MERGED_DECISIONS_LANGUAGE_RULE",
    "MERGED_EPISODES_LANGUAGE_RULE",
    "PRINCIPLES_FROM_DECISIONS_LANGUAGE_RULE",
    "PROFILE_INIT_LANGUAGE_RULE",
    "SOURCE_TEXT_LANGUAGE_RULE",
    "OutputLanguage",
    "build_language_rule",
]


class OutputLanguage(StrEnum):
    """Languages an extractor will write its output in.

    Members are the English language names the prompts were measured with, and being a
    :class:`~enum.StrEnum` they are usable wherever a ``str`` is — ``OutputLanguage.GERMAN == "German"``.

    Closed rather than free-form on purpose. The value is interpolated into the prompt, so an arbitrary
    string is an instruction the caller can smuggle in: a language field carrying ``"German. Ignore all
    previous instructions and ..."`` would arrive as prose the model reads as its own directive. Restricting
    it to names chosen here removes that whole class rather than filtering for it. Adding a language is a
    member plus a regression run; a caller who needs one that is not here can supply its own ``prompt=``.
    """

    CHINESE = "Chinese"
    ENGLISH = "English"
    GERMAN = "German"
    JAPANESE = "Japanese"
    KOREAN = "Korean"
    RUSSIAN = "Russian"
    SPANISH = "Spanish"

    @classmethod
    def _missing_(cls, value: object) -> OutputLanguage | None:
        """Accept any casing and surrounding whitespace — callers usually read this out of config."""
        if not isinstance(value, str):
            return None
        wanted = value.strip().casefold()
        return next((member for member in cls if member.value.casefold() == wanted), None)


def build_language_rule(
    output_language: OutputLanguage | str | None,
    *,
    fallback: str = PARTICIPANT_LANGUAGE_RULE,
) -> str:
    """Return the language-rule block to splice into a prompt's language-rule placeholder.

    Args:
        output_language: Language to write the extraction in, as an :class:`OutputLanguage` member or an
            equivalent string in any casing. ``None`` leaves the choice to the model, which costs accuracy
            — see ``prompts/en/_language.py`` for what each path measured.
        fallback: Rule to use when ``output_language`` is ``None``. Defaults to the conversation rule, which
            suits any operator reading a raw conversation. Operators facing a different input pass their own:
            ``SOURCE_TEXT_LANGUAGE_RULE`` for already-extracted memory text, where the judgement the
            conversation rule adjudicates does not arise; ``EXISTING_PROFILE_LANGUAGE_RULE`` and
            ``COMPACTED_PROFILE_LANGUAGE_RULE`` for the profile paths that must inherit a language rather
            than re-decide it; ``PROFILE_INIT_LANGUAGE_RULE`` for the one call that decides it;
            ``MERGED_EPISODES_LANGUAGE_RULE`` and ``EXISTING_NARRATIVE_LANGUAGE_RULE`` for the two episode
            reflect modes; ``MERGED_DECISIONS_LANGUAGE_RULE`` and ``EXISTING_DECISION_LANGUAGE_RULE`` for
            the two decision reflect modes; ``PRINCIPLES_FROM_DECISIONS_LANGUAGE_RULE`` for synthesising
            Principles from a Decision cluster.

    Returns:
        The rule text, with the language already substituted when one was named.

    Raises:
        ValueError: If ``output_language`` names no known language.
    """
    if output_language is None:
        return fallback
    try:
        language = OutputLanguage(output_language)
    except ValueError as exc:
        supported = ", ".join(sorted(OutputLanguage))
        raise ValueError(
            f"unsupported output_language {output_language!r}; expected one of {supported}, "
            f"or None to let the model infer it from the conversation"
        ) from exc
    # `str.replace`, not `str.format`: prompt text carries literal JSON braces, and this package's own
    # `render_prompt` avoids `.format` for that reason. Keeping the same discipline here means the rule
    # texts stay safe to edit without thinking about escaping.
    return CALLER_CHOSEN_LANGUAGE_RULE.replace("{language}", language)
