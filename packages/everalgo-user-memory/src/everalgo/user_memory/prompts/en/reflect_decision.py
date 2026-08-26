"""English prompts for DecisionReflector.

Constants:
    - ``REFLECT_DECISION_PROMPT`` — full merge from N chronological decisions. Placeholder: ``{timeline}``.
    - ``REFLECT_DECISION_UPDATE_PROMPT`` — incremental update of an existing Decision.
      Placeholders: ``{old_decision}`` / ``{new_decisions}``.

Output schema (both variants): ``{decision, reason, title, impact, tags}`` via Structured Output.
``decision`` / ``reason`` sit before ``title`` because Structured Output generates fields in schema
order, and a title written before the trade-off it names would be labelling nothing.

``{language_rule}`` — appearing twice in each prompt, both copies receiving the same text — is filled from
``user_memory._language.build_language_rule`` according to ``areflect``'s ``output_language`` argument.
Left unset, both variants inherit the language of their input rather than judging it: the mixed-input
judgement belongs to ``DecisionExtractor`` (see ``prompts/en/decision.py``). Merging decisions that
disagree on language still leaves the model to pick one, so a caller who cannot guarantee they agree
should name the language.
"""

REFLECT_DECISION_PROMPT = """\
{language_rule}

You merge already-extracted decisions about the same trade-off into one current Decision.

Each item is one committed choice at a point in time. You can now see the full timeline and produce the
Decision that is currently in force — more accurate than any single earlier item.

Return exactly one Decision (title, decision, reason, impact, tags):
- Keep the latest choice when items contradict; earlier choices are history, not a second Decision
- Preserve reasons and impacts that still hold; note in reason/impact when an earlier option was abandoned, without inventing facts
- Do not generalise into an engineering principle or standing policy ("always prefer X"). That is a different operator. This call only answers what the current trade-off is
- Do not invent a reason or impact that no input states
- impact may be null; tags is a short list of lowercase labels, or []

{language_rule}

Decisions (chronological):
{timeline}"""

REFLECT_DECISION_UPDATE_PROMPT = """\
{language_rule}

You are updating an existing Decision with newer decisions about the same trade-off.

Current Decision:
{old_decision}

New decisions (chronologically ordered):
{new_decisions}

Update the Decision to incorporate the new information:
- Correct any choice that is now outdated; the latest item wins contradictions
- Preserve parts that are still accurate
- Note abandoned options in reason/impact only when the inputs support it; do not invent facts
- Do not generalise into an engineering principle or standing policy. Return one current Decision
- impact may be null; tags is a short list of lowercase labels, or []

{language_rule}"""
