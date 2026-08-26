"""English prompt for DecisionExtractor.

Placeholders: ``{CONVERSATION_TEXT}`` (uppercase). Rendered via
:func:`everalgo.prompts.render_prompt`, not :py:meth:`str.format` — it mirrors ``.format``'s brace-escape
semantics but leaves an absent placeholder verbatim instead of raising.
Output schema: JSON object ``{"decisions": [{title, decision, reason, impact, tags}, ...]}``.

There is no per-sender placeholder. The extractor runs once for the whole MemCell; ``owner_id`` is
unbound in the DTO and filled later by the caller.

``{language_rule}`` — appearing twice, both copies receiving the same text — is filled from
``user_memory._language.build_language_rule`` according to ``aextract``'s ``output_language`` argument.
"""

DECISION_GENERATION_PROMPT = """
{language_rule}

You extract committed decisions from one conversation slice. A decision is a trade-off that has already
been made, not a summary of the chat, not a prediction of what someone might do later, and not a portrait
of a person.

## What counts as a decision

Extract an item only when the speakers have actually chosen one option over another, in one of these
forms:

1. **Explicit technical choice** — a language, library, runtime, protocol, or tool was selected and the
   alternative was rejected or left behind.
2. **Architectural trade-off** — a structure, boundary, or integration approach was chosen, with a reason
   that constrains later work.
3. **Long-term strategy** — a standing priority or policy was adopted (for example iterating quickly
   rather than optimizing now).

One item is one trade-off. If the same choice is restated, keep a single item. Most slices contain none;
that is the expected result.

## What does not count

Do not extract:

- Options still under discussion, brainstorms, "maybe", "let's try", or a plan that has not been adopted
- One-off operations (ran tests, switched a branch, scheduled a meeting)
- Predictions of future behaviour (that is foresight, not a decision)
- Preferences, identity, or standing traits (that is a profile)
- Isolated verifiable facts with no trade-off (that is an atomic fact)
- Reasons or impacts that do not appear in the conversation — never invent them

If nothing in the slice meets the three forms above, return an empty array. An empty array is a correct
successful answer. Do not invent a decision so that the array is non-empty.

## Output format

Return a JSON object with a "decisions" array. Do not emit owner_id, timestamp, session_id, or parent_id.

{{
  "decisions": [
    {{
      "title": "short name for this trade-off",
      "decision": "what was chosen",
      "reason": "why it was chosen, grounded in the conversation",
      "impact": "what this constrains later, or null if the conversation does not say",
      "tags": ["architecture", "runtime"]
    }}
  ]
}}

`impact` may be null. `tags` is a short list of lowercase labels; use [] when none are obvious.

## Example (extract)

conversation:
```text
[2026-08-24T10:00:00Z] Alice: Core Agent Runtime should stay in Python so we can evolve agent capability quickly. Device-side runtime stays in Rust for stability. Device features will come in through APIs rather than sharing the core runtime.
[2026-08-24T10:01:00Z] Bob: Agreed. Python for the core, Rust on device, API boundary between them.
```

{{
  "decisions": [
    {{
      "title": "Agent Runtime language choice",
      "decision": "Use Python for the core Agent Runtime and Rust for the device-side runtime.",
      "reason": "Python fits fast evolution of agent capability; Rust fits stable device operation.",
      "impact": "Device capabilities connect to the core Agent through APIs.",
      "tags": ["architecture", "runtime"]
    }}
  ]
}}

## Example (empty)

conversation:
```text
[2026-08-24T11:00:00Z] Alice: Tests are green. I'll look at the cache design next week.
[2026-08-24T11:00:30Z] Bob: Sounds good.
```

{{
  "decisions": []
}}

## Input

conversation:
```text
{CONVERSATION_TEXT}
```

{language_rule}

Return the JSON object now. If the conversation has no committed decision, return {{"decisions": []}}.

"""
