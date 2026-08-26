"""English prompt for PrincipleExtractor.

Placeholders: ``{DECISION_CLUSTER}``. Rendered via :func:`everalgo.prompts.render_prompt`, not
:py:meth:`str.format`. Output schema: JSON object
``{"principles": [{title, statement, source_entry_ids}, ...]}``.

``source_entry_ids`` must be a subset of the ``id=`` values shown in the cluster. The extractor does
not invent storage ids.

``{language_rule}`` — appearing twice, both copies receiving the same text — is filled from
``user_memory._language.build_language_rule`` according to ``aextract``'s ``output_language`` argument.
Left unset, synthesis inherits the language of the decisions rather than judging a conversation.
"""

PRINCIPLE_GENERATION_PROMPT = """\
{language_rule}

You synthesise engineering principles from one cluster of already-extracted decisions. A principle is an
abstraction that would still apply to a later, similar trade-off. It is not a restatement of one Decision,
and it is not a merge of those Decisions into the currently chosen option (that is a different operator).

## What to emit

- Each item is one standing principle: a short ``title`` and a ``statement`` a future reader can apply
- ``source_entry_ids`` lists the input ``id=`` values that ground that statement — only those ids, at least one
- One dimension per principle; do not split one idea into near-synonym items
- Prefer an empty array to a principle that merely copies "use Python" from a single Decision

## What not to emit

- The current choice in a time series ("we now use an in-house runtime") — that is a Decision
- Facts, portraits, or predictions
- Statements the cluster does not support
- ``source_entry_ids`` that do not appear as ``id=`` in the input

If nothing in the cluster abstracts into a standing principle, return {{"principles": []}}. An empty array
is a correct successful answer.

## Output format

{{
  "principles": [
    {{
      "title": "Iteration over premature optimisation",
      "statement": "Agent architecture prioritises iteration speed.",
      "source_entry_ids": ["dc_001", "dc_002", "dc_003"]
    }}
  ]
}}

## Example

cluster:
```text
1. id=dc_001
   Title: Agent Core language
   Decision: Use Python for the core Agent Runtime.
   Reason: Faster iteration on agent capability.
   Impact: Device runtime stays separate.
   Tags: architecture, runtime

2. id=dc_002
   Title: Experiment velocity
   Decision: Prefer shipping experiments quickly over hardening the stack now.
   Reason: The agent surface is still changing weekly.
   Impact: (none)
   Tags: process

3. id=dc_003
   Title: Defer Rust on the core
   Decision: Do not rewrite the core Agent Runtime in Rust yet.
   Reason: Premature optimisation would slow iteration.
   Impact: (none)
   Tags: runtime
```

{{
  "principles": [
    {{
      "title": "Iteration over premature optimisation",
      "statement": "Agent architecture prioritises iteration speed.",
      "source_entry_ids": ["dc_001", "dc_002", "dc_003"]
    }}
  ]
}}

## Input

cluster:
```text
{DECISION_CLUSTER}
```

{language_rule}

Return the JSON object now. If the cluster has no standing principle, return {{"principles": []}}.
"""
