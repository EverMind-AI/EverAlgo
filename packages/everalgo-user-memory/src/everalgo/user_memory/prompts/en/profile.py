"""English prompts for ProfileExtractor.

Three prompts cover the three calls: ``PROFILE_INITIAL_EXTRACTION_PROMPT`` builds a profile from a
conversation, ``PROFILE_UPDATE_PROMPT`` emits operations against a stored one, ``PROFILE_COMPACT_PROMPT``
rewrites a profile that has outgrown its item limit. The first two inject a ``{target_user}`` so extraction
is scoped to a single speaker in multi-party conversations; compaction only rewrites stored items.

What a profile IS, and how the two buckets divide, is defined once in ``_PORTRAIT`` and spliced into all
three. Each prompt previously carried its own wording, which is how compaction fell behind: it went on
merging items and refining tags under rules the other two had left behind, undoing their work on the one
path that rewrites everything. A shared definition cannot drift that way.

Placeholders & rendering: all three templates use single-brace placeholders and are rendered via
:func:`everalgo.prompts.render_prompt`, which mirrors :py:meth:`str.format`'s brace-escape semantics — the
JSON examples escape their literal braces as ``{{ }}`` — but leaves an absent placeholder verbatim rather
than raising.

Every rule here answers a measured failure; the numbers are in this package's CHANGELOG. Two properties
are easy to break while tidying:

* **A rule repeated in two places may be load-bearing.** Removing one copy of the "also / as well as /
  comma-list" prohibition as a duplicate raised explicit items from 0.63-0.75 per run to 1.30-1.40, and the
  regression appeared in ``explicit_info``, which the trait rule does not govern. Signals here are not
  partitioned by section.
* **An example beats the rule it contradicts.** ``evidence`` kept naming the speaker in 71/71 items while a
  rule forbade it, because one example demonstrated ``"In Oct 2024 user mentioned..."``. Compaction put
  brackets on 40/40 traits where the other paths put them on 0/44, the only difference being an
  ``(e.g., [Risk-Averse])``. Keep every example consistent with every rule.
* **The naming example has to name the actual subject.** ``_ITEM_RULES`` interpolates ``{target_user}`` rather
  than carrying a name, and a fixed placeholder is not a substitute: replacing the interpolation with a literal
  ``"Alice"`` — a name absent from the conversation — dropped ``implicit_traits`` from 0.95 to 0.00/0.05/0.00 per
  run over three repeats, against a same-version spread of 0.05. Naming the person who is actually speaking is
  what anchors who the portrait is about; a stranger's name reads as another participant.
* **Four rules written against a record of the work all regressed something else.** What the assistant explained,
  what a tool turned out to support, a setting that was changed — these are durable, assert nothing about the
  person, and still reach ``explicit_info`` at roughly 1.4-1.8 items per run on a purely operational corpus. Naming
  the subject matter ("facts about the code, the tooling, the tests") tripled the count on that corpus; stating it
  by form ("a record of the work is not a portrait") dropped the operational-turn check from 20/20 to 8/20; adding
  it to the bucket definition dropped it to 6/20; widening the assistant clause to name what it reported, to 11/20.
  Every attempt raised the total item count, which is the pattern: at this length an added prohibition dilutes the
  ones already here. Do not add a fifth without measuring both directions.
* **Do not hand the model a list of label names.** A six-name vocabulary of dimensions (role / languages &
  tools / process / …) was added to stop one fact drawing a fresh category name each run, qualified with
  "written in the output language". All 20 runs came back with the English labels verbatim under
  ``output_language=CHINESE`` — ``[role]``, ``[process]`` — and 10/20 chained two dimensions into one
  description under the borrowed label. Bold literals get copied, not translated. Reverted.
"""

# --------------------------------------------------------------------------------------------------
# Shared blocks. Spliced into all three prompts so a rule cannot hold on one path and not another.
# --------------------------------------------------------------------------------------------------

_PORTRAIT = """\
A profile is a **portrait of a person**, not a log of what happened. Nothing enters it because it occurred — only because it is still true the next time you meet this person. A long, busy conversation can correctly yield nothing at all.

**explicit_info** — what this person **stated** about themselves: their role, the constraints they work under, their environment, the preferences they voiced. One clear statement is enough. It must also still hold next time: "I'm tired today", "I'm in a hurry right now", "I'm annoyed at this bug", "I'm on leave next week" were genuinely said and do not belong in a portrait, and there is nowhere to record when they expire. Anything you would have to infer from what they did does not belong here at all: watching an action tells you the action happened, not that this person holds a standing property. A single operation is an event, not a portrait fact.

**implicit_traits** — the half of the portrait they did not state outright: how they decide, what they insist on, what they avoid. A conversation with any substance should not leave this empty. Each trait needs **two or more signals**, and a signal is something this person **chose or asserted** — never an operation they carried out. Switching a branch, running a test, fixing an import are actions; two actions are not two signals about who someone is. The two need not share a topic — one disposition can show through on unrelated subjects — but the same statement twice is one signal, not two.

**Which bucket an item goes in is decided by whether this person said it, never by what the fact is about.** Anything traceable to a sentence they uttered is explicit_info — including how their team requires work to be done, what they refuse to touch, what they insist you do — however much it also reveals about them. implicit_traits holds only what no sentence of theirs states. Rephrasing a statement as a disposition ("insists on the team's merge process") does not move it across: it is still the thing they said, and it belongs in explicit_info once, not in both buckets.

**Never restate an action as a capability, duty, skill, familiarity, interest or attitude.** Having run the tests once is not being responsible for testing, and it is not caring about test results or an interest in testing either; asking what a skipped test was is a question, not an attitude towards testing; reading a log is not attentiveness. Do not call this person responsible for, involved in, capable of, familiar with, interested in, attentive to or concerned with anything unless they said so themselves. This is about the claim, not the wording, and holds in every language. A pattern you inferred from behaviour goes to implicit_traits, not to explicit_info reworded.

**A question records nothing but the question, and being told something is not knowing it.** Whatever the assistant explained is the assistant's contribution to the conversation; it never becomes this person's knowledge, familiarity or concern, however specifically they asked for it. Wanting to know why a test was skipped, and being given the reason, leaves the portrait exactly as it was.

An item already on file making such a claim is **not** rescued by the evidence attached to it. That evidence records an occasion; the claim asserts a standing property; an occasion cannot establish one. Such an item was never a portrait item and is to be removed, not rewritten and not merged."""

_ITEM_RULES = """\
**One item per dimension.** A category names a durable dimension of the person, not a kind of activity, and that dimension gets exactly one item whose description covers the whole of it. "works mainly in Python, some Rust, refuses front-end JavaScript" is ONE item, not three. A fact about a dimension already on file extends that item; a dimension not yet on file gets its own. Reuse a category or trait name already in use rather than coining a near-synonym — two names for one dimension is how a duplicate gets in. Dimensions are coarse: which languages someone uses, what their team's process requires, which tools they work on are each one dimension, not one per language, rule or tool. When unsure whether a dimension is new, it is not.

**One item, one dimension.** A description states that one dimension and nothing else. Never chain unrelated dimensions together with "also", "as well as" or a comma-list under a catch-all label — that is a pile of events wearing one label, no better than adding them separately. Where neither belongs in a portrait, write nothing.

**Never name the subject.** A profile is read as being about this one person, so state each item directly: "Works mainly in Python", never "{target_user} works mainly in Python", "the user works mainly in Python" or "{target_user_id} works mainly in Python". This holds in "evidence" too — give what was said and when, not who said it."""

_TARGET_USER = """\
**TARGET USER: {target_user}**
This may be a multi-speaker conversation; each line is tagged with the speaker's ``user_id``. {verb} only for the speaker whose ``user_id`` equals ``{target_user_id}``; anything stated by or about another participant belongs to THAT participant. Do not treat assistant suggestions as this person's traits. The label and id above only locate their lines — neither appears in your output."""

_ITEM_SHAPE = """\
An explicit_info item is {{"category", "description", "evidence"}}; an implicit_traits item is {{"trait", "description", "basis", "evidence"}}. Keep each description to one natural sentence. "evidence" gives when something was said and quotes it, naming no one — e.g. "2024-10-03: '...'". "basis" names the signals themselves — the choices or assertions you are reading the disposition from, each one findable in the conversation. Restating the requirement ("two or more signals", "multiple instances", "repeated choices") is not a basis; if you cannot name the signals, the trait does not belong here at all."""


# --------------------------------------------------------------------------------------------------
# INIT — build a profile from a conversation.
# --------------------------------------------------------------------------------------------------

PROFILE_INITIAL_EXTRACTION_PROMPT = (
    """
{language_rule}

You are a "User Profile Analyst". Read the conversation below and build a user profile.

"""
    + _PORTRAIT
    + "\n\n"
    + _ITEM_RULES
    + "\n\n"
    + _TARGET_USER.replace("{verb}", "Build a profile")
    + """

【Output】
Either list may be empty, and often should be. A conversation made only of operations — requests, commands, tool output, the assistant's answers — with nothing this person said about themselves yields `"explicit_info": []`. That IS the correct answer there; inventing a fact to fill it is the error. Do not treat producing items as the goal.

"""
    + _ITEM_SHAPE
    + """

Output JSON directly, nothing else:
```json
{{"explicit_info": [{{"category": "...", "description": "...", "evidence": "..."}}],
  "implicit_traits": [{{"trait": "...", "description": "...", "basis": "...", "evidence": "..."}}]}}
```

{language_rule}

【Conversation】
{conversation_text}"""
)


# --------------------------------------------------------------------------------------------------
# UPDATE — emit operations against a stored profile.
# --------------------------------------------------------------------------------------------------

PROFILE_UPDATE_PROMPT = (
    """
{language_rule}

You are a user profile updater. Decide which operations to perform on the stored profile.

"""
    + _PORTRAIT
    + "\n\n"
    + _ITEM_RULES
    + "\n\n"
    + _TARGET_USER.replace("{verb}", "Update the profile")
    + """

【Stored Profile】(each item carries an index)
{current_profile}

【Conversation Records】
{conversations}

【Operations】
- **update**: an existing item gains a correction, a supplement or fresh evidence. **This is the operation for a fresh occurrence of something already recorded** — more evidence for a trait you have, a fact that sharpens an existing one, or a statement that contradicts it. Reaching for "add" here is the most common error on this task.
- **add**: a dimension not yet on file at all. If a stored item already covers the dimension, use "update" — a second item under an existing category or trait name is always wrong.
- **delete**: this person explicitly negates it; it has expired ("traveling next week", already past); it is too trivial to be a portrait ("wants pizza today"); a new statement contradicts it outright; **or the stored item asserts a duty, capability, skill, familiarity, interest or attitude this person never claimed** — delete those on sight, whatever evidence they carry, and do not settle for rewriting them.
- **none**: no operation needed — often the right answer. A conversation made only of operations contains nothing about who this person is, however much of it there is. "none" is complete and correct; manufacturing an operation is not.

【Rules】
1. **Index semantics**: every index resolves against the profile snapshot shown above, numbered exactly as it appears there. Operations within one response never shift each other's indices — do not adjust an index to compensate for another operation in the same list. "add" takes no index. Indices for explicit_info and implicit_traits are independent.
2. **Keep an item internally consistent**: fields you omit from an "update" keep their stored values, so when you rewrite a "description" carry its matching "evidence" in the same operation — otherwise the item asserts one thing while quoting the opposite.
3. Before "add", read the names already in use above. If a similar item exists under any wording, "update" it instead.

"""
    + _ITEM_SHAPE
    + """

【Output】
Nothing to do:
```json
{{
  "operations": [
    {{"action": "none"}}
  ],
  "update_note": "conversation contains nothing about this person"
}}
```
One or more operations:
```json
{{
  "operations": [
    {{"action": "add", "type": "explicit_info", "data": {{"category": "...", "description": "...", "evidence": "..."}}}},
    {{"action": "add", "type": "implicit_traits", "data": {{"trait": "...", "description": "...", "basis": "...", "evidence": "..."}}}},
    {{"action": "update", "type": "explicit_info", "index": 0, "data": {{"description": "...", "evidence": "..."}}}},
    {{"action": "delete", "type": "implicit_traits", "index": 1, "reason": "..."}}
  ],
  "update_note": "..."
}}
```
Every operation object closes with exactly two braces before the comma: one for "data", one for the operation itself.

{language_rule}"""
)


# --------------------------------------------------------------------------------------------------
# COMPACT — rewrite a profile that has outgrown its limit.
# --------------------------------------------------------------------------------------------------

PROFILE_COMPACT_PROMPT = (
    """
{language_rule}

The stored profile holds {total_items} items (explicit_info + implicit_traits combined), over the limit of {max_items}. Rewrite it down to **{max_items} items TOTAL** across both lists — not {max_items} each.

"""
    + _PORTRAIT
    + "\n\n"
    + _ITEM_RULES
    + """

【How to compact】Work in this order.
1. **Delete everything that was never a portrait item, before merging anything.** An action restated as a capability, duty, skill, familiarity, interest or attitude ("responsible for testing", "capable of debugging", "involved in version control", "cares about test results", "attentive to CI"); a passing state; anything trivial or already expired. Delete these outright — do NOT merge them: folding twelve "responsible for testing" entries into one still leaves a claim this person never made, and it now looks well-founded. Being numerous is not evidence, and neither is the evidence line attached to each one: it records an occasion, the claim asserts a standing property. Twelve such items become **zero** items, not one.
2. **Move anything misfiled.** A trait that merely restates something this person said is an explicit_info fact — move it into the dimension covering it, and do not leave a copy behind in implicit_traits.
3. **Then collapse each remaining dimension into its one item.** Several items under one category, or under near-synonymous categories, are one dimension that was allowed to split — merge those under the name already in use.
4. **Keep the evidence** of every item you keep, and keep it naming no one.
5. Prefer deleting a weak item over shortening a strong one. Reaching {max_items} by trimming good descriptions is the wrong trade. Ending well under {max_items} is fine — the limit is a ceiling, not a target.

【Stored Profile】
{profile_text}

"""
    + _ITEM_SHAPE
    + """

【Output】
explicit_info + implicit_traits must total no more than {max_items} items.
```json
{{"explicit_info": [{{"category": "...", "description": "...", "evidence": "..."}}],
  "implicit_traits": [{{"trait": "...", "description": "...", "basis": "...", "evidence": "..."}}],
  "compact_note": "what was merged and what was dropped"}}
```

{language_rule}"""
)
