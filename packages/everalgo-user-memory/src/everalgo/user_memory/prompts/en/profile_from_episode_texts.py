"""English prompts for ProfileExtractor's Episode-text path.

These prompts deliberately mirror the current MemCell Profile prompts where their semantics match, while
keeping Episode-specific grounding and category selection isolated across all four stages. The MemCell prompts
remain separate so changing one input contract cannot silently alter the other.
"""

from everalgo.user_memory.prompts.en.profile import _PORTRAIT

_ITEM_RULES = """\
**One fact, one item.** An item states a single atomic fact about this person, in one or two short sentences — plain, direct and easy to read. Never chain separate facts together with "also", "as well as" or a comma-list: "prefers small pull requests, insists on rebase-before-merge, and reviews are done in the morning" is THREE items, not one. A statement that genuinely forms one fact stays one item — "works mainly in Python with some Rust" is one fact about their language stack — but the moment a second independent fact appears, it gets its own item.

**A category groups items; it never merges them.** A category names a durable semantic dimension of the person, and several atomic facts may share one category. Name the dimension of the FACT itself, never the topic of the Episode where it surfaced. Do not broaden a category until unrelated dimensions fit under it, and do not change a fact's meaning to fit a category.

**Never name the subject in a description.** A profile is read as being about this one person, so write every description as a subjectless declarative sentence: "Works mainly in Python", never "{target_user} works mainly in Python", "the user works mainly in Python" or the imperative "Use Python". Evidence and basis are source-grounding fields, so their excerpts may retain names exactly as the Episode narrative wrote them."""

_SOURCE_RULES = """\
The source records are third-person Episode narratives rather than raw conversation turns. In the profile rules below, "conversation" means the supplied Episode narratives. Treat a fact as stated only when a narrative explicitly attributes that fact, preference or constraint to {target_user}; a quotation mark is not required. An action alone remains an event, not an explicit_info fact."""

_ITEM_SHAPE = """\
An explicit_info item is {{"category", "description", "evidence"}}; an implicit_traits item is {{"trait", "description", "basis"}} — no evidence field, its grounding is the basis. Keep each description to one or two short subjectless declarative sentences — concise and plain, never imperative; if a description needs "also" or a semicolon to hold together, it is more than one item. "evidence" and "basis" are JSON strings, never arrays; when either carries two grounds, combine them inside that one string. "evidence" contains at most two verifiable narrative excerpts or faithful paraphrases from the supplied Episode narratives. A paraphrase keeps the narrative's attribution: never turn it into a direct user quotation unless the source itself contains that quotation, and never invent wording, dates, speaker attribution or Episode identifiers. "basis" names the signals themselves — the choices or assertions you are reading the disposition from, each one findable in the Episode narratives and faithfully summarised when not excerpted, with no fabricated user quotation. Restating the requirement ("one clear signal", "multiple instances", "repeated choices") is not a basis; if you cannot name the signals, the trait does not belong here at all."""

_CATEGORY_RULES = """\
**Apply the available-category rule only to explicit_info.category.** For every explicit fact you output or update, determine its semantic dimension from that fact's own meaning, then choose the most accurate matching category from the 【Available Categories】 section. Never sacrifice category accuracy merely to reuse a listed category or reduce the number of categories. If the list is empty or no listed category accurately fits, create a necessary, concise and semantically accurate category. The list is not a whitelist. It does not constrain implicit_traits.trait; name each trait from the disposition it actually describes."""

_EPISODE_PRIORITY_RULES = """\
**Absolute highest priorities: factual correctness and category accuracy are co-equal.** Apply both gates during extraction and every maintenance rewrite; never trade either one for the other. They outrank recall, category count, category reuse and stylistic polish. Return the required valid JSON, but never distort a fact or its category for any lower-priority objective. The source material is the supplied Episode narratives for initial/update work and the shown stored claims plus their grounding for compact/regroup work.

**Admit only durable facts clearly supported by the source and attributable to the profile owner.** Do not transfer assistant statements or facts about other participants to this person. Exclude one-off actions, passing or short-term states, expiring concrete plans, standalone questions, and team-wide or organisation-wide procedures that apply regardless of who occupies the role. Keep an event only when the source supports that it has continuing explanatory value for the long-term portrait. When attribution, durability or grounding is uncertain, omit or delete the item; never improve recall by guessing. Evidence and basis remain faithful narrative excerpts or paraphrases in one string, never invented facts, attribution or user quotations.

**Require cross-Episode support for every implicit trait.** For this Episode-text path, this stricter gate overrides any general guidance that one signal is enough or that substantive input should produce a trait. Infer a stable disposition only from at least two mutually independent, consistent signals attributable to the profile owner across different Episode narratives. Repetition of one account, several details in one Episode, assistant or other-participant behaviour, or sheer input richness does not satisfy this gate. If the basis cannot faithfully name the qualifying signals, leave implicit_traits empty or delete the stored trait.
Do not generate an implicit trait merely because the input is detailed or because an output with both buckets looks richer."""


PROFILE_INITIAL_FROM_EPISODE_TEXTS_PROMPT = (
    """
{language_rule}

You are a user-profile analyst. From the Episode narratives given in the 【Episode Narratives】 section at the end, build a profile of {target_user}, following the rules below.

"""
    + _SOURCE_RULES
    + "\n\n"
    + _PORTRAIT
    + "\n\n"
    + _ITEM_RULES
    + "\n\n"
    + _CATEGORY_RULES
    + "\n\n"
    + _EPISODE_PRIORITY_RULES
    + """

【Output】
"""
    + _ITEM_SHAPE
    + """

Output JSON directly, nothing else:
```json
{{"explicit_info": [{{"category": "...", "description": "...", "evidence": "..."}}],
  "implicit_traits": [{{"trait": "...", "description": "...", "basis": "..."}}]}}
```

{language_rule}

【Available Categories】
{available_categories}

【Episode Narratives】
{episode_texts}"""
)


PROFILE_UPDATE_FROM_EPISODE_TEXTS_PROMPT = (
    """
{language_rule}

You are a user-profile updater. A stored profile and new Episode narratives are given in the 【Stored Profile】 and 【Episode Narratives】 sections at the end; decide which operations to apply to the stored profile of {target_user}, following the rules below.

"""
    + _SOURCE_RULES
    + "\n\n"
    + _PORTRAIT
    + "\n\n"
    + _ITEM_RULES
    + "\n\n"
    + _CATEGORY_RULES
    + "\n\n"
    + _EPISODE_PRIORITY_RULES
    + """

【Operations】
- **update**: an existing item's own fact gains a correction, a contradiction, a sharper wording or fresh evidence. Update acts on the SAME fact only — it never widens an item to absorb a different fact, however close the topic. An updated description still states one fact in one or two short sentences.
- **add**: a fact not yet on file. Assign its category by the available-category rule above; adding a second copy of a fact already on file is always wrong, because that is an "update" carrying the fresh evidence.
- **delete**: this person explicitly negates it; it has expired ("traveling next week", already past); it is too trivial to be a portrait ("wants pizza today"); a new statement contradicts it outright; **or the stored item asserts a duty, capability, skill, familiarity, interest or attitude this person never claimed** — delete those on sight, whatever evidence they carry, and do not settle for rewriting them.
- **none**: no operation needed — often the right answer. A conversation made only of operations contains nothing about who this person is, however much of it there is. "none" is complete and correct; manufacturing an operation is not.

【Rules】
1. **Index semantics**: every index resolves against the profile snapshot exactly as numbered in the 【Stored Profile】 section. Operations within one response never shift each other's indices — do not adjust an index to compensate for another operation in the same list. "add" takes no index. Indices for explicit_info and implicit_traits are independent.
2. **Keep an item internally consistent**: fields you omit from an "update" keep their stored values, so when you rewrite a "description" carry its matching grounding field ("evidence" or "basis") in the same operation — otherwise the item asserts one thing while its grounding supports another.
3. **Add versus update is decided by the FACT.** Before "add", scan the stored items for the same fact under any wording — found means "update" that item. Not found means "add". For every explicit_info item an operation adds or updates, assign its category by the same available-category rule; this includes an update that otherwise changes only wording or grounding.

【Output】
"""
    + _ITEM_SHAPE
    + """

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
    {{"action": "add", "type": "implicit_traits", "data": {{"trait": "...", "description": "...", "basis": "..."}}}},
    {{"action": "update", "type": "explicit_info", "index": 0, "data": {{"category": "...", "description": "...", "evidence": "..."}}}},
    {{"action": "delete", "type": "implicit_traits", "index": 1, "reason": "..."}}
  ],
  "update_note": "..."
}}
```
Every operation object closes with exactly two braces before the comma: one for "data", one for the operation itself.

{language_rule}

【Available Categories】
{available_categories}

【Stored Profile】(each item carries an index)
{current_profile}

【Episode Narratives】
{episode_texts}"""
)


PROFILE_COMPACT_FROM_EPISODE_TEXTS_PROMPT = (
    """
{language_rule}

The stored profile given in the 【Stored Profile】 section at the end is over its caps: it holds {total_items} items (explicit_info + implicit_traits combined) against a limit of {max_items} TOTAL, no more than {max_per_category} items under any one category or trait name, and every description within one or two short sentences. Rewrite it back inside all three, following the rules below.

"""
    + _PORTRAIT
    + "\n\n"
    + _ITEM_RULES
    + "\n\n"
    + _CATEGORY_RULES
    + "\n\n"
    + _EPISODE_PRIORITY_RULES
    + """

【How to compact】Work in this order.
1. **Delete everything that was never a portrait item, before touching anything else.** An action restated as a capability, duty, skill, familiarity, interest or attitude ("responsible for testing", "capable of debugging", "involved in version control", "cares about test results", "attentive to CI"); a passing state; anything trivial or already expired. Delete these outright — do NOT merge them: folding twelve "responsible for testing" entries into one still leaves a claim this person never made, and it now looks well-founded. Being numerous is not evidence, and neither is the evidence line attached to each one: it records an occasion, the claim asserts a standing property. Twelve such items become **zero** items, not one.
2. **Move anything misfiled.** A trait that merely restates something this person said is an explicit_info fact — move it to explicit_info under the category covering its dimension, and do not leave a copy behind in implicit_traits.
3. **Split every item that chains several facts.** A description held together by "also", semicolons or a comma-list of independent facts becomes several items — one fact each, one or two short sentences each, all under the same category, each keeping the evidence that belongs to its own fact. Splitting is not a way around the caps: a fact that would then be deleted as trivial should simply be deleted.
4. **Merge only restatements of the SAME fact.** Two items saying one thing in different words become one item keeping the better wording. Distinct facts stay distinct items — never collapse a dimension's items into one summary item; that recreates the blob this rewrite exists to remove.
5. **Reclassify every kept explicit_info item.** Apply the same available-category rule independently to every fact. Split mixed groups and correct inaccurate categories even when that creates a category not listed; classification renames an item but never changes its claim. For implicit_traits, choose a semantically accurate trait without treating the available category list as a constraint.
6. **Keep the grounding** of every item you keep — preserve its Episode-narrative form with at most two excerpts or faithful paraphrases in one string; never turn a paraphrase into a direct user quotation or invent missing wording, dates, attribution or identifiers.
7. Prefer deleting a weak item over merging or shortening a strong one. The caps are met by deleting, merging restatements and regrouping — NEVER by inventing a claim or grounding. Ending well under the caps is fine — they are ceilings, not targets.

【Output】
"""
    + _ITEM_SHAPE
    + """

explicit_info + implicit_traits must total no more than {max_items} items, with no more than {max_per_category} under any one category or trait name. Output JSON directly, nothing else:
```json
{{"explicit_info": [{{"category": "...", "description": "...", "evidence": "..."}}],
  "implicit_traits": [{{"trait": "...", "description": "...", "basis": "..."}}],
  "compact_note": "what was split, merged and dropped"}}
```

{language_rule}

【Available Categories】
{available_categories}

【Stored Profile】
{profile_text}"""
)


PROFILE_REGROUP_FROM_EPISODE_TEXTS_PROMPT = (
    """
{language_rule}

One group of a stored user profile has grown past its cap: {count} items share the {label_field} name "{label}", against a limit of {max_per_category} per name. The group's items are given in the 【Items】 section at the end; reorganise THIS GROUP ONLY, following the rules below — the rest of the profile is not shown and must not be assumed.

"""
    + _PORTRAIT
    + "\n\n"
    + _ITEM_RULES
    + "\n\n"
    + _CATEGORY_RULES
    + "\n\n"
    + _EPISODE_PRIORITY_RULES
    + """

【How to regroup】Work in this order.
1. **Merge restatements of the SAME fact** into one item, keeping the better wording and at most two Episode-narrative evidence excerpts or faithful paraphrases in one grounding string. Distinct facts stay distinct items.
2. **Split the name if it has swallowed several dimensions.** Reclassify every kept explicit_info item by the same available-category rule. For implicit_traits, use concise trait names that accurately describe each disposition without treating the available category list as a constraint. Renaming never rewrites a description.
3. **Delete only what was never a portrait item**: an action restated as a capability, duty, skill, familiarity, interest or attitude; a passing state; anything trivial or expired. Deleting is NOT a way to meet the cap — a real fact stays, filed under a better name.
4. **Never change an item's bucket.** These items stay in the bucket they came from, whatever their wording suggests; never invent content, and keep every claim and grounding within what the shown items support.

【Output】
"""
    + _ITEM_SHAPE
    + """

Return every kept item (renamed or not), nothing else:
```json
{{"items": [{{...}}],
  "regroup_note": "what was merged, renamed and dropped"}}
```

{language_rule}

【Available Categories】
{available_categories}

【Items】(currently under "{label}")
{items_text}"""
)
