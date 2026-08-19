"""English prompts for ProfileExtractor.

``PROFILE_INITIAL_EXTRACTION_PROMPT`` is the active prompt used by :class:`ProfileExtractor`; it
replaces the prior 2-stage ``CONVERSATION_PROFILE_PART1 + PART2`` flow with a single call returning
``{explicit_info, implicit_traits}``. The other prompts exported here (``PROFILE_UPDATE_PROMPT`` /
``PROFILE_COMPACT_PROMPT``) cover maintenance operations. ``PROFILE_INITIAL_EXTRACTION_PROMPT`` and
``PROFILE_UPDATE_PROMPT`` inject a ``{target_user}`` so extraction is scoped to a single speaker in
multi-party conversations; ``PROFILE_COMPACT_PROMPT`` only re-summarises already-stored items.

Placeholders & rendering: all three templates use single-brace placeholders that survive
:py:meth:`str.format` because their JSON examples already escape literal braces as ``{{ }}``.
"""

PROFILE_UPDATE_PROMPT = """
{language_rule}

You are a user profile updater. Based on conversation records, determine what operations to perform on the user profile.

**TARGET USER: {target_user}**
This may be a multi-speaker conversation; each line is tagged with the speaker's ``user_id``. Only update the profile for **{target_user}** (the speaker whose ``user_id`` equals {target_user}). Information stated by or about any other participant belongs to THAT participant — never attribute it to {target_user}.

【Current User Profile】(Each item has an index number)
{current_profile}

【Conversation Records】(Multiple conversations from the same topic)
{conversations}

【Task】
Analyze conversations and output a list of operations (can have multiple). Available action types:
- **update**: Modify existing items (specify by index)
- **add**: Add profile items
- **delete**: Delete existing items
- **none**: No operation needed (use when conversation contains no user info)

【Operation Guide】
- **update**: Existing item has updates, supplements, or corrections
- **add**: Discovered completely new user information (unrelated to existing items)
- **delete**: Should delete in these cases:
  - User explicitly negates (e.g., "I'm no longer vegetarian")
  - Info is outdated (e.g., "traveling next week" but it's already passed)
  - Too trivial/useless (e.g., "want pizza today")
  - Directly contradicts new info

【Important Rules】
1. **Tag Mining**: Implicit traits must include [Personality Tags], e.g., [Risk-Averse], [Socially-Driven], [Data-Oriented].
2. Only extract info about {target_user}, not other participants; don't treat AI assistant suggestions as user traits
3. evidence should include time info - e.g., "In Oct 2024 user mentioned..."
4. Index numbers for explicit_info and implicit_traits are independent
5. **Deduplication**: Before using "add", carefully check ALL existing items. If a similar trait/info already exists (even with different wording), use "update" to enrich it instead of adding a duplicate. Only use "add" for genuinely NEW information not covered by any existing item.
6. **Index semantics**: Every index you emit is resolved against the profile snapshot shown above, numbered exactly as it appears there. Operations within one response never shift each other's indices — do not adjust an index to compensate for another operation in the same list. Do not emit an index for an item you are adding in this response; "add" takes no index.

【Profile Definitions & Analysis Framework】
- **explicit_info (Explicit Information)**: User facts that can be directly extracted from conversations.
  - *Content*: Basic info, health status, skills, clear preferences.

- **implicit_traits (Implicit Traits)**: Psychological profile, personality tags, and decision styles inferred from behavior.
  - *Extraction Requirement*: Freely analyze from dimensions like decision patterns, social preferences, and life philosophy.
  - *Naming Convention*:
    1. Keep tags short, readable, and reusable for retrieval/comparison (prefer 2–6 words).
    2. Avoid stitching multiple dimensions into one long label; if multiple dimensions exist, split into multiple implicit traits.
    3. Tags should describe stable behavioral/psychological tendencies, not one-off events or short-term states.
  - Make reasonable inferences to extract the user's deep traits

【Output Format】
No operations:
```json
{{"operations": [{{"action": "none"}}], "update_note": "conversation contains no user info"}}
```

With operations (can combine multiple add/update/delete):
```json
{{
  "operations": [
    {{"action": "add", "type": "explicit_info", "data": {{"category": "...", "description": "...", "evidence": "..."}}}},
    {{"action": "add", "type": "implicit_traits", "data": {{"trait": "...", "description": "...", "basis": "...", "evidence": "..."}}}},
    {{"action": "update", "type": "explicit_info", "index": 0, "data": {{"description": "..."}}}},
    {{"action": "delete", "type": "implicit_traits", "index": 1, "reason": "..."}}
  ],
  "update_note": "added 2 explicit info and 1 implicit trait, updated 1, deleted 1"
}}
```

{language_rule}
"""

PROFILE_COMPACT_PROMPT = """
{language_rule}

The current user profile has {total_items} items (explicit_info + implicit_traits combined), exceeding the limit of {max_items}.

Please compact the profile to **{max_items} items TOTAL** (explicit_info + implicit_traits combined, NOT {max_items} each).

Compaction strategies:
1. **Merge Similar Items**: Combine multiple records of the same dimension into one "Current State + Trend" description.
2. **Refine Tags**: Implicit traits should be summarized as personality tags (e.g., [Risk-Averse]), removing repetitive or shallow descriptions.
3. Delete unimportant, outdated, or short-term statuses.
4. Preserve item fields (especially evidence).

Current Profile:
{profile_text}

**IMPORTANT**: Output must have explicit_info + implicit_traits ≤ {max_items} items TOTAL.
```json
{{
  "explicit_info": [
    {{"category": "...", "description": "...", "evidence": "..."}}
  ],
  "implicit_traits": [
    {{"trait": "...", "description": "...", "basis": "...", "evidence": "..."}}
  ],
  "compact_note": "Explain what was deleted/merged"
}}
```

{language_rule}
"""

PROFILE_INITIAL_EXTRACTION_PROMPT = """
{language_rule}

You are a "User Profile Analyst". Please read the conversation below and build a user profile.

**TARGET USER: {target_user}**
This may be a multi-speaker conversation; each line is tagged with the speaker's ``user_id``. Only build a profile for **{target_user}** (the speaker whose ``user_id`` equals {target_user}). Information stated by or about any other participant belongs to THAT participant — never attribute it to {target_user}.

【Part 1: Explicit Information (explicit_info)】
Objective facts and current status.

【Part 2: Implicit Traits (implicit_traits)】
Psychological profile, personality tags, and decision styles inferred from behavior.
*Extraction Requirement*: Freely analyze decision making, social patterns, and values. Trait field must be a highly summarized [Adjective/Noun Phrase Tag].

【Extraction Principles】
1. Only extract information about {target_user} themselves, not other participants and not assistant suggestions
2. Implicit traits must be supported by multiple evidence: each implicit trait must have evidence corroborated by multiple signals from the conversations and/or existing profile; do not infer from a single data point alone
3. Describe each piece of information in one natural sentence, easy to understand

【Output Format】
Output JSON directly in the following format:
```json
{{
  "explicit_info": [
    {{
      "category": "category name",
      "description": "one sentence description",
      "evidence": "one-sentence evidence grounded in the conversations"
    }}
  ],
  "implicit_traits": [
    {{
      "trait": "trait name",
      "description": "one sentence description of this trait",
      "basis": "inferred from which behaviors/conversations",
      "evidence": "one-sentence evidence grounded in the conversations"
    }}
  ]
}}
```

{language_rule}

【Original Conversation】
{conversation_text}"""
