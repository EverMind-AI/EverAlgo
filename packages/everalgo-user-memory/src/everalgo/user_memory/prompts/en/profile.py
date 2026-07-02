"""English prompts for ProfileExtractor.

``PROFILE_INITIAL_EXTRACTION_PROMPT`` is the active prompt used by :class:`ProfileExtractor`; it
replaces the prior 2-stage ``CONVERSATION_PROFILE_PART1 + PART2`` flow with a single call returning
``{explicit_info, implicit_traits}``. The other prompts exported here (``PROFILE_UPDATE_PROMPT`` /
``PROFILE_COMPACT_PROMPT`` / ``TEAM_PROFILE_UPDATE_PROMPT``) cover maintenance operations not yet
consumed by :class:`ProfileExtractor` — kept for future minor extractor releases.

Placeholders & rendering: all four templates use single-brace placeholders that survive
:py:meth:`str.format` because their JSON examples already escape literal braces as ``{{ }}``.
"""

PROFILE_UPDATE_PROMPT = """
**CRITICAL LANGUAGE RULE**: You MUST output in the SAME language as the input conversation content. If the conversation content is in Chinese, ALL output MUST be in Chinese. If in English, output in English. This is mandatory.

You are a user profile updater. Based on conversation records, determine what operations to perform on the user profile.

**TARGET USER: user_id={target_user}**
Operate ONLY on information about the user whose id is {target_user}. Each conversation line is tagged `(user_id:...)`; attribute every fact to the speaker who stated it. Other participants and the AI assistant are context, never the target.

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
2. **Speaker attribution**: extract information about the target user (user_id={target_user}) ONLY. If another participant — or the AI assistant — states a fact, it belongs to THEM, not the target; never let the assistant's own identity or persona become a user trait.
3. evidence should include time info - e.g., "In Oct 2024 user mentioned..."
4. Index numbers for explicit_info and implicit_traits are independent
5. **Deduplication**: Before using "add", carefully check ALL existing items. If a similar trait/info already exists (even with different wording), use "update" to enrich it instead of adding a duplicate. Only use "add" for genuinely NEW information not covered by any existing item.
6. **Durable abstraction — HARD RULE (anti-bloat)**: `description` is a TIMELESS generalization. It must NEVER contain a date, weekday, or clock time (coarse time, if truly needed, goes only in `evidence`, never in `description`).
   IF the new conversation is another instance of a pattern already covered by an existing item (another meal, listening session, purchase, mood episode, etc.) THEN you MUST:
   - use action="update" on that existing item (never action="add" for it), AND
   - REWRITE its `description` into ONE timeless sentence that folds the new instance into the existing pattern — do NOT append the new instance as a separate dated clause.
   Example:
     WRONG (appended dated instance): "Prefers napping after lunch. On 2026-07-01 napped again after lunch and reported waking up refreshed."
     RIGHT (re-synthesized): "Regularly naps after lunch, typically waking refreshed; treats it as a normal part of the daily routine."
   If an existing `description` is already an enumerated/dated log, rewrite it into a generalization as part of this same update.
   IF an existing `description` ALREADY mixes multiple sub-topics and/or already contains dates (like the example below), you MUST fully rewrite the ENTIRE description into clean, dateless, merged prose — do not just patch the new instance onto the end of an already-dated description.
   Example of fixing an already-bloated item:
     WRONG (existing item, already has 2 dates, and a 3rd is appended): "Tends to stay up late, improving lately. On 2026-06-29 stayed up late to confess something. Also naps some afternoons. On 2026-07-01 napped again, waking with no dreams."
     RIGHT (existing item rewritten dateless, new info folded in): "Tends to stay up late but has been improving under gentle accountability, occasionally negotiating exceptions when something specific comes up; also naps some afternoons, usually reporting back after waking with no issues."
   Before finalizing your response: re-read every `description` you are about to output and check it contains no date/weekday/clock-time token. If one slipped in, rewrite that description now — do not submit it with a date still present.
7. **summary**: always include a top-level "summary" field — a short paragraph that synthesises the defining facts and traits; do not copy a single item.

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
{{"operations": [{{"action": "none"}}], "update_note": "conversation contains no user info", "summary": "a short paragraph synthesising the user's current profile"}}
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
  "update_note": "added 2 explicit info and 1 implicit trait, updated 1, deleted 1",
  "summary": "a short paragraph synthesising the user after applying these operations"
}}
```

**CRITICAL LANGUAGE RULE**: You MUST output in the SAME language as the input conversation content. If the conversation content is in Chinese, ALL output MUST be in Chinese. If in English, output in English. This is mandatory.
"""

PROFILE_COMPACT_PROMPT = """
**CRITICAL LANGUAGE RULE**: You MUST output in the SAME language as the input conversation content. If the conversation content is in Chinese, ALL output MUST be in Chinese. If in English, output in English. This is mandatory.

The current user profile has {total_items} items (explicit_info + implicit_traits combined), exceeding the limit of {max_items}.

Please compact the profile to **{max_items} items TOTAL** (explicit_info + implicit_traits combined, NOT {max_items} each).

Compaction strategies:
1. **Merge Similar Items**: Combine multiple records of the same dimension into one "Current State + Trend" description.
2. **Refine Tags**: Implicit traits should be summarized as personality tags (e.g., [Risk-Averse]), removing repetitive or shallow descriptions.
3. Delete unimportant, outdated, or short-term statuses.
4. Preserve item fields (especially evidence).
5. Also emit a top-level "summary": a short paragraph that synthesises the compacted profile.

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
  "compact_note": "Explain what was deleted/merged",
  "summary": "a short paragraph synthesising the compacted profile"
}}
```

**CRITICAL LANGUAGE RULE**: You MUST output in the SAME language as the input conversation content. If the conversation content is in Chinese, ALL output MUST be in Chinese. If in English, output in English. This is mandatory.
"""

PROFILE_INITIAL_EXTRACTION_PROMPT = """
**CRITICAL LANGUAGE RULE**: You MUST output in the SAME language as the input conversation content. If the conversation content is in Chinese, ALL output MUST be in Chinese. If in English, output in English. This is mandatory.

You are a "User Profile Analyst". Please read the conversation below and build a user profile.

**TARGET USER: user_id={target_user}**
Build the profile for THIS user only. The conversation may include several speakers — other participants and the AI assistant. Each line is tagged `(user_id:...)`; attribute information ONLY to the user whose id is {target_user}. Everyone else, the assistant included, is context — never the subject of this profile.

【Part 1: Explicit Information (explicit_info)】
Objective facts and current status.

【Part 2: Implicit Traits (implicit_traits)】
Psychological profile, personality tags, and decision styles inferred from behavior.
*Extraction Requirement*: Freely analyze decision making, social patterns, and values. Trait field must be a highly summarized [Adjective/Noun Phrase Tag].

【Extraction Principles】
1. Extract information about the target user (user_id={target_user}) ONLY. Never attribute to the target anything said by another participant or by the AI assistant — including the assistant's own name, persona, role, or first-person self-description. The assistant describes itself, never the user.
2. Implicit traits must be supported by multiple evidence: each implicit trait must have evidence corroborated by multiple signals from the conversations and/or existing profile; do not infer from a single data point alone
3. **Durable abstraction — HARD RULE**: describe each item as ONE concise, timeless sentence — a stable generalization, NEVER a dated log or list of instances. `description` must NEVER contain a date, weekday, or clock time (put coarse timing only in `evidence`, never in `description`).
   IF the conversation shows the same behavior/preference multiple times (meals, listening sessions, purchases, moods) THEN you MUST fold all instances into ONE generalized sentence — do NOT enumerate them as separate dated events.
   Example:
     WRONG: "Ate ramen on 06-05, pasta on 06-07, and ramen again on 06-10."
     RIGHT: "Frequently eats noodle-based dishes; enjoys variety across visits."
   Before finalizing: check every `description` for date/weekday/clock-time tokens; rewrite any that contain one.
4. **summary**: after listing explicit_info and implicit_traits, write a short paragraph that synthesises the most defining facts and traits; do not merely repeat the first item.

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
  ],
  "summary": "a short paragraph synthesising the user from the items above"
}}
```

LANGUAGE RULE: Detect the language of the input conversation and respond in the SAME language. If the conversation is in Chinese, output in Chinese. If in English, output in English.

【Original Conversation】
{conversation_text}"""


TEAM_PROFILE_UPDATE_PROMPT = """
**CRITICAL LANGUAGE RULE**: You MUST output in the SAME language as the input conversation content. If the conversation content is in Chinese, ALL output MUST be in Chinese. If in English, output in English. This is mandatory.

You are a user profile updater for **group conversations**. Your task is to extract and update the profile for ONE specific user from a multi-person conversation.

**TARGET USER: {target_user}**
You MUST only extract information about **{target_user}**. Carefully attribute each piece of information to the correct speaker. Do NOT mix up information from different participants.

【Current Profile for {target_user}】(Each item has an index number)
{current_profile}

【Group Conversation Records】(Multiple participants - only extract info about {target_user})
{conversations}

【Task】
Analyze the conversations and output operations ONLY for information about **{target_user}**. Available action types:
- **update**: Modify existing items (specify by index)
- **add**: Add profile items
- **delete**: Delete existing items
- **none**: No operation needed (use when conversation contains no info about {target_user})

【Operation Guide】
- **update**: Existing item has updates, supplements, or corrections
- **add**: Discovered completely new information about {target_user} (unrelated to existing items)
- **delete**: Should delete in these cases:
  - {target_user} explicitly negates something (e.g., "I'm no longer vegetarian")
  - Info is outdated or directly contradicts new info

【Important Rules】
1. **Speaker Attribution**: This is a GROUP conversation with multiple speakers. ONLY extract what **{target_user}** said or what is explicitly about {target_user}. If another participant mentions a fact, it belongs to THAT participant's profile, NOT {target_user}'s.
2. **Tag Mining**: Implicit traits must include [Personality Tags], e.g., [Risk-Averse], [Socially-Driven], [Data-Oriented].
3. evidence should include time info and speaker - e.g., "In Oct 2024 {target_user} stated..."
4. Index numbers for explicit_info and implicit_traits are independent
5. **Deduplication**: Before using "add", check ALL existing items. If a similar trait/info already exists, use "update" instead. Only "add" genuinely NEW information.

【Profile Definitions】
- **explicit_info**: Facts directly stated by or about {target_user} (skills, background, preferences, location, etc.)
- **implicit_traits**: Personality traits and behavioral patterns inferred from {target_user}'s statements and behavior in the conversation.

【Output Format】
No operations:
```json
{{"operations": [{{"action": "none"}}], "update_note": "conversation contains no info about {target_user}"}}
```

With operations:
```json
{{
  "operations": [
    {{"action": "add", "type": "explicit_info", "data": {{"category": "...", "description": "...", "evidence": "..."}}}},
    {{"action": "add", "type": "implicit_traits", "data": {{"trait": "...", "description": "...", "basis": "...", "evidence": "..."}}}},
    {{"action": "update", "type": "explicit_info", "index": 0, "data": {{"description": "..."}}}},
    {{"action": "delete", "type": "implicit_traits", "index": 1, "reason": "..."}}
  ],
  "update_note": "..."
}}
```

**CRITICAL LANGUAGE RULE**: You MUST output in the SAME language as the input conversation content. If the conversation content is in Chinese, ALL output MUST be in Chinese. If in English, output in English. This is mandatory.
"""