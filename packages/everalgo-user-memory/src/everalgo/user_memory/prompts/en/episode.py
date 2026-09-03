"""English prompts for EpisodeExtractor.

Constants:
    - ``DEFAULT_CUSTOM_INSTRUCTIONS`` — block of instructions injected into the {custom_instructions} slot.
    - ``EPISODE_DECLARE_LANGUAGE_ADDITION`` — measured fallback-only instruction that makes the model declare
      the participants' language before generating the episode.
    - ``EPISODE_GENERATION_PROMPT`` — generic retrieval-optimised episode generation. Placeholders:
      ``{conversation_start_time}`` / ``{conversation}`` / ``{custom_instructions}`` / ``{language_rule}``.
    - ``USER_EPISODE_GENERATION_PROMPT`` — user-centred variant focused on a specific ``{user_name}``;
      additional placeholder ``{user_name}`` on top of the four above. Used by
      ``EpisodeExtractor`` when ``sender_id`` is provided. Pass ``sender_id=None`` to fall back to
      the generic ``EPISODE_GENERATION_PROMPT``.
    - ``SUMMARY_COMPRESS_PROMPT`` — repair prompt for an over-wide ``summary``: shortens that
      summary itself (placeholders ``{language_rule}`` / ``{summary_text}``), returning plain text rather than JSON. Used
      by the width guard in ``episode.py``; its stricter repair target must remain within that guard.

Output schemas: the generic variant emits
``{"user_language": str, "title": str, "content": str, "summary": str}``; the user-centred variant emits
``{"title": str, "content": str, "summary": str}``, with fallback generation asking for
``user_language`` first through ``EPISODE_DECLARE_LANGUAGE_ADDITION``.

``summary`` is a display preview of ``content``, and it is listed after ``content`` in every field
spec and example on purpose: the model emits JSON left to right, so a ``summary`` placed first would
be written before the narrative it summarises — a second independent pass over the conversation
rather than a compression of the record. Keep it last when editing these prompts.

``{language_rule}`` — appears twice in each prompt and both copies receive the same text — is filled from
``user_memory._language.build_language_rule`` according to ``aextract``'s ``output_language`` argument:
the caller's language when one is named, otherwise a rule asking the model to follow the participants.
A replacement prompt that drops the placeholder silently opts out of output-language control; one that
keeps it inherits the behaviour. Whether a given output language is actually retrievable depends on the
tokenizer used by the caller's search layer — a Chinese-only tokenizer cannot index Japanese kana, Hangul
or Cyrillic at all.

Contract for a caller-supplied ``prompt=`` override: code stores ``content`` verbatim (see
``episode.py::_build_episode``), so every time a reader sees is one these prompts made the model write
— which is why the time rules below are load-bearing rather than cosmetic. A replacement prompt takes
over that contract: it must require an absolute, ``UTC``-labelled time for the events it narrates, or
the stored episode carries no time a downstream LLM can read.
"""

DEFAULT_CUSTOM_INSTRUCTIONS = """
Follow these principles when generating episodic memories:
1. Each episode should be a complete, independent story or event
2. Preserve all important information including names, time, location, emotions, etc.
3. Use declarative language to describe episodes, not dialogue format
4. Highlight key information and emotional changes
5. Ensure episode content is easy to retrieve later
"""

#: Appended to custom instructions only when ``output_language`` is unset. The wording is measured; rerun the
#: language evaluation before changing it.
EPISODE_DECLARE_LANGUAGE_ADDITION = """\


Before writing, decide which language the participants speak to each other; text they paste, quote, or ask to process is not their speech. Add "user_language" as the FIRST field of the JSON output (e.g. "Chinese", "English"), then write title, content and summary in that language.\
"""

EPISODE_GENERATION_PROMPT = """
{language_rule}

You are an episodic memory generation expert. Please convert the following conversation into an episodic memory.

Conversation start time: {conversation_start_time}
Conversation content:
{conversation}

Custom instructions:
{custom_instructions}

IMPORTANT TIME HANDLING:
- Use the provided "Conversation start time" only as context for when this conversation/episode began; do not copy it into content or use it to resolve time references
- When the conversation mentions a resolvable non-absolute time reference, preserve the original expression and resolve it to an absolute date or range at the same granularity
- Preserve vague references without a deterministic boundary, such as "recently", "later", or "someday", without inventing an absolute date or range
- Each message has its own timestamp. Resolve the reference using the timestamp of the specific message that contains it — NOT the conversation start time
- Format: "original expression (absolute date or range)". For a message timestamped "2023-05-15 10:00 UTC (Monday)", write "last Friday (2023-05-12)" or, using an ISO Monday-Sunday calendar week, "last week (2023-05-08 to 2023-05-14)"
- This dual format supports both absolute and relative time-based questions
- Every absolute time that states a clock time MUST carry the UTC zone label: write "2024-03-14 15:00 UTC", never "2024-03-14 15:00" and never a bare "15:00". A date with no clock time needs no label — "last Friday (2023-07-21)" is already correct. Clock times quoted from what a speaker said keep their original wording ("at 3:30 PM"), because the speaker's own timezone is unknown

Please generate a structured episodic memory and return only a JSON object containing the following fields:
{{
    "user_language": "The exact English name of the language used for title, content, and summary (e.g. Chinese, English, French)",
    "title": "A concise, descriptive title that accurately summarizes the theme (fewer than 20 words; use only as many words as needed)",
    "content": "A concise factual record of the conversation in third-person narrative. It must include all important information: who participated at what time, what was discussed, what decisions were made, what emotions were expressed, and what plans or outcomes were formed. Write it as a chronological account focusing on observable actions and direct statements. Remove redundant expressions and verbose descriptions while preserving all facts, entities (names, dates, locations), and specific details. Keep the content concise without losing key information.",
    "summary": "A faithful preview of this episode in 1-3 short sentences (at most ~50 English words / ~100 Chinese characters). COMPRESS, never restate: do not reuse content's sentences — write shorter new ones that name the main participants and what actually happened, including the outcome or decision reached. Faithfully summarize the content without inventing facts or distorting their meaning. You may omit minor details, but every included detail must remain faithful to the content. Do not refer to the record itself — no 'this conversation', 'the above', 'the user asked'."
}}

Requirements:
1. The title should be specific and easy to search (including key topics/activities), use fewer than 20 words, and never be padded solely to make it longer.
2. The content must include all important information from the conversation while being concise.
3. Convert the dialogue format into a narrative description.
4. Maintain chronological order and causal relationships.
5. Use third-person unless explicitly first-person.
6. Include specific details that aid keyword search, especially concrete activities, places, and objects.
7. For resolvable time references, use the message's own timestamp and preserve the original granularity: "original expression (absolute date or range)". Preserve vague references without inventing dates.
8. When describing decisions or actions, naturally include the reasoning or motivation behind them, but avoid repetitive explanations.
9. Use specific names consistently rather than pronouns to avoid ambiguity in retrieval.
10. CONCISENESS AND REDUNDANCY REMOVAL:
    - Remove redundant expressions and verbose descriptions
    - Avoid repeating the same information in different ways
    - Eliminate unnecessary filler words and phrases
    - Keep sentences direct and to the point
    - Preserve all facts, entities (names, dates, locations), and specific details
    - Maintain the core meaning and important information
    - Aim for content length similar to or shorter than the original conversation
11. CRITICAL DETAIL PRESERVATION:
   - Person Names: Always include full names of people mentioned (e.g., "went to yoga with Amy's colleague, Rob" not just "went to yoga with a colleague")
   - Special Nouns & Entities: Preserve all proper nouns, brand names, place names, organization names exactly as mentioned
   - Item Names: Include specific product names, book titles, movie names, restaurant names, etc.
   - Quantities & Numbers: Record exact numbers, amounts, prices, percentages, dates, times (e.g., "ordered 3 pizzas" not "ordered pizzas")
   - Specific Activities: Use precise activity descriptions (e.g., "practiced hot yoga" not just "exercised")
   - Time Points: Include all specific times mentioned (e.g., "at 3:30 PM", "every Tuesday", "twice a week")
12. FREQUENCY INFORMATION:
   - Record recurring activities and their frequency (e.g., "goes to yoga class every Tuesday and Thursday")
   - Note patterns of behavior (e.g., "mentioned calling mom three times during the conversation")
   - Include habitual actions (e.g., "usually has coffee at 8 AM before work")
   - Document repetition counts (e.g., "asked about the project status twice")

Example:
If the conversation start time is "2024-03-14 15:00 UTC (Thursday)" and the conversation is about Caroline planning to go hiking:
{{
    "user_language": "English",
    "title": "Caroline Plans an Early Saturday Start for a Mount Rainier Sunrise Hike",
    "content": "Caroline expressed interest in hiking this weekend (2024-03-16 to 2024-03-17) and sought advice. She wanted to see the sunrise at Mount Rainier. When asked about gear by Melanie, Caroline received suggestions: hiking boots, warm clothing, flashlight, water, and high-energy food. Caroline decided to leave early Saturday morning (2024-03-16) to catch the sunrise and planned to invite friends. She was excited about the trip.",
    "summary": "Caroline planned a sunrise hike at Mount Rainier for the weekend (2024-03-16 to 2024-03-17) and asked for advice. Melanie suggested boots, warm clothing, a flashlight, water and high-energy food. Caroline settled on an early Saturday start and planned to invite friends, excited for the trip."
}}

{language_rule}

The "user_language" value is metadata and MUST always use the exact English name of the selected language. The language rule applies to title, content, and summary; it does not translate this metadata value.

Return only the JSON object, do not add any other text:
"""

USER_EPISODE_GENERATION_PROMPT = """
{language_rule}

You are a professional event recorder and episodic memory generation expert, specialised in capturing events relevant to a specific user from a conversation.
Your task is to focus on {user_name} — objectively record what he/she saw, heard, said, and did, and turn it into a coherent, accurate event record.

User name: {user_name}

Conversation start time: {conversation_start_time}
Conversation content:
{conversation}

Custom instructions:
{custom_instructions}

Please follow these principles:

1.  **Objective, neutral perspective**:
    *   Ground every detail in the facts; the goal is to record events relevant to {user_name}, not to cast him/her as the protagonist.
    *   **Strictly distinguish roles**: if {user_name} is only a recipient of information or an inquirer, the record must reflect that clearly. For example, write "{user_name} learned that..." or "{user_name} asked about..." rather than implying he/she drove the discussion.
    *   Filter out parts of the conversation that have no direct bearing on {user_name}, but keep the context needed to understand the event.

2.  **Coherent narrative**:
    *   Connect {user_name}'s key actions and decisions into a fluent narrative. For example, instead of "{user_name} agreed", write "After listening to everyone's suggestions, {user_name} eventually agreed to the plan" — surface the context.
    *   Make {user_name}'s role and behavioural arc in the event clear.

3.  **Distil the core experience**:
    *   This is more than recording — it is distillation. Summarise {user_name}'s key decisions, important plans, commitments received, or problems encountered during the conversation.
    *   Weave these core points naturally into the narrative.

4.  **Faithful to the source and easy to understand**:
    *   All content must be based on the original conversation, but you may reorganise and express it in a more readable way.
    *   Avoid adding subjective speculation or commentary not present in the source.

Output quality requirements (continued from the principles above):

5.  **CONCISENESS AND REDUNDANCY REMOVAL**:
    - Remove redundant expressions and verbose descriptions
    - Avoid repeating the same information in different ways
    - Keep sentences direct and to the point
    - Aim for content length similar to or shorter than the original conversation

6.  **CRITICAL DETAIL PRESERVATION**:
    - Person Names: Always include full names of people mentioned (e.g., "discussed with Bob, Amy's colleague" not just "discussed with a colleague")
    - Special Nouns & Entities: Preserve all proper nouns, brand names, place names, organisation names exactly as mentioned
    - Item Names: Include specific product names, book titles, movie names, restaurant names, etc.
    - Quantities & Numbers: Record exact numbers, amounts, prices, percentages, dates, times (e.g., "ordered 3 pizzas" not "ordered pizzas")
    - Specific Activities: Use precise activity descriptions (e.g., "practiced hot yoga" not just "exercised")
    - Time Points: Include all specific times mentioned (e.g., "at 3:30 PM", "every Tuesday", "twice a week")

7.  **FREQUENCY INFORMATION**:
    - Record recurring activities and their frequency (e.g., "goes to yoga class every Tuesday and Thursday")
    - Note patterns of behaviour (e.g., "mentioned calling mom three times during the conversation")
    - Include habitual actions (e.g., "usually has coffee at 8 AM before work")
    - Document repetition counts (e.g., "asked about the project status twice")

8.  **TIME REFERENCES — DUAL FORMAT**:
    - Use the provided "Conversation start time" only as context for when this conversation/episode began; do not copy it into content or use it to resolve time references
    - ANY resolvable non-absolute time reference must be resolved to an absolute date or range at the same granularity using the timestamp of the specific message that contains it — NOT the conversation start time
    - Examples of resolvable references: "yesterday", "last Friday", "last week" — any granularity
    - Preserve vague references without a deterministic boundary, such as "recently", "later", or "someday", without inventing an absolute date or range
    - Preserve both the original expression AND the resolved absolute date or range
    - Format: "original expression (absolute date or range)". For a message timestamped "2023-05-15 10:00 UTC (Monday)", write "last Friday (2023-05-12)" or, using an ISO Monday-Sunday calendar week, "last week (2023-05-08 to 2023-05-14)"
    - Every absolute time that states a clock time MUST carry the UTC zone label: write "2024-03-14 15:00 UTC", never "2024-03-14 15:00" and never a bare "15:00". A date with no clock time needs no label — "last Friday (2023-07-21)" is already correct. Clock times quoted from what a speaker said keep their original wording ("at 3:30 PM"), because the speaker's own timezone is unknown

9.  **TITLE SHAPE AND LENGTH**:
    - Use fewer than 20 words. There is no minimum length; never pad the title solely to make it longer
    - The title does not need to begin with or include `{user_name}`. Include the name only when needed to disambiguate participants; otherwise prioritise the core event and key outcome

10. **NARRATIVE FORM**:
    - Use third-person throughout (consistent with principle 1)
    - Maintain chronological order and causal relationships between events. Note: this refers to inter-event causality (e.g., "after listening to suggestions → decided to leave Saturday"), not {user_name}'s role in those events (principle 1 still governs role attribution).

Please generate a structured episodic memory and return only a JSON object containing the following three fields:
{{
    "title": "(A title of fewer than 20 words naming the core event and key outcome; use only as many words as needed and include `{user_name}` only when needed to disambiguate participants)",
    "content": "(A {user_name}-centred objective, coherent narrative.)",
    "summary": "A faithful preview of this episode in 1-3 short sentences (at most ~50 English words / ~100 Chinese characters). COMPRESS, never restate: do not reuse content's sentences — write shorter new ones that name the main participants and what actually happened, including the outcome or decision reached. Faithfully summarize the content without inventing facts or distorting their meaning. You may omit minor details, but every included detail must remain faithful to the content. Do not refer to the record itself — no 'this conversation', 'the above', 'the user asked'."
}}

Example (incorrect style — too subjective):
{{
    "title": "{user_name} led the weekend hiking planning",
    "content": "{user_name} initiated the plan to hike Mount Rainier this weekend and led the gear discussion. He/she made the final call on departure time and arranged to invite friends.",
    "summary": "{user_name} took charge of the weekend Mount Rainier hiking plan, drove the gear discussion and decided when everyone would leave."
}}

Example (correct style — objective recording):
{{
    "title": "Early Saturday Start Chosen for a Mount Rainier Sunrise Hike with Friends",
    "content": "{user_name} proposed hiking Mount Rainier this weekend in the conversation and asked for gear suggestions. After listening to others' advice on hiking boots and warm clothing, {user_name} decided to leave early Saturday morning (2024-03-16) to catch the sunrise, and mentioned inviting friends along.",
    "summary": "{user_name} proposed a weekend hike to Mount Rainier and asked for gear advice. After hearing suggestions on boots and warm clothing, {user_name} chose to leave early Saturday (2024-03-16) for the sunrise and mentioned inviting friends."
}}

Self-check list:
- Is the return strictly in JSON format?
- Do `title`, `content` and `summary` all strictly revolve around `{user_name}`?
- Does `summary` read on its own, and say only what `content` already says?
- Is `summary` at most 3 short sentences and much shorter than `content`?
- Is `content` a fluent experiential narrative, not a heap of scattered facts?
- Are all key points woven naturally into the narrative?

{language_rule}

Return only the JSON object, do not add any other text:
"""


SUMMARY_COMPRESS_PROMPT = """
{language_rule}

The text below is an episode summary that exceeds the display limit. Shorten it to 1-3 short sentences, at most ~50 English words / ~100 Chinese characters. Shorten only by removing complete sentences or clauses. Keep all retained wording unchanged; do not paraphrase, merge statements, add facts, or infer. Return ONLY the shortened summary — no JSON, no quotes, no explanations.

Summary:
{summary_text}
"""
