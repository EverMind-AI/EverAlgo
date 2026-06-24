"""English prompts for EpisodeExtractor.

Constants:
    - ``DEFAULT_CUSTOM_INSTRUCTIONS`` — block of instructions injected into the {custom_instructions} slot.
    - ``EPISODE_GENERATION_PROMPT`` — generic retrieval-optimised episode generation. Placeholders:
      ``{conversation_start_time}`` / ``{conversation}`` / ``{custom_instructions}``.
    - ``USER_EPISODE_GENERATION_PROMPT`` — user-centred variant focused on a specific ``{user_name}``;
      additional placeholder ``{user_name}`` on top of the three above. Used by
      ``EpisodeExtractor`` when ``sender_id`` is provided. Pass ``sender_id=None`` to fall back to
      the generic ``EPISODE_GENERATION_PROMPT``.

Output schema (both variants): ``{"title": str, "content": str}``.
"""

DEFAULT_CUSTOM_INSTRUCTIONS = """
Follow these principles when generating episodic memories:
1. Each episode should be a complete, independent story or event
2. Preserve all important information including names, time, location, emotions, etc.
3. Use declarative language to describe episodes, not dialogue format
4. Highlight key information and emotional changes
5. Ensure episode content is easy to retrieve later
"""

EPISODE_GENERATION_PROMPT = """
You are an episodic memory generation expert. Please convert the following conversation into an episodic memory.

Conversation start time: {conversation_start_time}
Conversation content:
{conversation}

Custom instructions:
{custom_instructions}

IMPORTANT TIME HANDLING:
- "Conversation start time" indicates when this conversation began
- Each message has its own timestamp. ANY non-absolute time reference must be resolved to an absolute date using the timestamp of the specific message that contains it — NOT the conversation start time
- Examples: "yesterday", "last Friday", "last summer", "recently" — any granularity
- Preserve both the original expression AND the resolved absolute date
- Format: "original expression (absolute date)" — e.g., "last summer (summer 2022)", "last Friday (July 21, 2023)"
- This dual format supports both absolute and relative time-based questions

Please generate a structured episodic memory and return only a JSON object containing the following two fields:
{{
    "title": "A concise, descriptive title that accurately summarizes the theme (10-20 words)",
    "content": "A concise factual record of the conversation in third-person narrative. It must include all important information: who participated at what time, what was discussed, what decisions were made, what emotions were expressed, and what plans or outcomes were formed. Write it as a chronological account focusing on observable actions and direct statements. Remove redundant expressions and verbose descriptions while preserving all facts, entities (names, dates, locations), and specific details. Keep the content concise without losing key information. Resolve every non-absolute time reference using the individual message's timestamp, not the conversation start time."
}}

Requirements:
1. The title should be specific and easy to search (including key topics/activities).
2. The content must include all important information from the conversation while being concise.
3. Convert the dialogue format into a narrative description.
4. Maintain chronological order and causal relationships.
5. Use third-person unless explicitly first-person.
6. Include specific details that aid keyword search, especially concrete activities, places, and objects.
7. For time references, resolve ANY non-absolute time expression using the message's own timestamp: "original expression (absolute date)" — e.g., "last summer (summer 2022)".
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
If the conversation start time is "March 14, 2024 (Thursday) at 3:00 PM UTC" and the conversation is about Caroline planning to go hiking:
{{
    "title": "Caroline's Mount Rainier Hiking Plan March 14, 2024: Weekend Adventure Planning Session",
    "content": "On March 14, 2024 at 3:00 PM UTC, Caroline expressed interest in hiking this weekend (March 16-17, 2024) and sought advice. She wanted to see the sunrise at Mount Rainier. When asked about gear by Melanie, Caroline received suggestions: hiking boots, warm clothing, flashlight, water, and high-energy food. Caroline decided to leave early Saturday morning (March 16, 2024) to catch the sunrise and planned to invite friends. She was excited about the trip."
}}

Return only the JSON object, do not add any other text:
"""

USER_EPISODE_GENERATION_PROMPT = """
**CRITICAL LANGUAGE RULE**: You MUST output in the SAME language as the input conversation content. If the conversation content is in Chinese, ALL output (title and content) MUST be in Chinese. If in English, output in English. This is mandatory.

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
    - ANY non-absolute time reference must be resolved to an absolute date using the timestamp of the specific message that contains it — NOT the conversation start time
    - Examples: "yesterday", "last Friday", "last summer", "recently" — any granularity
    - Preserve both the original expression AND the resolved absolute date
    - Format: "original expression (absolute date)" — e.g., "last summer (summer 2022)", "last Friday (July 21, 2023)"

9.  **TITLE LENGTH AND PREFIX**:
    - Keep the title within 15-25 words
    - Include the `{user_name}` prefix as shown in the schema below

10. **NARRATIVE FORM**:
    - Use third-person throughout (consistent with principle 1)
    - Maintain chronological order and causal relationships between events. Note: this refers to inter-event causality (e.g., "after listening to suggestions → decided to leave Saturday"), not {user_name}'s role in those events (principle 1 still governs role attribution).

Please generate a structured episodic memory and return only a JSON object containing the following two fields:
{{
    "title": "(Record of [core event] about `{user_name}`, YYYY-MM-DD)",
    "content": "(A {user_name}-centred objective, coherent narrative.)"
}}

Example (incorrect style — too subjective):
{{
    "title": "{user_name} led the weekend hiking planning",
    "content": "{user_name} initiated the plan to hike Mount Rainier this weekend and led the gear discussion. He/she made the final call on departure time and arranged to invite friends."
}}

Example (correct style — objective recording):
{{
    "title": "{user_name} participating in weekend Mount Rainier hike planning (2024-03-14)",
    "content": "{user_name} proposed hiking Mount Rainier this weekend in the conversation and asked for gear suggestions. After listening to others' advice on hiking boots and warm clothing, {user_name} decided to leave early Saturday morning (2024-03-16) to catch the sunrise, and mentioned inviting friends along."
}}

Self-check list:
- Is the return strictly in JSON format?
- Do both `title` and `content` strictly revolve around `{user_name}`?
- Is `content` a fluent experiential narrative, not a heap of scattered facts?
- Are all key points woven naturally into the narrative?

**CRITICAL LANGUAGE RULE**: You MUST output in the SAME language as the input conversation content. If the conversation content is in Chinese, ALL output (title and content) MUST be in Chinese. If in English, output in English. This is mandatory.

Return only the JSON object, do not add any other text:
"""
