"""The episode prompt of the EverAlgo LoCoMo reference run, pinned verbatim as a test fixture.

Source: ``git show d9fe22e:packages/everalgo-user-memory/src/everalgo/user_memory/prompts/en/episode.py``, the
commit the reference run used (everalgo-user-memory 0.3.0 by version number; its ``IMPORTANT TIME HANDLING`` block
differs from the ``everalgo-user-memory/v0.3.0`` tag's). Only the generic prompt and its default instructions are
kept, copied character for character with their line widths untouched: rewrapping a line changes what the run sent.
The template is ``str.format`` style, so ``{{`` and ``}}`` stand for single literal braces.

``MESSAGE_BLOCK`` and ``TIME_FORMAT`` are how that run rendered each message of ``{conversation}``: the f-string of
``_render_conversation`` in the same commit's ``everalgo/user_memory/episode.py`` with its expressions made the
fields ``timestamp``, ``speaker`` and ``content``, and the English format of
``everalgo.llm.format.format_natural_language_time``. The run joined the blocks with single newlines.
"""

from __future__ import annotations

__all__ = ["DEFAULT_CUSTOM_INSTRUCTIONS", "EPISODE_GENERATION_PROMPT", "MESSAGE_BLOCK", "TIME_FORMAT"]

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

MESSAGE_BLOCK = """
                {{
                    "timestamp": {timestamp},
                    "speaker": {speaker},
                    "content": {content}
                }}"""

TIME_FORMAT = "%B %d, %Y (%A) at %I:%M %p UTC"
