"""English prompts for EpisodeExtractor: a system message of rules and a user message of data.

The text is the episode prompt of the EverAlgo LoCoMo reference run (everalgo commit d9fe22e), sentence for
sentence, laid out the way every other operator's prompt is: the rules in the system message, the data in the user
message, each block in an XML-style tag instead of a heading, static blocks before the call-level ones. Two things
differ from that run's text: the worked example names its start time in the shared time shape, and the user message
closes with a reminder (below). What moved:

- The SYSTEM message holds the role sentence, then ``<important_time_handling>``, ``<what_to_write>`` (the request
  for the two-field JSON object with its described skeleton), ``<requirements>`` (the twelve numbered
  requirements), ``<example>`` and ``<reply_format>`` (the closing "Return only the JSON object" line), in the
  reference run's order. Each tag names the heading it replaces; ``<what_to_write>`` and ``<reply_format>`` had no
  heading and take the names every operator uses for those blocks. :func:`assemble_system_prompt` closes the message
  with ``<custom_instructions>``: the caller's text, or the reference run's five default principles
  (:data:`DEFAULT_CUSTOM_INSTRUCTIONS`) when the caller passes none, as that run filled the slot. The block moves
  from the middle of the prompt to the end of the system message, where a caller's text always goes.
- The USER message carries the data: ``<conversation_start_time>`` (the first message's time) and
  ``<conversation>``, each message laid out as the reference run laid it out, an indented pseudo-JSON block whose
  values are unquoted. Every time is the shared prompt shape with its weekday, ``2024-03-14 15:00 UTC (Thursday)``
  (:func:`everalgo.utils.format_time`), in the caller's zone. The message closes with ``<reminder>``.

The reminder is a static block after the data: write the episode following every rule, give every relative time its
absolute date in parentheses, keep every fact, name, number, date and stated reason. With the rules read only before
the data, the model followed them less (an 80-memcell LoCoMo probe: shorter episodes, fewer resolved dates); the
system message, and so its prompt-cache prefix, is unchanged.

Three deliberate exceptions to the operator conventions: no ``<output_language>`` block unless the caller names a
language (the reference run had no language rule); no ``summary`` field in the reply, the episode's summary being the
content's first 200 characters, as the reference run filled it; and the ``<reminder>`` after the data, so the user
message does not hold data only.

Constants:
    - ``EPISODE_SUMMARY_FIELD_RULE`` — what a summary is, used by ``prompts.en.episode_summary`` for the summary
      written for an edited or backfilled episode. The extraction prompt no longer asks for a summary.
    - ``DEFAULT_CUSTOM_INSTRUCTIONS`` — the reference run's default principles, sent when the caller passes none.
    - ``EPISODE_SYSTEM_PROMPT`` — the static rules, the role sentence through ``<reply_format>``. No placeholder.
    - ``EPISODE_USER_PROMPT`` — the data, then the static ``<reminder>``. Placeholders: ``{{conversation_start_time}}``
      and ``{{conversation}}``.

Output schema: ``{"title": str, "content": str}``, held to ``episode._replies._EpisodeReply``, in the reference
run's order.
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_CUSTOM_INSTRUCTIONS",
    "EPISODE_SUMMARY_FIELD_RULE",
    "EPISODE_SYSTEM_PROMPT",
    "EPISODE_USER_PROMPT",
    "assemble_system_prompt",
]

from everalgo.utils import append_custom_instructions, paragraphs, tagged

EPISODE_SUMMARY_FIELD_RULE = (
    "A faithful preview of this episode in 1-3 short sentences (at most ~50 English words / ~100 Chinese characters). "
    "Compress, never restate: do not reuse content's sentences — write shorter new ones that name the main participants "
    "and what actually happened, including the outcome or decision reached. You may omit minor details. Do not refer "
    "to the record itself — no 'this conversation', 'the above', 'the user asked'."
)

DEFAULT_CUSTOM_INSTRUCTIONS = """\
Follow these principles when generating episodic memories:
1. Each episode should be a complete, independent story or event
2. Preserve all important information including names, time, location, emotions, etc.
3. Use declarative language to describe episodes, not dialogue format
4. Highlight key information and emotional changes
5. Ensure episode content is easy to retrieve later"""

_ROLE = (
    "You are an episodic memory generation expert. Please convert the following conversation into an episodic memory."
)

_IMPORTANT_TIME_HANDLING = """\
- "Conversation start time" indicates when this conversation began
- Each message has its own timestamp. ANY non-absolute time reference must be resolved to an absolute date using the timestamp of the specific message that contains it — NOT the conversation start time
- Examples: "yesterday", "last Friday", "last summer", "recently" — any granularity
- Preserve both the original expression AND the resolved absolute date
- Format: "original expression (absolute date)" — e.g., "last summer (summer 2022)", "last Friday (July 21, 2023)"
- This dual format supports both absolute and relative time-based questions"""

_WHAT_TO_WRITE = """\
Please generate a structured episodic memory and return only a JSON object containing the following two fields:
{
    "title": "A concise, descriptive title that accurately summarizes the theme (10-20 words)",
    "content": "A concise factual record of the conversation in third-person narrative. It must include all important information: who participated at what time, what was discussed, what decisions were made, what emotions were expressed, and what plans or outcomes were formed. Write it as a chronological account focusing on observable actions and direct statements. Remove redundant expressions and verbose descriptions while preserving all facts, entities (names, dates, locations), and specific details. Keep the content concise without losing key information. Resolve every non-absolute time reference using the individual message's timestamp, not the conversation start time."
}"""

_REQUIREMENTS = """\
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
   - Document repetition counts (e.g., "asked about the project status twice")"""

_EXAMPLE = """\
If the conversation start time is "2024-03-14 15:00 UTC (Thursday)" and the conversation is about Caroline planning to go hiking:
{
    "title": "Caroline's Mount Rainier Hiking Plan March 14, 2024: Weekend Adventure Planning Session",
    "content": "On March 14, 2024 at 3:00 PM UTC, Caroline expressed interest in hiking this weekend (March 16-17, 2024) and sought advice. She wanted to see the sunrise at Mount Rainier. When asked about gear by Melanie, Caroline received suggestions: hiking boots, warm clothing, flashlight, water, and high-energy food. Caroline decided to leave early Saturday morning (March 16, 2024) to catch the sunrise and planned to invite friends. She was excited about the trip."
}"""

_REPLY_FORMAT = "Return only the JSON object, do not add any other text:"

EPISODE_SYSTEM_PROMPT = paragraphs(
    _ROLE,
    tagged("important_time_handling", _IMPORTANT_TIME_HANDLING),
    tagged("what_to_write", _WHAT_TO_WRITE),
    tagged("requirements", _REQUIREMENTS),
    tagged("example", _EXAMPLE),
    tagged("reply_format", _REPLY_FORMAT),
)

_REMINDER = (
    "Write the episode for the conversation above, following every rule in the system message: give every relative "
    "time expression its absolute date in parentheses, and keep every fact, name, number, date and stated reason "
    "from the conversation."
)

EPISODE_USER_PROMPT = paragraphs(
    tagged("conversation_start_time", "{{conversation_start_time}}"),
    tagged("conversation", "{{conversation}}"),
    tagged("reminder", _REMINDER),
)


def assemble_system_prompt(*, custom_instructions: str | None, language_rule: str | None) -> str:
    """Build the system message: the static rules, the language rule when a language is named, then the instructions.

    Both values are placed verbatim and are never scanned for placeholders, so braces in them stay literal.

    Args:
        custom_instructions: The caller's instructions for the ``<custom_instructions>`` block, the message's last;
            ``None`` or a blank string sends :data:`DEFAULT_CUSTOM_INSTRUCTIONS` there instead, as the reference run
            did.
        language_rule: The ``<output_language>`` text for a language the caller named, as
            ``user_memory._language.build_language_rule`` returns it; ``None`` leaves the block out.

    Returns:
        The system message.
    """
    system = EPISODE_SYSTEM_PROMPT
    if language_rule is not None:
        system = paragraphs(system, tagged("output_language", language_rule))
    if custom_instructions is None or not custom_instructions.strip():
        custom_instructions = DEFAULT_CUSTOM_INSTRUCTIONS
    return append_custom_instructions(system, custom_instructions)
