"""English prompts for EpisodeReflector.

Constants:
    - ``REFLECT_EPISODE_PROMPT`` — full merge from N chronological episodes. Placeholder: ``{timeline}``.
    - ``REFLECT_EPISODE_UPDATE_PROMPT`` — incremental update of an existing narrative.
      Placeholders: ``{old_episode}`` / ``{new_episodes}``.

Output schema (both variants): ``{"content": str, "title": str}`` via Structured Output.
"""

REFLECT_EPISODE_PROMPT = """\
You are a memory consolidation assistant.

Below are the following episode summaries about the same topic, listed chronologically.
Each episode was written at a point in time with limited context. You can now
see the full timeline and produce a narrative that is more accurate and complete
than any individual episode.

Merge them into a single coherent narrative that:
- Preserves ALL factual details: names, dates, locations, specific actions, quantities, and status changes
- Resolves contradictions by keeping the latest state
- Maintains chronological flow with dates preserved
- Removes redundant information
- Ends with a brief summary of the current state as of the latest episode

Episodes:
{timeline}"""

REFLECT_EPISODE_UPDATE_PROMPT = """\
You are updating an existing memory narrative with new information.

Current narrative:
{old_episode}

New episodes (chronologically ordered):
{new_episodes}

Update the narrative to incorporate the new information:
- Correct any statements that are now outdated
- Append new events in chronological position
- Preserve content that is still accurate
- Maintain all factual details: names, dates, locations, specific actions
- End with an updated summary of the current state"""
