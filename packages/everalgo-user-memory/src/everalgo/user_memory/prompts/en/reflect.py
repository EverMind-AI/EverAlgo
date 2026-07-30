"""English prompts for EpisodeReflector.

Constants:
    - ``REFLECT_EPISODE_PROMPT`` — full merge from N chronological episodes. Placeholder: ``{timeline}``.
    - ``REFLECT_EPISODE_UPDATE_PROMPT`` — incremental update of an existing narrative.
      Placeholders: ``{old_episode}`` / ``{new_episodes}``.

Output schema (both variants): ``{"content": str, "title": str}`` via Structured Output.

Both variants merge already-extracted episodes, so they inherit the language those episodes were written
in rather than judging it: the mixed-input judgement belongs to the extractor that read the raw
conversation. See ``prompts/en/episode.py`` for that judgement.
"""

REFLECT_EPISODE_PROMPT = """\
**CRITICAL LANGUAGE RULE**: You MUST write ALL output in the SAME language as the episodes you are merging. Merging never changes the language — do not translate. This is mandatory.

You are a memory consolidation assistant.

Below are the following episode summaries about the same topic, listed chronologically.
Each episode was written at a point in time with limited context. You can now
see the full timeline and produce a narrative that is more accurate and complete
than any individual episode.

Merge them into a single coherent narrative that:
- Preserves ALL factual details: names, dates, locations, specific actions, quantities, and status changes
- Resolves contradictions by keeping the latest state
- Maintains chronological flow with dates preserved
- Keeps every time exactly as the episodes wrote it. Every absolute time that states a clock time MUST carry the UTC zone label ("2024-03-14 15:00 UTC", never "2024-03-14 15:00" and never a bare "15:00"); a date with no clock time needs none. Do NOT reformat, convert, or drop a time the episodes already carry — no episode may lose its time in the merge
- Removes redundant information
- Ends with a brief summary of the current state as of the latest episode

**CRITICAL LANGUAGE RULE**: You MUST write ALL output in the SAME language as the episodes you are merging. Merging never changes the language — do not translate. This is mandatory.

Episodes:
{timeline}"""

REFLECT_EPISODE_UPDATE_PROMPT = """\
**CRITICAL LANGUAGE RULE**: You MUST write ALL output in the SAME language as the existing narrative you are updating. Updating never changes the language — do not translate, even if the new episodes are written in a different language. This is mandatory.

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
- Keep every time exactly as the narrative and the new episodes wrote it. Every absolute time that states a clock time MUST carry the UTC zone label ("2024-03-14 15:00 UTC", never "2024-03-14 15:00" and never a bare "15:00"); a date with no clock time needs none. Do NOT reformat, convert, or drop a time that is already there
- End with an updated summary of the current state

**CRITICAL LANGUAGE RULE**: You MUST write ALL output in the SAME language as the existing narrative you are updating. Updating never changes the language — do not translate, even if the new episodes are written in a different language. This is mandatory."""
