"""English prompt for EpisodeExtractor.aextract."""

EPISODE_EXTRACT_PROMPT_EN = """You are an episodic memory generation expert. Given a conversation slice (MemCell), extract structured Episode memories.

Conversation:
{memcell_text}

Conversation timestamp (Unix epoch ms): {timestamp}

Instructions:
1. Identify each distinct episodic event — a complete "what happened" trace with participants, place, time, action, outcome.
2. Convert dialogue format into third-person narrative.
3. Preserve names, dates, locations, decisions, emotions.
4. Use the conversation timestamp as the episode time anchor.
5. Generate a unique id for each episode (e.g. "ep_<random>").
6. Use a stable owner_id from the conversation (default "u_default" if unclear).

Output format (JSON only, no prose):
{{
  "episodes": [
    {{
      "id": "<string>",
      "owner_id": "<string>",
      "episode": "<narrative text>",
      "timestamp": <int>
    }}
  ]
}}

Note: parent_type and parent_id will be auto-filled by the caller; do not emit them.
"""
