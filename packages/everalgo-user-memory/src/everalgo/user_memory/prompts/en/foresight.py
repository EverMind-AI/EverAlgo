"""English prompt for ForesightExtractor.aextract."""

FORESIGHT_EXTRACT_PROMPT_EN = """You are a foresight prediction expert. Given a conversation slice (MemCell), extract anticipated future events or commitments mentioned by the participants.

Conversation:
{memcell_text}

Conversation timestamp (Unix epoch ms): {timestamp}

Instructions:
1. Identify each anticipated future event — explicit commitments ("I'll do X by Friday"), implicit plans ("we should review Y next week"), open intents ("hoping to ship Z this quarter").
2. For each foresight, record:
   - foresight: a third-person summary of the anticipated event.
   - evidence: the conversation phrasing that signals it (a short quote or paraphrase).
3. Use the conversation timestamp as the anchor; do not invent specific future timestamps.
4. Generate a unique id (e.g. "fs_<random>").
5. Use a stable owner_id from the conversation (default "u_default" if unclear).
6. Return an empty list if no foresights are present.

Output format (JSON only, no prose):
{{
  "foresights": [
    {{
      "id": "<string>",
      "owner_id": "<string>",
      "foresight": "<string>",
      "evidence": "<string>",
      "timestamp": <int>
    }}
  ]
}}

Note: parent_type and parent_id will be auto-filled by the caller; do not emit them.
"""
