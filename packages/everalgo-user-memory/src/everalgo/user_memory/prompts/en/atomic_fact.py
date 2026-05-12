"""English prompt for AtomicFactExtractor.aextract."""

ATOMIC_FACT_EXTRACT_PROMPT_EN = """You are an atomic-fact extraction expert. Given a conversation slice (MemCell), extract single, verifiable assertions about what happened or what is true.

Conversation:
{memcell_text}

Conversation timestamp (Unix epoch ms): {timestamp}

Instructions:
1. Identify each atomic fact — a single verifiable assertion that stands on its own.
2. Do NOT emit:
   - compound claims (split them into separate facts),
   - opinions / preferences / emotional states,
   - hypotheticals or future intents (those belong to Foresight),
   - sweeping generalizations not grounded in the conversation.
3. Phrase each fact in third person, present-or-past tense, with explicit subject.
   - Good: "Alice scheduled a 3pm meeting with Bob on 2024-03-14."
   - Bad: "They had a chat." / "Alice is a nice colleague."
4. Use the conversation timestamp as the time anchor.
5. Generate a unique id (e.g. "af_<random>").
6. Use a stable owner_id from the conversation (default "u_default" if unclear).
7. Return an empty list if no atomic facts are present.

Output format (JSON only, no prose):
{{
  "atomic_facts": [
    {{
      "id": "<string>",
      "owner_id": "<string>",
      "fact": "<string>",
      "timestamp": <int>
    }}
  ]
}}

Note: parent_type and parent_id will be auto-filled by the caller; do not emit them.
"""
