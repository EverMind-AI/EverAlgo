"""English prompt for ProfileExtractor.aextract."""

PROFILE_EXTRACT_PROMPT_EN = """You are a user-profile synthesis expert. Given a current conversation slice (MemCell) and a list of summaries from the same user's prior MemCells, produce a long-term profile snapshot.

Current conversation:
{current_memcell_text}

Prior conversation cluster (chronological summaries):
{cluster_summaries}

Current timestamp (Unix epoch ms): {timestamp}

Instructions:
1. Synthesize **long-term, stable user traits** — not one-off events, not anticipated commitments.
   - In scope: interests, habits, communication style, recurring preferences, skills, recurring topics, decision-making patterns.
   - Out of scope: discrete events ("Alice scheduled a meeting"), single commitments ("user will send draft Friday").
2. Required output fields:
   - id: a stable unique identifier (e.g. "pf_<owner_id>" or random "pf_<random>").
   - owner_id: the user this profile describes (default "u_default" if unclear).
   - summary: a one-paragraph narrative profile (3-6 sentences).
   - timestamp: use the current timestamp passed above.
3. Optional fields — you MAY emit any of the following as extra JSON keys when supported by evidence in the cluster; otherwise omit them entirely (do NOT emit empty placeholders):
   - interests: list[str]
   - habits: list[str]
   - preferences: dict[str, str]
   - hard_skills: list[str]
   - communication_style: str
   - decision_patterns: list[str]
4. If the cluster is empty, base the profile solely on the current MemCell and acknowledge the limited evidence in the summary.

Output format (JSON only, no prose):
{{
  "id": "<string>",
  "owner_id": "<string>",
  "summary": "<one-paragraph profile narrative>",
  "timestamp": <int>
}}

You may add additional keys (interests / habits / preferences / hard_skills / communication_style / decision_patterns) when supported. Do NOT emit parent_id or parent_type — Profile is user-level aggregate, not per-MemCell.
"""
