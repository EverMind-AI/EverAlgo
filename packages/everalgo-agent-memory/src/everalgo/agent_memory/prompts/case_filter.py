"""Prompt for AgentCaseExtractor LLM filter (step 7)."""

AGENT_CASE_FILTER_PROMPT = """Determine whether this agent interaction is worth extracting as a reusable problem-solving experience. Apply a HIGH threshold — only extract interactions that clearly demonstrate a complete problem-solving process.

The interaction may be pure conversation OR contain a single round of tool calls. Both types must meet the same quality bar.

Conversation:
{messages}

SKIP (return {{"worth_extracting": false}}) — default unless clearly valuable:
- Casual chitchat, greetings, small talk
- Opinion/preference exchange with no actionable outcome
- Simple factual Q&A (e.g., "What is X?" with a direct answer)
- Single-turn Q&A (one user message + one assistant response)
- Multi-turn but loosely related topics (user asks unrelated follow-ups, no progressive deepening)
- Information gathering without problem resolution (user asks questions but no concrete problem is solved)
- Generic advice that anyone could give (e.g., "try restarting", "check the docs")
- Lifestyle or personal preference conversations (e.g., activity planning, movie/book/food discussions, hobby sharing, travel recommendations) — these are inherently personal and lack a transferable problem-solving methodology
- Emotional support or empathetic conversations without a concrete, replicable resolution strategy
- Single-round tool call that performs a trivial lookup or simple data fetch without meaningful reasoning, diagnosis, or multi-step problem solving (e.g., a single search call that returns a direct answer)
- Conversations where the assistant only uses basic conversational competence — structured lists, follow-up questions, empathetic acknowledgment, curated recommendations, topic transitions. These are standard LLM dialogue patterns, NOT domain-specific problem-solving expertise
- Recommendation or suggestion conversations (e.g., "what movie should I watch", "what should I cook", "where should I travel") — even if multi-turn with progressive refinement, unless the recommendation requires specialized technical diagnosis or domain expertise beyond general knowledge
- Conversations where the "problem" is merely a personal decision or preference choice (e.g., choosing a hiking trail, picking a recipe, selecting a gift) rather than a technical/analytical problem with objectively verifiable steps

EXTRACT (return {{"worth_extracting": true}}) — ALL conditions must be met:
1. A specific, concrete problem is identified (not vague or generic)
2. The conversation shows progressive deepening: each turn builds on the previous, narrowing down the solution
3. The problem reaches a resolution or clear actionable conclusion
4. The approach involves non-trivial reasoning, diagnosis, or domain expertise that would be valuable to replay
5. The solution methodology is transferable — a different agent facing a similar problem class could follow the same steps to reach a similar outcome (personal taste, lifestyle advice, and subjective recommendations do NOT qualify)
6. The extracted skill would go BEYOND baseline LLM capabilities — ask: "Would a competent LLM without this experience handle this significantly worse?" If the answer is no, it is not worth extracting. Examples of baseline capabilities that do NOT qualify: making structured suggestion lists, asking clarifying questions, showing empathy, giving general-knowledge recommendations, summarizing information

**Borderline cases** — when you are unsure, lean toward SKIP.

Return JSON:
{{"worth_extracting": true/false, "reason": "1 sentence, less than 20 tokens"}}
"""
