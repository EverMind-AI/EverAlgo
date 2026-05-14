"""Prompt for AgentSkillExtractor success path (max(quality_score) >= 0.5 cases)."""

AGENT_SKILL_SUCCESS_EXTRACT_PROMPT = """You are an expert at extracting reusable problem-solving strategies from concrete agent task cases.

You will receive:
1. **New case(s)** from a cluster of semantically similar tasks — all with quality_score >= 0.5.
2. **Existing skills** previously extracted for this cluster (each with an index number; may be empty).

Your job: distill **actionable strategies** into reusable **Skills** via incremental operations. Maintain as few skills as the evidence warrants.

**What makes a GOOD skill:**
- Reasoning principles WITH concrete patterns: teaches HOW to think, not just what to do
- Decision branches that cover the different problem variants seen across cases
- A FEW well-chosen examples that illustrate distinct branches — not an exhaustive catalog

**What makes a BAD skill:**
- Too abstract: "Analyze constraints" without showing what analysis looks like in practice
- Too narrow: A single solution template that only works for one exact case
- **Bloated**: Listing dozens of case-specific details (names, dates, institutions, compounds, etc.) inside parentheses or comma-separated lists. Each How/Decision/e.g. field should contain 1-2 illustrative examples, NOT an inventory of every case seen

**Field-level requirements:**

- **description** (HARD LIMIT: max 150 tokens, must be under 500 characters):
  - One-sentence summary of the **abstract problem class** this skill solves — describe the general pattern, NOT specific cases.
  - Do NOT list multiple scenarios, entity names, or case-specific details.
  - Append `Keywords:` with up to 10 general terms (no specific names, numbers, or case-specific phrases).
  - Example: "Identifies academic researchers by cross-referencing biographical constraints with publication records. Keywords: researcher identification, biographical verification, publication matching, academic search"

- **content** (max 2000 tokens): Markdown format:
  ```markdown
  ## Steps
  1. <reasoning action — what to think about, not just what to do>
     - How: <principle explaining WHY this step works>
     - Decision: If <condition A> → <action>; If <condition B> → <alternative>
     - e.g., <specific real example with entity names/numbers from cases>
     - Check: <what to verify before proceeding>
  2. ...

  ## Pitfalls <- ONLY from actual failed steps in cases; otherwise OMIT
  - <specific mistake from a real case> — <what went wrong and how to avoid>
  ```

  **HARD RULES for content:**
  - **Max 5 steps.**
  - **Max 2 examples per step.** Each example MUST be a SHORT, single-sentence illustration of a distinct decision branch. Do NOT list multiple sub-examples inside parentheses or comma-separated lists.
  - **Decision branches**: REQUIRED when the next action depends on what was found. For linear steps with no branching, Decision may be omitted. Each Decision should have at most 3 branches.
  - **Max 4 pitfalls.** When adding a new one beyond 4, replace the most generic existing pitfall.
  - **No parenthetical catalogs**: FORBIDDEN to stuff dozens of case-specific terms (names, dates, compounds, institutions, etc.) inside a single parenthetical `(e.g., X, Y, Z, ...)`. Keep each field concise — generalize the pattern, illustrate with 1-2 examples only.

[New AgentCase(s) to integrate]
{new_case_json}

[Existing skills for this cluster](Each item has an index number)
{existing_skills_json}

[Task]
Analyze the new case(s) and output a list of operations (add / update / none).

[Operation Guide — follow in order]

**Step 1: Overlap Check (mandatory before every add/update decision)**
For each new case, compare against each existing skill:
  a. List the core steps of the new case's approach (the main actions that drove the outcome).
  b. List the core steps of the existing skill.
  c. Count how many of the new case's core steps are already covered by the existing skill.
  d. Compute coverage = (covered steps) / (total core steps in new case).
  e. Conclusion:
     - Coverage >= 60% → the case falls within this skill's problem pattern → **update** candidate.
     - Coverage < 60% → different problem pattern → **add** candidate.
     - If uncertain, default to **update**.

**Step 2: Execute the decided operation**

- **add**: The new case tackles a **different problem pattern** (coverage < 60% against all existing skills). Create a new skill. confidence = `0.5`.

- **update**: The new case overlaps an existing skill (coverage >= 60%). Enrich it with new Decision branches, better examples, or sharper How explanations.
  - You MAY substantially rewrite content (restructure steps, replace examples, refine How explanations), but **preserve existing verified content unless the new case directly contradicts it**.
  - **CRITICAL: The updated content MUST stay within 2000 tokens. Do NOT simply append new content — replace weaker examples with stronger ones, merge redundant steps, and compress prose. If the existing content is already long, aggressively condense it while preserving the core logic.**
  - **CRITICAL: The updated description MUST stay under 500 characters. Generalize — do NOT accumulate case-specific details.**
  - **Hypothesis promotion rule**: If the existing skill contains `## Potential Steps`, treat this update as a **promotion** — rewrite as `## Steps` using the new case as primary source. confidence = `0.6`.
  - **Confidence-only update**: If the new case merely confirms the existing skill without adding new decision logic or better examples, bump confidence only.

- **none**: Trivially duplicate — no new decision branches, no new examples worth keeping, no confidence change needed.

[Confidence Anchoring Rules]
- **New skill (add)**: confidence = `0.5`
- **Promoted skill (hypothesis → verified)**: confidence = `0.6`
- **Update with new decision branch**: confidence = existing + `0.1` (cap 0.95)
- **Confirming update (no new logic)**: confidence = existing + `0.05` (cap 0.95)
- **Contradicting case**: confidence = existing - `0.2`; add contradiction to Pitfalls; optionally add a new skill if the contradiction reveals a genuinely different pattern

**CRITICAL LANGUAGE RULE**: Output in the SAME language as the input conversation content.

[Output Format]
```json
{{
  "operations": [
    {{"action": "add", "data": {{"name": "Short descriptive name (max 10 words)", "description": "...", "content": "## Steps\\n1. ...", "confidence": 0.5}}}},
    {{"action": "update", "index": 0, "data": {{"content": "...", "confidence": 0.7}}}},
    {{"action": "update", "index": 1, "data": {{"confidence": 0.65}}}},
    {{"action": "none"}}
  ],
  "update_note": "Overlap check: new case core steps=[X, Y, Z]. skill[0] covers X and Y (67% overlap) → update. ..."
}}
```
"""
