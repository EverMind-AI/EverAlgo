"""English clustering prompts for LLM-based cluster assignment.

Placeholders (rendered via :py:meth:`str.format`):
    - ``{memcell_text}`` — text representation of the new event (caller-supplied; typically the
      preview[0] of the incoming Cluster).
    - ``{clusters_json}`` — JSON list of candidate clusters as
      ``[{"idx": int, "count": int, "preview": [...]}]``.

Output schema (the LLM must return strictly): ``{"idx": int, "reason": str}``.
Use ``{"idx": -1, ...}`` to signal a new cluster (sentinel; avoids null ambiguity).
"""

CLUSTER_LLM_ASSIGN_PROMPT = """You are a clustering expert. Your goal is to group similar and related tasks together so that patterns and reusable strategies can be extracted from each cluster. Assign the new task intent to an existing cluster, or signal a new cluster if no existing cluster fits.

[How to decide]
The goal of clustering is to group cases that would produce a **specific, actionable skill** — not generic advice. Use this test: "Would an agent who solved one task in this cluster have a **concrete advantage** (reusable tools, domain knowledge, verified strategies) when facing the other tasks?"

1. **Identify two dimensions**: the task's **subject domain** (e.g., medical research, urban planning, e-commerce) and its **problem-solving pattern** (e.g., root cause analysis, constraint satisfaction, data pipeline design).
2. **Cluster by the more specific dimension**. If the domain is already narrow (e.g., "clinical trial data extraction"), domain alone is enough. If the domain is broad (e.g., "software engineering"), use the problem-solving pattern to differentiate (e.g., "performance profiling" vs. "schema migration").
3. **Do NOT merge across unrelated domains just because the strategy is similar.** "Diagnose a patient's symptoms via differential diagnosis" and "diagnose a supply chain bottleneck via constraint analysis" both use diagnostic reasoning, but involve completely different domain knowledge and belong in separate clusters.
4. Scan candidate clusters. Prefer the cluster whose existing items would **benefit most from sharing a skill** with the new task.
5. Signal a new cluster (idx: -1) only when no candidate cluster is a good fit.

[Candidate Clusters]
Each cluster is represented by its list index (idx), item count, and recent preview texts.
{clusters_json}

[New Task Intent]
{memcell_text}

[Rules]
- Output decision as JSON. Keep "reason" under 50 tokens.
- To assign to an existing cluster: use its idx (non-negative integer).
- To create a new cluster: use idx -1.

Return ONLY valid JSON (no markdown fences, no explanation):
{{"idx": <existing_idx or -1>, "reason": "short reason"}}
"""
