"""English clustering prompts — verbatim port from opensource.

Source: ``opensource/evermemos-opensource/src/memory_layer/prompts/en/agent_prompts.py:421-445``
(``AGENT_CLUSTER_LLM_ASSIGN_PROMPT``). Renamed to drop the ``AGENT_`` prefix because EverAlgo
``clustering`` is a neutral operator consumed by both ``user_memory`` and ``agent_memory`` packages —
naming should not lean on either side.

Placeholders (rendered via :py:meth:`str.format`):
    - ``{memcell_text}`` — text representation of the new event (caller-supplied; typically the
      ``task_intent`` or ``episode`` body).
    - ``{clusters_json}`` — JSON list of candidate clusters as
      ``[{"cluster_id": ..., "item_count": ..., "recent_task_intents": [...]}]``.
    - ``{next_new_id}`` — three-digit zero-padded suffix the LLM should use if it decides to create a
      new cluster (e.g. ``"007"`` → ``"cluster_007"``).

Output schema (the LLM must return strictly): ``{"cluster_id": str, "reason": str}``.
"""

CLUSTER_LLM_ASSIGN_PROMPT = """You are a clustering expert. Your goal is to group similar and related tasks together so that patterns and reusable strategies can be extracted from each cluster. Assign the new task intent to an existing cluster, or create a new one if no existing cluster fits.

[How to decide]
The goal of clustering is to group cases that would produce a **specific, actionable skill** — not generic advice. Use this test: "Would an agent who solved one task in this cluster have a **concrete advantage** (reusable tools, domain knowledge, verified strategies) when facing the other tasks?"

1. **Identify two dimensions**: the task's **subject domain** (e.g., medical research, urban planning, e-commerce) and its **problem-solving pattern** (e.g., root cause analysis, constraint satisfaction, data pipeline design).
2. **Cluster by the more specific dimension**. If the domain is already narrow (e.g., "clinical trial data extraction"), domain alone is enough. If the domain is broad (e.g., "software engineering"), use the problem-solving pattern to differentiate (e.g., "performance profiling" vs. "schema migration").
3. **Do NOT merge across unrelated domains just because the strategy is similar.** "Diagnose a patient's symptoms via differential diagnosis" and "diagnose a supply chain bottleneck via constraint analysis" both use diagnostic reasoning, but involve completely different domain knowledge and belong in separate clusters.
4. Scan candidate clusters. Prefer the cluster whose existing items would **benefit most from sharing a skill** with the new task.
5. Create a new cluster only when no candidate cluster is a good fit.

[Candidate Clusters]
Each cluster is represented by its cluster_id, item_count, and most recent task intents.
{clusters_json}

[New Task Intent]
{memcell_text}

[Rules]
- Output decision as JSON. Keep "reason" under 50 tokens.
- To assign: use an existing cluster_id. To create new: use "cluster_{next_new_id}".

Return ONLY valid JSON (no markdown fences, no explanation):
{{"cluster_id": "<existing_cluster_id or cluster_{next_new_id}>", "reason": "short reason"}}
"""
