"""Chinese clustering prompts — byte-equivalent re-export of the English template.

Opensource's ``prompts/zh/cluster_prompts.py`` ships a stub comment "Cluster prompts moved to
agent_prompts.py", and the zh ``agent_prompts.py`` re-exports the en variant verbatim — confirming the
clustering LLM template is language-neutral (responses must match the candidate corpus' language; the
template itself works either way). EverAlgo mirrors that decision.
"""

from everalgo.clustering.prompts.en.cluster import CLUSTER_LLM_ASSIGN_PROMPT

__all__ = ["CLUSTER_LLM_ASSIGN_PROMPT"]
