"""Chinese clustering prompts — re-export of the English template.

The clustering LLM template is language-neutral: responses match the candidate corpus' language while
the template itself works either way. This module re-exports the English constant so callers that
import from ``prompts/zh/`` get the same prompt without duplication.
"""

from everalgo.clustering.prompts.en.cluster import CLUSTER_LLM_ASSIGN_PROMPT

__all__ = ["CLUSTER_LLM_ASSIGN_PROMPT"]
