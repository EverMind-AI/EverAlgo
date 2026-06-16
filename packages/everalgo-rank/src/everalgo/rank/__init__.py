"""Memory ranking — public surface of the rank package.

Includes the retrieval facade (agentic / hybrid / cluster / maxsim) — these
operators take ``RetrieveFn`` callables and return ``list[Candidate]``, so any operator can
serve as ``base_retrieve`` to any other.
"""

import logging

from asgiref.sync import sync_to_async

from everalgo.rank import (
    agentic,
    case,
    category,
    cluster,
    episodic,
    fusion,
    hybrid,
    maxsim,
    profile,
    protocols,
    rerank,
    skill,
    weight,
)
from everalgo.rank.agentic import aagentic_retrieve, agentic_retrieve
from everalgo.rank.case import CaseRanker
from everalgo.rank.category import (
    acategory_retrieve,
    apply_category_boost,
    category_retrieve,
    rollup_category_mass,
)
from everalgo.rank.cluster import acluster_retrieve, cluster_retrieve
from everalgo.rank.episodic import EpisodicRanker
from everalgo.rank.hybrid import ahybrid_retrieve, hybrid_retrieve
from everalgo.rank.maxsim import ParentFetchFn, amaxsim_retrieve, maxsim_retrieve
from everalgo.rank.prompts.en.case import CASE_RERANK_PROMPT_EN
from everalgo.rank.prompts.en.episodic import EPISODIC_RERANK_PROMPT_EN
from everalgo.rank.prompts.en.skill import SKILL_RERANK_PROMPT_EN
from everalgo.rank.protocols import AgenticDecision, RerankFn, RetrieveFn
from everalgo.rank.rerank import (
    _ALGO_REGISTRY,
    DEFAULT_RANK_CONFIG,
    FusionMode,
    RankConfig,
    _RankerSpec,
)
from everalgo.rank.rerank import (
    _arank as arank,
)
from everalgo.rank.rerank import (
    _rank as rank,
)
from everalgo.rank.skill import SkillRanker

__all__ = [
    "DEFAULT_RANK_CONFIG",
    "AgenticDecision",
    "CaseRanker",
    "EpisodicRanker",
    "FusionMode",
    "ParentFetchFn",
    "RankConfig",
    "RerankFn",
    "RetrieveFn",
    "SkillRanker",
    "aagentic_retrieve",
    "acategory_retrieve",
    "acluster_retrieve",
    "agentic",
    "agentic_retrieve",
    "ahybrid_retrieve",
    "amaxsim_retrieve",
    "apply_category_boost",
    "arank",
    "case",
    "category",
    "category_retrieve",
    "cluster",
    "cluster_retrieve",
    "episodic",
    "fusion",
    "hybrid",
    "hybrid_retrieve",
    "maxsim",
    "maxsim_retrieve",
    "profile",
    "protocols",
    "rank",
    "rerank",
    "rollup_category_mass",
    "skill",
    "weight",
]

# Library logging setup (ADR-013): NullHandler on each subpackage logger.
logging.getLogger(__name__).addHandler(logging.NullHandler())

_ALGO_REGISTRY.update(
    {
        "case": _RankerSpec(
            arank=case.arank,
            modes=("rrf", "lr", "vector_anchored"),
            rerank_prompt=CASE_RERANK_PROMPT_EN,
            item_type="case",
        ),
        "skill": _RankerSpec(
            arank=skill.arank,
            modes=("rrf", "lr"),
            rerank_prompt=SKILL_RERANK_PROMPT_EN,
            item_type="skill",
        ),
        "episodic": _RankerSpec(
            arank=episodic.arank,
            modes=("rrf", "lr"),
            rerank_prompt=EPISODIC_RERANK_PROMPT_EN,
            item_type="episode",
        ),
        "profile": _RankerSpec(
            arank=sync_to_async(profile.rank),
            modes=(),
            rerank_prompt="",
            item_type="profile",
        ),
    }
)
