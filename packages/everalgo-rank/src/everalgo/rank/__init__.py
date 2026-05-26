"""Memory ranking — public surface of the rank package."""

import logging

from asgiref.sync import sync_to_async

from everalgo.rank import case, episodic, fusion, profile, rerank, skill, weight
from everalgo.rank.case import CaseRanker
from everalgo.rank.episodic import EpisodicRanker
from everalgo.rank.prompts.en.case import CASE_RERANK_PROMPT_EN
from everalgo.rank.prompts.en.episodic import EPISODIC_RERANK_PROMPT_EN
from everalgo.rank.prompts.en.skill import SKILL_RERANK_PROMPT_EN
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
    "CaseRanker",
    "EpisodicRanker",
    "FusionMode",
    "RankConfig",
    "SkillRanker",
    "arank",
    "case",
    "episodic",
    "fusion",
    "profile",
    "rank",
    "rerank",
    "skill",
    "weight",
]

# Library logging setup (ADR-013): NullHandler on each subpackage logger.
logging.getLogger(__name__).addHandler(logging.NullHandler())

_ALGO_REGISTRY.update(
    {
        "case": _RankerSpec(
            arank=case.arank,
            modes=("rrf", "lr", "vector_anchored", "agentic"),
            rerank_prompt=CASE_RERANK_PROMPT_EN,
            item_type="case",
        ),
        "skill": _RankerSpec(
            arank=skill.arank,
            modes=("rrf", "lr", "agentic"),
            rerank_prompt=SKILL_RERANK_PROMPT_EN,
            item_type="skill",
        ),
        "episodic": _RankerSpec(
            arank=episodic.arank,
            modes=("rrf", "lr", "mrag", "agentic"),
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
