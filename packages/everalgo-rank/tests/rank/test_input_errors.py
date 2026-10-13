"""The caller-input checks raise ``InvalidInputError``, with the offender located and the remedy named."""

from __future__ import annotations

import pytest

from everalgo.rank import RankConfig, skill
from everalgo.rank.agentic import _format_docs
from everalgo.types import Candidate, RankInput
from everalgo.utils import InvalidInputError


def test_a_candidate_without_episode_text_is_named_by_id() -> None:
    candidate = Candidate(id="cand-7", score=0.5, metadata={"episode": {"subject": "s", "content": ""}})

    with pytest.raises(InvalidInputError, match=r"candidate 'cand-7' has an empty metadata\['episode'\]\['content'\]"):
        _format_docs([candidate])


async def test_rerank_without_an_llm_says_what_to_pass() -> None:
    rank_input = RankInput(
        query="q",
        memory_type="skill",
        sparse_candidates=[],
        dense_candidates=[Candidate(id="c1", score=0.5, metadata={})],
        top_k=1,
    )

    with pytest.raises(InvalidInputError, match="set enable_rerank=False"):
        await skill.arank(rank_input, config=RankConfig(fusion_mode="rrf"), enable_rerank=True)
