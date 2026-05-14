"""Stub schema existence + minimal-instantiation tests."""


def test_foresight_importable_with_required_fields() -> None:
    """Foresight is no longer a stub (MR 1); minimal-instantiation needs required fields."""
    from everalgo.types import Foresight

    fs = Foresight(
        id="fs_001",
        owner_id="u1",
        foresight="x",
        evidence="x",
        timestamp=1,
        parent_id="mc_001",
    )
    assert fs.id == "fs_001"
    assert fs.parent_type == "memcell"


def test_atomic_fact_importable_with_required_fields() -> None:
    """AtomicFact is no longer a stub (MR 2); minimal-instantiation needs required fields."""
    from everalgo.types import AtomicFact

    af = AtomicFact(
        id="af_001",
        owner_id="u1",
        fact="x",
        timestamp=1,
        parent_id="mc_001",
    )
    assert af.id == "af_001"
    assert af.parent_type == "memcell"


def test_profile_importable_with_required_fields() -> None:
    """Profile is no longer a stub (MR 3); minimal-instantiation needs required fields."""
    from everalgo.types import Profile

    pf = Profile(
        id="pf_001",
        owner_id="u1",
        summary="s",
        timestamp=1,
    )
    assert pf.id == "pf_001"
    # User-level aggregate — no parent_type / parent_id fields exist on the model.
    assert not hasattr(pf, "parent_type")


def test_agent_types_importable() -> None:
    """AgentCase / AgentSkill are no longer stubs (sub-project 4); fields are required."""
    from everalgo.types import AgentCase, AgentSkill

    case = AgentCase(
        id="c_001",
        timestamp=1,
        parent_id="mc_001",
        task_intent="solve X",
    )
    assert case.id == "c_001"
    assert case.parent_type == "memcell"
    assert case.quality_score == 0.5

    skill = AgentSkill(id="s_001", cluster_id="cl_001")
    assert skill.id == "s_001"
    assert skill.confidence == 0.0
    assert skill.maturity_score == 0.6
    assert "vector" not in skill.model_dump()  # AgentSkill no longer carries vector fields


def test_raw_types_importable() -> None:
    from everalgo.types import RawData, RawFile

    assert RawData().id == ""
    assert RawFile(uri="").uri == ""


def test_parsed_and_knowledge_importable() -> None:
    from everalgo.types import KnowledgeMemory, ParsedContent

    assert ParsedContent().id == ""
    assert KnowledgeMemory().id == ""


def test_rank_io_importable() -> None:
    from everalgo.types import (
        Candidate,
        FactCandidate,
        RankInput,
        RankOutput,
        ScoredItem,
    )

    rank_input = RankInput(query="hello", memory_type="episodic")
    assert rank_input.query == "hello"
    assert rank_input.memory_type == "episodic"
    assert rank_input.sparse_candidates == []
    assert rank_input.dense_candidates == []
    assert rank_input.episode_to_facts == {}
    assert rank_input.top_k == 10

    assert RankOutput().items == []
    assert Candidate(id="c1", score=0.5).source == "other"
    assert FactCandidate(id="f1", parent_episode_id="ep1", score=0.42).score == 0.42
    assert ScoredItem(id="x", score=0.9, item_type="episode").metadata == {}
