"""Stub schema existence + minimal-instantiation tests."""


def test_foresight_importable_with_required_fields() -> None:
    """Foresight is no longer a stub (MR 1); minimal-instantiation needs required fields."""
    from everalgo.types import Foresight

    fs = Foresight(
        owner_id="u1",
        foresight="x",
        evidence="x",
        timestamp=1,
    )
    assert fs.owner_id == "u1"
    assert not hasattr(fs, "parent_type")


def test_atomic_fact_importable_with_required_fields() -> None:
    """AtomicFact is no longer a stub (MR 2); minimal-instantiation needs required fields."""
    from everalgo.types import AtomicFact

    af = AtomicFact(
        owner_id="u1",
        content="x",
        timestamp=1,
    )
    assert af.owner_id == "u1"
    assert not hasattr(af, "parent_type")


def test_profile_importable_with_required_fields() -> None:
    """Profile is no longer a stub (MR 3); minimal-instantiation needs required fields."""
    from everalgo.types import Profile

    pf = Profile(
        owner_id="u1",
        summary="s",
        timestamp=1,
    )
    # User-level aggregate — no parent_type / parent_id fields exist on the model.
    assert not hasattr(pf, "parent_type")


def test_decision_importable_with_required_fields() -> None:
    from everalgo.types import Decision

    dc = Decision(
        owner_id="u1",
        title="t",
        decision="d",
        reason="r",
        timestamp=1,
    )
    assert dc.owner_id == "u1"
    assert dc.tags == []
    assert not hasattr(dc, "parent_type")
    assert not hasattr(dc, "parent_id")
    assert not hasattr(dc, "session_id")


def test_principle_importable_with_required_fields() -> None:
    from everalgo.types import Principle

    pr = Principle(
        owner_id="u1",
        title="t",
        statement="s",
        timestamp=1,
    )
    assert pr.owner_id == "u1"
    assert pr.source_entry_ids == []
    assert not hasattr(pr, "parent_type")
    assert not hasattr(pr, "parent_id")
    assert not hasattr(pr, "session_id")


def test_agent_types_importable() -> None:
    """AgentCase / AgentSkill are no longer stubs (sub-project 4); fields are required."""
    from everalgo.types import AgentCase, AgentSkill

    case = AgentCase(
        id="c_001",
        timestamp=1,
        task_intent="solve X",
    )
    assert case.id == "c_001"
    assert not hasattr(case, "parent_type")
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


def test_parsed_importable() -> None:
    from everalgo.types import Modality, ParsedContent

    # ParsedContent gained a concrete schema in the parser migration (no more `id` TBD).
    pc = ParsedContent()
    assert pc.text == ""
    assert pc.modality is Modality.UNKNOWN


def test_knowledge_importable_with_required_fields() -> None:
    """KnowledgeMemory is no longer a stub; minimal-instantiation needs required fields."""
    from everalgo.types import KnowledgeMemory

    km = KnowledgeMemory(
        doc_id="doc1",
        topic_index=0,
        topic="Root Title",
        summary="s",
        depth=0,
        topic_path="Root Title",
    )
    assert km.doc_id == "doc1"
    assert km.topic_index == 0
    # Root convention: parent is None, content / refs / labels default to empty.
    assert km.parent_index is None
    assert km.content == ""
    assert km.block_refs == []
    assert km.children_index == []
    assert km.content_labels == []


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
