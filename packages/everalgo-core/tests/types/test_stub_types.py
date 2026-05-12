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
    from everalgo.types import AgentCase, AgentSkill

    assert AgentCase().id == ""
    assert AgentSkill().id == ""


def test_raw_types_importable() -> None:
    from everalgo.types import RawData, RawFile

    assert RawData().id == ""
    assert RawFile(uri="").uri == ""


def test_parsed_and_knowledge_importable() -> None:
    from everalgo.types import KnowledgeMemory, ParsedContent

    assert ParsedContent().id == ""
    assert KnowledgeMemory().id == ""


def test_rank_io_importable() -> None:
    from everalgo.types import RankInput, RankOutput

    assert RankInput().memory_type == ""
    assert RankOutput().items == []
