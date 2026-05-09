"""Stub schema existence + minimal-instantiation tests."""


def test_foresight_atomic_fact_profile_importable() -> None:
    from evercore.types import AtomicFact, Foresight, Profile

    assert AtomicFact().id == ""
    assert Foresight().id == ""
    assert Profile().id == ""


def test_agent_types_importable() -> None:
    from evercore.types import AgentCase, AgentSkill

    assert AgentCase().id == ""
    assert AgentSkill().id == ""


def test_raw_types_importable() -> None:
    from evercore.types import RawData, RawFile

    assert RawData().id == ""
    assert RawFile(uri="").uri == ""


def test_parsed_and_knowledge_importable() -> None:
    from evercore.types import KnowledgeMemory, ParsedContent

    assert ParsedContent().id == ""
    assert KnowledgeMemory().id == ""


def test_rank_io_importable() -> None:
    from evercore.types import RankInput, RankOutput

    assert RankInput().memory_type == ""
    assert RankOutput().items == []
