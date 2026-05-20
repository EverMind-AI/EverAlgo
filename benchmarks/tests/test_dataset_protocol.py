import pytest
from pydantic import ValidationError

from benchmarks.common.dataset import Conversation, Dataset, Message, QAPair


def test_message_is_immutable():
    """frozen=True must prevent post-construction mutation."""
    m = Message(id="m1", role="user", content="hi", timestamp=1, sender_id="u_a", sender_name="A")
    assert m.id == "m1"
    with pytest.raises(ValidationError):
        m.content = "mutated"  # type: ignore[misc]


def test_qa_pair_required_fields():
    qa = QAPair(question_id="q1", conv_id="c1", question="?", golden_answer="A", category="1")
    assert qa.category == "1"


def test_qa_pair_metadata_is_per_instance():
    """Guard against regression to default={} (which would share dict across instances)."""
    a = QAPair(question_id="q1", conv_id="c1", question="?", golden_answer="A", category="1")
    b = QAPair(question_id="q2", conv_id="c1", question="?", golden_answer="A", category="1")
    assert a.metadata is not b.metadata


def test_conversation_holds_messages():
    m = Message(id="m1", role="user", content="hi", timestamp=1, sender_id="u_a")
    conv = Conversation(id="c1", speakers=("u_a",), messages=(m,))
    assert conv.messages[0].id == "m1"


def test_dataset_protocol_compiles():
    """Smoke test that Dataset Protocol has all expected methods."""
    methods = {
        "load_conversations",
        "load_qa_pairs",
        "category_label",
        "filter_categories",
        "judge_prompt",
        "answer_prompt",
    }
    assert methods <= set(dir(Dataset))
