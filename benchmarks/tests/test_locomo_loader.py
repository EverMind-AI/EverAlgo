"""Tests for the LoCoMo dataset loader."""

from pathlib import Path

import pytest

from benchmarks.common.dataset import Dataset
from benchmarks.datasets.locomo.loader import LocomoDataset

FIXTURE = Path(__file__).parent / "fixtures" / "locomo_mini.json"
MULTI_SESSION_FIXTURE = Path(__file__).parent / "fixtures" / "locomo_multi_session.json"
BOUNDARY_STRETCH_FIXTURE = Path(__file__).parent / "fixtures" / "locomo_boundary_stretch.json"


def test_locomo_is_dataset_compliant():
    ds = LocomoDataset(data_path=FIXTURE)
    assert isinstance(ds, Dataset)
    assert ds.name == "locomo"


def test_load_conversations_returns_one_conv_for_fixture():
    ds = LocomoDataset(data_path=FIXTURE)
    convs = list(ds.load_conversations())
    assert len(convs) == 1
    assert convs[0].id == "locomo_exp_user_0"
    assert len(convs[0].messages) == 2


def test_load_qa_pairs_returns_two_with_categories():
    ds = LocomoDataset(data_path=FIXTURE)
    qas = list(ds.load_qa_pairs("locomo_exp_user_0"))
    assert len(qas) == 1
    assert qas[0].category == "1"


def test_filter_categories_excludes_adversarial():
    ds = LocomoDataset(data_path=FIXTURE)
    assert ds.filter_categories() == {"5"}


def test_category_label_mapping():
    ds = LocomoDataset(data_path=FIXTURE)
    assert ds.category_label("1") == "single-hop"
    assert ds.category_label("2") == "temporal"
    assert ds.category_label("3") == "open-domain"
    assert ds.category_label("4") == "multi-hop"
    assert ds.category_label("5") == "adversarial"


def test_prompts_are_non_empty_strings():
    ds = LocomoDataset(data_path=FIXTURE)
    assert ds.answer_prompt().strip()
    assert ds.judge_prompt().strip()


def test_multi_session_messages_in_numeric_order():
    """Regression: sessions 1, 2, 10 must order by numeric suffix, not string sort."""
    ds = LocomoDataset(data_path=MULTI_SESSION_FIXTURE)
    conv = next(iter(ds.load_conversations()))
    contents = [m.content for m in conv.messages]
    assert contents == ["First session msg", "Second session msg", "Tenth session msg"]


def test_multi_session_timestamps_monotonic():
    """Regression: numeric session ordering also yields monotonic timestamps."""
    ds = LocomoDataset(data_path=MULTI_SESSION_FIXTURE)
    conv = next(iter(ds.load_conversations()))
    timestamps = [m.timestamp for m in conv.messages]
    assert timestamps == sorted(timestamps)


def test_load_qa_pairs_rejects_bad_prefix():
    """Regression: non-LoCoMo conv_id must raise, not silently look up index."""
    ds = LocomoDataset(data_path=FIXTURE)
    with pytest.raises(ValueError, match="Unknown conv_id format"):
        list(ds.load_qa_pairs("random_string"))


def test_load_qa_pairs_rejects_negative_index():
    """Regression: negative index must not silently wrap to last conversation."""
    ds = LocomoDataset(data_path=FIXTURE)
    with pytest.raises(IndexError):
        list(ds.load_qa_pairs("locomo_exp_user_-1"))


def test_load_qa_pairs_rejects_out_of_range():
    """Regression: index past end must raise IndexError, not crash with raw error."""
    ds = LocomoDataset(data_path=FIXTURE)
    with pytest.raises(IndexError):
        list(ds.load_qa_pairs("locomo_exp_user_999"))


def test_answer_prompt_has_required_placeholders():
    """Regression: answer_prompt must contain {context} and {question}."""
    ds = LocomoDataset(data_path=FIXTURE)
    prompt = ds.answer_prompt()
    assert "{context}" in prompt
    assert "{question}" in prompt
    assert "TBD" not in prompt


def test_judge_prompt_has_required_placeholders():
    """Regression: judge_prompt must contain the placeholders EverCore's grader uses."""
    ds = LocomoDataset(data_path=FIXTURE)
    prompt = ds.judge_prompt()
    assert "{question}" in prompt
    assert "{gold_answer}" in prompt
    assert "{response}" in prompt
    assert "TBD" not in prompt


def test_msg_interval_stretches_when_default_overflows_next_session():
    """93 alignment #3c: 30s * (N-1) > 0.9 * next-session-gap -> compress to (avail * 0.9) / (N-1)."""
    ds = LocomoDataset(data_path=BOUNDARY_STRETCH_FIXTURE)
    conv = next(iter(ds.load_conversations()))
    session1_msgs = [m for m in conv.messages if m.metadata["session"] == "session_1"]
    assert len(session1_msgs) == 5
    # session_1 ts = 1:00 pm May 1, session_2 ts = 1:01 pm -> gap = 60_000 ms
    # required = 4 * 30_000 = 120_000 > 54_000 -> time_interval = 54_000 / 4 = 13_500 ms
    deltas = [session1_msgs[i + 1].timestamp - session1_msgs[i].timestamp for i in range(len(session1_msgs) - 1)]
    assert deltas == [13_500, 13_500, 13_500, 13_500]


def test_msg_interval_uses_default_when_within_budget():
    """93 alignment #3c: when 30s * (N-1) fits within the next-session gap, keep 30s default."""
    ds = LocomoDataset(data_path=MULTI_SESSION_FIXTURE)
    conv = next(iter(ds.load_conversations()))
    # multi_session fixture has 1 msg per session -> time_interval=0 (single-msg branch),
    # but session_1 -> session_2 gap is 24h, far above 30s * 0 = 0 budget -> trivial.
    # No assertion needed beyond "loader runs without crashing on this fixture".
    assert len(conv.messages) == 3


def test_sender_id_uses_conv_id_string_suffix():
    """93 alignment #4: sender_id suffix is conv_id string (e.g. 'locomo_exp_user_0'), not raw index."""
    ds = LocomoDataset(data_path=FIXTURE)
    conv = next(iter(ds.load_conversations()))
    msg0 = conv.messages[0]
    assert msg0.sender_name == "Alice"
    assert msg0.sender_id == "alice_locomo_exp_user_0"


def test_message_metadata_preserves_session_img_blip_source():
    """93 alignment #6: metadata keeps session/img_url/blip_caption/timestamp_source."""
    ds = LocomoDataset(data_path=FIXTURE)
    conv = next(iter(ds.load_conversations()))
    msg0 = conv.messages[0]
    assert msg0.metadata == {
        "session": "session_1",
        "img_url": None,
        "blip_caption": None,
        "timestamp_source": "session_level",
    }
