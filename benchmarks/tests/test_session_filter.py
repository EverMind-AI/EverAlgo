"""Tests for LoCoMo session filter support."""

from __future__ import annotations

from pathlib import Path

from benchmarks.datasets.locomo.loader import LocomoDataset

FIXTURE = Path(__file__).parent / "fixtures" / "locomo_session_filter.json"


class TestSessionFilter:
    def test_no_filter_loads_all(self):
        """session_filter=None loads all messages from all sessions."""
        ds = LocomoDataset(data_path=FIXTURE)
        convs = list(ds.load_conversations())
        assert len(convs) == 2
        # Conv 0: 3 sessions x 1 msg each; Conv 1: 2 sessions x 1 msg each
        assert len(convs[0].messages) == 3
        assert len(convs[1].messages) == 2

    def test_filter_keeps_only_specified_sessions(self):
        """session_filter={0: [1, 2]} retains only sessions 1 and 2 from conv 0."""
        ds = LocomoDataset(data_path=FIXTURE, session_filter={0: [1, 2]})
        convs = list(ds.load_conversations())
        conv_0 = convs[0]
        assert len(conv_0.messages) == 2
        session_keys = [m.metadata["session"] for m in conv_0.messages]
        assert session_keys == ["session_1", "session_2"]

    def test_filter_excludes_unspecified_conv(self):
        """Conv index absent from session_filter is yielded with zero messages."""
        ds = LocomoDataset(data_path=FIXTURE, session_filter={0: [1]})
        convs = list(ds.load_conversations())
        # Conv 1 is not in the filter — yielded but with empty messages.
        assert len(convs) == 2
        conv_1 = convs[1]
        assert conv_1.id == "locomo_exp_user_1"
        assert len(conv_1.messages) == 0

    def test_filter_nonexistent_session_yields_empty_conv(self):
        """Filtering to a session id that does not exist yields 0 messages."""
        ds = LocomoDataset(data_path=FIXTURE, session_filter={0: [9999], 1: [9999]})
        convs = list(ds.load_conversations())
        assert len(convs) == 2
        assert len(convs[0].messages) == 0
        assert len(convs[1].messages) == 0

    def test_filter_preserves_message_metadata(self):
        """Filtered messages still carry the correct session key in metadata."""
        ds = LocomoDataset(data_path=FIXTURE, session_filter={0: [3], 1: [2]})
        convs = list(ds.load_conversations())
        assert convs[0].messages[0].metadata["session"] == "session_3"
        assert convs[1].messages[0].metadata["session"] == "session_2"

    def test_filter_multiple_convs(self):
        """session_filter can target different sessions in different conversations."""
        ds = LocomoDataset(data_path=FIXTURE, session_filter={0: [2, 3], 1: [1]})
        convs = list(ds.load_conversations())
        assert len(convs[0].messages) == 2
        assert len(convs[1].messages) == 1

    def test_filter_does_not_break_conv_ids(self):
        """Conversation IDs are stable regardless of session_filter."""
        ds = LocomoDataset(data_path=FIXTURE, session_filter={0: [1]})
        convs = list(ds.load_conversations())
        assert convs[0].id == "locomo_exp_user_0"
        assert convs[1].id == "locomo_exp_user_1"

    def test_empty_session_list_yields_empty_conv(self):
        """An empty session list for a conv index produces zero messages."""
        ds = LocomoDataset(data_path=FIXTURE, session_filter={0: [], 1: [1]})
        convs = list(ds.load_conversations())
        assert len(convs[0].messages) == 0
        assert len(convs[1].messages) == 1
