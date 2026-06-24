"""LoCoMo dataset loader.

Parses snap-research/locomo ``locomo10.json`` format into the benchmark's
``Conversation`` / ``QAPair`` value types. Adversarial questions (category 5)
are reported via ``filter_categories`` for the scoring layer to exclude.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

from benchmarks.common.dataset import Conversation, Message, QAPair
from benchmarks.datasets.locomo.prompts import ANSWER_PROMPT, JUDGE_PROMPT, JUDGE_SYSTEM_PROMPT

_CATEGORY_LABELS: Final = {
    "1": "single-hop",
    "2": "temporal",
    "3": "open-domain",
    "4": "multi-hop",
    "5": "adversarial",
}


def _parse_session_messages(
    session_msgs: list[dict[str, Any]],
    *,
    conv_id: str,
    session_key: str,
    base_ts: int,
    time_interval: float,
) -> list[Message]:
    """Parse raw LoCoMo session messages into ``Message`` objects."""
    result: list[Message] = []
    for i_in_session, msg in enumerate(session_msgs):
        speaker = msg.get("speaker", "")
        content = msg.get("text", "")
        if msg.get("img_url"):
            blip_caption = msg.get("blip_caption", "an image")
            content = f"[{speaker} shared an image: {blip_caption}] {content}"
        result.append(
            Message(
                id=msg["dia_id"],
                role="user",
                content=content,
                timestamp=int(base_ts + i_in_session * time_interval),
                sender_id=f"{speaker.lower().replace(' ', '_')}_{conv_id}",
                sender_name=speaker,
                metadata={
                    "session": session_key,
                    "img_url": msg.get("img_url"),
                    "blip_caption": msg.get("blip_caption"),
                    "timestamp_source": "session_level",
                },
            )
        )
    return result


class LocomoDataset:
    """LoCoMo benchmark dataset adapter.

    Args:
        data_path: Path to the ``locomo10.json`` file.
        session_filter: Optional mapping of conversation index to allowed session indices.
            When set, only the listed sessions are loaded; conversations absent from the
            dict yield empty ``Conversation`` objects to keep downstream indexes stable.
    """

    name = "locomo"

    def __init__(self, data_path: Path, *, session_filter: dict[int, list[int]] | None = None) -> None:
        self._data_path = data_path
        self._session_filter = session_filter
        with data_path.open(encoding="utf-8") as f:
            self._raw = json.load(f)

    def load_conversations(self) -> Iterable[Conversation]:
        """Yield one ``Conversation`` per entry in the raw LoCoMo file.

        When ``session_filter`` is set, only sessions listed for each conversation
        index are included. Conversations whose index is absent from the filter dict
        are yielded with zero messages (they are not skipped, so downstream stage
        indexes remain stable).
        """
        for i, entry in enumerate(self._raw):
            conv = entry["conversation"]
            speakers = (conv.get("speaker_a", "A"), conv.get("speaker_b", "B"))
            conv_id = f"locomo_exp_user_{i}"

            allowed_sessions: set[int] | None = None
            if self._session_filter is not None:
                if i not in self._session_filter:
                    yield Conversation(id=conv_id, speakers=speakers, messages=())
                    continue
                allowed_sessions = set(self._session_filter[i])

            messages: list[Message] = []
            session_keys = sorted(
                [k for k in conv if k.startswith("session_") and not k.endswith("_date_time")],
                key=_session_index,
            )
            session_timestamps = [_parse_timestamp(conv.get(f"{key}_date_time")) for key in session_keys]

            for session_idx, key in enumerate(session_keys):
                if allowed_sessions is not None and _session_index(key) not in allowed_sessions:
                    continue
                raw_msgs = conv[key]
                if not raw_msgs:
                    continue
                current_ts = session_timestamps[session_idx]
                next_ts = session_timestamps[session_idx + 1] if session_idx + 1 < len(session_keys) else None
                interval = _compute_msg_interval_ms(len(raw_msgs), current_ts, next_ts)
                messages.extend(
                    _parse_session_messages(
                        raw_msgs, conv_id=conv_id, session_key=key, base_ts=current_ts, time_interval=interval
                    )
                )

            yield Conversation(id=conv_id, speakers=speakers, messages=tuple(messages))

    def load_qa_pairs(self, conv_id: str) -> Iterable[QAPair]:
        """Yield all scorable QA pairs attached to the given conversation id.

        Filters out categories listed in :meth:`filter_categories` (LoCoMo
        adversarial category ``"5"``) at the loader boundary so that no
        downstream stage burns LLM / embedding / reranker budget on questions
        that will never be scored. ``question_id`` preserves the original
        ``qa{N}`` index from the raw file for joinability with upstream reference artifacts.
        """
        prefix = "locomo_exp_user_"
        if not conv_id.startswith(prefix):
            raise ValueError(f"Unknown conv_id format: {conv_id!r}")
        try:
            idx = int(conv_id.removeprefix(prefix))
        except ValueError as exc:
            raise ValueError(f"Malformed conv_id suffix: {conv_id!r}") from exc
        if idx < 0 or idx >= len(self._raw):
            raise IndexError(f"conv_id out of range: {conv_id!r}")
        entry = self._raw[idx]
        excluded = self.filter_categories()
        for qa_idx, qa in enumerate(entry.get("qa", [])):
            category = str(qa.get("category", ""))
            if category in excluded:
                continue
            yield QAPair(
                question_id=f"{conv_id}_qa{qa_idx}",
                conv_id=conv_id,
                question=qa.get("question", ""),
                golden_answer=str(qa.get("answer", "")),
                category=category,
            )

    def category_label(self, category: str) -> str:
        """Map a LoCoMo category code (``"1"``..``"5"``) to a human label."""
        return _CATEGORY_LABELS.get(category, f"unknown-{category}")

    def filter_categories(self) -> set[str]:
        """Return categories excluded from scoring (adversarial = ``"5"``)."""
        return {"5"}

    def judge_prompt(self) -> str:
        """LLM-judge prompt template for this dataset."""
        return JUDGE_PROMPT

    def judge_system_prompt(self) -> str:
        """System-role message sent before the judge user prompt.

        Splitting system from user gives the judge consistent role framing.
        """
        return JUDGE_SYSTEM_PROMPT

    def answer_prompt(self) -> str:
        """Answer-generation prompt template for this dataset."""
        return ANSWER_PROMPT


def _compute_msg_interval_ms(num_messages: int, current_ts: int, next_ts: int | None) -> float:
    """Compute per-message time interval in milliseconds for a session.

    Strategy: prefer 30 s (30_000 ms) between messages; compress only when the
    default would spill into the next session.  A 10 % buffer is kept so the
    last message of the current session never lands right on the start of the
    next one.

    Args:
        num_messages: Number of messages in the current session.
        current_ts: Current session start timestamp in milliseconds since epoch.
        next_ts: Next session start timestamp in milliseconds since epoch,
            or ``None`` when this is the last session.

    Returns:
        Interval in milliseconds (possibly fractional) to add per message index.
        Zero when ``num_messages <= 1``.
    """
    default_interval_ms: float = 30_000.0
    if num_messages <= 1:
        return 0.0
    required_duration = (num_messages - 1) * default_interval_ms
    if next_ts is None:
        return default_interval_ms
    available_duration = float(next_ts - current_ts)
    if available_duration <= 0:
        return default_interval_ms
    if required_duration > available_duration * 0.9:
        return (available_duration * 0.9) / (num_messages - 1)
    return default_interval_ms


def _session_index(key: str) -> int:
    """Extract numeric suffix from session_<N> for numeric sorting."""
    return int(key.split("_", 1)[1])


def _parse_timestamp(text: str | None) -> int:
    """Parse LoCoMo's '1:00 pm on 1 May, 2023' style into ms since epoch.

    Returns 0 on failure - timestamps are advisory metadata, not required.
    """
    if not text:
        return 0
    try:
        dt = datetime.strptime(text, "%I:%M %p on %d %B, %Y")
        return int(dt.replace(tzinfo=UTC).timestamp() * 1000)
    except ValueError:
        return 0
