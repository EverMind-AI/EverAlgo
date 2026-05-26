"""LoCoMo dataset loader.

Parses snap-research/locomo ``locomo10.json`` format into the benchmark's
``Conversation`` / ``QAPair`` value types. Adversarial questions (category 5)
are reported via ``filter_categories`` for the scoring layer to exclude.
"""

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from benchmarks.common.dataset import Conversation, Message, QAPair
from benchmarks.datasets.locomo.prompts import ANSWER_PROMPT, JUDGE_PROMPT, JUDGE_SYSTEM_PROMPT

_CATEGORY_LABELS: Final = {
    "1": "single-hop",
    "2": "temporal",
    "3": "open-domain",
    "4": "multi-hop",
    "5": "adversarial",
}


class LocomoDataset:
    """LoCoMo benchmark dataset adapter."""

    name = "locomo"

    def __init__(self, data_path: Path) -> None:
        self._data_path = data_path
        with data_path.open(encoding="utf-8") as f:
            self._raw = json.load(f)

    def load_conversations(self) -> Iterable[Conversation]:
        """Yield one ``Conversation`` per entry in the raw LoCoMo file."""
        for i, entry in enumerate(self._raw):
            conv = entry["conversation"]
            speakers = (conv.get("speaker_a", "A"), conv.get("speaker_b", "B"))
            conv_id = f"locomo_exp_user_{i}"

            messages: list[Message] = []
            session_keys = sorted(
                [k for k in conv if k.startswith("session_") and not k.endswith("_date_time")],
                key=_session_index,
            )
            # Pre-parse all session timestamps so #3c can look ahead to next session.
            session_timestamps = [_parse_timestamp(conv.get(f"{key}_date_time")) for key in session_keys]

            for session_idx, key in enumerate(session_keys):
                session_msgs = conv[key]
                if not session_msgs:
                    continue
                current_session_ts = session_timestamps[session_idx]
                next_ts = session_timestamps[session_idx + 1] if session_idx + 1 < len(session_keys) else None
                time_interval = _compute_msg_interval_ms(len(session_msgs), current_session_ts, next_ts)

                for i_in_session, msg in enumerate(session_msgs):
                    speaker = msg.get("speaker", "")
                    content = msg.get("text", "")
                    # Prepend image caption when a message carries ``img_url`` so the
                    # downstream LLMs (BoundaryDetector / Episode / AtomicFact) see the
                    # visual cue. ~15.5% of LoCoMo messages have ``img_url``; dropping
                    # them loses real signal for retrieval and answer generation.
                    if msg.get("img_url"):
                        blip_caption = msg.get("blip_caption", "an image")
                        content = f"[{speaker} shared an image: {blip_caption}] {content}"
                    messages.append(
                        Message(
                            id=msg["dia_id"],
                            role="user",  # LoCoMo has no system/assistant distinction
                            content=content,
                            timestamp=int(current_session_ts + i_in_session * time_interval),
                            # 93 alignment #4: suffix is conv_id string, not raw list index.
                            # ``f"{speaker.lower().replace(' ','_')}_{conv_id}"``.
                            sender_id=f"{speaker.lower().replace(' ', '_')}_{conv_id}",
                            sender_name=speaker,
                            # 93 alignment #6: preserve session / img / blip / timestamp_source.
                            # dia_id is already stored as Message.id; not duplicated here.
                            metadata={
                                "session": key,
                                "img_url": msg.get("img_url"),
                                "blip_caption": msg.get("blip_caption"),
                                "timestamp_source": "session_level",
                            },
                        )
                    )

            yield Conversation(
                id=conv_id,
                speakers=speakers,
                messages=tuple(messages),
            )

    def load_qa_pairs(self, conv_id: str) -> Iterable[QAPair]:
        """Yield all scorable QA pairs attached to the given conversation id.

        Filters out categories listed in :meth:`filter_categories` (LoCoMo
        adversarial category ``"5"``) at the loader boundary so that no
        downstream stage burns LLM / embedding / reranker budget on questions
        that will never be scored. ``question_id`` preserves the original
        ``qa{N}`` index from the raw file for joinability with EverCore artifacts.
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
