"""EpisodeExtractor — MemCell → Episode (user memory write path).

Shows how a ``MemCell`` built directly from two ``ChatMessage`` objects flows
through ``EpisodeExtractor.aextract(mc, sender_id=...)`` to produce a single
``Episode`` with ``owner_id``, ``subject``, ``episode`` text, and ``timestamp``.

Uses ``FakeLLMClient`` so no API key is needed.

Run:
    uv run python examples/03_user_memory_episode.py
"""

from __future__ import annotations

import asyncio
import json

from everalgo.llm.types import ChatResponse
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import ChatMessage, Episode, MemCell
from everalgo.user_memory.episode import EpisodeExtractor

# ---------------------------------------------------------------------------
# Scripted LLM response — {title, content} shape parsed by EpisodeExtractor.
# ---------------------------------------------------------------------------

_EPISODE_JSON = json.dumps(
    {
        "title": "Alice asks about Python async retry semantics",
        "content": (
            "Alice initiated a discussion on Python async retry patterns. "
            "The assistant offered to send a follow-up document the next week."
        ),
    }
)


def _make_memcell() -> MemCell:
    """Two-turn dialogue: user asks about async retries, assistant acknowledges."""
    return MemCell(
        items=[
            ChatMessage(
                id="m1",
                role="user",
                content="I've been getting into Python async lately. Can you walk me through retry semantics?",
                timestamp=1_700_000_000_000,
                sender_id="u_alice",
                sender_name="Alice",
            ),
            ChatMessage(
                id="m2",
                role="assistant",
                content="Sure — what failure mode are you seeing? I'll send a follow-up doc next week.",
                timestamp=1_700_000_001_000,
                sender_id="assistant",
            ),
        ],
        timestamp=1_700_000_001_000,
    )


async def main() -> None:
    """Extract a single Episode from a MemCell and print its fields."""
    fake = FakeLLMClient(responses=[ChatResponse(content=_EPISODE_JSON, model="fake")])
    mc = _make_memcell()

    episode: Episode = await EpisodeExtractor(llm=fake).aextract(mc, sender_id="u_alice")

    print(f"owner_id  : {episode.owner_id!r}")
    print(f"subject   : {episode.subject!r}")
    print(f"episode   : {episode.episode!r}")
    print(f"timestamp : {episode.timestamp}")


if __name__ == "__main__":
    asyncio.run(main())
