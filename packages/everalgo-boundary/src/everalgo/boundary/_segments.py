"""Where a conversation is cut: the index arithmetic of both paths, with no model call.

The model only ever sees ``ChatMessage`` items, so a batch cut is first decided over the chat messages alone and
then carried over to the full item list: a cut before a chat message falls right before it, which keeps the tool
items between two chat messages with the earlier one; tool items ahead of the first chat message join the first
segment; a cut after the last chat message closes the stream and takes the trailing tool items with it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from everalgo.boundary._render import _count_history_tokens
from everalgo.types import ChatMessage

if TYPE_CHECKING:
    from collections.abc import Sequence

    from everalgo.types import ConversationItem


# ---------------------------------------------------------------------------
# Batch path
# ---------------------------------------------------------------------------


def _chat_view(items: Sequence[ConversationItem]) -> tuple[list[int], list[ChatMessage]]:
    """The ``ChatMessage`` items in order, and the position of each in ``items``."""
    chat = [(position, item) for position, item in enumerate(items) if isinstance(item, ChatMessage)]
    return [position for position, _ in chat], [message for _, message in chat]


def _force_split_cuts(messages: Sequence[ChatMessage], hard_token_limit: int, hard_message_limit: int) -> list[int]:
    """Cuts that close leading segments until what is left fits under both hard limits, ascending.

    The last cut, if any, is where the window handed to the model starts. A window of one message is never cut
    further, however long it is.
    """
    cuts: list[int] = []
    start = 0
    while len(messages) - start > 1 and _exceeds_limits(messages[start:], hard_token_limit, hard_message_limit):
        start += _find_force_split_point(messages[start:], hard_token_limit, hard_message_limit)
        cuts.append(start)
    return cuts


def _exceeds_limits(messages: Sequence[ChatMessage], hard_token_limit: int, hard_message_limit: int) -> bool:
    """Whether ``messages`` reach either hard limit."""
    return _count_history_tokens(messages) >= hard_token_limit or len(messages) >= hard_message_limit


def _find_force_split_point(messages: Sequence[ChatMessage], hard_token_limit: int, hard_message_limit: int) -> int:
    """Binary-halving search for a force-split index; floor 1 so even a single very-long message is accepted."""
    if len(messages) <= 1:
        return len(messages)
    candidate = min(hard_message_limit - 1, len(messages) - 1)
    while candidate > 1 and _count_history_tokens(messages[:candidate]) >= hard_token_limit:
        candidate = max(1, candidate // 2)
    return candidate


def _item_cuts(chat_cuts: Sequence[int], chat_positions: Sequence[int], n_items: int) -> list[int]:
    """Carry cuts over the chat messages to positions among all ``n_items`` items.

    Args:
        chat_cuts: Ascending cut points over the chat messages, each in ``1..len(chat_positions)``; a cut at ``k``
            falls between the ``k``-th chat message and the next.
        chat_positions: The positions ``_chat_view(items)`` returns.
        n_items: ``len(items)``.

    Returns:
        Each cut as the position of the first item after it; a cut after the last chat message is ``n_items``.
    """
    return [chat_positions[cut] if cut < len(chat_positions) else n_items for cut in chat_cuts]


# ---------------------------------------------------------------------------
# Incremental (step) path
# ---------------------------------------------------------------------------


def _needs_force_split(
    history: Sequence[ChatMessage], new: ChatMessage, hard_token_limit: int, hard_message_limit: int
) -> bool:
    """Whether ``history + [new]`` reaches a hard limit with enough history to close a segment without the model.

    With fewer than two history messages the model still decides, so a single oversized message is never cut alone.
    """
    reaches_limit = _count_history_tokens([*history, new]) >= hard_token_limit or len(history) + 1 >= hard_message_limit
    return reaches_limit and len(history) >= 2
