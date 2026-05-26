"""Typed wrapper around ``tqdm.asyncio.tqdm.gather``.

``tqdm`` ships no type stubs, so every direct call site needs ``# type: ignore``
plus follow-up suppression for ``reportUnknownMemberType`` /
``reportUnknownVariableType``. Funneling the ignores through one generic helper
keeps the stage code typed end-to-end (``list[T]`` flows through) and confines
the stub suppressions to a single line.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Awaitable


async def gather_with_progress[T](
    *coros: Awaitable[T],
    desc: str,
    unit: str = "it",
) -> list[T]:
    """Run ``coros`` concurrently with a tqdm progress bar.

    Mirrors ``asyncio.gather`` semantics but renders a tqdm progress bar driven
    by completion order. Concurrency is bounded by the caller (typically via an
    ``asyncio.Semaphore`` inside each coroutine), not by this helper.
    """
    from tqdm.asyncio import tqdm as async_tqdm  # type: ignore[import-untyped]

    return cast(
        "list[T]",
        await async_tqdm.gather(  # pyright: ignore[reportUnknownMemberType]
            *coros, desc=desc, unit=unit, dynamic_ncols=True
        ),
    )
