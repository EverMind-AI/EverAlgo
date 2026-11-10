"""Boundary detection for agent memory: ``everalgo.boundary``'s public surface, re-exported unchanged.

Agent trajectories mix ``ChatMessage`` with ``ToolCallRequest`` / ``ToolCallResult``; ``BoundaryDetector.adetect_batch``
takes them as they are, and its ``Cuts`` index the full trajectory, so ``to_memcells`` keeps the tool items in the
cells.
"""

from __future__ import annotations

__all__ = ["BoundaryDetector", "Cuts", "to_memcells"]

from everalgo.boundary import BoundaryDetector, Cuts, to_memcells
