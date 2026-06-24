"""Fixed test set for Reflection evaluation.

Each test case defines a topic line with specific sessions from a LoCoMo conversation,
the Reflection capability dimensions it tests, and the fixed QA list for evaluation.
Session lists match the TOML config's [session_filter] section.
QA lists are pre-filtered: Category 1-4 only, answers confirmed in selected sessions.
"""

from __future__ import annotations

from typing import Any

TEST_CASES: list[dict[str, Any]] = [
    {
        "id": "andrew_pets_apartment",
        "conv": 5,
        "sessions": [1, 2, 5, 7, 10, 12, 18, 24, 27, 28],
        "dimensions": ["cumulative_chain", "deduplication"],
        "description": "Andrew pet acquisition 0→3 + repeated apartment search",
    },
    {
        "id": "john_career",
        "conv": 2,
        "sessions": [3, 8, 19, 25, 28],
        "dimensions": ["conflict_resolution"],
        "description": "John exam fail→pass, promotion→job loss",
    },
    {
        "id": "nate_tournaments_pets",
        "conv": 3,
        "sessions": [1, 2, 6, 10, 12, 14, 17, 19, 20, 22, 27, 28],
        "dimensions": ["cumulative_chain", "deduplication", "entity_integration"],
        "description": "Nate 9 tournaments / 7 wins + pet accumulation",
    },
    {
        "id": "evan_sam_multi",
        "conv": 8,
        "sessions": [1, 2, 4, 5, 6, 7, 8, 12, 13, 14, 16, 18, 19, 20, 21, 22, 23, 24],
        "dimensions": ["conflict_resolution", "cumulative_chain", "entity_integration", "deduplication"],
        "description": "Evan Prius arc + relationship chain + Sam health oscillation",
    },
    {
        "id": "deborah_anna",
        "conv": 7,
        "sessions": [3, 4, 8, 10, 15, 19],
        "dimensions": ["entity_integration"],
        "description": "Deborah-Anna relationship scattered across sessions",
    },
]

# QA lists will be populated by a one-time data exploration script that:
# 1. Loads locomo10.json QAs for each test case's conv
# 2. Filters to Cat 1-4
# 3. Verifies answer exists in selected sessions' content
# 4. Writes back the fixed list here
# For now, QA filtering happens at evaluation time based on category + evidence_num
FILTER_CATEGORIES: set[str] = {"1", "2", "3", "4"}
