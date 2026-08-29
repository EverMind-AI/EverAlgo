"""Cross-language text width in ASCII-equivalent units.

Shared by profile (item width backstop) and episode (summary width cap). Lives in its
own module so neither extractor imports the other for a yardstick.
"""

from __future__ import annotations

import unicodedata


def ascii_width(text: str) -> int:
    """Length in ASCII-equivalent units: East Asian Wide/Fullwidth characters count 2, the rest 1.

    One cross-language yardstick for text length — 200 units reads as one to two short sentences
    in English and in CJK alike, with no per-language threshold table to maintain.
    """
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)
