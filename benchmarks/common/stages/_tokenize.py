"""NLTK tokenization helpers shared by the Index and Search stages."""

from __future__ import annotations

from typing import Any

import nltk  # type: ignore[import-untyped]
from nltk.tokenize import word_tokenize  # type: ignore[import-untyped]


def ensure_nltk() -> None:
    """Download required NLTK data if not present."""
    for find_path, download_id in [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
    ]:
        try:
            nltk.data.find(find_path)  # type: ignore[no-untyped-call]
        except LookupError:
            nltk.download(download_id, quiet=True)  # type: ignore[no-untyped-call]


def tokenize(text: str, stemmer: Any, stop_words: set[str]) -> list[str]:
    """Lower -> tokenize -> keep alpha words len>=2 not stopword -> stem.

    Must be identical between index-time (``index.py``) and query-time (``search.py``).

    Args:
        text: Raw text to tokenize.
        stemmer: A stemmer with a ``stem(token)`` method (e.g. ``PorterStemmer``).
        stop_words: Set of lowercase stopwords to exclude.

    Returns:
        List of stemmed tokens.
    """
    if not text:
        return []
    tokens: list[str] = word_tokenize(text.lower())  # type: ignore[no-untyped-call]
    return [str(stemmer.stem(t)) for t in tokens if t.isalpha() and len(t) >= 2 and t not in stop_words]
