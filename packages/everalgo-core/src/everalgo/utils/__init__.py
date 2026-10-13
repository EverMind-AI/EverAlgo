"""General-purpose tools any EverAlgo package may use, one kind per module.

Prompt strings are not tools: they live next to their algorithm, at ``<subpkg>/prompts/{en,zh}/<name>.py``
(AGENTS.md §5).
"""

from __future__ import annotations

__all__ = [
    "InvalidInputError",
    "ascii_width",
    "count_tokens",
    "decode_bytes",
    "encode",
    "fetch_uri",
    "format_utc_date",
    "format_utc_time",
    "json_default",
    "paragraphs",
    "parse_utc_date",
    "render_chat_line",
    "render_prompt",
    "speaker",
    "tagged",
    "truncate_text",
]

import logging

from everalgo.utils.chat_lines import render_chat_line, speaker
from everalgo.utils.errors import InvalidInputError
from everalgo.utils.http import fetch_uri
from everalgo.utils.json import json_default
from everalgo.utils.markup import paragraphs, tagged
from everalgo.utils.templates import render_prompt
from everalgo.utils.text import ascii_width, decode_bytes
from everalgo.utils.timestamps import format_utc_date, format_utc_time, parse_utc_date
from everalgo.utils.tokens import count_tokens, encode, truncate_text

# Library logging setup (ADR-013): NullHandler on each subpackage logger.
logging.getLogger(__name__).addHandler(logging.NullHandler())
