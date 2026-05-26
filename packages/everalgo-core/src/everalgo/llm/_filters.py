"""Logging filters for ``everalgo.llm`` — redacts sensitive header values."""

from __future__ import annotations

import logging
import re
from typing import Final, override

__all__ = ["SensitiveHeadersFilter"]


_REDACTED: Final[str] = "<redacted>"

# Matches header / parameter names that conventionally carry credentials.
# Case-insensitive; covers Authorization, Api-Key / api_key, X-Api-Key,
# Bearer tokens. Keep narrow — false positives leak data; false negatives
# leak credentials.
_SENSITIVE_KEY_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(authorization|api[-_]?key|x-api-key|bearer)",
    re.IGNORECASE,
)


class SensitiveHeadersFilter(logging.Filter):
    """Redact sensitive header values in ``LogRecord.args`` mappings.

    Default-attached to ``everalgo.llm``; ``filter()`` always returns ``True`` — it mutates payload, never
    drops records. Only mapping-shaped ``record.args`` are scanned; tuple / ``None`` args pass through.
    Does not scan request/response bodies (body PII is out of scope for a header filter).
    """

    @override
    def filter(self, record: logging.LogRecord) -> bool:
        """Redact sensitive entries from ``record.args`` in place; always returns ``True``."""
        if isinstance(record.args, dict):
            record.args = {
                key: (_REDACTED if _SENSITIVE_KEY_PATTERN.search(str(key)) else value)
                for key, value in record.args.items()
            }
        return True
