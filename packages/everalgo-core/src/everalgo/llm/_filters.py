"""Internal logging filters for ``everalgo.llm``.

The single export is ``SensitiveHeadersFilter``, attached at import time
to the ``everalgo.llm`` logger by ``everalgo/llm/__init__.py``. It redacts
authorization-style header values from ``LogRecord.args`` when those args
are supplied as a mapping (the form OpenAI / Anthropic / urllib3 use when
logging HTTP request metadata).

Mirrors ``openai/_utils/_logs.py`` ``SensitiveHeadersFilter`` and
``anthropic/_utils/_logs.py`` ``SensitiveHeadersFilter`` — same redaction
pattern, same default-on behaviour. See ADR-013 §6.
"""

from __future__ import annotations

import logging
import re
from typing import Final

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

    Default-attached to the ``everalgo.llm`` logger; cannot be disabled
    by configuration. The intent is fail-safe: a developer enabling
    ``DEBUG`` on ``everalgo.llm`` must never see live API keys in stderr.

    Filter Behaviour
    ----------------
    - ``record.args`` is a mapping: keys matching ``_SENSITIVE_KEY_PATTERN``
      have their values replaced with ``"<redacted>"``. Other entries pass
      through untouched.
    - ``record.args`` is a tuple / ``None``: untouched. Positional ``%``-format
      args (the standard ``logger.debug("foo=%s", foo)`` shape) carry no
      header semantics, so there is nothing to redact.
    - The record itself is never dropped — ``filter()`` always returns
      ``True``; this Filter mutates payload, it does not gate emission.

    Notes
    -----
    Does NOT scan request / response bodies. Body content is filtered by
    convention: ADR-013 §5 forbids logging full request / response bodies
    at ``DEBUG`` unless the user explicitly sets ``EVERALGO_LLM_LOG_BODY=1``.
    Body PII / model-echoed inputs are out of scope for a header filter.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Redact sensitive entries from ``record.args`` in place.

        Parameters
        ----------
        record : logging.LogRecord
            The record about to be handled. ``record.args`` may be a tuple,
            mapping, or ``None`` (see :py:meth:`logging.Logger.debug`).

        Returns
        -------
        bool
            Always ``True`` — the record is emitted; only its ``args`` payload
            may be mutated.
        """
        if isinstance(record.args, dict):
            record.args = {
                key: (_REDACTED if _SENSITIVE_KEY_PATTERN.search(str(key)) else value)
                for key, value in record.args.items()
            }
        return True
