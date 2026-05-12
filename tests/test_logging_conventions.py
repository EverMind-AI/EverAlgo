"""Cross-distribution logging convention checks (ADR-013).

Verifies the contract that every published subpackage attaches a
``NullHandler`` to its own ``everalgo.<subpkg>`` logger (library logging
contract per Python HOWTO + requests / urllib3 / google-cloud-python
pattern), and that ``everalgo.llm`` additionally attaches a default-on
``SensitiveHeadersFilter`` (OpenAI / Anthropic SDK pattern).

The test lives at the repo-root ``tests/`` rather than under any single
distribution because the contract spans every distribution — a per-package
test would have to be duplicated 11 times.
"""

from __future__ import annotations

import importlib
import logging

import pytest

from everalgo.llm._filters import SensitiveHeadersFilter

# All 11 subpackages that publish a public ``everalgo.<name>`` logger.
# Order = (logger_name, importable_module_path). They are identical, but
# kept separate so future internal subpackages can be added without
# automatically being treated as public loggers.
_SUBPACKAGES: list[tuple[str, str]] = [
    ("everalgo.types", "everalgo.types"),
    ("everalgo.llm", "everalgo.llm"),
    ("everalgo.prompts", "everalgo.prompts"),
    ("everalgo.testing", "everalgo.testing"),
    ("everalgo.boundary", "everalgo.boundary"),
    ("everalgo.clustering", "everalgo.clustering"),
    ("everalgo.rank", "everalgo.rank"),
    ("everalgo.parser", "everalgo.parser"),
    ("everalgo.user_memory", "everalgo.user_memory"),
    ("everalgo.agent_memory", "everalgo.agent_memory"),
    ("everalgo.knowledge", "everalgo.knowledge"),
]


@pytest.mark.parametrize(("logger_name", "module_path"), _SUBPACKAGES)
def test_subpackage_attaches_null_handler(logger_name: str, module_path: str) -> None:
    """Each subpackage's ``__init__`` attaches one ``NullHandler`` to its own logger.

    The library-logging contract (Python logging HOWTO, requests / urllib3
    pattern) requires every public library logger to carry a ``NullHandler``
    so a consumer that has not configured logging never sees "No handlers
    could be found" warnings under older Python compatibility paths and
    sees a clean default under modern Python.
    """
    importlib.import_module(module_path)
    handlers = logging.getLogger(logger_name).handlers
    null_handlers = [h for h in handlers if isinstance(h, logging.NullHandler)]
    assert len(null_handlers) >= 1, (
        f"{logger_name} must carry at least one NullHandler (ADR-013); found {[type(h).__name__ for h in handlers]}"
    )


def test_llm_logger_attaches_sensitive_headers_filter() -> None:
    """``everalgo.llm`` carries a default-on ``SensitiveHeadersFilter`` (ADR-013 §6)."""
    importlib.import_module("everalgo.llm")
    filters = logging.getLogger("everalgo.llm").filters
    matched = [f for f in filters if isinstance(f, SensitiveHeadersFilter)]
    assert len(matched) >= 1, (
        f"everalgo.llm must carry a SensitiveHeadersFilter (ADR-013 §6); found {[type(f).__name__ for f in filters]}"
    )


def test_filter_redacts_authorization_in_dict_args() -> None:
    """``SensitiveHeadersFilter`` replaces values whose keys match the sensitive-key pattern."""
    flt = SensitiveHeadersFilter()
    record = logging.LogRecord(
        name="everalgo.llm.test",
        level=logging.DEBUG,
        pathname=__file__,
        lineno=0,
        msg="headers=%s",
        args={"Authorization": "Bearer sk-secret", "Content-Type": "application/json"},
        exc_info=None,
    )
    assert flt.filter(record) is True
    assert isinstance(record.args, dict)
    assert record.args["Authorization"] == "<redacted>"
    assert record.args["Content-Type"] == "application/json"


def test_filter_redacts_api_key_variants() -> None:
    """Covers ``api_key`` / ``api-key`` / ``X-Api-Key`` / ``Bearer`` keys."""
    flt = SensitiveHeadersFilter()
    record = logging.LogRecord(
        name="everalgo.llm.test",
        level=logging.DEBUG,
        pathname=__file__,
        lineno=0,
        msg="headers=%s",
        args={
            "api_key": "k1",
            "api-key": "k2",
            "X-Api-Key": "k3",
            "Bearer-Token": "k4",
            "user": "alice",
        },
        exc_info=None,
    )
    flt.filter(record)
    assert isinstance(record.args, dict)
    assert record.args["api_key"] == "<redacted>"
    assert record.args["api-key"] == "<redacted>"
    assert record.args["X-Api-Key"] == "<redacted>"
    assert record.args["Bearer-Token"] == "<redacted>"
    assert record.args["user"] == "alice"


def test_filter_passes_through_tuple_args_unchanged() -> None:
    """Positional ``%``-format args (the standard ``logger.debug("foo=%s", foo)`` shape) are untouched."""
    flt = SensitiveHeadersFilter()
    record = logging.LogRecord(
        name="everalgo.llm.test",
        level=logging.DEBUG,
        pathname=__file__,
        lineno=0,
        msg="provider=%s tokens=%d",
        args=("openai", 42),
        exc_info=None,
    )
    flt.filter(record)
    assert record.args == ("openai", 42)


def test_filter_passes_through_none_args() -> None:
    """``record.args = None`` (no-arg log calls) must not raise."""
    flt = SensitiveHeadersFilter()
    record = logging.LogRecord(
        name="everalgo.llm.test",
        level=logging.DEBUG,
        pathname=__file__,
        lineno=0,
        msg="no args",
        args=None,
        exc_info=None,
    )
    assert flt.filter(record) is True
    assert record.args is None


def test_namespace_root_has_no_handler() -> None:
    """The ``everalgo`` namespace root must not carry handlers (PEP 420 no-init contract).

    Subpackage loggers propagate to the namespace root; the root itself
    remains unconfigured so that consumer applications retain full control
    over where ``everalgo.*`` records ultimately land. See ADR-013 §2.
    """
    handlers = logging.getLogger("everalgo").handlers
    assert handlers == [], (
        "everalgo (namespace root) must carry no handlers — "
        f"PEP 420 namespace cannot host an __init__.py; found {handlers}"
    )
