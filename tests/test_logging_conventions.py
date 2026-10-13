"""Cross-distribution logging convention checks (ADR-013).

Verifies the contract that every published subpackage attaches a
``NullHandler`` to its own ``everalgo.<subpkg>`` logger (library logging
contract per Python HOWTO + requests / urllib3 / google-cloud-python
pattern).

The test lives at the repo-root ``tests/`` rather than under any single
distribution because the contract spans every distribution — a per-package
test would have to be duplicated 11 times.
"""

from __future__ import annotations

import importlib
import logging

import pytest

# All 11 subpackages that publish a public ``everalgo.<name>`` logger.
# Order = (logger_name, importable_module_path). They are identical, but
# kept separate so future internal subpackages can be added without
# automatically being treated as public loggers.
_SUBPACKAGES: list[tuple[str, str]] = [
    ("everalgo.types", "everalgo.types"),
    ("everalgo.llm", "everalgo.llm"),
    ("everalgo.utils", "everalgo.utils"),
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
