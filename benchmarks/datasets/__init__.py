"""Dataset auto-discovery.

Each subpackage of ``benchmarks.datasets`` that exposes a class attribute
matching the convention ``<Name>Dataset`` with ``name: str`` is registered
automatically. Future datasets only need to add a subdirectory with a
class — no changes to ``common/`` required.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["discover_datasets"]


def discover_datasets() -> dict[str, Callable[[Any], Any]]:
    """Walk ``benchmarks.datasets.*`` subpackages and return name -> factory.

    Returns:
        Mapping of ``dataset.name`` (str) to a factory callable taking
        ``data_path`` (typically ``pathlib.Path``) and returning a Dataset instance.
    """
    import benchmarks.datasets as pkg

    registry: dict[str, Callable[[Any], Any]] = {}
    for module_info in pkgutil.iter_modules(pkg.__path__):
        if not module_info.ispkg:
            continue
        mod = importlib.import_module(f"benchmarks.datasets.{module_info.name}")
        for attr_name in dir(mod):
            if not attr_name.endswith("Dataset"):
                continue
            cls = getattr(mod, attr_name)
            if isinstance(cls, type) and hasattr(cls, "name"):
                registry[cls.name] = cls  # type: ignore[index,attr-defined]
    return registry
