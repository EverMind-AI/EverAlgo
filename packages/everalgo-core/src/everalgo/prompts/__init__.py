"""Prompt validators (placeholder + length checks) shared across the library.

Per ``docs/design.md`` §1.4: concrete prompt strings live next to their
algorithm in each subpackage at ``<subpkg>/prompts/{en,zh}/<name>.py`` as
module-level constants — not in this directory.
"""

__all__: list[str] = []
