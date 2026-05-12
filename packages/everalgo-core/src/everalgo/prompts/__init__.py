"""Prompt utilities shared across the library.

Per ``docs/design.md`` §1.4: concrete prompt strings live next to their algorithm in each subpackage at
``<subpkg>/prompts/{en,zh}/<name>.py`` as module-level constants — not in this directory. This package only
ships the cross-cutting helpers (rendering, validation) every operator reuses.
"""

import logging

from everalgo.prompts.render import render_prompt

__all__ = ["render_prompt"]

# Library logging setup (ADR-013): NullHandler on each subpackage logger.
logging.getLogger(__name__).addHandler(logging.NullHandler())
