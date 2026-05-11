#!/usr/bin/env python3
r"""MR title lint for AGENTS.md section 6 - Gitmoji + Conventional Commits.

Used by ``.gitlab-ci.yml`` ``mr-title-lint`` job. Invoked as::

    python3 scripts/check_mr_title.py "$CI_MERGE_REQUEST_TITLE"

Exits 0 if the title matches ``<emoji> <type>(<scope>)?(!)?: <description>``,
1 otherwise.

Implementation note: this script replaces an earlier ``grep -E`` regex that
used ``\xNN`` byte escapes. POSIX ERE does not honour ``\xNN``; GNU grep
treats it as the literal four characters ``\``, ``x``, ``N``, ``N``.
Switching to Python regex with explicit Unicode codepoint classes
sidesteps the issue entirely.
"""

from __future__ import annotations

import re
import sys

ALLOWED_TYPES = "feat|fix|perf|refactor|revert|docs|style|test|build|ci|chore"

# Leading Gitmoji glyph class - one or more characters from:
#   U+2600..U+27BF   BMP misc-symbols & dingbats (covers Sparkles, Recycle, Check Mark)
#   U+1F300..U+1FAFF SMP emoji block (covers Bug, Wrench, Memo, Rocket)
#   U+FE0F           Variation Selector-16 (used by Recycle and similar compound glyphs)
#   U+200D           Zero-Width Joiner (used by family/profession emoji)
TITLE_RE = re.compile(
    r"^"
    r"[☀-➿\U0001F300-\U0001FAFF️‍]+"
    r"\s+"
    rf"(?:{ALLOWED_TYPES})"
    r"(?:\([^)]+\))?"
    r"!?"
    r":\s"
    r".+"
    r"$"
)


def main(title: str) -> int:
    """Validate the MR title and emit guidance on failure.

    Args:
        title: The merge request title (typically ``$CI_MERGE_REQUEST_TITLE``).

    Returns:
        Exit code: 0 if valid, 1 otherwise.
    """
    if TITLE_RE.match(title):
        print(f"MR title OK: {title}")
        return 0

    print("MR title must follow Gitmoji + Conventional Commits:")
    print("  <emoji> <type>(<scope>)?: <description>")
    print()
    print(f"Got: {title!r}")
    print()
    print("Examples:")
    print("  ✨ feat(clustering): add cluster_by_llm decision prompt")
    print("  \U0001f41b fix(boundary): correct token count for emoji-only chat")
    print("  \U0001f4dd docs(release): bump everalgo-rank to 0.2.0")
    print()
    print(f"Allowed types: {ALLOWED_TYPES.replace('|', ' | ')}")
    print("See AGENTS.md section 6 (Branching & Commits) for the convention.")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else ""))
