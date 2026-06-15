## Summary

<!-- What does this MR do and why? -->

## Checklist

- [ ] MR title follows `<emoji> <type>(<scope>): <description>` format
- [ ] `CHANGELOG.md` updated: if this MR adds/changes/removes user-visible behaviour, a one-line entry has been added under `## [Unreleased]` in the affected package's `packages/everalgo-<dist>/CHANGELOG.md`
- [ ] Tests added or updated
- [ ] `uv run ruff check . && uv run ruff format --check .` passes
- [ ] `uv run mypy . && uv run pyright` passes
- [ ] `uv run pytest` passes
