# Changelog — EverCore Monorepo

This is the **umbrella overview** for the EverCore monorepo. Each of the
8 distributions ships its own independent `CHANGELOG.md` with full history.

The format follows the [huggingface/transformers](https://github.com/huggingface/transformers)
/ [huggingface/accelerate](https://github.com/huggingface/accelerate) convention:
a root file with a current-version overview table, pointing at per-distribution
changelogs. Each distribution follows its own SemVer cadence — there is no
umbrella version (see [`docs/design.md`](docs/design.md) §1.3 "Why no meta package").

## Current published versions

| Distribution | Version | Released | Changelog |
|---|---|---|---|
| `evercore-core` | 0.1.0 | 2026-05-11 | [packages/evercore-core/CHANGELOG.md](packages/evercore-core/CHANGELOG.md) |
| `evercore-boundary` | 0.1.0 | 2026-05-11 | [packages/evercore-boundary/CHANGELOG.md](packages/evercore-boundary/CHANGELOG.md) |
| `evercore-clustering` | 0.1.0 | 2026-05-11 | [packages/evercore-clustering/CHANGELOG.md](packages/evercore-clustering/CHANGELOG.md) |
| `evercore-rank` | 0.1.0 | 2026-05-11 | [packages/evercore-rank/CHANGELOG.md](packages/evercore-rank/CHANGELOG.md) |
| `evercore-parser` | 0.1.0 | 2026-05-11 | [packages/evercore-parser/CHANGELOG.md](packages/evercore-parser/CHANGELOG.md) |
| `evercore-user-memory` | 0.1.0 | 2026-05-11 | [packages/evercore-user-memory/CHANGELOG.md](packages/evercore-user-memory/CHANGELOG.md) |
| `evercore-agent-memory` | 0.1.0 | 2026-05-11 | [packages/evercore-agent-memory/CHANGELOG.md](packages/evercore-agent-memory/CHANGELOG.md) |
| `evercore-knowledge` | 0.1.0 | 2026-05-11 | [packages/evercore-knowledge/CHANGELOG.md](packages/evercore-knowledge/CHANGELOG.md) |

## How releases work

See [`README.md` § Cutting a release](README.md#cutting-a-release) for the
full workflow. In short:

1. Bump `version` in `packages/evercore-<dist>/pyproject.toml`.
2. Run `git cliff --tag evercore-<dist>/v<X.Y.Z> --include-path 'packages/evercore-<dist>/**' --prepend packages/evercore-<dist>/CHANGELOG.md`.
3. Review and manually polish the generated section.
4. Update the version + release-date row above for that distribution.
5. Commit (`📝 docs(<dist>): release notes for v<X.Y.Z>`), push the per-distribution tag, CI publishes to PyPI.

## Why two-tier (root + per-dist)?

- **Per-distribution history is the source of truth.** Each `packages/evercore-*/CHANGELOG.md`
  ships inside its own wheel/sdist — PyPI users browsing the project page see only
  that distribution's changes, not the entire monorepo's churn.
- **Root file is the navigation index.** Engineers landing on the GitLab project
  homepage need one place that answers "what version is each distribution at
  right now?" without clicking into 8 subdirectories.
- **Industrial precedent.** HuggingFace transformers + accelerate, scipy
  (`doc/release/`), Apache Airflow (`RELEASE_NOTES.rst` + per-provider files)
  all use the two-tier pattern.
