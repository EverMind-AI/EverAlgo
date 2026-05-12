# Changelog — EverAlgo Monorepo

This is the **umbrella overview** for the EverAlgo monorepo. Each of the
8 distributions ships its own independent `CHANGELOG.md` with full history.

The format follows the [huggingface/transformers](https://github.com/huggingface/transformers)
/ [huggingface/accelerate](https://github.com/huggingface/accelerate) convention:
a root file with a current-version overview table, pointing at per-distribution
changelogs. Each distribution follows its own SemVer cadence — there is no
umbrella version (see [`docs/design.md`](docs/design.md) §1.3 "Why no meta package").

## Versions in `main`

No distribution has been published to PyPI yet. The table below tracks the
in-development version declared in each `packages/everalgo-*/pyproject.toml`.

| Distribution | Version | Changelog |
|---|---|---|
| `everalgo-core` | 0.1.0 (unreleased) | [packages/everalgo-core/CHANGELOG.md](packages/everalgo-core/CHANGELOG.md) |
| `everalgo-boundary` | 0.1.0 (unreleased) | [packages/everalgo-boundary/CHANGELOG.md](packages/everalgo-boundary/CHANGELOG.md) |
| `everalgo-clustering` | 0.1.0 (unreleased) | [packages/everalgo-clustering/CHANGELOG.md](packages/everalgo-clustering/CHANGELOG.md) |
| `everalgo-rank` | 0.1.0 (unreleased) | [packages/everalgo-rank/CHANGELOG.md](packages/everalgo-rank/CHANGELOG.md) |
| `everalgo-parser` | 0.1.0 (unreleased) | [packages/everalgo-parser/CHANGELOG.md](packages/everalgo-parser/CHANGELOG.md) |
| `everalgo-user-memory` | 0.1.0 (unreleased) | [packages/everalgo-user-memory/CHANGELOG.md](packages/everalgo-user-memory/CHANGELOG.md) |
| `everalgo-agent-memory` | 0.1.0 (unreleased) | [packages/everalgo-agent-memory/CHANGELOG.md](packages/everalgo-agent-memory/CHANGELOG.md) |
| `everalgo-knowledge` | 0.1.0 (unreleased) | [packages/everalgo-knowledge/CHANGELOG.md](packages/everalgo-knowledge/CHANGELOG.md) |

## How releases work

See [`README.md` § Cutting a release](README.md#cutting-a-release) for the
full workflow. In short:

1. Bump `version` in `packages/everalgo-<dist>/pyproject.toml`.
2. Run `git cliff --tag everalgo-<dist>/v<X.Y.Z> --include-path 'packages/everalgo-<dist>/**' --prepend packages/everalgo-<dist>/CHANGELOG.md`.
3. Review and manually polish the generated section.
4. Update the version + release-date row above for that distribution.
5. Commit (`📝 docs(<dist>): release notes for v<X.Y.Z>`), push the per-distribution tag, CI publishes to PyPI.

## Why two-tier (root + per-dist)?

- **Per-distribution history is the source of truth.** Each `packages/everalgo-*/CHANGELOG.md`
  ships inside its own wheel/sdist — PyPI users browsing the project page see only
  that distribution's changes, not the entire monorepo's churn.
- **Root file is the navigation index.** Engineers landing on the GitLab project
  homepage need one place that answers "what version is each distribution at
  right now?" without clicking into 8 subdirectories.
- **Industrial precedent.** HuggingFace transformers + accelerate, scipy
  (`doc/release/`), Apache Airflow (`RELEASE_NOTES.rst` + per-provider files)
  all use the two-tier pattern.
