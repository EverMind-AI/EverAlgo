# Version Policy

## Semantic Versioning

Every EverAlgo distribution follows [Semantic Versioning 2.0.0](https://semver.org/):

- `PATCH` — backward-compatible bug fixes.
- `MINOR` — backward-compatible new features.
- `MAJOR` — incompatible API changes.

During the **`0.x` series** (pre-stable), any release may include breaking changes.
`1.0.0` is the stability commitment point: once `everalgo-core` reaches `1.0.0`, its public API will not break within a major version.

---

## Independent version cadences

Each distribution has its own version number and its own release timeline.
Upgrading `everalgo-rank` does not require touching `everalgo-user-memory`.

Dependencies between distributions use **loose constraints** (a lower bound plus a next-major upper bound), the same independent-versioning approach as other namespace monorepos such as [`google-cloud-python`](https://github.com/googleapis/google-cloud-python) (many distributions sharing `google.cloud.*`, each on its own version) and Apache Airflow providers:

```toml
# In everalgo-user-memory/pyproject.toml
everalgo-core>=0.2.0,<2.0.0
everalgo-boundary>=0.2.0,<2.0.0
```

The upper bound spans the entire next major version so diamond-dependency resolution works without manual coordination across distributions.

---

## Supported Python versions

EverAlgo requires **Python 3.12 or later**.
Support for a Python version is dropped when it reaches [end-of-life](https://devguide.python.org/versions/) status.
Such a drop is a `MINOR` bump during `0.x` and a `MAJOR` bump post `1.0.0`.

---

## Deprecation policy

Before removing a public API:

1. Mark it `deprecated` in the docstring and emit a `DeprecationWarning` via `warnings.warn(..., stacklevel=2)`.
2. Keep the deprecated symbol for **at least one `MINOR` release**.
3. Remove it in the next `MAJOR` bump (or in `0.x`, the next `MINOR` bump).

---

## Changelog

Per-distribution changelogs live at `packages/everalgo-<name>/CHANGELOG.md` and are generated from Conventional Commit messages using [git-cliff](https://git-cliff.org/).
A summary table at the repo root [`CHANGELOG.md`](../CHANGELOG.md) covers the current release cycle across all distributions.

---

## Release cadence

There is no fixed release schedule.
Releases are driven by feature readiness and bug severity.
All releases land on `main` via Merge Request and are tagged with the distribution name and version, for example `everalgo-clustering/v0.2.0`.
