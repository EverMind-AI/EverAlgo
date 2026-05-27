# Releasing

The verified procedure for cutting a release of an EverAlgo distribution, as exercised for the
`0.2.0` release. This is the authoritative reference; the `README.md` § "Cutting a release"
summary points here.

## Model

- **Per-distribution, independent versions.** Each package owns its `version` and its own tag
  `everalgo-<dist>/v<semver>` (precedent: `google-cloud-python`, Apache Airflow providers).
- **Tag-triggered publish.** Pushing a release tag triggers the `publish` stage in
  `.gitlab-ci.yml`, which builds that one distribution and uploads it to PyPI via **Trusted
  Publishing over GitLab OIDC** — no long-lived token.
- `everalgo-knowledge` is **not published** (placeholder carrying `Private :: Do Not Upload`).

## One-time prerequisites (on PyPI, by a project owner)

For each published distribution, add a GitLab Trusted Publisher:
<https://docs.pypi.org/trusted-publishers/adding-a-publisher/> — namespace `npc-work/aic/ai`,
project `everalgo`, top-level pipeline file `.gitlab-ci.yml`.

## Procedure

### 1. Pre-flight — on `main`, everything green

```bash
git checkout main && git pull
uv run pytest                       # CI also runs the 3.12 / 3.13 / 3.14 matrix
uv run ruff check . && uv run ruff format --check .
uv run mypy . && uv run pyright
```

### 2. Bump versions

Compute each package's next version from its Conventional Commits and write it in — git-cliff +
uv, no extra tooling:

```bash
VER=$(git cliff --include-path "packages/everalgo-<dist>/**" --bumped-version)
(cd packages/everalgo-<dist> && uv version "${VER#*/v}")
```

For a coordinated baseline (e.g. all packages to `0.2.0`), set them directly with `uv version`.
Raise internal `everalgo-*` dependency lower bounds to the new version, then sync the lockfile
(CI runs `--frozen`, so a stale lock fails the build):

```bash
uv sync --all-packages
```

### 3. Archive CHANGELOGs

Move each published package's `## [Unreleased]` to `## [<ver>] - <date>` and open a fresh empty
`## [Unreleased]`; update the root `CHANGELOG.md` overview table. Keep `everalgo-knowledge`
unarchived (not published).

git-cliff can draft a section (`git cliff --include-path 'packages/everalgo-<dist>/**' --tag
everalgo-<dist>/v<ver> --unreleased --prepend <CHANGELOG>`), but the output is a **draft** — edit
for clarity. Note git-cliff renders commit-title-level entries, not the per-API detail in the
hand-written history; the changelog step always involves some human polish.

### 4. Land on `main`

Open an MR for the bump + changelog. **If it spans many scoped commits, merge with a merge
commit, not squash** — squashing collapses them into one and breaks git-cliff's per-commit
changelog grouping.

### 5. Tag and publish — ONE TAG AT A TIME

> ⚠️ **Push release tags one at a time.** Pushing several tags in a single `git push` makes
> GitLab drop the pipeline for some of them (observed during 0.2.0: 6 tags pushed at once, 2
> pipelines never created). Push each tag in its own `git push`, or verify after each push that
> its pipeline was actually created.

Publish a **canary** (one package) first to confirm the OIDC path end to end, then the rest:

```bash
git checkout main && git pull

# canary
git tag everalgo-core/v<ver>
git push origin everalgo-core/v<ver>            # watch the pipeline's publish job go green

# canary OK -> push the remaining tags, one per push
git tag everalgo-boundary/v<ver> && git push origin everalgo-boundary/v<ver>
git tag everalgo-clustering/v<ver> && git push origin everalgo-clustering/v<ver>
# ... one per distribution
```

A failed OIDC upload does **not** burn the version number — PyPI registers a version only on a
successful upload — so a failed publish is safe to fix and retry under the same tag.

### 6. Verify on PyPI

```bash
for d in core boundary clustering rank parser user-memory agent-memory; do
  curl -s "https://pypi.org/pypi/everalgo-$d/json" \
    | python3 -c "import sys,json;print('everalgo-$d', '<ver>' in json.load(sys.stdin)['releases'])"
done
```

## Auth mechanism

The `publish` job uses GitLab `id_tokens` (`aud: pypi`) + `twine>=6.1.0`. twine reads the OIDC
token from `PYPI_ID_TOKEN` and exchanges it with PyPI's configured Trusted Publisher itself — no
`curl` mint-token step, no stored credentials. The `twine>=6.1.0` pin is load-bearing: Trusted
Publishing support landed in twine 6.1.0; an older twine silently skips OIDC and 401s.
Reference: <https://docs.pypi.org/trusted-publishers/using-a-publisher/>
