# Releasing

The normative procedure for cutting a release of an EverAlgo distribution. This document is
written for durability: the **invariants** section captures what does not change as tooling
evolves; the **default path** captures the common single-distribution flow with concrete
commands; the **coordinated baseline** section captures the rare multi-package case; and
**rationale** explains the "why" behind every non-obvious rule, so future maintainers can
reason their way through tool migrations without rewriting policy.

The README's "Cutting a release" section is a pointer here. Tag-trigger logic lives in
`.gitlab-ci.yml`'s `publish` job; tooling versions are pinned there, not here.

## Invariants

These hold regardless of which tools we use:

1. **Per-distribution independent versions.** Every published package owns its own `version`
   in `packages/everalgo-<dist>/pyproject.toml` and its own SemVer cadence. There is no
   umbrella version. Bumping one distribution does not require bumping any other. Precedent:
   `google-cloud-python`, Apache Airflow providers.
2. **Tag is the release trigger.** Pushing a tag matching `everalgo-<dist>/v<X.Y.Z>` to the
   GitLab remote triggers exactly the `publish` job for that one distribution. No tag, no
   release.
3. **CHANGELOG is source of truth.** Each `packages/everalgo-<dist>/CHANGELOG.md` ships
   inside the wheel and is the authoritative per-package history. The root `CHANGELOG.md`
   is a repository-level timeline and index, not a substitute.
4. **PyPI auth via Trusted Publishing.** No long-lived tokens are stored. The CI `publish`
   job mints a short-lived OIDC token (`aud: pypi`); the uploader exchanges it with PyPI's
   configured Trusted Publisher. Reference: <https://docs.pypi.org/trusted-publishers/>.
5. **`main` is protected.** Direct push is denied. All changes — including release bumps —
   land via merge request.
6. **SemVer with the 0.x exception.** Per <https://semver.org/#spec-item-4>, in the `0.y.z`
   range "anything may change at any time"; we use that exception to remove or rename
   public API at minor bumps while still on 0.x. From `1.0.0` onward, strict SemVer
   applies — breaking changes require a major bump.

## One-time prerequisites

Done once per distribution by a PyPI project owner: add a GitLab Trusted Publisher
(<https://docs.pypi.org/trusted-publishers/adding-a-publisher/>), pointing at this project
and the top-level `.gitlab-ci.yml`. For a distribution that has never been published, use
PyPI's **pending publisher** flow (<https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/>) — it creates the PyPI project and configures the publisher in one step;
the first tag-triggered upload activates it automatically.

## Keeping `[Unreleased]` up to date

Every MR that adds, changes, or removes user-visible behaviour **must** include a one-line entry in the affected package's `## [Unreleased]` section (`packages/everalgo-<dist>/CHANGELOG.md`). This is part of the MR, not a separate task.

**What counts as user-visible:** new public API, changed behaviour, removed API, bug fixes, performance changes, dependency changes. Internal refactors, test-only changes, and CI tweaks do not need an entry.

**Format:** use the Keep a Changelog subsection that fits (`### Added`, `### Changed`, `### Fixed`, `### Removed`, `### Performance`). One line per change, written for the user who will read the release notes — describe impact, not commit mechanics.

**Why this matters:** at release time, the `[Unreleased]` section is promoted to the new version section as-is. If it is empty, the release author must reconstruct the changelog from git history under time pressure, which produces worse release notes and slows down the release. The MR author is the person with the most context on what changed and why — they should write the entry.

**AI assistants (Claude Code / Cursor / Copilot):** when you create an MR that touches files under `packages/everalgo-<dist>/`, check whether the change is user-visible. If so, append the appropriate entry to `packages/everalgo-<dist>/CHANGELOG.md` under `## [Unreleased]` before committing. Do not wait for the human to ask.

---

## Default path — single-distribution bump

Use this whenever a release touches only one distribution. The vast majority of releases
follow this path.

### 1. Pre-flight

On `main`, with the working tree clean:

```bash
git checkout main && git pull --ff-only
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy . && uv run pyright
```

CI re-runs the same checks across the supported Python matrix; failing locally is faster.

### 2. Decide the version

Pick `<X.Y.Z>` per SemVer (with the 0.x exception, Invariant 6). Two paths:

- **Manual** — always correct. Inspect commits since the previous tag and decide:
  ```bash
  git log everalgo-<dist>/v<prev>..main -- packages/everalgo-<dist>/
  ```
- **Suggestion** — informative, never authoritative for breaking changes:
  ```bash
  git cliff --include-path 'packages/everalgo-<dist>/**' --bumped-version
  ```
  This promotes patch / minor by Conventional-Commit type, but does **not** detect API
  removals or renames; for any breaking change override its suggestion manually.

### 3. Edit the three files

a. **`packages/everalgo-<dist>/pyproject.toml`** — set `version = "<X.Y.Z>"`.

b. **`packages/everalgo-<dist>/CHANGELOG.md`** — move the contents of `## [Unreleased]`
   into a new `## [<X.Y.Z>] - <YYYY-MM-DD>` section, leave a fresh empty `## [Unreleased]`,
   and update the link references at the bottom (add a new `[<X.Y.Z>]:` line; retarget
   `[Unreleased]:` to compare from the new tag).

   `git cliff --include-path 'packages/everalgo-<dist>/**' --tag everalgo-<dist>/v<X.Y.Z>
   --unreleased` can produce a draft, but the output is commit-title level (and often empty
   when commit bodies omit Conventional-Commit keywords). Always edit to per-API detail by
   hand: release notes describe user-visible impact and migration, not mechanical commit
   subjects.

c. **Root `CHANGELOG.md`** — two edits:
   - Update this distribution's row in the overview table to `<X.Y.Z>`.
   - Insert a new top-of-timeline section
     `## [everalgo-<dist>/<X.Y.Z>] - <YYYY-MM-DD>` containing one paragraph of context and
     a link to the package CHANGELOG. The root timeline is for repo-level chronology; the
     package CHANGELOG holds the authoritative detail.

d. **Dependency audit** — verify version floors and declared dependencies before tagging.

   **Completeness: every runtime import must be declared.** Grep the package source for all
   external `import` / `from ... import` statements that execute outside `TYPE_CHECKING`
   blocks. Each resolved top-level package must appear in `[project.dependencies]`. A
   missing entry means `pip install everalgo-<dist>` in a clean venv will fail at runtime
   even though the workspace venv (which installs everything) masks the problem.

   ```bash
   # Quick audit: list external runtime imports, compare against pyproject.toml
   grep -rh 'from \|^import ' packages/everalgo-<dist>/src/ --include='*.py' \
     | grep -v '# Deferred' | sort -u
   # Cross-check each top-level package against [project.dependencies].
   ```

   **Floor correctness: every `everalgo-*` floor must cover the APIs actually used.** For
   each workspace dependency (`everalgo-core`, `everalgo-parser`, ...) declared with a
   `>=X.Y.Z` floor, verify that the floor version actually exports every type, function,
   and field the package imports — including fields added to existing types. If the package
   uses `CategorySpec` (introduced in core 0.3.0) but the floor says `>=0.2.0`, users who
   install an older core will get `ImportError` or `ValidationError` at runtime.

   ```bash
   # List all imports from workspace dependencies
   grep -rh 'from everalgo\.' packages/everalgo-<dist>/src/ --include='*.py' | sort -u
   # For each imported symbol: which version of the dependency introduced it?
   # Raise the floor if the current floor predates that version.
   ```

   **Downstream propagation: raise floors in packages that depend on *this* package** if
   they must adopt a new feature or handle a changed API. Most single-package bumps do not
   need this — only bumps that introduce a feature/fix downstream packages must adopt.

   ```bash
   # Find all workspace consumers of the bumped package
   grep -rl 'everalgo-<dist>' packages/everalgo-*/pyproject.toml
   ```

e. **Documentation sweep** — grep for stale references to the package's status, version, or
   API surface across the entire repo. `README.md` ships inside the wheel, so stale text
   lands on PyPI. `AGENTS.md` feeds AI coding assistants, so stale claims propagate into
   generated code. The sweep must cover at least:

   ```bash
   # Find all docs that mention this distribution by name
   grep -rn 'everalgo-<dist>\|everalgo.<subpkg>' \
     AGENTS.md README.md docs/ packages/everalgo-<dist>/README.md \
     --include='*.md' | grep -iE 'stub|placeholder|not.published|not.implemented|reserved|TODO'
   ```

   Common stale patterns to look for:
   - "NOT YET IMPLEMENTED", "not published", "placeholder", "namespace reserved" — a
     previously-stubbed distribution now ships real code.
   - Version numbers in prose ("most at `0.2.0`") — often forgotten when the table is
     updated but the paragraph is not.
   - `__all__` or public-API lists in docs that do not reflect new exports.
   - `pyproject.toml` classifiers (`Development Status :: 1 - Planning`,
     `Private :: Do Not Upload`) that no longer apply.

   This step is especially critical for **first-time publications** (stub → real) where
   references to the old status are scattered across files written months earlier.

### 4. Sync the lockfile

```bash
uv sync --all-packages
```

CI runs `uv sync --frozen`; a stale lockfile fails the build.

### 5. Branch, commit, open the MR

```bash
git checkout -b release/<dist>-v<X.Y.Z>
git add CHANGELOG.md \
        packages/everalgo-<dist>/CHANGELOG.md \
        packages/everalgo-<dist>/pyproject.toml \
        uv.lock
git commit -m "🔖 release(<dist>): <X.Y.Z> — <one-line summary>"
git push -u origin release/<dist>-v<X.Y.Z>

glab mr create --target-branch main \
  --title "🔖 release(<dist>): <X.Y.Z> — <one-line summary>" \
  --squash-before-merge --remove-source-branch \
  --description '<motivation; breaking-change call-out if any; migration snippet if any; pre-flight green>'
```

**Squash is correct for a single-commit release bump.** Use a merge commit only when the
MR carries multiple meaningfully-scoped commits whose granularity is worth preserving for
future git-cliff runs — that case belongs to the coordinated-baseline path below.

### 6. Land on `main` and tag

After the MR merges:

```bash
git checkout main && git pull --ff-only
git tag everalgo-<dist>/v<X.Y.Z>
git push origin everalgo-<dist>/v<X.Y.Z>
```

The tag push triggers the `publish` job in `.gitlab-ci.yml`. Lint and test jobs are
intentionally excluded from tag pipelines (see `.python-job` rules); `main` is already
green from Step 1.

### 7. Verify, in this order

Trust signals strictly proceed from job trace → PyPI; do not invert this order.

a. **Confirm the publish job succeeded** before touching PyPI:
   ```bash
   glab ci view --branch everalgo-<dist>/v<X.Y.Z>
   ```
   Look for `Uploading everalgo_<dist>-<X.Y.Z>.tar.gz` and `Job succeeded`.

b. **Then** query PyPI, with a cache-bust query parameter:
   ```bash
   curl -s "https://pypi.org/pypi/everalgo-<dist>/json?$(date +%s)" \
        -H "Cache-Control: no-cache" \
     | python3 -c "import sys, json; print(json.load(sys.stdin)['info']['version'])"
   ```

The PyPI JSON index lags the actual upload by up to a few minutes (see Rationale). Seeing
the old version on a first query immediately after the job finishes is normal; the upload
itself is already live for `pip install`.

A failed OIDC upload **does not consume** the version number — PyPI registers versions
only on successful upload — so a failed publish is safe to fix and retry under the same
tag without bumping again.

## Coordinated multi-package baseline

Use this only for cross-package API coordination, packaging-metadata sweeps, or moving N
distributions to the same baseline version in one MR. This is rare; the 0.2.0 release was
the canonical example.

Differences from the default path:

1. **Step 3** runs once per affected package, including the dependency audit (3.d) which
   must cover cross-package floor raises across *all* distributions in the coordinated set.
   The root `CHANGELOG.md` gets a single shared `## [<X.Y.Z>] - <date>` section
   summarising the coordinated baseline rather than one section per package.
2. **Step 5** typically carries many scoped commits (per-package `chore` / `fix` /
   `refactor`). Merge with a **merge commit, not squash** — squashing collapses N scoped
   commits into one and destroys git-cliff's per-commit grouping. In the GitLab UI,
   override the project default for this MR; do not pass `--squash-before-merge` to
   `glab`.
3. **Step 6** pushes N tags. **One tag per `git push` invocation.** Pushing multiple tag
   refs in a single push event makes GitLab silently drop pipeline creation for some of
   them. Push, wait for the pipeline to appear, then push the next.
4. **Step 6, canary first.** Push one tag (typically `everalgo-core`) and watch its publish
   job go end-to-end green before pushing the others. This catches Trusted-Publisher or
   OIDC misconfiguration on one package rather than N.

Steps 1, 2, 4, and 7 are identical to the default path.

## Rationale

Each rule above rests on a constraint, not on a current command. These are the constraints,
so future maintainers can reason about tool migrations without rewriting policy.

- **Why tags trigger publish, not branches.** Mutable refs are the wrong trigger for an
  immutable artifact: a PyPI release cannot be replaced under the same version. Tags are
  immutable by convention and align one-to-one with releases.
- **Why one tag per `git push`.** A GitLab behaviour: pushing multiple tag refs in a single
  push event can drop pipeline creation for trailing tags. Observed during the 0.2.0
  release (six tags pushed in one invocation; two pipelines never created). Defensive
  practice regardless of whether GitLab eventually fixes it.
- **Why squash for one commit, merge for many.** git-cliff groups by individual commit;
  squashing N scoped commits collapses the signal. A single release-bump commit has no
  granularity to lose, so squash is preferable (cleaner mainline history).
- **Why Trusted Publishing instead of a stored token.** OIDC tokens are short-lived and
  audience-bound, shrinking blast radius if the CI environment is ever compromised. The
  twine version floor exists because earlier twine silently skipped the OIDC handshake and
  returned 401; the current floor lives in `.gitlab-ci.yml`.
- **Why a failed upload does not burn the version.** PyPI registers a version only on
  successful upload. Retrying under the same tag is safe; no TestPyPI dry run is required.
- **Why the PyPI JSON index lags.** `pypi.org/pypi/<pkg>/json` is served via Fastly CDN
  with an independent freshness window. A successful upload is immediately installable via
  `pip install`, but the JSON index can take a minute or two to reflect it. Trust the job
  trace's `Job succeeded` over the JSON.
- **Why changelog polish is non-negotiable.** Conventional-Commit subjects describe what
  changed mechanically; release notes describe user-visible impact, migration steps, and
  breaking-change call-outs. No generator produces the latter from the former.
- **Why the 0.x exception is used explicitly.** SemVer's 0.x clause permits breaking
  changes at any time; we use it deliberately to remove or rename public API at 0.x minor
  bumps. From 1.0.0 onward, breaking changes require a major bump and a deprecation cycle.
- **Why the root CHANGELOG carries a per-release section in single-package bumps.** The
  table-only alternative leaves the root file without a chronology; readers landing at the
  repo root would have to walk N package CHANGELOGs to reconstruct what shipped when. A
  one-paragraph root section per release preserves the timeline without duplicating
  detail.
- **Why a documentation sweep is mandatory before tagging.** `README.md` is baked into the
  sdist and wheel by hatchling; stale text ("not published", "NOT YET IMPLEMENTED") lands
  on the PyPI project page and cannot be corrected without a new release. `AGENTS.md`
  feeds AI coding assistants (Claude Code, Cursor, Copilot), so a stale claim like
  "knowledge is a stub" propagates into every AI-generated answer. Historical example:
  the `everalgo-knowledge` 0.1.0 release MR updated all 3 CHANGELOGs and the root table,
  but missed 4 prose references to "NOT YET IMPLEMENTED — not published" in `AGENTS.md`,
  `README.md` (×2), and `docs/api/README.md` — caught only at tag time.
- **Why a dependency audit is mandatory before release.** The uv workspace installs all
  8 distributions into a shared venv, so a missing `[project.dependencies]` entry or a
  too-low version floor is invisible during development and CI — every symbol resolves
  because sibling packages are co-installed. The bug surfaces only when an end user runs
  `pip install everalgo-<dist>` in isolation, which is the exact scenario a release
  creates. Historical example: `everalgo-knowledge` shipped code that `from asgiref.sync
  import async_to_sync` in four modules but never declared `asgiref` as a dependency; the
  workspace venv masked the gap because `everalgo-rank` (a sibling, not a dependency of
  knowledge) pulls in `asgiref`. A second example from the same release: knowledge
  imported `CategorySpec` (introduced in core 0.3.0) but declared `everalgo-core>=0.2.0`,
  meaning users who already had core 0.2.x installed would hit `ImportError` at runtime.

## Where tool versions live

Current minimum versions are pinned in `.gitlab-ci.yml`:

- `uv` runner image — see the `image:` line on the `publish` job.
- `twine` floor — see the `--with 'twine>=...'` flag in the `publish` job.

If those pins move, this document does not need to change.
