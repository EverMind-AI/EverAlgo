# Rename `evercore` → `everalgo` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Workspace-wide rename of the `evercore` brand to `everalgo` across all 8 distributions, namespace packages, configuration files, docs, ADRs, CHANGELOGs, and CI scripts. Implements the design at `docs/superpowers/specs/2026-05-11-rename-evercore-to-everalgo-design.md`.

**Architecture:** Mechanical three-case string substitution (`evercore` → `everalgo`, `EverCore` → `EverAlgo`, `EVERCORE` → `EVERALGO`) plus 16 directory renames via `git mv` (8 `packages/evercore-*` + 8 `src/evercore` namespace roots). All changes squash into one MR titled `♻️ refactor(repo): rename evercore → everalgo across the workspace`.

**Tech Stack:** `git mv` for renames, `perl -i -pe` for text substitution, `rg --files-with-matches` for discovery, `uv` for workspace verification, `ruff` / `mypy` / `pytest` / `pre-commit` as quality gates.

---

## File Structure

This plan is a rename — it does NOT create new source files. The affected paths fall into 5 buckets:

**Bucket 1 — Directory renames (16 total, all via `git mv`):**
- `packages/evercore-{core,boundary,clustering,rank,parser,user-memory,agent-memory,knowledge}` → `packages/everalgo-...` (8)
- `packages/everalgo-*/src/evercore` → `packages/everalgo-*/src/everalgo` (8, the namespace root inside each dist)

**Bucket 2 — `pyproject.toml` files (9 total):**
- Root `pyproject.toml` — workspace `members`, tool table dependencies, `mypy_path`, `known-first-party`
- 8 dist `pyproject.toml` — `name`, `dependencies`, setuptools `packages`

**Bucket 3 — Source code (Python files across all 8 dists + workspace `tests/`):**
- Approx 80+ `.py` files; all references are `from evercore.xxx import ...` or `import evercore.xxx`

**Bucket 4 — Configuration:**
- `cliff.toml` — `tag_pattern`, `commit_parsers`, header comment
- `.gitlab-ci.yml` — sanity check, currently no `evercore` strings expected
- `.pre-commit-config.yaml` — sanity check
- `.gitignore` — sanity check

**Bucket 5 — Documentation:**
- `README.md` / `AGENTS.md` / `CLAUDE.md` (symlink) / `.cursorrules` (symlink)
- `CHANGELOG.md` (root) + 8 × `packages/*/CHANGELOG.md`
- `docs/design.md`, `docs/decisions/*.md`, `docs/reference/README.md`
- `docs/superpowers/specs/*.md` — 5 already-shipped specs, including filename rename (`2026-05-07-evercore-*` → `2026-05-07-everalgo-*`)
- `local/superpowers/plans/*.md` — 2 retained plans
- `scripts/check_mr_title.py` — verify no `evercore` string in error output

**Hard exclusion (never touched by any substitution):**
- `docs/superpowers/specs/2026-05-11-rename-evercore-to-everalgo-design.md` (this spec)
- `docs/superpowers/plans/2026-05-11-rename-evercore-to-everalgo.md` (this plan)
- `.git/`, `.venv/`, `**/__pycache__/`, `**/*.egg-info/`, `uv.lock`

---

## Task 1: Directory & package metadata rename

**Files:**
- Rename: `packages/evercore-*` × 8 → `packages/everalgo-*`
- Rename: `packages/everalgo-*/src/evercore` × 8 → `.../src/everalgo`
- Modify: `pyproject.toml` (root)
- Modify: `packages/everalgo-*/pyproject.toml` × 8

This task brings the workspace into a state where `uv sync --all-packages` resolves successfully. Source code imports remain `from evercore.xxx`, so the editable installs work but runtime import paths are intentionally broken — fixed in Task 2.

- [ ] **Step 1: Rename the 8 outer dist directories.**

```bash
cd /Users/admin/Documents/evermemos/evercore
git mv packages/evercore-core         packages/everalgo-core
git mv packages/evercore-boundary     packages/everalgo-boundary
git mv packages/evercore-clustering   packages/everalgo-clustering
git mv packages/evercore-rank         packages/everalgo-rank
git mv packages/evercore-parser       packages/everalgo-parser
git mv packages/evercore-user-memory  packages/everalgo-user-memory
git mv packages/evercore-agent-memory packages/everalgo-agent-memory
git mv packages/evercore-knowledge    packages/everalgo-knowledge
ls packages/
```

Expected: `ls packages/` lists 8 `everalgo-*` entries and zero `evercore-*` entries.

- [ ] **Step 2: Rename the 8 inner namespace roots.**

```bash
for d in packages/everalgo-*; do
  git mv "$d/src/evercore" "$d/src/everalgo"
done
find packages -type d -name evercore
```

Expected: `find` returns nothing (no `evercore` directories remain).

- [ ] **Step 3: Update root `pyproject.toml`.**

Replace these blocks (use Edit tool, exact line numbers verified before edit):

```toml
# line 2
name = "evercore-workspace"            →   name = "everalgo-workspace"

# lines 14-21 dependency table
evercore-core         = { workspace = true }   →   everalgo-core         = { workspace = true }
evercore-boundary     = { workspace = true }   →   everalgo-boundary     = { workspace = true }
evercore-clustering   = { workspace = true }   →   everalgo-clustering   = { workspace = true }
evercore-rank         = { workspace = true }   →   everalgo-rank         = { workspace = true }
evercore-parser       = { workspace = true }   →   everalgo-parser       = { workspace = true }
evercore-user-memory  = { workspace = true }   →   everalgo-user-memory  = { workspace = true }
evercore-agent-memory = { workspace = true }   →   everalgo-agent-memory = { workspace = true }
evercore-knowledge    = { workspace = true }   →   everalgo-knowledge    = { workspace = true }

# lines 25-32 [project] dependencies list
"evercore-core",          →   "everalgo-core",
"evercore-boundary",      →   "everalgo-boundary",
"evercore-clustering",    →   "everalgo-clustering",
"evercore-rank",          →   "everalgo-rank",
"evercore-parser",        →   "everalgo-parser",
"evercore-user-memory",   →   "everalgo-user-memory",
"evercore-agent-memory",  →   "everalgo-agent-memory",
"evercore-knowledge",     →   "everalgo-knowledge",

# line 66
known-first-party = ["evercore", "tests"]      →   known-first-party = ["everalgo", "tests"]

# line 98
mypy_path = "packages/evercore-core/src:packages/evercore-boundary/src:..."   →
mypy_path = "packages/everalgo-core/src:packages/everalgo-boundary/src:packages/everalgo-clustering/src:packages/everalgo-rank/src:packages/everalgo-parser/src:packages/everalgo-user-memory/src:packages/everalgo-agent-memory/src:packages/everalgo-knowledge/src"

# comments around mypy table (`evercore.*` PEP 420 namespace ... `evercore.types`)
all `evercore.*` → `everalgo.*`
```

- [ ] **Step 4: Update each dist `pyproject.toml`.**

For each of `packages/everalgo-{core,boundary,clustering,rank,parser,user-memory,agent-memory,knowledge}/pyproject.toml`:

```toml
name = "evercore-<dist>"           →   name = "everalgo-<dist>"
dependencies = [
  "evercore-core>=0.1.0,<2.0.0",   →   "everalgo-core>=0.1.0,<2.0.0",
  ...
]
[tool.setuptools]
packages = ["src/evercore"]        →   packages = ["src/everalgo"]
```

Use `perl -i -pe 's/evercore/everalgo/g' packages/everalgo-*/pyproject.toml` — these files only contain dist-name references, no `EverCore` brand strings.

- [ ] **Step 5: Verify workspace metadata resolves.**

```bash
rm -rf .venv
uv sync --all-packages --group dev
```

Expected: `uv sync` exits 0, `.venv/` rebuilt, `uv pip list | grep everalgo` shows all 8 distributions installed editable.

- [ ] **Step 6: Commit Task 1.**

```bash
git add -A
git commit -m "$(cat <<'EOF'
♻️ refactor(repo): rename dist directories & pyproject metadata to everalgo

Step 1/N of evercore → everalgo workspace rename. Renames the 8 outer dist
directories (packages/evercore-* → packages/everalgo-*) and 8 inner PEP 420
namespace roots (src/evercore → src/everalgo). Updates root pyproject.toml
workspace members, dependency table, mypy_path, and ruff first-party list.
Updates each dist's pyproject.toml name + dependencies + setuptools packages.

Source imports still reference `from evercore.xxx` — fixed in next step.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Source & test content sweep

**Files:**
- All `.py` files under `packages/everalgo-*/src/`
- All `.py` files under `packages/everalgo-*/tests/` and root `tests/`
- `scripts/check_mr_title.py`

After this task, `uv run pytest` must pass; source code references are coherent.

- [ ] **Step 1: Discover all `.py` files that mention `evercore` (excluding meta-docs).**

```bash
rg --files-with-matches -g '*.py' -g '!docs/superpowers/specs/2026-05-11-rename-*' -g '!docs/superpowers/plans/2026-05-11-rename-*' 'evercore|EverCore|EVERCORE' | tee /tmp/everalgo-py-files.txt | wc -l
```

Expected: a count between 50 and 120 (initial grep gave ~80 `.py` files).

- [ ] **Step 2: Run the three-case substitution on every discovered `.py` file.**

```bash
cat /tmp/everalgo-py-files.txt | xargs perl -i -pe 's/\bevercore\b/everalgo/g; s/\bEverCore\b/EverAlgo/g; s/\bEVERCORE\b/EVERALGO/g'
```

Notes on regex:
- `\b` word boundary protects against false positives like `myevercore_foo` (none in practice but defensive)
- Three separate substitutions preserve case correctly (no smart-mapping)

- [ ] **Step 3: Substitute TOML / YAML / CFG config files (excluding pyproject — handled in Task 1, but extras like `cliff.toml`, `.gitlab-ci.yml`, `.pre-commit-config.yaml`, `.gitignore`, `mypy.ini`).**

```bash
rg --files-with-matches -g '*.toml' -g '*.yml' -g '*.yaml' -g '*.cfg' -g '*.ini' \
   -g '!docs/superpowers/specs/2026-05-11-rename-*' -g '!docs/superpowers/plans/2026-05-11-rename-*' \
   -g '!uv.lock' \
   'evercore|EverCore|EVERCORE' | tee /tmp/everalgo-config-files.txt
xargs perl -i -pe 's/\bevercore\b/everalgo/g; s/\bEverCore\b/EverAlgo/g; s/\bEVERCORE\b/EVERALGO/g' < /tmp/everalgo-config-files.txt
```

- [ ] **Step 4: Re-verify `uv sync` after config sweep.**

```bash
uv sync --all-packages --group dev
```

Expected: exit 0.

- [ ] **Step 5: Verify imports resolve and tests pass.**

```bash
uv run pytest -x
```

Expected: 167 tests pass (same as pre-rename). Failure modes:
- `ModuleNotFoundError: No module named 'evercore'` → some import path missed; re-run Step 2 on the offending file
- `ModuleNotFoundError: No module named 'everalgo'` → site-packages not refreshed; `rm -rf .venv && uv sync --all-packages --group dev`

- [ ] **Step 6: Verify `ruff check` and `ruff format` pass.**

```bash
uv run ruff check .
uv run ruff format --check .
```

Expected: both report all-passed.

- [ ] **Step 7: Verify `mypy` passes.**

```bash
uv run mypy .
```

Expected: Success / 0 errors.

- [ ] **Step 8: Commit Task 2.**

```bash
git add -A
git commit -m "$(cat <<'EOF'
♻️ refactor(repo): rename evercore → everalgo in source code & config

Substitutes 'evercore' / 'EverCore' / 'EVERCORE' across all .py / .toml /
.yml / .yaml / .cfg / .ini files. All 167 tests pass; ruff check, ruff
format, and mypy clean.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Documentation & changelog sweep

**Files:**
- `README.md` / `AGENTS.md` (the source of truth; `CLAUDE.md` and `.cursorrules` are symlinks)
- `CHANGELOG.md` (root) + 8 × `packages/everalgo-*/CHANGELOG.md`
- `docs/design.md`
- `docs/decisions/*.md`
- `docs/reference/README.md`
- `docs/superpowers/specs/2026-05-0[7-8]-*.md` (5 already-shipped specs — also rename the **filenames**)
- `local/superpowers/plans/*.md` (2 retained plans)

After this task, `grep -r 'evercore\|EverCore'` against `docs/`, `README.md`, `AGENTS.md`, and `CHANGELOG.md` returns only this rename-spec and this rename-plan.

- [ ] **Step 1: Rename 5 already-shipped spec filenames.**

```bash
cd /Users/admin/Documents/evermemos/evercore
git mv docs/superpowers/specs/2026-05-07-evercore-foundation-design.md     docs/superpowers/specs/2026-05-07-everalgo-foundation-design.md
git mv docs/superpowers/specs/2026-05-08-evercore-llm-injection-design.md  docs/superpowers/specs/2026-05-08-everalgo-llm-injection-design.md
git mv docs/superpowers/specs/2026-05-08-evercore-llm-stack-design.md      docs/superpowers/specs/2026-05-08-everalgo-llm-stack-design.md
git mv docs/superpowers/specs/2026-05-08-evercore-reference-impl-design.md docs/superpowers/specs/2026-05-08-everalgo-reference-impl-design.md
git mv docs/superpowers/specs/2026-05-08-evercore-testing-toolkit-design.md docs/superpowers/specs/2026-05-08-everalgo-testing-toolkit-design.md
ls docs/superpowers/specs/
```

Expected: 6 spec files total (5 renamed + this rename spec; the rename spec keeps its `evercore-to-everalgo` filename intentionally).

- [ ] **Step 2: Discover all markdown files touching the brand (excluding meta-docs).**

```bash
rg --files-with-matches -g '*.md' \
   -g '!docs/superpowers/specs/2026-05-11-rename-*' \
   -g '!docs/superpowers/plans/2026-05-11-rename-*' \
   'evercore|EverCore|EVERCORE' | tee /tmp/everalgo-md-files.txt | wc -l
```

Expected: a count between 20 and 50.

- [ ] **Step 3: Run the three-case substitution on every discovered `.md`.**

```bash
xargs perl -i -pe 's/\bevercore\b/everalgo/g; s/\bEverCore\b/EverAlgo/g; s/\bEVERCORE\b/EVERALGO/g' < /tmp/everalgo-md-files.txt
```

- [ ] **Step 4: Special-case the 2 GitLab clone URLs.**

The substitution in Step 3 already converted `gitlab.com/.../evercore.git` to `gitlab.com/.../everalgo.git`, but the new GitLab repo name is `EverAlgo` (PascalCase). Fix:

```bash
perl -i -pe 's{gitlab\.com:npc-work/aic/ai/everalgo\.git}{gitlab.com:npc-work/aic/ai/EverAlgo.git}g' AGENTS.md README.md
grep -n 'gitlab.com.*\.git' AGENTS.md README.md
```

Expected: 2 hits, both pointing to `EverAlgo.git`.

- [ ] **Step 5: Verify symlinks still resolve.**

```bash
ls -la CLAUDE.md .cursorrules
cat CLAUDE.md | head -3
```

Expected: both files are symlinks to `AGENTS.md`; `cat` shows the renamed content.

- [ ] **Step 6: Commit Task 3.**

```bash
git add -A
git commit -m "$(cat <<'EOF'
📝 docs(repo): rename evercore → everalgo across docs, ADRs, changelogs

Substitutes evercore / EverCore / EVERCORE across README, AGENTS, design.md,
ADRs, reference docs, shipped specs, local plans, root CHANGELOG, and 8
per-dist CHANGELOGs. Renames 5 shipped spec filenames. Updates 2 GitLab
clone URLs to the new EverAlgo repo path.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: CI tooling & cliff.toml sweep

**Files:**
- `cliff.toml` — header comment, `tag_pattern`, commit-parser regexes referencing `evercore-`
- `.gitlab-ci.yml`
- `.pre-commit-config.yaml`

- [ ] **Step 1: Substitute `cliff.toml`.**

```bash
perl -i -pe 's/\bevercore\b/everalgo/g; s/\bEverCore\b/EverAlgo/g; s/\bEVERCORE\b/EVERALGO/g' cliff.toml
grep -n 'tag_pattern\|evercore' cliff.toml
```

Expected: `tag_pattern` is now `everalgo-[a-z-]+/v[0-9]+\\.[0-9]+\\.[0-9]+`, no `evercore` remains.

- [ ] **Step 2: Substitute CI files (no-op if already clean).**

```bash
perl -i -pe 's/\bevercore\b/everalgo/g; s/\bEverCore\b/EverAlgo/g; s/\bEVERCORE\b/EVERALGO/g' .gitlab-ci.yml .pre-commit-config.yaml
grep -n 'evercore' .gitlab-ci.yml .pre-commit-config.yaml || echo "clean"
```

Expected: "clean" (these files do not contain dist names today, but the substitution is harmless and future-proof).

- [ ] **Step 3: Smoke-test `git-cliff` parses with the new `tag_pattern`.**

```bash
uv run git-cliff --tag everalgo-clustering/v0.2.0 --include-path 'packages/everalgo-clustering/**' --unreleased --strip header 2>&1 | head -20
```

Expected: command runs without error (output may be empty if no unreleased commits match — that is fine).

- [ ] **Step 4: Commit Task 4.**

```bash
git add -A
git commit -m "$(cat <<'EOF'
🔧 chore(ci): rename evercore → everalgo in cliff.toml and CI configs

Updates cliff.toml tag_pattern and commit-parser regexes. .gitlab-ci.yml
and .pre-commit-config.yaml unchanged in content (no dist-name references)
but swept for safety.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: End-to-end verification

This task does **not** commit code. It runs the full quality-gate suite and the final grep sweep. If anything fails, the implementer reopens Task 1-4 to fix.

- [ ] **Step 1: Clean rebuild of the workspace.**

```bash
rm -rf .venv
uv sync --all-packages --group dev
```

Expected: exit 0, all 8 `everalgo-*` distributions installed editable.

- [ ] **Step 2: Ruff check.**

```bash
uv run ruff check .
```

Expected: `All checks passed!`

- [ ] **Step 3: Ruff format check.**

```bash
uv run ruff format --check .
```

Expected: `XX files already formatted`.

- [ ] **Step 4: Mypy.**

```bash
uv run mypy .
```

Expected: `Success: no issues found in N source files`.

- [ ] **Step 5: Pytest.**

```bash
uv run pytest
```

Expected: 167 passed in approximately the same time as pre-rename.

- [ ] **Step 6: Pre-commit hook re-install + manual run.**

```bash
uv run pre-commit install
ls -la .git/hooks/pre-commit
uv run pre-commit run --all-files
```

Expected: `.git/hooks/pre-commit` exists, manual run reports all hooks Passed or Skipped.

- [ ] **Step 7: Final grep sweep — must return zero hits except the rename meta-docs.**

```bash
grep -rEn 'evercore|EverCore|EVERCORE' . \
  --include='*.py' --include='*.toml' --include='*.yml' --include='*.yaml' \
  --include='*.cfg' --include='*.ini' --include='*.md' --include='*.txt' \
  | grep -v -E '(\.git/|\.venv/|2026-05-11-rename-)' \
  | tee /tmp/everalgo-residual-hits.txt
wc -l /tmp/everalgo-residual-hits.txt
```

Expected: 0 lines.

- [ ] **Step 8: Explicitly verify extensionless files.**

```bash
grep -n 'evercore\|EverCore' LICENSE .gitignore .pre-commit-config.yaml cliff.toml || echo "clean"
```

Expected: `clean`.

- [ ] **Step 9: Verify `git log --follow` traces blame across renames.**

```bash
git log --follow --oneline packages/everalgo-core/pyproject.toml | head -5
```

Expected: history includes commits that originally touched `packages/evercore-core/pyproject.toml`.

---

## Task 6: Push & open MR

This task requires BOSS to have renamed the GitLab repository (`evercore` → `EverAlgo`) before pushing. If BOSS has not done so, pause and ask.

- [ ] **Step 1: Confirm with BOSS that GitLab repo rename is done.**

If not yet done, stop and report status.

- [ ] **Step 2: Switch remote URL.**

```bash
git remote -v
git remote set-url origin git@gitlab.com:npc-work/aic/ai/EverAlgo.git
git remote -v
```

Expected: `origin` now points to `EverAlgo.git`.

- [ ] **Step 3: Verify connectivity to new URL.**

```bash
git ls-remote origin HEAD
```

Expected: returns the current `main` commit SHA (`1c8b9d0` or newer).

- [ ] **Step 4: Push the feature branch.**

```bash
git push -u origin feat/rename-evercore-to-everalgo
```

Expected: push succeeds; output includes a "create merge request" hint URL.

- [ ] **Step 5: Open MR via `glab`.**

Squash-merge is enforced at the project level (GitLab Settings → Merge Requests → Squash commit template = `%{title}`), so no `--squash` flag is needed on `glab mr create`. The MR title becomes the squash commit message on `main`.

```bash
glab mr create \
  --title "♻️ refactor(repo): rename evercore → everalgo across the workspace" \
  --target-branch main \
  --remove-source-branch \
  --description "$(cat <<'EOF'
## Summary

Workspace-wide rename of the `evercore` brand to `everalgo`. Implements the
design in `docs/superpowers/specs/2026-05-11-rename-evercore-to-everalgo-design.md`.

- 16 directory renames (`packages/evercore-*` × 8 + `src/evercore` × 8)
- ~2127 lowercase, ~422 PascalCase, and 13 uppercase substitutions
- 9 `pyproject.toml` updates (workspace + 8 dists)
- `cliff.toml` `tag_pattern` updated
- 5 shipped specs renamed + content updated
- All ADRs, CHANGELOGs, README, AGENTS swept
- 2 GitLab clone URLs updated to new `EverAlgo.git` path

## Verification

- `uv sync --all-packages --group dev` ✓
- `uv run ruff check .` ✓
- `uv run ruff format --check .` ✓
- `uv run mypy .` ✓
- `uv run pytest` — 167 passed ✓
- `uv run pre-commit run --all-files` ✓
- Final grep sweep returns 0 hits (excluding the rename spec & plan, which
  are the only files that must keep `evercore` references)

## Out of scope (follow-up MRs)

- `memsys_opensource` documentation/comment references to evercore
  (~10 files, docstring-only, no runtime impact) — separate MR.
- Local `main` branch cleanup (12 pre-squash commits already merged into
  `origin/main` via MR !2) — BOSS to decide timing.
- `backup-before-rewrite` safety branch (from MR !2 history rewrite) —
  delete after this MR merges.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Expected: command returns an MR URL.

- [ ] **Step 6: Watch CI run; if any of the 5 jobs fail, diagnose and push fix commits.**

```bash
glab mr view --web
```

Expected: 5 jobs go green — `ruff-check`, `ruff-format`, `mypy`, `pytest`, `mr-title-lint`.

---

## Verification summary

Cumulative invariants that must hold at the end of all 6 tasks:

1. Zero `evercore` / `EverCore` / `EVERCORE` occurrences across the repository, **except** in the two rename meta-docs:
   - `docs/superpowers/specs/2026-05-11-rename-evercore-to-everalgo-design.md`
   - `docs/superpowers/plans/2026-05-11-rename-evercore-to-everalgo.md`
2. All 5 CI jobs green on the MR pipeline.
3. `git log --follow` traces blame across the renames.
4. No release tag is created (the rename is intentionally not a release event).
