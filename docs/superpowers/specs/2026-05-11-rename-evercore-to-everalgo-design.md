# Rename `evercore` → `everalgo` 设计文档

> **Status:** Approved by BOSS on 2026-05-11. Implementation plan in `docs/superpowers/plans/2026-05-11-rename-evercore-to-everalgo.md`.

## 1. 背景

EverCore 仓库整体改名为 EverAlgo。GitLab 仓库会由 BOSS 改名(`evercore` → `EverAlgo`),代码侧需要把 namespace / dist 名 / 目录名 / 文档 / 配置里的 `evercore / EverCore / EVERCORE` 全部替换为 `everalgo / EverAlgo / EVERALGO`。本次 rename 是 rebrand,不是 fork — 历史 ADR / CHANGELOG / spec 一并替换,不留尾巴。

无 release tag 已发到 PyPI(`git tag -l 'evercore*'` 为空),所以**没有外部已发版本需要兼容**,本次 rename 是 clean break。

## 2. 范围

### 2.1 大小写映射(三段,无歧义)

| 旧 | 新 | 出现位置 | 出现次数 |
|---|---|---|---|
| `evercore` | `everalgo` | 包名 / import 路径 / dist 名 / 目录名 / URL slug / `tag_pattern` | ~2127 |
| `EverCore` | `EverAlgo` | 品牌名 / 文档标题 / README 大标题 / commit subject | ~422 |
| `EVERCORE` | `EVERALGO` | env var 前缀(`EVERCORE_LLM_*`,主要在 `docs/design.md` 示例) | 13 |

替换工具:`perl -i -pe`(三段独立 case-sensitive 替换),不依赖"智能"映射避免边界 bug。

### 2.2 In Scope(全量替换)

**目录 rename(8 个,用 `git mv` 保留 blame)**:
- `packages/evercore-core` → `packages/everalgo-core`
- `packages/evercore-boundary` → `packages/everalgo-boundary`
- `packages/evercore-clustering` → `packages/everalgo-clustering`
- `packages/evercore-rank` → `packages/everalgo-rank`
- `packages/evercore-parser` → `packages/everalgo-parser`
- `packages/evercore-user-memory` → `packages/everalgo-user-memory`
- `packages/evercore-agent-memory` → `packages/everalgo-agent-memory`
- `packages/evercore-knowledge` → `packages/everalgo-knowledge`

**文件内容替换**:
- 全部 `*.py`(代码 + tests)— 主要是 `from evercore.xxx import ...`、模块 docstring
- 全部 `*.toml`(根 + 8 个 dist 的 `pyproject.toml` 的 `name = "evercore-*"`、workspace `members`、依赖关系如 `"evercore-core"`)
- `cliff.toml` 的 `tag_pattern` 与 commit parser 引用
- `.gitlab-ci.yml`(若有 dist 名引用)
- `.pre-commit-config.yaml`(若有引用)
- 根 `CHANGELOG.md` + 8 份 `packages/*/CHANGELOG.md`(含 `[0.1.0]` 历史条目)
- `README.md` / `AGENTS.md` / `CLAUDE.md`(symlink,实体跟 AGENTS.md)/ `.cursorrules`(symlink,实体跟 AGENTS.md)
- `docs/design.md`(主架构)
- `docs/decisions/`(ADR-001 ~ 010)
- `docs/superpowers/specs/` 其它已 ship 的设计文档 — **文件名也 rename**:`2026-05-07-evercore-foundation-design.md` → `2026-05-07-everalgo-foundation-design.md` 等(5 个)
- `local/superpowers/plans/`(本地计划存档,2 个文件)
- `scripts/check_mr_title.py`(error message / docstring 若有引用)
- `LICENSE`(若 Copyright 行提到 EverCore)
- GitLab clone URL 2 处(`AGENTS.md:73`、`README.md:34`):`gitlab.com:npc-work/aic/ai/evercore.git` → `.../EverAlgo.git`

### 2.3 Out of Scope

**本次 PR 不改**:
- `uv.lock` — 让 `uv sync --all-packages --group dev` 在 T8 验证时自动重生成
- `.git/` — 历史 commit 不可改
- `.venv/` — 在 T8 重建
- 本仓库以外的 sibling repo `memsys_opensource`(docstring 提到 "calls evercore",~10 个文件)— **follow-up**,本次任务结束后我另开独立 MR
- `memsys_enterprise` / `evermemos-backend` — 经 grep 验证完全无引用
- GitLab Issue / 历史 MR description / 历史 commit subject 里的 "evercore" — 改不动,无影响

**绝对禁止替换(meta 自指文档)**:
- 本 spec 自身:`docs/superpowers/specs/2026-05-11-rename-evercore-to-everalgo-design.md`
- 对应 plan:`docs/superpowers/plans/2026-05-11-rename-evercore-to-everalgo.md`

理由:这两个文件正文必然提到"evercore → everalgo"的映射关系,内容里的 evercore 是描述变更的元名词,不是被改的对象。脚本必须 exclude 这两个路径。

## 3. 分支与 MR 策略

- **Base**:`origin/main`(`1c8b9d0`,MR !2 squash-merge 后的最新 main)
- **Feature branch**:`feat/rename-evercore-to-everalgo`
- **单个大 MR,squash 合并**:rename 是原子动作,拆分无意义;reviewer 一次看清全貌
- **MR title**:`♻️ refactor(repo): rename evercore → everalgo across the workspace`
  - Gitmoji ♻️ + Conventional Commit `refactor` + scope `repo`(跨 dist 改动用 `repo`,符合 AGENTS.md §6)
  - 通过 `scripts/check_mr_title.py` mr-title-lint
- **commit 数量**:理想为单个 commit;允许拆 2-3 个(目录 rename / 内容替换 / 验证 fix)但最终都 squash 进 MR

## 4. GitLab 仓库改名协同

BOSS 完成 GitLab Web UI 改名(`evercore` → `EverAlgo`)后,我在 `git push` 前执行:

```bash
git remote set-url origin git@gitlab.com:npc-work/aic/ai/EverAlgo.git
git remote -v   # verify
```

GitLab 在 rename 后会自动配置 HTTP redirect 到新 URL 一段时间(默认 90 天),但显式切换更稳。如果在 BOSS 改名前 push,push 走旧 URL,自动 redirect 通常能 work(GitLab 对 git+SSH 也 redirect),但仍建议**先改 GitLab 名,再切 remote,再 push**。

## 5. 执行方法(T1-T9)

按 superpowers `subagent-driven-development` 流程,9 个 verifiable checkpoint:

| Task | 内容 | 验证 |
|---|---|---|
| T1 | 目录 rename:`git mv packages/evercore-* packages/everalgo-*` ×8 | `ls packages/` 显示 8 个 `everalgo-*` |
| T2 | 根 `pyproject.toml` + 8 个 dist `pyproject.toml` 替换 dist 名、workspace members、依赖 | `uv sync --all-packages` 通过 |
| T3 | 全仓 Python / TOML / YAML / config / script 文本三段替换(**必须 exclude `docs/superpowers/specs/2026-05-11-rename-*` 与 `docs/superpowers/plans/2026-05-11-rename-*`,见 §2.3**) | `grep -rE 'evercore\|EverCore\|EVERCORE' --include='*.py' --include='*.toml' --include='*.yml' --include='*.yaml' --include='*.cfg'` 为空 |
| T4 | `docs/` 全文替换 + spec 文件名 rename(`evercore-*` → `everalgo-*`)— **本 spec 与对应 plan 不动**(文件名也不改,见 §2.3) | `grep -r 'evercore\|EverCore\|EVERCORE' docs/` 仅命中本 spec 与对应 plan |
| T5 | 根 `CHANGELOG.md` + 8 份 `packages/*/CHANGELOG.md` 替换 | `grep -r 'evercore\|EverCore' CHANGELOG.md packages/*/CHANGELOG.md` 为空 |
| T6 | `README.md` / `AGENTS.md` 替换 + 2 处 GitLab URL 替换 | `grep -E 'evercore\|EverCore' README.md AGENTS.md` 为空(除示例 tag 名注解) |
| T7 | `cliff.toml` `tag_pattern` 更新(`evercore-` → `everalgo-`) + `.gitlab-ci.yml` 等 CI 配置 verify | `grep evercore cliff.toml .gitlab-ci.yml .pre-commit-config.yaml` 为空 |
| T8 | **End-to-end 验证套件**: | |
| | (a) `rm -rf .venv && uv sync --all-packages --group dev` | exit 0 |
| | (b) `uv run ruff check .` | All checks passed |
| | (c) `uv run ruff format --check .` | All files formatted |
| | (d) `uv run mypy .` | Success / 0 errors |
| | (e) `uv run pytest` | 167 passed |
| | (f) `uv run pre-commit run --all-files` | All hooks Passed |
| | (g) `uv run pre-commit install` + `ls -la .git/hooks/pre-commit` | exists |
| | (h) 最终 grep 扫描:`grep -rEn 'evercore\|EverCore\|EVERCORE' . --include='*.py' --include='*.toml' --include='*.yml' --include='*.yaml' --include='*.cfg' --include='*.md' --include='*.txt' \| grep -v -E '(\.git/\|\.venv/\|2026-05-11-rename-)'` + 显式 `grep -n 'evercore\|EverCore' LICENSE .gitignore .pre-commit-config.yaml`(无扩展名文件) | 两条命令均输出空 |
| T9 | commit + (BOSS 改完 GitLab 仓库名后)切 remote + push + 开 MR | MR 链接返回 |

## 6. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| `uv sync` 解析失败(workspace members 路径错) | 中 | T8 红 → 回退修 T2 | T2 后立即 `uv sync` 验证,不等到 T8 |
| 替换误伤本 spec/plan 的元描述 | 中 | spec 自指混乱 | 替换脚本 hard-coded exclude `2026-05-11-rename-*` |
| 替换误伤外部 reference(URL 锚点 / 论文 / 项目名) | 低 | 链接 404 / 引用失真 | 替换后 grep `EverCore` / `evercore` 在 docs 中,人工 review 剩余 hit 是否为外部引用(目前已扫描:除自身外仅有内部引用) |
| pre-commit hook 在 rename 后 broken | 低 | commit 阶段失败 | T8 (f)(g) 显式验证 hook 安装与运行 |
| ruff format 因 import 路径变化重 wrap | 低 | format check fail | `ruff format` 在 T3 后立即 run 一次自动修正 |
| GitLab Web 改名时机错配导致 push 失败 | 低 | push 报 404 | 改名后 `set-url`,GitLab 90 天 redirect 兜底 |

## 7. 验收标准

- 全套 lint / type / test 通过(T8 a-f)
- meta-exclude 后 `grep -rE 'evercore\|EverCore\|EVERCORE'` 0 hit(T8 h)
- 8 个 dist 目录正确 rename,`git log --follow packages/everalgo-core/pyproject.toml` 能追溯到原 `evercore-core/pyproject.toml`(blame 不断)
- MR 通过 CI 全部 5 个 job:`ruff-check` / `ruff-format` / `mypy` / `pytest` / `mr-title-lint`
- `evercore-*/v*` tag 正则在 `cliff.toml` 已更新为 `everalgo-*/v*`,但**不打实际 tag**(本次仅 rename,不发版)

## 8. 后续(本 MR 之外)

1. memsys_opensource 文档/comment 引用 → 我另开独立 MR 处理(约 10 个文件,docstring 替换)
2. BOSS 完成 GitLab Web UI rename → 切 remote
3. 本地 main 分支与 origin/main 分歧的清理(本地 12 个 commit 已 squash 进 origin),由 BOSS 决定时机执行 `git branch -m main main-old && git checkout -b main origin/main`(destructive,不擅自做)
4. `backup-before-rewrite` 分支删除(MR !2 历史重写时建的保险绳,merge 完了删)

## 9. 不接受的备选方案

- **per-package 拆 MR**:拒绝。`evercore.*` namespace 跨 dist,改一半 import 全断,违反 trunk-based 原则
- **codemod 工具 `libcst` / `bowler`**:拒绝。过度工程,grep + perl -i 三段 case sed 足够,无 Python AST 级歧义(没有把 `evercore` 当变量名而非字符串的场景)
- **保留 ADR 历史名字**:拒绝(BOSS 已选 A 全量替换)
- **本仓库 + memsys_opensource 一次 PR 完成**:拒绝。不同仓库 = 不同 CI = 不同 reviewer,跨仓库一次 PR 反 monorepo-外的常规协作模式
