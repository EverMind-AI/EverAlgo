# EverAlgo —— AI 编码助手上下文指南(中文版)

> **本文件是英文版 [`AGENTS.md`](AGENTS.md) 的中文并行翻译**,方便中文母语的工程师/AI 助手快速了解项目。**`AGENTS.md` 仍是 single source of truth(唯一权威版本)**;`CLAUDE.md`、`.cursorrules` 都是它的符号链接。两份文件出现冲突时以英文版为准。

如果你是 AI 助手(Claude Code / Cursor / Copilot / Codex 等)第一次接入这个仓库,先读这份文件,再读 `docs/design.md` 拿完整架构,接着读 `docs/decisions/` 下相关的 ADR(架构决策记录),最后再动代码。

---

## 1. 项目定位

**EverAlgo** 是一个用于**记忆抽取与排序**的算法库 —— *不是*服务、*不是*框架。

- **纯算法库**。所有 memory 的抽取 / 融合 / 重排策略都在这里。库本身**无状态**:不连数据库、不读写文件系统、不持有任何业务状态。
- **两条主轴**。每个 operator(算子)都落到下面两条主轴之一,合同对称(无状态、内存进内存出):
  - **Extract(抽取)** —— 写路径。输入:结构化单元(比如 `MemCell`)。输出:结构化记忆(`Episode` / `Profile` / `Case` / `Skill` / …)。
  - **Rank(排序)** —— 读路径。输入:多路召回候选 + 预先取好的跨记忆关联。输出:排序后的记忆列表。Ranker **不做任何存储 I/O**;跨记忆关联(如 `Episode → AtomicFact`)由调用方提前查好后传进来。
- **编排在上游**。什么时候调、按什么顺序、并发多少、怎么持久化到 markdown 文件系统 —— 这些全部由 **evermem** 负责。EverAlgo 不关心调用方是开源还是云端商业版,两条线共用同一份算法代码。

要更深入的背景和动机,看 `docs/design.md` §1.1。

---

## 2. 仓库结构

```
everalgo/                              # monorepo,uv 虚拟工作区
├── pyproject.toml                     # 工作区根,[tool.uv] package = false
├── uv.lock                            # 由 `uv sync` 生成 —— 不要手动编辑
├── AGENTS.md  ← canonical 英文版      # CLAUDE.md 和 .cursorrules 都是它的 symlink
├── AGENTS.zh.md  ← 你在这里            # 并行中文翻译
├── README.md
├── LICENSE                            # MIT
├── .gitignore  .gitlab-ci.yml
├── docs/
│   ├── design.md                      # ⚠️ 必读 —— 完整架构
│   ├── decisions/                     # ADR(ADR-001…012)—— 想挑战设计前先读
│   ├── concepts/                      # 高层架构笔记
│   └── reference/                     # API 参考(按 distribution 切分)
├── packages/
│   ├── everalgo-core/                 # types、llm(含 providers)、prompts、testing
│   ├── everalgo-boundary/             # MemCell 抽取器 + tokenize / split
│   ├── everalgo-clustering/           # cluster_by_geometry / cluster_by_llm
│   ├── everalgo-rank/                 # 4 个 ranker + fusion / weight / rerank
│   ├── everalgo-parser/               # 多模态原始文件 → ParsedContent
│   ├── everalgo-user-memory/          # Episode / Foresight / AtomicFact / Profile
│   ├── everalgo-agent-memory/         # AgentCase / AgentSkill
│   └── everalgo-knowledge/            # KnowledgeMemory
└── tests/
```

8 个可发布的 distribution(发行包)通过 [PEP 420](https://peps.python.org/pep-0420/) **原生 namespace package(命名空间包)**机制共享 **`everalgo.*` 命名空间**:每个 `packages/*/src/everalgo/` 目录都**故意不放** `__init__.py`,而子包(`everalgo/<subpkg>/__init__.py`)是常规 package。这是 [PyPA 推荐的布局](https://packaging.python.org/en/latest/guides/packaging-namespace-packages/#native-namespace-packages),适用于纯 Py3 + pip 安装的项目。效果是:即便 `everalgo-user-memory` 和 `everalgo-boundary` 来自不同的发行包,`from everalgo.user_memory import EpisodeExtractor` 也能正常工作。工业界参考:`google-cloud-*`(100+ 个发行包共享 `google.cloud.*`)、`sphinxcontrib-*`(6 个官方 Sphinx 扩展共享 `sphinxcontrib.*`)。

开发流程构建在 **uv 虚拟 workspace** 之上(根目录 `[tool.uv] package = false`,成员在 `packages/*` 下)。同样形态的项目:[Apache Airflow](https://github.com/apache/airflow)(100+ workspace 成员 + 单 root lockfile)和 [pydantic-ai](https://github.com/pydantic/pydantic-ai)。注意这俩只是 *uv workspace* 的参考 —— Airflow 的 `airflow.providers.*` 用的是 pkgutil 风格的老式 namespace,不是 PEP 420;pydantic-ai 用了三个独立 namespace,不是共享一个。LangChain / LlamaIndex 只用作 *monorepo* 布局的参考 —— 它俩都没用 uv workspace,而是每个包独立 venv + lockfile。

**依赖拓扑**(完整图和理由见 `docs/design.md` §1.3):

```
                                everalgo-core
                                     ▲
       ┌────────────┬────────────┬──┴───────────┬───────────┐
       │            │            │              │           │
   boundary    clustering        rank         parser
       ▲            ▲                                       ▲
       └────────────┤                                       │
                    │                                       │
            user-memory ── agent-memory          everalgo-knowledge
```

---

## 3. 快速开始

```bash
# 前置:Python 3.12+ 和 uv (https://docs.astral.sh/uv/)
git clone git@gitlab.com:npc-work/aic/ai/everalgo.git
cd everalgo

# 把 8 个包都以 editable 方式装进共享 venv
uv sync --all-packages

# 跑整个 workspace 的测试
uv run pytest

# Lint + 格式检查
uv run ruff check .
uv run ruff format --check .

# 类型检查
uv run mypy .
```

只想动单个包?只同步它的依赖:

```bash
uv sync --package everalgo-clustering
uv run pytest packages/everalgo-clustering/tests/   # 等单包测试目录建好后
```

### Pre-commit hook(必装)

仓库自带 `.pre-commit-config.yaml`,在每次 commit 时跑 `ruff check --fix` + `ruff format` + 一组常规清理(行尾空白、文件末尾换行、merge-conflict 标记、超大文件、行尾符、YAML / TOML 语法)。这套工作流跟 sklearn、pydantic、dspy、langchain、pandas、numpy 一致。

**每次 clone 之后都要装一遍并验证**(hook 是每个 clone 独立的状态,**不在仓库里**,新 clone / 新机器默认就是没装的状态):

```bash
uv sync --all-packages --group dev   # 把 pre-commit 装进 workspace venv
uv run pre-commit install            # 创建 .git/hooks/pre-commit
ls -la .git/hooks/pre-commit         # 必须存在且可执行
```

如果第三步显示 `No such file or directory`,说明 `install` 步骤**悄无声息地失败了**,后面**每一次 `git commit` 都会默默跳过 lint**。必须先修好再干活。

#### 常见坑:`--all-files` ≠ hook 装好了

`uv run pre-commit run --all-files` 是**手动**触发的命令。它只能验证 hook 配置本身是不是健康的,**并不说明** `git commit` 真的会自动触发它。只有 `.git/hooks/pre-commit` 文件存在并可执行的时候,hook 才会自动跑。

这个坑是真实踩过的:跑 `--all-files` 看到 "9/9 Passed",可能掩盖了一整个 sprint 没装 hook 的事实 —— 每次 `git commit` 都默默绕过 lint,直到 CI 抛出本该被本地 hook 拦下的违规才暴露。**装完 `install` 一定要 `ls .git/hooks/pre-commit` 验证一下。**

#### 常用用法

开 MR 前对全树跑一遍(把 hook 装好之前提交的东西也覆盖到):

```bash
uv run pre-commit run --all-files
```

定期更新 pinned 的 hook 版本:

```bash
uv run pre-commit autoupdate
```

#### 故意**不**放进 pre-commit 的东西

- **`mypy` / `pyright`** —— 在 8 个包的 PEP 420 workspace 上做严格类型检查每次要好几秒,放进 commit 会让人觉得卡;改由 CI 把关(pydantic / sklearn / openai-python / anthropic-sdk-python 做法相同)。
- **`pytest`** —— 同上,CI 兜底。

### 编辑器集成(推荐)

Pre-commit 在 commit 时才跑。想要每次敲键盘就拿到反馈,顺手装上 ruff 的编辑器插件:

- **VSCode / Cursor**:装 [Ruff 扩展](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff)。打开保存时格式化,让编辑器自动跑 `ruff check --fix` 和 `ruff format`。
- **PyCharm / IntelliJ**:装 [Ruff 插件](https://plugins.jetbrains.com/plugin/20574-ruff)。
- **Vim / Neovim**:通过 LSP 配置 ruff(`ruff-lsp` 或 `nvim-lspconfig` 内置 LSP)。

CI 流水线(`.gitlab-ci.yml`)在每个 MR 上重跑 `ruff check .` + `ruff format --check .` + `mypy .` + `pyright`,作为最终兜底。两个类型检查器都跑是因为它们抓的东西略有不同 —— `openai-python` 和 `anthropic-sdk-python` 也是这套双检查方案。pre-commit 和编辑器是为了 **反馈速度**,CI 才是闸门。

---

## 4. 常用命令

| 操作 | 命令 |
|---|---|
| 安装 workspace(editable) | `uv sync --all-packages --group dev` |
| 跑全部测试 | `uv run pytest` |
| 跑指定测试 | `uv run pytest path/to/test.py::test_name -v` |
| Lint | `uv run ruff check .` |
| 格式化 | `uv run ruff format .` |
| 类型检查(mypy) | `uv run mypy .` |
| 类型检查(pyright) | `uv run pyright` |
| 构建单个 distribution | `cd packages/everalgo-core && uv build` |
| 给某个包加运行时依赖 | `uv add --package everalgo-clustering numpy` |
| 给 workspace 加 dev 工具 | `uv add --group dev pytest-asyncio` |

参考:[uv workspace 文档](https://docs.astral.sh/uv/concepts/projects/workspaces/)。

---

## 5. 代码规范

完整理由在 `docs/design.md` §1.4 和 ADR 010 / 011。硬性规则:

- **命名契约 —— `a` 前缀代表 async**。`aextract` / `arank` / `adetect` / `aparse` 这些方法是**原生异步**(做真实 I/O —— LLM、网络等),要用 `await` 调用;没有 `a` 前缀的(`rank` / `extract` / `count_tokens` / `rrf` 等)是**同步**(纯计算,不做 I/O),直接调用。跟 `dspy.acall` / `litellm.acompletion` / `instructor.AsyncInstructor` 同样的约定。
- **I/O 算子:async 优先 + sync 桥接**。原生 async 走 `asyncio`;sync 版本通过 `asgiref.async_to_sync` 派生,给非事件循环的调用方用(CLI 脚本、普通单元测试)。**不要**在已经跑着事件循环的地方调 sync 桥接版。
- **纯计算算子:只提供 sync**。`fusion.rrf`、`_tokenize.count_tokens`、聚类距离计算等不包 async 壳。跟 numpy / scipy / sklearn / pandas 一致。
- **Prompt 作为 Python 字符串模块**。具体 prompt 字符串以模块级常量的形式放在 `<subpkg>/prompts/{en,zh}/<name>.py`。改 prompt = 改 `.py` 文件。**不用**外部的 `.md` / `.yaml` / `.toml` prompt 存储。调用方定制:每次调用传 `prompt=` 参数(细粒度)、或启动时 monkey-patch 模块常量(粗粒度)。
- **用 `Protocol` 做类型,不用 `ABC`**。EverAlgo 的算子无状态,实现方不需要继承任何东西。详见 ADR 011。
- **算法代码里不搞依赖注入**。模块级函数 + 全局 config + 测试里 monkeypatch。算法作者从写代码到能跑应该只差一个回车,别强加框架仪式感。
- **I/O 算子的 sync 桥接:照 ADR 010 第 199-214 行,写一行 `extract = async_to_sync(aextract)`;不要引入 `DualInterface` mixin**。这让类型推导可预期,也避免 metaclass 黑魔法,跟 ADR 里展示的模式一致。`async_to_sync` 这个 helper 来自 `asgiref.sync`。
- **Lint 配置**。整个 workspace 的 ruff 配置在根 `pyproject.toml`(`line-length = 120`,目标版本从 `requires-python = ">=3.12"` 推出,规则集是 pytorch + pydantic-ai 的交集)。docstring 用 NumPy 风格 —— 跟 numpy / scipy / scikit-learn / pandas 一致,是科学 Python 算法库的行业标准。
- **日志规范**。LLM / I/O 路径上用 `logger = logging.getLogger(__name__)`,配懒计算的 `%`-format(`logger.debug("count=%d", n)` —— **绝不**在 log 调用里写 f-string),`except` 块里用 `logger.exception(...)`。用户行为类问题和废弃提示用 `warnings.warn(..., stacklevel=2)`;纯算法错误用 `raise ValueError(...)` 带详细信息(numpy 风格 —— `shapes (3,4) and (5,6) not aligned` 这种)。每个公开子包的 `__init__.py` 都已经挂了 `NullHandler`;`everalgo.llm` 默认开了 `SensitiveHeadersFilter`。**库代码里禁用**:`logging.basicConfig`、`addHandler`(除了 `NullHandler` 都不行)、`setLevel`、显式 `propagate = True/False`,以及任何模块级的 `logging.warning(...)` / `logging.error(...)` / `logging.getLogger()`(不传 name)/ `logging.root.*` —— 这些都打到 root logger,是 application 的活儿。**DEBUG 日志里禁用**:请求/响应 body、prompt 文本、模型输出(Filter 只能脱敏 header,body 里的 PII 它看不见)。性能计时是用户的事(`cProfile` / `line_profiler` / `%timeit`);库不打 duration。ruff 规则集 `G` + `LOG` + `TRY` 在 lint 阶段强制以上规则。完整理由、级别语义和 logging-vs-warnings-vs-exception 决策矩阵见 [ADR 013](docs/decisions/013-logging-conventions.md)。
- **手动换行时把 `line-length = 120` 用满**。Python 注释 / docstring 和 TOML / YAML 注释,每行写到 100–115 字符再换行 —— 不要出于习惯在 70 / 79 / 80 / 88 / 100 字符就预换行。`E501` 在 ignore 列表里,ruff 也不会标过短的行,所以这是写作纪律不是 lint。3 行注释如果能干净地压到 2 行 @120,就写 2 行。例外:bullet 列表、代码块、NumPy docstring 的 `name : type` 字段(故意一行一条),以及那些自然断句更好读的地方。本仓库的 markdown 文件**故意**用**一段一行**(不硬换行)的风格 —— Prettier 默认就这么排版,GitHub 渲染也干净 —— 所以 `.md` 散文豁免 100–115 规则,靠编辑器软换行。
- **代码、配置、commit message 只用英文**。所有 Python 代码、注释、标识符、`pyproject.toml` 注释、CI 文件、commit message 必须是英文。`evermem` 用 pre-commit hook 强制这条,这里同样适用。`docs/` 下的设计讨论文档(`design.md`、`decisions/`、`concepts/`)可以保留中文 —— 它们反映的是设计讨论时的工作语言,不是代码本身。

---

## 6. 分支与提交

**分支策略:trunk-based(主干开发)**(参考 DSPy / scikit-learn / instructor / pydantic —— 四个 Python 算法库参考样板都这么做;不搞 GitFlow)。

- `main` 是唯一长期存活的分支。它**受 GitLab 保护**(Settings → Repository → Protected branches):所有人都不能直接 push,落到 `main` 的唯一路径是 Merge Request。
- 功能开发走短期分支:`feat/<topic>` / `fix/<bug>` / `docs/<topic>` / `refactor/<topic>`。开 MR → squash merge 到 `main`。
- 发布 = 在 `main` 上打 tag,按 distribution 维度的 SemVer:`everalgo-clustering/v0.2.0`。每个 distribution 独立版本节奏(HuggingFace 模式;见 `docs/design.md` §1.3 和 `README.md` "Cutting a release")。
- 维护分支(`0.1.X-fixes`)**只在**已发布版本需要 back-port 时才开;默认不开。

**Commit message:Gitmoji + Conventional Commits**。格式:`<emoji> <type>(<scope>): <description>`。

```text
✨ feat(clustering): add cluster_by_llm decision prompt zh variant
🐛 fix(boundary): correct token count for emoji-only chat segments
♻️ refactor(rank): extract shared fusion helper from case / skill rankers
✅ test(user-memory): cover EpisodeExtractor tail-merge edge case
📝 docs(design): clarify §2.4 cluster_previews shape
```

允许的 `type`:`feat` / `fix` / `docs` / `style` / `refactor` / `perf` / `test` / `build` / `ci` / `chore` / `revert`。

**MR 标题是关键路径**。GitLab 配置(Settings → Merge Requests → Squash commit template = `%{title}`)让 MR 标题原样落到 `main` 上的 squash commit。MR 标题**必须**符合上面的格式,因为 release notes 生成器(`git cliff`,见 `cliff.toml` + `README.md` "Cutting a release")会解析这些 message 来组织每个 distribution 的 CHANGELOG。

**Scope = 去掉 `everalgo-` 前缀的 distribution 名**。用 `clustering` / `rank` / `core` / `boundary` / `parser` / `user-memory` / `agent-memory` / `knowledge`。跨切面的改动(CI、monorepo 工具、根目录文档)用 `ci` / `release` / `repo` / `design` / `docs` 作 scope,或者干脆不带 scope。

**Squash 对 per-distribution 过滤很关键**。`git cliff --include-path 'packages/everalgo-<name>/**'` 按改动路径过滤 commit。Squash merge 保证一个 commit = 一个 MR = 一条带 scope 的 Conventional-Commit,这正是 git-cliff 分组的单位。

---

## 7. 新加一个算法算子的步骤

加新的 extractor / ranker / clusterer 时按这份 checklist:

1. **选子包**。基于产品轴(user_memory / agent_memory / knowledge)或工具轴(boundary / clustering / rank / parser)选 `packages/everalgo-<dist>/src/everalgo/<subpkg>/`。拿不准看 `docs/design.md` §1.2。
2. **建模块**。`<subpkg>/<operator>.py` —— 模块级函数,或一个无状态类实现 `everalgo.protocols` 里对应的 Protocol。
3. **写 prompt(如果用 LLM)**。把 prompt 字符串作为模块级常量放进 `<subpkg>/prompts/en/<operator>.py`(需要中文变体的话再加 `zh/<operator>.py`)。
4. **重新导出公开 API**。如果这个算子是其所在 facade 子包公开 API 的一部分,加进 `<subpkg>/__init__.py` 的 re-export 区段和 `__all__`。re-export 模式见 `docs/design.md` §1.3。
5. **接好依赖**。如果新代码引入了新的第三方库,用 `uv add --package everalgo-<dist> <library>` 加,会自动更新对应包的 `pyproject.toml`。
6. **写测试**。用 `everalgo.testing.fake_llm` 避免真实 API 调用;用 `everalgo.testing.assertions` 做结构化记忆断言。
7. **本地跑全套 lint + 格式化 + 类型检查 + 测试** 后再开 MR(`uv run ruff check . && uv run ruff format --check . && uv run mypy . && uv run pyright && uv run pytest`)。
8. **记录架构决策**。如果这个算子涉及非平凡的设计选择(新的公开 Protocol、新的 distribution 边界、打破某个约定),在 `docs/decisions/` 下加一个新的 ADR,编号接最新的之后。

---

## 8. 加一个新 LLM Provider

Provider 嵌在 `everalgo-core` 的 `everalgo/llm/providers/<provider>/` 下(根据 ADR 004 —— provider **嵌套**在 `llm` 里,不单独发包;沿用 litellm / instructor / dspy / llama-index 的惯例)。

1. 建 `everalgo/llm/providers/<provider>/__init__.py` 和 `client.py`。
2. 实现 `everalgo.protocols` 里的 `LLMClient` Protocol(sync + async 的 chat / stream)。
3. 在 `everalgo/llm/routing.py` 注册自动识别(URL 或环境变量识别 —— 照已有 pattern 写)。
4. 把 provider 原生异常映射到统一的 `LLMError` 子类树。
5. 只有 provider 需要特殊 prompt 格式时才加 per-provider prompt(罕见 —— 多数 provider 都是 OpenAI 兼容的)。
6. 测试放在 `packages/everalgo-core/tests/llm/providers/<provider>/`。CI 有真实 key 的时候**不要**在 HTTP 层 mock;否则用 `respx` 录 fixture。
7. 公开 API 有变就更新 `docs/design.md` §2.5 和 `AGENTS.md`。

---

## 9. 测试规范

- **`asyncio_mode = "auto"`** 已在整个 workspace 配好(见根 `pyproject.toml`);普通 `async def test_*()` 不用任何装饰器就能跑。
- **用 `everalgo.testing.fake_llm`** 做确定性 LLM 回放。单元测试里**不要**在 HTTP 层打桩 —— provider 改协议就崩。
- **跨包集成冒烟测试** 放在 workspace 根的 `tests/` 目录。Per-distribution 的单测 等单个 distribution 体量长起来后,colocate 在 `packages/everalgo-<name>/tests/` 下(参考 pydantic-ai 的 `pydantic_ai_slim/tests/`)。
- **默认 `pytest` 不打真实网络**。Provider 的网络测试加 `@pytest.mark.integration`,用环境变量门禁。

---

## 10. 参考资料

| 主题 | 在哪里 |
|---|---|
| 架构(权威) | [`docs/design.md`](docs/design.md) |
| 架构决策(ADR) | [`docs/decisions/`](docs/decisions/) |
| `evermem` 合同来源 | Confluence —— 链接见 `docs/design.md` 的 header |
| uv workspace 概念 | https://docs.astral.sh/uv/concepts/projects/workspaces/ |
| PEP 420 namespace 包 | https://peps.python.org/pep-0420/ |
| PEP 8(风格)/ 257(docstring)/ 484(类型注解) | https://peps.python.org/ |
| Conventional Commits | https://www.conventionalcommits.org/ |
| Gitmoji | https://gitmoji.dev/ |

---

## 11. 编辑这份文件

这份文件是人类工程师和 AI 助手在本仓库的协作契约。改它的时候请:

1. **保持英文版 `AGENTS.md` 为 canonical**(权威版本)。`CLAUDE.md` 和 `.cursorrules` 继续是 symlink。`AGENTS.zh.md` 是并行翻译,如果与英文版冲突以英文版为准。
2. **改动后同步翻译**。改英文版的同一个 MR 里把中文版也同步过来,避免漂移。如果只是 typo / 措辞调整,中文版可以下一个 MR 跟。
3. **每个具体决策都要有出处**。引用 `docs/design.md` 的某节、某个 ADR、或公开规范 / 明星项目 URL。不要写没根据的断言。
4. **仓库结构或工作流变了**,§2(布局)和 §3-§4(命令)要在**同一个 MR**里同步更新。
