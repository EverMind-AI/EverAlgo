# EverAlgo Documentation

## Contents

| Document | Purpose |
|---|---|
| [index.md](index.md) | Project front door — what EverAlgo is, distributions, where to start |
| [installation.md](installation.md) | Install options, prerequisites, troubleshooting |
| [getting-started.md](getting-started.md) | 5-minute end-to-end runnable example |
| [version-policy.md](version-policy.md) | SemVer policy, Python version support, deprecation |
| [contributing.md](contributing.md) | How to contribute; links to AGENTS.md for the full rules |
| [concepts/architecture.md](concepts/architecture.md) | High-level architecture, subpackage layout, naming, LLM injection |
| [concepts/stateless-design.md](concepts/stateless-design.md) | What business-stateless means, including the parser I/O boundary |
| [concepts/async-sync-bridge.md](concepts/async-sync-bridge.md) | The `a`-prefix convention and the sync bridge |
| [concepts/stage1-boundary-detection-flow.md](concepts/stage1-boundary-detection-flow.md) | Stage 1 extract-base flow |
| [concepts/stage5-agentic-retrieval-flow.md](concepts/stage5-agentic-retrieval-flow.md) | Stage 5 agentic retrieval flow |
| [releasing.md](releasing.md) | Release process and checklist |
| [api/README.md](api/README.md) | Index linking to each package README and the generated API reference |

## Diátaxis map

```
Tutorials  →  getting-started.md
Reference  →  api/README.md  +  packages/*/README.md
Concepts   →  concepts/
How-to     →  releasing.md  +  (see AGENTS.md §7–§8 for operator and provider checklists)
```
