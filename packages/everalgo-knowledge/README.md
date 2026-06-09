# everalgo-knowledge

> [!WARNING]
> **NOT YET IMPLEMENTED — placeholder distribution.** All entry points raise `NotImplementedError`. The `everalgo.knowledge` namespace is reserved on PyPI. Track progress in [`docs/concepts/architecture.md`](../../docs/concepts/architecture.md).

File-based knowledge extractors for EverAlgo — turns parsed multimodal content (`ParsedContent`) into `KnowledgeMemory` records.

See the umbrella project: [EverAlgo monorepo](../../README.md) and the architecture document at [`docs/concepts/architecture.md`](../../docs/concepts/architecture.md).

## Install

```bash
# This package is NOT published to PyPI (Private :: Do Not Upload classifier).
# Install within the workspace only:
uv sync --package everalgo-knowledge
```

## Public surface (stubbed)

| Symbol | Module | Role |
|---|---|---|
| `KnowledgeExtractor` | `everalgo.knowledge` | Stub — raises `NotImplementedError` |

## Related distributions

- [`everalgo-parser`](../everalgo-parser/) — produces `ParsedContent` from raw files (fully implemented except video, which is deferred pending ADR)
- [`everalgo-core`](../everalgo-core/) — `KnowledgeMemory` type is defined here
