# everalgo-parser

> [!WARNING]
> **NOT YET IMPLEMENTED — placeholder distribution.** All entry points raise `NotImplementedError`. The `everalgo.parser` namespace is reserved on PyPI. Track progress in [`docs/concepts/architecture.md`](../../docs/concepts/architecture.md).

Multimodal parsing for EverAlgo — dispatches raw file inputs (image / audio / document / video / url) to `ParsedContent`. Used by `everalgo-knowledge` for file ingestion and by evermem's inline-parse step.

See the umbrella project: [EverAlgo monorepo](../../README.md) and the architecture document at [`docs/concepts/architecture.md`](../../docs/concepts/architecture.md).

## Install

```bash
pip install everalgo-parser
# Auto-pulls: everalgo-core
```

## Public surface (stubbed)

| Symbol | Module | Role |
|---|---|---|
| `aparse(raw_file)` | `everalgo.parser` | Async dispatch by `raw_file.mime` — raises `NotImplementedError` |
| `parse(raw_file)` | `everalgo.parser` | Sync bridge — raises `NotImplementedError` |
| `audio`, `document`, `image`, `url`, `video` | `everalgo.parser` | Sub-module stubs |

## Related distributions

- [`everalgo-core`](../everalgo-core/) — `ParsedContent` and `RawFile` types are defined here
- [`everalgo-knowledge`](../everalgo-knowledge/) — consumes `ParsedContent` once both are implemented
