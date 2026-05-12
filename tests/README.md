# tests/

**Cross-distribution end-to-end tests only.** Intra-distribution unit tests
live next to each package under `packages/everalgo-<name>/tests/` (mirrors
the pydantic-ai layout).

This directory exists to cover the one class of bug that per-package tests
structurally cannot catch: **contract drift between distributions**. A
`boundary` unit test asserts its own output shape; a `user-memory` unit test
asserts it consumes a well-formed input. Each passes in isolation while the
two contracts silently diverge — that's the gap an E2E here closes.

Expect ~1 e2e file per cross-distribution data-flow path (today: 1 covering
`messages → MemCell → Episode`; future candidates include `parser → knowledge`
and `boundary → clustering → rank`). Not a catch-all bucket; each file must
exercise a real cross-package contract and stay under ~100 lines.
