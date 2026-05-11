# tests/

Cross-package integration tests live here. Per-package unit tests should live
next to the package itself once it grows enough mass — i.e. a future
`packages/everalgo-<name>/tests/` directory — mirroring the pydantic-ai layout.

Until then, this directory holds workspace-wide smoke tests (e.g. import
sanity, namespace package resolution across distributions).
