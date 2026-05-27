# Security Policy

## Supported versions

EverAlgo is published as eight independently-versioned distributions under the shared `everalgo.*`
namespace. Security fixes are applied to the latest released version of each distribution.

| Distribution | Supported |
|---|---|
| `everalgo-*` (latest `0.1.x` release of each) | ✅ |
| Older pre-releases / yanked versions | ❌ |

## Reporting a vulnerability

**Do not open a public issue for security vulnerabilities.**

Report suspected vulnerabilities privately by email to **Evermind@shanda.com** with:

- the affected distribution(s) and version(s),
- a description of the issue and its impact,
- a minimal reproduction or proof of concept if available.

We aim to acknowledge a report within **3 business days** and to provide a remediation timeline
after triage. Please give us a reasonable window to release a fix before any public disclosure;
we will credit reporters who wish to be acknowledged.

## Scope

EverAlgo is a **stateless algorithm library** — it does not connect to databases, read or write
the filesystem, or manage credentials. The caller injects all I/O (LLM clients, storage) through
the `LLMClient` / `RetrieveFn` / `RerankFn` interfaces. Vulnerabilities in caller-supplied clients
or in the orchestration layer (e.g. `evermem`) are out of scope for this repository.
