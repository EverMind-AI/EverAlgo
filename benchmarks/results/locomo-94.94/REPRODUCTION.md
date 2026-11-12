# LoCoMo Benchmark Reproduction Guide

EverAlgo on the [LoCoMo](https://github.com/silin159/locomo) long-conversation benchmark: **94.94%** majority-vote accuracy (1462/1540) in a single run of the 7-stage pipeline on everalgo `main` at `c073a97`. That code is what was released as everalgo-core 0.7.0rc1, everalgo-boundary 0.5.0rc1, everalgo-user-memory 0.9.0rc5 and everalgo-rank 0.5.0rc5 (see Notes). This supersedes `locomo-93.51` as the current reference.

## Quick Start — Full Run from Scratch

```bash
# Prerequisites: Python 3.12+, uv, API keys in benchmarks/.env
#   OPENROUTER_API_KEY=...
#   DEEPINFRA_API_KEY=...
git checkout c073a97
uv sync --all-packages --group dev
bash benchmarks/scripts/download_locomo.sh
uv run python -m benchmarks.cli --dataset locomo --run-name my-run
```

The run uses `benchmarks/config.toml` unchanged. Results land in `benchmarks/results/locomo-my-run/`. Run one benchmark at a time (see Notes).

## Configuration

| Stage | Setting |
|---|---|
| Boundary detection | Step mode: one `BoundaryDetector.adetect_step` call per message (`extract_smart_mask = true`), with everalgo-boundary's step hard limits of 8,192 tokens / 50 messages |
| Reply format | Strict JSON schema on every LLM call (`response_format_mode = "json_schema"`, the client default) |
| Extraction and retrieval LLM | `openai/gpt-4.1-mini` via OpenRouter, temperature 0.3, `max_tokens` 16,384, 60 s timeout |
| Episode | everalgo-user-memory's episode prompt; no output language named, so the output follows the conversation's language |
| Reflection | Off (`enable_reflection = false`) |
| Atomic facts | Extracted from each final episode |
| Index | BM25 over atomic facts + episode subject + the first 200 characters of the episode body (NLTK tokenization); embeddings `Qwen/Qwen3-Embedding-4B` at 1024 dimensions over atomic facts + subject; clusters at cosine 0.70 within a 7-day gap |
| Retrieval | Agentic: dense top 50 + BM25 top 50 → RRF (k = 40) → cluster expansion (top 10 clusters) → rerank with `Qwen/Qwen3-Reranker-4B` → top 10 → LLM sufficiency check → if insufficient, 3 sub-queries and a second round → 10 episodes to the answer stage |
| Answer | `openai/gpt-4.1-mini`, temperature 0.0 |
| Judge | `openai/gpt-4o-mini`, temperature 0.0, 3 independent runs, majority vote |

The full configuration dump is in `report.txt` (Config section) and `report.json`.

## Scoring

Stage 7 runs 3 independent judge passes per question. Every one of the 1540 questions received all three verdicts.

- **Majority-vote** (headline): a question is correct if at least 2 of 3 judges agree. **94.94%** (1462/1540).
- **Mean-of-runs**: average accuracy across the 3 judge runs. **94.78%** (judge std 0.0003).
- **Re-judged against corrected gold answers**: also **94.94%** (1462/1540) for this run's agentic retrieval. "Re-judged" means: [dial481/locomo-audit](https://github.com/dial481/locomo-audit) disputes the gold answer of 99 of the 1540 questions (its 156 entries, less the 57 that flag only a wrong evidence citation). The answers this run already generated for those 99 were judged again against the audit's corrected answers, with the same judge prompt, model and three-run majority vote; nothing was re-retrieved or re-answered, and the other 1441 verdicts stand. 12 of the 99 turned correct and 12 turned wrong. The re-judge is not one of the seven stages; it was run afterwards on the stage 6 answers.

Per-category (majority-vote): C1 single-hop 93.26% (263/282) · C2 temporal 93.15% (299/321) · C3 open-domain 86.46% (83/96) · C4 multi-hop 97.15% (817/841).

## Notes

- **Package versions.** `report.json` lists the versions `main` declared at `c073a97` (core 0.6.0, boundary 0.4.2, user-memory 0.9.0rc4, rank 0.5.0rc4). The release commit that followed (`f7fd0ce`) changed only version numbers, changelogs and dependency floors, so the code measured is that of core 0.7.0rc1, boundary 0.5.0rc1, user-memory 0.9.0rc5 and rank 0.5.0rc5.
- **The run was resumed twice.** Concurrent runs hit DeepInfra rate limits (HTTP 429), which the benchmark's HTTP retry does not retry, so this run was executed in three invocations: stages 1–2, then 3–4, then 5–7, each starting from the previous stages' artifacts on disk with `--stages` (the supported way to re-run later stages). `report.txt` and `profile.json` are written by the last invocation and list stages 5–7 only; all seven stages are below.

| Stage | Duration | Tokens (prompt / completion) |
|---|---|---|
| 1 extract | 945.1 s | 372,414 / 120,205 (estimates from `stats_conv_*.json`) |
| 2 reflect | 0.0 s (off) | — |
| 3 enrich | 170.2 s | not recorded by the stage |
| 4 index | 2.3 s | — |
| 5 search | 1006.1 s | 4,241,690 / 269,067 |
| 6 answer | 414.1 s | 5,566,061 / 808,042 |
| 7 evaluate | 179.6 s | 2,157,135 / 179,185 |

## Models & External Services

| Role | Model | Provider |
|------|-------|----------|
| Boundary / Episode / Atomic facts / Retrieval / Answer | `gpt-4.1-mini` | OpenRouter |
| Embedding | `Qwen3-Embedding-4B` (1024 dim) | DeepInfra |
| Reranker | `Qwen3-Reranker-4B` | DeepInfra |
| Judge | `gpt-4o-mini` (temperature 0, 3 runs) | OpenRouter |

## Files in This Directory

| File | Description |
|------|-------------|
| `REPRODUCTION.md` | This guide |
| `report.txt` | Human-readable results: both accuracy metrics, per-category breakdown, full config, stage 5–7 timings |
| `report.json` | Machine-readable version of the above |
| `profile.json` | Per-stage token and duration profile of the last invocation (stages 5–7) |

> Only the top-level summary (`report.txt` / `report.json` / `profile.json`) and this note are committed. Stage outputs and `eval_results.json` stay gitignored; regenerate them by re-running. The previous 7-stage reference remains at `locomo-93.51`, and the historical 5-stage run at `locomo-93.05`.
