# LoCoMo Benchmark Reproduction Guide

EverAlgo on the [LoCoMo](https://github.com/silin159/locomo) long-conversation benchmark: **93.51%** majority-vote accuracy (1440/1540), on the current **7-stage** pipeline. This supersedes the historical 5-stage `locomo-93.05` run as the current canonical reference.

## Quick Start — Full Run from Scratch

```bash
# Prerequisites: Python 3.12+, uv, API keys in benchmarks/.env
#   OPENROUTER_API_KEY=...
#   DEEPINFRA_API_KEY=...
uv sync --all-packages --group dev
uv run python -m benchmarks --dataset locomo --run-name my-run
```

This runs all **7 stages** end-to-end (extract_base → reflect → enrich → index → search → answer → evaluate; ~75 min, ~$9 at the token counts in `profile.json`). Results land in `benchmarks/results/locomo-my-run/`.

## Scoring

Stage 7 runs 3 independent LLM judge passes per question (`gpt-4o-mini`, temperature=0). Two accuracy metrics are reported (full breakdown in `report.txt`):

- **Majority-vote** (headline): a question is correct if ≥2 of 3 judges agree. **93.51%** (1440/1540).
- **Mean-of-runs** (conservative): average accuracy across the 3 judge runs. **93.33%** (judge std 0.0012).

Per-category (majority-vote): C1 single-hop 92.20% (260/282) · C2 temporal 91.59% (294/321) · C3 open-domain 79.17% (76/96) · C4 multi-hop 96.31% (810/841).

## Notes

- **Reflection was OFF.** `enable_reflection: False` (see `report.txt` Config). Reflection is not part of this canonical configuration.
- **Judge failures count against the total.** 103 of 1540 questions had all judge passes fail and are counted as incorrect in the 1540 denominator (the prior 93.05 run had 109 such failures — same accounting).

## Models & External Services

| Role | Model | Provider |
|------|-------|----------|
| Extract / Reflect / Enrich / Search / Answer | `gpt-4.1-mini` | OpenRouter |
| Embedding | `Qwen3-Embedding-4B` (1024 dim) | DeepInfra |
| Reranker | `Qwen3-Reranker-4B` | DeepInfra |
| Judge | `gpt-4o-mini` (temperature=0, 3 runs) | OpenRouter |

Full configuration is in `report.txt` (Config section) and `profile.json`.

## Files in This Directory

| File | Description |
|------|-------------|
| `REPRODUCTION.md` | This guide |
| `report.txt` | Human-readable results — both accuracy metrics, per-category breakdown, full config, stage timings |
| `report.json` | Machine-readable version of the above |
| `profile.json` | Per-stage token/duration profile at run time |

> Only the top-level summary (`report.txt` / `report.json` / `profile.json`) and this note are committed. Stage 1–6 intermediate outputs (~626 MB) and `eval_results.json` stay gitignored — regenerate by re-running. The historical 5-stage run remains at `locomo-93.05` for reference.
