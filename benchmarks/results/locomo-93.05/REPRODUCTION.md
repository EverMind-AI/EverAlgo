# LoCoMo Benchmark Reproduction Guide

EverAlgo on the [LoCoMo](https://github.com/silin159/locomo) long-conversation benchmark: **93.05%** accuracy (1433/1540).

## Quick Start — Full Run from Scratch

```bash
# Prerequisites: Python 3.12+, uv, API keys in benchmarks/.env
#   OPENROUTER_API_KEY=...
#   DEEPINFRA_API_KEY=...
uv sync --all-packages --group dev
uv run python -m benchmarks --dataset locomo --run-name my-run
```

This runs all 5 stages end-to-end (~55 min, ~$7). Results land in `benchmarks/results/locomo-my-run/`.

> Stage 1 (extract) uses LLM calls with non-zero temperature, so each run produces slightly different
> MemCells. Expect ±1pp accuracy variation across runs.

## Partial Re-run from Cached Artifacts

Stages 1-2 (extract + index) are the most expensive and LLM-dependent. To skip them and only re-run
retrieval + answer + evaluation, download the pre-built artifacts from the GitHub Release:

```bash
cd benchmarks/results && mkdir -p locomo-93.05 && cd locomo-93.05

# Download the archive from the release page, then extract:
tar xzf locomo-93.05-artifacts.tar.gz

cd ../../..
uv run python -m benchmarks --dataset locomo --run-name 93.05 --stages 3 4 5
```

**Release page**: [GitHub](https://github.com/EverMind-AI/EverAlgo/releases/tag/benchmarks/v93.05)

| Archive | Size | Contents |
|---------|------|----------|
| `locomo-93.05-artifacts.tar.gz` | 204 MB | All 5 stages: `stage1_extract/` + `stage2_index/` + stage 3-5 outputs + `profile.json` + `run.log` |

## Scoring

Stage 5 runs 3 independent LLM judge passes per question (`gpt-4o-mini`, temperature=0).
Two accuracy metrics are reported:

- **Majority-vote** (headline): a question is correct if ≥2 of 3 judges agree. **93.05%** (1433/1540).
- **Mean-of-runs** (conservative): average accuracy across the 3 judge runs. **92.92%**.

Per-category breakdown and both metrics are in `report.txt`.

## Models & External Services

| Role | Model | Provider |
|------|-------|----------|
| Extract / Search / Answer | `gpt-4.1-mini` | OpenRouter |
| Embedding | `Qwen3-Embedding-4B` (1024 dim) | DeepInfra |
| Reranker | `Qwen3-Reranker-4B` | DeepInfra |
| Judge | `gpt-4o-mini` (temperature=0, 3 runs) | OpenRouter |

Full configuration is in `report.txt` (Config section) and `profile.json`.

## Files in This Directory

| File | Description |
|------|-------------|
| `REPRODUCTION.md` | This guide |
| `report.txt` | Human-readable results with both accuracy metrics, per-category breakdown, config, and stage timings |
| `report.json` | Machine-readable version of the above |
| `profile.json` | Frozen config snapshot + package versions at run time |
