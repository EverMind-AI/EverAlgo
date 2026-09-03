# EverAlgo Benchmarks

End-to-end benchmarks for the EverAlgo algorithm library.

## Quick start

Run every command from the **repository root**. The CLI resolves `benchmarks/.env`
and the default dataset path relative to the current working directory, so doing
`cd benchmarks` first would break both.

```bash
# 1. Install the workspace (benchmarks is a uv workspace member).
uv sync --all-packages --group dev

# 2. Download the LoCoMo dataset (-> benchmarks/datasets/locomo/data/locomo10.json).
bash benchmarks/scripts/download_locomo.sh

# 3. Provide API keys: OpenRouter (LLM) + DeepInfra (embedding / reranker).
cp benchmarks/.env.example benchmarks/.env   # then fill in the keys

# 4. Smoke test (1 conversation, a few minutes), then a full run.
uv run python -m benchmarks.cli --dataset locomo --smoke --run-name smoke
uv run python -m benchmarks.cli --dataset locomo --run-name v1
```

## Stages

The pipeline has seven stages (1=extract\_base, 2=reflect, 3=enrich, 4=index,
5=search, 6=answer, 7=evaluate). Each writes its output to disk, so stages can
be re-run independently -- a resumed stage reads the previous stage's on-disk
artifact:

```bash
# Re-run search -> answer -> evaluate, reusing stages 1-4 output.
uv run python -m benchmarks.cli --dataset locomo --run-name v1 --stages 5 6 7
```

### Stage overview

| Stage | Output directory | What it does |
|-------|------------------|--------------|
| 1 Extract Base | `stage1_extract_base/` | Boundary detection, MemCell segmentation, Episode extraction, Episode embedding, Clustering |
| 2 Reflect | `stage2_reflect/` | Merge episodes within 2+ member clusters (optional, off by default — set `enable_reflection=true`) |
| 3 Enrich | `stage3_enrich/` | Extract atomic facts + embeddings from final episodes |
| 4 Index | `stage4_index/` | Build BM25 + embedding + cluster indices |
| 5 Search | `stage5_search/` | Agentic multi-round retrieval |
| 6 Answer | `stage6_answer/` | Generate answers from retrieved episodes |
| 7 Evaluate | `stage7_evaluate/` | LLM-as-judge scoring |

See [`docs/pipeline.md`](docs/pipeline.md) for the full data flow.

## Config

Defaults live in `benchmarks/config.toml` (single source of truth). Override any
field via a named TOML file under `benchmarks/`:

```bash
# Load benchmarks/fast.toml (unset fields fall back to BenchmarkConfig defaults).
uv run python -m benchmarks.cli --dataset locomo --config fast --run-name fast-run
```

### Running specific conversations

```bash
# Run only conversation 0 (0-based index).
uv run python -m benchmarks.cli --dataset locomo --conv 0 --run-name conv0

# Run conversations 0, 3, and 5.
uv run python -m benchmarks.cli --dataset locomo --conv 0 3 5 --run-name subset
```

## Output

Results land in `benchmarks/results/<dataset>-<run-name>/`: `report.txt`,
`report.json`, `profile.json`, plus per-stage subdirectories and `run.log`.
