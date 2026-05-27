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

The pipeline has five stages (1=extract, 2=index, 3=search, 4=answer, 5=evaluate).
Each writes its output to disk, so stages can be re-run independently — a resumed
stage reads the previous stage's on-disk artifact:

```bash
# Re-run search -> answer -> evaluate, reusing stage 1 + 2 output.
uv run python -m benchmarks.cli --dataset locomo --run-name v1 --stages 3 4 5
```

See [`docs/pipeline.md`](docs/pipeline.md) for the full data flow.

## Output

Results land in `benchmarks/results/<dataset>-<run-name>/`: `report.txt`,
`report.json`, `profile.json`, plus per-stage subdirectories and `run.log`.
