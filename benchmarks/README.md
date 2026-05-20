# EverAlgo Benchmarks

End-to-end benchmarks for the EverAlgo algorithm library.

## Quick start

```bash
cd benchmarks
cp .env.example .env  # fill in API keys
uv sync
python -m benchmarks.cli --dataset locomo --smoke   # smoke test
python -m benchmarks.cli --dataset locomo --run-name v1  # full run
```

## Output

Results land in `results/<dataset>-<run-name>/`.
