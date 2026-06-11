# Benchmark Variance Analysis

## Setup

- **Code**: `origin/main` at `a0d3609` (includes `524ffd0` retrieval alignment refactor)
- **Dataset**: LoCoMo full, 10 conversations, 1540 QA
- **Config**: agentic retrieval, `gpt-4.1-mini` (temp=0.3), `gpt-4o-mini` judge ×3, all other params at default
- **Runs**: 5, identical code and config

## Results

| Run | Majority-Vote | Mean-of-Runs | Cat1/282 | Cat2/321 | Cat3/96 | Cat4/841 |
|---|---|---|---|---|---|---|
| Run 1 | 92.79% | 92.75% | 253 | 290 | 73 | 813 |
| Run 2 | 92.73% | 92.75% | 256 | 290 | 70 | 812 |
| Run 3 | 92.40% | 92.38% | 253 | 288 | 72 | 810 |
| Run 4 | 92.47% | 92.42% | 252 | 290 | 74 | 808 |
| Run 5 | **93.12%** | **93.05%** | 258 | 292 | 77 | 807 |

## Summary

| Metric | Value |
|---|---|
| Mean (majority / mean-of-runs) | 92.70% / 92.67% |
| Max | **93.12%** (majority), **93.05%** (mean-of-runs) |
| Min | 92.40% |
| Std | 0.29pp |
| Range | [92.40%, 93.12%] (0.71pp) |

## Key Findings

1. **93.05% is reachable but not central**: the median run lands at 92.7%. Hitting 93.05% requires ~5 runs at >50% probability.
2. **Run-to-run variance is 0.29pp std** (0.71pp range), driven by OpenRouter routing non-determinism (89% of total variance). Judge contributes only 11%.
3. **Cat4 multi-hop is the anchor**: 96.31% ±0.30pp. Cat3 open-domain is the most volatile: 76.25% ±2.70pp (7-question swing on 96 total).
4. **Agentic multi-query supplementary retrieval ratio stable**: 30.8% ±0.6pp — about 31% of questions trigger a second-round retrieval after the sufficiency check judges the first round insufficient.
5. **Total token cost per full run**: ~13.3M tokens (prompt + completion, across all 5 stages).
