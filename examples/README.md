# Examples

Runnable quickstart scripts for each major EverAlgo operator. Every example uses
`everalgo.testing.fake_llm.FakeLLMClient` so it runs offline with no API key.

```bash
uv run python examples/01_boundary_chat.py
```

| # | File | What it shows |
|---|---|---|
| 01 | [`01_boundary_chat.py`](01_boundary_chat.py) | Chat → MemCell via `BoundaryDetector` |
| 02 | [`02_clustering_geometry.py`](02_clustering_geometry.py) | Online incremental clustering with `cluster_by_geometry` |
| 03 | [`03_user_memory_episode.py`](03_user_memory_episode.py) | MemCell → `Episode` via `EpisodeExtractor` |
| 04 | [`04_agent_memory_case.py`](04_agent_memory_case.py) | Agent trajectory → `AgentCase` via `AgentCaseExtractor` |
| 05 | [`05_rank_rerank.py`](05_rank_rerank.py) | LLM-based reranking with `arerank` |
| 06 | [`06_full_user_memory_pipeline.py`](06_full_user_memory_pipeline.py) | Full pipeline: chat → boundary → 4 user-memory extractors |

For deeper architecture see [`docs/concepts/architecture.md`](../docs/concepts/architecture.md).
