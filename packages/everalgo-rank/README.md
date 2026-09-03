# everalgo-rank

Memory retrieval and ranking for EverAlgo. The package has two public layers: composable retrieval strategies over caller-injected `RetrieveFn` / `RerankFn` callables, and four business rankers over pre-fetched `RankInput` data. EverAlgo owns the algorithms but never the caller's storage or model clients.

See the umbrella project: [EverAlgo monorepo](../../README.md) and the architecture document at [`docs/concepts/architecture.md`](../../docs/concepts/architecture.md).

## Install

```bash
pip install everalgo-rank
```

## What this distribution provides

| Symbol | Role |
|---|---|
| `ahybrid_retrieve` / `hybrid_retrieve` | Parallel dense + sparse recall followed by RRF or LR fusion |
| `aagentic_retrieve` / `agentic_retrieve` | LLM sufficiency check with optional multi-query or refined-query Round 2 over any base retriever |
| `acluster_retrieve` / `cluster_retrieve` | Select top clusters from base recall and expand their full membership |
| `amaxsim_retrieve` / `maxsim_retrieve` | Child recall → parent max-pool → parent hydration |
| `acategory_retrieve` / `category_retrieve` | Base recall → category-mass rollup → caller rerank → category boost |
| `RetrieveFn` / `RerankFn` | Async callable contracts implemented by the caller; storage/index/model clients stay inside the callbacks |
| `EpisodicRanker` | Class facade — episodic memory ranking; LLM bound at construction |
| `CaseRanker` | Class facade — agent case ranking; LLM bound at construction |
| `SkillRanker` | Class facade — agent skill ranking; LLM bound at construction |
| `profile.rank` | Module-level sync function — profile ranking; cosine sort + dedup, no LLM |
| `RankConfig` | Frozen pydantic config: `fusion_mode`, `rrf_k`, `alpha`, etc. |
| `FusionMode` | Literal type: `"rrf"` / `"lr"` / `"vector_anchored"` |
| `DEFAULT_RANK_CONFIG` | `RankConfig()` with defaults (`fusion_mode="rrf"`, `rrf_k=60`) |
| `arank` / `rank` | Top-level async / sync dispatch — routes to the registered ranker by `RankInput.memory_type` |
| `arerank` / `rerank` | Low-level LLM rerank step (not top-level: import from `everalgo.rank.rerank`) |

## Retrieval composition quick start

```python
from everalgo.rank import ahybrid_retrieve

hits = await ahybrid_retrieve(
    "Python async retry patterns",
    dense_retrieve=dense_retrieve,
    sparse_retrieve=sparse_retrieve,
    top_n=20,
)
```

`dense_retrieve` and `sparse_retrieve` are caller-owned async functions with the shape `(query: str, top_k: int) -> list[Candidate]`. They can bind any vector store, keyword index, or test fixture; `ahybrid_retrieve` only runs and fuses them.

## Business-ranker quick start

```python
import asyncio
import json

from everalgo.llm.types import ChatResponse
from everalgo.rank import EpisodicRanker, DEFAULT_RANK_CONFIG
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import Candidate, RankInput

_RERANK_JSON = json.dumps({"ranked": [{"id": "ep_a", "score": 0.95}]})

async def main() -> None:
    fake = FakeLLMClient(responses=[ChatResponse(content=_RERANK_JSON, model="fake")])
    ranker = EpisodicRanker(llm=fake)

    rank_input = RankInput(
        query="Python async retry patterns",
        memory_type="episodic",
        dense_candidates=[
            Candidate(id="ep_a", score=0.9, metadata={"episode": "Alice asked about async retries."}),
            Candidate(id="ep_b", score=0.5, metadata={"episode": "Team discussed lunch."}),
        ],
        top_k=1,
    )

    output = await ranker.arank(rank_input, config=DEFAULT_RANK_CONFIG, enable_rerank=True)
    for item in output.items:
        print(f"{item.id}  score={item.score:.2f}")

asyncio.run(main())
```

## LLM rerank step

`arerank` is the standalone LLM reranking primitive used internally by the ranker facades and exposed for direct use:

```python
from everalgo.rank.rerank import arerank
from everalgo.rank.prompts.en.episodic import EPISODIC_RERANK_PROMPT_EN
from everalgo.types import Candidate

reranked = await arerank(
    candidates,
    prompt=EPISODIC_RERANK_PROMPT_EN,
    top_k=5,
    llm=client,
)
```

## API surface

| Symbol | Module | Signature summary |
|---|---|---|
| `ahybrid_retrieve` | `everalgo.rank` | `(query, *, dense_retrieve, sparse_retrieve, ...) → list[Candidate]` |
| `aagentic_retrieve` | `everalgo.rank` | `(query, *, base_retrieve, llm, rerank_fn=None, ...) → tuple[list[Candidate], AgenticDecision]` |
| `acluster_retrieve` | `everalgo.rank` | `(query, *, base_retrieve, clusters, all_docs, ...) → list[Candidate]` |
| `amaxsim_retrieve` | `everalgo.rank` | `(query, *, child_retrieve, parent_fetch, ...) → list[Candidate]` |
| `acategory_retrieve` | `everalgo.rank` | `(query, *, base_retrieve, rerank_fn, ...) → list[Candidate]` |
| `EpisodicRanker` | `everalgo.rank` | `__init__(*, llm)` → `arank(rank_input, *, config, prompt, enable_rerank, ...) → RankOutput` |
| `CaseRanker` | `everalgo.rank` | Same shape as `EpisodicRanker` |
| `SkillRanker` | `everalgo.rank` | Same shape as `EpisodicRanker` |
| `profile.rank` | `everalgo.rank.profile` | `(rank_input, *, threshold=0.0) → RankOutput` — sync, no LLM |
| `RankConfig` | `everalgo.rank` | `fusion_mode: FusionMode = "rrf"`, `rrf_k: int = 60`, `alpha: float = 1.0`, `expand_limit: int = 3` |
| `arank` | `everalgo.rank` | Top-level async dispatch by `rank_input.memory_type` |
| `arerank` | `everalgo.rank.rerank` | `(items, *, prompt, top_k, llm) → list[Candidate]` |

## Cross-links

- [`everalgo-core`](../everalgo-core/) — `RankInput`, `RankOutput`, `Candidate`, `ScoredItem`, `LLMClient`
- [`everalgo-user-memory`](../everalgo-user-memory/) — produces `Episode` / `AtomicFact` / `Profile` that the caller's recall layer packages into `Candidate` for ranking
- [`everalgo-agent-memory`](../everalgo-agent-memory/) — produces `AgentCase` / `AgentSkill` candidates
