# everalgo-boundary

Chat boundary detection for EverAlgo — segments a flat list of `ChatMessage` objects into coherent `MemCell` slices using an LLM-based batch algorithm.

See the umbrella project: [EverAlgo monorepo](../../README.md) and the architecture document at [`docs/concepts/architecture.md`](../../docs/concepts/architecture.md).

## Install

```bash
pip install everalgo-boundary
```

For the user-scenario class facade, install [`everalgo-user-memory`](../everalgo-user-memory/) instead — it re-exports `BoundaryDetector` which wraps this package.

## What this distribution provides

| Symbol | Role |
|---|---|
| `detect_boundaries` | Low-level async function: `(list[ChatMessage], *, llm, is_final, ...) → DetectionResult` |
| `DetectionResult` | `NamedTuple(cells: list[MemCell], tail: list[ChatMessage])` |
| `WorkspaceMemCellExtractor` | Placeholder stub for Jira / Email / Confluence (raises `NotImplementedError`) |

The class-style facades (`BoundaryDetector` for user-scenario chat, `AgentBoundaryDetector` for agent trajectories with tool calls) live in [`everalgo-user-memory`](../everalgo-user-memory/) and [`everalgo-agent-memory`](../everalgo-agent-memory/) respectively.

## Quick start

```python
import asyncio
import json

from everalgo.boundary import detect_boundaries
from everalgo.llm.types import ChatMessage as LLMChatMessage, ChatResponse
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import ChatMessage

_BOUNDARY_JSON = json.dumps(
    {"reasoning": "single topic", "boundaries": [], "should_wait": False}
)

async def main() -> None:
    fake = FakeLLMClient(responses=[ChatResponse(content=_BOUNDARY_JSON, model="fake")])
    messages = [
        ChatMessage(id="m1", role="user",   content="Let's talk about deployment.",     timestamp=1_700_000_000_000, sender_id="u_alice"),
        ChatMessage(id="m2", role="assistant", content="Sure — what's the target env?",  timestamp=1_700_000_001_000, sender_id="assistant"),
        ChatMessage(id="m3", role="user",   content="K8s. Switching topic: lunch?",     timestamp=1_700_000_002_000, sender_id="u_alice"),
    ]

    # Streaming: hold `tail` between calls; pass prior tail + new messages each time.
    result = await detect_boundaries(messages, llm=fake)
    cells, tail = result  # NamedTuple unpacking

    # End-of-session: tail is forced into the last cell.
    result = await detect_boundaries(messages, llm=fake, is_final=True)
    assert result.tail == []
    for mc in result.cells:
        print(mc.timestamp, len(mc.items))


asyncio.run(main())
```

## The streaming state machine

`detect_boundaries` deliberately holds back trailing messages as `tail` — the LLM cannot know whether a conversation continues beyond the last seen message. The caller maintains state:

```python
tail: list[ChatMessage] = []

for batch in incoming_batches:
    result = await detect_boundaries(tail + batch, llm=client)
    await persist(result.cells)
    tail = result.tail

# Session ends — flush everything.
final = await detect_boundaries(tail, llm=client, is_final=True)
await persist(final.cells)
```

## Tokenizer utilities

`everalgo._tokenize` (in `everalgo-core`) exposes two module-private utilities used by boundary algorithms; not part of the public surface:

- `count_tokens(text: str) → int` — token count under OpenAI `o200k_base` encoding via [`tiktoken`](https://github.com/openai/tiktoken).
- `force_split(text: str, *, max_tokens: int) → list[str]` — last-resort token-bounded chunking; no semantic awareness.

## Stubs

`WorkspaceMemCellExtractor` (Jira / Email / Confluence) is a placeholder in `v0.x` — all methods raise `NotImplementedError`. Implementation lands in a future minor bump when the `RawData` contract is finalised.

## Related distributions

- [`everalgo-user-memory`](../everalgo-user-memory/) — `BoundaryDetector` class facade for chat scenarios
- [`everalgo-agent-memory`](../everalgo-agent-memory/) — `AgentBoundaryDetector` class facade for agent trajectories
