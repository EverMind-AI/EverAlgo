# everalgo-boundary

MemCell boundary extractors — `ChatMemCellExtractor` / `WorkspaceMemCellExtractor` / `AgentMemCellExtractor`, plus shared `_tokenize` / `force_split` / boundary prompt helpers.

See the umbrella project: [EverAlgo monorepo](../../README.md) and the architecture document at [`docs/design.md`](../../docs/design.md).

## Quick start

> **Interface contract is defined; implementation is a stub.** Calls to `ChatMemCellExtractor.adetect`
> currently raise `NotImplementedError`. The contract below is what the real impl will satisfy.

```python
import asyncio

import everalgo
from everalgo.boundary import ChatMemCellExtractor
from everalgo.llm.providers.openai_compat import OpenAICompatClient
from everalgo.types import Message, MessageRole


async def main() -> None:
    everalgo.configure(llm=OpenAICompatClient(model="gpt-4o-mini"))
    msgs = [
        Message(role=MessageRole.USER, content="Let's talk about deployment.", timestamp=1700000000000),
        Message(role=MessageRole.ASSISTANT, content="Sure — what's the target environment?", timestamp=1700000001000),
        Message(role=MessageRole.USER, content="Switching topic — what's for lunch?", timestamp=1700000002000),
    ]
    # Streaming form: persist `tail` between calls.
    cells, tail = await ChatMemCellExtractor().adetect(msgs)

    # End-of-session form: tail is folded into the last cell.
    cells, tail = await ChatMemCellExtractor().adetect(msgs, is_final=True)
    assert tail == []
    for mc in cells:
        print(mc.id, len(mc.messages))


asyncio.run(main())
```

## Tokenizer

`everalgo.boundary._tokenize` exposes two utilities (module-private — used by boundary algorithms, not part of the public surface):

- `count_tokens(text: str) -> int` — counts tokens under OpenAI's `o200k_base` encoding via [`tiktoken`](https://github.com/openai/tiktoken). Matches GPT-4o / GPT-5 / o-series tokenization.
- `force_split(text: str, *, max_tokens: int) -> list[str]` — last-resort token-bounded chunking for caller-side prompt fitting. No semantic awareness; use `ChatMemCellExtractor` for semantic boundaries.

## Stubs

`WorkspaceMemCellExtractor` (Jira / Email / Confluence) and `AgentMemCellExtractor` (agent execution trace) are placeholder stubs in v0.x — `NotImplementedError`. Implementations land in a future minor bump when their data contracts (`RawData`) are finalised.
