# everalgo-user-memory

User-side memory products — `EpisodeExtractor`, `ForesightExtractor`, `AtomicFactExtractor`, `ProfileExtractor`. Re-exports `ChatMemCellExtractor` / `WorkspaceMemCellExtractor` from `everalgo-boundary` for facade convenience.

See the umbrella project: [EverAlgo monorepo](../../README.md) and the architecture document at [`docs/design.md`](../../docs/design.md).

## Quick start

All four extractors are stateless callable classes following the same shape: `aextract(memcell, *, llm=None, prompt=None)` returning a typed memory list (or a single `Profile` for `ProfileExtractor`). The `extract = async_to_sync(aextract)` sync bridge is available for non-event-loop callers.

```python
import asyncio

import everalgo
from everalgo.boundary import ChatMemCellExtractor
from everalgo.llm.providers.openai_compat import OpenAICompatClient
from everalgo.types import Message, MessageRole
from everalgo.user_memory import (
    AtomicFactExtractor,
    EpisodeExtractor,
    ForesightExtractor,
    ProfileExtractor,
)


async def main() -> None:
    everalgo.configure(llm=OpenAICompatClient(model="gpt-4o-mini"))

    msgs = [
        Message(role=MessageRole.USER, content="Schedule a 3pm sync with Alice; I'll follow up Friday.", timestamp=1700000000000),
        Message(role=MessageRole.ASSISTANT, content="Done. Invite sent.", timestamp=1700000001000),
    ]
    cells, _tail = await ChatMemCellExtractor().adetect(msgs, is_final=True)
    mc = cells[0]

    episodes = await EpisodeExtractor().aextract(mc)         # list[Episode]
    foresights = await ForesightExtractor().aextract(mc)     # list[Foresight]
    facts = await AtomicFactExtractor().aextract(mc)         # list[AtomicFact]

    # Profile takes the current cell plus a caller-fetched prior cluster
    profile = await ProfileExtractor().aextract(
        mc, cluster_episodes=[],                             # caller passes prior MemCells here
    )                                                        # -> Profile (single)


asyncio.run(main())
```

### Customizing prompts

Per call:

```python
from everalgo.user_memory.prompts.zh.episode import EPISODE_EXTRACT_PROMPT_ZH

episodes = await EpisodeExtractor().aextract(mc, prompt=EPISODE_EXTRACT_PROMPT_ZH)
```

Globally (one-time monkey-patch at startup):

```python
import everalgo.user_memory.prompts.en.foresight as _fs

_fs.FORESIGHT_EXTRACT_PROMPT_EN = my_custom_prompt
```

### Testing

The `everalgo.testing` subpackage ships `FakeLLMClient` (handler / scripted modes) and `assert_X_shape`
helpers for each memory type. See the in-tree pattern at
[`tests/integration/test_boundary_to_episode_e2e.py`](../../tests/integration/test_boundary_to_episode_e2e.py).
