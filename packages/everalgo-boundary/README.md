# everalgo-boundary

Boundary detection for EverAlgo — cuts a conversation into coherent `MemCell` slices with an LLM, either in one batch call or one message at a time. Agent trajectories with tool calls and tool results go through the same class: only the chat messages reach the model, and the tool items stay in the cells.

See the umbrella project: [EverAlgo monorepo](../../README.md) and the architecture document at [`docs/concepts/architecture.md`](../../docs/concepts/architecture.md).

## Install

```bash
pip install everalgo-boundary
```

[`everalgo-user-memory`](../everalgo-user-memory/) and [`everalgo-agent-memory`](../everalgo-agent-memory/) both depend on this package and re-export `BoundaryDetector` unchanged.

## What this distribution provides

| Symbol | Role |
|---|---|
| `BoundaryDetector` | Class binding an LLM client: `adetect` / `detect` (batch, over `ConversationItem`), `adetect_step` / `detect_step` (incremental, over `ChatMessage`) |
| `adetect_boundaries` | Async function behind `BoundaryDetector.adetect`: `(Sequence[ConversationItem], *, llm, is_final, ...) → DetectionResult` |
| `detect_boundaries` | Sync bridge over `adetect_boundaries` (`asgiref.async_to_sync`); never call it inside a running event loop |
| `adetect_boundary_step` | Async function behind `BoundaryDetector.adetect_step`: single-step boundary decision → `BoundaryDecision` |
| `DetectionResult` | `NamedTuple(cells: list[MemCell], tail: list[ConversationItem], tail_complete: bool \| None)` |

### Tool items

`adetect_boundaries` renders only the `ChatMessage` items for the model and carries its cuts back to the full list:
tool items between two chat messages go with the earlier one, tool items ahead of the first chat message with the
first cell, and trailing tool items into the tail (into the last cell when `is_final=True`). Every item lands exactly
once. Without any `ChatMessage` no LLM is called and every item goes to the tail.

### Hard limits

Both paths cut without the model once a window reaches a hard limit: 65536 tokens or 500 messages by default
(`hard_token_limit` / `hard_msg_limit` on the batch path, `hard_token_limit` / `hard_message_limit` on
`adetect_step`).

### Prompts

Each model call sends two messages. The system message holds the rules —
`BATCH_BOUNDARY_DETECT_SYSTEM_PROMPT` or `STEP_BOUNDARY_DETECT_SYSTEM_PROMPT` in
`everalgo.boundary.prompts.en.chat` — and is the same text on every call, so a prompt cache can share all of it.
The user message holds only the data, rendered from `BATCH_BOUNDARY_DETECT_USER_PROMPT` (`<messages count="N">`) or
`STEP_BOUNDARY_DETECT_USER_PROMPT` (`<conversation_history>`, `<time_gap>`, `<new_messages>`). Each message line is
`[N] [YYYY-MM-DD HH:MM UTC (Weekday)] speaker: text` on the batch path and the same without `[N]` on the step path.

`custom_instructions=` on `adetect_boundaries`, `adetect_boundary_step` and the `BoundaryDetector` methods adds the
host's text to the bundled rules, never replacing them: it is placed verbatim, never rendered as a template, in a
`<custom_instructions>` block that is always the system message's last, and `None` or a blank string leaves it out.
The user message is unchanged and always rendered by the operator. There is no `prompt=`: the bundled rules cannot be
replaced per call.

## Quick start

```python
import asyncio
import json

from everalgo.boundary import adetect_boundaries
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import ChatMessage

_BOUNDARY_JSON = json.dumps(
    {"reasoning": "single topic", "boundaries": [], "tail_complete": True}
)

async def main() -> None:
    # One scripted reply per LLM call; `llm.achat` returns the reply text (see `everalgo.llm.LLMClient`).
    fake = FakeLLMClient(responses=[_BOUNDARY_JSON, _BOUNDARY_JSON])
    messages = [
        ChatMessage(id="m1", role="user",   content="Let's talk about deployment.",     timestamp=1_700_000_000_000, sender_id="u_alice"),
        ChatMessage(id="m2", role="assistant", content="Sure — what's the target env?",  timestamp=1_700_000_001_000, sender_id="assistant"),
        ChatMessage(id="m3", role="user",   content="K8s. Switching topic: lunch?",     timestamp=1_700_000_002_000, sender_id="u_alice"),
    ]

    # Streaming: hold `tail` between calls; pass prior tail + new messages each time.
    result = await adetect_boundaries(messages, llm=fake)
    cells, tail, tail_complete = result  # NamedTuple unpacking

    # `tail_complete` is the LLM's verdict on the tail, not a restatement of it being non-empty: True
    # means the trailing segment already forms a complete episode, so it can be closed now; False
    # means it is not complete yet (it only opens a topic, or is only media placeholders, a bare "ok",
    # a system notification, an ambiguous 30-min-to-4-hour gap), so hold it for the next batch.
    # `None` means no path judged it — see the DetectionResult docstring.

    # End-of-session: tail is forced into the last cell.
    result = await adetect_boundaries(messages, llm=fake, is_final=True)
    assert result.tail == []
    for mc in result.cells:
        print(mc.timestamp, len(mc.items))


asyncio.run(main())
```

## The streaming state machine

`adetect_boundaries` deliberately holds back trailing messages as `tail` — the LLM cannot know whether a conversation continues beyond the last seen message. The caller maintains state:

```python
tail: list[ConversationItem] = []

for batch in incoming_batches:
    result = await adetect_boundaries(tail + batch, llm=client)
    await persist(result.cells)
    tail = result.tail

# Session ends — flush everything.
final = await adetect_boundaries(tail, llm=client, is_final=True)
await persist(final.cells)
```

## Failures

Each call makes one LLM request and validates the reply against a strict pydantic model, which it also passes as
`response_format`. A reply that does not fit — not JSON, a missing or extra key, a non-integer boundary — raises
`everalgo.llm.LLMOutputError`; nothing is retried, and the client's own exceptions propagate unchanged. Retry policy
belongs to the caller.

## Tokenizer utilities

Boundary detection counts tokens with `everalgo.utils.count_tokens` (in `everalgo-core`), which uses OpenAI's `o200k_base` encoding via [`tiktoken`](https://github.com/openai/tiktoken).

## Stubs

`WorkspaceMemCellExtractor` (Jira / Email / Confluence) is a placeholder in `v0.x` — all methods raise `NotImplementedError`. It is deliberately excluded from `everalgo.boundary.__all__`; import it from `everalgo.boundary.workspace` directly if you need the reserved name. Implementation lands in a future minor bump when the `RawData` contract is finalised.

## Related distributions

- [`everalgo-user-memory`](../everalgo-user-memory/) — re-exports `BoundaryDetector` beside the user-memory extractors
- [`everalgo-agent-memory`](../everalgo-agent-memory/) — re-exports `BoundaryDetector` beside the agent-memory extractors
