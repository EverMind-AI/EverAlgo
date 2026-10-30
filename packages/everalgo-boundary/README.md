# everalgo-boundary

Boundary detection for EverAlgo — cuts a conversation into coherent `MemCell` slices with an LLM, either in one batch call or one message at a time. Agent trajectories with tool calls and tool results go through the same class: only the chat messages reach the model, and the tool items stay in the cells.

See the umbrella project: [EverAlgo monorepo](../../README.md) and the architecture document at [`docs/concepts/architecture.md`](../../docs/concepts/architecture.md).

## Install

```bash
pip install everalgo-boundary==0.4.2
```

[`everalgo-user-memory`](../everalgo-user-memory/) and [`everalgo-agent-memory`](../everalgo-agent-memory/) both depend on this package and re-export `BoundaryDetector` unchanged.

## What this distribution provides

| Symbol | Role |
|---|---|
| `BoundaryDetector` | Class binding an LLM client: `adetect` / `detect` (batch, over `ConversationItem`), `adetect_step` / `detect_step` (incremental, over `ChatMessage`) |
| `adetect_boundaries` | Async function behind `BoundaryDetector.adetect`: `(Sequence[ConversationItem], *, llm, is_final, ...) → DetectionResult` |
| `detect_boundaries` | Sync bridge over `adetect_boundaries` (`asgiref.async_to_sync`); never call it inside a running event loop |
| `adetect_boundary_step` | Async function behind `BoundaryDetector.adetect_step`: single-step boundary decision → `BoundaryDecision` |
| `DetectionResult` | `NamedTuple(cells: list[MemCell], tail: list[ConversationItem])` |

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
Batch records escape XML delimiters in names and content, indent continuation lines, and retain empty content
as `[N] [empty message]`. Material inside these records is data, not instructions to the detector.

The batch input is an open streaming window of one or more messages. An episode is one coherent event,
activity or independent discussion, including its execution, outcome and related feedback. Different steps,
revisions, acknowledgements or dates alone do not establish a new episode. With insufficient evidence of a
new independent episode, the prompt asks the model to keep the messages together.

Batch labels start at **0**. Each boundary `b` names the first message of a new episode, so the preceding
slice is `messages[:b]`. For example, `[2]` groups messages 0 and 1, then starts the next episode at 2.
The valid range is `1 <= b < N`; the final input message can start a new episode. Tool items are not numbered.
The strict batch response contains only `reasoning` and `boundaries`; reasoning is requested in one sentence of
at most 25 words explaining all boundary decisions. It does not predict whether the trailing episode has finished.

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
    {"reasoning": "All messages concern the same deployment decision.", "boundaries": []}
)

async def main() -> None:
    # One scripted reply per LLM call; `llm.achat` returns the reply text (see `everalgo.llm.LLMClient`).
    fake = FakeLLMClient(responses=[_BOUNDARY_JSON, _BOUNDARY_JSON])
    messages = [
        ChatMessage(id="m1", role="user",   content="Let's talk about deployment.",     timestamp=1_700_000_000_000, sender_id="u_alice"),
        ChatMessage(id="m2", role="assistant", content="Sure — what's the target env?",  timestamp=1_700_000_001_000, sender_id="assistant"),
        ChatMessage(id="m3", role="user",   content="K8s. Which deployment strategy should I use?", timestamp=1_700_000_002_000, sender_id="u_alice"),
    ]

    # Streaming: hold `tail` between calls; pass prior tail + new messages each time.
    result = await adetect_boundaries(messages, llm=fake)
    cells, tail = result  # NamedTuple unpacking

    # Persist the closed cells and carry the tail into the next batch.

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

## Migrating from 0.4.1

**Breaking:** `DetectionResult` now has two fields on both batch and step paths. Replace
`cells, tail, tail_complete = result` with `cells, tail = result`; remove `result.tail_complete` accesses and
completion-based branches. Persist `result.cells`, retain `result.tail`, and use the host's idle or session-end
flush policy. The library does not schedule automatic flushing. The step prompt and decision schema are unchanged.

The model's strict batch schema also drops `tail_complete`. Update scripted replies and custom LLM adapters to
return only `reasoning` and `boundaries`; the removed field is rejected as an extra key. Custom batch instructions
must use zero-based new-episode start labels. The numeric cut position itself remains a Python slice boundary.

This release replaces completion prediction because an open window does not establish whether more related
messages will arrive. It groups related execution, outcomes and feedback into an episode and requests a
one-sentence explanation of all boundary decisions in at most 25 words. The word limit is a prompt instruction,
not a schema validator or a truncation step. Effects, response length and speed remain model-dependent.

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
