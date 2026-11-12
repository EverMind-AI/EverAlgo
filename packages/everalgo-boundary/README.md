# everalgo-boundary

Boundary detection for EverAlgo — decides where to cut a conversation into coherent `MemCell` slices with an LLM, either in one batch call or one message at a time. Both paths return indices into the caller's own input, never the content; the caller slices it, or builds the cells with `to_memcells`. Agent trajectories with tool calls and tool results go through the same class: only the chat messages reach the model, and the indices count the tool items too, so they stay in the cells.

See the umbrella project: [EverAlgo monorepo](../../README.md) and the architecture document at [`docs/concepts/architecture.md`](../../docs/concepts/architecture.md).

## Install

```bash
pip install everalgo-boundary==0.5.0
```

[`everalgo-user-memory`](../everalgo-user-memory/) and [`everalgo-agent-memory`](../everalgo-agent-memory/) both depend on this package and re-export `BoundaryDetector`, `Cuts` and `to_memcells` unchanged.

## What this distribution provides

| Symbol | Role |
|---|---|
| `BoundaryDetector` | The only entry, binding an LLM client: `adetect_batch` / `detect_batch` (one window of `ConversationItem`) and `adetect_step` / `detect_step` (the judged history plus one new `ChatMessage`); both return `Cuts`. The `detect_*` names are `asgiref.async_to_sync` bridges; never call them inside a running event loop |
| `Cuts` | `NamedTuple(ends: list[int], tail_start: int)`: indices into the input the method was given |
| `to_memcells` | `(items, cuts) → list[MemCell]`: one cell per closed segment, stamped with its last chat message's time |

### Cut indices

A cut is the position of the first item of a new segment, which is also where the segment before it ends. Segments
are half-open like Python slices: segment `i` is `items[ends[i-1]:ends[i]]` (the first starts at 0), and the part
to carry into the next call is `items[tail_start:]`. Indices count every item of the input, tool items included.
`ends` is strictly increasing with `0 < ends[0]` and `ends[-1] <= len(items)`; `0 <= tail_start <= len(items)`.

- **Batch**: the model's boundary `b` is already a cut (the first message of a new episode), carried from the chat
  messages to the full item list. Segments never overlap: `tail_start` is the last cut, `0` without one, and
  `is_final=True` adds the cut `len(items)` so the tail is empty. For six messages and a reply of `[2, 4]`:
  `Cuts([2, 4], 4)`, or `Cuts([2, 4, 6], 6)` when final.
- **Step**: `items` is the history already judged plus one new message; with `n` history messages the result is
  one of three:

| Outcome | `ends` | `tail_start` | Meaning |
|---|---|---|---|
| No cut (warmup included) | `[]` | `0` | the new message joins the history |
| Cut, no bridge (`n <= smart_mask_threshold`) | `[n]` | `n` | the history closes; the new message starts the next segment |
| Cut, bridge (`n > smart_mask_threshold`) | `[n]` | `n - 1` | the history closes; its last message also starts the next segment |

A model cut and a hard-limit cut have the same shape, so a caller never needs to know which happened, nor anything
about the bridge: it slices by the indices.

### Step strategy

The whole step strategy is `adetect_step`'s, each part a keyword whose default is the recommended value: no model
call while the history is shorter than `warmup_messages` (2); above `smart_mask_threshold` (5) history messages
with `smart_mask` on, the model is not shown the history's last message, and a cut carries that message into the
next segment (the bridge); hard limits as below. Tool items are refused with `InvalidInputError`, as is empty input.

### Tool items

`adetect_batch` renders only the `ChatMessage` items for the model and carries its cuts back to the full list:
tool items between two chat messages go with the earlier one, tool items ahead of the first chat message with the
first segment, and trailing tool items into the tail (into the last segment when `is_final=True`). Every item lands
exactly once. Without any `ChatMessage` no LLM is called and the result is `Cuts([], 0)`.

### Hard limits

Both paths cut without the model once a hard limit is reached (`hard_token_limit` / `hard_message_limit` on both),
but the limits bound different things, so their defaults differ:

| Path | What the limit bounds | When it is reached | Default |
|---|---|---|---|
| Batch | the window sent to the model, which usually holds several segments | leading segments are cut off until what is left is under both limits | 65,536 tokens / 500 messages |
| Step | one segment, which every step call sends to the model whole | the history closes as a segment (needs at least two history messages) | 8,192 tokens / 50 messages |

### Prompts

Each model call sends two messages. The system message holds the rules —
`BATCH_BOUNDARY_DETECT_SYSTEM_PROMPT` or `STEP_BOUNDARY_DETECT_SYSTEM_PROMPT` in
`everalgo.boundary.prompts.en.chat` — and is the same text on every call, so a prompt cache can share all of it.
The user message holds only the data, rendered from `BATCH_BOUNDARY_DETECT_USER_PROMPT` (`<messages>`, then
`<message_count>`) or `STEP_BOUNDARY_DETECT_USER_PROMPT` (`<conversation_history>`, `<time_gap>`, `<new_messages>`).
The batch count comes after the records because it changes on every call: a window that only grew keeps the
previous call's records as its prefix, so the cache can reach past the system message. Each message line is
`[N] [YYYY-MM-DD HH:MM UTC (Weekday)] speaker: text` on the batch path and the same without `[N]` on the step path,
whose time gap reads like `Time gap: 5 minutes (recent conversation)`. Times are UTC unless the call passes `tz=`
(a `datetime.tzinfo`), which shows each message's local time with its offset, `[2024-04-10 22:00 UTC+08:00
(Wednesday)]`; the date and weekday follow that zone.
Batch records escape XML delimiters in names and content, indent continuation lines, and retain empty content
as `[N] [empty message]`. Material inside these records is data, not instructions to the detector.

Both paths judge by the same rules: the batch prompt's `<analysis_aspects>`, `<special_rules_for_common_patterns>`
and `<decision_principles>` are the step prompt's word for word, and its `<notes>` differ only in the first line.
Its role sentence and `<input>` frame them for a numbered window that more messages may follow, so its last
episode may still be ongoing. A `<time_gap>` line between consecutive records gives the time from one message to
the next, worded as on the step path (`<time_gap>Time gap: 2 days (long gap, likely new conversation)</time_gap>`).
On LoCoMo with one flush per conversation, the same prompt without these lines let days of same-topic talk merge
into one episode and scored lower; see the CHANGELOG.

Batch labels start at **0**. Each boundary `b` names the first message of a new episode, so the preceding
slice is `messages[:b]`. For example, `[2]` groups messages 0 and 1, then starts the next episode at 2.
The valid range is `1 <= b < N`; the final input message can start a new episode, and labels outside the range
are dropped. Tool items are not numbered. The strict batch response contains only `reasoning`, then
`boundaries`; `<reply_format>` shows it as a parseable skeleton, `{"reasoning": "One sentence of at most 25 words
naming the label of each message that starts a new episode, or none, and why", "boundaries": []}`. It does not
predict whether the trailing episode has finished.

The step prompt's rules and its time-gap labels are the EverAlgo LoCoMo reference run's (everalgo-boundary 0.2.1):
the rules sentence for sentence in tags instead of Markdown, nothing added; the time gap keeps that run's wording,
unit always plural (`1 minutes`). Its reply format is that run's sentence and field skeleton cut to the first two
fields, `{"reasoning", "should_end"}`: a deliberate, measured departure from that run,
which also asked for `confidence` and `topic_summary` (on LoCoMo, three runs each, the two-field reply scored the
same within noise and cost fewer output tokens per call). The skeleton (`"should_end": true/false`) is not a
parseable example reply: a deliberate exception to the LLM contract (§4.3), since every example tried moved the
decisions. The step reply model accepts those two keys only; `should_end` is required and `reasoning` defaults to
`""`. The message lines keep the shared chat-line shape above.

Both role sentences ask whether a message starts a new episode, and both ask for `reasoning` in one sentence of at
most 25 words: the batch prompt for the label of each message that starts a new episode, or none, and why; the step
prompt for whether the new dialogue starts a new episode, and why (`"should_end": true` means it does). The cap keeps
replies short (a median of 38–78 output tokens per batch call for qwen3-4b-instruct-2507, against 299 uncapped); the
role sentences, the labels in the batch `reasoning` and its `[]` example keep a small model's cuts where they were
without the cap. See the CHANGELOG for the measurements.

`custom_instructions=` on both `BoundaryDetector` methods and their sync bridges adds the host's text to the bundled
rules, never replacing them: it is placed verbatim, never rendered as a template, in a
`<custom_instructions>` block that is always the system message's last, and `None` or a blank string leaves it out.
The user message is unchanged and always rendered by the operator. There is no `prompt=`: the bundled rules cannot be
replaced per call.

## Quick start

```python
import asyncio
import json

from everalgo.boundary import BoundaryDetector, to_memcells
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import ChatMessage

_BOUNDARY_JSON = json.dumps(
    {"reasoning": "All messages concern the same deployment decision.", "boundaries": []}
)

async def main() -> None:
    # One scripted reply per LLM call; `llm.achat` returns the reply text (see `everalgo.llm.LLMClient`).
    detector = BoundaryDetector(llm=FakeLLMClient(responses=[_BOUNDARY_JSON, _BOUNDARY_JSON]))
    messages = [
        ChatMessage(id="m1", role="user",   content="Let's talk about deployment.",     timestamp=1_700_000_000_000, sender_id="u_alice"),
        ChatMessage(id="m2", role="assistant", content="Sure — what's the target env?",  timestamp=1_700_000_001_000, sender_id="assistant"),
        ChatMessage(id="m3", role="user",   content="K8s. Which deployment strategy should I use?", timestamp=1_700_000_002_000, sender_id="u_alice"),
    ]

    # Streaming: the indices point into `messages`; persist the closed cells, carry the tail.
    cuts = await detector.adetect_batch(messages)
    ends, tail_start = cuts  # NamedTuple unpacking
    cells, tail = to_memcells(messages, cuts), messages[tail_start:]

    # End-of-session: the tail is closed into the last cell.
    cuts = await detector.adetect_batch(messages, is_final=True)
    assert messages[cuts.tail_start:] == []
    for mc in to_memcells(messages, cuts):
        print(mc.timestamp, len(mc.items))


asyncio.run(main())
```

## The streaming state machine

`adetect_batch` deliberately holds back trailing messages as the tail — the LLM cannot know whether a conversation continues beyond the last seen message. The caller keeps the tail between calls:

```python
tail: list[ConversationItem] = []

for batch in incoming_batches:
    items = tail + batch
    cuts = await detector.adetect_batch(items)
    await persist(to_memcells(items, cuts))
    tail = items[cuts.tail_start:]

# Session ends — flush everything.
cuts = await detector.adetect_batch(tail, is_final=True)
await persist(to_memcells(tail, cuts))
```

The step path is the same loop one message at a time; the warmup, the mask and the bridge stay inside `adetect_step`:

```python
history: list[ChatMessage] = []

for message in incoming_messages:
    items = [*history, message]
    cuts = await detector.adetect_step(items)
    await persist(to_memcells(items, cuts))
    history = items[cuts.tail_start:]

# Session ends — the history left is the last cell.
if history:
    await persist(to_memcells(history, Cuts([len(history)], len(history))))
```

## Migrating from 0.4.2

**Breaking:** `BoundaryDetector` is the only entry, and both of its methods return `Cuts` instead of
`DetectionResult`.

| 0.4.2 | Now |
|---|---|
| `adetect_boundaries(items, llm=client, ...)` / `detect_boundaries(...)` | `BoundaryDetector(llm=client).adetect_batch(items, ...)` / `detect_batch(...)` |
| `BoundaryDetector.adetect` / `detect` | `adetect_batch` / `detect_batch` |
| `hard_msg_limit=` (batch) | `hard_message_limit=` |
| `adetect_step(history, new, ...)` | `adetect_step([*history, new], ...)` |
| the caller's own warmup (no step call before two history messages) | built into `adetect_step` (`warmup_messages=2`); drop the caller's |
| step defaults 65,536 tokens / 500 messages | 8,192 / 50; pass the old values explicitly to keep them |
| `result.cells` | `to_memcells(items, cuts)` |
| `result.tail` | `items[cuts.tail_start:]` |
| `adetect_boundary_step` / `BoundaryDecision` | private; `adetect_step` makes the one model call |
| step reply `{"reasoning", "should_end", "confidence", "topic_summary"}` | `{"reasoning", "should_end"}`; scripted or adapted step replies must drop the other two, which are now rejected |
| step log `boundary step reply: should_end=… confidence=…` and the empty-`topic_summary` warning | `boundary step reply: should_end=…`, no warning |
| batch system prompt: episode-continuity rules, an index contract and one worked example | the step prompt's rules framed for a numbered window; review custom batch instructions written against the old rules |
| batch user message: `<messages count="N">` around the numbered records | `<messages>` around the records with a `<time_gap>` line between consecutive ones, then `<message_count>` |

Apart from the step reply and the batch prompt, the messages sent to the model, the reply models and the log lines
are unchanged: for the same input and replies the cells and tails are the same as before.

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

Each call makes at most one LLM request and validates the reply against a strict pydantic model, which it also passes
as `response_format`. A reply that does not fit — not JSON, a missing or extra key, a non-integer boundary — raises
`everalgo.llm.LLMOutputError`; nothing is retried, and the client's own exceptions propagate unchanged. Retry policy
belongs to the caller.

## Tokenizer utilities

Boundary detection counts tokens with `everalgo.utils.count_tokens` (in `everalgo-core`), which uses OpenAI's `o200k_base` encoding via [`tiktoken`](https://github.com/openai/tiktoken).

## Stubs

`WorkspaceMemCellExtractor` (Jira / Email / Confluence) is a placeholder in `v0.x` — all methods raise `NotImplementedError`. It is deliberately excluded from `everalgo.boundary.__all__`; import it from `everalgo.boundary.workspace` directly if you need the reserved name. Implementation lands in a future minor bump when the `RawData` contract is finalised.

## Related distributions

- [`everalgo-user-memory`](../everalgo-user-memory/) — re-exports `BoundaryDetector`, `Cuts` and `to_memcells` beside the user-memory extractors
- [`everalgo-agent-memory`](../everalgo-agent-memory/) — re-exports `BoundaryDetector`, `Cuts` and `to_memcells` beside the agent-memory extractors
