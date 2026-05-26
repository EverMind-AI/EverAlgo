# everalgo-core

Foundation distribution of EverAlgo — data types, LLM client + provider routing, prompt helpers, and testing utilities shared by every other `everalgo-*` package.

See the umbrella project: [EverAlgo monorepo](../../README.md) and the architecture document at [`docs/concepts/architecture.md`](../../docs/concepts/architecture.md).

## Install

```bash
pip install everalgo-core
```

## What this distribution provides

| Subpackage | What it contains |
|---|---|
| `everalgo.types` | All shared data types: `MemCell`, `ChatMessage`, `ConversationItem`, `Episode`, `Foresight`, `AtomicFact`, `Profile`, `AgentCase`, `AgentSkill`, `RankInput`, `RankOutput`, `Candidate`, `ScoredItem`, `ParsedContent`, `RawFile`, tool-call types |
| `everalgo.llm` | `LLMClient` Protocol, `LLMConfig`, `ChatMessage` / `ChatResponse` / `Usage`, `LLMError`, `build_client` factory |
| `everalgo.llm.format` | `format_message_timestamp`, `format_natural_language_time` — timestamp rendering for LLM prompts |
| `everalgo.llm.parse` | `parse_llm_json_object` — three-tier robust JSON-object extraction |
| `everalgo.llm.providers.openai_compat` | `OpenAICompatClient` — wraps `openai.AsyncOpenAI` |
| `everalgo.prompts` | `render_prompt` — shared template substitution helper |
| `everalgo.testing` | `FakeLLMClient`, `CallRecord`, and `assert_*_shape` structural assertion helpers |

## Quick start

```python
from everalgo.llm import LLMConfig, build_client
from everalgo.llm.types import ChatMessage

config = LLMConfig(model="gpt-4o-mini", api_key="sk-...", base_url="https://api.openai.com/v1")
client = build_client(config)

response = await client.chat(
    messages=[ChatMessage(role="user", content="Hello")],
)
print(response.content)
```

### Using FakeLLMClient in tests

```python
import json
from everalgo.llm.types import ChatResponse
from everalgo.testing import FakeLLMClient

# Scripted mode — responses are returned in order
fake = FakeLLMClient(responses=[
    ChatResponse(content=json.dumps({"title": "T", "content": "C"}), model="fake"),
])

# Handler mode — full control over each call
from everalgo.llm.types import ChatMessage as LLMMsg
def handler(messages: list[LLMMsg], **_) -> ChatResponse:
    return ChatResponse(content="ok", model="fake")

fake = FakeLLMClient(handler=handler)
print(fake.call_count)  # 0
```

## LLM utilities

`everalgo.llm.format` and `everalgo.llm.parse` provide prompt-rendering helpers used by boundary detection and every extractor.

### `everalgo.llm.format` — timestamp rendering

```python
from everalgo.llm.format import format_message_timestamp, format_natural_language_time

# ISO 8601 UTC anchor — used as inline message prefix in prompts
# e.g. "2023-11-14T22:13:20Z"
ts_str = format_message_timestamp(1_700_000_000_000)

# Human-readable label for LLM time-of-day reasoning
# Example output: "November 14, 2023 (Tuesday) at 10:13 PM UTC"
label = format_natural_language_time(1_700_000_000_000, lang="en")
```

### `everalgo.llm.parse` — robust JSON extraction

```python
from everalgo.llm.parse import parse_llm_json_object

# Three-tier fallback: fence block → direct loads → outermost braces
result = parse_llm_json_object('```json\n{"key": "value"}\n```')
# → {"key": "value"}
```

`parse_llm_json_object(raw: str) -> dict[str, Any]` tries each strategy in order and raises `ValueError` if all three fail.

### `everalgo._tokenize` — token counting

`count_tokens(text: str) → int` and `force_split(text: str, *, max_tokens: int) → list[str]` are module-private utilities (used internally by boundary algorithms; not part of the public `everalgo.*` surface). They use OpenAI `o200k_base` encoding via [`tiktoken`](https://github.com/openai/tiktoken).

## API surface

| Symbol | Module | Role |
|---|---|---|
| `MemCell` | `everalgo.types` | Boundary-detected conversation slice; `items: list[ConversationItem]`, `timestamp: int` |
| `ChatMessage` | `everalgo.types` | Chat wire type: `id`, `role`, `content`, `timestamp`, `sender_id`, `sender_name` |
| `ConversationItem` | `everalgo.types` | `ChatMessage \| ToolCallRequest \| ToolCallResult` discriminated union |
| `Episode` / `Foresight` / `AtomicFact` / `Profile` | `everalgo.types` | User-side memory output types |
| `AgentCase` / `AgentSkill` | `everalgo.types` | Agent-side memory output types |
| `RankInput` / `RankOutput` / `Candidate` / `ScoredItem` | `everalgo.types` | Rank I/O contracts |
| `LLMClient` | `everalgo.llm` | Protocol — `chat(messages, *, model=None, ...) → ChatResponse` |
| `LLMConfig` | `everalgo.llm` | `model`, `api_key` (SecretStr), `base_url`, `temperature`, `max_tokens`, `timeout` |
| `build_client` | `everalgo.llm` | Factory: `LLMConfig → LLMClient` (returns `OpenAICompatClient`) |
| `LLMError` | `everalgo.llm` | Base exception raised by all providers |
| `OpenAICompatClient` | `everalgo.llm.providers.openai_compat` | Wraps `openai.AsyncOpenAI`; no retry logic |
| `format_message_timestamp` | `everalgo.llm.format` | ISO 8601 UTC string from a millisecond timestamp |
| `format_natural_language_time` | `everalgo.llm.format` | Human-readable timestamp label; `lang="en"` or `"zh"` |
| `parse_llm_json_object` | `everalgo.llm.parse` | Robust JSON-object parser with three-tier fallback |
| `render_prompt` | `everalgo.prompts` | Template substitution for `{placeholder}` patterns |
| `FakeLLMClient` | `everalgo.testing` | In-memory `LLMClient` double; `responses=[...]` or `handler=callable` |
| `CallRecord` | `everalgo.testing` | Single recorded `chat()` invocation (for test assertions) |
| `assert_episode_shape` | `everalgo.testing` | Structural assertion: required `Episode` fields are non-empty |
| `assert_foresight_shape` | `everalgo.testing` | Structural assertion for `Foresight` |
| `assert_atomic_fact_shape` | `everalgo.testing` | Structural assertion for `AtomicFact` |
| `assert_profile_shape` | `everalgo.testing` | Structural assertion for `Profile` |

## Related distributions

- [`everalgo-boundary`](../everalgo-boundary/) — uses `MemCell`, `ChatMessage`, `LLMClient`
- [`everalgo-clustering`](../everalgo-clustering/) — uses `MemCell`, `LLMClient`
- [`everalgo-rank`](../everalgo-rank/) — uses `RankInput`, `RankOutput`, `Candidate`, `ScoredItem`
- [`everalgo-user-memory`](../everalgo-user-memory/) — uses all user-side types + `LLMClient`
- [`everalgo-agent-memory`](../everalgo-agent-memory/) — uses agent-side types + `LLMClient`
