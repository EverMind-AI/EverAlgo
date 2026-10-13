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
| `everalgo.types` | Key shared data types: `MemCell`, `ChatMessage`, `ConversationItem`, user / agent / knowledge memories, ranking contracts, parsed / raw content, modality helpers, tool-call types |
| `everalgo.llm` | `LLMClient` Protocol, `LLMMessage` + content parts, `LLMOutputError`, `StrictOutputModel`, `parse_reply` / `describe_errors`, the request builders `to_openai_response_format` / `to_anthropic_response_format` / `to_gemini_response_format` / `to_ollama_response_format` |
| `everalgo.utils` | General-purpose tools, one kind per module: `templates` (`render_prompt`), `markup` (`tagged`, `paragraphs`), `chat_lines` (`render_chat_line`, `speaker`), `timestamps` (`format_utc_time`, `format_utc_date`, `parse_utc_date`), `tokens` (`encode`, `count_tokens`, `truncate_text`), `text` (`ascii_width`, `decode_bytes`), `json` (`json_default`), `http` (`fetch_uri`) |
| `everalgo.llm.schema` | The `to_<api>_response_format` request builders — a `response_format` model to each API's structured-output request value, refusing structures outside the strict subset |
| `everalgo.llm.openai_compat` | `OpenAICompatClient` — reference `LLMClient`, OpenAI Chat Completions over httpx with no provider SDK; `LLMError`, the error it raises (not part of the contract) |
| `everalgo.testing` | `FakeLLMClient`, `CallRecord`, `assert_*_shape` structural assertion helpers, and the `LLMClient` conformance suite (`everalgo.testing.conformance`) |

## Layout and responsibilities

```
everalgo.llm      the LLM contract, and the tools that help an implementation meet it
everalgo.types    the domain types operators take and return
everalgo.utils    general-purpose tools, one kind per module
everalgo.testing  test doubles and the LLMClient conformance suite

testing ──→ llm, types          llm, types and utils import none of each other
```

| Subpackage | Holds | Does not hold |
|---|---|---|
| `llm` | `LLMClient` and its request types, `LLMOutputError`, `StrictOutputModel` and `parse_reply`, the `to_<api>_response_format` builders, and reference implementations, one flat module per API (`openai_compat`) | Anything a prompt or an operator needs besides the contract |
| `types` | Data structures only: pydantic models, enums, and their constants | Helper functions; logic bound to one type is a method of that type |
| `utils` | Tools that are general-purpose, whatever the number of callers today; one kind per module (`templates`, `markup`, `chat_lines`, `timestamps`, `tokens`, `text`, `json`, `http`) | Tools with domain meaning, which stay next to the domain they serve, and prompt strings, which live in each package's `prompts/{en,zh}/` |
| `testing` | `FakeLLMClient`, the `assert_*_shape` helpers and the conformance suite | Production code |

Two rules hold across the workspace. Anything another distribution imports from core is public: the operator packages are released separately, so a private `_name` they depended on could change under them without a version bump. And core depends only on `httpx`, `pydantic` and `tiktoken`: a tool that needs a heavier library, such as Pillow or BeautifulSoup, stays in the package that needs it. `tests/test_core_layering.py` at the repository root enforces the import direction and both rules.

## Quick start

```python
from contextlib import aclosing

from everalgo.llm import LLMMessage, StrictOutputModel, parse_reply
from everalgo.llm.openai_compat import OpenAICompatClient


class Greeting(StrictOutputModel):
    text: str


async with aclosing(OpenAICompatClient(model="gpt-4o-mini", api_key="sk-...")) as client:
    reply = await client.achat(
        messages=[LLMMessage(role="user", content='Say hello. Reply with one JSON object: {"text": "..."}.')],
        response_format=Greeting,
    )
# The operator always parses the reply itself; a reply that does not match raises LLMOutputError.
greeting = parse_reply(Greeting, reply, operator="greeting")
```

`achat` returns the reply text. `response_format` is a pydantic model: the client's `response_format_mode` decides how it asks for it, through `to_openai_response_format` (`json_schema`, the default, sends the model's strict schema; `json_object` sends JSON mode; `prompt_only` sends nothing and relies on the prompt). The client then validates the reply against the model and retries a reply that fails, at most `invalid_reply_retries` times. Every model — nested sub-models included — inherits `StrictOutputModel` and gives its fields no defaults, so local validation accepts exactly what the strict schema accepts.

### Using FakeLLMClient in tests

```python
import json
from everalgo.llm import LLMMessage
from everalgo.testing import FakeLLMClient

# Scripted mode — reply texts are returned in order; an exception in the list is raised instead
fake = FakeLLMClient(responses=[
    json.dumps({"title": "T", "content": "C", "summary": "S"}),
    TimeoutError("upstream timed out"),
])

# Handler mode — full control over each call
def handler(messages: list[LLMMessage], **_) -> str:
    return '{"text": "ok"}'

fake = FakeLLMClient(handler=handler)
print(fake.call_count)  # 0
```

### Checking an `LLMClient` implementation

`everalgo.testing.conformance` is the executable half of the contract. Implement `LLMClientTestHarness` (build a client against a scripted upstream, report the requests it actually sent) and run every check in `REQUIRED_CHECKS` — for example `@pytest.mark.parametrize("check", REQUIRED_CHECKS)`. `OPTIONAL_CHECKS` cover recommendations; failing them is not a breach. `tests/llm/test_openai_compat_conformance.py` is the harness for the reference client.

## Utilities

`everalgo.utils` holds the general-purpose tools every package may use, one kind per module. Timestamps take milliseconds since the epoch and render in UTC, to the minute, as `YYYY-MM-DD HH:MM UTC` — the one shape every prompt shows a model; token counts use OpenAI's `o200k_base` encoding via [`tiktoken`](https://github.com/openai/tiktoken), a budget estimate for any model.

```python
from everalgo.utils import count_tokens, format_utc_time, render_prompt, truncate_text

render_prompt("Summarise {{text}}", None, text="...")  # the default template unless the caller passes one
format_utc_time(1_700_000_000_000)  # "2023-11-14 22:13 UTC"; weekday=True adds " (Tuesday)"
count_tokens("Hello, world!")  # 4
truncate_text(long_text, max_tokens=500)  # head and tail around a "[... trimmed N tokens ...]" marker
```

## API surface

| Symbol | Module | Role |
|---|---|---|
| `MemCell` | `everalgo.types` | Boundary-detected conversation slice; `items: list[ConversationItem]`, `timestamp: int` |
| `ChatMessage` | `everalgo.types` | Chat wire type: `id`, `role`, `content`, `timestamp`, `sender_id`, `sender_name` |
| `ConversationItem` | `everalgo.types` | `ChatMessage \| ToolCallRequest \| ToolCallResult` discriminated union |
| `Episode` / `Foresight` / `AtomicFact` / `Profile` | `everalgo.types` | User-side memory output types |
| `ExplicitInfoItem` / `ImplicitTraitItem` | `everalgo.types` | Typed items of `Profile.explicit_info` / `Profile.implicit_traits`; host keys ride through via `extra="allow"` |
| `AgentCase` / `AgentSkill` / `AgentProfilePatch` / `AgentProfileSignal` / `AgentProfileUpdate` | `everalgo.types` | Agent-side memory and config-update output types |
| `RankInput` / `RankOutput` / `Candidate` / `ScoredItem` | `everalgo.types` | Rank I/O contracts |
| `LLMClient` | `everalgo.llm` | Protocol — `achat(messages, *, response_format, temperature=None, max_tokens=None, **extra) → str` |
| `LLMMessage` | `everalgo.llm` | One request message: `role` (system / user / assistant) and `content` (`str` or content parts) |
| `LLMOutputError` | `everalgo.llm` | Raised by operators when the reply fails to parse or fails business validation |
| `StrictOutputModel` | `everalgo.llm` | Configuration-only base (`strict=True, extra="forbid"`) for every `response_format` model |
| `parse_reply` / `describe_errors` | `everalgo.llm` | Validate a reply against its `response_format` model, raising `LLMOutputError` that names each failing location, never the reply |
| `OpenAICompatClient` | `everalgo.llm.openai_compat` | Reference client, built from keyword arguments (`model`, `api_key`, `base_url`, `temperature`, `max_tokens`, `timeout`, `response_format_mode`, `invalid_reply_retries`, `transport_retries`) and released with `aclose()`; raises `LLMError` on refusal, absent content, `length` or `content_filter`; validates each reply against `response_format`, retries an invalid reply up to `invalid_reply_retries` times; resends a transport error, HTTP 408 / 409 / 429 / 5xx, a reply with no choice or an aborted generation (`abort`, `error`) up to `transport_retries` times |
| `to_openai_response_format` | `everalgo.llm` | `(model, mode)` → the value of the OpenAI Chat Completions `response_format` parameter, or `None` for `prompt_only` |
| `to_anthropic_response_format` | `everalgo.llm` | `(model, mode)` → the value of Anthropic's `output_config.format`, or `None`; keywords Anthropic does not take move into the description |
| `to_gemini_response_format` | `everalgo.llm` | `(model, mode)` → the Gemini `generationConfig` entries `responseMimeType` / `responseJsonSchema`; keywords Gemini does not list move into the description |
| `to_ollama_response_format` | `everalgo.llm` | `(model, mode)` → the value of Ollama's `format`: the schema, `"json"`, or `None` |
| `LLMError` | `everalgo.llm.openai_compat` | The reference client's error; not part of the contract |
| `format_utc_time` | `everalgo.utils` | `YYYY-MM-DD HH:MM UTC` from a millisecond timestamp, seconds truncated; `weekday=True` appends ` (Wednesday)` where the model resolves relative times |
| `format_utc_date` | `everalgo.utils` | `YYYY-MM-DD` UTC date from a millisecond timestamp or an aware datetime |
| `parse_utc_date` | `everalgo.utils` | Strict `YYYY-MM-DD` → aware UTC midnight, or `None` when not that shape or not a real date |
| `render_prompt` | `everalgo.utils` | Template substitution for `{{name}}` placeholders |
| `tagged` / `paragraphs` | `everalgo.utils` | An XML-style prompt block, open and close tag each alone on its line / blocks joined by one blank line |
| `render_chat_line` / `speaker` | `everalgo.utils` | One `ChatMessage` as `[YYYY-MM-DD HH:MM UTC (Wednesday)] speaker: text`, the line every operator shows a model / `sender_name`, else `sender_id` |
| `encode` / `count_tokens` | `everalgo.utils` | Token ids / token count under `o200k_base`; special-token literals count as ordinary text |
| `truncate_text` | `everalgo.utils` | Cut text to a token budget, keeping head and tail around a trim marker, or the head plus a suffix |
| `ascii_width` | `everalgo.utils` | Text length in ASCII-equivalent units: East Asian wide characters count 2 |
| `decode_bytes` | `everalgo.utils` | Decode bytes of unknown charset as UTF-8, else GBK, else Latin-1; never raises |
| `json_default` | `everalgo.utils` | `json.dumps(default=...)` fallback: datetime as ISO 8601, anything else as `str` |
| `fetch_uri` | `everalgo.utils` | Fetch an `http` / `https` URI; other schemes are refused, private addresses are not blocked |
| `FakeLLMClient` | `everalgo.testing` | In-memory `LLMClient` double; `responses=[...]` or `handler=callable` |
| `CallRecord` | `everalgo.testing` | Single recorded `achat()` invocation (for test assertions) |
| `LLMClientTestHarness` / `EffectiveRequest` / `UpstreamStub` / `NOT_SENT` | `everalgo.testing` | What an `LLMClient` implementation provides and sees when running the conformance suite |
| `REQUIRED_CHECKS` / `OPTIONAL_CHECKS` | `everalgo.testing` | The conformance checks (hard clauses / recommendations) |
| `assert_episode_shape` | `everalgo.testing` | Structural assertion: required `Episode` fields are non-empty |
| `assert_foresight_shape` | `everalgo.testing` | Structural assertion for `Foresight` |
| `assert_atomic_fact_shape` | `everalgo.testing` | Structural assertion for `AtomicFact` |
| `assert_profile_shape` | `everalgo.testing` | Structural assertion for `Profile` |

## Related distributions

- [`everalgo-boundary`](../everalgo-boundary/) — uses `MemCell`, `ChatMessage`, `LLMClient`
- [`everalgo-clustering`](../everalgo-clustering/) — uses `LLMClient` and core prompt helpers
- [`everalgo-rank`](../everalgo-rank/) — uses `RankInput`, `RankOutput`, `Candidate`, `ScoredItem`
- [`everalgo-user-memory`](../everalgo-user-memory/) — uses all user-side types + `LLMClient`
- [`everalgo-agent-memory`](../everalgo-agent-memory/) — uses agent-side types + `LLMClient`
