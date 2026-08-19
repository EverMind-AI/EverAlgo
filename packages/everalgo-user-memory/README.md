# everalgo-user-memory

User-side memory products for EverAlgo — four LLM-backed extractors (`EpisodeExtractor`, `ForesightExtractor`, `AtomicFactExtractor`, `ProfileExtractor`) plus a `BoundaryDetector` class facade that wraps `everalgo-boundary`.

See the umbrella project: [EverAlgo monorepo](../../README.md) and the architecture document at [`docs/concepts/architecture.md`](../../docs/concepts/architecture.md).

## Install

```bash
pip install everalgo-user-memory
# Auto-pulls: everalgo-core, everalgo-boundary
```

## Quick start

All extractors are stateless classes; pass `llm=` at construction time. The `sender_id` argument is always required and is not inferred from the conversation.

```python
import asyncio
import json

from everalgo.llm.types import ChatResponse
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import ChatMessage, MemCell
from everalgo.user_memory import (
    BoundaryDetector,
    EpisodeExtractor,
    ForesightExtractor,
    AtomicFactExtractor,
    ProfileExtractor,
)

_BOUNDARY_JSON = json.dumps({"reasoning": "single topic", "boundaries": [], "should_wait": False})
_EPISODE_JSON  = json.dumps({"title": "Alice asks about async retries", "content": "Alice explored async retry patterns."})
_FORE_JSON     = json.dumps([{"content": "Alice will read the follow-up doc", "evidence": "assistant promised a doc", "start_time": "2023-11-14", "end_time": "2023-11-21", "duration_days": 7}])
_FACT_JSON     = json.dumps({"atomic_facts": {"time": "Nov 14 2023", "atomic_fact": ["Alice is learning Python async."]}})
_PROFILE_JSON  = json.dumps({"explicit_info": [], "implicit_traits": [{"category": "Technical", "description": "Python developer."}]})


async def main() -> None:
    messages = [
        ChatMessage(id="m1", role="user",      content="I want to learn Python async retry patterns.", timestamp=1_700_000_000_000, sender_id="u_alice", sender_name="Alice"),
        ChatMessage(id="m2", role="assistant",  content="Sure — I'll send a follow-up doc next week.", timestamp=1_700_000_001_000, sender_id="assistant"),
    ]

    fake = FakeLLMClient(responses=[
        ChatResponse(content=_BOUNDARY_JSON, model="fake"),
        ChatResponse(content=_EPISODE_JSON,  model="fake"),
        ChatResponse(content=_FORE_JSON,     model="fake"),
        ChatResponse(content=_FACT_JSON,     model="fake"),
        ChatResponse(content=_PROFILE_JSON,  model="fake"),
    ])

    # Step 1: boundary detection → MemCell
    result = await BoundaryDetector(llm=fake).adetect(messages, is_final=True)
    mc = result.cells[0]

    # Step 2–4: user-memory extractors
    episode   = await EpisodeExtractor(llm=fake).aextract(mc, sender_id="u_alice")
    foresights = await ForesightExtractor(llm=fake).aextract(mc, sender_id="u_alice")
    facts      = await AtomicFactExtractor(llm=fake).aextract(mc, sender_id="u_alice")

    # Step 5: Profile takes a chronological Sequence[MemCell]; last is most recent
    profile = await ProfileExtractor(llm=fake).aextract([mc], sender_id="u_alice")

    print(episode.subject, profile.summary)


asyncio.run(main())
```

See [`examples/06_full_user_memory_pipeline.py`](../../examples/06_full_user_memory_pipeline.py) for the complete end-to-end example including geometry clustering.

## Choosing the output language

Every extraction method takes an `output_language`. Name one and the model writes in it; leave it out and the
model works the language out for itself, which is measurably less reliable:

```python
from everalgo.user_memory import EpisodeExtractor, OutputLanguage

# Caller decides. Zero wrong-language output over the regression corpus: seven languages, five models,
# every interference pattern it holds.
episode = await EpisodeExtractor(llm=client).aextract(
    mc, sender_id="u_alice", output_language=OutputLanguage.CHINESE
)

# Model decides. Roughly one extraction in nine comes back in the wrong language, and which cases fail
# depends on the model — one of the five measured never drifted, another drifted on a quarter of them.
episode = await EpisodeExtractor(llm=client).aextract(mc, sender_id="u_alice")
```

Plain strings work too, in any casing (`"chinese"`, `"German"`) — convenient when the value comes from
config. An unrecognised name raises `ValueError` rather than reaching the prompt.

**Decide the language once, upstream, and pass the same value to every call.** The alternative — deriving
one extractor's language from another's output — inherits that extractor's error rate, and for profiles the
consequence compounds: a profile updated without a named language inherits whatever language it already
says, so one wrong INIT persists through every later update. Passing the language on the update is the way
back out. Note also that `category` and `trait` labels are model-authored, so they follow the argument along
with the descriptions (`Location` versus `居住地`) — worth knowing if you group or filter on them.

## Customising prompts

Each extractor accepts a `prompt=` override per call, or the module-level constant can be monkey-patched at
startup for a global override:

```python
# Per-call override. A replacement keeping the {language_rule} placeholder keeps output-language control;
# one that drops it opts out.
episode = await EpisodeExtractor(llm=client).aextract(mc, sender_id="u_alice", prompt=my_custom_prompt)

# Global: replace the default prompt at startup
import everalgo.user_memory.prompts.en.foresight as _fs
_fs.FORESIGHT_GENERATION_PROMPT = my_custom_prompt
```

Prompts ship in English only. A parallel `prompts/zh/` tree used to carry translations and was removed:
prompt language turned out to dictate output language almost entirely, so the translations were an implicit
language switch maintained by hand. `output_language` does that job from one prompt tree, for languages
nobody has to translate a prompt into.

## API surface

```python
class BoundaryDetector:
    def __init__(self, *, llm: LLMClient) -> None: ...
    async def adetect(
        self, messages: list[ChatMessage], *, is_final: bool = False, prompt: str | None = None
    ) -> DetectionResult: ...

class EpisodeExtractor:
    def __init__(self, *, llm: LLMClient) -> None: ...
    async def aextract(
        self, memcell: MemCell, *,
        sender_id: str | None,           # None → generic whole-memcell episode (cheaper)
        prompt: str | None = None,
        custom_instructions: str | None = None,
    ) -> Episode: ...

class ForesightExtractor:
    def __init__(self, *, llm: LLMClient) -> None: ...
    async def aextract(
        self, memcell: MemCell, *, sender_id: str, prompt: str | None = None
    ) -> list[Foresight]: ...

class AtomicFactExtractor:
    def __init__(self, *, llm: LLMClient) -> None: ...
    async def aextract(
        self, memcell: MemCell, *,
        sender_id: str | None,           # None → generic facts not bound to any user
        prompt: str | None = None,
    ) -> list[AtomicFact]: ...

class ProfileExtractor:
    def __init__(self, *, llm: LLMClient) -> None: ...
    async def aextract(
        self, memcells: Sequence[MemCell], *,
        sender_id: str,
        old_profile: Profile | None = None,   # None → INIT mode; present → UPDATE mode
        prompt: str | None = None,
    ) -> Profile: ...
```

`EpisodeExtractor` has two modes: pass `sender_id=str` to extract a user-focused episode (uses `USER_EPISODE_GENERATION_PROMPT`); pass `sender_id=None` for a generic whole-memcell episode (uses `EPISODE_GENERATION_PROMPT`).

`ProfileExtractor` has two modes: `old_profile=None` triggers INIT extraction; passing an existing profile triggers UPDATE (LLM emits add/update/delete ops). When the merged profile exceeds an internal item count threshold a second compact LLM pass runs automatically — this is transparent to the caller.

All class methods have a sync bridge: `extractor.extract(...)` is `async_to_sync(aextract)` — only for non-event-loop callers (CLI scripts, plain unit tests).

## Testing

```python
from everalgo.testing import FakeLLMClient, assert_episode_shape

fake = FakeLLMClient(responses=[ChatResponse(content=_EPISODE_JSON, model="fake")])
episode = await EpisodeExtractor(llm=fake).aextract(mc, sender_id="u_alice")
assert_episode_shape(episode)
```

See the integration test pattern in [`tests/integration/`](../../tests/integration/).

## Related distributions

- [`everalgo-boundary`](../everalgo-boundary/) — `detect_boundaries` primitive used by `BoundaryDetector`
- [`everalgo-clustering`](../everalgo-clustering/) — geometry / LLM clustering for grouping MemCells before `ProfileExtractor`
- [`everalgo-rank`](../everalgo-rank/) — ranks `Episode`, `AtomicFact`, `Profile` candidates at read time
