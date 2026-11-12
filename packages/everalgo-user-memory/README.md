# everalgo-user-memory

User-side memory products for EverAlgo — four LLM-backed extractors (`EpisodeExtractor`, `ForesightExtractor`, `AtomicFactExtractor`, `ProfileExtractor`), an `EpisodeReflector` that merges several episodes into one narrative, an `EpisodeSummarizer` that writes an episode's summary from its memcell (beside extraction) or from an episode text already stored, plus `BoundaryDetector`, its `Cuts` result and the `to_memcells` helper, re-exported unchanged from `everalgo-boundary`.

See the umbrella project: [EverAlgo monorepo](../../README.md) and the architecture document at [`docs/concepts/architecture.md`](../../docs/concepts/architecture.md).

## Install

```bash
pip install everalgo-user-memory==0.9.0rc6
```

## Quick start

All extractors are stateless classes; pass `llm=` at construction time. The `sender_id` argument is always required and is not inferred from the conversation.

```python
import asyncio
import json

from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import ChatMessage, MemCell
from everalgo.user_memory import (
    BoundaryDetector,
    EpisodeExtractor,
    ForesightExtractor,
    AtomicFactExtractor,
    ProfileExtractor,
    to_memcells,
)

_BOUNDARY_JSON = json.dumps({"reasoning": "Explanation pending", "boundaries": []})
_EPISODE_JSON  = json.dumps({"title": "Alice asks about async retries", "content": "Alice explored async retry patterns."})
_FORE_JSON     = json.dumps({"foresights": [{"content": "Alice will read the follow-up doc", "evidence": "assistant promised a doc", "start_time": "2023-11-14", "end_time": "2023-11-21", "duration_days": 7}]})
_FACT_JSON     = json.dumps({"atomic_facts": {"atomic_fact": ["Alice is learning Python async."]}})
_PROFILE_JSON  = json.dumps({"explicit_info": [], "implicit_traits": [{"trait": "Pragmatic", "description": "Prefers minimal-ceremony tooling.", "basis": "chose ruff over black"}]})


async def main() -> None:
    messages = [
        ChatMessage(id="m1", role="user",      content="I want to learn Python async retry patterns.", timestamp=1_700_000_000_000, sender_id="u_alice", sender_name="Alice"),
        ChatMessage(id="m2", role="assistant",  content="Sure — I'll send a follow-up doc next week.", timestamp=1_700_000_001_000, sender_id="assistant"),
    ]

    fake = FakeLLMClient(responses=[
        _BOUNDARY_JSON,
        _EPISODE_JSON,
        _FORE_JSON,
        _FACT_JSON,
        _PROFILE_JSON,
    ])

    # Step 1: boundary detection → Cuts (indices into `messages`) → MemCell
    cuts = await BoundaryDetector(llm=fake).adetect_batch(messages, is_final=True)
    mc = to_memcells(messages, cuts)[0]

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

Every LLM-backed method takes an `output_language`. Name one and the model writes in it; leave it out and the
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
config. An unrecognised name raises `everalgo.utils.InvalidInputError` (a `ValueError`) rather than reaching the prompt.

**Decide the language once, upstream, and pass the same value to every call.** The alternative — deriving
one extractor's language from another's output — inherits that extractor's error rate, and for profiles the
consequence compounds: a profile updated without a named language inherits whatever language it already
says, so one wrong INIT persists through every later update. Passing the language on the update is the way
back out. Note also that `category` and `trait` labels are model-authored, so they follow the argument along
with the descriptions (`Location` versus `居住地`) — worth knowing if you group or filter on them.

What "leave it out" means depends on the operator. The four reading a raw conversation judge the language from
what the participants write, which is the 10.2% path above. `EpisodeExtractor` is one of them, although the EverAlgo
LoCoMo reference run whose prompt it reproduces sent no language rule; the figures above were measured on an earlier
episode prompt. The four reading already-extracted memory —
`AtomicFactExtractor.aextract_from_text`, `ProfileExtractor.aextract_from_episodes`,
`EpisodeReflector.areflect` and `EpisodeSummarizer.asummarize` — instead inherit the language of their input, which is a much easier call for a
single-language narrative but not free. The Profile Episode-text path and `areflect` can take several episodes,
so inputs that disagree on language leave the model to pick one; an update inherits the existing Profile or
narrative language. Name a language when inputs may disagree, or to move an existing result that is already in
the wrong one.

## Time zone

Every extractor that shows the model a time takes `tz=`, a `datetime.tzinfo` (default UTC): `EpisodeExtractor`, `AtomicFactExtractor` (both entry points), `ForesightExtractor` and `ProfileExtractor` (both entry points). Times render in that zone with its offset, `[2024-03-11 00:00 UTC+14:00 (Monday)] Alice: text`, and the date and English weekday follow the zone, so "yesterday" resolves against the user's own calendar. UTC renders as `2024-03-10 10:00 UTC (Sunday)`. Build the zone on the host side, for example `zoneinfo.ZoneInfo("Asia/Shanghai")`; everalgo adds no time-zone dependency.

## Customising prompts

Every operator that takes per-call instructions takes them as `custom_instructions=`: the text is added to the bundled rules
as the last block of the system message and never replaces them. `BoundaryDetector`, `EpisodeExtractor`, `ProfileExtractor`,
`AtomicFactExtractor.aextract_from_text`, `EpisodeReflector` and `EpisodeSummarizer` take it; `ForesightExtractor` and `AtomicFactExtractor.aextract`
take no per-call instructions at all.

`ProfileExtractor` takes `custom_instructions=`, and it appends rather than replaces: the host's
instructions follow the bundled rules of every stage the call runs — an `<additional_requirements>` block closing
the system message on the Episode-text path, an `【Additional Requirements】` section before the data on the
MemCell path — and the output shape stays fixed. On the Episode-text path the block also says the bundled
rules hold where the two conflict, so the instructions can tighten the rules but not loosen them. It is the host's own text, not an end user's: it reaches the
model as an instruction, so whatever a host forwards into it is the host's to vet.

```python
profile = await ProfileExtractor(llm=client).aextract_from_episodes(
    episodes, owner_id="u_alice", custom_instructions="Record dietary restrictions under a Health category."
)
```

`EpisodeExtractor` likewise takes only `custom_instructions=`: the host's text, placed verbatim in a
`<custom_instructions>` block after the bundled rules; the output shape stays fixed. `None` or a blank string does not
leave the block out: it holds the EverAlgo LoCoMo reference run's five default principles
(`prompts.en.episode.DEFAULT_CUSTOM_INSTRUCTIONS`), as that run filled it, and a host's text takes their place.

Prompts ship in English only. A parallel `prompts/zh/` tree used to carry translations and was removed:
prompt language turned out to dictate output language almost entirely, so the translations were an implicit
language switch maintained by hand. `output_language` does that job from one prompt tree, for languages
nobody has to translate a prompt into.

## API surface

```python
class BoundaryDetector:  # everalgo.boundary.BoundaryDetector, re-exported
    def __init__(self, *, llm: LLMClient) -> None: ...
    async def adetect_batch(
        self, items: Sequence[ConversationItem], *, is_final: bool = False, custom_instructions: str | None = None,
        tz: tzinfo = UTC, hard_token_limit: int = 65_536, hard_message_limit: int = 500,
    ) -> Cuts: ...
    async def adetect_step(
        self, items: Sequence[ChatMessage], *, custom_instructions: str | None = None, tz: tzinfo = UTC,
        warmup_messages: int = 2, smart_mask: bool = True, smart_mask_threshold: int = 5,
        hard_token_limit: int = 8_192, hard_message_limit: int = 50,
    ) -> Cuts: ...

class Cuts(NamedTuple):  # everalgo.boundary.Cuts, re-exported
    ends: list[int]  # segment i is items[ends[i-1]:ends[i]], the first starting at 0
    tail_start: int  # carry items[tail_start:] into the next call

def to_memcells(items: Sequence[ConversationItem], cuts: Cuts) -> list[MemCell]: ...  # everalgo.boundary, re-exported

class EpisodeExtractor:
    def __init__(self, *, llm: LLMClient) -> None: ...
    async def aextract(
        self, memcell: MemCell, *,
        sender_id: str | None,           # only sets Episode.owner_id; None → an episode bound to no user
        custom_instructions: str | None = None,   # host's instructions; None → the default principles
        output_language: OutputLanguage | str | None = None,   # None → the participants' language
    ) -> Episode: ...

class ForesightExtractor:
    def __init__(self, *, llm: LLMClient) -> None: ...
    async def aextract(
        self, memcell: MemCell, *,
        sender_id: str,
        output_language: OutputLanguage | str | None = None,   # None → the model infers it
    ) -> list[Foresight]: ...

class AtomicFactExtractor:
    def __init__(self, *, llm: LLMClient) -> None: ...
    async def aextract(
        self, memcell: MemCell, *,
        sender_id: str | None,           # None → generic facts not bound to any user
        output_language: OutputLanguage | str | None = None,   # None → the model infers it
    ) -> list[AtomicFact]: ...

    async def aextract_from_text(
        self, text: str, *,
        timestamp: int,                  # anchors relative dates the text mentions
        custom_instructions: str | None = None,   # host's instructions, the closing block
        output_language: OutputLanguage | str | None = None,   # None → inherited from the input
    ) -> list[AtomicFact]: ...

class ProfileExtractor:
    def __init__(self, *, llm: LLMClient) -> None: ...
    async def aextract(
        self, memcells: Sequence[MemCell], *,
        sender_id: str,
        old_profile: Profile | None = None,   # None → INIT mode; present → UPDATE mode
        custom_instructions: str | None = None,   # host's instructions, appended after the bundled rules
        output_language: OutputLanguage | str | None = None,   # None → the model infers it
    ) -> Profile: ...

    async def aextract_from_episodes(
        self, episodes: Sequence[Episode], *,   # dated narratives, any order; only the ones new to the profile
        owner_id: str,
        owner_name: str | None = None,     # non-blank name targets the owner; otherwise owner_id
        old_profile: Profile | None = None,
        categories: Sequence[str] | None = None,   # complete current explicit_info category snapshot
        custom_instructions: str | None = None,   # host's instructions, appended after the bundled rules
        output_language: OutputLanguage | str | None = None,   # None → inherited from Episode text/profile
    ) -> Profile: ...

class EpisodeReflector:
    def __init__(self, *, llm: LLMClient) -> None: ...
    async def areflect(
        self, episodes: Sequence[Episode], *,
        old_episode: Episode | None = None,   # None → INIT merge; present → UPDATE
        custom_instructions: str | None = None,   # host's instructions, the closing block
        output_language: OutputLanguage | str | None = None,   # None → inherited from the input
    ) -> Episode: ...

class EpisodeSummarizer:
    def __init__(self, *, llm: LLMClient) -> None: ...
    async def asummarize(
        self, episode_text: str, *,      # the stored narrative (Episode.episode); blank → InvalidInputError
        output_language: OutputLanguage | str | None = None,   # None → inherited from the text
        custom_instructions: str | None = None,   # host's instructions, the closing block
    ) -> str: ...
    async def asummarize_memcell(
        self, memcell: MemCell, *,       # the memcell EpisodeExtractor reads; no chat text → InvalidInputError
        output_language: OutputLanguage | str | None = None,   # None → the participant rule, as extraction
        custom_instructions: str | None = None,   # host's instructions, the closing block
        tz: tzinfo = UTC,                # zone the conversation's times are shown in
    ) -> str: ...
```

Every episode from `EpisodeExtractor` carries two model-written fields, `subject` (the title) and `episode` (the full
narrative), both required: if the model omits either or returns it blank, `aextract` raises `LLMOutputError` rather
than substituting a value. Its `summary` is the narrative's first 200 characters, cut in code as the EverAlgo LoCoMo
reference run cut it; the model writes none, and the cut can end mid-word.

`EpisodeReflector` has the model write all three fields, so a merged episode has a written preview of the merged
narrative.

`EpisodeSummarizer.asummarize_memcell` writes the preview from the memcell itself, so a host can make it beside
`EpisodeExtractor.aextract` and store the model's preview in place of the 200-character cut without waiting for the
narrative. The user message is the conversation exactly as extraction renders it (`<conversation_start_time>` and
`<conversation>`), without extraction's `<reminder>`; the system message is `<role>`, `<the_preview>` — only
`prompts.en.episode.EPISODE_SUMMARY_PREVIEW_RULE`, the first sentence of the summary description —, `<reply_format>`,
`<output_language>` (the named language, or extraction's participant rule) and, when the host passes some,
`<custom_instructions>`. The reply is held to `{"summary"}` and the 400-unit guard applies; a memcell with no chat
message text raises `InvalidInputError` before any call.

```python
import asyncio

from everalgo.user_memory import EpisodeExtractor, EpisodeSummarizer

episode, summary = await asyncio.gather(
    EpisodeExtractor(llm=client).aextract(memcell, sender_id=user_id),
    EpisodeSummarizer(llm=client).asummarize_memcell(memcell),
)
episode = episode.model_copy(update={"summary": summary})
```

`EpisodeSummarizer.asummarize` writes the preview alone, for an episode text already stored — a body a user edited, or an
extracted episode whose 200-character cut should give way to a written preview. It asks for the summary in the words
of `prompts.en.episode.EPISODE_SUMMARY_FIELD_RULE`, adds that the preview may state no fact the text does not and
keeps the text's time format, and cuts a summary wider than 400 units at a sentence boundary. One call —
a system message of rules (`EPISODE_SUMMARY_SYSTEM_PROMPT`, then `<output_language>` and, last, `<custom_instructions>`
when the host passes some) and a user message holding only the text in `<episode_text>`; the reply is held strictly to
`{"summary"}`, and a reply that fails or leaves it blank raises `LLMOutputError` with no retry. A blank `episode_text`
raises `InvalidInputError` before any call; no length bound is applied, the host owns its input budget.

```python
from everalgo.user_memory import EpisodeSummarizer, OutputLanguage

summary = await EpisodeSummarizer(llm=client).asummarize(
    episode.episode, output_language=OutputLanguage.CHINESE
)
episode = episode.model_copy(update={"summary": summary})
```

`EpisodeReflector.areflect` and `AtomicFactExtractor.aextract_from_text` each make one call — a system message of rules followed by the call-level `<output_language>` block and, only when the host passes some, a closing `<custom_instructions>` block, and a user message holding only the data: the numbered episodes (and, in UPDATE mode, the narrative being updated), or the text and its time. The reply is validated strictly — `{"content", "summary", "title"}` for reflect, `{"atomic_facts": {"atomic_fact"}}` for atomic facts — and a reply that fails raises `LLMOutputError` with no retry; blank fact sentences are dropped. `AtomicFactExtractor.aextract` keeps its single-message prompt, takes no `custom_instructions`, and is held to the same atomic-fact reply. None of the three takes a `prompt=` override.

`ForesightExtractor` makes one call at temperature 0.3 — a system message of rules (`FORESIGHT_SYSTEM_PROMPT`, then the call-level `<output_language>` block) and a user message holding only the data (`FORESIGHT_USER_PROMPT`: `<user_id>`, `<user_name>` and `<conversation>` elements, one `[YYYY-MM-DD HH:MM UTC (Weekday)] speaker: text` line per message, in the zone `tz=` names, UTC by default). The reply is validated strictly — `{"foresights": [{"content", "evidence", "start_time", "end_time", "duration_days"}]}` — and a reply that fails, or adds a key, raises `LLMOutputError` with no retry. The three time fields may be `null`: a missing `start_time` falls back to the date of the memcell's first message in that zone, and an `end_time` and a `duration_days` are each computed from the other; an `end_time` past year 9999 is dropped. Items with blank `content` are skipped and at most ten are kept.

`EpisodeExtractor` makes one call — a system message of rules (`EPISODE_SYSTEM_PROMPT`, the EverAlgo LoCoMo reference run's episode prompt sentence for sentence, then `<output_language>`: the language the host names, or the participant rule when it names none; and last `<custom_instructions>`: the host's text, or that run's five default principles) and a user message holding the data (`EPISODE_USER_PROMPT`: `<conversation_start_time>`, the first message's time, and `<conversation>`, each message with text as that run's pseudo-JSON block, every time as `2024-03-14 15:00 UTC (Thursday)` in the zone `tz=` names) and, after it, a static `<reminder>` to follow every rule, resolve every relative time and keep every fact — whatever `sender_id` is: the episode narrates the whole memcell, and `sender_id` only becomes `owner_id`. The reply is validated strictly against two fields in that run's order — `title`, `content` — and a reply that fails, adds a key such as `summary` or `language`, or leaves a field blank raises `LLMOutputError` with no retry. Named or not, the model declares no language: the Episode carries no `user_language`, so a host that needs the episode's language judges it from the text. `summary` is the content's first 200 characters.

`ProfileExtractor` accepts either chronological `MemCell` objects through `aextract` or dated generic/reflected `Episode` objects, in any order, through `aextract_from_episodes`. The Episode path resolves one target from non-blank `owner_name` or falls back to `owner_id`, skips narratives that do not contain that target, and raises before the first LLM call only when none contain it. Its optional `categories` argument is the complete current category snapshot for `explicit_info`: all four processing stages receive the same normalized list, select the most semantically accurate listed match, and may create a concise category when none fits; the list does not constrain `implicit_traits.trait`. Both paths use `old_profile=None` for INIT and an existing Profile for UPDATE and run compact and regroup maintenance transparently; the Episode path's COMPACT and REGROUP reply with operations on indexed items, so items they do not name keep their ids and dates. Neither path creates a new `implicit_traits` item, and an item carrying an identification or account number is dropped. On the Episode path every item the model writes must quote the batch's narratives, a habit or preference seen once waits on the profile's `pending` extra until a later batch repeats it, and matters the person asked not to be recorded are kept in its `do_not_record` extra; the host stores both extras with the profile and passes them back.

The merge is time-aware. Every profile item carries `observed_at`, the observation date of the narrative that last established it, and `Profile.timestamp` never moves backwards. An Episode observed before an item was established may add facts and evidence but cannot rewrite or delete that item — the rule is stated in the UPDATE prompt and enforced in code — so a backfilled older Episode arriving after a newer one cannot turn the profile back in time. Callers pass only the Episodes new to the profile; a batch that straddles the stored `Profile.timestamp` is split into a historical pass and a current pass. See the [Episode integration contract](../../docs/api/profile-from-episode-texts.md).

All class methods have a sync bridge: `extractor.extract(...)` is `async_to_sync(aextract)`, the Episode Profile method is exposed as `extract_from_episodes(...)`, and `EpisodeSummarizer.asummarize` / `asummarize_memcell` as `summarize(...)` / `summarize_memcell(...)` — only for non-event-loop callers (CLI scripts, plain unit tests).

## Testing

```python
from everalgo.testing import FakeLLMClient, assert_episode_shape

fake = FakeLLMClient(responses=[_EPISODE_JSON])
episode = await EpisodeExtractor(llm=fake).aextract(mc, sender_id="u_alice")
assert_episode_shape(episode)
```

See the integration test pattern in [`tests/integration/`](../../tests/integration/).

## Related distributions

- [`everalgo-boundary`](../everalgo-boundary/) — defines `BoundaryDetector`, `Cuts` and `to_memcells`
- [`everalgo-clustering`](../everalgo-clustering/) — geometry / LLM clustering for grouping MemCells before `ProfileExtractor`
- [`everalgo-rank`](../everalgo-rank/) — ranks `Episode`, `AtomicFact`, `Profile` candidates at read time
