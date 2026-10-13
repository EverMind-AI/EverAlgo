# Profile Extraction from Episodes

Status: implemented in EverAlgo; ready for EverOS integration.

This document defines the contract between EverOS and EverAlgo for extracting one user-owned `Profile` from dated generic or reflected Episodes. EverOS keeps storage identifiers and orchestration metadata and decides *which* Episodes are new to the profile; EverAlgo receives those Episodes with their observation dates, keeps the profile's timeline, and decides *what* each one may change.

## API

```python
from collections.abc import Sequence

from asgiref.sync import async_to_sync

from everalgo.types import Episode, Profile
from everalgo.user_memory import OutputLanguage


class ProfileExtractor:
    async def aextract_from_episodes(
        self,
        episodes: Sequence[Episode],
        *,
        owner_id: str,
        owner_name: str | None = None,
        old_profile: Profile | None = None,
        categories: Sequence[str] | None = None,
        custom_instructions: str | None = None,
        output_language: OutputLanguage | str | None = None,
    ) -> Profile:
        ...

    extract_from_episodes = async_to_sync(aextract_from_episodes)
```

`aextract_from_episodes` is the native asynchronous API. `extract_from_episodes` is the synchronous bridge for callers that are not running an event loop. It must not be called from a running event loop.

The existing [`ProfileExtractor.aextract`](../../packages/everalgo-user-memory/src/everalgo/user_memory/profile/_extractor.py) API remains unchanged. The two methods serve different inputs:

| Method | Input | Target resolution |
|---|---|---|
| `aextract` | Chronological `Sequence[MemCell]` | Validates `sender_id` against structured user messages |
| `aextract_from_episodes` | `Sequence[Episode]` in any order | Resolves one target reference from `owner_name` or `owner_id`, then selects Episodes whose narrative contains it |

## Parameters

### `episodes`

A non-empty sequence of `everalgo.types.Episode` values, each with a non-blank `episode` narrative and a `timestamp` (Unix epoch milliseconds) giving the observation date of what it narrates. Each item may be a generic Episode or a reflected Episode. Order does not matter: EverAlgo sorts the selected Episodes by `timestamp` before rendering them.

EverAlgo uses non-blank `owner_name` when available and falls back to `owner_id` otherwise. It excludes Episodes whose narrative does not contain the resolved target reference. At least one Episode must contain it.

Pass only the Episodes that are new to the stored profile since the previous call (see [EverOS responsibilities](#everos-responsibilities)). EverOS must deduplicate Episodes before the call; this API does not receive storage identifiers.

### `owner_id`

The authoritative identifier of the user whose Profile is being extracted. EverAlgo must copy this value to `Profile.owner_id`; it must never infer or replace the identifier from model output.

`owner_id` must be non-blank. In UPDATE mode, it must equal `old_profile.owner_id`.

### `owner_name`

The optional authoritative display name used to locate the target user inside Episode narratives. EverOS supplies it from trusted participant metadata when available; EverAlgo must not infer it from the narrative. `None` or a blank value makes EverAlgo use `owner_id` as the target reference.

The name is a target-selection hint only. The returned Profile is owned by `owner_id`, not by `owner_name`.

### `old_profile`

- `None` selects INIT mode and creates a new Profile.
- A `Profile` selects UPDATE mode and applies additions, updates, and deletions to that Profile.

`old_profile.timestamp` is read as the observation date the stored profile reflects; see [Time-aware merge](#time-aware-merge). After UPDATE the profile is brought back inside its caps in the same call: past 60 items in total it is compacted, and then every group holding more than 8 items under one category or trait name is regrouped. The group check reads COMPACT's result when COMPACT ran, so a group COMPACT leaves over its cap is regrouped too.

Both buckets hold typed items from `everalgo.types`:

| Field | `explicit_info[]` — `ExplicitInfoItem` | `implicit_traits[]` — `ImplicitTraitItem` |
|---|---|---|
| `description` | Required. The fact, one short sentence; its folded form is the item's identity. | Required. The pattern, one short sentence; same identity rule. |
| label | `category: str \| None` — the dimension the fact is filed under. | `trait: str \| None` — the trait label. |
| grounding | `evidence: str \| None` — source excerpt or faithful paraphrase. | `basis: str \| None` — the signals the inference rests on. |
| `observed_at` | `int \| None`, epoch ms; written by EverAlgo, see [Time-aware merge](#time-aware-merge). | Same. |
| any other key | Kept untouched through every merge (`extra="allow"`), and never shown to the model. | Same. |

A host that stores items as documents dumps them with `model_dump()` — `model_dump(exclude_unset=True)` leaves out fields that were never set — and passes them back through `Profile.model_validate`. An item with no `description`, or one that is not an object, fails that validation.

### `categories`

The complete category snapshot currently available for `explicit_info.category`. The caller is responsible for assembling the whole snapshot before the call; EverAlgo does not distinguish category sources or lifecycle states.

EverAlgo strips each string, ignores blank values, removes exact duplicates in first-seen order, and injects the resulting JSON list unchanged into INIT, UPDATE, COMPACT, and REGROUP. `None` and an empty sequence both render as `[]`. Passing a string instead of a sequence, or including a non-string element, raises `TypeError` before the LLM call.

The prompt defines a category as the dimension a fact belongs to, named after the fact itself rather than the topic of the narrative it appeared in, and asks the model to prefer a category from this list; when there is no list or no listed category fits, the model coins a short dimension name. The list is therefore a preference, not a whitelist, and it does not constrain `implicit_traits.trait`.

### `custom_instructions`

Optional instructions from the host, appended to the bundled rules of every stage the call runs — INIT or UPDATE, and any COMPACT or REGROUP that follows. The text of the parameter `custom_instructions` enters the system message as an `<additional_requirements>` block that closes it, after `<available_categories>`, opened by a fixed lead-in saying the text applies on top of the bundled rules, leaves the reply shape in `<reply_format>` as it is, and yields to the bundled rules where the two conflict — so it can tighten them but not loosen them. The text is sent verbatim and is never rendered as a template, so braces in it stay literal.

`None` or a blank string adds nothing, and the system message is exactly the bundled one. The parameter cannot replace or remove a bundled rule, and the output shape stays fixed; input validation, output ownership, and the time-aware merge rules are enforced in code either way.

The text is the host's own, not an end user's. It reaches the model as an instruction — the same reason `output_language` accepts only a closed set of language names rather than free text — so the host is responsible for anything it forwards into it. Placing the block last keeps every block before it a shared cache prefix; only the block itself varies with the text.

### `output_language`

The requested output language as an `OutputLanguage` value or equivalent case-insensitive string. `None` lets INIT follow the Episode narrative language and lets UPDATE, COMPACT and REGROUP keep the language most of the existing Profile is written in, so a Profile that is already mixed converges on one language. A named language applies to every field in every stage, whatever language the input is in, including anything already stored: it takes precedence over a stored Profile written in another language, and the UPDATE prompt asks for a stored item written in another language to be rewritten in it even when no narrative mentions that item.

## Target-owner validation

Validation and target selection finish before the first LLM call. EverAlgo checks every Episode independently; checking only the concatenated batch is not sufficient because a narrative without the target must not reach the prompt.

EverAlgo resolves the target reference once before validating the Episodes:

```python
target_user = owner_name.strip() if owner_name and owner_name.strip() else owner_id.strip()
```

For each `episodes[index]`:

1. Reject the call with `TypeError` if the item is not an `Episode`, and with `ValueError` if its narrative is blank.
2. Normalize the narrative and `target_user` consistently for comparison.
3. Retain the Episode if its narrative contains `target_user` as a literal target reference.
4. Skip the Episode otherwise, without passing its narrative to the LLM.
5. After checking the full sequence, reject the call with `ValueError` only if no Episodes remain.

Structural failures identify the invalid list index. An all-unmatched failure identifies the target value. Neither errors nor logs include Episode bodies.

Example error:

```text
no episodes reference target user 'Alice'
```

This selection is a deterministic safety guard, not identity proof. It prevents unmatched narratives from influencing extraction but cannot disambiguate two people with the same display name. EverOS remains responsible for supplying authoritative owner metadata and owner-scoped Episode batches.

## Time-aware merge

A profile is a portrait of the person *now*. EverOS may deliver Episodes out of observation order — a backfilled history, a device syncing old conversations — so the merge cannot treat every new narrative as the latest word. EverAlgo keeps two dates and enforces one rule.

**Dates**

- `Profile.timestamp` is the newest observation date the profile reflects. INIT sets it to the newest selected Episode; UPDATE sets it to `max(old_profile.timestamp, newest Episode in the batch)`. It never moves backwards.
- Every `explicit_info` and `implicit_traits` item carries `observed_at` (Unix epoch milliseconds): the observation date of the narrative that last established its `description`. INIT stamps every item with the batch's newest date. UPDATE stamps every added item and every rewritten description with the batch's newest date; an update that only changes grounding (`evidence` / `basis`) or the category keeps the item's date. Items rewritten by COMPACT or REGROUP take `Profile.timestamp`. Items stored before this field existed are read as `old_profile.timestamp`.

`observed_at` is written by EverAlgo only. The model sees it inside each stored item's JSON as a UTC minute (`2024-04-10 14:00 UTC`), the same shape as each narrative's `time` attribute; a reply whose operation writes `observed_at` fails validation and raises `LLMOutputError`.

**Rule**

An Episode observed *before* an item's `observed_at` is older knowledge about the person. It may add a fact not yet on file and may add evidence to an existing item, but it must not rewrite that item's description or delete it — a later Episode already established the current state. The UPDATE prompt states this rule — including that an earlier value of a dimension that already holds one goes into that item's evidence, never into a new item, and that the description may add that history in a short parenthesis after the current value — and `_apply_ops` enforces it: a delete, or an update that changes the description other than by adding a parenthesis after it, whose batch is older than the target item is dropped and logged with `reason=stale evidence`. An older batch's update whose description is the stored one followed by a parenthesis (compared case- and whitespace-folded, the stored description's closing `.` `!` `?` `。` `！` `？` set aside), such as `Lives in Chengdu (earlier in Chongqing).` over `Lives in Chengdu.`, lands: its description and evidence are taken, and the item keeps its `observed_at`, since the later Episode still holds the current value.

**Two passes**

The rule is judged per batch, against the batch's newest date. A batch that straddles `old_profile.timestamp` is therefore split: Episodes observed before it run first as a historical pass (they can only fill gaps), then Episodes observed after it run as a normal pass. Both passes send the same system message; the model applies the rule per item from the times it is shown, so the split exists only to give each pass one observation date. Real-time ingestion never has a historical half, so it pays no extra LLM call; only backfill and out-of-order delivery do, and only when one batch contains both kinds.

## Extraction behavior

The method performs exactly one owner-scoped Profile extraction for the supplied `owner_id`. A single generic Episode may be fanned out upstream to multiple owners, but EverOS must call this method separately for each owner because each owner receives a different Profile interpretation.

The returned Profile must satisfy all of the following:

- `Profile.owner_id == owner_id`.
- `Profile.timestamp` equals the newest selected Episode's `timestamp` in INIT, and `max(old_profile.timestamp, newest selected Episode)` in UPDATE.
- Every item carries an integer `observed_at` as defined above.
- No item established by a later Episode is rewritten or deleted on the strength of an earlier one.
- The target owner's facts and traits are not mixed with those of other participants.
- A single action (or one restated as a capability, skill, duty, interest or attitude), a passing state or what the person is working on right now, a rule that would hold for whoever sat in their seat, a decision taken once for one project, a single question, anything the assistant or another participant said, a fact about anyone other than the owner even when the owner tells it or it sits inside an item about the relationship (a relationship enters only as a standing feature of the owner's own life, such as a partner or someone they live with, and carries nothing about the other person; a bare relationship is not an item), and a one-off request about format or procedure never become items; an item whose attribution, durability or grounding is uncertain is left out.
- An `explicit_info` item is a fact a narrative explicitly attributes to the owner. An `implicit_traits` item is a stable pattern no single narrative states but several show, and its `basis` names the concrete signals — choices or requests the narratives record, not operations the owner carried out.
- INIT and UPDATE use the same merge and compaction semantics as `aextract`.
- No partial Profile is returned when validation or extraction fails.

The INIT and UPDATE system messages show where these lines fall with worked examples, in an `<examples>` block right after `<never_in_the_profile>`: INIT two and UPDATE one, each in its stage's reply shape and about made-up people. COMPACT and REGROUP carry none.

## Description style

Every `explicit_info[].description` and `implicit_traits[].description` must be a concise subjectless declarative sentence. It describes the owner directly without naming or substituting for the subject, and it must not be written as an imperative instruction.

Correct:

- `Works mainly in Python.`
- `Prefers concise, direct answers.`

Incorrect:

- `Alice works mainly in Python.` — names the subject.
- `The user prefers concise answers.` — substitutes a generic subject.
- `Use Python.` — changes a profile statement into an instruction.

## Evidence contract

The Episode path does not have access to raw conversation turns. Its evidence contract therefore differs from the `MemCell` path:

- `explicit_info[].evidence` contains a verifiable narrative excerpt or faithful paraphrase from the supplied Episode narratives.
- Evidence does not require an Episode number or synthetic identifier. Each narrative is shown to the model as `<narrative time="YYYY-MM-DD HH:MM UTC">…</narrative>`, its observation time in the attribute; the model may cite that date but must not invent others. Narrative text is not XML-escaped: only a literal closing tag of the prompt's own elements (`</narrative>`, `</episode_narratives>`, `</item>`, `</explicit_info>`, `</implicit_traits>`, `</stored_profile>`, `</group>`, in any case) has its `<` replaced by the fullwidth `＜` (U+FF1C), in narratives and in stored items alike, so it cannot close an element early.
- Evidence must not be presented as a direct user quotation unless the supplied Episode narrative itself contains that quotation.
- The model must not invent user wording, dates, or speaker attribution.
- `implicit_traits[].basis` names the signals behind the trait — which choices or requests, in which narratives — and must not invent user quotations; restating the requirement ("several signals") is not a basis.
- Both `evidence` and `basis` are scalar JSON strings, not arrays; faithful narrative paraphrases are allowed when they preserve the source meaning and attribution.

EverOS must treat this evidence as Episode-derived evidence, not raw-conversation evidence.

## Exceptions

The method raises `TypeError` before calling the LLM when:

- an item of `episodes` is not an `Episode`; or
- `categories` is not a sequence of strings or contains a non-string value.

The method raises `ValueError` before calling the LLM when:

- `episodes` is empty;
- an Episode narrative is blank;
- `owner_id` is blank;
- no Episode narrative contains the resolved target reference;
- `old_profile.owner_id` differs from `owner_id`; or
- `output_language` is unsupported.

The method raises `LLMOutputError` (from `everalgo.llm`) when an LLM reply is not a valid output of its stage: not parseable JSON, not the INIT, UPDATE, COMPACT or REGROUP shape, or an item shaped for the other bucket (an UPDATE operation whose `data` does not match its `type`, a REGROUP item not shaped for its group's bucket). Its message names each failing location, never the reply's content.

Exceptions raised by the injected LLM client propagate unchanged.

Validation failures are caller-data errors and should not be retried with the same input. LLM transport failures and malformed model responses may follow the upstream retry policy.

## EverOS responsibilities

Before calling EverAlgo, EverOS must:

1. Select the Episodes belonging to the target owner scope that the stored profile has **not yet consumed** — for example those written after a per-profile watermark. Passing the whole history on every call re-spends the prompt on facts the profile already reflects; passing only the newest few by observation date silently drops backfilled older ones.
2. Deduplicate them by the upstream Episode identifier.
3. Map each to an `everalgo.types.Episode` with its narrative and its observation `timestamp`. Order is irrelevant.
4. Resolve the authoritative `owner_id` and, when available, `owner_name` from participant metadata.
5. Supply the exact `owner_name` when one is available, or rely on `owner_id` otherwise; EverAlgo filters narratives that do not contain the resolved value.
6. Load the existing owner Profile, if any, and pass it as `old_profile`, items and `observed_at` values included.
7. Assemble and pass the complete current `explicit_info` category snapshot as `categories`, or pass `None` when no category is currently available.
8. Persist the returned Profile — `timestamp` and per-item `observed_at` included, since the next call reads them — and advance the watermark only after the persist succeeds; EverAlgo remains stateless.

The current EverOS Episode row already carries the Episode identifier, narrative text, observation timestamp, and `owner_id`. It does not carry `owner_name`, so the upstream integration should resolve the name from participant metadata when available and otherwise rely on the `owner_id` fallback.

## Async example

```python
from everalgo.types import Episode
from everalgo.user_memory import OutputLanguage, ProfileExtractor


rows = await episode_repository.list_unconsumed_for_owner(owner_id="user-123", after=existing_profile_watermark)
episodes = [
    Episode(owner_id=row.owner_id, episode=row.text, subject=row.subject, summary=row.summary, timestamp=row.timestamp_ms)
    for row in deduplicate_by_entry_id(rows)
]

profile = await ProfileExtractor(llm=llm).aextract_from_episodes(
    episodes,
    owner_id="user-123",
    owner_name="Alice",
    old_profile=existing_profile,
    categories=available_profile_categories,
    output_language=OutputLanguage.CHINESE,
)
```

When language selection is intentionally delegated to the model, omit `output_language` or pass `None`:

```python
profile = await ProfileExtractor(llm=llm).aextract_from_episodes(
    episodes,
    owner_id=owner_id,
    owner_name=owner_name,
    old_profile=existing_profile,
    categories=None,
    output_language=None,
)
```

## Integration acceptance criteria

The EverOS integration is ready when all of the following are covered by tests:

- INIT succeeds when `owner_name` is supplied and every narrative contains that name.
- INIT succeeds when `owner_name` is absent and every narrative contains `owner_id`.
- UPDATE succeeds when `old_profile.owner_id` matches `owner_id`.
- `categories=None`, an empty list, whitespace-only values, and exact duplicates have deterministic documented rendering.
- A non-string category value fails before LLM invocation.
- INIT, UPDATE, COMPACT, and REGROUP receive the same normalized category snapshot.
- A matching available category is selected by semantic accuracy, while a necessary category may be created when no listed category fits.
- The category snapshot does not constrain `implicit_traits.trait`.
- Narratives that lack the resolved target reference are absent from the LLM prompt; matching narratives appear oldest first, each with its observation time as an attribute.
- The call fails before LLM invocation when every narrative lacks the resolved target reference.
- The call fails before LLM invocation for a mismatched existing Profile owner.
- Duplicate Episode IDs are removed upstream before the batch is built.
- Only Episodes not yet consumed by the stored profile are passed, and the watermark advances only after the returned Profile is persisted.
- The persisted Profile round-trips `timestamp` and every item's `observed_at` unchanged into the next call's `old_profile`.
- An Episode observed before the stored state adds facts but leaves later-established items' descriptions in place, and `Profile.timestamp` does not move backwards.
- Two owners fanned out from one generic Episode are extracted in separate calls and produce separately owned Profiles.
- Evidence is traceable to Episode narratives and contains no fabricated user quotation.
- `custom_instructions`, when passed, is host-authored text and never forwards end-user input.

## Why this interface

Accepting `Sequence[Episode]` rather than bare strings keeps each narrative's observation date attached to it, which is what lets the merge tell older knowledge from newer state; the core `Episode` type already exists, so no new type is needed. `owner_id` always controls Profile ownership, while optional `owner_name` locates the person in model-written narratives that may not contain IDs. Resolving one target reference and filtering independently before extraction prevents unrelated narratives from influencing the Profile without discarding a usable batch because of one unrelated Episode; rejecting an all-unmatched batch still prevents extraction without owner evidence. The temporal rule lives in code rather than only in the prompt because it is deterministic: a model that ignores the instruction still cannot turn the profile back in time. Deciding *which* Episodes are new stays with EverOS, which owns the storage and the watermark; deciding *what* those Episodes may change stays with EverAlgo, which owns the merge. Accepting the current category snapshot separately keeps classification policy caller-owned while allowing the stateless extraction and maintenance stages to apply one consistent semantic rule. `custom_instructions` appends rather than replaces: a replacement also discards the rules that define the output shape and what may enter the profile, while an appended block leaves every bundled rule — and the shared cache prefix — intact.
