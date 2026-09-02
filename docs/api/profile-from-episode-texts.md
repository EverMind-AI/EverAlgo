# Profile Extraction from Episode Texts

Status: implemented in EverAlgo; ready for EverOS integration.

This document defines the contract between EverOS and EverAlgo for extracting one user-owned `Profile` from a chronological list of generic or reflected Episode narratives. It is intentionally text-based: EverOS keeps storage identifiers and orchestration metadata, while EverAlgo receives only the narrative content and the explicit target-user context required for profile extraction.

## API

```python
from collections.abc import Sequence

from asgiref.sync import async_to_sync

from everalgo.types import Profile
from everalgo.user_memory import OutputLanguage


class ProfileExtractor:
    async def aextract_from_episode_texts(
        self,
        episode_texts: Sequence[str],
        *,
        owner_id: str,
        timestamp: int,
        owner_name: str | None = None,
        old_profile: Profile | None = None,
        categories: Sequence[str] | None = None,
        prompt: str | None = None,
        output_language: OutputLanguage | str | None = None,
    ) -> Profile:
        ...

    extract_from_episode_texts = async_to_sync(aextract_from_episode_texts)
```

`aextract_from_episode_texts` is the native asynchronous API. `extract_from_episode_texts` is the synchronous bridge for callers that are not running an event loop. It must not be called from a running event loop.

The existing [`ProfileExtractor.aextract`](../../packages/everalgo-user-memory/src/everalgo/user_memory/profile.py) API remains unchanged. The two methods serve different inputs:

| Method | Input | Target resolution |
|---|---|---|
| `aextract` | Chronological `Sequence[MemCell]` | Validates `sender_id` against structured user messages |
| `aextract_from_episode_texts` | Chronological `Sequence[str]` of Episode narratives | Resolves one target reference from `owner_name` or `owner_id`, then selects narratives that contain it |

## Parameters

### `episode_texts`

A non-empty chronological sequence of non-blank Episode narrative bodies. Each item may be either a generic Episode narrative or a reflected Episode narrative.

EverAlgo uses non-blank `owner_name` when available and falls back to `owner_id` otherwise. It excludes items that do not contain the resolved target reference, while preserving the relative order of the remaining items. At least one item must contain the target.

The sequence must be ordered from oldest to newest. EverOS must deduplicate Episodes before converting them to strings because this API deliberately does not receive Episode IDs.

### `owner_id`

The authoritative identifier of the user whose Profile is being extracted. EverAlgo must copy this value to `Profile.owner_id`; it must never infer or replace the identifier from model output.

`owner_id` must be non-blank. In UPDATE mode, it must equal `old_profile.owner_id`.

### `owner_name`

The optional authoritative display name used to locate the target user inside Episode narratives. EverOS supplies it from trusted participant metadata when available; EverAlgo must not infer it from the narrative. `None` or a blank value makes EverAlgo use `owner_id` as the target reference.

The name is a target-selection hint only. The returned Profile is owned by `owner_id`, not by `owner_name`.

### `timestamp`

The Unix epoch timestamp in milliseconds assigned to the returned `Profile`. EverOS should pass the maximum timestamp of the deduplicated Episode batch.

This argument is explicit because `Sequence[str]` carries no timestamp metadata.

### `old_profile`

- `None` selects INIT mode and creates a new Profile.
- A `Profile` selects UPDATE mode and applies additions, updates, and deletions to that Profile.

The existing transparent compaction behavior is preserved after UPDATE.

### `categories`

The complete category snapshot currently available for `explicit_info.category`. The caller is responsible for assembling the whole snapshot before the call; EverAlgo does not distinguish category sources or lifecycle states.

EverAlgo strips each string, ignores blank values, removes exact duplicates in first-seen order, and injects the resulting JSON list unchanged into INIT, UPDATE, COMPACT, and REGROUP. `None` and an empty sequence both render as `[]`. Passing a string instead of a sequence, or including a non-string element, raises `TypeError` before the LLM call.

For every explicit fact processed by a stage, the model must choose the most semantically accurate matching category from this list. The list is not a whitelist: when no listed category accurately fits, the model may create a necessary, concise, semantically accurate category. Category reuse and category-count reduction must never override classification accuracy. This list does not constrain `implicit_traits.trait`.

### `prompt`

An optional prompt override. `None` selects the bundled Episode-text Profile prompt. Input validation and output ownership rules still apply when a custom prompt is used.

### `output_language`

The requested output language as an `OutputLanguage` value or equivalent case-insensitive string. `None` lets INIT follow the Episode narrative language and lets UPDATE preserve the existing Profile language.

## Target-owner validation

Validation and target selection finish before the first LLM call. EverAlgo checks every Episode text independently; checking only the concatenated batch is not sufficient because a narrative without the target must not reach the prompt.

EverAlgo resolves the target reference once before validating the texts:

```python
target_user = owner_name.strip() if owner_name and owner_name.strip() else owner_id.strip()
```

For each `episode_texts[index]`:

1. Reject the call if any item is blank or not a string.
2. Normalize each text and `target_user` consistently for comparison.
3. Retain the item if it contains `target_user` as a literal target reference.
4. Skip the item otherwise, without passing its body to the LLM.
5. After checking the full sequence, reject the call with `ValueError` only if no items remain.

Structural failures identify the invalid list index. An all-unmatched failure identifies the target value. Neither errors nor logs include Episode bodies.

Example error:

```text
no episode_texts reference target user 'Alice'
```

This selection is a deterministic safety guard, not identity proof. It prevents unmatched narratives from influencing extraction but cannot disambiguate two people with the same display name. EverOS remains responsible for supplying authoritative owner metadata and owner-scoped Episode batches.

## Extraction behavior

The method performs exactly one owner-scoped Profile extraction for the supplied `owner_id`. A single generic Episode may be fanned out upstream to multiple owners, but EverOS must call this method separately for each owner because each owner receives a different Profile interpretation.

The returned Profile must satisfy all of the following:

- `Profile.owner_id == owner_id`.
- `Profile.timestamp == timestamp`.
- The target owner's facts and traits are not mixed with those of other participants.
- Factual correctness and `explicit_info.category` accuracy are co-equal highest priorities across INIT, UPDATE, COMPACT, and REGROUP.
- Unsupported, misattributed, transient, expiring, question-only, or generally applicable team/organisation content is excluded even when retaining it would improve recall.
- INIT and UPDATE use the same merge and compaction semantics as `aextract`.
- No partial Profile is returned when validation or extraction fails.

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

The Episode-text path does not have access to raw conversation turns. Its evidence contract therefore differs from the `MemCell` path:

- `explicit_info[].evidence` contains a verifiable narrative excerpt or faithful paraphrase from the supplied Episode texts.
- Evidence does not require an Episode number or synthetic identifier.
- Evidence must not be presented as a direct user quotation unless the supplied Episode text itself contains that quotation.
- The model must not invent user wording, dates, or speaker attribution.
- `implicit_traits[].basis` names signals that can be located in the supplied Episode texts and must not invent user quotations.
- Both `evidence` and `basis` are scalar JSON strings, not arrays; faithful narrative paraphrases are allowed when they preserve the source meaning and attribution.

EverOS must treat this evidence as Episode-derived evidence, not raw-conversation evidence.

## Exceptions

The method raises `ValueError` before calling the LLM when:

- `episode_texts` is empty;
- an Episode text is blank;
- `owner_id` is blank;
- no Episode text contains the resolved target reference;
- `old_profile.owner_id` differs from `owner_id`; or
- `output_language` is unsupported.

It raises `TypeError` before calling the LLM when `categories` is not a sequence of strings or contains a non-string value.

The method also preserves the existing Profile extraction errors:

- `ValueError` when the LLM response violates the Profile schema;
- `json.JSONDecodeError` when the LLM response is not parseable JSON; and
- `LLMError` when the injected LLM client fails.

Validation failures are caller-data errors and should not be retried with the same input. LLM transport failures and malformed model responses may follow the upstream retry policy.

## EverOS responsibilities

Before calling EverAlgo, EverOS must:

1. Select Episodes belonging to the target owner scope.
2. Deduplicate them by the upstream Episode identifier.
3. Sort them by timestamp from oldest to newest.
4. Convert each Episode to its narrative body only.
5. Resolve the authoritative `owner_id` and, when available, `owner_name` from participant metadata.
6. Supply the exact `owner_name` when one is available, or rely on `owner_id` otherwise; EverAlgo filters narratives that do not contain the resolved value.
7. Pass the maximum Episode timestamp as `timestamp`.
8. Load the existing owner Profile, if any, and pass it as `old_profile`.
9. Assemble and pass the complete current `explicit_info` category snapshot as `categories`, or pass `None` when no category is currently available.
10. Persist the returned Profile and handle retries; EverAlgo remains stateless.

The current EverOS Episode event already carries the Episode identifier, narrative text, timestamp, and `owner_id`. It does not carry `owner_name`, so the upstream integration should resolve the name from participant metadata when available and otherwise rely on the `owner_id` fallback.

## Async example

```python
from everalgo.user_memory import OutputLanguage, ProfileExtractor


episode_records = await episode_repository.list_for_owner(owner_id="user-123")
deduplicated = deduplicate_by_entry_id(episode_records)
ordered = sorted(deduplicated, key=lambda episode: episode.timestamp_ms)

profile = await ProfileExtractor(llm=llm).aextract_from_episode_texts(
    [episode.text for episode in ordered],
    owner_id="user-123",
    owner_name="Alice",
    timestamp=max(episode.timestamp_ms for episode in ordered),
    old_profile=existing_profile,
    categories=available_profile_categories,
    output_language=OutputLanguage.CHINESE,
)
```

When language selection is intentionally delegated to the model, omit `output_language` or pass `None`:

```python
profile = await ProfileExtractor(llm=llm).aextract_from_episode_texts(
    episode_texts,
    owner_id=owner_id,
    owner_name=owner_name,
    timestamp=latest_timestamp_ms,
    old_profile=existing_profile,
    categories=None,
    output_language=None,
)
```

## Integration acceptance criteria

The EverOS integration is ready when all of the following are covered by tests:

- INIT succeeds when `owner_name` is supplied and every text contains that name.
- INIT succeeds when `owner_name` is absent and every text contains `owner_id`.
- UPDATE succeeds when `old_profile.owner_id` matches `owner_id`.
- `categories=None`, an empty list, whitespace-only values, and exact duplicates have deterministic documented rendering.
- A non-string category value fails before LLM invocation.
- INIT, UPDATE, COMPACT, and REGROUP receive the same normalized category snapshot.
- A matching available category is selected by semantic accuracy, while a necessary category may be created when no listed category fits.
- The category snapshot does not constrain `implicit_traits.trait`.
- Narratives that lack the resolved target reference are absent from the LLM prompt, while matching narratives retain their input order.
- The call fails before LLM invocation when every text lacks the resolved target reference.
- The call fails before LLM invocation for a mismatched existing Profile owner.
- Duplicate Episode IDs are removed upstream before the text list is built.
- Texts are passed in chronological order and `timestamp` equals the newest Episode timestamp.
- Two owners fanned out from one generic Episode are extracted in separate calls and produce separately owned Profiles.
- Evidence is traceable to Episode narratives and contains no fabricated user quotation.

## Why this interface

Accepting `Sequence[str]` keeps EverAlgo independent of EverOS persistence models and matches the stateless algorithm boundary. `owner_id` always controls Profile ownership, while optional `owner_name` locates the person in model-written narratives that may not contain IDs. Resolving one target reference and filtering independently before extraction prevents unrelated narratives from influencing the Profile without discarding a usable batch because of one unrelated Episode; rejecting an all-unmatched batch still prevents extraction without owner evidence. Accepting the current category snapshot separately keeps classification policy caller-owned while allowing the stateless extraction and maintenance stages to apply one consistent semantic rule.
