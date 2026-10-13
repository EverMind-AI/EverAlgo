# Profile Design Notes

Why `ProfileExtractor` (`everalgo-user-memory`, the `everalgo/user_memory/profile/` package) works the way it does. The code's docstrings state the contract — what each function does, takes, returns and raises; this page keeps the reasoning behind it, the alternatives that were tried and dropped, and the numbers that decided them. For the upstream integration contract of the Episode path, see [`docs/api/profile-from-episode-texts.md`](../api/profile-from-episode-texts.md).

```
              INIT (no stored profile)                    UPDATE (stored profile)
  inputs ──▶ model writes both buckets ──▶ dedupe ──┐     inputs ──▶ model writes add / update / delete ops
                                                     │                         │
                                                     │               _apply_ops: collect → validate → apply → dedupe
                                                     │                         │
                                                     │          total > 60 ──▶ COMPACT (whole profile)
                                                     │          then a label > 8 items or an item > 250 units ──▶ REGROUP (that group)
                                                     ▼                         ▼
                                                  Profile ◀────────────────────┘
```

## 1. Admission and buckets

**Two buckets of atomic items.** `explicit_info` holds facts a source states about the owner; `implicit_traits` holds stable patterns inferred across sources. Each item is one fact (`ExplicitInfoItem` / `ImplicitTraitItem` in `everalgo.types`). The first design packed one prose blob per dimension; later facts were folded into the owner's blob, which is what let items concatenate without bound (production blobs reached 722 characters). Items are now atomic and a dimension holds as many as it needs.

**The label is a group key, not an identity.** `category` (explicit) and `trait` (implicit) name the dimension an item is filed under, and many items share one. The label is used to canonicalise spelling and to group items for the caps; it never merges two items' content.

**Grounding first in every reply.** The reply models declare `evidence` / `basis` before `description`, and the label last. A strict JSON schema lists properties in declaration order, so the model is shown the excerpt before the claim it grounds, and the label once the claim is stated. The schema fixes which keys an item has, not the order the model writes them in — in one live run 43 of 60 UPDATE items kept it — so every `<reply_format>` also asks for the fields in the order shown, grounding first. That raises the share; nothing in the request makes it certain. Stored items are rendered back to the model in the same order (`_FIELD_ORDER`) whatever order the host stored their keys in, so every item it reads looks like the item it is asked to write. The core item models declare required fields first, the usual pydantic order; the two orders are deliberately not coupled.

**Each path has its own reply models.** The MemCell path's prompts ask for an `update_note` / `compact_note` / `regroup_note` and the Episode path's do not, so each path validates against its own models. The MemCell models repeat the Episode ones' fields rather than extend them, because an output model inherits `StrictOutputModel` directly ([LLM contract §2.2](llm-contract.md)). Whether an UPDATE item matches the bucket its op names is a business check after parsing (§4.1): the right shape depends on another field of the same op.

**Profile items are typed.** `Profile.explicit_info` / `implicit_traits` used to ride on `extra="allow"` as bare dicts, which forced defensive code on every access (`getattr(...) or []`, `isinstance(item, dict)`, casts) and let a malformed item travel silently until a merge tripped over it. They are now models. `description` is required because it is an item's identity (section 3): an item without one can be neither deduplicated nor compared, and the profile now refuses it at construction instead of carrying it along. The labels and the grounding stay optional, since the merge has always tolerated items stored without them. Items keep `extra="allow"`, so a host's own keys (ids, provenance) ride through every merge untouched. They are never rendered to the model: an id or a provenance tag is not something the model can judge or should copy, every item it reads should look like the item it is asked to write, and each key it shows costs tokens on every call. Rendering keeps the `_FIELD_ORDER` fields only.

## 2. Merge semantics

**Collect, then apply.** `_apply_ops` validates every op against the snapshot the model was shown before applying any of them. Applied one at a time to a mutating list, a delete would shift the index of every op after it, and an add would widen the bound an out-of-range index is checked against; both landed edits on a neighbouring item, producing a self-consistent but wrong profile. Deletes are removed in one filtering pass for the same reason. Ops that fail validation are dropped with a warning rather than repaired.

**The bucket name is a whitelist.** An op's `type` is looked up among the known buckets, so a misspelt bucket is rejected; an earlier `if/else` routed anything unrecognised into `implicit_traits`.

**`update` is a partial merge.** The op's fields are merged over the stored item's, so two updates on one index accumulate rather than replace. The prompts nevertheless ask the model to restate the whole item, so the merge does not depend on omitted fields.

**`delete` beats `update` on the same index.** A delete answers an explicit negation, an expiry or a contradiction. Keeping a rewritten copy of something the user has disowned is worse than losing one correction.

**`observed_at` keeps the timeline.** Every item carries the observation date of the source that last established its `description`. Only the library writes it: the model is shown it as a date (`2024-04-10 14:00 UTC`), and a reply that writes it back fails validation. An update or delete coming from a source observed *before* the item was established is stale evidence and is refused — a later source already settled the current state — while an update that only adds grounding still lands, since more evidence for a fact is never wrong. Without the date, a backfilled older Episode arriving after a newer one overwrote the newer state, because the UPDATE prompt otherwise treats every new narrative as the latest word.

**History after the current value is not a rewrite.** An update from an older source whose description is the stored one followed by a parenthesis — `Lives in Chengdu (earlier in Chongqing).` over `Lives in Chengdu.` — lands, and the item keeps its `observed_at`: the current value is still the one the later source established, so the date that protects it does not move. Models write this form on their own when they carry an earlier value into evidence, and refusing the whole update threw that evidence away with it. The rule is mechanical, on the `_normalize`-folded descriptions: drop the stored description's closing `.` `!` `?` `。` `！` `？`; the new description must open with what is left, and what follows, leading spaces aside, must open with `(` or `（`. Any other change — a replaced or reworded current value, a negation put before it (`No longer lives in Chengdu`), a contradiction around it (`曾经养狗，现在不养了` over `养狗`), history without a parenthesis — is still stale evidence, and so is a delete. From a newer source the same extension is an ordinary rewrite and takes the new date. What the rule cannot see is the parenthesis itself: `住在成都（但已搬走）` over `住在成都` lands, though it retracts the current value. Telling history from a retraction takes reading it, which the merge does not do; the prompt asks for history only.

**One date per pass.** The Episode path splits a batch around the stored `Profile.timestamp` and runs the older narratives as a `historical` pass before the `current` one. Mixing them in one pass would let the newer narrative's date vouch for a rewrite the older one motivated, so the split is what gives the stale guard its meaning. Both passes send the same system message: the rule for older narratives is one the model applies per item, from the dates it is shown. Real-time ingestion never has a historical half; only backfill and out-of-order delivery do.

**Dates the model never wrote.** Items stored before `observed_at` existed take the profile's own timestamp, the newest knowledge they can be assumed to reflect. Items rewritten by COMPACT or REGROUP cannot be traced to the items they came from, so they take the profile's timestamp too, which keeps them protected from older narratives. Rendering and merging date items the same way (`_dated_buckets`): an item the model sees undated is one it cannot reason about, and one the guard dates differently from what the model saw is a rule the model could not have followed.

**Logs carry no content.** Every merge log line is `key=value` counts, dates, lengths and enum values, never a prompt, a reply or an item's text — labels included, since the model writes them. The line formats are listed in the module docstring and are a contract.

## 3. Dedupe and labels

**Identity is the description alone**, case- and whitespace-folded. The label stays out because the model writes it freely — the same fact lands under `diet` one round and `preferences` the next — and a drifting label would be a way around the dedupe. `evidence` / `basis` stay out as provenance: the same fact quoted from a second source is still the same fact. A blank description is no identity, so such an item is kept rather than matched against every other blank one.

**Dedupe runs over the whole bucket**, not only over new items, which covers three sources of duplicates: an add of something already stored, an update that rewrites one item into a copy of another, and duplicates already sitting in a profile written before the check existed. Compaction is the only other backstop and does not run until the total cap is passed, so an update is a polluted profile's one way back. The earliest copy survives, which keeps `summary` (the first description) stable.

**Add labels are respelled, never merged.** An add whose label matches one on file, ignoring case and whitespace, takes the stored spelling, so one dimension cannot split into variants of its own name. Near-synonym labels are left to the prompt, with compaction as the backstop.

**The UPDATE prompt lists the labels in use before the items.** Deciding add versus update is a comparison against the labels, and asking for it inside a wall of JSON is what let near-synonyms accumulate.

Measured: with the labels left implicit in the JSON, 8 conversations produced 8 explicit items, 86% of them one-off actions restated as standing capabilities, with one category repeated 7 times.

## 4. Caps: COMPACT and REGROUP

**Limits.** At most 60 items in total and 8 per label. Items are atomic, so a profile spans more items than the one-blob-per-dimension design did; 60 covers its 30 dimensions at two facts each, and the per-label cap stops one dimension flooding the profile. INIT never compacts: a profile that was just born has nothing accumulated to clean up, and a second call would double the latency on the longest inputs.

**Width backstop.** An item whose description is wider than 250 units sends its group to REGROUP. Width is measured with `ascii_width` (East Asian wide characters count 2), so one threshold holds across languages: 200 units is about 200 English characters or 100 CJK characters. The backstop guards against runaway concatenation only; the style rule ("one or two short sentences") lives in the prompt.

**COMPACT only on the total cap.** Rewriting the whole profile to fix one overcrowded group deletes valid items in groups that were fine. A per-label breach gets a REGROUP pass scoped to that group, whose legal moves are to split, rename and merge restatements — not to delete. COMPACT remains the only path that removes items that never belonged in the profile.

**REGROUP after COMPACT.** COMPACT is told the per-label cap but does not always meet it, so the per-label check runs on its result exactly as it runs on an UPDATE that stayed under the total cap. Returning COMPACT's result unchecked left a group over its cap until the next UPDATE, which may never come for a quiet profile. A call past the total cap therefore costs COMPACT plus one REGROUP per group COMPACT left over its cap.

**REGROUP never loops.** A reply that still breaches the cap is accepted with a warning. Items with a blank description in a REGROUP reply are dropped and logged; everything outside the group passes through untouched.

Measured:

- Full-profile compaction triggered by one noisy group dropped 6 of 13 gold facts.
- UPDATE emits deletes of never-in-the-profile items in about 1 run out of 10, so COMPACT is still needed for them.
- Compliant English items measured at p95 183–198 width units; 250 catches the runaway tail without policing compliant items, while 200 would sit on the style budget and churn REGROUP on legitimate English items.

## 5. Language

`output_language` is a closed set (`OutputLanguage`), honoured by every stage. Passed on every call, it fixes a profile's language for its whole life with no inference anywhere. Left `None`, each stage falls back to its own judgement: INIT follows the participants, while UPDATE, COMPACT and REGROUP keep the language most of the profile is already written in, so a profile that is already mixed has one language to converge on. Inheriting is safe against a later conversation in another language splitting a profile in half, but cannot recover a profile whose language is already wrong; naming a language on an update is how such a profile is corrected.

A profile is written in one language. UPDATE replies with operations, so a stored item no narrative touches never passes through the model; `<how_to_update>` therefore asks for every stored item written in another language than the one `<output_language>` gives to be updated into it, nothing else changed, even when no narrative mentions it. COMPACT and REGROUP return every item they keep, so the language rule alone covers them. Accepted limits: a translating update changes the description, which the merge cannot tell from a rewrite, so it moves the item's `observed_at` to the batch's time; in a backfill batch, older than the item, the same update is refused as a stale rewrite and the mix waits for the next current batch; and the first UPDATE after a host starts naming a language may rewrite every stored item (at most the total cap of 60 items), so the host's reply budget has to allow for a full rewrite.

It is a closed set rather than free text because `custom_instructions` already reaches the model as an instruction: the host's own text, appended after the bundled rules, which cannot replace them or the output shape. Whatever a host forwards into it is the host's to vet.

Measured: INIT left to infer the language wrote the wrong one 18% of the time across four models, against 0% when the language was named. The per-path measurements are in `everalgo/user_memory/prompts/en/_language.py`.

## 6. Caching

Prompt caches match on a prefix, so the Episode path's system messages are ordered from least to most variable and keep per-request data out: the role and rules first, then `<output_language>`, `<available_categories>` and, when passed, `<additional_requirements>`. The system message of a stage is therefore identical across owners and requests that share a language, and every block before `<additional_requirements>` stays a shared prefix. Both UPDATE passes send the same system message for the same reason. The host's instructions go in as text, never rendered as a template, so braces in them stay literal. The MemCell prompts are a single user message and keep their layout.

## 7. What the model is shown

**Closing tags, not escaping.** The Episode path's user messages wrap their data in XML elements, but the text inside is not XML-escaped: escaping would show the model `&lt;` and `&amp;` in place of what the person wrote, in every narrative and item, to guard against a case that almost never occurs. The one real hazard is a narrative or item that spells out one of the prompt's own closing tags and so ends its element early. Exactly those strings (`</narrative>`, `</episode_narratives>`, `</item>`, `</explicit_info>`, `</implicit_traits>`, `</stored_profile>`, `</group>`, in any case) get a fullwidth `＜` (U+FF1C) for their `<`, which the model still reads as the same text; all other markup passes as written. The MemCell path shares the item renderer, so its stored items get the same treatment.
