"""The output-language rules every extraction prompt in this package splices in.

A prompt receives ``CALLER_CHOSEN_LANGUAGE_RULE`` when the caller named a language, and otherwise the
fallback its own operator picks: the participant rule for the operators reading a raw conversation, and a
rule of its own for each operator whose input is already-extracted memory text — those inherit a language
rather than judge one, and each names the specific input it inherits from. Two of the texts below,
``CALLER_CHOSEN_LANGUAGE_RULE`` and ``PARTICIPANT_LANGUAGE_RULE``, were selected by measurement rather than
judgement: eight arms over the same 0.3.2 episode prompt body, 1750 calls each across five models and
thirty-five conversations, differing only in this block. The inheriting rules were not measured — they
predate the argument and are carried over verbatim from the prompts that already used them.

Every rule here is spliced in twice, at the prompt's head and tail, so none of them may refer to its own
position in the prompt — a sentence pointing above or below is wrong at one of the two ends.

``CALLER_CHOSEN_LANGUAGE_RULE`` names the language outright and leaves the model nothing to decide. Zero
drift over 1400 samples spanning seven languages, every interference pattern in the corpus, and all five
models — no scenario, no model, and no language deviated.

``PARTICIPANT_LANGUAGE_RULE`` asks the model to work the language out. That costs roughly one wrong
language in thirteen, and no rewriting closed the gap: five wordings and one structural variant all landed
between 10% and 26%, each fixing some scenarios while breaking others, because the underlying cause is not
the wording. Chinese characters anywhere in the input pull the output towards Chinese — inside a user ID,
inside a pasted paragraph, inside a single term in an otherwise English sentence — while English inside a
Chinese conversation does not pull back. Six mirrored scenario pairs across eight arms showed no exception.
The residual failures therefore skew one way: a conversation that should yield English is the one at risk.

This rule is nonetheless the best of the self-judging variants, and the only one that never drifted on
plain single-language conversations, which is why it is the fallback rather than one of the others.

The measurements behind those two texts are recorded internally; the corpus and runner live outside the
repository because they need real models and five-figure call counts.
"""

# `{language}` is filled by `user_memory._language.build_language_rule`, not by the prompt renderer — the
# renderer only knows each prompt's own placeholders.
CALLER_CHOSEN_LANGUAGE_RULE = (
    "**CRITICAL LANGUAGE RULE**: Write ALL output fields in {language}. This is mandatory and overrides "
    "the language of the conversation content. Person names, user IDs, proper nouns and technical terms "
    "keep their original form regardless of the output language."
)

EXISTING_PROFILE_LANGUAGE_RULE = (
    "**CRITICAL LANGUAGE RULE**: You MUST write ALL output in the SAME language as the existing profile "
    "you are updating, including every personality tag. Do NOT switch languages even when the new "
    "conversation is written in a different language — the profile's language was fixed when it was first "
    "created. The tag examples in this prompt illustrate format and granularity only, not language. This "
    "is mandatory."
)
"""Fallback for the profile UPDATE path: inherit the language the profile is already written in.

Inheriting is what stops a later conversation in another language from splitting a profile in half, but it
is still a judgement the model has to make, and a wrong one is unrecoverable — every later update inherits
it in turn. A caller passing `output_language` on every update removes the judgement entirely and can also
correct a profile whose language already went wrong.
"""

COMPACTED_PROFILE_LANGUAGE_RULE = (
    "**CRITICAL LANGUAGE RULE**: You MUST write ALL output in the SAME language as the profile you are "
    "compacting, including every personality tag. Compaction never changes the profile's language. The tag "
    "examples in this prompt illustrate format and granularity only, not language. This is mandatory."
)
"""Fallback for compaction. Same reasoning as the update rule, minus the incoming conversation.

Compaction rewrites every item, so it is the one path that can change a whole profile's language in a single
call — which makes it the natural place to correct one, and the worst place to guess.
"""

MERGED_EPISODES_LANGUAGE_RULE = (
    "**CRITICAL LANGUAGE RULE**: You MUST write ALL output in the SAME language as the episodes you are "
    "merging. Merging never changes the language — do not translate. This is mandatory."
)
"""Fallback for merging episodes into one narrative.

The episodes were already extracted, so their language was settled upstream and merging inherits it. That
inheritance is only as good as the episodes agree, though: merging episodes written in different languages
asks the model to pick one, and nothing here says which. A caller who cannot guarantee its episodes share a
language should name the output language rather than leave the pick to the model.
"""

EXISTING_NARRATIVE_LANGUAGE_RULE = (
    "**CRITICAL LANGUAGE RULE**: You MUST write ALL output in the SAME language as the existing narrative "
    "you are updating. Updating never changes the language — do not translate, even if the new episodes are "
    "written in a different language. This is mandatory."
)
"""Fallback for updating a merged narrative. Same shape as the profile update rule.

The narrative already has a language and the incoming episodes may not share it, so the rule pins the
narrative's own — which is what stops an update in a second language from splitting it in half, and is
equally unrecoverable when the language it pins is already wrong. Naming a language is the way out.
"""

MERGED_DECISIONS_LANGUAGE_RULE = (
    "**CRITICAL LANGUAGE RULE**: You MUST write ALL output in the SAME language as the decisions you are "
    "merging. Merging never changes the language — do not translate. This is mandatory."
)
"""Fallback for merging already-extracted Decisions into one current Decision.

Same inherit-don't-translate shape as ``MERGED_EPISODES_LANGUAGE_RULE``. The decisions were extracted
upstream, so their language is already settled; re-judging here would translate a merge out of the
language its sources were written in. Disagreeing sources still leave the model to pick one — name a
language when the caller cannot guarantee they agree.
"""

EXISTING_DECISION_LANGUAGE_RULE = (
    "**CRITICAL LANGUAGE RULE**: You MUST write ALL output in the SAME language as the existing decision "
    "you are updating. Updating never changes the language — do not translate, even if the new decisions "
    "are written in a different language. This is mandatory."
)
"""Fallback for updating an already-merged Decision.

Pins the existing Decision's language, matching ``EXISTING_NARRATIVE_LANGUAGE_RULE``. A later decision
in another language must not split the record; a language that went wrong on INIT stays wrong until a
caller names one.
"""

SOURCE_TEXT_LANGUAGE_RULE = (
    "**CRITICAL LANGUAGE RULE**: You MUST output in the SAME language EPISODE_TEXT itself is written in. "
    "ALL output MUST match that language. This is mandatory."
)
"""Fallback for the operators whose input is already-extracted memory text rather than a conversation.

Deliberately one sentence where ``PARTICIPANT_LANGUAGE_RULE`` is three paragraphs. Everything those
paragraphs deal with — pasted documents, participants writing different languages, an identifier
contradicting the conversation — is a property of a live multi-party conversation. ``EPISODE_TEXT`` has
already been through extraction, so it is a single-language narrative and there is nothing to adjudicate:
the language was settled upstream and this layer inherits it.
"""

PARTICIPANT_LANGUAGE_RULE = (
    "**CRITICAL LANGUAGE RULE**: Write ALL output fields in the language the participants use when they "
    "talk to each other. If they talk in Chinese, write in Chinese; if in English, write in English; if in "
    "Japanese, write in Japanese; and so on for any language. This is mandatory.\n\n"
    "To find that language, read only the message contents — the sentences the participants address to "
    "one another: their questions, answers, opinions, decisions and reactions. Text they merely bring "
    "into the conversation is not theirs: a pasted document, a code block, an error message, a quoted "
    "passage, the contents of a link. Ignore all of it when deciding the language, however much of the "
    "conversation it occupies. Such material usually sits in the same message as the participant's own "
    "words, so split within a message rather than discarding the whole message.\n\n"
    "A foreign word or technical term inside a participant's own sentence does not change the judgement — "
    "go by the language the sentence is built in. Person names, user IDs, proper nouns and technical "
    "terms keep their original form in the output regardless of the output language."
)

PROFILE_INIT_LANGUAGE_RULE = (
    PARTICIPANT_LANGUAGE_RULE + "\n\n"
    "This is the call that fixes the profile's language: later update and compaction calls preserve "
    "whatever language you choose here, so every personality tag must be written in that language too — "
    "never in a different language from the rest of the profile."
)
"""Fallback for profile INIT: the participant rule, plus what makes this particular call different.

INIT is the only call that chooses a profile's language rather than inheriting it, and the two paths that
inherit it bind their personality tags explicitly, so the call that sets the precedent has to bind them
too — otherwise the tags are the one part of a profile no rule ever pins to its language.

The appended sentence is carried over verbatim from the pre-``output_language`` INIT prompt rather than
newly worded, and the participant rule it extends is untouched, so the eight-arm measurement of that rule
still describes the judgement this text asks for. What the extra sentence does to tag-language consistency
is not itself measured — the corpus judges the profile's prose, not its labels.
"""
