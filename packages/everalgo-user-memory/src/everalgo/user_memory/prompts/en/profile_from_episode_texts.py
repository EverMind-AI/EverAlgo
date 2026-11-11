"""English prompts for ProfileExtractor's Episode-text path.

Structure: every stage is a pair of messages. The SYSTEM message holds the rules, assembled from blocks in the
order a model works — the stage's role sentence, what the profile holds (``<what_the_profile_holds>``), what never
enters it (``<never_in_the_profile>``, the single home of every exclusion), worked examples of where that line
falls (``<examples>``; INIT and UPDATE only, each in its own reply shape), how this stage does its work
(``<how_to_update>``, ``<how_to_compact>`` or ``<how_to_regroup>``; INIT has none, the blocks before it say
everything INIT needs) and the reply format (``<reply_format>``: one sentence plus a bare, parseable JSON
skeleton, never a fence the model could copy; UPDATE, COMPACT and REGROUP reply with operations on indexed stored
items, so an item no operation names stays exactly as stored) — and closes with the blocks that vary between
requests, least- to most-variable: the output-language rule (``<output_language>``), the category snapshot
(``<available_categories>``) and, only when the caller supplies them, its additional requirements
(``<additional_requirements>``, see :func:`additional_requirements_block`). Prompt caches match on a prefix: the
language rule takes one of a few fixed texts, the snapshot comes from the caller on every request and the
additional requirements are whatever the caller writes, so this order lets the static prefix be shared across
languages and owners.
The USER message opens with ``<target_person>`` naming the person and then carries only the per-request data,
marked up the same way: narratives as ``<narrative time="...">`` elements, stored items as
``<item index="...">`` elements holding JSON, and in UPDATE the matters this person asked not to be recorded as
``<matter>`` elements in ``<do_not_record>``. A rule appears once and never names the person: the rules say "the
target person", so the system text is the same for every owner and every request.

One prompt, one delimiter format. Both messages delimit every block and every input element with an XML-style
tag, carry times and indices as attributes, and keep JSON for data inside a tag; no rule text uses Markdown
emphasis, headings, numbered steps or code fences.
That is what the prompting guides of the current model generation do: the OpenAI GPT-5.x guides wrap each
instruction block in a descriptive snake_case tag, Anthropic's best practices ask for XML tags wherever a prompt
mixes instructions, context and variable input, and Gemini's guidance asks for one format per prompt. With the
input marked up this way it explains itself, so no block describes how to read it.

This path owns its blocks; the MemCell prompts are a separate path with their own definitions.
"""

from __future__ import annotations

from everalgo.utils import paragraphs, tagged


# Worked examples: their people, jobs and places appear in no test case, so they cannot teach a test's answer.
def _examples(lead: str, *examples: str) -> str:
    """Build an ``<examples>`` block: the lead sentence, then each example in its own ``<example>`` element.

    Args:
        lead: The sentence saying what the examples show and how to use them.
        *examples: Each example's body, already marked up, its ``<reply>`` last.

    Returns:
        The tagged block, a blank line after the lead and between examples.
    """
    return tagged("examples", paragraphs(lead, *(tagged("example", example) for example in examples)))


# --------------------------------------------------------------------------------------------------
# Shared blocks
# --------------------------------------------------------------------------------------------------

_TARGET_LINE = "<target_person>{{target_user}}</target_person>"

_WHAT_THE_PROFILE_HOLDS = paragraphs(
    "A profile is what an assistant should know about this person before the next conversation: who they are, "
    "their life and the people in it, their background and skills, what they like and dislike, their long-term "
    "goals, how they want to be treated, and their stable habits and dispositions. Major facts about their past — "
    "where they studied, where they are from, the jobs they have held, the big events in their life — are standing "
    "properties too. It is not a log of what happened.",
    "A profile has two lists: explicit_info and implicit_traits.",
    "explicit_info holds the facts about this person that a narrative states directly. A fact counts as stated "
    "directly only when a narrative explicitly attributes it to this person. Who they are is read the same way from "
    'how they plainly speak of their own work and family: "we make procurement software" while they write its '
    'requirements gives their role and their organization; "my wife Lin Lan" gives that they are married to Lin Lan. '
    "The identity item names the role, the organization with what it does, and the city, as fully as the narratives "
    "allow. Each item has three fields. evidence "
    "is what supports the fact: narrative excerpts or faithful paraphrases, with their dates, in one string, one per "
    "occasion the fact was established or changed; a repeat of the same occasion adds nothing. A paraphrase keeps "
    "the narrative's attribution and invents no wording, date, attribution or identifier. description is the fact "
    'itself: one fact per item, in one or two short sentences without a subject, stating the current value only — no dates of arrangements, no step-by-step history ("used to … then …"), no detail that will be stale next month. A habit or routine is stated with its rough frequency, not its weekdays, clock times or distances; a health condition with how it is managed over time, not drug names, doses, target values or appointment schedules. category is the dimension the fact belongs to (identity, skills, preferences and the like), named '
    "after the fact itself, not after the topic of the narrative it appeared in. Prefer a category listed in "
    "<available_categories>, and when there is no list or none fits, coin a short dimension name. How this person "
    "wants the assistant to work — reply language and style, how documents should be laid out, default reminder "
    "times, number and date formats and the like — is one explicit_info item at most, a short summary of those "
    "standing preferences kept current: a new standing rule updates that item instead of adding another, and a "
    "request for this one time changes nothing.",
    "implicit_traits holds the stable patterns that no single narrative states but that show across several. A "
    "trait is a grounded inference: you read a pattern out of signals and name those signals in its basis. A signal "
    "is a choice or request of theirs that a narrative records, not an operation they carried out; the same thing "
    "retold several times is one signal. Consulting the assistant is what every user does and is not a signal; what "
    "they chose or asked for, and how, can be. A fact a narrative states directly stays in explicit_info; neither "
    "that fact nor the occasions it was stated is a signal, and a trait resting on one is that fact restated. Each "
    "item has three fields. basis names the concrete signals behind the inference — which choices or requests, in "
    'which narratives — in one string; restating the requirement ("several signals", "repeated choices") is not a '
    "basis, and if you cannot name the signals the trait does not belong here. Every signal in a basis is something "
    "a narrative shown records, in its own terms; a basis invents no signal, wording or date to reach a count. "
    "description is what the pattern is: one pattern per item, in one or two short sentences without a subject. "
    "trait is the name of the pattern, a short phrase.",
    # The extractor creates no new trait unless built with ``new_traits=True``; this paragraph tells the model so and says where
    # what this person states about their values, situation and habits goes instead. The implicit_traits definition
    # above stays for the stored traits an update may still revise or delete.
    "This system adds no new implicit_traits item: a first profile leaves the list empty and an update adds none. "
    "What this person says about their own values, principles and priorities — what matters most to them, rules "
    "they hold themselves to, how they want to come across — is stated directly, so it is an explicit_info item, "
    "under the category that fits it, and so is a standing situation they describe (on leave, caring for a child). "
    "So is a hobby or habit they describe as their own or that recurs across narratives (shoots film, travels two "
    "or three times a year, runs before work): one explicit_info item for it, kept up to date as it changes.",
)

_NEVER_IN_THE_PROFILE_LIST = """\
Whatever the wording and whatever evidence is attached, these never become items:
- A single action, or a single action restated as a capability, skill, duty, interest or attitude. Having run the tests once is not being responsible for testing.
- A passing state, anything trivial, and what this person is working on right now: the task at hand, an ongoing project, a plan with a date. A long-term goal of their own (a book they mean to publish this year, a race they train for) is not such a plan: it enters, under its current name and target.
- A rule that would hold for whoever sat in their seat (a team process, an SOP, how a pipeline works), and a decision taken once for one project.
- A single question, and anything the assistant or another participant said: being told something is not knowing it.
- Work the assistant or an agent carried out at this person's request, credited to them as their own doing or skill. They asked an agent to fix the build, write the scraper or set up the server: what enters is what they own, direct or decided ("runs the CrossGate browser project, built with coding agents"), never that they wrote, built, configured or fixed it themselves, and never a skill read off the agent's work. Advice or analysis the assistant gave is the assistant's, not their view, unless they adopt it in their own words.
- A fact about anyone other than the target person, even one the target person told, and even inside an item about the relationship. "My brother is an accountant in Shanghai" gives nothing: the accounting and Shanghai are his, and "Has a brother" on its own is an empty item. A relationship enters only as a standing feature of this person's own life — a partner, someone they live with — and then with nothing about the other person.
- A one-off request about format or procedure: how to answer this time, which tool to use for this task.
- Anything said as a joke, half in jest or as a what-if ("if I can't keep this up I'll go open a café"), however plainly the narrative records it: it is not a goal, a plan or a fact.
- A self-description said once in passing ("I'm a worrier", "I'm lazy", "I'm quite reliable") and a change tried for only days (gave up afternoon coffee this week): one remark or one attempt is not who this person is.
- A routine that has only just started, restarted or been signed up for, or that they only mean to keep up (bought a gym card this week, went for a first run again today, asked for a reminder to run three mornings a week): it is not yet a habit. A routine they describe as how they live now ("I usually sleep around two", "I swim three mornings a week") is one. A long-standing habit stays through a temporary interruption (an injury, a trip, a busy month), and the interruption itself is a passing state that does not enter; a habit only weeks or months old that they stop is removed.
- A taste, habit or trait of someone else that came up beside this person's own (a cousin who skips coriander, a colleague who drinks no coffee): it is that person's, not theirs.
- A label this person rejected or did not accept: a trait someone else attributed to them that they denied or left unanswered, or a self-description they denied. One later occasion that seems to fit the label does not override their denial.
- Anything this person asked not to be recorded or remembered, together with every later mention of the same matter in any wording: the request stands until they withdraw it.
- Identification and account numbers (ID card, bank card, insurance, passport, passwords) and single readings of a measurement (one day's blood pressure or glucose); the standing condition behind readings may enter, the readings never.
- Work standards this person set for a team, a document or a deliverable (report formats, review checklists, hand-over rules): they describe the work, not the person. The principle behind them is different: a principle this person holds about how they work — people make the final call, customer data stays isolated, claims stay modest — is a value of theirs and belongs in the profile.
- Arrangements of the people around this person: who stays to help, when parents move, a child's school timetable, who stands in for them at work. The relationship itself enters with the names this person uses ("Married to Lin Lan; two children, Xiaoman and Anan"), and so does a standing duty of their own toward someone (they help manage their father's diabetes, their parents now live in the same compound); the other person's own plans and schedule do not."""

# Only the stages that read stored items carry this sentence; INIT has nothing on file to remove.
_STORED_CLAIM_IS_REMOVED = (
    "A stored item that makes such a claim is removed, not rewritten and not merged: its evidence records an "
    "occasion, and an occasion cannot establish a standing property."
)

_LEAVE_UNCERTAIN_OUT = "When attribution, durability or grounding is uncertain, leave the item out."

_NEVER_IN_THE_PROFILE_INIT = tagged(
    "never_in_the_profile", paragraphs(_NEVER_IN_THE_PROFILE_LIST, _LEAVE_UNCERTAIN_OUT)
)

_NEVER_IN_THE_PROFILE_STORED = tagged(
    "never_in_the_profile",
    paragraphs(_NEVER_IN_THE_PROFILE_LIST, f"{_STORED_CLAIM_IS_REMOVED} {_LEAVE_UNCERTAIN_OUT}"),
)

_JSON_OUTPUT_RULE = "Reply with one JSON object, nothing else."
# The schema fixes which keys an item has, not their order; asking for grounding first is all the prompt can do.
_FIELD_ORDER_RULE = "Write each item's fields in the order shown, evidence or basis first."
_REPLY_RULES = f"{_JSON_OUTPUT_RULE} {_FIELD_ORDER_RULE}"

_EXPLICIT_ITEM_SKELETON = '{"evidence": "...", "description": "...", "category": "..."}'
_IMPLICIT_ITEM_SKELETON = '{"basis": "...", "description": "...", "trait": "..."}'

_OUTPUT_LANGUAGE = tagged("output_language", "{{language_rule}}")

_AVAILABLE_CATEGORIES = tagged("available_categories", "{{available_categories}}")

_ADDITIONAL_REQUIREMENTS_LEAD = (
    "They apply on top of the rules above; the reply shape in <reply_format> stays as it is. Where they conflict "
    "with the rules above, the rules above hold."
)


def _system_prompt(role: str, never_in_the_profile: str, *stage_blocks: str, examples: str | None = None) -> str:
    """Assemble a stage's system template: the shared static blocks around its own, the per-request blocks last.

    Args:
        role: The stage's role sentence, which opens the message untagged.
        never_in_the_profile: The ``<never_in_the_profile>`` variant for the stage.
        *stage_blocks: The stage's own tagged blocks, in order, ending with ``<reply_format>``.
        examples: The stage's ``<examples>`` block, placed right after ``<never_in_the_profile>``; None for a stage
            that shows none.

    Returns:
        The system template.
    """
    return "\n\n".join(
        (
            role,
            tagged("what_the_profile_holds", _WHAT_THE_PROFILE_HOLDS),
            never_in_the_profile,
            *(() if examples is None else (examples,)),
            *stage_blocks,
            _OUTPUT_LANGUAGE,
            _AVAILABLE_CATEGORIES,
        )
    )


def additional_requirements_block(text: str) -> str:
    """Build the ``<additional_requirements>`` block that closes a system message when the caller has requirements.

    The text is placed verbatim after a fixed lead-in saying it adds to the bundled rules, leaves the output shape
    as it is and yields to the bundled rules where the two conflict. Append the block to an already-rendered
    message: the text is not a template, so braces in it stay literal.

    Args:
        text: The caller's additional requirements.

    Returns:
        The tagged block, the lead-in on its first line.
    """
    return tagged("additional_requirements", f"{_ADDITIONAL_REQUIREMENTS_LEAD}\n{text}")


# --------------------------------------------------------------------------------------------------
# Item checks (INIT and UPDATE) — every item carries a quote the system looks for, and an item claiming a standing
# pattern waits for a second occasion unless its quote shows regularity; ``profile/_pending.py`` enforces both. The
# block closes the system message after the per-request blocks, where it was evaluated; moving it ahead of them
# would let the static prefix be shared again but changes what the model reads.
# --------------------------------------------------------------------------------------------------

_PATTERN_FIELD = (
    'Every item also carries "pattern". "pattern": true when the item claims a standing pattern of this person — a '
    "habit or routine, a like or dislike, a hobby, a personality or emotional trait, or a way they work, manage, "
    "decide or communicate (how they run a team, how they answer messages, rules they keep at work), whether you "
    'read it out of what they did or they said it about themselves. "pattern": false for facts about who they are, '
    "their family and relationships, their job, background, skills, health, life events and goals; for the one item "
    "that sums up how they want the assistant to work; and for what they say matters most to them in life (family "
    "before work, health first)."
)

_PENDING_RULE_INIT = (
    "The system does not write a pattern item the first time it appears: it keeps the item aside and writes it only "
    "when narratives of a later update show the same pattern again. Propose pattern items exactly as you would any "
    "item, under the same rules; the system decides when each is written."
)

_PENDING_RULE_UPDATE = (
    "The system does not write a pattern item the first time it appears: it keeps the item in <pending_patterns> and "
    "writes it only when narratives of a later update show the same pattern again. Propose a new pattern item with "
    "add exactly as you would any item; the system decides when it is written. Before writing your operations, go "
    "through <pending_patterns> one entry at a time and ask whether these narratives show that same pattern again — "
    "another occasion of the habit (another swim, another late night), the like or dislike voiced again, the same way "
    'of working seen again. For each entry they do, reply with a confirm operation naming the entry: {"action": '
    '"confirm", "type": "explicit_info", "pending": <entry index>, "data": <the item in full, as an add writes it, '
    "its evidence naming the earlier occasion and the new one, its quote taken from these narratives>}. Never add an "
    "item that repeats a pending entry: confirm it instead. An entry these narratives only share a topic with, "
    "contradict, or that this person negated or asked not to record, is left alone."
)

_QUOTE_RULE = (
    "Every item you write — each item of a first profile, each add, and each update that changes a description — "
    'also carries "quote": words copied character for character from one narrative in <episode_narratives>, the '
    "clause that itself states what the description says, 10 to 60 characters. Copy exactly: no paraphrase, no "
    "translation, no summary. For a pattern item, quote the words that show it is regular or standing for this "
    'person when a narrative has them, and copy into "regular" the few words of your quote that say so: how often '
    "(every day, each week, twice a month, on Wednesdays), always, usually, often, never, a habit or routine, for "
    "years, from now on, by default, without fail. One date or day (this Saturday), once, again, very, or a plan does "
    'not make it regular: then "regular" is "". The system writes a pattern item at once when its "regular" words '
    "stand in its quote and keeps the others pending. The system looks for every quote in the narratives and "
    "discards an item whose quote is not there."
)

_NO_GENDER_RULE = (
    "Narratives often guess this person's gender, so no description refers to this person with a gendered word — he "
    "or she, his or her, or the like in the output language — unless they state their gender themselves; the subject "
    "is left out."
)

_ANNOUNCED_CHANGE_RULE = (
    "A change this person announces for later — leaving a job tomorrow, moving next month, a leave that starts "
    "soon — is not the current state yet: until a narrative reports that it has happened, items go on stating the "
    "present, and the announced change enters at most as a plan or goal of theirs."
)

_ITEM_CHECKS_INIT = tagged(
    "item_checks",
    paragraphs(
        _PATTERN_FIELD,
        _PENDING_RULE_INIT,
        _QUOTE_RULE,
        _ANNOUNCED_CHANGE_RULE,
        _NO_GENDER_RULE,
        'An explicit_info item then reads: {"evidence": "...", "description": "...", "category": "...", '
        '"pattern": false, "quote": "...", "regular": ""}',
    ),
)

_ITEM_CHECKS_UPDATE = tagged(
    "item_checks",
    paragraphs(
        _PATTERN_FIELD,
        _PENDING_RULE_UPDATE,
        _QUOTE_RULE,
        _ANNOUNCED_CHANGE_RULE,
        _NO_GENDER_RULE,
        'An add then reads: {"action": "add", "type": "explicit_info", "data": {"evidence": "...", "description": '
        '"...", "category": "...", "pattern": true, "quote": "...", "regular": "..."}}.',
    ),
)

# --------------------------------------------------------------------------------------------------
# INIT
# --------------------------------------------------------------------------------------------------

_ROLE_INIT = (
    "You are a user-profile analyst. Build the profile of the target person from the Episode narratives. The target "
    "person is named in <target_person>. The Episode narratives are in <episode_narratives>. Follow the rules below."
)

# INIT lists the matters this person asked not to be recorded in do_not_record; the extractor keeps them on the
# profile and every later UPDATE shows them in <do_not_record>.
_REPLY_FORMAT_INIT = paragraphs(
    "Both lists may be empty when the narratives carry nothing about this person. When this person asks that "
    "something not be recorded or remembered, list the matter in do_not_record, named in a few words as narrowly as "
    "they meant it, with its evidence, and write no item about it; the system keeps the list for every later update. "
    f"do_not_record is empty otherwise. {_REPLY_RULES}",
    '{"explicit_info": ['
    + _EXPLICIT_ITEM_SKELETON
    + '], "implicit_traits": ['
    + _IMPLICIT_ITEM_SKELETON
    + '], "do_not_record": [{"topic": "...", "evidence": "..."}]}',
)

_EXAMPLE_OPERATIONS_ONLY = """\
<episode_narratives>
<narrative time="2026-03-02 09:00 UTC">Ravi asked the assistant to turn a CSV of monthly sales figures into a pivot table and to put the totals in bold. The assistant produced the table; Ravi then asked for the same data as a bar chart.</narrative>
</episode_narratives>
<reply>
{"explicit_info": [], "implicit_traits": [], "do_not_record": []}
</reply>"""

_EXAMPLE_FACTS_AND_A_PATTERN = """\
<episode_narratives>
<narrative time="2026-01-10 10:00 UTC">Ines said she runs a small bookshop in Lisbon and that her cousin, a lawyer, handles the paperwork. She asked how to price a new line of notebooks; when the assistant proposed a price, she asked to see last quarter's sales first.</narrative>
<narrative time="2026-02-14 15:30 UTC">Ines asked for help drafting an email to a book distributor and wanted the figures from her spreadsheet checked before sending it. She mentioned she wants to open a second shop one day. She mentioned she lives with her mother.</narrative>
<narrative time="2026-03-20 11:00 UTC">Ines asked the assistant to rewrite a job advertisement and to keep the tone formal this time.</narrative>
</episode_narratives>
<reply>
{
  "explicit_info": [
    {"evidence": "2026-01-10: said she runs a small bookshop in Lisbon", "description": "Runs a small bookshop in Lisbon.", "category": "Identity"},
    {"evidence": "2026-02-14: said she wants to open a second shop one day", "description": "Wants to open a second shop one day.", "category": "Goals"},
    {"evidence": "2026-02-14: said she lives with her mother", "description": "Lives with her mother.", "category": "Relationships"}
  ],
  "implicit_traits": [
    {"basis": "2026-01-10: asked to see last quarter's sales before accepting a proposed price; 2026-02-14: wanted the spreadsheet figures checked before sending the email", "description": "Checks the numbers before acting on advice.", "trait": "Verifies before deciding"}
  ],
  "do_not_record": []
}
</reply>"""

_EXAMPLES_INIT = _examples(
    "Two worked examples. They show what enters the profile and what stays out; copy their judgement, not their "
    "wording.",
    _EXAMPLE_OPERATIONS_ONLY,
    _EXAMPLE_FACTS_AND_A_PATTERN,
)

PROFILE_INITIAL_FROM_EPISODE_TEXTS_SYSTEM_PROMPT = (
    _system_prompt(
        _ROLE_INIT,
        _NEVER_IN_THE_PROFILE_INIT,
        tagged("reply_format", _REPLY_FORMAT_INIT),
        examples=_EXAMPLES_INIT,
    )
    + f"\n\n{_ITEM_CHECKS_INIT}"
)

PROFILE_INITIAL_FROM_EPISODE_TEXTS_USER_PROMPT = "\n\n".join(
    (_TARGET_LINE, tagged("episode_narratives", "{{episode_texts}}"))
)

# --------------------------------------------------------------------------------------------------
# UPDATE
# --------------------------------------------------------------------------------------------------

_ROLE_UPDATE = (
    "You are a user-profile analyst. Update the stored profile of the target person from the Episode narratives. "
    "Express the update as operations on the stored profile. The target person is named in <target_person>. The "
    "stored profile is in <stored_profile>. The Episode narratives are in <episode_narratives>; the stored profile "
    "does not reflect them yet, and they may be older than it. Follow the rules below."
)

_HOW_TO_UPDATE = paragraphs(
    'add or update is decided by the fact, not the wording. Before "add", scan the stored items for the same fact '
    'under any wording — found means "update". Adding a second copy of a stored fact is always wrong.',
    "An update restates the whole item — its category or trait, its description and its evidence or basis, the "
    "unchanged ones included — so a description always travels with its grounding. An update keeps the item on the "
    "same fact: a correction, a sharper wording or a changed value of that fact goes into the item. A fact of another "
    "dimension — a goal beside a job, a habit beside a hobby — is an add, and what <never_in_the_profile> excludes "
    "goes nowhere.",
    "The profile is written in one language, the one <output_language> gives. A stored item written in any other "
    "language is updated: its category or trait, its description and its evidence or basis rewritten in that "
    "language and nothing else changed, even when no narrative mentions it.",
    "delete when this person explicitly negates it, a new statement contradicts it outright, it has expired or is "
    "trivial, or it is on the <never_in_the_profile> list.",
    "A change in a standing situation — a move, a new job or role, a new way of getting to work or of working, a "
    "habit stopped or paused, family moving in or away, a plan cancelled or postponed, a goal changed or a project renamed — makes other stored "
    "items untrue as well. When a narrative reports one, go through every stored item it touches (commute, place "
    "of work, routines tied to the old home or job, the old name, care arranged at the old place) and update or "
    "delete each, so that no item still describes the old situation as the current one. A temporary "
    "interruption — an injury, a trip, a busy month — is not such a change.",
    "When this person asks that something not be recorded or remembered, reply with a forget operation naming the "
    "matter in a few words as narrowly as they meant it — the thing they asked about, not its whole area — and delete every stored item that touches it. The system keeps each such matter in "
    "<do_not_record> for every later update: an item about a matter listed there never enters, in any wording and "
    "however it is mentioned again, and a stored item that touches one is deleted.",
    "none is often the right answer. Narratives made only of operations contain nothing about who this person is, "
    "however many there are; manufacturing an operation is not.",
    "Newer state stands; older narratives only add to it. Every stored item shows observed_at, the time of the "
    "narrative that last established its description, and every narrative shows its own time. A narrative whose "
    'time is before an item\'s observed_at may "add" a fact not yet stored and may "update" that item with '
    'additional evidence, but it must not rewrite the item\'s description or "delete" it. An earlier value of a '
    "dimension that already holds one goes into that item's evidence, never into a new item; the description may "
    "add that history in a short parenthesis after the current value, but the current value stays as written. The "
    "system maintains observed_at and discards operations that break this rule; leave observed_at to the system.",
)

_REPLY_FORMAT_UPDATE = paragraphs(
    f"In update and delete, index is the item's index attribute in <stored_profile>. {_REPLY_RULES}",
    """\
Nothing to do:
{
  "operations": [
    {"action": "none"}
  ]
}""",
    f"""\
One or more operations:
{{
  "operations": [
    {{"action": "add", "type": "explicit_info", "data": {_EXPLICIT_ITEM_SKELETON}}},
    {{"action": "add", "type": "implicit_traits", "data": {_IMPLICIT_ITEM_SKELETON}}},
    {{"action": "update", "type": "explicit_info", "index": 0, "data": {_EXPLICIT_ITEM_SKELETON}}},
    {{"action": "delete", "type": "implicit_traits", "index": 1, "reason": "..."}},
    {{"action": "forget", "topic": "...", "evidence": "..."}}
  ]
}}""",
)

_EXAMPLE_NAMED_LANGUAGE_SAME_SUBJECT = """\
With <output_language> naming English:
<stored_profile>
<explicit_info>
<item index="0">{"evidence": "2025-11-03：说自己在京都经营一家花店", "description": "在京都经营一家花店。", "category": "身份", "observed_at": "2025-11-03 10:00 UTC"}</item>
</explicit_info>
<implicit_traits>
</implicit_traits>
</stored_profile>
<episode_narratives>
<narrative time="2026-01-12 14:00 UTC">Hana mentioned that her flower shop in Kyoto is in the Gion district, and asked for a caption for a photo of the new shelf.</narrative>
</episode_narratives>
<reply>
{
  "operations": [
    {"action": "update", "type": "explicit_info", "index": 0, "data": {"evidence": "2025-11-03: said she runs a flower shop in Kyoto; 2026-01-12: said the shop is in the Gion district", "description": "Runs a flower shop in Kyoto's Gion district.", "category": "Identity"}}
  ]
}
</reply>"""

_EXAMPLES_UPDATE = _examples(
    "One worked example. It shows what enters the profile and what stays out; copy its judgement, not its wording.",
    _EXAMPLE_NAMED_LANGUAGE_SAME_SUBJECT,
)

PROFILE_UPDATE_FROM_EPISODE_TEXTS_SYSTEM_PROMPT = (
    _system_prompt(
        _ROLE_UPDATE,
        _NEVER_IN_THE_PROFILE_STORED,
        tagged("how_to_update", _HOW_TO_UPDATE),
        tagged("reply_format", _REPLY_FORMAT_UPDATE),
        examples=_EXAMPLES_UPDATE,
    )
    + f"\n\n{_ITEM_CHECKS_UPDATE}"
)

# <do_not_record> comes before the stored profile and narratives it governs; "(none)" when the list is empty.
# <pending_patterns> closes the message: the pattern items held for a second occasion, "(none)" when there are none.
PROFILE_UPDATE_FROM_EPISODE_TEXTS_USER_PROMPT = "\n\n".join(
    (
        _TARGET_LINE,
        tagged("do_not_record", "{{do_not_record}}"),
        tagged("stored_profile", "{{stored_profile}}"),
        tagged("episode_narratives", "{{episode_texts}}"),
        tagged("pending_patterns", "{{pending}}"),
    )
)

# --------------------------------------------------------------------------------------------------
# COMPACT — the profile as a whole is over its caps. The reply is operations on the indexed stored items, so every
# item it does not name keeps its host keys (ids) and observed_at; a rewrite of the whole profile kept neither.
# --------------------------------------------------------------------------------------------------

_ROLE_COMPACT = (
    "You are a user-profile analyst. Bring the stored profile of the target person back inside its caps. Express "
    "the change as operations on the stored items; every item you do not name stays exactly as it is. The target "
    "person is named in <target_person>. The stored profile is in <stored_profile>. Follow the rules below."
)

# The first requirement carries the one ordering that matters: merging excluded items first turns twelve of them
# into one surviving item instead of none.
_HOW_TO_COMPACT = paragraphs(
    "delete everything on the <never_in_the_profile> list before merging anything: twelve such items become zero "
    "items, not one.",
    'merge only restatements of the same fact: name every index in "indices" and give the merged item in "data", '
    "keeping the better wording and every occasion's grounding. Distinct facts stay distinct items even within one "
    "dimension.",
    'split an item that chains several facts: give one item per fact in "items", each with its own grounding; a '
    "fact that would then be trivial is left out.",
    'move a trait that merely restates a fact a narrative states directly into explicit_info: "data" is the '
    "explicit_info item it becomes.",
    "update an item only to put it under the right category or trait name; its claim stays as it is.",
    "Prefer deleting a weak item to merging or shortening a strong one; every claim and grounding you write comes "
    "from the items you were given, and ending well under the caps is fine. Each index appears in one operation at "
    "most.",
)

_COMPACT_OPERATIONS_SKELETON = f"""\
{{
  "operations": [
    {{"action": "delete", "type": "explicit_info", "index": 3, "reason": "..."}},
    {{"action": "merge", "type": "explicit_info", "indices": [1, 4], "data": {_EXPLICIT_ITEM_SKELETON}}},
    {{"action": "split", "type": "explicit_info", "index": 2, "items": [{_EXPLICIT_ITEM_SKELETON}, {_EXPLICIT_ITEM_SKELETON}]}},
    {{"action": "move", "type": "implicit_traits", "index": 0, "data": {_EXPLICIT_ITEM_SKELETON}}},
    {{"action": "update", "type": "implicit_traits", "index": 1, "data": {_IMPLICIT_ITEM_SKELETON}}}
  ]
}}"""

_REPLY_FORMAT_COMPACT = paragraphs(
    "After your operations, explicit_info + implicit_traits must total no more than {{max_items}} items, with no "
    "more than {{max_per_category}} under any one category or trait name. index and indices are the index "
    f"attributes in <stored_profile>; type is the list the index belongs to. {_REPLY_RULES}",
    _COMPACT_OPERATIONS_SKELETON,
)

PROFILE_COMPACT_FROM_EPISODE_TEXTS_SYSTEM_PROMPT = _system_prompt(
    _ROLE_COMPACT,
    _NEVER_IN_THE_PROFILE_STORED,
    tagged("how_to_compact", _HOW_TO_COMPACT),
    tagged("reply_format", _REPLY_FORMAT_COMPACT),
)

PROFILE_COMPACT_FROM_EPISODE_TEXTS_USER_PROMPT = "\n\n".join(
    (_TARGET_LINE, tagged("stored_profile", "{{profile_text}}", total_items="{{total_items}}"))
)

# --------------------------------------------------------------------------------------------------
# REGROUP — ONE overcrowded group, the rest of the profile unseen; operations on the group's indexed items.
# --------------------------------------------------------------------------------------------------

_ROLE_REGROUP = (
    "You are a user-profile analyst. Reorganise one group of the stored profile of the target person; the group is "
    "over its cap. Express the change as operations on the group's items; every item you do not name stays exactly "
    "as it is. The target person is named in <target_person>. The group and its items are in <group>. The rest of "
    "the profile is not shown; work from the items in <group> alone. Follow the rules below."
)

_HOW_TO_REGROUP = paragraphs(
    'merge restatements of the same fact into one item: name every index in "indices" and give the merged item in '
    '"data", keeping the better wording and the grounding in the form <what_the_profile_holds> gives.',
    "update an item to file it under a better name when the group's name has swallowed several dimensions, choosing "
    "the category as <what_the_profile_holds> defines it; renaming leaves its description as it is.",
    'split an item that chains several facts into one item per fact in "items".',
    "delete only what is on the <never_in_the_profile> list; every real fact stays, filed under a better name.",
    "Keep every item in the bucket it came from; every claim and grounding comes from the shown items. Each index "
    "appears in one operation at most.",
)

REGROUP_REPLY_SKELETONS: dict[str, str] = {
    "explicit_info": f"""\
{{
  "operations": [
    {{"action": "merge", "type": "explicit_info", "indices": [0, 3], "data": {_EXPLICIT_ITEM_SKELETON}}},
    {{"action": "update", "type": "explicit_info", "index": 1, "data": {_EXPLICIT_ITEM_SKELETON}}},
    {{"action": "split", "type": "explicit_info", "index": 2, "items": [{_EXPLICIT_ITEM_SKELETON}, {_EXPLICIT_ITEM_SKELETON}]}},
    {{"action": "delete", "type": "explicit_info", "index": 4, "reason": "..."}}
  ]
}}""",
    "implicit_traits": f"""\
{{
  "operations": [
    {{"action": "merge", "type": "implicit_traits", "indices": [0, 3], "data": {_IMPLICIT_ITEM_SKELETON}}},
    {{"action": "update", "type": "implicit_traits", "index": 1, "data": {_IMPLICIT_ITEM_SKELETON}}},
    {{"action": "split", "type": "implicit_traits", "index": 2, "items": [{_IMPLICIT_ITEM_SKELETON}, {_IMPLICIT_ITEM_SKELETON}]}},
    {{"action": "delete", "type": "implicit_traits", "index": 4, "reason": "..."}}
  ]
}}""",
}
"""The REGROUP reply skeleton for each bucket, filled into the system template's ``{{reply_skeleton}}`` slot.

A group belongs to one bucket and every operation on it names that bucket and carries that bucket's item shape, so
the skeleton shows them. One fixed text per bucket keeps the system message identical for every group of a bucket.
"""

_REPLY_FORMAT_REGROUP = paragraphs(
    "After your operations the group may hold no more than {{max_per_category}} items under any one category or "
    f"trait name. index and indices are the index attributes in <group>; type is the group's bucket. {_REPLY_RULES}",
    "{{reply_skeleton}}",
)

PROFILE_REGROUP_FROM_EPISODE_TEXTS_SYSTEM_PROMPT = _system_prompt(
    _ROLE_REGROUP,
    _NEVER_IN_THE_PROFILE_STORED,
    tagged("how_to_regroup", _HOW_TO_REGROUP),
    tagged("reply_format", _REPLY_FORMAT_REGROUP),
)

PROFILE_REGROUP_FROM_EPISODE_TEXTS_USER_PROMPT = "\n\n".join(
    (
        _TARGET_LINE,
        tagged(
            "group",
            "{{items_text}}",
            bucket="{{bucket}}",
            label_field="{{label_field}}",
            label="{{label}}",
            count="{{count}}",
        ),
    )
)
