"""English prompts for LanguageDetector: name the primary language of a chat record.

The system message is the task, the judging rule, one worked example and the reply line; the user message is the
record, one ``[role] text`` block per chat message inside ``<chat_record>``, with no sender names. The text was
selected by measurement on ``qwen3-4b-instruct-2507``, two runs each with the same answer both times, every message cut
to its head, middle and tail as :class:`~everalgo.user_memory.LanguageDetector` cuts it by default. On 271 production
conversations labelled by hand, 190 scored and 81 shown but not scored (task briefs, agent dispatches, platform notices
and test tenants), it named the labelled language for 189 scored and 77 unscored records; on 13 conversations written
to mix one language's words into another, for 12. The scored miss is a Chinese conversation dense with English
technical terms, named English; the written one is an English conversation carrying a whole Chinese address, named
Chinese. These are development figures — the text was chosen against these same sets — so fresh held-out data has to
confirm them.

What the rule encodes, and why:

- The record as a whole, not one participant: the question is what language the conversation is carried on in.
- The users first, the assistant when the users' text cannot show it: a one-word message, an image, or a task brief
  a scheduler or another agent sent on a user's behalf (usually English whatever the user speaks).
- Of the assistant's messages, only what it says to the user: an agent narrating its own steps in English before a
  Chinese report is still a Chinese conversation.
- Carried material counts for nothing: pasted documents, code, logs and paths are in whatever language they were
  written in.
- A few words of another language do not move the conversation: names, places, brands, terms and short phrases from
  other languages are ordinary in any conversation, and the sentences decide. The worked example shows it, an English
  exchange carrying two Chinese words named English; before it, English conversations carrying a Chinese address,
  company, landmark or phrase were named Chinese. The renderer drops sender names for the same reason.

The labels are Chinese, English, Japanese and Other, Other standing for every other language; the detector returns it
as English, the one language written for everybody else. Alternatives measured and dropped: listing every
``OutputLanguage`` value beside Other (the model never chose Other, naming Thai and Persian records Chinese); three
labels with English defined as "any other language" and no example value (about forty Chinese records named English,
pure Chinese conversations among them); against the previous text, which said only that names and IDs say nothing
about the language, the general rule without the example (2 scored records fixed, 2 broken) and the example without
the rule (4 fixed, 1 broken), where this text fixed 7 and broke 1; and a second, opposite example, a Chinese exchange
carrying English terms named Chinese, beside the first: 11 Chinese records then named English with the rule and 19
without it.

Constants:
    - ``LANGUAGE_LABELS`` — the labels the reply may give, in the order the prompt lists them.
    - ``LANGUAGE_SYSTEM_PROMPT`` — the static system message. No placeholder.
    - ``LANGUAGE_USER_PROMPT`` — the data. Placeholder: ``{{chat_record}}``.

Output schema: ``{"language": <one of LANGUAGE_LABELS>}``, held to ``language._LanguageReply``.
"""

from __future__ import annotations

__all__ = ["LANGUAGE_LABELS", "LANGUAGE_SYSTEM_PROMPT", "LANGUAGE_USER_PROMPT", "assemble_system_prompt"]

from everalgo.utils import append_custom_instructions

# The measured order; ``language._LanguageReply`` and a test hold the set to these four.
LANGUAGE_LABELS = ("Chinese", "English", "Japanese", "Other")

LANGUAGE_SYSTEM_PROMPT = (
    "You are given a chat record. Name its primary language: the language the conversation itself is carried on in."
    "\n\n"
    "Judge the record as a whole from what the participants write to each other. The users' messages count first. "
    "The assistant's replies decide when the users' messages are too short to show a language, or are a task brief "
    "or template sent on a user's behalf rather than written to the assistant. Of the assistant's messages, only what "
    "it says to the user counts, such as answers and reports; notes on what it is doing while it works do not. "
    "Material that is only carried along is not the conversation: pasted documents, code, logs, error messages, tool "
    "output, quoted passages, file paths. A few words in another language do not change the conversation's language: "
    "people routinely mix in names, places, brands, terms and short phrases from other languages. Judge by the "
    "language the sentences are written in."
    "\n\n"
    "Example:\n<chat_record>\n"
    "[user] I've been finding cafés on 小红书, but every 种草 post looks the same. Any better way to pick one?\n\n"
    "[assistant] Check the post dates and cross-check the map reviews; recent photos are a good sign.\n"
    '</chat_record>\n{"language": "English"}'
    "\n\n"
    'Reply as JSON, {"language": "Chinese"}, using one of: ' + ", ".join(LANGUAGE_LABELS) + "."
)

LANGUAGE_USER_PROMPT = "<chat_record>\n{{chat_record}}\n</chat_record>"


def assemble_system_prompt(*, custom_instructions: str | None) -> str:
    """Build the system message: the static prompt, then the caller's instructions when there are any.

    Args:
        custom_instructions: The caller's instructions for the ``<custom_instructions>`` block, the message's last,
            placed verbatim and never scanned for placeholders. ``None`` or a blank string leaves the block out.

    Returns:
        The system message.
    """
    return append_custom_instructions(LANGUAGE_SYSTEM_PROMPT, custom_instructions)
