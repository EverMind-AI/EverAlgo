"""English prompts for LanguageDetector: name the primary language of a chat record.

The system message is the task, the judging rule and the reply line; the user message is the record, one
``[role] text`` block per chat message inside ``<chat_record>``, with no sender names. The text was selected by
measurement on ``qwen3-4b-instruct-2507``, five runs each, every message cut to its head, middle and tail as
:class:`~everalgo.user_memory.LanguageDetector` cuts it by default: on 219 production conversations labelled by hand,
157 scored and 62 automation records (task briefs, agent dispatches, platform notices) shown but not scored, it named
the labelled language in every run of 155 scored and 61 automation records. The two scored misses are pulled to
Chinese: an English conversation about a company with a Chinese name, and a Persian one. These are development-set
figures — the prompt was revised against both sets — so fresh held-out data has to confirm them.

What the rule encodes, and why:

- The record as a whole, not one participant: the question is what language the conversation is carried on in.
- The users first, the assistant when the users' text cannot show it: a one-word message, an image, or a task brief
  a scheduler or another agent sent on a user's behalf (usually English whatever the user speaks).
- Of the assistant's messages, only what it says to the user: an agent narrating its own steps in English before a
  Chinese report is still a Chinese conversation.
- Carried material and names count for nothing: pasted documents, code, logs and paths are in whatever language they
  were written in, and a sender's name says nothing about the language they write. The renderer drops names too.

The labels are Chinese, English, Japanese and Other, Other standing for every other language; the detector returns it
as English, the one language written for everybody else. Three alternatives were measured and dropped. Listing every
``OutputLanguage`` value beside Other: the model never chose Other, naming Thai and Persian records Chinese. Three
labels with English defined as "any other language" and no example value: about forty Chinese records were named
English, pure Chinese conversations among them. A closing sentence that names, terms and a requested reply language
do not change the language written: no miss fixed, the Thai record back to Chinese.

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
    "output, quoted passages, file paths. Names and IDs say nothing about the language."
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
