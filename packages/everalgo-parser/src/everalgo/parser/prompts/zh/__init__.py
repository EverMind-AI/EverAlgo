"""Chinese prompts for parser operators."""

from everalgo.parser.prompts.zh.audio import PROMPT_FOR_AUDIO
from everalgo.parser.prompts.zh.document import HTML_USER_PROMPT, PROMPT_FOR_FILE, PROMPT_FOR_HTML
from everalgo.parser.prompts.zh.image import MERGE_USER_PROMPT, PROMPT_FOR_MERGE, PROMPT_FOR_PICTURE

__all__ = [
    "HTML_USER_PROMPT",
    "MERGE_USER_PROMPT",
    "PROMPT_FOR_AUDIO",
    "PROMPT_FOR_FILE",
    "PROMPT_FOR_HTML",
    "PROMPT_FOR_MERGE",
    "PROMPT_FOR_PICTURE",
]
