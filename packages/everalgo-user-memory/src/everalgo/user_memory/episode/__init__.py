"""Write one Episode — a titled third-person narrative with a short summary — from a MemCell.

Modules, each importing only those below it: ``_extractor`` (:class:`EpisodeExtractor`) wires ``_render`` (the system
and user messages the model is shown), ``_replies`` (the reply model and the LLM callsite held to it) and ``_summary``
(the width guard on the stored summary).

Log lines, all on ``everalgo.user_memory.episode`` whichever module writes them; none carries a prompt or model text:

    ``extracting episode: <n> items, <n> rendered chars, output_language=<value>`` (INFO)
        Once per call, before the model is asked. ``rendered chars`` is the user message's length: the
        conversation as rendered in its ``<conversation>`` element. ``output_language`` is the caller's argument as
        given.
    ``episode extracted: content <n> units, summary <n> units`` (INFO)
        Once per call that returns an Episode, widths in ASCII-equivalent units after the summary guard.
    ``episode summary over cap (<n> > <cap> units)`` (WARNING)
        The model wrote a summary wider than the cap; its rate is the production model's violation rate.
    ``episode summary truncated: <n> -> <n> units`` (INFO)
        Follows every over-cap line: the stored summary's width after the sentence-boundary cut.
"""

from __future__ import annotations

__all__ = ["EpisodeExtractor"]

from everalgo.user_memory.episode._extractor import EpisodeExtractor
