"""Tests for EpisodeSummarizer: one call, a strict one-field reply, no retry, the extraction path's width guard.

Coverage layers:
- Contract  — the call requests ``_SummaryReply`` with a system and a user message; the sync bridge
- Error path — an invalid reply or a blank summary raises LLMOutputError without the reply; client exceptions propagate
- Input     — a blank ``episode_text`` or an unknown language is InvalidInputError before any call
- Summary   — an over-wide summary is cut at a sentence boundary, never re-requested
- Log lines — one INFO line of widths and the truncation flag, never text
"""

from __future__ import annotations

import inspect
import json
from typing import Any

import pytest
from pydantic import ValidationError

from everalgo.llm import LLMOutputError, StrictOutputModel, to_openai_response_format
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.user_memory import EpisodeSummarizer
from everalgo.user_memory.episode import EpisodeSummarizer as PackageEpisodeSummarizer
from everalgo.user_memory.episode._replies import _SummaryReply
from everalgo.user_memory.episode._summary import _SUMMARY_WIDTH_CAP
from everalgo.utils import InvalidInputError, ascii_width

_LOGGER = "everalgo.user_memory.episode"
_SECRET = "SUMMARY-SECRET-41c9"
_TEXT = "Alice met Bob on 2026-03-14 and they agreed to ship the report on Friday (2026-03-20)."
_SUMMARY = "Alice and Bob agreed to ship the report on 2026-03-20."


def _reply(summary: str = _SUMMARY) -> str:
    return json.dumps({"summary": summary})


async def _summarize(reply: str = "", text: str = _TEXT, **kwargs: Any) -> tuple[str, FakeLLMClient]:
    fake = FakeLLMClient(responses=[reply or _reply()])
    summary = await EpisodeSummarizer(llm=fake).asummarize(text, **kwargs)
    return summary, fake


class _ClientError(Exception):
    """A host's own call-failure type, which the operator must neither wrap nor retype."""


# ===========================================================================
# Contract
# ===========================================================================


def test_the_package_and_the_product_export_the_same_class() -> None:
    assert EpisodeSummarizer is PackageEpisodeSummarizer


def test_reply_model_is_strict_one_field_and_default_free() -> None:
    assert _SummaryReply.__bases__ == (StrictOutputModel,)
    assert list(_SummaryReply.model_fields) == ["summary"]
    assert _SummaryReply.model_fields["summary"].is_required()
    assert not _SummaryReply.__pydantic_decorators__.field_validators


def test_reply_model_fits_the_strict_subset() -> None:
    response_format = to_openai_response_format(_SummaryReply)

    assert response_format is not None
    assert response_format["json_schema"]["schema"]["type"] == "object"


def test_signature_takes_the_text_positionally_and_the_rest_by_keyword() -> None:
    parameters = inspect.signature(EpisodeSummarizer.asummarize).parameters

    assert list(parameters) == ["self", "episode_text", "output_language", "custom_instructions"]
    assert parameters["episode_text"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert parameters["output_language"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["custom_instructions"].kind is inspect.Parameter.KEYWORD_ONLY


async def test_valid_reply_becomes_the_summary_in_one_call() -> None:
    summary, fake = await _summarize()

    assert summary == _SUMMARY
    assert fake.call_count == 1


async def test_the_call_requests_the_reply_model_with_a_system_and_a_user_message() -> None:
    _, fake = await _summarize()

    assert fake.calls[0].response_format is _SummaryReply
    assert [message.role for message in fake.calls[0].messages] == ["system", "user"]


def test_sync_bridge_returns_the_summary() -> None:
    fake = FakeLLMClient(responses=[_reply()])

    assert EpisodeSummarizer(llm=fake).summarize(_TEXT) == _SUMMARY
    assert fake.call_count == 1


# ===========================================================================
# Error path — single call, no retry, no reply text in the message
# ===========================================================================


@pytest.mark.parametrize(
    ("reply", "failing_location"),
    [
        pytest.param(f"not json {_SECRET}", "<root>", id="not-json"),
        pytest.param(json.dumps({}), "summary:", id="summary-missing"),
        pytest.param(json.dumps({"summary": _SECRET, "title": "t"}), "title:", id="extra-field-forbidden"),
        pytest.param(json.dumps({"summary": 7}), "summary:", id="number-not-coerced"),
        pytest.param(json.dumps({"summary": None}), "summary:", id="null-not-accepted"),
    ],
)
async def test_reply_failing_the_model_raises_llm_output_error(reply: str, failing_location: str) -> None:
    fake = FakeLLMClient(responses=[reply])

    with pytest.raises(LLMOutputError) as excinfo:
        await EpisodeSummarizer(llm=fake).asummarize(_TEXT)

    assert isinstance(excinfo.value.__cause__, ValidationError)
    assert "episode summary reply is not a valid _SummaryReply" in str(excinfo.value)
    assert failing_location in str(excinfo.value)
    assert _SECRET not in str(excinfo.value)
    assert fake.call_count == 1


@pytest.mark.parametrize("blank", ["", "   ", "\n\t"])
async def test_blank_summary_raises_llm_output_error(blank: str) -> None:
    fake = FakeLLMClient(responses=[_reply(blank)])

    with pytest.raises(LLMOutputError) as excinfo:
        await EpisodeSummarizer(llm=fake).asummarize(_TEXT)

    assert str(excinfo.value) == "episode summary reply leaves fields blank: summary: empty after stripping whitespace"
    assert fake.call_count == 1


async def test_client_exception_propagates_unchanged() -> None:
    error = _ClientError("upstream down")
    fake = FakeLLMClient(responses=[error])

    with pytest.raises(_ClientError) as excinfo:
        await EpisodeSummarizer(llm=fake).asummarize(_TEXT)

    assert excinfo.value is error


# ===========================================================================
# Input — rejected before any call
# ===========================================================================


@pytest.mark.parametrize("blank", ["", "  ", "\n\t "])
async def test_blank_episode_text_is_invalid_input_before_any_call(blank: str) -> None:
    fake = FakeLLMClient(responses=[])

    with pytest.raises(InvalidInputError, match=r"got a blank episode_text, there is nothing to preview: pass "):
        await EpisodeSummarizer(llm=fake).asummarize(blank)

    assert fake.call_count == 0


async def test_unsupported_output_language_is_invalid_input_before_any_call() -> None:
    fake = FakeLLMClient(responses=[])

    with pytest.raises(InvalidInputError, match=r"output_language 'Klingon' names no supported language"):
        await EpisodeSummarizer(llm=fake).asummarize(_TEXT, output_language="Klingon")

    assert fake.call_count == 0


async def test_long_text_is_sent_whole() -> None:
    """No length bound: the host owns its input budget, and a silent cut would preview a different text."""
    text = "Alice met Bob. " * 2000

    _, fake = await _summarize(text=text)

    assert text in fake.calls[0].messages[1].content


# ===========================================================================
# Summary width guard — the extraction path's, truncation in place
# ===========================================================================

_LONG_SUMMARY = "SUMMARY_ONLY_MARKER. The team met and decided many things. " * 30


async def test_summary_at_the_cap_passes_untouched() -> None:
    at_cap = "a" * _SUMMARY_WIDTH_CAP

    summary, _ = await _summarize(_reply(at_cap))

    assert summary == at_cap


async def test_overwide_summary_is_cut_at_a_sentence_boundary_without_a_second_call() -> None:
    summary, fake = await _summarize(_reply(_LONG_SUMMARY))

    assert fake.call_count == 1
    assert ascii_width(summary) <= _SUMMARY_WIDTH_CAP
    assert summary.endswith("…")
    body = summary.removesuffix("…")
    assert (body + ". ") in _LONG_SUMMARY


async def test_overwide_cjk_summary_is_cut_at_a_sentence_boundary() -> None:
    long_cjk = "林然和周敏排查了支付接口的超时问题。" * 30

    summary, _ = await _summarize(_reply(long_cjk))

    assert ascii_width(summary) <= _SUMMARY_WIDTH_CAP
    assert summary.endswith("…")
    assert long_cjk.startswith(summary.removesuffix("…") + "。")


# ===========================================================================
# Log lines — widths and the truncation flag, never text
# ===========================================================================


async def test_a_compliant_summary_logs_one_line_without_text(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("INFO", logger=_LOGGER):
        summary, _ = await _summarize(text=f"{_SECRET} {_TEXT}")

    logged = [record.message for record in caplog.records]
    expected = (
        f"episode summarized: text {ascii_width(f'{_SECRET} {_TEXT}')} units, "
        f"summary {ascii_width(summary)} units, truncated=False"
    )
    assert logged == [expected]
    assert all(record.name == _LOGGER for record in caplog.records)


async def test_a_truncated_summary_logs_the_guard_lines_and_the_flag(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("INFO", logger=_LOGGER):
        summary, _ = await _summarize(_reply(_LONG_SUMMARY))

    by_level = [(record.levelname, record.message) for record in caplog.records]
    original = ascii_width(_LONG_SUMMARY)
    assert by_level == [
        ("WARNING", f"episode summary over cap ({original} > {_SUMMARY_WIDTH_CAP} units)"),
        ("INFO", f"episode summary truncated: {original} -> {ascii_width(summary)} units"),
        (
            "INFO",
            f"episode summarized: text {ascii_width(_TEXT)} units, summary {ascii_width(summary)} units, truncated=True",
        ),
    ]
    assert not any(_SECRET in message or "MARKER" in message for _, message in by_level)
