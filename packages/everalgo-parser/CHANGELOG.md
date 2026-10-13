# Changelog — everalgo-parser

All notable changes to this distribution will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- git-cliff-unreleased-start -->
## [Unreleased]

<!-- git-cliff-unreleased-end -->

## [0.3.0rc2] - 2026-10-13

### Changed

- Invalid caller input now raises `everalgo.utils.InvalidInputError` (a `ValueError` subclass, so existing handlers keep working) instead of a bare `ValueError`: a `RawFile` with neither content nor uri, with empty content, with an unsupported extension or modality, a uri that returns an empty body or a content type the parser cannot read. Each message names the field and what to set instead; a uri it quotes is cut to `scheme://host[:port]/path`, so a signed URL's query string, fragment and credentials never reach the message.
- Requires `everalgo-core>=0.6.0rc2` (the floor was `0.6.0rc1`): the package imports `everalgo.utils.InvalidInputError`, which `0.6.0rc1` does not define.

## [0.3.0rc1] - 2026-10-13

### Changed

- **BREAKING:** every LLM call follows the new `LLMClient` contract: it passes `response_format=_TextReply`, a strict pydantic model of one field, `{"text": "..."}`, reads the `str` `achat` returns and validates it with `everalgo.llm.parse_reply`. A reply that is not that object — plain text, a missing or extra key, a non-string `text` — raises `everalgo.llm.LLMOutputError` from the image, PDF, HTML, Office and audio parsers. Two algorithm-level fallbacks keep their behaviour and now also cover an invalid reply: a tall-image merge that fails or comes back empty joins the slices' OCR (`metadata["merge_fallback"] = "deterministic_join"`), and an inline email image whose OCR fails gets the `[image: <name>, OCR failed]` placeholder. A custom client passed as `llm` must accept the mandatory keyword and return the reply text. See `everalgo-core` for the contract.
- **BREAKING:** every call sends a system message and a user message instead of one user message. The system message is the prompt's rules, unchanged, followed by a `<reply_format>` block asking for `{"text": "..."}` (shared by the English and Chinese sets, `prompts/_reply_format.py`); `PROMPT_FOR_PICTURE`, `PROMPT_FOR_MERGE`, `PROMPT_FOR_FILE`, `PROMPT_FOR_HTML` and `PROMPT_FOR_AUDIO` are now these system messages. The user message is the input alone: the image, PDF or audio part with no text beside it; for HTML the cleaned page in `<html_content>` (new `HTML_USER_PROMPT`, slot `{{html}}`) instead of after a `---` line; for the tall-image merge the OCR texts, each in a `<part number="N">`, in `<ocr_parts count="N">` (new `MERGE_USER_PROMPT`, slots `{{count}}` and `{{parts}}`) instead of `=== Part N ===` sections in `PROMPT_FOR_MERGE`'s removed `{{parts_text}}` slot.
- **BREAKING:** `ParsedContent.metadata` no longer carries `model` or `finish_reason`: `achat` returns only the reply text, and the model actually served is the host's to record. The image, audio, PDF, HTML and Office results keep their other keys.
- **BREAKING:** operators call `LLMClient.achat`, the new name of the protocol method; a custom client passed as `llm` must implement `achat` instead of `chat`. See `everalgo-core` for the contract.
- The URL parser fetches through `everalgo.utils.fetch_uri`, which moved to `everalgo-core`; the `ValueError` for a scheme other than `http` / `https` now reads `fetch_uri: only http and https are supported, got scheme 'file'` instead of `fetch_uri: only http/https supported, got scheme='file'. file:// and other local-filesystem URIs are rejected by design (...)`, so match on the exception type, not the text.
- `everalgo-parser` no longer declares `httpx` itself: it makes no HTTP call of its own any more, and `everalgo-core` declares `httpx>=0.27` for the fetch.

## [0.2.1] - 2026-06-15

### Changed

- **License relicensed from MIT to Apache-2.0** as part of the pre-open-source security audit.

## [0.2.0] - 2026-05-27

> First archived changelog. Entries below accumulated since the initial `0.1.0` PyPI release (published manually, without a git tag or per-package changelog), so this `0.2.0` section also consolidates the previously-unarchived `0.1.0` surface.

### Added

- **PDF parser** — single multimodal LLM call, mirrors the upstream
  internal multimodal PDF parser reference implementation.
- **Image parser** — single LLM call for normal-ratio images; tall
  screenshots (height/width > 10) are split into overlapping vertical
  slices, OCR'd per slice, and merged with a second LLM call using
  `PROMPT_FOR_MERGE`. BMP / TIFF are transcoded to PNG via Pillow before
  upload (Gemini does not accept those MIMEs natively); SVG is rasterised
  via the optional `cairosvg` extra (`everalgo-parser[svg]`).
- **Audio parser** — ASR via multimodal LLM; covers
  mp3 / wav / m4a / amr / aiff / aac / ogg / flac.
- **HTML / `htm` handler** (in `document.aparse`) — bs4 noise cleanup
  (`clean_html_for_llm` drops `<script>` / `<style>` / `<nav>` / `<footer>`
  / `<iframe>` and bloat attributes) then LLM extraction.
- **Email (`.eml`) handler** (in `document.aparse`) — stdlib `email`
  parsing for headers + body, with inline-image OCR via `cid:` placeholder
  substitution.
- **Office document handler** (in `document.aparse`) — covers docx / xlsx
  / pptx / doc / xls / ppt / pages / key / numbers / odt / ods / odp /
  rtf via a LibreOffice subprocess (`soffice --convert-to pdf` → reuse PDF
  path). `soffice` must be installed on the host (see README).
- **URL parser** — fetches `http`/`https` URIs via httpx, then **dispatches
  by the fetched `Content-Type`**: `application/pdf` → PDF handler,
  `image/*` → image handler, `audio/*` → audio handler, `text/html` (or
  unknown) → HTML handler with Open Graph / Twitter Card / `<meta>` tag
  extraction into `ParsedContent.metadata`. `file://` and other local-
  filesystem URIs are rejected (AGENTS.md §1). Inner modality recorded in
  `metadata["inner_modality"]`. Metadata-extractor schema lifted from the
  upstream internal URL extractor reference implementation.
- **MIME-aware top-level dispatch** — `parser.aparse` now falls back to
  `mime` when `extension` is missing or `UNKNOWN`, so the canonical
  `RawFile(uri=..., mime="application/pdf")` example actually routes to
  the PDF handler. Added `Modality` helpers in `everalgo-core/types/
  modality.py`: `MIME_TO_MODALITY` / `MIME_TO_EXTENSION` /
  `get_modality_from_mime(mime)` / `get_extension_from_mime(mime)`.
- **Chinese prompts** at `prompts/zh/{image,audio,document}.py`
  (`PROMPT_FOR_PICTURE` / `PROMPT_FOR_MERGE` / `PROMPT_FOR_FILE` /
  `PROMPT_FOR_HTML` / `PROMPT_FOR_AUDIO`), mirroring the English set.
- **Optional `[svg]` extra** pulling in `cairosvg` for SVG rasterisation.
- **Re-exports** of `ParsedContent` / `Modality` / `RawFile` from
  `everalgo.parser` so callers can `from everalgo.parser import aparse,
  RawFile, ParsedContent`.

### Notes

- `video` submodule remains a stub — the upstream internal multimodal parser
  has no video implementation to port; backend selection pending an ADR.
- The parser layer deliberately omits retry / fallback / multi-key
  rotation / Prometheus metrics. Those are deployment concerns
  (ADR-012); algorithm-layer operators surface failures via `LLMError`
  and callers wrap as needed.

### Dependencies

- `pillow >= 10.0.0` — image transcode + aspect-ratio check.
- `beautifulsoup4 >= 4.12.0` — HTML cleanup + OG metadata extraction.
- `asgiref >= 3.8.0` — sync bridge (ADR-010).
- Optional `cairosvg >= 2.7.0` via `[svg]` extra.
- System: LibreOffice (`soffice` on PATH) for Office document parsing.

[Unreleased]: https://github.com/EverMind-AI/EverAlgo/compare/everalgo-parser/v0.3.0rc2...HEAD
[0.3.0rc2]: https://github.com/EverMind-AI/EverAlgo/compare/everalgo-parser/v0.3.0rc1...everalgo-parser/v0.3.0rc2
[0.3.0rc1]: https://github.com/EverMind-AI/EverAlgo/compare/everalgo-parser/v0.2.1...everalgo-parser/v0.3.0rc1
[0.2.1]: https://github.com/EverMind-AI/EverAlgo/compare/everalgo-parser/v0.2.0...everalgo-parser/v0.2.1
[0.2.0]: https://github.com/EverMind-AI/EverAlgo/releases/tag/everalgo-parser/v0.2.0
