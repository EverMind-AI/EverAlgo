# Changelog — everalgo-parser

All notable changes to this distribution will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- git-cliff-unreleased-start -->
## [Unreleased]
<!-- git-cliff-unreleased-end -->

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

[Unreleased]: https://github.com/EverMind-AI/EverAlgo/compare/everalgo-parser/v0.2.0...HEAD
[0.2.0]: https://github.com/EverMind-AI/EverAlgo/releases/tag/everalgo-parser/v0.2.0
