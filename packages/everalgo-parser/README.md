# everalgo-parser

Multimodal parsing — image / audio / document / video / url raw inputs into `ParsedContent`. Used by `everalgo-knowledge` for file ingestion and by evermem step 1 for inline parsing.

See the umbrella project: [EverAlgo monorepo](../../README.md) and the architecture document at [`docs/design.md`](../../docs/design.md).

## Quick start

```python
import everalgo
from everalgo.llm.config import LLMConfig
from everalgo.llm.providers.openai_compat import OpenAICompatClient
from everalgo.parser import aparse, RawFile

# Configure an LLM once (process-wide). The parser uses OpenAI-compatible
# clients; OpenRouter is the reference deployment (Gemini multimodal via
# OpenRouter passthrough).
everalgo.configure(OpenAICompatClient(LLMConfig(
    model="google/gemini-3-flash-preview",
    api_key="sk-or-v1-...",
    base_url="https://openrouter.ai/api/v1",
)))

# Bytes-in: caller already hydrated the file.
parsed = await aparse(RawFile(content=pdf_bytes, extension="pdf"))
print(parsed.text)

# URL-in: parser fetches over HTTP, then delegates to the HTML handler.
parsed = await aparse(RawFile(uri="https://example.com/article"))
print(parsed.metadata["title"], parsed.text[:500])
```

## Supported formats

| Modality | Extensions | Backend |
|----------|------------|---------|
| `PDF` | `pdf` | Multimodal LLM (single call, full doc) |
| `IMAGE` | `png` / `jpg` / `jpeg` / `webp` / `bmp` / `tiff` / `tif` / `svg` | Multimodal LLM; BMP/TIFF transcoded to PNG via Pillow; SVG rasterised via `cairosvg`; tall screenshots split + merged |
| `AUDIO` | `mp3` / `wav` / `m4a` / `amr` / `aiff` / `aac` / `ogg` / `flac` | Multimodal LLM ASR |
| `HTML` | `html` / `htm` | bs4 cleanup → LLM extraction |
| `EMAIL` | `eml` | stdlib `email` + inline-image OCR via the LLM |
| `DOCUMENT` | `docx` / `pptx` / `xlsx` / `doc` / `ppt` / `xls` / `pages` / `key` / `numbers` / `odt` / `ods` / `odp` / `rtf` | LibreOffice `soffice --convert-to pdf` → reuse PDF path |
| `URL` | (any `http`/`https` URI) | httpx fetch → HTML handler |
| `DIRECT` | `txt` / `md` / `csv` / `tsv` / `vtt` | UTF-8 decode, no LLM |
| `VIDEO` | — | Deferred (no upstream implementation; ADR pending) |

## Installation

```bash
pip install everalgo-parser            # core: pdf / image / audio / html / eml / direct / url
pip install 'everalgo-parser[svg]'     # adds SVG support (cairosvg)
```

### System dependency for Office documents

Office document parsing (docx / xlsx / pptx / …) shells out to **LibreOffice**, which is a system package, not a pip wheel. Install before parsing Office files:

```bash
# Ubuntu / Debian
sudo apt-get install -y libreoffice

# macOS
brew install --cask libreoffice
```

The parser detects `soffice` via `shutil.which("soffice")` and the canonical macOS Applications path. Missing → `RuntimeError` with install instructions when an Office file is parsed; non-Office paths are unaffected.

## Conventions

- `aparse(...)` is async; `parse(...)` is the sync bridge via `asgiref.async_to_sync`.
- Prompts live as module-level string constants under `prompts/{en,zh}/<operator>.py` ([AGENTS.md §5](../../AGENTS.md#5-code-style)). Swap languages by re-binding the constant at startup.
- The library is **stateless**: it never reads the filesystem and never owns business state. HTTP I/O (LLM calls, URL fetching) is explicitly allowed.
- No retry / fallback / metrics inside operators — surface failures via `LLMError`, let the caller wrap.

## Reference

- Architecture (definitive): [`docs/design.md`](../../docs/design.md) §1.2 / §2.3
- Schema source for PDF / image / audio / document / html / email: `evermemos-multimodal` (tag `prod-20260306-0331-v1`).
- Schema source for URL metadata extraction: `evermemos-opensource/src/common_utils/url_extractor.py`.
