# everalgo-knowledge

Extracts a flattened topic tree (`list[KnowledgeMemory]`) from parsed content (`ParsedContent`) via LLM.

## Use

```python
from everalgo.knowledge import KnowledgeExtractor
from everalgo.types import ParsedContent

parsed = ParsedContent(text="...", mime="text/markdown")
memories = await KnowledgeExtractor(llm=client).aextract(parsed, doc_id="doc1", title="My doc")
# memories[0] = synthetic doc root; rest = DFS-ordered topic nodes.
```

`llm`: any `everalgo.llm.LLMClient`, passed at construction. Every call sends a system message of rules and a user message of data, requests a strict reply model as `response_format`, and raises `everalgo.llm.LLMOutputError` when the reply does not validate (no retry; the host decides). Sync bridge: `extract(...)`.

`custom_instructions=` (on `aextract` and `aclassify_category`) adds the host's text to the bundled rules, never replacing them: it goes, verbatim and never rendered as a template, in a `<custom_instructions>` block closing the system message of every LLM call the extraction makes; `None` or a blank string adds nothing.

## Pipeline at a glance

```
ParsedContent
  → preprocess + atomize           (_block_split)
  → topic-tree LLM (per batch)     (prompts.en.topic_tree_extract)
  → cross-batch merge (>1 batch)   (prompts.en.content_merge + topic_merge)
  → postprocess: split unsplit leaves + assign uncovered blocks
  → tree assembly + DFS flatten    → list[KnowledgeMemory]
```

## Install

```bash
pip install everalgo-knowledge    # pulls everalgo-core + everalgo-parser
```

For development within the monorepo workspace:

```bash
uv sync --package everalgo-knowledge
```

## Live-LLM scripts

Run from the **repository root**. Put `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` in a `.env` file there first; each command block below sources it inline and is paste-and-run.

Smoke test — 2 fixtures, structural assertions only, no artifacts written:

```bash
set -a; source .env; set +a
uv run pytest -m integration packages/everalgo-knowledge/tests/functional/test_extraction_smoke.py
```

Run one doc — dumps result (JSON + optional HTML viz) under `tests/functional/outputs/`:

```bash
set -a; source .env; set +a
uv run python packages/everalgo-knowledge/examples/run_one_doc.py \
  packages/everalgo-knowledge/tests/functional/fixtures/idx_multi_topic.json --html
```

## Related distributions

- [`everalgo-parser`](../everalgo-parser/) — produces `ParsedContent` from raw files (fully implemented except video, which is deferred pending ADR)
- [`everalgo-core`](../everalgo-core/) — `KnowledgeMemory` type is defined here
