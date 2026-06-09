# Boundary Detection Flow

The boundary detection flow in benchmark stage 1. Splits a conversation's message list into multiple MemCells (memory units), where each MemCell represents a coherent conversation segment.

Scope: per-conversation. Stage 1 processes one conversation at a time.

---

## Step 1: Message Conversion

**Input**: raw conversation message list from the LoCoMo dataset

**Process** (`extract.py`):

1. Iterate over all messages in the conversation
2. Filter: keep only messages with `role` equal to `user` or `assistant` (`extract.py:80`). In the LoCoMo dataset, all messages have `role = "user"` (`loader.py:72`), so no messages are filtered out
3. Convert each message to `ChatMessage`: `id`, `role`, `content`, `timestamp` (ms epoch), `sender_id`, `sender_name`

**Output**: `list[ChatMessage]`, which in the LoCoMo scenario equals the full message list

---

## Step 2: Incremental Boundary Detection (Incremental Loop)

**Input**: `list[ChatMessage]` from Step 1

**Process** (`extract.py:225-286`, `user_memory/boundary.py:66-143`):

For each message, execute the following loop:

**2a. Front-2 Buffer** (`extract.py:273-276`):
- The first 2 messages are added directly to `history` without calling the LLM
- Starting from the 3rd message, each triggers one `adetect_step` call

**2b. Force-Split Short Circuit** (`boundary.py:113-121`):
- If `count_tokens(history + new) >= 8192` or `len(history) + 1 >= 50`, and `len(history) >= 2`
- Close `history` as a MemCell immediately without calling the LLM
- New history: when smart-mask is on, `[history[-1], new]` (bridge); otherwise `[new]`

**2c. Smart-Mask** (`boundary.py:111, 125`):
- Condition: `smart_mask=True` (default) and `len(history) > 5` (`smart_mask_threshold=5`)
- When active: the LLM only sees `history[:-1]` (masks everything before the last message), reducing token consumption
- Below threshold: the LLM sees the full `history`

**2d. Time Gap Calculation** (`boundary.py:126, 174-204`):
- Calculate the time difference between the last message in `history` and the `new` message
- Bucket into: <60s "immediate response", <1h "N minutes", <1day "N hours", >=1day "N days"
- Injected as `time_gap_info` into the prompt

**2e. LLM Judgment** (`boundary.py:127-133`):
- Call `adetect_boundary_step(masked_history, [new], llm=..., prompt=..., time_gap_info=...)`
- LLM returns JSON: `{should_end: bool, reasoning: str, confidence: float, topic_summary: str}`
- `gpt-4.1-mini`, temperature determined internally by `adetect_boundary_step`

**2f. State Transition** (`boundary.py:135-143`):
- `should_end=True`: close `history` as a MemCell, new history is `[history[-1], new]` (smart-mask bridge) or `[new]` (clean cut)
- `should_end=False`: `history = [*history, new]`, continue accumulating

**2g. Tail Flush** (`extract.py:282-284`):
- After all messages are processed, remaining messages in `history` are force-closed as the final MemCell

**Output**: `list[MemCell]`, where each MemCell contains a coherent conversation segment with `items: list[ChatMessage]` and `timestamp` (timestamp of the last message)

---

## Step 3: Per-MemCell Extraction (Episode + AtomicFact + Embedding)

**Input**: each MemCell from Step 2

**Process** (`extract.py:137-222`, `extract.py:349-356` parallel dispatch):

All MemCells are processed in parallel via `asyncio.gather`, with concurrency controlled by a semaphore (`extract.py:349-356`). For each MemCell:

**3a. Episode Extraction** (`extract.py:159-166`):
- `EpisodeExtractor(llm=llm).aextract(mc)` — compresses the MemCell's raw messages into a narrative summary
- Produces `episode.subject` (title) and `episode.episode` (body text)
- Empty body raises `ValueError` (fail-loud)
- JSON parse failures retry up to `max_attempts=5` times

**3b. Parallel: AtomicFact Extraction + Episode Body Embedding** (`extract.py:168-183`):
- `AtomicFactExtractor(llm=llm).aextract_from_text(episode_body)` — extracts verifiable atomic facts from the episode body
- Uses `EVENT_LOG_PROMPT` (`extract.py:174`)
- Simultaneously `embedding_client.embed([episode_body])` — embeds the episode body
- Both run in parallel via `asyncio.gather`
- No atomic facts returned raises `ValueError`

**3c. Atomic Fact Embedding** (`extract.py:189-190`):
- `embedding_client.embed(fact_strings)` — embeds all atomic fact strings
- Must wait for fact extraction in 3b to complete (sequential dependency)

**3d. Assembly and Serialization** (`extract.py:192-210`):
- `episode_summary = episode_body[:200] + "..."` (truncation, no LLM summarization, `extract.py:195`)
- Assembled into `EpisodeMemoryRecord` (subject, summary, content, embedding) + `AtomicFactRecord` (time, atomic_fact, fact_embeddings, timestamp)
- Serialized to dict

**Output**: each MemCell becomes a dict containing:
- `items`: original ChatMessage list
- `episode`: `{subject, summary, content, embedding}`
- `atomic_facts`: `{time, atomic_fact: [str, ...], fact_embeddings: [[float, ...], ...], timestamp}`

---

## Step 4: Clustering (Cluster by Geometry)

**Input**: all MemCells from Step 3 (each with an episode body embedding)

**Process** (`extract.py:435-467` -> `algorithm.py:33-63`):

MemCells are processed **sequentially in order** (cannot be parallelized; each step depends on the cluster state from the previous step, `extract.py:455-462`). For each MemCell:

**4a. Construct New Cluster** (`extract.py:388-393`):
- Use the MemCell's `episode.content_embeddings` as the centroid
- `last_ts` = the MemCell's timestamp
- `members` = `[mc_id]`
- `preview` = `[episode_body]`

**4b. Match Existing Clusters** (`_find_best_within_window`, `algorithm.py:168-192`):
- Iterate over all existing clusters, checking in order:
  1. **Time window**: `abs(new_ts - cluster.last_ts) > 7 days` (`cluster_max_time_gap_days=7.0`, `config.py`) -> skip
  2. **Cosine similarity**: cosine similarity between the new MemCell embedding and the cluster centroid
- Among all clusters within the time window, find the one with the highest cosine similarity (top-1)
- Highest score >= `threshold=0.70` (`cluster_similarity_threshold`, `config.py`) -> match successful
- Otherwise -> no match

**4c. Merge or Create** (`extract.py:400-405`, `algorithm.py:148-158`):
- **Match successful**: weighted-average centroid (`(existing.centroid * existing.count + new.centroid * 1) / total`), merge members lists, update `last_ts` to the larger of the two
- **No match**: create a new cluster, assign `id = "cluster_{len(existing)}"`

**Output**: `clusters_conv_<i>.json`, containing:
- `clusters`: each cluster's `{id, centroid, count, last_ts, members, preview}`
- `memcell_to_cluster`: `{memcell_id -> cluster_id}` reverse mapping

> **Key behavior**: the time window check happens before cosine — a cluster beyond the 7-day window will not be merged even if it is a perfect semantic match. This is why the adoption topic was split into 5 clusters in earlier experiments.
