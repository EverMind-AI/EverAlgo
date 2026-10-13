# LLM Contract

Status: final design, 2026-09-23. A Chinese companion is available: [llm-contract.zh-CN.md](llm-contract.zh-CN.md); this English version is the repository reference.

This document defines the contract between EverAlgo operators and the `LLMClient` implementation a host injects into them. It describes what the contract is, not how existing code migrates to it.

Position: EverAlgo is the party that specifies the protocol; it defines the contract and the conformance checks. How a client is constructed, how it retries, how it routes, and where its defaults live are entirely the implementation's business.

The key words "MUST", "MUST NOT", "SHOULD", "SHOULD NOT", and "MAY" in this document are to be interpreted as described in [RFC 2119](https://www.rfc-editor.org/rfc/rfc2119).

| Term | Meaning |
|---|---|
| operator | An EverAlgo algorithm that calls `LLMClient.achat` |
| implementation | The host-side `LLMClient` implementation |
| host | The application that builds the implementation, injects it into operators, and receives their exceptions; the implementation and the host are the same party |

---

## 1. Overview

### 1.1 In one sentence

One method, one call, multimodal input, **JSON output**. An operator declares the structure it wants with a pydantic model, then parses the reply and guards against bad output itself.

```
          Operator (EverAlgo)                        Implementation (host)
  ┌──────────────────────────────┐        ┌──────────────────────────────────┐
  │ prompt: asks for JSON        │        │ how to make the model produce    │
  │         + a parseable sample │ ─────► │   this structure                 │
  │ response_format: model class │ needs  │ how to validate, how to retry    │
  │                              │        │   — all the implementation's own │
  │ model_validate_json(text)    │ ◄───── │ expected: what comes back is     │
  │   fails → LLMOutputError     │  text  │   already a valid result         │
  │ business check               │        │                                  │
  │   fails → LLMOutputError     │        │                                  │
  └──────────────────────────────┘        └──────────────────────────────────┘
    knows the format, not the model          knows the model, not the strategy
```

### 1.2 Where the contract draws the line: can EverAlgo detect it?

| Can EverAlgo detect it? | Examples | How the contract treats it |
|---|---|---|
| **No** | An explicit `temperature` overwritten by a default; an image silently dropped, so the model may make up valid JSON about empty content; the upstream flags a refusal but the refusal text sits inside a valid JSON field; the upstream reports truncation but the body happens to be a closed, valid object | **Hard requirements (MUST)** (Section 3); required checks in the conformance suite |
| **Yes** | Output is not valid JSON; a required field is missing; truncation before the top-level object closes | **Recommendations only** (Section 7). The operator's parse failure raises `LLMOutputError` as a backstop (Section 4), but at that point the host can only rerun the whole operator, the most expensive option; handling it in the implementation's own layer first is cheapest |

---

## 2. Interface

### 2.1 Signature

```python
async def achat(
    self,
    messages: list[LLMMessage],
    *,
    response_format: type[BaseModel],     # required
    temperature: float | None = None,
    max_tokens: int | None = None,
    **extra: Any,
) -> str          # the model's output text; expected to pass response_format.model_validate_json
```

### 2.2 `response_format`: a pydantic model

The model is defined by EverAlgo and MUST satisfy the four rules below (enforced by EverAlgo's own unit tests; see 4.3):

| Rule | Why |
|---|---|
| **The top level is an object**: a `BaseModel` subclass; `RootModel` is not accepted | The truncation guard in 4.2 depends on this. `RootModel` is itself a `BaseModel` subclass (verified: `issubclass(RootModel, BaseModel) is True`), and the schema of `RootModel[int]` has `integer` at the top level, so a truncated number still parses |
| **Structural constraints only** | An implementation that validates against the model retries on failure; when a business rule fails, a retry usually does not help and only burns budget |
| **Inherit from `StrictOutputModel` (nested sub-models included); no field defaults** | So that local validation and the strict schema accept the same inputs; see below |
| **Stay within the strict subset** (8.2) | Otherwise the implementation cannot use native schema constraints, the most reliable path |

#### Defaults, type coercion, and extra fields are output semantics too

When an implementation uses native schema constraints, what it sends is the model's strict schema, built by EverAlgo's request builders: this makes **every field required** and adds `additionalProperties: false` (Section 8). A pydantic model with the default configuration judges differently during local validation. Verified:

| Input | Default model (`count: int = 0`) | `strict=True, extra="forbid"`, no default | strict schema |
|---|---|---|---|
| `{}` | Accepted, filled in as `count=0` | Rejected | Rejected (required) |
| `{"count":"3"}` | Accepted, coerced to the integer 3 | Rejected | Rejected (`type: integer`) |
| `{"count":3,"extra":true}` | Accepted, extra field ignored | Rejected | Rejected (`additionalProperties: false`) |
| `{"count":3}` | Accepted | Accepted | Accepted |

With a default model, the same operator always receives every field when the implementation uses native schema constraints, but under JSON mode or prompt-only a missing field is silently filled with its default and `"3"` is silently coerced to 3. **What the operator receives then depends on which technique the implementation chose**, and the implementation is free to choose it (3.2).

So models are written like this:

```python
class StrictOutputModel(BaseModel):
    """Base for every response_format model: config only."""

    model_config = ConfigDict(strict=True, extra="forbid")


class Fact(StrictOutputModel):
    content: str

class ReflectOutput(StrictOutputModel):
    facts: list[Fact]          # sub-models inherit the base too, otherwise the config does not carry over
```

- **Sub-models MUST inherit it too.** Pydantic configuration propagates only through inheritance, not through field types: for nested models the official docs say "each model has its own 'configuration boundary'" (*Configuration*, "Configuration propagation"); verified: with `forbid` set on the parent but not on the child, extra fields inside the child object are still accepted.
- **Fields MUST NOT have defaults.** A field that may be absent is written as `X | None`, and the model returns `null` explicitly (verified: under strict, a missing key is rejected and `null` is accepted, matching the strict schema). The base class cannot enforce this rule; the unit test in 4.3 checks it.
- Where a default value or lenient type coercion is genuinely needed, it MUST be deliberate, and the operator's unit tests MUST cover the prompt-only path.

#### Boundaries of `StrictOutputModel`

| Rule | Why |
|---|---|
| **Config only**: no fields, methods, or validators | Any behaviour would turn it into a superfluous abstraction layer |
| **One level only**: every model inherits it directly; no intermediate base class is derived from it | Avoids "change the parent and every child changes with it" |
| **Not part of the protocol**: it lives in everalgo-core for operator packages to import; the protocol signature stays `type[BaseModel]` | The implementation does not need to know about it |

Sharing configuration through a parent class is the approach pydantic's official docs recommend (*Configuration*, "Change behaviour globally": "you can create your own custom parent class with a custom configuration, as the configuration is inherited"). Host codebases already use the same pattern: a base class that holds only `model_config`, inherited by several operation models. Python itself (PEPs, official docs) makes no recommendation on this; it is a library convention. Django's "Abstract base classes" follow the same idea.

#### The line between structural constraints and business rules

| In the model (structural constraints) | After the operator parses (business rules) |
|---|---|
| Field types, required fields, `Literal` enums, nested sub-models | Whether values are reasonable, whether the result is empty, cross-field consistency, **whether the content is complete** (for example, whether OCR dropped part of the source; 4.2) |
| `Field` constraints expressible in JSON Schema (length, `pattern`) | Custom validators such as `field_validator` / `model_validator` |

### 2.3 Multimodal input

Multimodal input goes through `LLMMessage.content`, which accepts either `str` or `list[ContentPart]` (text plus image / pdf / audio data URIs). Serialised, it is the OpenAI / OpenRouter wire shape.

### 2.4 Naming

The protocol family uses a uniform `LLM` prefix: `LLMClient` / `LLMMessage` / `LLMOutputError`. How a client is constructed is not part of the protocol: it is the implementation's business. EverAlgo's own reference implementation, `OpenAICompatClient`, takes keyword arguments, as the openai and anthropic SDK clients do. The return value has no wrapper type; it is simply `str` (rationale in 6.4).

`LLMMessage` and the domain type `everalgo.types.chat.ChatMessage` are **two different things**. The latter is a conversation record with `id` / `timestamp` / `sender_id` / `sender_name`, and its `role` is only user or assistant. The former has only `role` and `content`, its `role` includes system, and it is one request sent to a model. The two are never converted into each other: an operator such as boundary detection renders domain messages into text and sends that text as the content of a user `LLMMessage`.

**The method is named `achat`: `chat`, not `generate`, with the `a` prefix.** Its input is a list of chat-style messages, and `chat` describes exactly that shape. `generate` comes from prompt-in interfaces such as HuggingFace `model.generate` and Google `models.generate_content`, which do not match this signature. The mainstream SDKs agree without exception: OpenAI has `chat.completions.create` / `.parse` (response type `ChatCompletion`), Anthropic has `messages.create` (response type `Message`), and litellm has `completion()` / `acompletion()` (response type `ModelResponse`). Type names need a prefix because they appear out of context in imports and annotations; a method name always hangs off an `LLMClient` instance, so it has no such problem. The `a` prefix is the repository-wide rule that every native-async operator name carries it; litellm draws the same line between `completion()` and `acompletion()`. There is no synchronous `chat` beside it: the protocol is what a host implements, a sync twin would double that work, and operators only ever call it from async code; the sync bridges live one level up, on the operators.

### 2.5 Errors: the contract defines only the one EverAlgo raises

```python
class LLMOutputError(Exception):
    ...   # the model's reply does not meet the operator's requirements (parse failure or failed business check)
```

| Exception | Raised by | When |
|---|---|---|
| `LLMOutputError` | **The operator** | Parse failure, or a failed business check (4.1) |
| The implementation's own exceptions | The implementation | Call failure, unsatisfiable parameters, upstream refusal or truncation, and so on. **The type is the implementation's choice; the contract does not prescribe it** |

**The contract does not define exceptions on the implementation's behalf.** The implementation and the host are the same party: the implementation raises, the host catches, and EverAlgo only sits in between, never catching them by type. What was checked (as of 2026-09-23):

| Check | Result |
|---|---|
| Operators catching call failures by type | **None.** The only two degradation points (in `everalgo.parser` document and image parsing) both use `except Exception` without looking at the type. `everalgo.user_memory` episode extraction no longer degrades: it makes one call, validates the reply against a strict model, and lets the client's exceptions through unchanged |
| Hosts relying on call exceptions defined by EverAlgo | No. A production host defines its own exception hierarchy and never imports EverAlgo's call exceptions |

So EverAlgo has only one rule to keep here: **exceptions raised by the implementation propagate unchanged, neither wrapped nor retyped** (4.3), so that the host can recognise them by its own types. The exceptions a host sees come from only three sources:

| Exception seen | Meaning |
|---|---|
| One of its own implementation's types | The call failed |
| `LLMOutputError` | The model's output did not meet the requirements |
| Anything else | An EverAlgo bug, or the caller passed invalid arguments |

`LLMOutputError` means only "the model did not produce the required output"; **whether to retry is the host's own decision** (Section 5).

> **Invalid arguments have a type too.** The table's last row says "the caller passed invalid arguments"; since this change that case raises `everalgo.utils.InvalidInputError` (a `ValueError` subclass), the caller-side counterpart of `LLMOutputError`. See "Two error types" in the `everalgo-core` README.

**When the protocol would need to define a call exception:** when an operator must distinguish "this call failed" from "some other error" and handle them differently. The operator does not know the host's types, so the protocol would then have to provide a common type. The existing degradation points need no such distinction, so none is defined.

---

## 3. Hard requirements for the implementation: what EverAlgo cannot detect

### 3.1 Explicit values are delivered as given

If a value is passed, that value is used; `None` means the implementation's configured default decides; if neither exists, the parameter is omitted from the request and left to the server. **The implementation's defaults MUST NOT override the caller's explicit values, wherever those defaults are stored.**

This holds for **every** upstream request, including retries the implementation initiates itself. In particular, the implementation **MUST NOT raise `max_tokens` to work around truncation**: the caller's value is the budget it wants, and raising it changes cost and latency on the caller's behalf.

**Why this is a hard requirement:** once these values are changed, the model still returns valid JSON, and EverAlgo sees nothing wrong.

### 3.2 `messages` MUST NOT be silently dropped or rewritten

Deliver what can be delivered; when delivery is known to be impossible (for example, an image sent to a text-only model), raise an error whose type the implementation chooses (2.5). **Silent dropping is forbidden.** An **equivalent format translation** made to fit the upstream **is not a rewrite**: the same semantics look different in different vendors' request bodies, and translation is the adapter's job (counterexample in 6.6).

**Why this is a hard requirement:** in the parser's OCR call, if the image is dropped the model may still follow the prompt and make up a well-formed `{"text": "..."}` (inference, not tested). Every check in the operator would then pass, wrong content would go straight into storage, and EverAlgo would have no way to notice.

Uncertainty caused by upstream servers silently ignoring parameters is **not the implementation's responsibility**: OpenAI-compatible servers ignore unknown parameters by default instead of returning an error, and the contract **has no standing** to require servers to return one.

**`response_format` is not covered by this rule.** Which format parameter to send, and whether to send one at all, is the implementation's technique, and EverAlgo does not prescribe it (recommendations in 7.2). A format requirement therefore can never be "undeliverable".

### 3.3 Cancellation propagates unchanged

The implementation MUST NOT swallow `asyncio.CancelledError` (for example, by catching it and continuing to retry).

**Why this is a hard requirement:** some operators issue several calls concurrently (`everalgo.agent_memory.case`, `everalgo.rank.agentic`, and `everalgo.rank.hybrid` use `asyncio.gather` / `TaskGroup`). A swallowed cancellation disables timeouts and cancellation in the layers above, and EverAlgo cannot notice.

### 3.4 An explicit upstream refusal MUST raise

When the upstream **explicitly flags** that the model refused to answer, the implementation MUST raise (a type of its choosing) and MUST NOT return the refusal text or empty content as a result.

**Why this is a hard requirement:** whether the operator's parse catches a refusal depends on what the refusal looks like. Verified (`model_validate_json`):

| What the refusal looks like | Operator parse |
|---|---|
| Plain text "Sorry, I can't help with this request." | Rejected |
| Empty content | Rejected |
| `{"text": "Sorry, I can't help with this request."}` | **Accepted** |

The third is exactly the `{"text": "..."}` shape that free-text call sites use (6.1). It is valid JSON with the right structure, and the operator cannot tell it apart. Only the implementation sees the refusal signal:

| Upstream | Refusal signal | Source |
|---|---|---|
| OpenAI / OpenAI-compatible | Non-empty `message.refusal` | openai SDK 2.35.1 `openai/types/chat/chat_completion_message.py:63` |
| Anthropic | `stop_reason: "refusal"` | Official docs *Handling stop reasons*: Claude declined to respond. |

Feasibility: the reference implementation, `everalgo.llm.openai_compat`, raises when it sees `refusal`; a production host adapter already raises on empty content, which covers API refusals.

**This rule covers only refusals the upstream explicitly flags.** If the model is not flagged and merely says "cannot process this" in natural language inside a field, the contract does not ask the implementation to guess, and the operator's generic parse cannot tell either. This is a known limitation: what business checks can recognise, the operator checks itself; what they cannot recognise stays unrecognised.

### 3.5 An explicit upstream report of truncation or abnormal termination: that body MUST NOT be returned

When the upstream **explicitly reports** that this generation was truncated or terminated abnormally, the implementation MUST retry or raise (a type of its choosing) and **MUST NOT return that body directly**. A retry is still bound by 3.1: `max_tokens` MUST NOT be raised.

**When the finish reason is missing, the implementation is not required to infer completeness.**

**Why this is a hard requirement:** the same reason as 3.4. The operator can detect truncation only if the object is left unclosed (4.2). The body of the following response passes the parse:

```
content       = {"text": "partial content"}
finish_reason = length
```

Verified: `model_validate_json` returns `text='partial content'` normally. Bodies that happen to be closed do occur: for example, the model hits the limit right after writing the closing brace, or the implementation or a gateway repairs the truncated JSON by closing the brackets (the latter is inference, not tested). Only the implementation sees the finish reason.

| Upstream | Truncation / abnormal-termination signal | Source |
|---|---|---|
| OpenAI / OpenAI-compatible | `finish_reason` is `length` or `content_filter` | openai SDK 2.35.1 `openai/types/chat/chat_completion.py:25` |
| vLLM and other compatible gateways | `finish_reason` is `abort` or `error` | Observed in practice and already handled by a production host adapter; see below |
| Anthropic | `stop_reason` is `max_tokens` or `model_context_window_exceeded` | Official docs *Handling stop reasons* |

`content_filter` means the upstream stopped generating because of content filtering; it is classed as abnormal termination.

Feasibility: a production host adapter already treats `length`, `abort`, `error`, and `content_filter` as failures.

**This rule does not promise complete content, nor does it hand the finish reason to the operator.** It only requires the implementation to act on explicit signals it can see; without a signal, content completeness belongs to business checks (4.2).

---

## 4. Operator side

### 4.1 How the operator handles the return value

```python
text = await self._llm.achat(messages=[...], response_format=ReflectOutput)
output = ReflectOutput.model_validate_json(text)   # fails → raise LLMOutputError
check_business_rules(output)                       # fails → raise LLMOutputError
```

1. **Always parse the reply itself.** Do not assume the implementation has validated it. A parse failure raises `LLMOutputError`.
2. **Check business rules** (the right-hand column of the line table in 2.2). A failure also raises `LLMOutputError`.

The operator **never retries** (Section 5) and **does not need `finish_reason`**: truncation the upstream reports is handled by the implementation (3.5), and truncation that leaves the top-level object unclosed is caught in step 1 (4.2). Making the operator match OpenAI's `length` or Anthropic's `max_tokens` would only leak provider knowledge into the algorithm layer.

### 4.2 Parsing catches "the object never closed", not "the content is incomplete"

**Truncation that leaves the top-level JSON object unclosed is caught by the parse; a successful parse does not guarantee complete content, and content completeness is the job of business checks.**

Basis for the first half: with the top level fixed as an object (2.2), truncation before the object closes always fails to parse. Verified (`model_validate_json`):

- For a 26-character object, **all** 25 of its proper prefixes fail validation;
- Enumerating every proper prefix by JSON value type: any truncation of an object, array, string, `true`, or `null` is invalid; **only a top-level number** stays valid after truncation (`12345` cut to `123`). Section 2.2 rules out a non-object top level, which closes that path too.

So whether or not the implementation validates, and whether or not the finish reason is reported, **syntactically unfinished JSON** never reaches business logic. The case where the upstream reports truncation but the body happens to be closed is stopped at the implementation's layer by 3.5.

Basis for the second half: the proper-prefix experiment only proves that "truncation before the object closes fails"; it cannot prove that content is complete. `{"text": "only part of the content was extracted"}` is a closed, valid object, and the parse cannot see whether it left out part of the source. That judgement belongs to business checks: what can be judged, the operator judges; what cannot is a known limitation.

### 4.3 Operator-side obligations

These are rules EverAlgo keeps for itself. The implementation cannot see strategy content, so **they are checked by EverAlgo's own unit tests and are not part of the implementation's conformance suite** (Section 9).

| Obligation | Why | How EverAlgo's unit tests check it |
|---|---|---|
| **The prompt states that the output is JSON and gives a parseable example** | The model learns what to output only from the prompt; if the implementation sends no format parameter, the prompt is the only constraint. It is required under JSON mode too: the parameter description in openai SDK 2.35.1 (`openai/types/beta/assistant_create_params.py:93-99`) requires separately instructing the model to output JSON in a message, otherwise the model may produce an "unending stream of whitespace" until it hits the token limit | The rendered prompt contains the format instruction; the example passes `model_validate_json` on the corresponding model |
| **The model carries structural constraints only** | 2.2 | No custom validators on the model |
| **The model (nested sub-models included) inherits `StrictOutputModel` and has no field defaults; exceptions MUST be deliberate** | 2.2: otherwise what the operator receives varies with the implementation's technique | Walk every `response_format` model and all of its sub-models, asserting each is a `StrictOutputModel` subclass and no field has a default; operators with an exception have a unit test covering the prompt-only path |
| **The model's top level is an object, and the model stays within the strict subset** | 2.2 / 8.2 | Run `to_openai_response_format` on every `response_format` model; it does not raise |
| **Parse the reply itself; raise `LLMOutputError` for both parse failures and failed business checks, never a bare `ValueError`; never retry on its own** | 4.1 / Section 5 | Operator unit tests |
| **Exceptions raised by the implementation propagate unchanged, neither wrapped nor retyped** | The host relies on its own types to recognise call failures (2.5). An explicit degradation point MAY catch the exception, but it MUST be deliberate | Operator unit test: a test double raises a custom exception, and the test asserts it surfaces unchanged |

Example: both reflect prompts (`everalgo.user_memory.prompts.en.reflect`) say "Reply with one JSON object, nothing else."

---

## 5. Who retries: operators never retry; the host decides by error type

| Layer | What it does | On failure |
|---|---|---|
| Implementation | What the hard requirements demand (Section 3); recommended to validate and retry in its own layer (7.1) | Decides on its own whether to retry or raise |
| **Operator** | Parse + business checks | Raises `LLMOutputError`; **never retries** |
| **Host** | Decides by error type whether to retry, dead-letter, or skip | — |

The operator has no retry mechanism, and should not have one: it does not know the budget, the cost, or which pipeline this call is on. Whether another generation is worth it is a question only the host can answer.

**Why the `LLMOutputError` type exists:** if an operator's parse failure raised a **bare `ValueError`**, the host could not tell "the model emitted garbage" from "the operator has a bug". It could only guess the source from the traceback and then fall back to the conservative disposition that is safe in case the bug is its own, which means treating the error as non-retryable. **As a result, an occasional malformed output is permanently dead-lettered**, when a single retry would most likely have fixed it.

**But an exception type describes the source of a failure, not whether it is recoverable:**

- `LLMOutputError` is **not necessarily worth retrying**: if the prompt itself is broken, or a business rule is one the model can never satisfy, retrying any number of times gives the same result;
- a caller passing invalid arguments gets `everalgo.utils.InvalidInputError` (a subclass of `ValueError`), which the host reads by type as terminal; a bare `ValueError` that still escapes is a programming error on the EverAlgo side (a reply model outside the strict schema subset, a prompt template placeholder, a client constructor argument), never a data problem the host can act on.

So the contract promises only that **the source is distinguishable**; whether to retry is decided by the host from budget, pipeline position, and retries already made.

**Retrying at the host is the most expensive layer**: it retries the whole operator call, re-reading data and re-rendering prompts, and a multi-pass operator such as UPDATE reruns every pass. That is why Section 7 recommends the implementation handle failures in its own layer first. If the implementation also retries internally, the two layers' retry counts multiply, and the host has to account for that when setting its policy.

---

## 6. Key trade-offs and rationale

### 6.1 Why every output MUST be JSON, with no free-text tier

**Benefits:**

- **Clean separation of responsibilities.** Whether to use JSON and which structure to use is decided by the strategy prompt and belongs to EverAlgo; which techniques the model supports belongs to the implementation. As long as every output is JSON, the host never needs to know which format each strategy wants.
- **The operator can guard itself.** Validation needs structure. If free-text output were allowed, truncated text would look like normal text and the operator could not detect it; if any JSON value were allowed at the top level, a truncated number would still parse. With every output a top-level object, truncation before the object closes always fails to parse (4.2). Content completeness still depends on business checks.

**Costs:**

1. **Call sites that naturally want free text need a wrapper.** Impact (as of 2026-09-23, non-test code, rough per-file screen):

   | Call site type | Count | How to write it |
   |---|---|---|
   | Its file already parses JSON | 26 | Declare the corresponding model |
   | Outputs free text (`everalgo.parser` image 3, document 4, audio 1) | 8 | Wrap in a model shaped like `{"text": "..."}` |
   | Already declares a schema (reflect) | 1 | Nothing |

   Once free text goes inside a JSON string, newlines and quotes must be escaped, so it costs somewhat more tokens.

2. **Whether format restrictions hurt reasoning quality is contested.** Tam et al., *Let Me Speak Freely?* (arXiv:2408.02442, 2024-08-05), observed a clear drop in reasoning ability under format restrictions, larger the stricter the format; the .txt team's reply *Say What You Mean* argues that this is a prompt-writing problem, and that with well-written prompts structured output actually scores 1–4 percentage points higher. A common mitigation is to put the fields that need reasoning before the conclusion fields, and the reflect prompt is already ordered that way. This has to be settled by experiment, not by literature.

3. **Dynamic-key `dict[str, X]` cannot be used** (it returned 400 in the 8.2 test); use `list[sub-model]` instead.

### 6.2 Why `response_format` is a pydantic model and not a JSON Schema dict

| | JSON Schema dict | pydantic model |
|---|---|---|
| Converting to a strict schema | Each operator converts on its own | The implementation calls EverAlgo's request builder for its API when it needs one (Section 8) |
| When the implementation wants to validate | It has to pull in the `jsonschema` library | Call `model_validate_json` directly |
| Do the implementation's and the operator's **local validations** agree? | No: when an `int` field receives `"3"` or an extra field appears, the schema and a default pydantic model judge oppositely (verified) | Yes, it is the same model |
| How the operator writes it | `response_format=converter(M)` | `response_format=M` |

Note that "local validations agree" does not mean "the three generation techniques constrain identically"; the latter requires the model to be configured as in 2.2.

The dict approach's only advantage is that the implementation does not have to write the roughly 50-line recursive "pydantic to strict schema" rewrite itself. Once EverAlgo's request builders do that rewrite, that advantage is gone. The OpenAI SDK's `chat.completions.parse()` also takes the model class directly (a pydantic `BaseModel`; dataclasses are accepted too; openai 2.35.1 `openai/lib/_parsing/_completions.py:270-277`). everalgo-core already depends on pydantic, and the protocol is a Python protocol, so "the implementation has no pydantic" cannot happen.

### 6.3 The call surface keeps four parameters, and `model` is not one of them

| Parameter | Owner | Conclusion | Basis |
|---|---|---|---|
| `response_format` | **The operator**: the structure is decided by the strategy prompt | Required | 6.1 |
| `max_tokens` | The host (budget); the operator overrides it when it knows its own output shape | Optional | everalgo-parser sets two budgets, 8 000 and 32 000, for two stages of the same pipeline, which a host's single per-scene config cannot express |
| `temperature` | The host (tuning bound to the model, and the host chose the model) | Optional | Of 35 call sites (as of 2026-09-23), 5 override it for a concrete reason (for example, OCR merging pins it to 0.0 for determinism); the rest leave it to the host |
| `model` | The host (scene routing + fallback to another model) | **Not on the call surface** | **Zero** of the 35 call sites (as of 2026-09-23) pass it; a host adapter receives it only to discard it |

"The model actually served" does need to be recorded (fallback switches models, gateways route), but **the one who needs it is the host, not the operator**: the host's provider already knows where it routed, and billing happens in that layer too. So it is not in the return value either; see 6.4.

### 6.4 The return value is just `str`, with no wrapper type

Every field was checked against operators' actual reads (as of 2026-09-23):

| Field | Operator reads | Conclusion | Reason |
|---|---|---|---|
| `content` | 41 | **It is the return value itself** | The business payload |
| `model` | 9 | ❌ | All 9 are **in everalgo-parser**, and all merely stuff it into `ParsedContent.metadata`; those two keys have **no consumer other than parser's own unit tests**, and benchmarks read it zero times. Attribution is the host's business; see 6.3 |
| `finish_reason` | 8 (**zero branch on it**) | ❌ | Likewise all in parser's metadata. Truncation the upstream reports is handled by the implementation (3.5), and truncation that leaves the object unclosed is caught by the operator's parse (4.2); it has no decision value |
| `parsed` | 3 | ❌ | The operator's own `model_validate_json` gives the exact type, with no cast needed |
| `usage` | **0** | ❌ | The implementation is already on the billing path itself, with no need to go through the operator |
| `raw` | **0** | ❌ | Same as above |

With only one field left, there should be no wrapper: **`achat()` returns `str` directly**. The operator always parses once itself (4.1); even if the implementation has parsed, it has no need to pass the result object back.

The only argument for keeping an empty shell is evolution: adding a field to a dataclass is backward compatible, whereas changing `str` into an object is not. But the table above shows that what operators will need in the future is also zero: `usage`, `finish_reason`, and retry counts are all host concerns. If the day comes when something must be added, adding a new method is cleaner than carrying an empty shell forever.

**Knock-on effect**: `ParsedContent.metadata` has no `model` or `finish_reason` keys.

### 6.5 `messages: list[LLMMessage]`, not `prompt: str`

"A list of messages with roles" is the shape every vendor shares, and it naturally accommodates multimodal content, system prompts, and multi-turn few-shot examples. Switching to `prompt: str` would require inventing two more parameters, `system` and `attachments`: two home-made mechanisms to work around a problem that needs no workaround.

But **do not treat it as a wire format that is identical field by field across vendors**: the detailed differences are real (see 6.6), and they are exactly what an adapter has to translate.

### 6.6 Why the contract does not prescribe the wire shape: a concrete counterexample

The differences between the Anthropic Messages API and OpenAI are not minor details (official docs checked on 2026-09-22):

| | OpenAI / OpenRouter | Anthropic Messages |
|---|---|---|
| System prompt | A `role="system"` entry in `messages` | **A top-level `system` parameter**; the docs say: there is no `"system"` role for input messages in the Messages API |
| Structured output | `response_format` | **No such parameter**; it uses `output_config.format`: `{"type": "json_schema", "schema": {...}}` |
| Images / PDF | `image_url` / `file` content part | `image` / `document` content block |

An Anthropic adapter that lifts the `role="system"` message into the top-level `system` is making an equivalent translation, which 3.2 allows. Whether it produces structured output through `output_config.format` or relies on the prompt alone is its own technique, and the contract does not care. Had the contract said "forward OpenAI fields verbatim", this correct implementation would violate it everywhere.

---

## 7. Recommendations for the implementation (not part of the contract)

None of the following is a contract requirement (they are SHOULD-level recommendations, non-normative); EverAlgo neither checks nor relies on them. They are written down because following them keeps the host's retry cost lowest.

The recommended flow puts the hard requirements of Section 3 and the recommendations below in order; `OpenAICompatClient.achat` follows it step by step (7.4). Each step is marked MUST (Section 3) or SHOULD (this section):

1. **Build the request once**, so every attempt sends it unchanged.
   - Send an explicit `temperature` / `max_tokens` exactly as given; only for `None` fall back to your own default, or leave the parameter out (MUST, 3.1).
   - Deliver every content part, non-text ones included, or raise (MUST, 3.2).
   - Choose the format parameter by what the model supports (SHOULD, 7.2).
2. **Send it until a reply passes every check**, a bounded number of times (SHOULD, 7.1, 7.3). Each attempt:
   1. **Send the request.** Let `asyncio.CancelledError` propagate (MUST, 3.3). Whether to resend after a transport failure is the implementation's own choice.
   2. **Check what the upstream reported.** A refusal raises (MUST, 3.4). A truncated or abnormally stopped generation is never returned (MUST, 3.5): raise on `length` or `content_filter`, which the identical request hits again; `abort` and `error` may be resent like a transport failure (SHOULD, 7.3).
   3. **Validate the reply** against `response_format` with `model_validate_json`, and send the request again if it fails (SHOULD, 7.1).

### 7.1 Validate before handing the reply to the operator

**Reason: retry granularity.** When the implementation finds the output unacceptable, it can simply resend one call on the spot, without even re-rendering the prompt; if the operator is left to find it, the host can only rerun the whole operator call (Section 5).

Recommended practice:

- Validate with `model_validate_json` on **the same model the operator supplied**, so the judgement matches the operator's exactly;
- Retry on failure, with a limit; once retries are exhausted, raise the implementation's own exception;
- When the finish reason is missing, go by the validation result (some gateways omit the finish reason on otherwise usable responses; judging by validation avoids discarding them).

Refusals, truncation, and abnormal termination that the upstream explicitly flags are not covered here: those are hard requirements (3.4, 3.5).

Pydantic's actual verdicts (verified, `model_validate_json`): anything other than the JSON itself is rejected, so such a reply is retried rather than handed over:

| Input | Result |
|---|---|
| Complete JSON | Accepted |
| Leading / trailing whitespace | Accepted |
| Extra trailing characters | Rejected |
| Wrapped in a ` ```json ` fence | Rejected |
| Preceded by a sentence of explanation | Rejected |

**Anti-pattern:** downgrading validation along with the request.

```python
json_mode = {"type": "json_object"}
text = await provider.achat(..., response_format=json_mode)
json.loads(text)      # only confirms valid JSON; not validated against the model, so a reply missing fields still reaches the operator
return text
```

This does not violate the contract (the operator's parse will still catch the problem), but it pushes a failure that should have been handled in the implementation's layer up to the layer where retrying costs the most.

### 7.2 Choose the format parameter by model capability

| The current model supports | What to send |
|---|---|
| Native schema constraints | The strict schema, from the builder for the API (Section 8); highest first-attempt success rate |
| JSON mode only | The JSON-mode parameter |
| Neither | Nothing; rely on the prompt alone (4.3 guarantees that the prompt states the format) |

The choice affects routing, so weigh it yourself. Measured (2026-09-22, `z-ai/glm-5.3-flash`, 32 endpoints):

| Format parameter in the request | Available endpoints |
|---|---|
| None | 32 |
| JSON mode | 29 |
| Native schema constraints | **24** (8 providers do not support structured outputs, including the official Z.AI provider) |

Moreover, OpenRouter's soft preference looks at the **value**: in the same experiment, with routing pinned towards three providers that do not support `response_format`, sending nothing or `{"type": "text"}` landed on the target provider all three times, while sending JSON mode pushed the request away all three times.

**EverAlgo's own reference implementation provides this switch.** `OpenAICompatClient` takes `response_format_mode: Literal["json_schema", "json_object", "prompt_only"]`, defaulting to `json_schema`. The core utility `everalgo.llm.to_openai_response_format(model, mode)` turns it into the OpenAI Chat Completions `response_format` parameter, so any OpenAI-compatible implementation can reuse it instead of writing its own (8.4). Because it is a constructor argument, the same operator can be compared across the three techniques by changing one value, with no change to operator code.

The value is named `prompt_only`, not `text`: it means "send no format parameter and rely on the prompt alone", and the model output **is still JSON**. Comparing against real free-text output requires changing the prompt and the parsing logic; that is an experiment on the EverAlgo side, not something a client config can do (6.1).

### 7.3 Retries and observability

- Retries need a limit, otherwise "hand back a valid result" turns into unbounded attempts;
- **Do not retry what the identical request hits again**: `length` and `content_filter`. Retrying the same prompt **does not necessarily** truncate again, but since `max_tokens` cannot be raised (3.1), the identical request runs into the same cap or trips the same filter, and retrying gambles with the caller's budget. `abort` and `error` are different: the server gave up on a generation it could have finished (preemption, an upstream blip), so resending them like a 5xx is reasonable. A production host adapter draws the same line: it raises at once on `length` and `content_filter` and retries `abort` and `error`;
- Record which format technique each call used and how many retries it took. Only with this data can the host see that "the model configured for this scene needs 3 retries on average; time to switch", and budget correctly for the multiplied retries of the two layers (Section 5).

### 7.4 The reference implementation follows all of the above

`OpenAICompatClient` is the implementation EverAlgo ships as the example to copy, so it follows every hard requirement in Section 3 and every recommendation in this section:

| Recommendation | What `OpenAICompatClient` does |
|---|---|
| Validate before handing over (7.1) | Validates each reply with the operator's own `response_format` model |
| Bounded retries (7.1, 7.3) | A reply that fails validation is retried at most `invalid_reply_retries` times (default 1); once they run out it raises `LLMError` with the last `ValidationError` as `__cause__` |
| Same request on every attempt (3.1) | The request is built once, so every attempt carries the caller's `max_tokens` unchanged |
| Do not retry what the identical request hits again (7.3) | A reported refusal, `length` or `content_filter` raises at once; `abort` and `error` are resent within `transport_retries` |
| Missing finish reason (7.1) | Decided by validation alone |
| Choose the format parameter (7.2) | `response_format_mode`, built by `to_openai_response_format` |
| Record technique and attempts (7.3) | Logs a warning for every failed validation with the attempt number, the model and the `response_format_mode`, and for every resend after a transient failure with the reason, the backoff and the attempt number |

It sends the request with httpx directly and depends on no provider SDK, so everalgo-core pulls none into a host that brings its own client. A connection error, a timeout, HTTP 408, 409, 429 or 5xx, a 200 with no choices (how OpenRouter answers some upstream rejections), or a generation the server aborted (`abort`, `error`) is resent at most `transport_retries` times (default 2), after an exponential backoff or the upstream's `Retry-After`; the statuses and backoff constants are those of openai SDK 2.35.1 (`openai/_base_client.py`).

The conformance suite runs against it with every required and every optional check passing (Section 9).

---

## 8. Utility functions for implementations

### 8.1 The strict schema every builder starts from

The `json_schema` mode of every builder (8.4) starts from one rewrite of `model.model_json_schema()`: every node that already has `type: "object"` and **has not declared** `additionalProperties` gets `additionalProperties: false`, and its `required` is filled with every property name; `default` entries whose value is `null` are stripped; `$ref` with sibling keys is expanded; a discriminated union's `oneOf` becomes `anyOf` and its `discriminator` is dropped (8.2); and it recurses through `$defs` / `definitions` / `properties` / `items` / `anyOf` / `allOf`.

These rewrites make it stricter than a pydantic model with the default configuration. Only when the model is configured as in 2.2 do it and local validation accept the same inputs.

The rewrite is private to `everalgo.llm.schema` and not exported: each API wraps the schema differently and each builder already returns the whole request value, so an implementation never needs the bare schema. An implementation for an API that has no builder yet can take `to_openai_response_format(model)["json_schema"]["schema"]` until one is added (8.4).

An OpenAI-compatible implementation assembles it like this:

```python
from everalgo.llm import to_openai_response_format

openai_response_format = to_openai_response_format(response_format, "json_schema")
# {"type": "json_schema",
#  "json_schema": {"name": response_format.__name__, "schema": <strict schema of response_format>, "strict": True}}
```

The basis for `strict: True`: the OpenAI docs say "Structured Outputs is a feature that ensures the model will always generate responses that adhere to your supplied JSON Schema", and "We recommend always using Structured Outputs instead of JSON mode when possible".

### 8.2 Supported subset

The rewrite is mechanical; it cannot itself repair structures that strict mode cannot express. Verified against the live API on 2026-09-22 with `openai/gpt-5.4-mini` (`strict: true`):

| pydantic construct | Result |
|---|---|
| Flat fields | ✅ |
| `X \| None` (nullable) | ✅ rendered as `anyOf` including `null` |
| Nested sub-models (`$defs` + `$ref`) | ✅ |
| `list[sub-model]` | ✅ |
| `Literal["a", "b"]` | ✅ |
| Fields with defaults | ✅ (the default is stripped; the field still goes into `required`) |
| A sub-model with `Field(description=...)` (produces `$ref` with sibling keys) | ✅ passes after expansion |
| Self-referencing recursive models | ✅ |
| Discriminated union (`Field(discriminator=...)`) | ✅ after the rewrite: `oneOf` becomes `anyOf`, `discriminator` is dropped; not part of the 2026-09-22 run, see below |
| **`dict[str, int]` (dynamic keys)** | **❌ 400** |

The pydantic schema for `dict[str, int]` is `{"type": "object", "additionalProperties": {"type": "integer"}}`. It **already** declares `additionalProperties`, so the rewrite does not override it: overriding it with `false` would turn "any key allowed" into "no key allowed", changing what the model means. Left alone, the node does not satisfy strict mode; the API says:

> Invalid schema for response_format 'WithDict': In context=(), 'required' is required to be supplied and to be an array including every key in properties. Extra required key 'counts' supplied.

So a dynamic-key `dict` is outside the supported subset; use `list[sub-model]` instead.

Unlike the dynamic-key `dict`, a discriminated union can be rewritten without changing what the model means. pydantic renders `Annotated[A | B, Field(discriminator="kind")]` as `{"oneOf": [...], "discriminator": {"propertyName": "kind", "mapping": {...}}}`, and openai SDK 2.35.1's `to_strict_json_schema` passes both keys through. Strict mode does not take `oneOf`: the "Supported types" list in the OpenAI Structured Outputs docs names `anyOf` and not `oneOf`, and the API error quoted in openai-agents-python issue #474 reads `In context=('properties', 'results', 'items'), 'oneOf' is not permitted.` Renaming it changes no reply the schema accepts: each arm fixes the discriminator field to its own `const`, so at most one arm matches and `anyOf` accepts the same replies, while local validation against the model (7.1) still picks the arm by the discriminator. `discriminator` is OpenAPI vocabulary that JSON Schema validation never reads (pydantic's own source marks it as OpenAPI), and none of the four APIs in 8.4 lists it, so it is dropped rather than moved into the description.

The rewritten shape has not been sent to the live API yet. Its ✅ rests on the documented `anyOf` support and the error above, not on a live call like the other rows; confirm it with one before the release that ships it.

Two common claims also disagree with the measurements: `pattern` **is supported** (in the same test run, schemas containing `pattern` returned 200; the restriction mainly concerns fine-tuned models), and the combining keyword that is genuinely unsupported is `allOf`. The rewrite flattens a single-element `allOf`, while a multi-element `allOf` is left as is and rejected at request time.

### 8.3 Build-time checks

When it meets a structure outside the subset, the rewrite raises as soon as the builder is called, instead of producing a schema that only fails with 400 at request time. The error message says which field, why it is not allowed, and what to use instead:

| Case | Handling |
|---|---|
| `additionalProperties: false` | Already satisfies strict mode; **kept, no error** |
| `additionalProperties: true` (open object) | Raises |
| `additionalProperties` is a schema (dynamic-key `dict[str, X]`) | Raises, suggesting `list[sub-model]` |
| Multi-element `allOf` | Raises (a single element is flattened) |
| `oneOf` next to an `anyOf` on the same node (pydantic never emits this) | Raises; merging the two would change what the node accepts |
| Top-level type is not `object` (e.g. `RootModel[int]`) | Raises; violates 2.2 |

These checks serve both sides: the operator's unit tests rely on them to confirm that the model stays within the subset (4.3), and the implementation never receives a schema that is bound to fail with 400.

### 8.4 Request builders: `to_<api>_response_format`, one per API

Each builder takes the `response_format` model and a `mode`, and returns what one API expects in its request:

| Function | Returns | `mode` |
|---|---|---|
| `to_openai_response_format` | Value of `response_format` (OpenAI Chat Completions and every OpenAI-compatible endpoint) | `json_schema` / `json_object` / `prompt_only` |
| `to_anthropic_response_format` | Value of `output_config.format` (Anthropic Messages) | `json_schema` / `prompt_only`: Anthropic has no JSON-mode level |
| `to_gemini_response_format` | The `generationConfig` entries `responseMimeType` and `responseJsonSchema` (Gemini `generateContent`) | `json_schema` / `json_object` / `prompt_only` |
| `to_ollama_response_format` | Value of `format` (Ollama `/api/chat`, `/api/generate`): the schema itself, `"json"`, or `None` | `json_schema` / `json_object` / `prompt_only` |

`json_schema` always starts from the strict schema (8.1). Where an API takes only part of JSON Schema, the builder keeps what that API lists and **moves every other keyword into the description** (`description\n\n{key: value, ...}`), where the model can still read it: Anthropic loses numeric bounds, string length, `pattern`, a `minItems` above 1 and unlisted string formats; Gemini loses `pattern`, string length, `multipleOf` and `const`. OpenAI and Ollama keep the schema as is. `prompt_only` means the field is left out.

| Choice | Why |
|---|---|
| Move unsupported keywords into the description, never refuse | This is how Anthropic's own SDK masks the gap (`transform_schema`, and the "How SDK transformation works" section of its docs: remove unsupported constraints, note them in the description, validate the reply against the original schema), and litellm mirrors it. It is the contract's own rule (3.2: the request may degrade, validation may not): the implementation still validates against the full model (7.1), so a dropped constraint is enforced locally. The builders never judge whether the upstream will accept the rest |
| Keep a list of what each API takes, not of what it lacks | An allowlist fails safe when it goes stale: a keyword an API starts to support stays in the description, one guided constraint fewer. A denylist that misses a newly unsupported keyword would send it and may get a 400 |
| One function per API, one naming pattern | An implementation targets one API and never switches at run time, so a `target` parameter buys it nothing; per-API functions show up in autocompletion, carry exact return types and docs, and adding an API touches no existing function. litellm organises its per-provider conversion the same way |
| `mode` lists only the levels that API has | Asking Anthropic for JSON mode is a type error when the code is written, not a `ValueError` at run time |
| Return what goes into the request, not fields to merge | APIs nest the setting at different paths, and a shallow `dict.update` of a nested field would overwrite whatever else the caller put there. Anthropic is the concrete case: `output_config` also holds `effort` |
| Gemini returns two entries | Gemini spreads the setting over two sibling `generationConfig` fields. Both are flat keys no other entry shares, so `generation_config.update(...)` cannot overwrite anything. The keys are the REST names, which the `google-genai` SDK also accepts (its models use a camel-case alias with `populate_by_name`) |

Request shapes, checked 2026-09-23:

| API | Source |
|---|---|
| Anthropic | platform.claude.com/docs/en/build-with-claude/structured-outputs; anthropic SDK 0.75.0 `types/beta/beta_output_config_param.py`, `beta_json_output_format_param.py`, `lib/_parse/_transform.py` |
| Gemini | google-genai SDK 1.53.0 `types.py` (`GenerateContentConfig.response_mime_type`, `response_json_schema` and its list of supported keywords) |
| Ollama | docs.ollama.com/capabilities/structured-outputs; github.com/ollama/ollama `docs/api.md` |

Other APIs are added once an implementation needs them, with their request shape checked against their own documentation.

---

## 9. Conformance suite

Documentation alone is not enough for a contract. As the party that specifies the protocol, EverAlgo delivers three things: **the contract, executable conformance checks, and test doubles**.

The checks target **the implementation** and verify only the hard requirements in Section 3. The operator-side obligations in 4.3 are checked by EverAlgo's own unit tests; the recommendations in Section 7 are listed separately as **optional checks**, which the implementation runs voluntarily, and failing them is not a contract violation.

### 9.1 Test harness

3.1 and 3.2 need to see **what was actually sent**, and 3.4 and 3.5 need to **inject upstream refusal and truncation signals**. Neither can be verified with just a client instance, so the implementation provides a harness. The contract specifies only its capabilities, not how it is implemented:

```python
class LLMClientTestHarness(Protocol):
    """Provided by the implementation to run the conformance checks."""

    def build(self, **defaults: Any) -> LLMClient:
        """Build a client with the given defaults (used to verify that explicit values win over defaults)."""

    def queue_upstream(self, *responses: UpstreamStub) -> None:
        """Queue what the upstream will return: a valid reply, truncation, refusal, a reply that does not fit the model."""

    def sent_requests(self) -> list[EffectiveRequest]:
        """Requests already sent, normalised into a semantic view."""
```

`EffectiveRequest` is a **semantic view**, not wire bytes, and contains only `temperature`, `max_tokens`, and `messages`. The contract does not prescribe the wire shape (6.6); requiring the suite to parse every vendor's wire format would bind the contract back to one vendor. The format parameter is not in the view: what to send is the implementation's technique, which the contract does not govern (3.2).

**The view MUST be normalised from the outbound request actually captured, not copied from the call arguments.** Copying the arguments would make the check a tautology. The implementation SHOULD place the capture point **as close to the wire as possible** (the HTTP transport layer, an httpx request hook, or similar), not at the entry of the `achat()` method. **The suite cannot enforce this**: if the implementation cuts corners and returns the arguments directly, the suite cannot tell. So this is **the implementation's obligation**; the suite is an aid, not a proof.

### 9.2 Required checks (Section 3 hard requirements)

| Check | Requirement |
|---|---|
| Explicit `temperature` / `max_tokens` reach the upstream and are not overridden by the implementation's defaults | 3.1 |
| When the implementation retries internally, `max_tokens` is not raised on any request | 3.1 |
| When `None` is passed, the implementation's default is used or the parameter is omitted; it is never a literal `None` | 3.1 |
| Non-text `ContentPart`s actually appear in the outbound `messages`; when they cannot be accepted, an error (of any type) is raised; they are never silently dropped | 3.2 |
| When the call is cancelled, `asyncio.CancelledError` propagates unchanged and is not swallowed | 3.3 |
| When the upstream flags a refusal (non-empty `refusal`, `stop_reason: "refusal"`, and the like), an error is raised even if the refusal content is valid JSON; neither the refusal text nor empty content is returned | 3.4 |
| When the upstream reports truncation or abnormal termination (`finish_reason` of `length` / `content_filter`, etc.), the implementation retries or raises even if the body is a closed, valid object; `max_tokens` is unchanged on retry | 3.5 |

### 9.3 Optional checks (Section 7 recommendations)

| Check | Recommendation |
|---|---|
| When the upstream returns a reply that does not fit the model (explanatory text, a fence, a missing required field, truncation that leaves the object unclosed), it is not handed to the operator | 7.1 |
| When the finish reason is missing and the content is valid, the reply is returned normally | 7.1 |
| The number of retries is bounded | 7.3 |
