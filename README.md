Gemini LLM Plugin for Fess
==========================

## Overview

This plugin provides Google Gemini integration for Fess's RAG (Retrieval-Augmented Generation) features. It enables Fess to use Google Gemini models for AI-powered search capabilities including intent detection, answer generation, document summarization, and FAQ handling.

## Download

See [Maven Repository](https://repo1.maven.org/maven2/org/codelibs/fess/fess-llm-gemini/).

## Requirements

- Fess 15.x or later
- Java 21 or later
- Google Gemini API key

## Installation

1. Download the plugin JAR from the Maven Repository
2. Place it in your Fess plugin directory
3. Restart Fess

For detailed instructions, see the [Plugin Administration Guide](https://fess.codelibs.org/14.19/admin/plugin-guide.html).

## Configuration

This plugin reads properties from two independent Fess configuration channels — they are not
interchangeable, and a value placed in the wrong file is silently ignored (no error, no log line):

- **`rag.llm.gemini.*` / `rag.chat.*`** (below) — LastaFlute `fess_config.properties`, loaded
  once at container boot; a change requires a restart.
- **`rag.llm.name`** (below), **`content_chunker.enabled`**, **`content_chunker.embedding.name`**,
  and **`content_chunker.embedding.gemini.*`** (see
  [Content Chunk Embedding](#content-chunk-embedding) below) — `conf/system.properties` (or a
  `-Dfess.system.<key>` JVM argument). Most of these are re-read live, without a restart; the
  exceptions are `content_chunker.embedding.gemini.timeout` and
  `content_chunker.embedding.gemini.availability.check.interval`, which are read once when the
  embedding client initializes and require a restart to pick up a change. Values are also visible
  under System Info > Config Info > App Properties in the admin UI; secret values such as
  `content_chunker.embedding.gemini.api.key` are masked there.

### Enabling the Plugin

| Property | Default | Description |
|----------|---------|-------------|
| `rag.llm.name` | - | Set to `gemini` to activate this plugin as Fess's RAG LLM backend. **A `conf/system.properties` key** (or `-Dfess.system.rag.llm.name=gemini`) — unlike every property in the table below, it is not read from `fess_config.properties`. |

### RAG Chat / LLM

Configure the following properties in `fess_config.properties`:

| Property | Default | Description |
|----------|---------|-------------|
| `rag.chat.enabled` | `false` | Enable RAG chat feature |
| `rag.llm.gemini.api.key` | - | Google Gemini API key (required) |
| `rag.llm.gemini.api.url` | `https://generativelanguage.googleapis.com/v1beta` | Gemini API endpoint URL |
| `rag.llm.gemini.model` | `gemini-3.1-flash-lite-preview` | Model name (e.g., `gemini-3-flash-preview`, `gemini-3.1-pro`, `gemini-2.5-flash`) |
| `rag.llm.gemini.timeout` | `60000` | HTTP request timeout in milliseconds |
| `rag.llm.gemini.availability.check.interval` | `60` | Interval (seconds) for checking API availability |
| `rag.llm.gemini.answer.context.max.chars` | `16000` | Maximum characters of retrieved context passed to the `answer` prompt |
| `rag.llm.gemini.summary.context.max.chars` | `16000` | Maximum characters of the document passed to the `summary` prompt |
| `rag.llm.gemini.faq.context.max.chars` | `10000` | Maximum characters of retrieved context passed to the `faq` prompt |
| `rag.llm.gemini.chat.evaluation.max.relevant.docs` | `3` | Maximum number of relevant documents for evaluation |
| `rag.llm.gemini.chat.evaluation.description.max.chars` | `500` | Maximum characters of each document description during evaluation |
| `rag.llm.gemini.history.max.chars` | `10000` | Maximum characters of conversation history |
| `rag.llm.gemini.history.assistant.max.chars` | `1000` | Maximum characters kept from each assistant turn |
| `rag.llm.gemini.history.assistant.summary.max.chars` | `1000` | Maximum characters kept from each assistant summary |
| `rag.llm.gemini.intent.history.max.messages` | `10` | Maximum history messages passed to the intent prompt |
| `rag.llm.gemini.intent.history.max.chars` | `5000` | Maximum history characters passed to the intent prompt |
| `rag.llm.gemini.<promptType>.temperature` | (built-in per-prompt-type default) | Overrides the built-in `temperature` default for this prompt type. Read by the base `AbstractLlmClient.applyPromptTypeParams`, like `.max.tokens` and `.thinking.budget` below. |
| `rag.llm.gemini.<promptType>.max.tokens` | (built-in per-prompt-type default; see [Extended Thinking](#extended-thinking)) | Overrides the built-in `maxOutputTokens` default for this prompt type. Setting it explicitly also skips the automatic thinking-headroom addition described below — the value you set is sent as-is. |
| `rag.llm.gemini.<promptType>.thinking.budget` | `0` | Overrides the built-in `thinkingBudget` default for this prompt type. See [Extended Thinking](#extended-thinking). |
| `rag.llm.gemini.thinking.level.enabled` | `auto` | Force whether `thinkingConfig.thinkingLevel` is sent instead of `thinkingConfig.thinkingBudget`. See [Extended Thinking](#extended-thinking). |
| `rag.llm.gemini.thinking.minimal.enabled` | `auto` | Force whether `MINIMAL` is a valid `thinkingLevel` for the resolved model. See [Extended Thinking](#extended-thinking). |
| `rag.llm.gemini.thinking.headroom.enabled` | `auto` | Force whether the extra thinking-token headroom is added to `maxOutputTokens`. See [Extended Thinking](#extended-thinking). |
| `rag.llm.gemini.retry.max` | `10` | Maximum HTTP attempts per call (the initial attempt plus retries) when the response status is `429`, `500`, `503` or `504` |
| `rag.llm.gemini.retry.base.delay.ms` | `2000` | Base delay (ms) for exponential backoff between attempts |
| `rag.llm.gemini.max.concurrent.requests` | `5` | Maximum concurrent in-flight requests to the API (read by the base `AbstractLlmClient`) |
| `rag.llm.gemini.concurrency.wait.timeout` | `30000` | Milliseconds a request waits for a concurrency slot before failing (read by the base `AbstractLlmClient`) |

`<promptType>` is one of: `intent`, `evaluation`, `unclear`, `noresults`, `docnotfound`,
`direct`, `faq`, `answer`, `summary`, `queryregeneration`. There is no prompt-type-less form of
these keys — `rag.llm.gemini.temperature` (without a `<promptType>` segment) is read by nothing
and is silently ignored.

`.context.max.chars` is the exception to that list: only the `answer`, `summary` and `faq` prompts
assemble a retrieved-context block, so only the three rows spelled out above are read. A
`rag.llm.gemini.intent.context.max.chars` (or any other prompt type) is read by nothing.

### Content Chunk Embedding

Fess's content-chunking RAG feature is turned on with `content_chunker.enabled=true` and pointed
at this plugin with `content_chunker.embedding.name=gemini` — both `conf/system.properties` keys,
like every property in the table below. Once enabled, the following properties configure
`GeminiEmbeddingClient`, which calls Gemini's `POST /models/{model}:batchEmbedContents` endpoint.

**Unlike the `rag.llm.gemini.*` properties above, every property in this table is configured in
`conf/system.properties`, not `fess_config.properties`** (equivalently, as a
`-Dfess.system.<key>` JVM argument). Fess core routes every `content_chunker.*` key through
`FessConfigImpl.getSystemProperty`, the same live-reloadable channel used across all
content-chunking providers, not the LastaFlute config-file channel loaded once at boot. `timeout`
and `availability.check.interval` are read once, when the client is initialized at startup, so
changing either requires a restart; every other property below is re-read on each call and takes
effect immediately. A value added to `fess_config.properties` for any key in this table is
silently ignored.

| Property | Default | Description |
|----------|---------|-------------|
| `content_chunker.embedding.gemini.api.key` | - | Google Gemini API key (required) |
| `content_chunker.embedding.gemini.api.url` | `https://generativelanguage.googleapis.com/v1beta` | Gemini API endpoint URL |
| `content_chunker.embedding.gemini.model` | `gemini-embedding-001` | Embedding model name |
| `content_chunker.embedding.gemini.document.task_type` | `RETRIEVAL_DOCUMENT` | Gemini `taskType` sent when embedding document/chunk texts. Set to an empty string to omit the field for models/API versions that don't support it. |
| `content_chunker.embedding.gemini.query.task_type` | `RETRIEVAL_QUERY` | Gemini `taskType` sent when embedding query texts. Set to an empty string to omit the field for models/API versions that don't support it. |
| `content_chunker.embedding.gemini.timeout` | `60000` | HTTP request timeout in milliseconds |
| `content_chunker.embedding.gemini.availability.check.interval` | `60` | Interval (seconds) for checking API availability |
| `content_chunker.embedding.gemini.retry.max` | `10` | Maximum HTTP retry attempts on `429` / `500` / `503` / `504` (no `502`, unlike the sibling Ollama client) |
| `content_chunker.embedding.gemini.retry.base.delay.ms` | `2000` | Base delay (ms) for exponential backoff between retries |

Also requires the shared `content_chunker.embedding.dimension` property (embedding vector
dimension, also a `conf/system.properties` value) to be set, independent of this plugin. Like
`GeminiLlmClient`, the API key is sent via the `x-goog-api-key` HTTP request header, not appended
to the URL.

#### Batch size and token quotas

`embedDocuments` / `embedQuery` split their input into sequential sub-batches of at most 100
texts, because Gemini rejects any larger `batchEmbedContents` call. That cap counts *requests*,
not tokens, so a full sub-batch of large chunks can still exceed the project's tokens-per-minute
quota — and that `429` is not one a retry can outgrow, because every attempt re-sends the same
oversized batch. Measured against a free-tier key (~30,000 TPM): 100 chunks of 800 Japanese
characters is ~42,900 tokens and fails on all 10 attempts (~5 minutes of backoff), after which
core's per-document fallback re-sends the same chunks and fails identically, leaving the
document without vectors.

If embedding runs show sustained `429`s, lower `content_chunker.length.chunk_size` (also a
`conf/system.properties` key) so that 100 chunks fit inside the quota, or raise the quota.
Lowering `content_chunker.job.bulk_size` does not help on its own: it bounds how many
*documents* go into one call, not the chunk count of a single large document.

### Authentication

The plugin authenticates by sending the API key as the `x-goog-api-key` HTTP request header (Google's recommended method). Keys are never appended to the URL as `?key=…`, so they do not appear in URL access logs.

### Extended Thinking

The plugin automatically translates a single request-level `thinkingBudget` (integer token allowance) to whatever shape the resolved model expects:

- **Gemini 2.x** (e.g. `gemini-2.5-flash`) – sent as `thinkingConfig.thinkingBudget` (integer).
- **Gemini 3.x** (e.g. `gemini-3-flash-preview`, `gemini-3.1-flash-lite-preview`, `gemini-3.1-pro`) – sent as `thinkingConfig.thinkingLevel` with bucket mapping `<=0 → MINIMAL` (Flash / Flash-Lite) or `LOW` (Pro, which does not support `MINIMAL`), `<=4096 → MEDIUM`, `>4096 → HIGH`. `thinkingLevel` and `thinkingBudget` are alternatives per [Google's API documentation](https://ai.google.dev/docs).

Thinking parts (response parts marked `thought: true`) are automatically filtered out before the visible response is delivered. Override per prompt type via `rag.llm.gemini.<type>.thinking.budget`, `rag.llm.gemini.<type>.temperature` and `rag.llm.gemini.<type>.max.tokens` (the visible-output cap; setting `<type>.max.tokens` explicitly also skips the thinking-headroom addition below).

The built-in per-prompt-type visible-output budgets (the `maxOutputTokens` default, before any
headroom) are:

| Prompt type | Visible-output default |
|-------------|------------------------|
| `evaluation`, `queryregeneration` | 256 |
| `intent`, `unclear`, `noresults`, `docnotfound` | 512 |
| `direct`, `faq` | 2048 |
| `summary` | 4096 |
| `answer` | 8192 |

They are sized for non-English (e.g. Japanese) responses, where 1 character ≈ 1–2 tokens.

Because Gemini 3.x models always spend some tokens on thinking — even at the lowest `thinkingLevel` bucket — the plugin adds an extra 1024 tokens of thinking headroom on top of each prompt type's visible-output budget by default, so responses do not truncate with `finishReason=MAX_TOKENS`. So a Gemini 3.x `answer` request defaults to `maxOutputTokens=9216`, a Gemini 2.x one to `8192`.

Which of the three behaviours above applies to a given model id is normally inferred from the name (`gemini-3-*`, `gemini-3.*`, or exactly `gemini-3`); each can also be forced explicitly:

| Property | Overrides | `auto` resolves to |
|----------|-----------|--------------------|
| `rag.llm.gemini.thinking.level.enabled` | Send `thinkingConfig.thinkingLevel` instead of `thinkingConfig.thinkingBudget` | the model id is Gemini 3 |
| `rag.llm.gemini.thinking.minimal.enabled` | Allow `MINIMAL` as the lowest `thinkingLevel` (otherwise `LOW`) | `thinking.level.enabled` resolved to true **and** the model id contains `flash` |
| `rag.llm.gemini.thinking.headroom.enabled` | Add the 1024-token thinking headroom to the per-prompt-type default `maxOutputTokens` | `thinking.level.enabled` resolved to true |

Both halves of the name rule are matched case-insensitively, so `GEMINI-3-FLASH` classifies exactly as `gemini-3-flash` does. With every key left at `auto` the chained rows above are exactly the name rule too, since `thinking.level.enabled` then resolves to "the model id is Gemini 3".

Each takes `auto` (the default), `true` or `false`, case-insensitively; a blank value is treated as `auto`, and any other unrecognized value also falls back to `auto` (logging a one-time warning per key/value) rather than `false`, so a typo cannot silently disable a capability. Surrounding whitespace never matters — the config channel returns values already trimmed, and the plugin trims again defensively — so `...enabled=true ` reads as `true`. See [Classifying an unrecognized model id](#classifying-an-unrecognized-model-id) for when and how to use them.

`thinking.level.enabled` is the classification key: set it, and the other two follow it on `auto`.
Forcing it to `true` on a model id the name rule does not recognise therefore also turns the
thinking headroom on (so the reply is not truncated) and picks that id's lowest thinking level —
`MINIMAL` when the id contains `flash`, `LOW` otherwise, which is what Gemini 3 Pro accepts. Set
`thinking.headroom.enabled` or `thinking.minimal.enabled` explicitly only when you need to
override that derived answer; each remains independently forceable to `true` or `false`.

Setting `thinking.minimal.enabled=false` is also the only way to reach `LOW` on a Flash model:
`LOW` has no bucket of its own for a positive `thinkingBudget` (`<=4096` maps to `MEDIUM`,
`>4096` to `HIGH`), so on the `auto` path a Flash id always maps a non-positive budget to
`MINIMAL` and never to `LOW`.

### Classifying an unrecognized model id

`isGemini3` only recognises ids that start with `gemini-3-` or `gemini-3.`, or that equal `gemini-3` exactly (case-insensitively). A future `gemini-4*` id, or a Vertex/gateway route id such as `publishers/google/models/gemini-3-flash`, does not match, so `usesThinkingLevel`, `supportsMinimalThinking` and `usesThinkingHeadroom` all fall back to their Gemini 2.x behaviour unless overridden. Classify such an id with `thinking.level.enabled` — for example, a Gemini-3-generation Flash model reached through a gateway route id the name rule cannot parse:

```properties
rag.llm.gemini.api.url=https://my-gemini-gateway.example.com/v1beta
rag.llm.gemini.model=publishers/google/models/gemini-3-flash
rag.llm.gemini.thinking.level.enabled=true
```

That one line is enough: the headroom follows it, and the level chosen at a non-positive budget
still keys off the route id, so this Flash route gets `MINIMAL` while
`publishers/google/models/gemini-3-pro` — same single line — gets `LOW`, the level Pro accepts.
Add `thinking.headroom.enabled` or `thinking.minimal.enabled` only to override one of those.

Notes:

- The keys above live in `fess_config.properties` and are resolved through `getOrDefault`, the same channel as every other `rag.llm.gemini.*` property, so `-Dfess.config.rag.llm.gemini.<key>=<value>` also works as a JVM-level override. Restart Fess after changing any of them.
- Setting `thinking.headroom.enabled=false` for a model that actually needs the headroom means the 1024 tokens are never added, so the per-prompt-type `maxOutputTokens` budget ends up smaller than the model needs. What this codebase verifies is how the client *handles* a truncated reply, not what the live API returns: given a successful HTTP `200` whose body carries `finishReason=MAX_TOKENS`, the client delivers the partial text and logs a WARN rather than throwing. Since truncation is not signalled by an HTTP status, that WARN is the symptom to look for.
- Where to see it: the plugin logs a WARN on `org.codelibs.fess.llm.gemini` whenever a response ends on anything other than `STOP` or `FINISH_REASON_UNSPECIFIED` (both of which are treated as normal completions). Look for `[LLM:GEMINI] Chat finished abnormally. finishReason=MAX_TOKENS, ...` (non-streaming) or `[LLM:GEMINI] Stream finished abnormally. ... finishReason=MAX_TOKENS, ...` (streaming); both carry the `model=` that produced it. The streaming INFO line `[LLM:GEMINI] Stream completed.` additionally reports `candidatesTokens` and `thoughtsTokens`, which show how much of the budget the thinking spend consumed. No DEBUG level is needed for either WARN.
- This plugin's tests run entirely against a mock server, so no claim is made here about how the live Gemini API responds to an unexpected `thinkingLevel` / `thinkingBudget` field beyond the two fields being alternatives per Google's documentation, cited above.

## Features

- **Intent Detection** - Determines user intent (search, summary, FAQ, unclear) and generates Lucene queries
- **Answer Generation** - Generates answers based on search results with citation support
- **Document Summarization** - Summarizes specific documents
- **FAQ Handling** - Provides direct, concise answers to FAQ-type questions
- **Relevance Evaluation** - Identifies the most relevant documents for answer generation
- **Streaming Support** - Real-time response streaming via Server-Sent Events (`?alt=sse`)
- **Availability Checking** - Validates API availability at configurable intervals
- **Extended Thinking** - Model-aware thinking config: integer `thinkingBudget` for Gemini 2.x and bucketed `thinkingLevel` (`MINIMAL`/`LOW`/`MEDIUM`/`HIGH`) for Gemini 3.x, each behaviour independently overridable (see [Extended Thinking](#extended-thinking))

## Gemini API Endpoints Used

The API key is supplied via the `x-goog-api-key` request header on every call (it is not appended to the URL).

- `GET /models` - Lists available models for availability checking
- `POST /models/{model}:generateContent` - Performs chat completion
- `POST /models/{model}:streamGenerateContent?alt=sse` - Performs streaming chat completion (Server-Sent Events)

## Development

### Building from Source

```bash
mvn clean package
```

### Running Tests

```bash
mvn test
```

## License

Apache License 2.0
