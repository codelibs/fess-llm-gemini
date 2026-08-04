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
| `rag.llm.gemini.chat.context.max.chars` | `4000` | Maximum characters for context in chat |
| `rag.llm.gemini.chat.evaluation.max.relevant.docs` | `3` | Maximum number of relevant documents for evaluation |
| `rag.llm.gemini.retry.max` | `10` | Maximum HTTP retry attempts on `429` / `5xx` |
| `rag.llm.gemini.retry.base.delay.ms` | `2000` | Base delay (ms) for exponential backoff between retries |

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
- **Gemini 3.x** (e.g. `gemini-3-flash-preview`, `gemini-3.1-flash-lite-preview`, `gemini-3.1-pro`) – sent as `thinkingConfig.thinkingLevel` with bucket mapping `<=0 → MINIMAL` (Flash / Flash-Lite) or `LOW` (Pro, which does not support `MINIMAL`), `<=4096 → MEDIUM`, `>4096 → HIGH`.

Thinking parts (response parts marked `thought: true`) are automatically filtered out before the visible response is delivered. Override per prompt type via `rag.llm.gemini.<type>.thinking.budget` (and `rag.llm.gemini.<type>.max.tokens` for the visible-output cap).

## Features

- **Intent Detection** - Determines user intent (search, summary, FAQ, unclear) and generates Lucene queries
- **Answer Generation** - Generates answers based on search results with citation support
- **Document Summarization** - Summarizes specific documents
- **FAQ Handling** - Provides direct, concise answers to FAQ-type questions
- **Relevance Evaluation** - Identifies the most relevant documents for answer generation
- **Streaming Support** - Real-time response streaming via Server-Sent Events (`?alt=sse`)
- **Availability Checking** - Validates API availability at configurable intervals
- **Extended Thinking** - Model-aware thinking config: integer `thinkingBudget` for Gemini 2.x and bucketed `thinkingLevel` (`MINIMAL`/`LOW`/`MEDIUM`/`HIGH`) for Gemini 3.x

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
