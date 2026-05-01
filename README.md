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

Configure the following properties in `fess_config.properties`:

| Property | Default | Description |
|----------|---------|-------------|
| `rag.llm.name` | - | Set to `gemini` to use this plugin |
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
