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
| `rag.llm.gemini.model` | `gemini-3-flash-preview` | Model name (e.g., `gemini-2.5-flash`, `gemini-2.0-flash`) |
| `rag.llm.gemini.timeout` | `60000` | HTTP request timeout in milliseconds |
| `rag.llm.gemini.availability.check.interval` | `60` | Interval (seconds) for checking API availability |
| `rag.llm.gemini.chat.context.max.chars` | `4000` | Maximum characters for context in chat |
| `rag.llm.gemini.chat.evaluation.max.relevant.docs` | `3` | Maximum number of relevant documents for evaluation |

### Extended Thinking (Gemini 3)

For models that support extended thinking, you can configure the thinking budget via `thinkingConfig.thinkingBudget` in the generation config. Thinking parts (marked with `thought: true`) are automatically filtered out from responses.

## Features

- **Intent Detection** - Determines user intent (search, summary, FAQ, unclear) and generates Lucene queries
- **Answer Generation** - Generates answers based on search results with citation support
- **Document Summarization** - Summarizes specific documents
- **FAQ Handling** - Provides direct, concise answers to FAQ-type questions
- **Relevance Evaluation** - Identifies the most relevant documents for answer generation
- **Streaming Support** - Real-time response streaming via newline-delimited JSON
- **Availability Checking** - Validates API availability at configurable intervals
- **Extended Thinking** - Support for Gemini 3 thinking models with configurable thinking budget

## Gemini API Endpoints Used

- `GET /models?key={apiKey}` - Lists available models for availability checking
- `POST /models/{model}:generateContent?key={apiKey}` - Performs chat completion
- `POST /models/{model}:streamGenerateContent?key={apiKey}` - Performs streaming chat completion

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
