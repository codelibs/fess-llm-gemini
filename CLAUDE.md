# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is `fess-llm-gemini`, a Fess plugin that integrates Google Gemini as an LLM backend for Fess's RAG (Retrieval-Augmented Generation) features. It implements `AbstractLlmClient` from the core Fess project.

## Build Commands

```bash
# Build (requires fess-parent to be installed first)
mvn clean package

# Run tests
mvn test

# Run a single test
mvn test -Dtest=GeminiLlmClientTest

# Run a single test method
mvn test -Dtest=GeminiLlmClientTest#testChat

# Install fess-parent (required before first build)
cd ../fess-parent && mvn install -Dgpg.skip=true
```

Java 21 is required.

## Architecture

### Single-class plugin
The entire plugin is one main class: `GeminiLlmClient` extends `AbstractLlmClient` (from `fess` core). It provides:
- `chat()` - synchronous Gemini API call via `generateContent`
- `streamChat()` - streaming via `streamGenerateContent` with manual JSON brace-depth parsing
- `checkAvailabilityNow()` - health check via `GET /models`
- `buildRequestBody()` - converts Fess's `LlmMessage` list to Gemini's format (system messages go to `systemInstruction`, assistant role maps to `model`)
- `applyDefaultParams()` - sets per-prompt-type defaults (temperature, maxTokens, thinkingBudget) for: intent, evaluation, unclear, noresults, docnotfound, direct, faq, answer, summary

### DI Configuration
`src/main/resources/fess_llm++.xml` is a LastaDi component definition that wires `GeminiLlmClient` as a bean with all prompt templates injected via property setters. The `++` suffix means it auto-loads as a Fess plugin component.

### Configuration Properties
All runtime config is read from `ComponentUtil.getFessConfig()` with prefix `rag.llm.gemini.*`.

### Test Infrastructure
Tests use `UnitFessTestCase` (extends `WebContainerTestCase` from utflute-lastaflute) with `test_app.xml` for DI container setup. HTTP calls are mocked via OkHttp's `MockWebServer`. The test class creates a `TestableGeminiLlmClient` inner subclass that overrides config methods to point at the mock server.

## Coding Conventions

- Apache License 2.0 header on all Java files
- Code formatting enforced by `formatter-maven-plugin` (runs during build)
- License headers enforced by `license-maven-plugin`
- Use `final` on local variables and parameters
- Logger: Log4j2 with `[LLM:GEMINI]` prefix for debug messages
- Error handling: wrap in `LlmException` with error codes from `resolveErrorCode()`
