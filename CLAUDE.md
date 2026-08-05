# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is `fess-llm-gemini`, a Fess plugin that integrates Google Gemini as both an LLM backend for
Fess's RAG (Retrieval-Augmented Generation) features and an embedding provider for content-chunk
RAG. It implements `AbstractLlmClient` (`GeminiLlmClient`) and `AbstractEmbeddingClient`
(`GeminiEmbeddingClient`) from the core Fess project.

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

### Two client classes
The plugin has two production classes, each implementing a different core abstraction and
reading a different config prefix through a different config channel (see
[Configuration Properties](#configuration-properties)):

- `GeminiLlmClient` extends `AbstractLlmClient` (from `fess` core), prefix `rag.llm.gemini.*`. It provides:
  - `chat()` - synchronous Gemini API call via `generateContent`
  - `streamChat()` - streaming via `streamGenerateContent` with manual JSON brace-depth parsing
  - `checkAvailabilityNow()` - health check via `GET /models`
  - `buildRequestBody()` - converts Fess's `LlmMessage` list to Gemini's format (system messages go to `systemInstruction`, assistant role maps to `model`)
  - `applyDefaultParams()` - sets per-prompt-type defaults (temperature, maxTokens, thinkingBudget) for: intent, evaluation, unclear, noresults, docnotfound, direct, faq, answer, summary, queryregeneration
- `GeminiEmbeddingClient` extends `AbstractEmbeddingClient` (from `fess` core), prefix
  `content_chunker.embedding.gemini.*`. Calls Gemini's `POST /models/{model}:batchEmbedContents`
  endpoint for `embedDocuments()`/`embedQuery()`, splitting inputs larger than 100 texts into
  sequential sub-batches.

### DI Configuration
`src/main/resources/fess_llm++.xml` is a LastaDi component definition that wires `GeminiLlmClient` as a bean with all prompt templates injected via property setters. The `++` suffix means it auto-loads as a Fess plugin component. `GeminiEmbeddingClient` has no such XML wiring: core resolves it by component name (`geminiEmbeddingClient`) or via `EmbeddingClientManager`'s registration fallback.

### Configuration Properties

The two classes use two different config channels — they are not interchangeable:

- `GeminiLlmClient`: `rag.llm.gemini.*` (and `rag.chat.*`), read via
  `ComponentUtil.getFessConfig().getOrDefault(...)` and the inherited
  `AbstractLlmClient#getConfigInt(...)`, both ultimately backed by `getOrDefault` —
  `fess_config.properties` (LastaFlute `ObjectiveConfig`, loaded once at container boot) plus
  the `-Dfess.config.*` JVM override.
- `GeminiEmbeddingClient`: `content_chunker.embedding.gemini.*`, read via the inherited
  `AbstractEmbeddingClient#getConfigString(...)` and `#getConfigInt(...)`, both ultimately backed
  by `ComponentUtil.getFessConfig().getSystemProperty(...)` — `conf/system.properties` (or
  `-Dfess.system.<key>`), visible under System Info > Config Info > App Properties (secret values
  such as `api.key` are masked there). This matches how Fess core routes every
  `content_chunker.*` key. Most properties are re-read live on every call; `timeout` and
  `availability.check.interval` are the exceptions — both are consumed exactly once, in
  `AbstractEmbeddingClient#init()` / `#startAvailabilityCheck()` (`GeminiEmbeddingClient` does not
  override `init()`), so changing either requires a restart.

A value for a `content_chunker.embedding.gemini.*` key placed in `fess_config.properties` (or a
`rag.llm.gemini.*` key placed in `conf/system.properties`) is silently ignored.

Two keys read by `GeminiLlmClient` are deliberately not of the `rag.llm.gemini.*` shape:

- `getLlmType()` reads `rag.llm.name` via `getSystemProperty` — the `conf/system.properties`
  channel `GeminiEmbeddingClient` uses, not the LLM client's own. This is the key that selects
  the plugin, so it cannot live behind the plugin's own prefix.
- `isRagChatEnabled()` reads `rag.chat.enabled`, a core Fess key carrying no
  `rag.llm.gemini.` prefix.

Base-class keys (`max.concurrent.requests`, `concurrency.wait.timeout`, the
`<promptType>.temperature` / `.max.tokens` / `.thinking.budget` triple) are composed from
`getConfigPrefix()` in `AbstractLlmClient`, so they do carry this plugin's prefix even though
`GeminiLlmClient` never names them.

### Test Infrastructure
Tests use `UnitFessTestCase` (extends `WebContainerTestCase` from utflute-lastaflute) with `test_app.xml` for DI container setup. HTTP calls are mocked via OkHttp's `MockWebServer`. `GeminiLlmClientTest` creates a `TestableGeminiLlmClient` inner subclass that overrides config methods to point at the mock server; `GeminiEmbeddingClientTest` does the same with `TestableGeminiEmbeddingClient`.

A second seam covers the real (non-overridden) config-read methods on `GeminiEmbeddingClient`: `GeminiEmbeddingClientConfigChannelTest`, plus several `test_get*_notConfigured`/`test_getDimension_*`/`test_getRetryBaseDelayMs_nonNumericValue_returnsDefault` methods in `GeminiEmbeddingClientTest`, construct a plain `new GeminiEmbeddingClient()` (no subclass) and inject values directly into the `systemProperties` component via `ComponentUtil.getSystemProperties().setProperty(...)`/`.remove(...)`, restoring in a `finally` block. That component is a JVM-lifetime singleton shared across test classes (registered as `org.codelibs.fess.unit.TestSystemProperties` in `test_app.xml`), so both test classes override `isUseOneTimeContainer()` to get a fresh container per test method instead of relying on `finally`-only cleanup.

### Logging keys

`streamChat` emits a single `[LLM:GEMINI] Stream completed.` INFO line per call carrying:
`chunkCount`, `objectCount`, `firstChunkMs`, `elapsedTime`, `finishReason`,
`promptTokens`, `candidatesTokens`, `thoughtsTokens`, `totalTokens`.

When `finishReason` is anything other than `STOP` / `FINISH_REASON_UNSPECIFIED`,
both `chat()` and `streamChat()` emit an extra WARN line so truncation
(`MAX_TOKENS`) and content blocking (`SAFETY`, `RECITATION`,
`PROHIBITED_CONTENT`, `BLOCKLIST`, `SPII`, `IMAGE_SAFETY`,
`MALFORMED_FUNCTION_CALL`, `OTHER`) can be alerted on without enabling DEBUG.

Enable `org.codelibs.fess.llm.gemini` at DEBUG level to additionally log:
- the JSON request body sent to Gemini (`requestBody=`),
- HTTP status + `Content-Type` of the streaming response,
- each parsed JSON object from the stream (`streamObject#N json=`).

The completion line additionally records `responseId` for request
correlation and `cachedContentTokens` when context caching is in use.
WARN lines are also emitted for `promptFeedback.blockReason` (input
blocked) and for candidate `safetyRatings` whenever the response stops on
an abnormal `finishReason` such as `SAFETY` / `LANGUAGE` / `RECITATION` /
`PROHIBITED_CONTENT`.

### Auth & retries

Gemini API key is sent as the `x-goog-api-key` HTTP header (recommended by
Google), not via `?key=` query parameter — keys never appear in URL logs.

Retries: HTTP `429`, `500`, `503`, `504` are retried, up to
`rag.llm.gemini.retry.max` *attempts* in total per call — the initial attempt
plus retries, not that many retries on top of it (default `10`, see
`getRetryMaxAttempts` / `executeWithRetry`) — with exponential backoff
starting at `rag.llm.gemini.retry.base.delay.ms` (default `2000`) and ±20%
jitter. Streaming retries only the initial connect — once the response body
starts flowing, partial-stream errors propagate immediately.

### Model-aware thinking

`thinkingBudget` (integer, Gemini 2.x) and `thinkingLevel` (`MINIMAL`/`LOW`/
`MEDIUM`/`HIGH`, Gemini 3.x) are mutually exclusive on the wire — alternatives per
[Google's API documentation](https://ai.google.dev/docs). `usesThinkingLevel(model)`
decides which field is sent, defaulting (`auto`) to `isGemini3(model)` — a
config-free name-rule predicate (prefix match on `gemini-3-*` / `gemini-3.*`, or
exactly `gemini-3`, case-insensitive, blank-safe) that is only the *inference* the
configurable predicates fall back to, not itself a decision point. The client
translates the request-level `thinkingBudget` to whichever field
`usesThinkingLevel` selects:

- Gemini 2.x: `thinkingBudget` is sent as-is.
- Gemini 3.x: `thinkingBudget` is mapped to `thinkingLevel`. The `<=0`
  bucket is model-aware: it maps to `MINIMAL` on Gemini 3 Flash and Gemini
  3.1 Flash-Lite (which support `MINIMAL`), and to `LOW` on Gemini 3 Pro /
  Gemini 3.1 Pro (which do not). `<=4096` maps to `MEDIUM`, `>4096` to
  `HIGH`.

On `auto`, `isGemini3` matches only `gemini-3-*`, `gemini-3.*` and exactly
`gemini-3`, so a future `gemini-4*` id, or a Vertex/gateway route id such as
`publishers/google/models/gemini-3-flash`, falls through to the Gemini 2.x branch
of every decision below unless the corresponding property is forced. This is
**not** a compatible-endpoint story: Gemini's URL shape, `x-goog-api-key` header
and response parser are all vendor-specific, and `rag.llm.gemini.api.url` already
covers a Gemini-protocol gateway. What the overrides address is model-name churn.
The shape chosen here is one key per wire effect, *chained*:
`thinking.level.enabled` is the classification key, and the other two derive from
its resolved value when left on `auto`. One setting therefore classifies an
unrecognised id correctly end to end, while each effect stays independently
forceable for the cases where the derived answer is wrong. A single root
"is this a new-generation model" key could not express combinations such as "send
`thinkingLevel` but add no headroom"; naming the effects rather than the
generation also avoids a key name that ages against Gemini generations the way
`gemini3.enabled` would.

| Property | Overrides | `auto` resolves to |
|----------|-----------|--------------------|
| `rag.llm.gemini.thinking.level.enabled` | `usesThinkingLevel(model)` - send `thinkingLevel` instead of `thinkingBudget` | `isGemini3(model)` |
| `rag.llm.gemini.thinking.minimal.enabled` | `supportsMinimalThinking(model)` - allow `MINIMAL` as the lowest level | `usesThinkingLevel(model)` and the id contains `flash` (case-insensitive; a blank id is guarded, since a forced `usesThinkingLevel` no longer shields the `toLowerCase`) |
| `rag.llm.gemini.thinking.headroom.enabled` | `usesThinkingHeadroom(model)` - add `GEMINI3_THINKING_HEADROOM` to the per-prompt-type default `maxOutputTokens` (see [Default generation parameters](#default-generation-parameters)) | `usesThinkingLevel(model)` |

With all three keys unset the chained rows collapse back onto the name rule,
because `usesThinkingLevel(model)` is then exactly `isGemini3(model)` — the
chaining changes nothing on the default path and only takes effect once
`thinking.level.enabled` is forced.

Values are `auto` (default) / `true` / `false`, case-insensitive; a blank value is `auto`, and any
other unrecognized value degrades to `auto` with a WARN emitted once per key/value — never to
`false`, so a typo cannot silently disable a capability. Surrounding whitespace is a non-issue:
the config channel returns values already trimmed (`FessConfigImpl` ends in
`filterPropertyAsDefault` -> LastaFlute `filterPropertyTrimming` -> `String.trim()`, and the
`-Dfess.config.*` lookup sits upstream of it), and `getCapabilityOverride` trims again defensively
so the parsing does not depend on that. They are read with `getConfigString(suffix, default)`,
which goes through `getOrDefault` (`fess_config.properties` / `-Dfess.config.*`). Do **not** switch
this to `AbstractEmbeddingClient#getConfigString`, which reads `conf/system.properties` — a
different channel from every other `rag.llm.gemini.*` key. `GeminiLlmClientCapabilityConfigTest`
exists to catch exactly that regression.

`supportsMinimalThinking` names a different wire effect from `usesThinkingLevel` —
*which* level string is sent at a non-positive budget, not *whether* a level string
is sent — so it keeps its own key even though it derives from `usesThinkingLevel`
on `auto`: an operator who forces `thinking.level.enabled=true` for a Gemini 3 Pro
route id gets `LOW` from the derivation (the id has no `flash` in it), and can
force `MINIMAL` on or off from there. Note also that `LOW` has no bucket of its
own above zero (`<=4096` is `MEDIUM`, `>4096` is `HIGH`), so on a Flash model the
only route to `LOW` is a non-positive budget with `thinking.minimal.enabled=false`
— a combination the `auto` path cannot produce.

Because the derivation is chained, `thinking.level.enabled=true` on its own is the
correct way to classify an unrecognised Gemini 3 id: it also turns on the headroom
that prevents `MAX_TOKENS` truncation. Forcing all three keys blindly is the
mistake to avoid — `thinking.minimal.enabled=true` on a Pro route id sends
`MINIMAL`, a level Pro rejects.

### Default generation parameters

The default model (`rag.llm.gemini.model`) is `gemini-3.1-flash-lite-preview`,
chosen as the most cost-effective Gemini option. Other Gemini 3.x models
(e.g. `gemini-3-flash-preview`, `gemini-3-pro`) and Gemini 2.x models
(e.g. `gemini-2.5-flash`) are supported by setting the property accordingly.

All prompt types default to `thinkingBudget=0`. The visible-output budgets are:
`intent=512, evaluation=256, queryregeneration=256, docnotfound=512,
unclear=512, noresults=512, direct=2048, faq=2048, summary=4096, answer=8192`.

The per-step values are sized for non-English (e.g. Japanese) responses, where
1 char ≈ 1–2 tokens. `intent` and `docnotfound` are 512 instead of 256 because
the JSON reasoning field and the polite multi-bullet "document not found"
message can both exceed 256 visible tokens in Japanese.

Because Gemini 3.x always emits some thinking tokens (even at the lowest
`thinkingLevel` bucket — `MINIMAL` on Flash/Flash-Lite, `LOW` on Pro — which
is the bucket `thinkingBudget=0` maps to), the default `maxOutputTokens` is
**headroom-aware**: when `usesThinkingHeadroom(model)` is true (on `auto` that
means whenever `usesThinkingLevel(model)` is true — by default, a Gemini 3.x
model id; separately forceable via `rag.llm.gemini.thinking.headroom.enabled`,
see [Model-aware thinking](#model-aware-thinking)),
an extra `GEMINI3_THINKING_HEADROOM` (1024 tokens) is added on top of each prompt
type's visible budget so responses do not truncate with
`finishReason=MAX_TOKENS`. When headroom is not needed, defaults are unchanged
because `thinkingBudget=0` actually disables thinking on the 2.x wire format.

Override per prompt type via `rag.llm.gemini.<type>.temperature`,
`rag.llm.gemini.<type>.thinking.budget` and `rag.llm.gemini.<type>.max.tokens` in
`fess_config.properties` (or `-Dfess.config....`). All three are read by the
**base** `AbstractLlmClient.applyPromptTypeParams` (not by `GeminiLlmClient`
itself) before `applyDefaultParams` runs, so an explicit value always wins over
the built-in default — in particular, an explicit `<type>.max.tokens` also
suppresses the `GEMINI3_THINKING_HEADROOM` addition above, since the headroom is
only added when this class computes the default itself.

## Coding Conventions

- Apache License 2.0 header on all Java files
- Code formatting enforced by `formatter-maven-plugin` (runs during build)
- License headers enforced by `license-maven-plugin`
- Use `final` on local variables and parameters
- Logger: Log4j2 with `[LLM:GEMINI]` prefix for debug messages
- Error handling: wrap in `LlmException` with error codes from `resolveErrorCode()`
