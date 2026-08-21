/*
 * Copyright 2012-2025 CodeLibs Project and the Others.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
 * either express or implied. See the License for the specific language
 * governing permissions and limitations under the License.
 */
package org.codelibs.fess.embedding.gemini;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.regex.Pattern;

import org.apache.hc.client5.http.classic.methods.HttpGet;
import org.apache.hc.client5.http.classic.methods.HttpPost;
import org.apache.hc.core5.http.ContentType;
import org.apache.hc.core5.http.ParseException;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.apache.hc.core5.http.io.entity.StringEntity;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.codelibs.core.lang.StringUtil;
import org.codelibs.fess.embedding.AbstractEmbeddingClient;
import org.codelibs.fess.embedding.EmbeddingException;
import org.codelibs.fess.gemini.GeminiApiUrl;
import org.codelibs.fess.util.CredentialUrlUtil;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

/**
 * Embedding client implementation for Google Gemini.
 * Calls Gemini's {@code POST /models/{model}:batchEmbedContents} endpoint.
 *
 * <p>Because that endpoint caps a single call at {@link #MAX_BATCH_SIZE} requests, inputs larger
 * than the cap are split into sequential sub-batches and their vectors concatenated in input
 * order (see {@link #callEmbedApi(String, List, String)}); this is transparent to callers.
 *
 * <p>Retry semantics, timeout handling, and the {@code x-goog-api-key} auth
 * header mirror {@code org.codelibs.fess.llm.gemini.GeminiLlmClient}'s
 * conventions rather than the sibling {@code OllamaEmbeddingClient}'s: only
 * {@code 429/500/503/504} are retried (no {@code 502}) and connect-level
 * {@link IOException}/{@link ParseException} are never retried. A single backoff
 * sleep is capped at {@link #MAX_BACKOFF_MS} (like {@code OllamaEmbeddingClient}, and
 * unlike {@code GeminiLlmClient}) so a persistently-throttled sub-batch cannot stall
 * the caller for minutes on one sleep.
 *
 * <p>{@link #embedDocuments(List)} and {@link #embedQuery(List)} share the
 * same HTTP call machinery but pass Gemini's {@code taskType} request
 * parameter, analogous to how {@code OllamaEmbeddingClient} applies a
 * {@code search_document: }/{@code search_query: } text prefix for the same
 * document-vs-query distinction.
 *
 * <p>The two also differ in the text they send, which is separate from {@code taskType}:
 * a query reaching {@link #embedQuery(List)} on the RAG path is a Fess query string built
 * by the LLM's intent step, so its query syntax is stripped first (see
 * {@link #toPlainQuery(String)}). Document text is embedded exactly as given.
 *
 * @see <a href="https://ai.google.dev/gemini-api/docs/embeddings">Gemini Embeddings API</a>
 */
public class GeminiEmbeddingClient extends AbstractEmbeddingClient {

    private static final Logger logger = LogManager.getLogger(GeminiEmbeddingClient.class);

    /** Shared ObjectMapper instance for JSON processing. */
    protected static final ObjectMapper objectMapper = new ObjectMapper();

    /** The name identifier for the Gemini embedding client. */
    protected static final String NAME = "gemini";

    /**
     * Configuration property key suffix holding the API endpoint. Combined with
     * {@link #getConfigPrefix()} via {@link #apiUrlConfigKey()} to build the full key named in
     * URL-rejection messages, so that name can never drift from the key {@link #getApiUrl()}
     * actually reads.
     */
    private static final String API_URL_CONFIG_SUFFIX = "api.url";

    /**
     * Configuration property key suffix holding the retry base delay, shared between
     * {@link #getRetryBaseDelayMs()}'s {@code getConfigString} read and its invalid-value WARN
     * message so the two can never drift apart.
     */
    private static final String CONFIG_RETRY_BASE_DELAY_MS = "retry.base.delay.ms";

    private static final String CONFIG_DOCUMENT_TASK_TYPE = "document.task_type";
    private static final String CONFIG_QUERY_TASK_TYPE = "query.task_type";

    /** Default {@code taskType} sent when embedding document/chunk texts. */
    protected static final String DEFAULT_DOCUMENT_TASK_TYPE = "RETRIEVAL_DOCUMENT";

    /** Default {@code taskType} sent when embedding query texts. */
    protected static final String DEFAULT_QUERY_TASK_TYPE = "RETRIEVAL_QUERY";

    /**
     * Model-name prefix of the legacy Gemini/PaLM embedding line ({@code embedding-001},
     * {@code embedding-gecko-001}) that predates Matryoshka (MRL) support and therefore rejects
     * the {@code outputDimensionality} request field with a non-retryable {@code 400}. Every
     * current model - {@code gemini-embedding-001}/{@code -2}, {@code text-embedding-004}/{@code
     * -005}, {@code text-multilingual-embedding-*} - accepts it and does not match this prefix.
     */
    private static final String LEGACY_FIXED_DIMENSION_MODEL_PREFIX = "embedding-";

    /**
     * Native (untruncated) output dimensionality of {@code gemini-embedding-001}. Google's
     * embeddings documentation guarantees unit-length vectors only at this size; any smaller
     * configured dimension is produced by MRL truncation and must be normalized manually
     * (see {@link #isNormalizationRequired(String, int)}).
     */
    static final int NATIVE_DIMENSION = 3072;

    /** Hard cap on a single backoff sleep, regardless of the computed exponential delay. */
    private static final long MAX_BACKOFF_MS = 60_000L;

    /**
     * Maximum number of texts allowed in a single {@code batchEmbedContents} call. Gemini's
     * endpoint rejects any larger batch with HTTP {@code 400} and the verbatim message
     * {@code "BatchEmbedContentsRequest.requests: at most 100 requests can be in one batch"}.
     * {@link #callEmbedApi(String, List, String)} therefore splits the input into sequential
     * sub-batches of at most this size, regardless of whether those texts came from one document
     * or many (a {@code 400} here is not retryable, and core's per-document fallback would fail
     * identically for a single document whose chunk count exceeds this cap).
     */
    static final int MAX_BATCH_SIZE = 100;

    /**
     * A {@code +} or {@code -} that begins a term. Mid-token both are ordinary characters
     * ({@code gemini-embedding-001}, {@code C++}), so the match is anchored to the start
     * of the string or to whitespace and the whitespace itself is preserved.
     */
    private static final Pattern QUERY_TERM_PREFIX = Pattern.compile("(^|\\s)[+\\-](?=\\S)");

    /**
     * A field restriction such as {@code title:}. The field name is a schema name rather than
     * something the user asked about, so it is removed with its colon instead of being left
     * behind as a term. Deliberately ASCII-only ({@code \w}), which is what Fess field names are.
     */
    private static final Pattern QUERY_FIELD_PREFIX = Pattern.compile("\\b\\w+:");

    /**
     * A boost ({@code ^2}) or fuzzy/proximity ({@code ~1}) marker together with its number.
     * Removed as a unit: dropping only the {@code ^} of {@code "Fess"^2} would glue the boost
     * factor onto the term and embed {@code Fess2}.
     */
    private static final Pattern QUERY_BOOST_OR_FUZZY = Pattern.compile("[\\^~]\\d*(?:\\.\\d+)?");

    /** Grouping, phrase, range and wildcard markup, plus the two-character boolean operators. */
    private static final Pattern QUERY_SYNTAX_CHARS = Pattern.compile("[\"()\\[\\]{}*?\\\\]|&&|\\|\\|");

    /** Boolean and range keywords, which Lucene reads as operators rather than as terms. */
    private static final Pattern QUERY_KEYWORDS = Pattern.compile("\\b(?:AND|OR|NOT|TO)\\b");

    /** Collapses the gaps left where markup was removed. */
    private static final Pattern WHITESPACE_RUN = Pattern.compile("\\s+");

    /**
     * Whether the userinfo-bearing {@code api.url} has already been reported. The availability
     * probe runs on a timer, so an unguarded ERROR would repeat the same line forever; this latches
     * it to one report per broken configuration and re-arms once the URL is fixed.
     */
    private final AtomicBoolean userInfoApiUrlReported = new AtomicBoolean();

    /**
     * Default constructor.
     */
    public GeminiEmbeddingClient() {
        // Default constructor
    }

    @Override
    public String getName() {
        return NAME;
    }

    @Override
    protected boolean checkAvailabilityNow() {
        final String apiUrl = getApiUrl();
        // Checked before the API key, because an operator who put credentials in the URL has
        // usually left the key unset; reporting only "apiKey is blank" would hide the real problem.
        if (reportUserInfoApiUrl(apiUrl)) {
            return false;
        }
        final String apiKey = getApiKey();
        if (StringUtil.isBlank(apiKey)) {
            if (logger.isDebugEnabled()) {
                logger.debug("[Embedding:GEMINI] Gemini is not available. apiKey is blank");
            }
            return false;
        }
        if (StringUtil.isBlank(apiUrl)) {
            if (logger.isDebugEnabled()) {
                logger.debug("[Embedding:GEMINI] Gemini is not available. apiUrl is blank");
            }
            return false;
        }
        final String maskedUrl = CredentialUrlUtil.maskCredentialInUrl(apiUrl);
        try {
            final String url = GeminiApiUrl.appendPath(apiUrl, "/models");
            final HttpGet request = GeminiApiUrl.createGet(url, apiUrlConfigKey());
            request.addHeader("x-goog-api-key", apiKey);
            try (var response = getHttpClient().execute(request)) {
                final int statusCode = response.getCode();
                final boolean available = statusCode >= 200 && statusCode < 300;
                if (logger.isDebugEnabled()) {
                    logger.debug("[Embedding:GEMINI] Gemini availability check. url={}, statusCode={}, available={}", maskedUrl, statusCode,
                            available);
                }
                return available;
            }
        } catch (final Exception e) {
            if (logger.isDebugEnabled()) {
                // The URI-rejection message no longer carries the URL at all; masking here as well
                // covers any other exception on this branch whose message may quote it.
                logger.debug("[Embedding:GEMINI] Gemini is not available. url={}, error={}", maskedUrl,
                        CredentialUrlUtil.maskCredentialInUrl(e.getMessage()));
            }
            return false;
        }
    }

    @Override
    public List<float[]> embedDocuments(final List<String> texts) {
        return callEmbedApi("embedDocuments", texts, getDocumentTaskType());
    }

    /**
     * Generates embedding vectors for the given query texts, with Fess/Lucene query syntax
     * removed first (see {@link #toPlainQuery(String)}).
     *
     * <p>The request differs from {@link #embedDocuments(List)} in its {@code taskType}, and the
     * text differs too: a query arriving here on the RAG path is a Fess query string assembled by
     * the intent step, and its operators are markup rather than words.
     *
     * @param texts the query texts to embed, in order
     * @return the list of vectors, one per input text, in the same order
     * @throws EmbeddingException if the provider call fails or returns an unusable response
     */
    @Override
    public List<float[]> embedQuery(final List<String> texts) {
        if (texts == null || texts.isEmpty()) {
            return callEmbedApi("embedQuery", texts, getQueryTaskType());
        }
        final List<String> plainTexts = new ArrayList<>(texts.size());
        for (final String text : texts) {
            final String plain = toPlainQuery(text);
            if (logger.isDebugEnabled() && plain != null && !plain.equals(text)) {
                logger.debug("[Embedding:GEMINI] Removed query syntax before embedding. from={}, to={}", text, plain);
            }
            plainTexts.add(plain);
        }
        return callEmbedApi("embedQuery", plainTexts, getQueryTaskType());
    }

    /**
     * Removes Fess/Lucene query syntax so what gets embedded is the terms the user asked about.
     *
     * <p>On the RAG path fess core embeds the query the LLM's intent step produced, and this
     * plugin's own {@code intentDetectionPrompt} instructs that step to emit Fess syntax
     * ({@code +required}, {@code (a OR b)}, {@code title:"x"^2}, quoted phrases). Those
     * operators are not words: embedded verbatim they are noise in the vector, and the chunks
     * chosen for the answer prompt are ranked against that vector.
     *
     * <p><b>Scope.</b> In fess 15.8.0 exactly two call sites reach {@code embedQuery}.
     * {@code SemanticChunkSearcher#search} calls it only after its own {@code isPlainQuery()}
     * returned true, and every construct removed here is one that
     * {@code SemanticChunkSearcher.QUERY_SYNTAX_PATTERN} already rejects - so for that call site
     * this method is the identity and the semantic branch embeds exactly what it embedded
     * before. The behaviour therefore changes only on the other call site,
     * {@code DefaultChatContentFetcher#resolveQueryVector}, which is the one that needs it.
     *
     * <p>A string that survives unchanged is returned as-is, whitespace included, so the
     * identity above is exact rather than approximate. A string left empty by the removals -
     * a query made only of operators - falls back to the original, because a blank input is
     * rejected by the API and degrading to the previous behaviour beats failing the chat.
     *
     * @param text the query text, may be null
     * @return the text with query syntax removed, or the original text if nothing was removed
     *         or nothing would remain
     */
    protected String toPlainQuery(final String text) {
        if (StringUtil.isBlank(text)) {
            return text;
        }
        String work = QUERY_TERM_PREFIX.matcher(text).replaceAll("$1");
        work = QUERY_FIELD_PREFIX.matcher(work).replaceAll(StringUtil.EMPTY);
        work = QUERY_BOOST_OR_FUZZY.matcher(work).replaceAll(StringUtil.EMPTY);
        // Replaced with a space, not with nothing: "(a)(b)" must not become the single term "ab".
        work = QUERY_SYNTAX_CHARS.matcher(work).replaceAll(" ");
        work = QUERY_KEYWORDS.matcher(work).replaceAll(StringUtil.EMPTY);
        if (work.equals(text)) {
            return text;
        }
        final String plain = WHITESPACE_RUN.matcher(work).replaceAll(" ").trim();
        return plain.isEmpty() ? text : plain;
    }

    /**
     * Embeds the given texts, splitting them into sequential sub-batches of at most
     * {@link #MAX_BATCH_SIZE} texts so no single {@code batchEmbedContents} call exceeds
     * Gemini's documented per-call cap (see {@link #MAX_BATCH_SIZE}). The split is applied
     * uniformly regardless of how many documents the texts came from, so a single call with
     * more than {@link #MAX_BATCH_SIZE} texts (e.g. one long document's chunk list) is embedded
     * correctly rather than failing. Each sub-batch is embedded by an independent
     * {@link #callEmbedApiBatch(String, List, String)} call (each with its own retry budget), and
     * the resulting vectors are concatenated in input order. Shared by
     * {@link #embedDocuments(List)} and {@link #embedQuery(List)}.
     *
     * <p>Reassembly is all-or-nothing: if any sub-batch fails after retries are exhausted, the
     * failure propagates as an {@link EmbeddingException} and no partial result is returned, since
     * a partial vector list would corrupt the chunk-to-vector mapping upstream.
     *
     * @param operation log label, e.g. {@code "embedDocuments"} or {@code "embedQuery"}
     * @param texts the input texts to embed, in order
     * @param taskType the Gemini {@code taskType} value to send, or blank to omit the field
     * @return the parsed vectors, one per input text, in the same order
     * @throws EmbeddingException if any sub-batch call fails or returns an unusable response
     */
    private List<float[]> callEmbedApi(final String operation, final List<String> texts, final String taskType) {
        if (texts == null || texts.isEmpty()) {
            return Collections.emptyList();
        }
        final int total = texts.size();
        if (total <= MAX_BATCH_SIZE) {
            return callEmbedApiBatch(operation, texts, taskType);
        }
        final List<float[]> vectors = new ArrayList<>(total);
        for (int start = 0; start < total; start += MAX_BATCH_SIZE) {
            final int end = Math.min(start + MAX_BATCH_SIZE, total);
            vectors.addAll(callEmbedApiBatch(operation, texts.subList(start, end), taskType));
        }
        return vectors;
    }

    /**
     * Calls the {@code batchEmbedContents} endpoint once with the given texts (assumed to be at
     * most {@link #MAX_BATCH_SIZE} in size; callers split larger inputs via
     * {@link #callEmbedApi(String, List, String)}), tagging every nested request with
     * {@code taskType} (see {@link #buildRequestBody(List, String, int, String)}).
     *
     * @param operation log label, e.g. {@code "embedDocuments"} or {@code "embedQuery"}
     * @param texts the input texts to embed in this single call, in order
     * @param taskType the Gemini {@code taskType} value to send, or blank to omit the field
     * @return the parsed vectors, one per input text, in the same order
     * @throws EmbeddingException if the provider call fails or returns an unusable response
     */
    private List<float[]> callEmbedApiBatch(final String operation, final List<String> texts, final String taskType) {
        requireNoUserInfoApiUrl();
        final String model = getModel();
        final String url = GeminiApiUrl.appendPath(getApiUrl(), "/models/" + model + ":batchEmbedContents");
        final String maskedUrl = CredentialUrlUtil.maskCredentialInUrl(url);
        final int dimension = getDimension();
        final Map<String, Object> requestBody = buildRequestBody(texts, model, dimension, taskType);
        final long startTime = System.currentTimeMillis();

        try {
            final String json = objectMapper.writeValueAsString(requestBody);
            return executeWithRetry(operation, () -> {
                final HttpPost httpRequest = GeminiApiUrl.createPost(url, apiUrlConfigKey());
                httpRequest.addHeader("x-goog-api-key", getApiKey());
                httpRequest.setEntity(new StringEntity(json, ContentType.APPLICATION_JSON));
                try (var response = getHttpClient().execute(httpRequest)) {
                    final int statusCode = response.getCode();
                    if (statusCode < 200 || statusCode >= 300) {
                        if (isRetryableStatus(statusCode)) {
                            throw new RetryableHttpException(statusCode, response.getReasonPhrase());
                        }
                        logger.warn("[Embedding:GEMINI] API error. url={}, statusCode={}, message={}", maskedUrl, statusCode,
                                response.getReasonPhrase());
                        throw new EmbeddingException("Gemini API error: " + statusCode + " " + response.getReasonPhrase());
                    }
                    final String responseBody = response.getEntity() != null ? EntityUtils.toString(response.getEntity()) : "";
                    final List<float[]> vectors = parseEmbedResponse(responseBody, texts.size(), dimension);
                    logger.info("[Embedding:GEMINI] {} response received. count={}, elapsedTime={}ms", operation, vectors.size(),
                            System.currentTimeMillis() - startTime);
                    return vectors;
                }
            });
        } catch (final EmbeddingException e) {
            throw e;
        } catch (final Exception e) {
            logger.warn("[Embedding:GEMINI] Failed to call Gemini embed API. url={}, error={}", maskedUrl, e.getMessage(), e);
            throw new EmbeddingException("Failed to call Gemini embed API", e);
        }
    }

    /**
     * Builds the {@code batchEmbedContents} request body: one nested
     * {@code EmbedContentRequest} entry per input text, in the same order.
     * Each entry's {@code model} field carries the {@code "models/"} prefix
     * (Gemini's proto convention), distinct from the outer URL's
     * {@code {model}} path segment which does not.
     *
     * @param texts the input texts, in order
     * @param model the configured model name (without the {@code "models/"} prefix)
     * @param dimension the configured output dimensionality, sent on every nested request whose
     *            model supports it (see {@link #supportsOutputDimensionality(String)})
     * @param taskType the Gemini {@code taskType} value to set on every nested request, or blank to omit the field
     * @return the request body as a map, ready for JSON serialization
     */
    protected Map<String, Object> buildRequestBody(final List<String> texts, final String model, final int dimension,
            final String taskType) {
        // Invariant across the loop: hoisted so a 100-text batch does not recompute them per entry.
        final String qualifiedModel = "models/" + model;
        final boolean sendDimension = supportsOutputDimensionality(model);
        final boolean sendTaskType = StringUtil.isNotBlank(taskType);
        final List<Map<String, Object>> requests = new ArrayList<>(texts.size());
        for (final String text : texts) {
            final Map<String, Object> content = new HashMap<>(4);
            content.put("parts", List.of(Map.of("text", text)));

            final Map<String, Object> embedRequest = new HashMap<>(8);
            embedRequest.put("model", qualifiedModel);
            embedRequest.put("content", content);
            if (sendDimension) {
                embedRequest.put("outputDimensionality", dimension);
            }
            if (sendTaskType) {
                embedRequest.put("taskType", taskType);
            }
            requests.add(embedRequest);
        }
        final Map<String, Object> body = new HashMap<>(4);
        body.put("requests", requests);
        return body;
    }

    /**
     * Determines whether {@code model} accepts the {@code outputDimensionality} request field
     * (Matryoshka / MRL truncation). All current Gemini embedding models -
     * {@code gemini-embedding-001}/{@code -2}, {@code text-embedding-004}/{@code -005}, and
     * {@code text-multilingual-embedding-*} - support it; only the legacy
     * {@code embedding-001}/{@code embedding-gecko-*} line (fixed at 768 dimensions, predating
     * MRL) rejects it with a non-retryable {@code 400}. Sending it unconditionally would break
     * those legacy models, so {@link #buildRequestBody(List, String, int, String)} gates it here,
     * mirroring how {@code OpenAiEmbeddingClient} gates its analogous {@code dimensions} parameter
     * by model.
     *
     * @param model the configured model name (without the {@code "models/"} prefix)
     * @return {@code true} when the model accepts an {@code outputDimensionality} field
     */
    protected boolean supportsOutputDimensionality(final String model) {
        return StringUtil.isNotBlank(model) && !model.startsWith(LEGACY_FIXED_DIMENSION_MODEL_PREFIX);
    }

    /**
     * Parses the {@code batchEmbedContents} response body into a list of vectors,
     * validating that the returned vector count matches {@code expectedCount}
     * and that every vector's length matches {@code dimension}. Gemini's
     * {@code embeddings} field is a {@code repeated} proto field with no
     * separate index/id, so this method assumes the API returns embeddings in
     * request order and reassembles them positionally; positional order is the
     * only correlation available (unlike the OpenAI sibling's index-based
     * reassembly).
     *
     * <p>Vectors produced by MRL truncation are L2-normalized here (see
     * {@link #isNormalizationRequired(String, int)}).
     *
     * @param responseBody the raw JSON response body
     * @param expectedCount the expected number of vectors (= number of input texts)
     * @param dimension the expected vector dimension
     * @return the parsed vectors, in response order
     * @throws EmbeddingException if the response is malformed or a count/dimension mismatch is detected
     */
    protected List<float[]> parseEmbedResponse(final String responseBody, final int expectedCount, final int dimension) {
        final JsonNode jsonNode;
        try {
            jsonNode = objectMapper.readTree(responseBody);
        } catch (final IOException e) {
            throw new EmbeddingException("Failed to parse Gemini embed response", e);
        }
        final JsonNode embeddingsNode = jsonNode.path("embeddings");
        if (!embeddingsNode.isArray()) {
            throw new EmbeddingException("Gemini embed response missing 'embeddings' array");
        }
        if (embeddingsNode.size() != expectedCount) {
            throw new EmbeddingException(
                    "Gemini embed response count mismatch: expected=" + expectedCount + ", actual=" + embeddingsNode.size());
        }
        final boolean normalize = isNormalizationRequired(getModel(), dimension);
        final List<float[]> vectors = new ArrayList<>(embeddingsNode.size());
        int index = 0;
        for (final JsonNode entry : embeddingsNode) {
            final JsonNode valuesNode = entry.path("values");
            if (!valuesNode.isArray() || valuesNode.size() != dimension) {
                throw new EmbeddingException("Gemini embed vector dimension mismatch: expected=" + dimension + ", actual="
                        + (valuesNode.isArray() ? valuesNode.size() : -1));
            }
            final float[] vector = new float[dimension];
            for (int i = 0; i < dimension; i++) {
                final JsonNode componentNode = valuesNode.get(i);
                if (componentNode == null || !componentNode.isNumber()) {
                    throw new EmbeddingException("Gemini embed vector component is not numeric: index=" + index + ", position=" + i);
                }
                // isNumber() accepts an overflowing literal like 1e999 (parsed as +Infinity); guard
                // it so a non-finite component never reaches the kNN index as a poisoned vector.
                final float component = (float) componentNode.asDouble();
                if (!Float.isFinite(component)) {
                    throw new EmbeddingException("Gemini embed vector component is not finite: index=" + index + ", position=" + i);
                }
                vector[i] = component;
            }
            if (normalize) {
                normalizeL2(vector);
            }
            vectors.add(vector);
            index++;
        }
        return vectors;
    }

    /**
     * Determines whether parsed vectors need manual L2 normalization. Gemini's embedding models
     * return unit-length vectors only at their native {@link #NATIVE_DIMENSION} output; Google's
     * embeddings documentation states verbatim that for {@code gemini-embedding-001} - this
     * client's default model - "you must manually normalize non-3072 dimensions", because MRL
     * (Matryoshka) truncation drops the tail components and therefore shortens the vector.
     *
     * <p>This matters because core's {@code ChunkVectorHelper} accepts {@code space_type} values
     * that are not scale-invariant ({@code innerproduct}, {@code l2}, {@code l1}, ...): with a
     * truncated, non-normalized vector those metrics silently bias ANN ranking toward chunks whose
     * vectors happen to have a larger norm. The default {@code cosinesimil} is scale-invariant and
     * therefore unaffected, which is why the defect is silent.
     *
     * <p>Normalization is applied only when MRL truncation is actually in effect: the model must
     * accept {@code outputDimensionality} at all (see {@link #supportsOutputDimensionality(String)},
     * which excludes the legacy fixed-dimension line) and the configured dimension must differ from
     * {@link #NATIVE_DIMENSION}.
     *
     * @param model the configured model name (without the {@code "models/"} prefix)
     * @param dimension the configured output dimensionality
     * @return {@code true} when parsed vectors should be L2-normalized
     */
    protected boolean isNormalizationRequired(final String model, final int dimension) {
        return supportsOutputDimensionality(model) && dimension != NATIVE_DIMENSION;
    }

    /**
     * L2-normalizes {@code vector} in place. A zero-norm vector is left unchanged: it carries no
     * direction to preserve, and dividing by zero would write {@code NaN} components that the
     * finite-component guard above has just ruled out and that would poison the kNN index.
     * A non-finite norm (only reachable if the squared sum overflows) is likewise left alone.
     *
     * @param vector the vector to normalize in place
     */
    static void normalizeL2(final float[] vector) {
        double sumOfSquares = 0.0;
        for (final float component : vector) {
            sumOfSquares += (double) component * component;
        }
        final double norm = Math.sqrt(sumOfSquares);
        if (norm <= 0.0 || !Double.isFinite(norm)) {
            return;
        }
        for (int i = 0; i < vector.length; i++) {
            vector[i] = (float) (vector[i] / norm);
        }
    }

    /**
     * Gets the Gemini API key.
     *
     * @return the API key
     */
    protected String getApiKey() {
        return getConfigString("api.key", "");
    }

    /**
     * Gets the Gemini API URL, with any single trailing {@code /} stripped so appending a fixed
     * path ({@code /models}, {@code /models/{model}:batchEmbedContents}) never yields a {@code //}.
     *
     * @return the normalized API URL
     */
    protected String getApiUrl() {
        return GeminiApiUrl.stripTrailingSlash(getConfigString(API_URL_CONFIG_SUFFIX, "https://generativelanguage.googleapis.com/v1beta"));
    }

    /**
     * Builds the full {@code api.url} configuration key (e.g.
     * {@code content_chunker.embedding.gemini.api.url}) from {@link #getConfigPrefix()} and
     * {@link #API_URL_CONFIG_SUFFIX}, so the name reported in error messages can never drift from
     * the key {@link #getApiUrl()} actually reads.
     *
     * @return the fully-qualified {@code api.url} configuration key
     */
    private String apiUrlConfigKey() {
        return getConfigPrefix() + "." + API_URL_CONFIG_SUFFIX;
    }

    /**
     * Reports a userinfo-bearing {@code api.url} and tells the caller to treat the client as
     * unavailable.
     *
     * <p>This is deliberately a fail-closed report rather than a thrown exception. The availability
     * check is reachable from {@code init()}, which LastaDi runs as a {@code postConstruct} while
     * assembling the container: an exception escaping it aborts the assembly and stops the server
     * from starting, turning one mistyped property into a total outage. Reporting the client
     * unavailable degrades exactly the feature that cannot work, which is all a userinfo-bearing
     * URL can ever do anyway - HttpClient rejects such a request URI unconditionally, so the
     * endpoint was already unreachable; only the diagnosis changes.
     *
     * <p>The ERROR is emitted once per broken configuration, because the availability check runs on
     * a timer and would otherwise repeat it for the lifetime of the JVM. It names the setting and
     * the supported alternative but never any part of the URL.
     *
     * @param apiUrl the configured API URL (may be {@code null} or blank)
     * @return {@code true} when the URL carries userinfo and no request may be issued
     */
    protected boolean reportUserInfoApiUrl(final String apiUrl) {
        if (!CredentialUrlUtil.hasUserInfo(apiUrl)) {
            userInfoApiUrlReported.set(false);
            return false;
        }
        if (userInfoApiUrlReported.compareAndSet(false, true)) {
            logger.error("[Embedding:GEMINI] Gemini is not available. {}", GeminiApiUrl.userInfoRejectionMessage(apiUrlConfigKey()));
        }
        return true;
    }

    /**
     * Refuses a userinfo-bearing {@code api.url} before a request URI is built from it.
     *
     * <p>Throwing is safe here in a way it is not in the availability check: this runs only on an
     * explicit embed call, never during container initialization. The refusal replaces an opaque
     * failure raised deep inside HttpClient whose surrounding log lines render the configured URL
     * through a mask that whitespace in the credential defeats.
     *
     * @throws EmbeddingException if the configured API URL carries a userinfo component
     */
    protected void requireNoUserInfoApiUrl() {
        if (CredentialUrlUtil.hasUserInfo(getApiUrl())) {
            throw new EmbeddingException(GeminiApiUrl.userInfoRejectionMessage(apiUrlConfigKey()));
        }
    }

    /**
     * Gets the configured Gemini embedding model name.
     *
     * @return the model name (default {@code gemini-embedding-001})
     */
    protected String getModel() {
        return getConfigString("model", "gemini-embedding-001");
    }

    @Override
    protected int getTimeout() {
        return getConfigInt("timeout", 60000);
    }

    /**
     * Returns the maximum number of attempts (initial + retries) for a single HTTP call.
     * Configured via {@code content_chunker.embedding.gemini.retry.max} (default {@code 10}).
     *
     * @return the maximum number of attempts.
     */
    protected int getRetryMaxAttempts() {
        return getConfigInt("retry.max", 10);
    }

    /**
     * Returns the base delay in milliseconds for exponential backoff between retries.
     * Configured via {@code content_chunker.embedding.gemini.retry.base.delay.ms} (default {@code 2000}).
     *
     * @return the base retry delay in milliseconds.
     */
    protected long getRetryBaseDelayMs() {
        return getConfigLong(CONFIG_RETRY_BASE_DELAY_MS, 2000L);
    }

    @Override
    protected String getConfigPrefix() {
        return "content_chunker.embedding.gemini";
    }

    /**
     * Gets the {@code taskType} sent with every nested request when embedding
     * document/chunk texts (see {@link #embedDocuments(List)}). Defaults to
     * {@code "RETRIEVAL_DOCUMENT"}; set to an empty string to omit the field
     * for models/API versions that don't support it.
     *
     * @return the configured document task type
     */
    protected String getDocumentTaskType() {
        return getConfigString(CONFIG_DOCUMENT_TASK_TYPE, DEFAULT_DOCUMENT_TASK_TYPE);
    }

    /**
     * Gets the {@code taskType} sent with every nested request when embedding
     * query texts (see {@link #embedQuery(List)}). Defaults to
     * {@code "RETRIEVAL_QUERY"}; set to an empty string to omit the field
     * for models/API versions that don't support it.
     *
     * @return the configured query task type
     */
    protected String getQueryTaskType() {
        return getConfigString(CONFIG_QUERY_TASK_TYPE, DEFAULT_QUERY_TASK_TYPE);
    }

    /**
     * Functional interface for the retryable HTTP call body executed by {@link #executeWithRetry}.
     *
     * @param <T> the call result type.
     */
    @FunctionalInterface
    interface HttpCall<T> {
        /**
         * Executes the HTTP call.
         *
         * @return the call result.
         * @throws IOException on I/O failure.
         * @throws ParseException on response parse failure.
         */
        T call() throws IOException, ParseException;
    }

    /**
     * Internal signaling exception thrown by the HTTP call body to indicate that the
     * received status code is retryable (per {@link #isRetryableStatus(int)}). Caught by
     * {@link #executeWithRetry(String, HttpCall)}; never escapes the client.
     */
    static final class RetryableHttpException extends RuntimeException {
        private static final long serialVersionUID = 1L;
        /** The HTTP status code that triggered the retry. */
        final int statusCode;
        /** The HTTP reason phrase associated with {@link #statusCode}. */
        final String reason;

        /**
         * Creates a new {@code RetryableHttpException}.
         *
         * @param statusCode the HTTP status code.
         * @param reason the HTTP reason phrase.
         */
        RetryableHttpException(final int statusCode, final String reason) {
            super("retryable http error: " + statusCode + " " + reason);
            this.statusCode = statusCode;
            this.reason = reason;
        }
    }

    /**
     * Returns whether the given HTTP status code should be retried. Retryable statuses
     * are {@code 429} (RESOURCE_EXHAUSTED), {@code 500} (INTERNAL),
     * {@code 503} (UNAVAILABLE), and {@code 504} (DEADLINE_EXCEEDED). Unlike
     * {@code OllamaEmbeddingClient}, {@code 502} is deliberately excluded, matching
     * {@code GeminiLlmClient.isRetryableStatus}.
     *
     * @param statusCode the HTTP status code.
     * @return {@code true} when the status is retryable.
     */
    static boolean isRetryableStatus(final int statusCode) {
        return statusCode == 429 || statusCode == 500 || statusCode == 503 || statusCode == 504;
    }

    /**
     * Executes {@code call} with retry on {@link RetryableHttpException}. {@link IOException},
     * {@link ParseException}, and {@link EmbeddingException} (RuntimeException, NOT caught here)
     * are all propagated immediately without retry. Backoff is exponential
     * ({@code base * 2^(attempt-1)}) with ±20% jitter, capped per-sleep at {@link #MAX_BACKOFF_MS}.
     * Sleep duration honors {@link #getRetryBaseDelayMs()} and the attempt count is bounded by
     * {@link #getRetryMaxAttempts()}.
     *
     * @param operation the operation label used in log messages (e.g. {@code "embedDocuments"}).
     * @param call the HTTP call body.
     * @param <T> the call result type.
     * @return the call result on success.
     * @throws IOException if the call throws a non-retryable {@link IOException} or the retry
     *             budget is exhausted.
     * @throws ParseException if the call throws {@link ParseException}.
     */
    <T> T executeWithRetry(final String operation, final HttpCall<T> call) throws IOException, ParseException {
        final int maxAttempts = Math.max(1, getRetryMaxAttempts());
        final long baseDelay = Math.max(0L, getRetryBaseDelayMs());
        for (int attempt = 1; attempt <= maxAttempts; attempt++) {
            try {
                return call.call();
            } catch (final RetryableHttpException e) {
                if (attempt == maxAttempts) {
                    logger.warn("[Embedding:GEMINI] {} retry exhausted. attempts={}, lastStatus={}", operation, attempt, e.statusCode);
                    throw new IOException("Gemini API retryable error: " + e.statusCode + " " + e.reason, e);
                }
                sleepBackoff(operation, attempt, maxAttempts, baseDelay, e);
            }
        }
        throw new IllegalStateException("executeWithRetry exited without exception or success");
    }

    /**
     * Sleeps the computed backoff interval. Restores interrupt status if interrupted.
     * Unlike {@code GeminiLlmClient} (and this client's own earlier behavior), the per-sleep
     * delay is capped at {@link #MAX_BACKOFF_MS} by {@link #computeBackoffMs(int, long)} so a
     * persistently-throttled sub-batch cannot stall the caller (e.g. {@code ChunkVectorJob}'s
     * sequential sub-batch fan-out) for many minutes on a single sleep; the worst-case aggregate
     * wait is therefore roughly {@link #getRetryMaxAttempts()} × {@link #MAX_BACKOFF_MS}.
     *
     * @param operation log label.
     * @param attempt 1-based current attempt index.
     * @param maxAttempts total attempts including the first.
     * @param baseDelay base delay in milliseconds (already clamped to {@code >= 0}).
     * @param cause the {@link RetryableHttpException} that triggered the retry.
     * @throws IOException if the sleep is interrupted.
     */
    private void sleepBackoff(final String operation, final int attempt, final int maxAttempts, final long baseDelay,
            final RetryableHttpException cause) throws IOException {
        final long sleepMs = computeBackoffMs(attempt, baseDelay);
        logger.info("[Embedding:GEMINI] {} retrying. attempt={}/{}, status={}, sleepMs={}", operation, attempt, maxAttempts,
                cause.statusCode, sleepMs);
        try {
            Thread.sleep(sleepMs);
        } catch (final InterruptedException ie) {
            Thread.currentThread().interrupt();
            throw new IOException("Retry interrupted", ie);
        }
    }

    /**
     * Computes the capped exponential-backoff delay ({@code baseDelay * 2^(attempt-1)}) with
     * ±20% jitter, clamped to {@code [0, }{@link #MAX_BACKOFF_MS}{@code ]}. The clamp is applied
     * after jitter so the returned value never exceeds the cap; without it a high attempt with the
     * default 2000ms base would sleep many minutes on a single retry.
     *
     * @param attempt 1-based current attempt index.
     * @param baseDelay base delay in milliseconds (already clamped to {@code >= 0}).
     * @return the sleep duration in milliseconds, within {@code [0, MAX_BACKOFF_MS]}.
     */
    static long computeBackoffMs(final int attempt, final long baseDelay) {
        final long jitter = (long) (baseDelay * 0.2 * ThreadLocalRandom.current().nextDouble(-1.0, 1.0)); // +/-20%
        final long delay = (long) (baseDelay * Math.pow(2, attempt - 1)) + jitter;
        return Math.max(0L, Math.min(MAX_BACKOFF_MS, delay));
    }
}
