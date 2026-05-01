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
package org.codelibs.fess.llm.gemini;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;

import org.apache.hc.client5.http.classic.methods.HttpGet;
import org.apache.hc.client5.http.classic.methods.HttpPost;
import org.apache.hc.core5.http.ContentType;
import org.apache.hc.core5.http.ParseException;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.apache.hc.core5.http.io.entity.StringEntity;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.codelibs.core.lang.StringUtil;
import org.codelibs.fess.llm.AbstractLlmClient;
import org.codelibs.fess.llm.LlmChatRequest;
import org.codelibs.fess.llm.LlmChatResponse;
import org.codelibs.fess.llm.LlmException;
import org.codelibs.fess.llm.LlmMessage;
import org.codelibs.fess.llm.LlmStreamCallback;
import org.codelibs.fess.util.ComponentUtil;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;

/**
 * LLM client implementation for Google Gemini API.
 *
 * Google Gemini provides cloud-based LLM services including Gemini models.
 * This client supports both synchronous and streaming chat completions.
 *
 * @author FessProject
 * @see <a href="https://ai.google.dev/docs">Google AI for Developers</a>
 */
public class GeminiLlmClient extends AbstractLlmClient {

    private static final Logger logger = LogManager.getLogger(GeminiLlmClient.class);
    /** The name identifier for the Gemini LLM client. */
    protected static final String NAME = "gemini";

    /** Gemini role for model responses (equivalent to "assistant" in OpenAI). */
    protected static final String ROLE_MODEL = "model";

    /**
     * Summary of a single streamChat invocation. Exposed for diagnostics, not part of the LLM SPI.
     */
    public static final class StreamSummary {
        /** Number of text chunks delivered to the callback. */
        public final int chunkCount;
        /** Number of complete JSON objects parsed from the stream. */
        public final int objectCount;
        /** Last observed {@code finishReason} value, or {@code null} if none was reported. */
        public final String finishReason;
        /** Last observed top-level {@code responseId} value, or {@code null} if none was reported. */
        public final String responseId;
        /** Last observed {@code promptTokenCount}, or {@code null} if absent. */
        public final Integer promptTokenCount;
        /** Last observed {@code cachedContentTokenCount}, or {@code null} if absent. */
        public final Integer cachedContentTokenCount;
        /** Last observed {@code candidatesTokenCount}, or {@code null} if absent. */
        public final Integer candidatesTokenCount;
        /** Last observed {@code thoughtsTokenCount}, or {@code null} if absent. */
        public final Integer thoughtsTokenCount;
        /** Last observed {@code totalTokenCount}, or {@code null} if absent. */
        public final Integer totalTokenCount;
        /** Milliseconds from request start to first text chunk. */
        public final long firstChunkMs;
        /** Total milliseconds from request start to stream end. */
        public final long elapsedMs;

        StreamSummary(final int chunkCount, final int objectCount, final String finishReason, final String responseId,
                final Integer promptTokenCount, final Integer cachedContentTokenCount, final Integer candidatesTokenCount,
                final Integer thoughtsTokenCount, final Integer totalTokenCount, final long firstChunkMs, final long elapsedMs) {
            this.chunkCount = chunkCount;
            this.objectCount = objectCount;
            this.finishReason = finishReason;
            this.responseId = responseId;
            this.promptTokenCount = promptTokenCount;
            this.cachedContentTokenCount = cachedContentTokenCount;
            this.candidatesTokenCount = candidatesTokenCount;
            this.thoughtsTokenCount = thoughtsTokenCount;
            this.totalTokenCount = totalTokenCount;
            this.firstChunkMs = firstChunkMs;
            this.elapsedMs = elapsedMs;
        }
    }

    /** Test hook; not thread-safe. Set once before invoking streamChat from a single thread. */
    private java.util.function.Consumer<StreamSummary> streamSummaryConsumer;

    /**
     * Test hook: receives the per-call {@link StreamSummary} right after the completion log line.
     *
     * @param consumer summary consumer; pass {@code null} to clear.
     */
    void setStreamSummaryConsumer(final java.util.function.Consumer<StreamSummary> consumer) {
        this.streamSummaryConsumer = consumer;
    }

    /**
     * Default constructor.
     */
    public GeminiLlmClient() {
        // Default constructor
    }

    @Override
    public String getName() {
        return NAME;
    }

    @Override
    protected boolean checkAvailabilityNow() {
        final String apiKey = getApiKey();
        if (StringUtil.isBlank(apiKey)) {
            if (logger.isDebugEnabled()) {
                logger.debug("[LLM:GEMINI] Gemini is not available. apiKey is blank");
            }
            return false;
        }
        final String apiUrl = getApiUrl();
        if (StringUtil.isBlank(apiUrl)) {
            if (logger.isDebugEnabled()) {
                logger.debug("[LLM:GEMINI] Gemini is not available. apiUrl is blank");
            }
            return false;
        }
        try {
            final String url = apiUrl + "/models";
            final HttpGet request = new HttpGet(url);
            request.addHeader("x-goog-api-key", apiKey);
            try (var response = getHttpClient().execute(request)) {
                final int statusCode = response.getCode();
                final boolean available = statusCode >= 200 && statusCode < 300;
                if (logger.isDebugEnabled()) {
                    logger.debug("[LLM:GEMINI] Gemini availability check. url={}, statusCode={}, available={}", apiUrl, statusCode,
                            available);
                }
                return available;
            }
        } catch (final Exception e) {
            if (logger.isDebugEnabled()) {
                logger.debug("[LLM:GEMINI] Gemini is not available. url={}, error={}", apiUrl, e.getMessage());
            }
            return false;
        }
    }

    @Override
    public LlmChatResponse chat(final LlmChatRequest request) {
        final String model = getModelName(request);
        final String url = buildApiUrl(model, false);
        final Map<String, Object> requestBody = buildRequestBody(request);
        final long startTime = System.currentTimeMillis();

        if (logger.isDebugEnabled()) {
            logger.debug("[LLM:GEMINI] Sending chat request to Gemini. url={}, model={}, messageCount={}", maskApiKeyInUrl(url), model,
                    request.getMessages().size());
        }

        try {
            final String json = objectMapper.writeValueAsString(requestBody);
            if (logger.isDebugEnabled()) {
                logger.debug("[LLM:GEMINI] requestBody={}", json);
            }
            final HttpPost httpRequest = new HttpPost(url);
            httpRequest.addHeader("x-goog-api-key", getApiKey());
            httpRequest.setEntity(new StringEntity(json, ContentType.APPLICATION_JSON));

            return executeWithRetry("chat", () -> {
                try (var response = getHttpClient().execute(httpRequest)) {
                    final int statusCode = response.getCode();
                    if (statusCode < 200 || statusCode >= 300) {
                        String errorBody = "";
                        if (response.getEntity() != null) {
                            try {
                                errorBody = EntityUtils.toString(response.getEntity());
                            } catch (final IOException e) {
                                // ignore
                            }
                        }
                        logger.warn("[LLM:GEMINI] API error. url={}, statusCode={}, message={}, body={}", maskApiKeyInUrl(url), statusCode,
                                response.getReasonPhrase(), errorBody);
                        if (isRetryableStatus(statusCode)) {
                            throw new RetryableHttpException(statusCode, response.getReasonPhrase());
                        }
                        throw new LlmException("Gemini API error: " + statusCode + " " + response.getReasonPhrase(),
                                resolveErrorCode(statusCode));
                    }

                    final String responseBody = response.getEntity() != null ? EntityUtils.toString(response.getEntity()) : "";
                    if (logger.isDebugEnabled()) {
                        logger.debug("[LLM:GEMINI] responseBody={}", responseBody);
                    }
                    final JsonNode jsonNode = objectMapper.readTree(responseBody);

                    final LlmChatResponse chatResponse = new LlmChatResponse();
                    if (jsonNode.has("candidates") && jsonNode.get("candidates").isArray() && jsonNode.get("candidates").size() > 0) {
                        final JsonNode firstCandidate = jsonNode.get("candidates").get(0);
                        if (firstCandidate.has("content") && firstCandidate.get("content").has("parts")) {
                            final JsonNode parts = firstCandidate.get("content").get("parts");
                            if (parts.isArray()) {
                                final StringBuilder textContent = new StringBuilder();
                                for (int i = 0; i < parts.size(); i++) {
                                    final JsonNode part = parts.get(i);
                                    // Skip thinking parts
                                    if (part.has("thought") && part.get("thought").asBoolean(false)) {
                                        continue;
                                    }
                                    if (part.has("text")) {
                                        textContent.append(part.get("text").asText());
                                    }
                                }
                                if (textContent.length() > 0) {
                                    chatResponse.setContent(textContent.toString());
                                }
                            }
                        }
                        if (firstCandidate.has("finishReason") && !firstCandidate.get("finishReason").isNull()) {
                            chatResponse.setFinishReason(firstCandidate.get("finishReason").asText());
                        }
                    }
                    if (jsonNode.has("modelVersion")) {
                        chatResponse.setModel(jsonNode.get("modelVersion").asText());
                    } else {
                        chatResponse.setModel(model);
                    }
                    Integer cachedTokens = null;
                    if (jsonNode.has("usageMetadata")) {
                        final JsonNode usage = jsonNode.get("usageMetadata");
                        if (usage.has("promptTokenCount")) {
                            chatResponse.setPromptTokens(usage.get("promptTokenCount").asInt());
                        }
                        if (usage.has("cachedContentTokenCount")) {
                            cachedTokens = usage.get("cachedContentTokenCount").asInt();
                        }
                        if (usage.has("candidatesTokenCount")) {
                            chatResponse.setCompletionTokens(usage.get("candidatesTokenCount").asInt());
                        }
                        if (usage.has("totalTokenCount")) {
                            chatResponse.setTotalTokens(usage.get("totalTokenCount").asInt());
                        }
                    }
                    String responseId = null;
                    if (jsonNode.has("responseId") && !jsonNode.get("responseId").isNull()) {
                        responseId = jsonNode.get("responseId").asText();
                    }

                    logger.info(
                            "[LLM:GEMINI] Chat response received. model={}, responseId={}, promptTokens={}, cachedContentTokens={}, completionTokens={}, totalTokens={}, contentLength={}, elapsedTime={}ms",
                            chatResponse.getModel(), responseId, chatResponse.getPromptTokens(), cachedTokens,
                            chatResponse.getCompletionTokens(), chatResponse.getTotalTokens(),
                            chatResponse.getContent() != null ? chatResponse.getContent().length() : 0,
                            System.currentTimeMillis() - startTime);
                    if (isAbnormalFinishReason(chatResponse.getFinishReason())) {
                        logger.warn("[LLM:GEMINI] Chat finished abnormally. finishReason={}, contentLength={}, model={}",
                                chatResponse.getFinishReason(), chatResponse.getContent() != null ? chatResponse.getContent().length() : 0,
                                chatResponse.getModel());
                    }
                    if (jsonNode.has("promptFeedback")) {
                        final JsonNode pf = jsonNode.get("promptFeedback");
                        if (pf.has("blockReason") && !pf.get("blockReason").isNull()) {
                            logger.warn("[LLM:GEMINI] Prompt blocked. blockReason={}, safetyRatings={}, model={}",
                                    pf.get("blockReason").asText(), stringifySafetyRatings(pf), chatResponse.getModel());
                        }
                    }
                    if (isAbnormalFinishReason(chatResponse.getFinishReason()) && jsonNode.has("candidates")
                            && jsonNode.get("candidates").size() > 0) {
                        final JsonNode firstCandidate = jsonNode.get("candidates").get(0);
                        final String safetyDetail = stringifySafetyRatings(firstCandidate);
                        if (safetyDetail != null) {
                            logger.warn("[LLM:GEMINI] Candidate safety ratings. finishReason={}, safetyRatings={}, model={}",
                                    chatResponse.getFinishReason(), safetyDetail, chatResponse.getModel());
                        }
                    }

                    return chatResponse;
                }
            });
        } catch (final LlmException e) {
            throw e;
        } catch (final RetryableHttpException e) {
            // Defensive: executeWithRetry consumes RetryableHttpException, so this should never fire today.
            // Documents intent and protects future refactors.
            throw new LlmException("Gemini API retryable exhausted", LlmException.ERROR_CONNECTION, e);
        } catch (final Exception e) {
            logger.warn("[LLM:GEMINI] Failed to call Gemini API. url={}, error={}", maskApiKeyInUrl(url), e.getMessage(), e);
            throw new LlmException("Failed to call Gemini API", LlmException.ERROR_CONNECTION, e);
        }
    }

    @Override
    public void streamChat(final LlmChatRequest request, final LlmStreamCallback callback) {
        final String model = getModelName(request);
        final String url = buildApiUrl(model, true);
        final Map<String, Object> requestBody = buildRequestBody(request);
        final long startTime = System.currentTimeMillis();

        if (logger.isDebugEnabled()) {
            logger.debug("[LLM:GEMINI] Starting streaming chat request to Gemini. url={}, model={}, messageCount={}", maskApiKeyInUrl(url),
                    model, request.getMessages().size());
        }

        try {
            final String json = objectMapper.writeValueAsString(requestBody);
            if (logger.isDebugEnabled()) {
                logger.debug("[LLM:GEMINI] requestBody={}", json);
            }
            final HttpPost httpRequest = new HttpPost(url);
            httpRequest.addHeader("x-goog-api-key", getApiKey());
            httpRequest.setEntity(new StringEntity(json, ContentType.APPLICATION_JSON));

            executeWithRetry("streamChat", () -> {
                try (var response = getHttpClient().execute(httpRequest)) {
                    final int statusCode = response.getCode();
                    if (logger.isDebugEnabled()) {
                        final var ctHeader = response.getFirstHeader("Content-Type");
                        logger.debug("[LLM:GEMINI] streamGenerateContent http response. statusCode={}, contentType={}", statusCode,
                                ctHeader != null ? ctHeader.getValue() : null);
                    }
                    if (statusCode < 200 || statusCode >= 300) {
                        String errorBody = "";
                        if (response.getEntity() != null) {
                            try {
                                errorBody = EntityUtils.toString(response.getEntity());
                            } catch (final IOException | ParseException e) {
                                // ignore
                            }
                        }
                        logger.warn("[LLM:GEMINI] Streaming API error. url={}, statusCode={}, message={}, body={}", maskApiKeyInUrl(url),
                                statusCode, response.getReasonPhrase(), errorBody);
                        if (isRetryableStatus(statusCode)) {
                            throw new RetryableHttpException(statusCode, response.getReasonPhrase());
                        }
                        throw new LlmException("Gemini API error: " + statusCode + " " + response.getReasonPhrase(),
                                resolveErrorCode(statusCode));
                    }

                    if (response.getEntity() == null) {
                        logger.warn("[LLM:GEMINI] Empty response from Gemini streaming API. url={}", maskApiKeyInUrl(url));
                        throw new LlmException("Empty response from Gemini");
                    }

                    int chunkCount = 0;
                    int objectCount = 0;
                    long firstChunkTime = 0;
                    String lastFinishReason = null;
                    String lastResponseId = null;
                    String lastBlockReason = null;
                    String lastSafetyRatings = null;
                    Integer promptTokenCount = null;
                    Integer cachedContentTokenCount = null;
                    Integer candidatesTokenCount = null;
                    Integer thoughtsTokenCount = null;
                    Integer totalTokenCount = null;
                    try (BufferedReader reader =
                            new BufferedReader(new InputStreamReader(response.getEntity().getContent(), StandardCharsets.UTF_8))) {
                        final StringBuilder jsonBuffer = new StringBuilder();
                        int braceDepth = 0;
                        boolean inString = false;
                        boolean streamDone = false;

                        String line;
                        while ((line = reader.readLine()) != null && !streamDone) {
                            if (StringUtil.isBlank(line)) {
                                continue;
                            }
                            // SSE comment line, e.g. ": keepalive"
                            if (line.charAt(0) == ':') {
                                continue;
                            }
                            // SSE terminator sometimes used by proxies / older deployments
                            final String trimmed = line.trim();
                            if ("data: [DONE]".equals(trimmed) || "[DONE]".equals(trimmed)) {
                                streamDone = true;
                                continue;
                            }

                            for (int ci = 0; ci < line.length() && !streamDone; ci++) {
                                final char c = line.charAt(ci);

                                if (braceDepth == 0) {
                                    // Outside JSON object - skip array-level delimiters
                                    if (c == '{') {
                                        braceDepth = 1;
                                        jsonBuffer.setLength(0);
                                        jsonBuffer.append(c);
                                    }
                                    continue;
                                }

                                // Inside JSON object
                                jsonBuffer.append(c);

                                if (inString) {
                                    if (c == '\\') {
                                        // Escape sequence - append next char(s) and skip
                                        ci++;
                                        if (ci < line.length()) {
                                            final char escaped = line.charAt(ci);
                                            jsonBuffer.append(escaped);
                                            if (escaped == 'u') {
                                                // Unicode escape sequence - skip 4 hex digits
                                                final int end = Math.min(ci + 4, line.length() - 1);
                                                for (int ui = ci + 1; ui <= end; ui++) {
                                                    jsonBuffer.append(line.charAt(ui));
                                                }
                                                ci = end;
                                            }
                                        }
                                    } else if (c == '"') {
                                        inString = false;
                                    }
                                } else {
                                    if (c == '"') {
                                        inString = true;
                                    } else if (c == '{') {
                                        braceDepth++;
                                    } else if (c == '}') {
                                        braceDepth--;
                                        if (braceDepth == 0) {
                                            // Complete JSON object accumulated
                                            final String jsonStr = jsonBuffer.toString();
                                            jsonBuffer.setLength(0);

                                            try {
                                                final JsonNode jsonNode = objectMapper.readTree(jsonStr);
                                                objectCount++;
                                                if (logger.isDebugEnabled()) {
                                                    logger.debug("[LLM:GEMINI] streamObject#{} json={}", objectCount, jsonStr);
                                                }
                                                if (jsonNode.has("responseId") && !jsonNode.get("responseId").isNull()) {
                                                    lastResponseId = jsonNode.get("responseId").asText();
                                                }
                                                if (jsonNode.has("promptFeedback")) {
                                                    final JsonNode pf = jsonNode.get("promptFeedback");
                                                    if (pf.has("blockReason") && !pf.get("blockReason").isNull()) {
                                                        lastBlockReason = pf.get("blockReason").asText();
                                                    }
                                                }
                                                if (jsonNode.has("usageMetadata")) {
                                                    final JsonNode usage = jsonNode.get("usageMetadata");
                                                    if (usage.has("promptTokenCount")) {
                                                        promptTokenCount = usage.get("promptTokenCount").asInt();
                                                    }
                                                    if (usage.has("cachedContentTokenCount")) {
                                                        cachedContentTokenCount = usage.get("cachedContentTokenCount").asInt();
                                                    }
                                                    if (usage.has("candidatesTokenCount")) {
                                                        candidatesTokenCount = usage.get("candidatesTokenCount").asInt();
                                                    }
                                                    if (usage.has("thoughtsTokenCount")) {
                                                        thoughtsTokenCount = usage.get("thoughtsTokenCount").asInt();
                                                    }
                                                    if (usage.has("totalTokenCount")) {
                                                        totalTokenCount = usage.get("totalTokenCount").asInt();
                                                    }
                                                }

                                                boolean done = false;
                                                if (jsonNode.has("candidates") && jsonNode.get("candidates").isArray()
                                                        && jsonNode.get("candidates").size() > 0) {
                                                    final JsonNode firstCandidate = jsonNode.get("candidates").get(0);
                                                    final String safetyDetail = stringifySafetyRatings(firstCandidate);
                                                    if (safetyDetail != null) {
                                                        lastSafetyRatings = safetyDetail;
                                                    }
                                                    final String reason = extractFinishReason(firstCandidate);
                                                    if (reason != null) {
                                                        lastFinishReason = reason;
                                                        done = true;
                                                    }

                                                    if (firstCandidate.has("content") && firstCandidate.get("content").has("parts")) {
                                                        final JsonNode parts = firstCandidate.get("content").get("parts");
                                                        boolean textSent = false;
                                                        if (parts.isArray()) {
                                                            for (int pi = 0; pi < parts.size(); pi++) {
                                                                final JsonNode part = parts.get(pi);
                                                                // Skip thinking parts
                                                                if (part.has("thought") && part.get("thought").asBoolean(false)) {
                                                                    continue;
                                                                }
                                                                if (part.has("text")) {
                                                                    callback.onChunk(part.get("text").asText(), done);
                                                                    if (chunkCount == 0) {
                                                                        firstChunkTime = System.currentTimeMillis() - startTime;
                                                                    }
                                                                    chunkCount++;
                                                                    textSent = true;
                                                                }
                                                            }
                                                        }
                                                        if (done && !textSent) {
                                                            callback.onChunk("", true);
                                                        }
                                                    } else if (done) {
                                                        callback.onChunk("", true);
                                                    }

                                                    if (done) {
                                                        streamDone = true;
                                                    }
                                                }
                                            } catch (final JsonProcessingException e) {
                                                logger.warn("[LLM:GEMINI] Failed to parse streaming response. json={}", jsonStr, e);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    final long elapsed = System.currentTimeMillis() - startTime;
                    logger.info(
                            "[LLM:GEMINI] Stream completed. chunkCount={}, objectCount={}, firstChunkMs={}, elapsedTime={}ms, responseId={}, finishReason={}, promptTokens={}, cachedContentTokens={}, candidatesTokens={}, thoughtsTokens={}, totalTokens={}",
                            chunkCount, objectCount, firstChunkTime, elapsed, lastResponseId, lastFinishReason, promptTokenCount,
                            cachedContentTokenCount, candidatesTokenCount, thoughtsTokenCount, totalTokenCount);
                    if (isAbnormalFinishReason(lastFinishReason)) {
                        logger.warn(
                                "[LLM:GEMINI] Stream finished abnormally. responseId={}, finishReason={}, chunkCount={}, candidatesTokens={}, thoughtsTokens={}, model={}",
                                lastResponseId, lastFinishReason, chunkCount, candidatesTokenCount, thoughtsTokenCount, model);
                    }
                    if (lastBlockReason != null) {
                        logger.warn("[LLM:GEMINI] Stream prompt blocked. blockReason={}, model={}", lastBlockReason, model);
                    }
                    if (isAbnormalFinishReason(lastFinishReason) && lastSafetyRatings != null) {
                        logger.warn("[LLM:GEMINI] Stream candidate safety ratings. finishReason={}, safetyRatings={}, model={}",
                                lastFinishReason, lastSafetyRatings, model);
                    }
                    if (streamSummaryConsumer != null) {
                        streamSummaryConsumer.accept(new StreamSummary(chunkCount, objectCount, lastFinishReason, lastResponseId,
                                promptTokenCount, cachedContentTokenCount, candidatesTokenCount, thoughtsTokenCount, totalTokenCount,
                                firstChunkTime, elapsed));
                    }
                    return null;
                }
            });
        } catch (final LlmException e) {
            callback.onError(e);
            throw e;
        } catch (final IOException e) {
            logger.warn("[LLM:GEMINI] Failed to stream from Gemini API. url={}, error={}", maskApiKeyInUrl(url), e.getMessage(), e);
            final LlmException llmException = new LlmException("Failed to stream from Gemini API", LlmException.ERROR_CONNECTION, e);
            callback.onError(llmException);
            throw llmException;
        } catch (final ParseException e) {
            logger.warn("[LLM:GEMINI] Failed to stream from Gemini API. url={}, error={}", maskApiKeyInUrl(url), e.getMessage(), e);
            final LlmException llmException = new LlmException("Failed to stream from Gemini API", LlmException.ERROR_CONNECTION, e);
            callback.onError(llmException);
            throw llmException;
        }
    }

    /**
     * Gets the model name from the request or config.
     *
     * @param request the chat request
     * @return the model name
     */
    protected String getModelName(final LlmChatRequest request) {
        String model = request.getModel();
        if (StringUtil.isBlank(model)) {
            model = getModel();
        }
        return model;
    }

    /**
     * Builds the API URL for the specified model.
     *
     * @param model the model name
     * @param stream whether this is a streaming request
     * @return the complete API URL
     */
    protected String buildApiUrl(final String model, final boolean stream) {
        final String apiUrl = getApiUrl();
        final String action = stream ? "streamGenerateContent" : "generateContent";
        final StringBuilder url = new StringBuilder().append(apiUrl).append("/models/").append(model).append(":").append(action);
        if (stream) {
            url.append("?alt=sse");
        }
        return url.toString();
    }

    /**
     * Builds the request body for the Gemini API.
     *
     * @param request the chat request
     * @return the request body as a map
     */
    protected Map<String, Object> buildRequestBody(final LlmChatRequest request) {
        final Map<String, Object> body = new HashMap<>();

        String systemMessage = null;
        final List<LlmMessage> conversationMessages = new ArrayList<>();
        for (final LlmMessage message : request.getMessages()) {
            if (LlmMessage.ROLE_SYSTEM.equals(message.getRole())) {
                if (systemMessage == null) {
                    systemMessage = message.getContent();
                } else {
                    systemMessage = systemMessage + "\n" + message.getContent();
                }
            } else {
                conversationMessages.add(message);
            }
        }

        if (systemMessage != null) {
            final Map<String, Object> systemInstruction = new HashMap<>();
            final List<Map<String, String>> systemParts = new ArrayList<>();
            final Map<String, String> textPart = new HashMap<>();
            textPart.put("text", systemMessage);
            systemParts.add(textPart);
            systemInstruction.put("parts", systemParts);
            body.put("systemInstruction", systemInstruction);
        }

        final List<Map<String, Object>> contents = conversationMessages.stream().map(this::convertMessage).collect(Collectors.toList());
        body.put("contents", contents);

        final Map<String, Object> generationConfig = new HashMap<>();
        if (request.getTemperature() != null) {
            generationConfig.put("temperature", request.getTemperature());
        }
        if (request.getMaxTokens() != null) {
            generationConfig.put("maxOutputTokens", request.getMaxTokens());
        }
        if (request.getThinkingBudget() != null) {
            final Map<String, Object> thinkingConfig = new HashMap<>();
            final String model = getModelName(request);
            if (isGemini3(model)) {
                thinkingConfig.put("thinkingLevel", budgetToThinkingLevel(request.getThinkingBudget()));
            } else {
                thinkingConfig.put("thinkingBudget", request.getThinkingBudget());
            }
            generationConfig.put("thinkingConfig", thinkingConfig);
        }
        if (!generationConfig.isEmpty()) {
            body.put("generationConfig", generationConfig);
        }

        return body;
    }

    /**
     * Converts an LlmMessage to a map for the Gemini API request.
     *
     * @param message the message to convert
     * @return the message as a map
     */
    protected Map<String, Object> convertMessage(final LlmMessage message) {
        final Map<String, Object> map = new HashMap<>();

        String role = message.getRole();
        if (LlmMessage.ROLE_ASSISTANT.equals(role)) {
            role = ROLE_MODEL;
        }
        map.put("role", role);

        final List<Map<String, String>> parts = new ArrayList<>();
        final Map<String, String> textPart = new HashMap<>();
        textPart.put("text", message.getContent());
        parts.add(textPart);
        map.put("parts", parts);

        return map;
    }

    /**
     * Masks the API key in a URL by replacing the key value with "***".
     *
     * @param url the URL that may contain an API key parameter
     * @return the URL with the API key masked
     */
    protected String maskApiKeyInUrl(final String url) {
        if (url == null) {
            return null;
        }
        return url.replaceAll("([?&])key=[^&]*", "$1key=***");
    }

    /**
     * Gets the Gemini API key.
     *
     * @return the API key
     */
    protected String getApiKey() {
        return ComponentUtil.getFessConfig().getOrDefault("rag.llm.gemini.api.key", "");
    }

    /**
     * Gets the Gemini API URL.
     *
     * @return the API URL
     */
    protected String getApiUrl() {
        return ComponentUtil.getFessConfig().getOrDefault("rag.llm.gemini.api.url", "https://generativelanguage.googleapis.com/v1beta");
    }

    @Override
    protected String getModel() {
        return ComponentUtil.getFessConfig().getOrDefault("rag.llm.gemini.model", "gemini-3.1-flash-lite-preview");
    }

    @Override
    protected int getTimeout() {
        return getConfigInt("timeout", 60000);
    }

    @Override
    protected String getConfigPrefix() {
        return "rag.llm.gemini";
    }

    @Override
    protected void applyPromptTypeParams(final LlmChatRequest request, final String promptType) {
        super.applyPromptTypeParams(request, promptType);
        applyDefaultParams(request, promptType);
    }

    /**
     * Extra {@code maxOutputTokens} headroom added to Gemini 3 default budgets so the
     * mandatory thinking tokens (the {@code thinkingLevel=LOW} bucket still consumes a
     * few hundred tokens per response) do not crowd out the visible reply.
     *
     * <p>Gemini 2.x honours {@code thinkingBudget=0} as "thinking off" so the visible
     * budget alone is enough; Gemini 3.x always emits some thinking tokens even at the
     * lowest level, so the cap must cover both.
     */
    static final int GEMINI3_THINKING_HEADROOM = 1024;

    /**
     * Resolves the default {@code maxOutputTokens} for a prompt type, adding
     * {@link #GEMINI3_THINKING_HEADROOM} when the resolved model is a Gemini 3.x model
     * so the mandatory thinking spend does not eat into the visible-output budget.
     *
     * @param visibleTokens the budget required for the actual visible response.
     * @param gemini3 {@code true} when the active model is Gemini 3.x.
     * @return the resolved default {@code maxOutputTokens}.
     */
    static int defaultMaxTokens(final int visibleTokens, final boolean gemini3) {
        return gemini3 ? visibleTokens + GEMINI3_THINKING_HEADROOM : visibleTokens;
    }

    /**
     * Applies default generation parameters for the Gemini API free tier.
     * Only sets defaults when user has not configured the parameter.
     *
     * <p>The {@code maxOutputTokens} default is model-aware: for Gemini 3.x models the
     * mandatory thinking token spend (even at {@code thinkingLevel=LOW}) is added on top
     * of each prompt type's visible-output budget so responses do not get truncated with
     * {@code finishReason=MAX_TOKENS}. Gemini 2.x defaults are unchanged.
     *
     * @param request the LLM chat request
     * @param promptType the prompt type (e.g. "intent", "evaluation", "answer")
     */
    protected void applyDefaultParams(final LlmChatRequest request, final String promptType) {
        final boolean gemini3 = isGemini3(getModelName(request));
        switch (promptType) {
        case "intent":
            if (request.getTemperature() == null) {
                request.setTemperature(0.1);
            }
            if (request.getMaxTokens() == null) {
                // Intent JSON includes a query string with Fess query syntax plus a
                // reasoning field; in non-English locales (e.g. Japanese) reasoning
                // tokens can easily push the visible output above 256.
                request.setMaxTokens(defaultMaxTokens(512, gemini3));
            }
            if (request.getThinkingBudget() == null) {
                request.setThinkingBudget(0);
            }
            break;
        case "evaluation":
            if (request.getTemperature() == null) {
                request.setTemperature(0.1);
            }
            if (request.getMaxTokens() == null) {
                // Tiny JSON: {"relevant_indexes":[..],"has_relevant":bool}
                request.setMaxTokens(defaultMaxTokens(256, gemini3));
            }
            if (request.getThinkingBudget() == null) {
                request.setThinkingBudget(0);
            }
            break;
        case "unclear":
        case "noresults":
            if (request.getTemperature() == null) {
                request.setTemperature(0.7);
            }
            if (request.getMaxTokens() == null) {
                request.setMaxTokens(defaultMaxTokens(512, gemini3));
            }
            if (request.getThinkingBudget() == null) {
                request.setThinkingBudget(0);
            }
            break;
        case "docnotfound":
            if (request.getTemperature() == null) {
                request.setTemperature(0.7);
            }
            if (request.getMaxTokens() == null) {
                // Polite multi-bullet message that re-prints the requested URL; sized
                // to match unclear/noresults so non-English (e.g. Japanese) responses
                // do not truncate.
                request.setMaxTokens(defaultMaxTokens(512, gemini3));
            }
            if (request.getThinkingBudget() == null) {
                request.setThinkingBudget(0);
            }
            break;
        case "direct":
        case "faq":
            if (request.getTemperature() == null) {
                request.setTemperature(0.7);
            }
            if (request.getMaxTokens() == null) {
                request.setMaxTokens(defaultMaxTokens(2048, gemini3));
            }
            if (request.getThinkingBudget() == null) {
                request.setThinkingBudget(0);
            }
            break;
        case "answer":
            if (request.getTemperature() == null) {
                request.setTemperature(0.5);
            }
            if (request.getMaxTokens() == null) {
                request.setMaxTokens(defaultMaxTokens(8192, gemini3));
            }
            if (request.getThinkingBudget() == null) {
                request.setThinkingBudget(0);
            }
            break;
        case "summary":
            if (request.getTemperature() == null) {
                request.setTemperature(0.3);
            }
            if (request.getMaxTokens() == null) {
                request.setMaxTokens(defaultMaxTokens(4096, gemini3));
            }
            if (request.getThinkingBudget() == null) {
                request.setThinkingBudget(0);
            }
            break;
        case "queryregeneration":
            if (request.getTemperature() == null) {
                request.setTemperature(0.3);
            }
            if (request.getMaxTokens() == null) {
                request.setMaxTokens(defaultMaxTokens(256, gemini3));
            }
            if (request.getThinkingBudget() == null) {
                request.setThinkingBudget(0);
            }
            break;
        default:
            break;
        }
    }

    @Override
    protected int getAvailabilityCheckInterval() {
        return getConfigInt("availability.check.interval", 60);
    }

    @Override
    protected boolean isRagChatEnabled() {
        return Boolean.parseBoolean(ComponentUtil.getFessConfig().getOrDefault("rag.chat.enabled", "false"));
    }

    @Override
    protected String getLlmType() {
        return ComponentUtil.getFessConfig().getSystemProperty("rag.llm.name", "ollama");
    }

    @Override
    protected int getContextMaxChars(final String promptType) {
        final String key = "rag.llm.gemini." + promptType + ".context.max.chars";
        final String configValue = ComponentUtil.getFessConfig().getOrDefault(key, null);
        if (configValue != null) {
            final int value = Integer.parseInt(configValue);
            if (value > 0) {
                return value;
            }
            logger.warn("Invalid context max chars for promptType={}: {}. Using default.", promptType, value);
        }
        switch (promptType) {
        case "answer":
            return 16000;
        case "summary":
            return 16000;
        case "faq":
            return 10000;
        default:
            return 10000;
        }
    }

    @Override
    protected int getEvaluationMaxRelevantDocs() {
        return getConfigInt("chat.evaluation.max.relevant.docs", 3);
    }

    @Override
    protected int getEvaluationDescriptionMaxChars() {
        return getConfigInt("chat.evaluation.description.max.chars", 500);
    }

    @Override
    protected int getHistoryMaxChars() {
        return getConfigInt("history.max.chars", 10000);
    }

    @Override
    protected int getIntentHistoryMaxMessages() {
        return getConfigInt("intent.history.max.messages", 10);
    }

    @Override
    protected int getIntentHistoryMaxChars() {
        return getConfigInt("intent.history.max.chars", 5000);
    }

    @Override
    public int getHistoryAssistantMaxChars() {
        return getConfigInt("history.assistant.max.chars", 1000);
    }

    @Override
    public int getHistoryAssistantSummaryMaxChars() {
        return getConfigInt("history.assistant.summary.max.chars", 1000);
    }

    /**
     * Extracts the {@code finishReason} string from a candidate JSON node.
     *
     * @param candidate the candidate JSON node (may be {@code null}).
     * @return the {@code finishReason} text, or {@code null} when absent, JSON-null, blank, or the literal "null".
     */
    static String extractFinishReason(final JsonNode candidate) {
        if (candidate == null || !candidate.has("finishReason")) {
            return null;
        }
        final JsonNode node = candidate.get("finishReason");
        if (node.isNull()) {
            return null;
        }
        final String text = node.asText();
        if (StringUtil.isBlank(text) || "null".equals(text)) {
            return null;
        }
        return text;
    }

    /**
     * Returns whether the given {@code finishReason} indicates an abnormal stream/chat completion.
     *
     * @param reason the {@code finishReason} text (may be {@code null}).
     * @return {@code false} for {@code null}, {@code "STOP"}, and {@code "FINISH_REASON_UNSPECIFIED"}; {@code true} otherwise
     *         (e.g. {@code MAX_TOKENS}, {@code SAFETY}, {@code RECITATION}, {@code BLOCKLIST}, {@code PROHIBITED_CONTENT},
     *         {@code SPII}, {@code MALFORMED_FUNCTION_CALL}, {@code IMAGE_SAFETY}, {@code OTHER}).
     */
    static boolean isAbnormalFinishReason(final String reason) {
        if (reason == null || "STOP".equals(reason) || "FINISH_REASON_UNSPECIFIED".equals(reason)) {
            return false;
        }
        // LANGUAGE / SAFETY / RECITATION / BLOCKLIST / PROHIBITED_CONTENT / SPII /
        // MALFORMED_FUNCTION_CALL / IMAGE_SAFETY / OTHER all flow through this
        // catch-all so the WARN sibling log fires on any non-STOP completion.
        return true;
    }

    /**
     * Renders a candidate's {@code safetyRatings} array as a single-line diagnostic
     * string of the form {@code "[CATEGORY:PROBABILITY(blocked),...]"}. Categories or
     * probabilities that are missing are rendered as {@code UNKNOWN}; the
     * {@code (blocked)} suffix appears only when the rating's {@code blocked} flag
     * is {@code true}.
     *
     * @param candidate the candidate JSON node (may be {@code null}); also accepts a
     *            {@code promptFeedback} node, since both expose a {@code safetyRatings}
     *            array with the same shape.
     * @return the formatted ratings string, or {@code null} when no {@code safetyRatings}
     *         array is present.
     */
    private static String stringifySafetyRatings(final JsonNode candidate) {
        if (candidate == null || !candidate.has("safetyRatings")) {
            return null;
        }
        final JsonNode ratings = candidate.get("safetyRatings");
        if (!ratings.isArray()) {
            return null;
        }
        final StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < ratings.size(); i++) {
            final JsonNode r = ratings.get(i);
            if (i > 0) {
                sb.append(",");
            }
            sb.append(r.path("category").asText("UNKNOWN")).append(":").append(r.path("probability").asText("UNKNOWN"));
            if (r.has("blocked") && r.get("blocked").asBoolean(false)) {
                sb.append("(blocked)");
            }
        }
        sb.append("]");
        return sb.toString();
    }

    /**
     * Detects whether a model id refers to the Gemini 3 generation, which uses
     * {@code thinkingLevel} (LOW/MEDIUM/HIGH) instead of the integer
     * {@code thinkingBudget} accepted by Gemini 2.x.
     *
     * @param model the model id, for example {@code "gemini-3-flash"} or {@code "gemini-3.1-pro"}.
     * @return {@code true} when the id starts with {@code "gemini-3"}; {@code false} when {@code null} or any other generation.
     */
    static boolean isGemini3(final String model) {
        if (model == null) {
            return false;
        }
        return model.startsWith("gemini-3-") || model.startsWith("gemini-3.") || "gemini-3".equals(model);
    }

    /**
     * Translates a Gemini 2.x {@code thinkingBudget} (integer token allowance) into the
     * Gemini 3.x {@code thinkingLevel} bucket. Mapping: {@code <=0 -> LOW}, {@code <=4096 -> MEDIUM},
     * {@code >4096 -> HIGH}.
     *
     * @param budget the requested thinking budget in tokens.
     * @return one of {@code "LOW"}, {@code "MEDIUM"}, {@code "HIGH"}.
     */
    static String budgetToThinkingLevel(final int budget) {
        if (budget <= 0) {
            return "LOW";
        }
        if (budget <= 4096) {
            return "MEDIUM";
        }
        return "HIGH";
    }

    /**
     * Returns the maximum number of attempts (initial + retries) for a single HTTP call.
     * Configured via {@code rag.llm.gemini.retry.max} (default {@code 10}).
     *
     * @return the maximum number of attempts.
     */
    protected int getRetryMaxAttempts() {
        return getConfigInt("retry.max", 10);
    }

    /**
     * Returns the base delay in milliseconds for exponential backoff between retries.
     * Configured via {@code rag.llm.gemini.retry.base.delay.ms} (default {@code 2000}).
     *
     * @return the base retry delay in milliseconds.
     */
    protected long getRetryBaseDelayMs() {
        return Long.parseLong(ComponentUtil.getFessConfig().getOrDefault(getConfigPrefix() + ".retry.base.delay.ms", "2000"));
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
     * Executes {@code call} with retry on {@link RetryableHttpException}. {@link IOException},
     * {@link ParseException}, and {@link LlmException} (RuntimeException, NOT caught here) are
     * all propagated immediately without retry. Backoff is exponential
     * ({@code base * 2^(attempt-1)}) with ±20% jitter. Sleep duration honors
     * {@link #getRetryBaseDelayMs()} and the cap is {@link #getRetryMaxAttempts()}.
     *
     * @param operation the operation label used in log messages (e.g. {@code "chat"}).
     * @param call the HTTP call body.
     * @param <T> the call result type.
     * @return the call result on success.
     * @throws IOException if the call throws a non-retryable {@link IOException} or the retry
     *             budget is exhausted.
     * @throws ParseException if the call throws {@link ParseException}.
     */
    private <T> T executeWithRetry(final String operation, final HttpCall<T> call) throws IOException, ParseException {
        final int maxAttempts = Math.max(1, getRetryMaxAttempts());
        final long baseDelay = Math.max(0L, getRetryBaseDelayMs());
        IOException lastIo = null;
        ParseException lastParse = null;
        for (int attempt = 1; attempt <= maxAttempts; attempt++) {
            try {
                return call.call();
            } catch (final RetryableHttpException e) {
                if (attempt == maxAttempts) {
                    logger.warn("[LLM:GEMINI] {} retry exhausted. attempts={}, lastStatus={}", operation, attempt, e.statusCode);
                    throw new IOException("Gemini API retryable error: " + e.statusCode + " " + e.reason, e);
                }
                final long jitter = (long) (baseDelay * 0.2 * ThreadLocalRandom.current().nextDouble(-1.0, 1.0)); // ±20%
                final long delay = (long) (baseDelay * Math.pow(2, attempt - 1)) + jitter;
                logger.info("[LLM:GEMINI] {} retrying. attempt={}/{}, status={}, sleepMs={}", operation, attempt, maxAttempts, e.statusCode,
                        Math.max(0, delay));
                try {
                    Thread.sleep(Math.max(0, delay));
                } catch (final InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    throw new IOException("Retry interrupted", ie);
                }
            } catch (final IOException e) {
                lastIo = e;
                break;
            } catch (final ParseException e) {
                lastParse = e;
                break;
            }
        }
        if (lastIo != null) {
            throw lastIo;
        }
        throw lastParse;
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
     * {@code 503} (UNAVAILABLE), and {@code 504} (DEADLINE_EXCEEDED).
     *
     * @param statusCode the HTTP status code.
     * @return {@code true} when the status is retryable.
     */
    static boolean isRetryableStatus(final int statusCode) {
        return statusCode == 429 || statusCode == 500 || statusCode == 503 || statusCode == 504;
    }

}
