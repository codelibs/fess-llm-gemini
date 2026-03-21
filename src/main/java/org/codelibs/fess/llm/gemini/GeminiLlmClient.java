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
            final String url = apiUrl + "/models?key=" + apiKey;
            final HttpGet request = new HttpGet(url);
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
            httpRequest.setEntity(new StringEntity(json, ContentType.APPLICATION_JSON));

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
                if (jsonNode.has("usageMetadata")) {
                    final JsonNode usage = jsonNode.get("usageMetadata");
                    if (usage.has("promptTokenCount")) {
                        chatResponse.setPromptTokens(usage.get("promptTokenCount").asInt());
                    }
                    if (usage.has("candidatesTokenCount")) {
                        chatResponse.setCompletionTokens(usage.get("candidatesTokenCount").asInt());
                    }
                    if (usage.has("totalTokenCount")) {
                        chatResponse.setTotalTokens(usage.get("totalTokenCount").asInt());
                    }
                }

                logger.info(
                        "[LLM:GEMINI] Chat response received. model={}, promptTokens={}, completionTokens={}, totalTokens={}, contentLength={}, elapsedTime={}ms",
                        chatResponse.getModel(), chatResponse.getPromptTokens(), chatResponse.getCompletionTokens(),
                        chatResponse.getTotalTokens(), chatResponse.getContent() != null ? chatResponse.getContent().length() : 0,
                        System.currentTimeMillis() - startTime);

                return chatResponse;
            }
        } catch (final LlmException e) {
            throw e;
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
            httpRequest.setEntity(new StringEntity(json, ContentType.APPLICATION_JSON));

            try (var response = getHttpClient().execute(httpRequest)) {
                final int statusCode = response.getCode();
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
                    throw new LlmException("Gemini API error: " + statusCode + " " + response.getReasonPhrase(),
                            resolveErrorCode(statusCode));
                }

                if (response.getEntity() == null) {
                    logger.warn("[LLM:GEMINI] Empty response from Gemini streaming API. url={}", maskApiKeyInUrl(url));
                    throw new LlmException("Empty response from Gemini");
                }

                int chunkCount = 0;
                long firstChunkTime = 0;
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

                                            boolean done = false;
                                            if (jsonNode.has("candidates") && jsonNode.get("candidates").isArray()
                                                    && jsonNode.get("candidates").size() > 0) {
                                                final JsonNode firstCandidate = jsonNode.get("candidates").get(0);
                                                if (firstCandidate.has("finishReason") && !firstCandidate.get("finishReason").isNull()
                                                        && !"null".equals(firstCandidate.get("finishReason").asText())) {
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

                logger.info("[LLM:GEMINI] Stream completed. chunkCount={}, firstChunkMs={}, elapsedTime={}ms", chunkCount, firstChunkTime,
                        System.currentTimeMillis() - startTime);
            }
        } catch (final LlmException e) {
            callback.onError(e);
            throw e;
        } catch (final IOException e) {
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
        final String apiKey = getApiKey();
        final String action = stream ? "streamGenerateContent" : "generateContent";
        return apiUrl + "/models/" + model + ":" + action + "?key=" + apiKey;
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
            thinkingConfig.put("thinkingBudget", request.getThinkingBudget());
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
        return ComponentUtil.getFessConfig().getOrDefault("rag.llm.gemini.model", "gemini-3-flash-preview");
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
     * Applies default generation parameters for the Gemini API free tier.
     * Only sets defaults when user has not configured the parameter.
     *
     * @param request the LLM chat request
     * @param promptType the prompt type (e.g. "intent", "evaluation", "answer")
     */
    protected void applyDefaultParams(final LlmChatRequest request, final String promptType) {
        switch (promptType) {
        case "intent":
        case "evaluation":
            if (request.getTemperature() == null) {
                request.setTemperature(0.1);
            }
            if (request.getMaxTokens() == null) {
                request.setMaxTokens(256);
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
                request.setMaxTokens(512);
            }
            break;
        case "docnotfound":
            if (request.getTemperature() == null) {
                request.setTemperature(0.7);
            }
            if (request.getMaxTokens() == null) {
                request.setMaxTokens(256);
            }
            break;
        case "direct":
        case "faq":
            if (request.getTemperature() == null) {
                request.setTemperature(0.7);
            }
            if (request.getMaxTokens() == null) {
                request.setMaxTokens(1024);
            }
            break;
        case "answer":
            if (request.getTemperature() == null) {
                request.setTemperature(0.5);
            }
            if (request.getMaxTokens() == null) {
                request.setMaxTokens(2048);
            }
            break;
        case "summary":
            if (request.getTemperature() == null) {
                request.setTemperature(0.3);
            }
            if (request.getMaxTokens() == null) {
                request.setMaxTokens(2048);
            }
            break;
        case "queryregeneration":
            if (request.getTemperature() == null) {
                request.setTemperature(0.3);
            }
            if (request.getMaxTokens() == null) {
                request.setMaxTokens(256);
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

}
