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

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.core.LogEvent;
import org.apache.logging.log4j.core.appender.AbstractAppender;
import org.apache.logging.log4j.core.config.Property;
import org.codelibs.fess.llm.LlmChatRequest;
import org.codelibs.fess.llm.LlmChatResponse;
import org.codelibs.fess.llm.LlmException;
import org.codelibs.fess.llm.LlmMessage;
import org.codelibs.fess.llm.LlmStreamCallback;
import org.codelibs.fess.llm.gemini.GeminiLlmClient.StreamSummary;
import org.codelibs.fess.unit.UnitFessTestCase;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInfo;

import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import okhttp3.mockwebserver.RecordedRequest;

public class GeminiLlmClientTest extends UnitFessTestCase {

    private TestableGeminiLlmClient client;
    private MockWebServer mockServer;

    @Override
    protected void setUp(TestInfo testInfo) throws Exception {
        super.setUp(testInfo);
        client = new TestableGeminiLlmClient();
        mockServer = new MockWebServer();
        mockServer.start();
    }

    @Override
    protected void tearDown(TestInfo testInfo) throws Exception {
        if (client != null) {
            client.destroy();
        }
        if (mockServer != null) {
            mockServer.shutdown();
        }
        super.tearDown(testInfo);
    }

    @Test
    public void test_getName() {
        assertEquals("gemini", client.getName());
    }

    @Test
    public void test_isAvailable_noApiKey() {
        client.setTestApiKey("");
        client.setTestApiUrl("https://generativelanguage.googleapis.com/v1beta");
        assertFalse(client.isAvailable());
    }

    @Test
    public void test_isAvailable_nullApiKey() {
        client.setTestApiKey(null);
        client.setTestApiUrl("https://generativelanguage.googleapis.com/v1beta");
        assertFalse(client.isAvailable());
    }

    @Test
    public void test_isAvailable_noApiUrl() {
        client.setTestApiKey("test-api-key");
        client.setTestApiUrl("");
        assertFalse(client.isAvailable());
    }

    @Test
    public void test_isAvailable_nullApiUrl() {
        client.setTestApiKey("test-api-key");
        client.setTestApiUrl(null);
        assertFalse(client.isAvailable());
    }

    @Test
    public void test_isAvailable_valid() throws IOException {
        // Mock the models endpoint for availability check
        mockServer.enqueue(new MockResponse().setBody("{\"models\":[]}").addHeader("Content-Type", "application/json"));
        setupClientForMockServer();
        assertTrue(client.isAvailable());
    }

    @Test
    public void test_convertMessage_user() {
        final LlmMessage message = LlmMessage.user("Hello, how are you?");
        final Map<String, Object> result = client.convertMessage(message);

        assertEquals("user", result.get("role"));
        @SuppressWarnings("unchecked")
        final List<Map<String, String>> parts = (List<Map<String, String>>) result.get("parts");
        assertEquals(1, parts.size());
        assertEquals("Hello, how are you?", parts.get(0).get("text"));
    }

    @Test
    public void test_convertMessage_assistant() {
        final LlmMessage message = LlmMessage.assistant("I'm doing well, thank you!");
        final Map<String, Object> result = client.convertMessage(message);

        // Assistant should be converted to "model" for Gemini
        assertEquals("model", result.get("role"));
        @SuppressWarnings("unchecked")
        final List<Map<String, String>> parts = (List<Map<String, String>>) result.get("parts");
        assertEquals(1, parts.size());
        assertEquals("I'm doing well, thank you!", parts.get(0).get("text"));
    }

    @Test
    public void test_buildRequestBody_defaultValues() {
        client.setTestModel("gemini-2.5-flash");

        final LlmChatRequest request = new LlmChatRequest().setTemperature(0.7).setMaxTokens(4096).addUserMessage("Hello");

        final Map<String, Object> body = client.buildRequestBody(request);

        // Check contents
        @SuppressWarnings("unchecked")
        final List<Map<String, Object>> contents = (List<Map<String, Object>>) body.get("contents");
        assertEquals(1, contents.size());
        assertEquals("user", contents.get(0).get("role"));

        @SuppressWarnings("unchecked")
        final List<Map<String, String>> parts = (List<Map<String, String>>) contents.get(0).get("parts");
        assertEquals("Hello", parts.get(0).get("text"));

        // Check generation config
        @SuppressWarnings("unchecked")
        final Map<String, Object> generationConfig = (Map<String, Object>) body.get("generationConfig");
        assertEquals(0.7, generationConfig.get("temperature"));
        assertEquals(4096, generationConfig.get("maxOutputTokens"));

        // No system instruction for this request
        assertNull(body.get("systemInstruction"));
    }

    @Test
    public void test_buildRequestBody_withSystemMessage() {
        client.setTestModel("gemini-2.5-flash");
        client.setTestTemperature(0.7);
        client.setTestMaxTokens(4096);

        final LlmChatRequest request = new LlmChatRequest().addSystemMessage("You are a helpful assistant.").addUserMessage("Hello");

        final Map<String, Object> body = client.buildRequestBody(request);

        // Check system instruction
        @SuppressWarnings("unchecked")
        final Map<String, Object> systemInstruction = (Map<String, Object>) body.get("systemInstruction");
        assertNotNull(systemInstruction);

        @SuppressWarnings("unchecked")
        final List<Map<String, String>> systemParts = (List<Map<String, String>>) systemInstruction.get("parts");
        assertEquals(1, systemParts.size());
        assertEquals("You are a helpful assistant.", systemParts.get(0).get("text"));

        // Check contents (should not include system message)
        @SuppressWarnings("unchecked")
        final List<Map<String, Object>> contents = (List<Map<String, Object>>) body.get("contents");
        assertEquals(1, contents.size());
        assertEquals("user", contents.get(0).get("role"));
    }

    @Test
    public void test_buildRequestBody_withMultipleSystemMessages() {
        client.setTestModel("gemini-2.5-flash");
        client.setTestTemperature(0.7);
        client.setTestMaxTokens(4096);

        final LlmChatRequest request = new LlmChatRequest().addSystemMessage("You are a helpful assistant.")
                .addSystemMessage("Always be polite.")
                .addUserMessage("Hello");

        final Map<String, Object> body = client.buildRequestBody(request);

        // Check system instruction (should be combined)
        @SuppressWarnings("unchecked")
        final Map<String, Object> systemInstruction = (Map<String, Object>) body.get("systemInstruction");
        assertNotNull(systemInstruction);

        @SuppressWarnings("unchecked")
        final List<Map<String, String>> systemParts = (List<Map<String, String>>) systemInstruction.get("parts");
        assertEquals(1, systemParts.size());
        assertEquals("You are a helpful assistant.\nAlways be polite.", systemParts.get(0).get("text"));
    }

    @Test
    public void test_buildRequestBody_withRequestTemperature() {
        client.setTestModel("gemini-2.5-flash");
        client.setTestTemperature(0.7);
        client.setTestMaxTokens(4096);

        final LlmChatRequest request = new LlmChatRequest().setTemperature(0.5).addUserMessage("Hello");

        final Map<String, Object> body = client.buildRequestBody(request);

        @SuppressWarnings("unchecked")
        final Map<String, Object> generationConfig = (Map<String, Object>) body.get("generationConfig");
        assertEquals(0.5, generationConfig.get("temperature"));
    }

    @Test
    public void test_buildRequestBody_withRequestMaxTokens() {
        client.setTestModel("gemini-2.5-flash");
        client.setTestTemperature(0.7);
        client.setTestMaxTokens(4096);

        final LlmChatRequest request = new LlmChatRequest().setMaxTokens(1000).addUserMessage("Hello");

        final Map<String, Object> body = client.buildRequestBody(request);

        @SuppressWarnings("unchecked")
        final Map<String, Object> generationConfig = (Map<String, Object>) body.get("generationConfig");
        assertEquals(1000, generationConfig.get("maxOutputTokens"));
    }

    @Test
    public void test_buildRequestBody_multipleMessages() {
        client.setTestModel("gemini-2.5-flash");
        client.setTestTemperature(0.7);
        client.setTestMaxTokens(4096);

        final LlmChatRequest request = new LlmChatRequest().addSystemMessage("You are a helpful assistant.")
                .addUserMessage("What is the weather?")
                .addAssistantMessage("I cannot access weather information.")
                .addUserMessage("OK");

        final Map<String, Object> body = client.buildRequestBody(request);

        @SuppressWarnings("unchecked")
        final List<Map<String, Object>> contents = (List<Map<String, Object>>) body.get("contents");
        assertEquals(3, contents.size());

        // First message: user
        assertEquals("user", contents.get(0).get("role"));
        @SuppressWarnings("unchecked")
        final List<Map<String, String>> parts0 = (List<Map<String, String>>) contents.get(0).get("parts");
        assertEquals("What is the weather?", parts0.get(0).get("text"));

        // Second message: model (converted from assistant)
        assertEquals("model", contents.get(1).get("role"));
        @SuppressWarnings("unchecked")
        final List<Map<String, String>> parts1 = (List<Map<String, String>>) contents.get(1).get("parts");
        assertEquals("I cannot access weather information.", parts1.get(0).get("text"));

        // Third message: user
        assertEquals("user", contents.get(2).get("role"));
        @SuppressWarnings("unchecked")
        final List<Map<String, String>> parts2 = (List<Map<String, String>>) contents.get(2).get("parts");
        assertEquals("OK", parts2.get(0).get("text"));
    }

    @Test
    public void test_buildApiUrl_nonStreaming() {
        client.setTestApiUrl("https://generativelanguage.googleapis.com/v1beta");
        client.setTestApiKey("test-key");

        final String url = client.buildApiUrl("gemini-2.5-flash", false);

        assertEquals("https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=test-key", url);
    }

    @Test
    public void test_buildApiUrl_streaming() {
        client.setTestApiUrl("https://generativelanguage.googleapis.com/v1beta");
        client.setTestApiKey("test-key");

        final String url = client.buildApiUrl("gemini-2.5-flash", true);

        assertEquals("https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent?key=test-key&alt=sse",
                url);
    }

    @Test
    public void test_getModelName_fromRequest() {
        client.setTestModel("gemini-2.5-flash");

        final LlmChatRequest request = new LlmChatRequest().setModel("gemini-1.5-pro").addUserMessage("Hello");

        assertEquals("gemini-1.5-pro", client.getModelName(request));
    }

    @Test
    public void test_getModelName_fromConfig() {
        client.setTestModel("gemini-2.5-flash");

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");

        assertEquals("gemini-2.5-flash", client.getModelName(request));
    }

    @Test
    public void test_getModelName_blankModelUsesDefault() {
        client.setTestModel("gemini-2.5-flash");

        final LlmChatRequest request = new LlmChatRequest().setModel("").addUserMessage("Hello");

        assertEquals("gemini-2.5-flash", client.getModelName(request));
    }

    @Test
    public void test_getModelName_nullModelUsesDefault() {
        client.setTestModel("gemini-2.5-flash");

        final LlmChatRequest request = new LlmChatRequest().setModel(null).addUserMessage("Hello");

        assertEquals("gemini-2.5-flash", client.getModelName(request));
    }

    @Test
    public void test_init() {
        client.setTestTimeout(30000);
        client.init();
        assertNotNull(client.getHttpClient());
    }

    @Test
    public void test_getHttpClient_lazyInitialization() {
        client.setTestTimeout(60000);
        // First call should initialize the client
        assertNotNull(client.getHttpClient());
        // Second call should return the same client
        assertNotNull(client.getHttpClient());
    }

    @Test
    public void test_buildRequestBody_withThinkingBudget() {
        client.setTestModel("gemini-3-flash-preview");
        client.setTestTemperature(0.7);
        client.setTestMaxTokens(4096);

        final LlmChatRequest request = new LlmChatRequest().setThinkingBudget(1024).addUserMessage("Hello");

        final Map<String, Object> body = client.buildRequestBody(request);

        @SuppressWarnings("unchecked")
        final Map<String, Object> generationConfig = (Map<String, Object>) body.get("generationConfig");
        assertNotNull(generationConfig);

        @SuppressWarnings("unchecked")
        final Map<String, Object> thinkingConfig = (Map<String, Object>) generationConfig.get("thinkingConfig");
        assertNotNull(thinkingConfig);
        assertEquals(1024, thinkingConfig.get("thinkingBudget"));
    }

    @Test
    public void test_buildRequestBody_withThinkingBudgetZero() {
        client.setTestModel("gemini-3-flash-preview");
        client.setTestTemperature(0.3);
        client.setTestMaxTokens(500);

        final LlmChatRequest request = new LlmChatRequest().setThinkingBudget(0).addUserMessage("Detect intent");

        final Map<String, Object> body = client.buildRequestBody(request);

        @SuppressWarnings("unchecked")
        final Map<String, Object> generationConfig = (Map<String, Object>) body.get("generationConfig");
        assertNotNull(generationConfig);

        @SuppressWarnings("unchecked")
        final Map<String, Object> thinkingConfig = (Map<String, Object>) generationConfig.get("thinkingConfig");
        assertNotNull(thinkingConfig);
        assertEquals(0, thinkingConfig.get("thinkingBudget"));
    }

    @Test
    public void test_buildRequestBody_withoutThinkingBudget() {
        client.setTestModel("gemini-2.5-flash");

        final LlmChatRequest request = new LlmChatRequest().setTemperature(0.7).setMaxTokens(4096).addUserMessage("Hello");

        final Map<String, Object> body = client.buildRequestBody(request);

        @SuppressWarnings("unchecked")
        final Map<String, Object> generationConfig = (Map<String, Object>) body.get("generationConfig");
        assertNotNull(generationConfig);
        assertNull(generationConfig.get("thinkingConfig"));
    }

    @Test
    public void test_buildRequestBody_noSystemMessage() {
        client.setTestModel("gemini-2.5-flash");
        client.setTestTemperature(0.7);
        client.setTestMaxTokens(4096);

        final LlmChatRequest request =
                new LlmChatRequest().addUserMessage("Hello").addAssistantMessage("Hi there!").addUserMessage("How are you?");

        final Map<String, Object> body = client.buildRequestBody(request);

        // No system instruction
        assertNull(body.get("systemInstruction"));

        // Check contents
        @SuppressWarnings("unchecked")
        final List<Map<String, Object>> contents = (List<Map<String, Object>>) body.get("contents");
        assertEquals(3, contents.size());
    }

    @Test
    public void test_convertMessage_preservesUserRole() {
        final LlmMessage message = LlmMessage.user("Test message");
        final Map<String, Object> result = client.convertMessage(message);

        assertEquals("user", result.get("role"));
    }

    @Test
    public void test_convertMessage_convertsAssistantToModel() {
        final LlmMessage message = LlmMessage.assistant("Test response");
        final Map<String, Object> result = client.convertMessage(message);

        // "assistant" should be converted to "model" for Gemini API
        assertEquals("model", result.get("role"));
    }

    // ========== chat() method tests ==========

    @Test
    public void test_chat_success() throws IOException {
        final String responseJson = """
                {
                    "candidates": [{
                        "content": {
                            "parts": [{
                                "text": "Hello! How can I help you today?"
                            }],
                            "role": "model"
                        },
                        "finishReason": "STOP"
                    }],
                    "usageMetadata": {
                        "promptTokenCount": 10,
                        "candidatesTokenCount": 20,
                        "totalTokenCount": 30
                    },
                    "modelVersion": "gemini-2.0-flash"
                }
                """;

        mockServer.enqueue(new MockResponse().setBody(responseJson).addHeader("Content-Type", "application/json"));

        setupClientForMockServer();

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");
        final LlmChatResponse response = client.chat(request);

        assertEquals("Hello! How can I help you today?", response.getContent());
        assertEquals("STOP", response.getFinishReason());
        assertEquals("gemini-2.0-flash", response.getModel());
        assertEquals(10, response.getPromptTokens());
        assertEquals(20, response.getCompletionTokens());
        assertEquals(30, response.getTotalTokens());
    }

    @Test
    public void test_chat_successWithMinimalResponse() throws IOException {
        final String responseJson = """
                {
                    "candidates": [{
                        "content": {
                            "parts": [{
                                "text": "Response text"
                            }]
                        }
                    }]
                }
                """;

        mockServer.enqueue(new MockResponse().setBody(responseJson).addHeader("Content-Type", "application/json"));

        setupClientForMockServer();

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");
        final LlmChatResponse response = client.chat(request);

        assertEquals("Response text", response.getContent());
    }

    @Test
    public void test_chat_successWithoutModelVersion() throws IOException {
        final String responseJson = """
                {
                    "candidates": [{
                        "content": {
                            "parts": [{
                                "text": "Response text"
                            }]
                        }
                    }]
                }
                """;

        mockServer.enqueue(new MockResponse().setBody(responseJson).addHeader("Content-Type", "application/json"));

        setupClientForMockServer();

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");
        final LlmChatResponse response = client.chat(request);

        // Should use the configured model name when modelVersion is not in response
        assertEquals("gemini-2.0-flash", response.getModel());
    }

    @Test
    public void test_chat_errorResponse_withBody() throws IOException {
        final String errorJson = """
                {
                    "error": {
                        "code": 401,
                        "message": "API key not valid. Please pass a valid API key.",
                        "status": "UNAUTHENTICATED"
                    }
                }
                """;

        mockServer.enqueue(new MockResponse().setResponseCode(401).setBody(errorJson).addHeader("Content-Type", "application/json"));

        setupClientForMockServer();

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");

        try {
            client.chat(request);
            fail("Expected LlmException to be thrown");
        } catch (final LlmException error) {
            assertTrue(error.getMessage().contains("401"));
        }
    }

    @Test
    public void test_chat_errorResponse_quotaExceeded() throws IOException {
        final String errorJson = """
                {
                    "error": {
                        "code": 429,
                        "message": "Resource has been exhausted",
                        "status": "RESOURCE_EXHAUSTED"
                    }
                }
                """;

        mockServer.enqueue(new MockResponse().setResponseCode(429).setBody(errorJson).addHeader("Content-Type", "application/json"));

        setupClientForMockServer();

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");

        try {
            client.chat(request);
            fail("Expected LlmException to be thrown");
        } catch (final LlmException error) {
            assertTrue(error.getMessage().contains("429"));
        }
    }

    @Test
    public void test_chat_errorResponse_serverError() throws IOException {
        final String errorJson = """
                {
                    "error": {
                        "code": 500,
                        "message": "Internal error encountered",
                        "status": "INTERNAL"
                    }
                }
                """;

        mockServer.enqueue(new MockResponse().setResponseCode(500).setBody(errorJson).addHeader("Content-Type", "application/json"));

        setupClientForMockServer();

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");

        try {
            client.chat(request);
            fail("Expected LlmException to be thrown");
        } catch (final LlmException error) {
            assertTrue(error.getMessage().contains("500"));
        }
    }

    @Test
    public void test_chat_errorResponse_emptyBody() throws IOException {
        mockServer.enqueue(new MockResponse().setResponseCode(503).setBody("").addHeader("Content-Type", "application/json"));

        setupClientForMockServer();

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");

        try {
            client.chat(request);
            fail("Expected LlmException to be thrown");
        } catch (final LlmException error) {
            assertTrue(error.getMessage().contains("503"));
        }
    }

    @Test
    public void test_chat_emptyCandidates() throws IOException {
        final String responseJson = """
                {
                    "candidates": []
                }
                """;

        mockServer.enqueue(new MockResponse().setBody(responseJson).addHeader("Content-Type", "application/json"));

        setupClientForMockServer();

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");
        final LlmChatResponse response = client.chat(request);

        assertNull(response.getContent());
    }

    @Test
    public void test_chat_nullFinishReason() throws IOException {
        final String responseJson = """
                {
                    "candidates": [{
                        "content": {
                            "parts": [{
                                "text": "Test"
                            }]
                        },
                        "finishReason": null
                    }]
                }
                """;

        mockServer.enqueue(new MockResponse().setBody(responseJson).addHeader("Content-Type", "application/json"));

        setupClientForMockServer();

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");
        final LlmChatResponse response = client.chat(request);

        assertEquals("Test", response.getContent());
        assertNull(response.getFinishReason());
    }

    @Test
    public void test_chat_partialUsageMetadata() throws IOException {
        final String responseJson = """
                {
                    "candidates": [{
                        "content": {
                            "parts": [{
                                "text": "Test"
                            }]
                        }
                    }],
                    "usageMetadata": {
                        "promptTokenCount": 5
                    }
                }
                """;

        mockServer.enqueue(new MockResponse().setBody(responseJson).addHeader("Content-Type", "application/json"));

        setupClientForMockServer();

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");
        final LlmChatResponse response = client.chat(request);

        assertEquals(5, response.getPromptTokens());
        assertNull(response.getCompletionTokens());
        assertNull(response.getTotalTokens());
    }

    @Test
    public void test_chat_emptyParts() throws IOException {
        final String responseJson = """
                {
                    "candidates": [{
                        "content": {
                            "parts": []
                        }
                    }]
                }
                """;

        mockServer.enqueue(new MockResponse().setBody(responseJson).addHeader("Content-Type", "application/json"));

        setupClientForMockServer();

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");
        final LlmChatResponse response = client.chat(request);

        assertNull(response.getContent());
    }

    // ========== streamChat() method tests ==========

    @Test
    public void test_streamChat_success() throws IOException {
        final String streamResponse = """
                [
                {"candidates":[{"content":{"parts":[{"text":"Hello"}],"role":"model"}}]},
                {"candidates":[{"content":{"parts":[{"text":" World"}],"role":"model"}}]},
                {"candidates":[{"content":{"parts":[{"text":"!"}],"role":"model"},"finishReason":"STOP"}]}
                ]
                """;

        mockServer.enqueue(new MockResponse().setBody(streamResponse).addHeader("Content-Type", "application/json"));

        setupClientForMockServer();

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");
        final List<String> chunks = new ArrayList<>();
        final AtomicBoolean doneReceived = new AtomicBoolean(false);

        client.streamChat(request, new LlmStreamCallback() {
            @Override
            public void onChunk(final String content, final boolean done) {
                chunks.add(content);
                if (done) {
                    doneReceived.set(true);
                }
            }

            @Override
            public void onError(final Throwable error) {
                fail("Unexpected error: " + error.getMessage());
            }
        });

        assertEquals(3, chunks.size());
        assertEquals("Hello", chunks.get(0));
        assertEquals(" World", chunks.get(1));
        assertEquals("!", chunks.get(2));
        assertTrue(doneReceived.get());
    }

    @Test
    public void test_streamChat_multipleChunks() throws IOException {
        final String streamResponse = """
                [
                {"candidates":[{"content":{"parts":[{"text":"A"}]}}]},
                {"candidates":[{"content":{"parts":[{"text":"B"}]}}]},
                {"candidates":[{"content":{"parts":[{"text":"C"}]}}]},
                {"candidates":[{"content":{"parts":[{"text":"D"}]},"finishReason":"STOP"}]}
                ]
                """;

        mockServer.enqueue(new MockResponse().setBody(streamResponse).addHeader("Content-Type", "application/json"));

        setupClientForMockServer();

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");
        final StringBuilder content = new StringBuilder();
        final AtomicInteger chunkCount = new AtomicInteger(0);

        client.streamChat(request, new LlmStreamCallback() {
            @Override
            public void onChunk(final String chunk, final boolean done) {
                content.append(chunk);
                chunkCount.incrementAndGet();
            }

            @Override
            public void onError(final Throwable error) {
                fail("Unexpected error: " + error.getMessage());
            }
        });

        assertEquals("ABCD", content.toString());
        assertEquals(4, chunkCount.get());
    }

    @Test
    public void test_streamChat_errorResponse_withBody() throws IOException {
        final String errorJson = """
                {
                    "error": {
                        "code": 429,
                        "message": "Resource has been exhausted",
                        "status": "RESOURCE_EXHAUSTED"
                    }
                }
                """;

        mockServer.enqueue(new MockResponse().setResponseCode(429).setBody(errorJson).addHeader("Content-Type", "application/json"));

        setupClientForMockServer();

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");
        final AtomicBoolean errorReceived = new AtomicBoolean(false);

        try {
            client.streamChat(request, new LlmStreamCallback() {
                @Override
                public void onChunk(final String content, final boolean done) {
                    fail("Should not receive chunks on error");
                }

                @Override
                public void onError(final Throwable error) {
                    errorReceived.set(true);
                }
            });
            fail("Expected LlmException to be thrown");
        } catch (final LlmException error) {
            assertTrue(error.getMessage().contains("429"));
            assertTrue(errorReceived.get());
        }
    }

    @Test
    public void test_streamChat_errorResponse_serverError() throws IOException {
        final String errorJson = """
                {
                    "error": {
                        "code": 500,
                        "message": "Internal error encountered"
                    }
                }
                """;

        mockServer.enqueue(new MockResponse().setResponseCode(500).setBody(errorJson).addHeader("Content-Type", "application/json"));

        setupClientForMockServer();

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");
        final AtomicBoolean errorReceived = new AtomicBoolean(false);

        try {
            client.streamChat(request, new LlmStreamCallback() {
                @Override
                public void onChunk(final String content, final boolean done) {
                    fail("Should not receive chunks on error");
                }

                @Override
                public void onError(final Throwable error) {
                    errorReceived.set(true);
                }
            });
            fail("Expected LlmException to be thrown");
        } catch (final LlmException error) {
            assertTrue(error.getMessage().contains("500"));
            assertTrue(errorReceived.get());
        }
    }

    @Test
    public void test_streamChat_emptyBody() throws IOException {
        // Empty body with MockWebServer doesn't result in null body
        // Just verify that no chunks are received
        mockServer.enqueue(new MockResponse().setResponseCode(200).setBody("").addHeader("Content-Type", "application/json"));

        setupClientForMockServer();

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");
        final List<String> chunks = new ArrayList<>();

        client.streamChat(request, new LlmStreamCallback() {
            @Override
            public void onChunk(final String content, final boolean done) {
                chunks.add(content);
            }

            @Override
            public void onError(final Throwable error) {
                fail("Unexpected error: " + error.getMessage());
            }
        });

        // No chunks should be received for empty body
        assertEquals(0, chunks.size());
    }

    @Test
    public void test_streamChat_finishReasonStop() throws IOException {
        final String streamResponse = """
                [
                {"candidates":[{"content":{"parts":[{"text":"Test"}]},"finishReason":"STOP"}]}
                ]
                """;

        mockServer.enqueue(new MockResponse().setBody(streamResponse).addHeader("Content-Type", "application/json"));

        setupClientForMockServer();

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");
        final AtomicBoolean doneReceived = new AtomicBoolean(false);
        final List<String> chunks = new ArrayList<>();

        client.streamChat(request, new LlmStreamCallback() {
            @Override
            public void onChunk(final String content, final boolean done) {
                chunks.add(content);
                if (done) {
                    doneReceived.set(true);
                }
            }

            @Override
            public void onError(final Throwable error) {
                fail("Unexpected error: " + error.getMessage());
            }
        });

        assertEquals(1, chunks.size());
        assertEquals("Test", chunks.get(0));
        assertTrue(doneReceived.get());
    }

    @Test
    public void test_streamChat_finishReasonMaxTokens() throws IOException {
        final String streamResponse = """
                [
                {"candidates":[{"content":{"parts":[{"text":"Truncated"}]},"finishReason":"MAX_TOKENS"}]}
                ]
                """;

        mockServer.enqueue(new MockResponse().setBody(streamResponse).addHeader("Content-Type", "application/json"));

        setupClientForMockServer();

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");
        final AtomicBoolean doneReceived = new AtomicBoolean(false);

        client.streamChat(request, new LlmStreamCallback() {
            @Override
            public void onChunk(final String content, final boolean done) {
                if (done) {
                    doneReceived.set(true);
                }
            }

            @Override
            public void onError(final Throwable error) {
                fail("Unexpected error: " + error.getMessage());
            }
        });

        assertTrue(doneReceived.get());
    }

    @Test
    public void test_streamChat_logsFinishReasonAndUsage() throws IOException {
        final String streamResponse =
                """
                        [
                        {"candidates":[{"content":{"parts":[{"text":"Hello"}],"role":"model"}}]},
                        {"candidates":[{"content":{"parts":[{"text":" world"}],"role":"model"},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":12,"candidatesTokenCount":2,"totalTokenCount":14,"thoughtsTokenCount":0}}
                        ]
                        """;

        mockServer.enqueue(new MockResponse().setBody(streamResponse).addHeader("Content-Type", "text/event-stream"));
        setupClientForMockServer();

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hi");
        final List<String> chunks = new ArrayList<>();
        final AtomicReference<StreamSummary> summaryRef = new AtomicReference<>();

        client.setStreamSummaryConsumer(summaryRef::set);

        client.streamChat(request, new LlmStreamCallback() {
            @Override
            public void onChunk(final String content, final boolean done) {
                chunks.add(content);
            }

            @Override
            public void onError(final Throwable error) {
                fail("Unexpected: " + error.getMessage());
            }
        });

        assertEquals(2, chunks.size());
        final StreamSummary s = summaryRef.get();
        assertNotNull(s);
        assertEquals("STOP", s.finishReason);
        assertEquals(2, s.objectCount);
        assertEquals(2, s.chunkCount);
        assertEquals(Integer.valueOf(2), s.candidatesTokenCount);
        assertEquals(Integer.valueOf(0), s.thoughtsTokenCount);
    }

    @Test
    public void test_streamChat_sseFormatTwoEvents() throws IOException {
        final String streamResponse = "" + "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hel\"}],\"role\":\"model\"}}]}\n"
                + "\n"
                + "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"lo\"}],\"role\":\"model\"},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":3,\"candidatesTokenCount\":2,\"totalTokenCount\":5}}\n"
                + "\n";

        mockServer.enqueue(new MockResponse().setBody(streamResponse).addHeader("Content-Type", "text/event-stream"));
        setupClientForMockServer();

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hi");
        final List<String> chunks = new ArrayList<>();
        final AtomicBoolean doneReceived = new AtomicBoolean(false);
        client.streamChat(request, new LlmStreamCallback() {
            @Override
            public void onChunk(final String content, final boolean done) {
                chunks.add(content);
                if (done) {
                    doneReceived.set(true);
                }
            }

            @Override
            public void onError(final Throwable error) {
                fail("Unexpected: " + error.getMessage());
            }
        });

        assertEquals(2, chunks.size());
        assertEquals("Hel", chunks.get(0));
        assertEquals("lo", chunks.get(1));
        assertTrue(doneReceived.get());
    }

    @Test
    public void test_streamChat_sseTerminatorDoneIsIgnored() throws IOException {
        final String streamResponse =
                "" + "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"x\"}],\"role\":\"model\"},\"finishReason\":\"STOP\"}]}\n"
                        + "\n" + "data: [DONE]\n" + "\n";
        mockServer.enqueue(new MockResponse().setBody(streamResponse).addHeader("Content-Type", "text/event-stream"));
        setupClientForMockServer();

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hi");
        final List<String> chunks = new ArrayList<>();
        client.streamChat(request, new LlmStreamCallback() {
            @Override
            public void onChunk(final String content, final boolean done) {
                chunks.add(content);
            }

            @Override
            public void onError(final Throwable error) {
                fail("Unexpected: " + error.getMessage());
            }
        });

        assertEquals(1, chunks.size());
        assertEquals("x", chunks.get(0));
    }

    @Test
    public void test_streamChat_sseCommentLinesIgnored() throws IOException {
        // Some proxies send heartbeats as ":keepalive" SSE comments
        final String streamResponse = "" + ": ping\n"
                + "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"y\"}],\"role\":\"model\"},\"finishReason\":\"STOP\"}]}\n"
                + "\n";
        mockServer.enqueue(new MockResponse().setBody(streamResponse).addHeader("Content-Type", "text/event-stream"));
        setupClientForMockServer();

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hi");
        final List<String> chunks = new ArrayList<>();
        client.streamChat(request, new LlmStreamCallback() {
            @Override
            public void onChunk(final String content, final boolean done) {
                chunks.add(content);
            }

            @Override
            public void onError(final Throwable error) {
                fail("Unexpected: " + error.getMessage());
            }
        });
        assertEquals(1, chunks.size());
        assertEquals("y", chunks.get(0));
    }

    @Test
    public void test_streamChat_acceptsApplicationJsonArray() throws IOException {
        // Backwards-compat: when alt=sse is overridden by config or proxy, server may still serve an array
        final String streamResponse = """
                [
                {"candidates":[{"content":{"parts":[{"text":"A"}],"role":"model"},"finishReason":"STOP"}]}
                ]
                """;
        mockServer.enqueue(new MockResponse().setBody(streamResponse).addHeader("Content-Type", "application/json"));
        setupClientForMockServer();

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hi");
        final List<String> chunks = new ArrayList<>();
        client.streamChat(request, new LlmStreamCallback() {
            @Override
            public void onChunk(final String content, final boolean done) {
                chunks.add(content);
            }

            @Override
            public void onError(final Throwable error) {
                fail("Unexpected: " + error.getMessage());
            }
        });
        assertEquals(1, chunks.size());
        assertEquals("A", chunks.get(0));
    }

    @Test
    public void test_streamChat_malformedJson() throws IOException {
        final String streamResponse = """
                [
                {"candidates":[{"content":{"parts":[{"text":"Hello"}]}}]},
                {invalid json},
                {"candidates":[{"content":{"parts":[{"text":" World"}]},"finishReason":"STOP"}]}
                ]
                """;

        mockServer.enqueue(new MockResponse().setBody(streamResponse).addHeader("Content-Type", "application/json"));

        setupClientForMockServer();

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");
        final List<String> chunks = new ArrayList<>();

        client.streamChat(request, new LlmStreamCallback() {
            @Override
            public void onChunk(final String content, final boolean done) {
                if (!content.isEmpty()) {
                    chunks.add(content);
                }
            }

            @Override
            public void onError(final Throwable error) {
                // Malformed JSON is logged but doesn't stop streaming
            }
        });

        // Should still receive valid chunks
        assertTrue(chunks.size() >= 2);
        assertTrue(chunks.contains("Hello"));
        assertTrue(chunks.contains(" World"));
    }

    @Test
    public void test_streamChat_ignoresArrayBrackets() throws IOException {
        final String streamResponse = """
                [
                {"candidates":[{"content":{"parts":[{"text":"Test"}]},"finishReason":"STOP"}]}
                ]
                """;

        mockServer.enqueue(new MockResponse().setBody(streamResponse).addHeader("Content-Type", "application/json"));

        setupClientForMockServer();

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");
        final List<String> chunks = new ArrayList<>();

        client.streamChat(request, new LlmStreamCallback() {
            @Override
            public void onChunk(final String content, final boolean done) {
                chunks.add(content);
            }

            @Override
            public void onError(final Throwable error) {
                fail("Unexpected error: " + error.getMessage());
            }
        });

        assertEquals(1, chunks.size());
        assertEquals("Test", chunks.get(0));
    }

    @Test
    public void test_streamChat_handlesCommasInStreamFormat() throws IOException {
        // Gemini streaming format often has commas between JSON objects
        final String streamResponse = """
                [
                {"candidates":[{"content":{"parts":[{"text":"A"}]}}]}
                ,
                {"candidates":[{"content":{"parts":[{"text":"B"}]},"finishReason":"STOP"}]}
                ]
                """;

        mockServer.enqueue(new MockResponse().setBody(streamResponse).addHeader("Content-Type", "application/json"));

        setupClientForMockServer();

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");
        final StringBuilder content = new StringBuilder();

        client.streamChat(request, new LlmStreamCallback() {
            @Override
            public void onChunk(final String chunk, final boolean done) {
                content.append(chunk);
            }

            @Override
            public void onError(final Throwable error) {
                fail("Unexpected error: " + error.getMessage());
            }
        });

        assertEquals("AB", content.toString());
    }

    @Test
    public void test_streamChat_emptyCandidates() throws IOException {
        final String streamResponse = """
                [
                {"candidates":[]}
                ]
                """;

        mockServer.enqueue(new MockResponse().setBody(streamResponse).addHeader("Content-Type", "application/json"));

        setupClientForMockServer();

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");
        final List<String> chunks = new ArrayList<>();

        client.streamChat(request, new LlmStreamCallback() {
            @Override
            public void onChunk(final String content, final boolean done) {
                chunks.add(content);
            }

            @Override
            public void onError(final Throwable error) {
                fail("Unexpected error: " + error.getMessage());
            }
        });

        // No chunks should be added for empty candidates
        assertEquals(0, chunks.size());
    }

    // ========== destroy() tests ==========

    @Test
    public void test_destroy_closesHttpClient() {
        client.setTestTimeout(30000);
        client.init();
        assertNotNull(client.getHttpClient());
        client.destroy();
        // After destroy, calling getHttpClient() triggers re-init
        // Verify no exception is thrown during destroy
    }

    @Test
    public void test_destroy_beforeInit() {
        // destroy before init should not throw
        client.destroy();
    }

    // ========== Request format verification tests ==========

    @Test
    public void test_chat_verifyRequestFormat() throws Exception {
        final String responseJson = """
                {
                    "candidates": [{
                        "content": {
                            "parts": [{
                                "text": "Response"
                            }]
                        }
                    }]
                }
                """;

        mockServer.enqueue(new MockResponse().setBody(responseJson).addHeader("Content-Type", "application/json"));

        setupClientForMockServer();

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");
        client.chat(request);

        final RecordedRequest recorded = mockServer.takeRequest();
        assertEquals("POST", recorded.getMethod());
        assertTrue(recorded.getPath().contains("/models/gemini-2.0-flash:generateContent"));
        assertTrue(recorded.getPath().contains("key=test-key"));
        assertEquals("application/json; charset=UTF-8", recorded.getHeader("Content-Type"));

        // Verify body contains expected structure
        final String body = recorded.getBody().readUtf8();
        assertTrue(body.contains("\"contents\""));
        assertTrue(body.contains("Hello"));
    }

    @Test
    public void test_streamChat_verifyRequestFormat() throws Exception {
        final String streamResponse = """
                [
                {"candidates":[{"content":{"parts":[{"text":"Test"}]},"finishReason":"STOP"}]}
                ]
                """;

        mockServer.enqueue(new MockResponse().setBody(streamResponse).addHeader("Content-Type", "application/json"));

        setupClientForMockServer();

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");
        client.streamChat(request, new LlmStreamCallback() {
            @Override
            public void onChunk(final String content, final boolean done) {
            }

            @Override
            public void onError(final Throwable error) {
            }
        });

        final RecordedRequest recorded = mockServer.takeRequest();
        assertEquals("POST", recorded.getMethod());
        assertTrue(recorded.getPath().contains("/models/gemini-2.0-flash:streamGenerateContent"));
        assertTrue(recorded.getPath().contains("key=test-key"));
    }

    // ========== checkAvailabilityNow() tests ==========

    @Test
    public void test_checkAvailabilityNow_success() throws Exception {
        mockServer.enqueue(new MockResponse().setBody("{\"models\":[]}").addHeader("Content-Type", "application/json"));

        setupClientForMockServer();

        assertTrue(client.checkAvailabilityNow());

        final RecordedRequest recorded = mockServer.takeRequest();
        assertEquals("GET", recorded.getMethod());
        assertTrue(recorded.getPath().contains("/models"));
        assertTrue(recorded.getPath().contains("key=test-key"));
    }

    @Test
    public void test_checkAvailabilityNow_serverError() throws IOException {
        mockServer.enqueue(new MockResponse().setResponseCode(500).setBody("Internal Server Error"));

        setupClientForMockServer();

        assertFalse(client.checkAvailabilityNow());
    }

    @Test
    public void test_isAvailable_serverError() throws IOException {
        mockServer.enqueue(new MockResponse().setResponseCode(401).setBody("Unauthorized"));

        setupClientForMockServer();

        assertFalse(client.isAvailable());
    }

    @Test
    public void test_streamChat_serviceUnavailable() throws IOException {
        mockServer.enqueue(new MockResponse().setResponseCode(503).setBody("Service Unavailable"));

        setupClientForMockServer();

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");
        final AtomicBoolean errorReceived = new AtomicBoolean(false);

        try {
            client.streamChat(request, new LlmStreamCallback() {
                @Override
                public void onChunk(final String content, final boolean done) {
                    fail("Should not receive chunks on error");
                }

                @Override
                public void onError(final Throwable error) {
                    errorReceived.set(true);
                }
            });
            fail("Expected LlmException to be thrown");
        } catch (final LlmException error) {
            assertTrue(error.getMessage().contains("503"));
            assertTrue(errorReceived.get());
        }
    }

    @Test
    public void test_chat_serviceUnavailable() throws IOException {
        mockServer.enqueue(new MockResponse().setResponseCode(503).setBody("Service Unavailable"));

        setupClientForMockServer();

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");

        try {
            client.chat(request);
            fail("Expected LlmException to be thrown");
        } catch (final LlmException error) {
            assertTrue(error.getMessage().contains("503"));
        }
    }

    // ========== Multi-line JSON and thinking token tests ==========

    @Test
    public void test_streamChat_multiLineJson() throws IOException {
        // Gemini 3 sends multi-line JSON objects in streaming responses
        final String streamResponse = "[{\n" //
                + "  \"candidates\": [{\n" //
                + "    \"content\": {\n" //
                + "      \"parts\": [{\n" //
                + "        \"text\": \"Hello\"\n" //
                + "      }],\n" //
                + "      \"role\": \"model\"\n" //
                + "    }\n" //
                + "  }]\n" //
                + "}\n" //
                + ",{\n" //
                + "  \"candidates\": [{\n" //
                + "    \"content\": {\n" //
                + "      \"parts\": [{\n" //
                + "        \"text\": \" World\"\n" //
                + "      }],\n" //
                + "      \"role\": \"model\"\n" //
                + "    },\n" //
                + "    \"finishReason\": \"STOP\"\n" //
                + "  }]\n" //
                + "}]";

        mockServer.enqueue(new MockResponse().setBody(streamResponse).addHeader("Content-Type", "application/json"));

        setupClientForMockServer();

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");
        final List<String> chunks = new ArrayList<>();
        final AtomicBoolean doneReceived = new AtomicBoolean(false);

        client.streamChat(request, new LlmStreamCallback() {
            @Override
            public void onChunk(final String content, final boolean done) {
                chunks.add(content);
                if (done) {
                    doneReceived.set(true);
                }
            }

            @Override
            public void onError(final Throwable error) {
                fail("Unexpected error: " + error.getMessage());
            }
        });

        assertEquals(2, chunks.size());
        assertEquals("Hello", chunks.get(0));
        assertEquals(" World", chunks.get(1));
        assertTrue(doneReceived.get());
    }

    @Test
    public void test_streamChat_skipsThinkingParts() throws IOException {
        // Gemini 3 models may include thinking parts with "thought": true
        final String streamResponse = """
                [
                {"candidates":[{"content":{"parts":[{"thought":true,"text":"Let me think about this..."}],"role":"model"}}]},
                {"candidates":[{"content":{"parts":[{"text":"Actual response"}],"role":"model"},"finishReason":"STOP"}]}
                ]
                """;

        mockServer.enqueue(new MockResponse().setBody(streamResponse).addHeader("Content-Type", "application/json"));

        setupClientForMockServer();

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");
        final List<String> chunks = new ArrayList<>();
        final AtomicBoolean doneReceived = new AtomicBoolean(false);

        client.streamChat(request, new LlmStreamCallback() {
            @Override
            public void onChunk(final String content, final boolean done) {
                chunks.add(content);
                if (done) {
                    doneReceived.set(true);
                }
            }

            @Override
            public void onError(final Throwable error) {
                fail("Unexpected error: " + error.getMessage());
            }
        });

        // Thinking part should be skipped, only actual response should be received
        assertEquals(1, chunks.size());
        assertEquals("Actual response", chunks.get(0));
        assertTrue(doneReceived.get());
    }

    @Test
    public void test_streamChat_multiLineJsonWithThinking() throws IOException {
        // Gemini 3 multi-line format with thinking parts
        final String streamResponse = "[{\n" //
                + "  \"candidates\": [{\n" //
                + "    \"content\": {\n" //
                + "      \"parts\": [{\n" //
                + "        \"thought\": true,\n" //
                + "        \"text\": \"Thinking about the question...\"\n" //
                + "      }],\n" //
                + "      \"role\": \"model\"\n" //
                + "    }\n" //
                + "  }]\n" //
                + "},\n" //
                + "{\n" //
                + "  \"candidates\": [{\n" //
                + "    \"content\": {\n" //
                + "      \"parts\": [{\n" //
                + "        \"text\": \"The answer is 42.\"\n" //
                + "      }],\n" //
                + "      \"role\": \"model\"\n" //
                + "    },\n" //
                + "    \"finishReason\": \"STOP\"\n" //
                + "  }]\n" //
                + "}]";

        mockServer.enqueue(new MockResponse().setBody(streamResponse).addHeader("Content-Type", "application/json"));

        setupClientForMockServer();

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("What is the answer?");
        final List<String> chunks = new ArrayList<>();
        final AtomicBoolean doneReceived = new AtomicBoolean(false);

        client.streamChat(request, new LlmStreamCallback() {
            @Override
            public void onChunk(final String content, final boolean done) {
                chunks.add(content);
                if (done) {
                    doneReceived.set(true);
                }
            }

            @Override
            public void onError(final Throwable error) {
                fail("Unexpected error: " + error.getMessage());
            }
        });

        assertEquals(1, chunks.size());
        assertEquals("The answer is 42.", chunks.get(0));
        assertTrue(doneReceived.get());
    }

    @Test
    public void test_chat_skipsThinkingParts() throws IOException {
        // Non-streaming chat with thinking parts
        final String responseJson = """
                {
                    "candidates": [{
                        "content": {
                            "parts": [
                                {"thought": true, "text": "Let me reason about this..."},
                                {"text": "The actual answer."}
                            ],
                            "role": "model"
                        },
                        "finishReason": "STOP"
                    }],
                    "usageMetadata": {
                        "promptTokenCount": 10,
                        "candidatesTokenCount": 50,
                        "totalTokenCount": 60
                    }
                }
                """;

        mockServer.enqueue(new MockResponse().setBody(responseJson).addHeader("Content-Type", "application/json"));

        setupClientForMockServer();

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");
        final LlmChatResponse response = client.chat(request);

        // Should skip thinking part and only return actual text
        assertEquals("The actual answer.", response.getContent());
        assertEquals("STOP", response.getFinishReason());
    }

    @Test
    public void test_chat_onlyThinkingParts() throws IOException {
        // Edge case: response contains only thinking parts
        final String responseJson = """
                {
                    "candidates": [{
                        "content": {
                            "parts": [
                                {"thought": true, "text": "Thinking..."}
                            ],
                            "role": "model"
                        },
                        "finishReason": "STOP"
                    }]
                }
                """;

        mockServer.enqueue(new MockResponse().setBody(responseJson).addHeader("Content-Type", "application/json"));

        setupClientForMockServer();

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hello");
        final LlmChatResponse response = client.chat(request);

        // No non-thinking text content
        assertNull(response.getContent());
    }

    @Test
    public void test_chat_logsFinishReasonInInfo() throws IOException {
        final String responseJson = """
                {
                    "candidates": [{
                        "content": {"parts": [{"text": "Tru"}], "role": "model"},
                        "finishReason": "MAX_TOKENS"
                    }],
                    "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 1, "totalTokenCount": 6}
                }
                """;
        mockServer.enqueue(new MockResponse().setBody(responseJson).addHeader("Content-Type", "application/json"));
        setupClientForMockServer();

        final LlmChatRequest request = new LlmChatRequest().addUserMessage("Hi");
        final LlmChatResponse response = client.chat(request);

        assertEquals("Tru", response.getContent());
        assertEquals("MAX_TOKENS", response.getFinishReason());
        // The fact that the WARN actually fired is verified manually via log capture in CI; here we
        // just guard the response shape and that isAbnormalFinishReason classifies it.
        assertTrue(GeminiLlmClient.isAbnormalFinishReason(response.getFinishReason()));
    }

    // ========== buildApiUrl streaming alt=sse tests ==========

    @Test
    public void test_buildApiUrl_streamingUsesAltSse() {
        setupClientForMockServer();
        final String url = client.testBuildApiUrl("gemini-2.5-flash", true);
        assertTrue("expected ?alt=sse in URL but was: " + url, url.contains("alt=sse"));
        assertTrue("expected key parameter in URL", url.contains("key="));
        assertTrue("expected streamGenerateContent action", url.contains(":streamGenerateContent"));
    }

    @Test
    public void test_buildApiUrl_nonStreamingHasNoAltSse() {
        setupClientForMockServer();
        final String url = client.testBuildApiUrl("gemini-2.5-flash", false);
        assertFalse("non-stream URL should NOT carry alt=sse: " + url, url.contains("alt=sse"));
        assertTrue(url.contains(":generateContent"));
    }

    @Test
    public void test_maskApiKeyInUrl_withAltSse() {
        setupClientForMockServer();
        final String masked = client.testMaskApiKeyInUrl(
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent?key=SECRET&alt=sse");
        assertEquals("https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent?key=***&alt=sse",
                masked);
    }

    // ========== Stream completion log diagnostics tests ==========
    // These tests verify that streamChat emits finishReason and usageMetadata
    // in its "Stream completed" INFO log so that 1-character / truncated-response
    // cases (MAX_TOKENS, SAFETY, RECITATION, thinking-budget exhaustion) are
    // distinguishable from chunk count alone.

    @Test
    public void test_streamChat_logIncludesFinishReasonStop() throws IOException {
        final String streamResponse = """
                [
                {"candidates":[{"content":{"parts":[{"text":"Hi"}]},"finishReason":"STOP"}]}
                ]
                """;
        mockServer.enqueue(new MockResponse().setBody(streamResponse).addHeader("Content-Type", "application/json"));
        setupClientForMockServer();

        final List<String> messages = new ArrayList<>();
        runStreamWithCapturedLogs(messages, () -> {
            client.streamChat(new LlmChatRequest().addUserMessage("Hello"), new NoopCallback());
        });

        final String completed = findCompletionLog(messages);
        assertNotNull(completed, "Stream completed log not emitted");
        assertTrue(completed.contains("finishReason=STOP"), "missing finishReason=STOP: " + completed);
    }

    @Test
    public void test_streamChat_logIncludesFinishReasonMaxTokens() throws IOException {
        final String streamResponse = """
                [
                {"candidates":[{"content":{"parts":[{"text":"!"}]},"finishReason":"MAX_TOKENS"}]}
                ]
                """;
        mockServer.enqueue(new MockResponse().setBody(streamResponse).addHeader("Content-Type", "application/json"));
        setupClientForMockServer();

        final List<String> messages = new ArrayList<>();
        runStreamWithCapturedLogs(messages, () -> {
            client.streamChat(new LlmChatRequest().addUserMessage("Hello"), new NoopCallback());
        });

        final String completed = findCompletionLog(messages);
        assertNotNull(completed, "Stream completed log not emitted");
        assertTrue(completed.contains("finishReason=MAX_TOKENS"), "missing finishReason=MAX_TOKENS: " + completed);
    }

    @Test
    public void test_streamChat_logIncludesFinishReasonSafety() throws IOException {
        // SAFETY may arrive without any text parts
        final String streamResponse = """
                [
                {"candidates":[{"finishReason":"SAFETY"}]}
                ]
                """;
        mockServer.enqueue(new MockResponse().setBody(streamResponse).addHeader("Content-Type", "application/json"));
        setupClientForMockServer();

        final List<String> messages = new ArrayList<>();
        final AtomicBoolean doneReceived = new AtomicBoolean(false);
        runStreamWithCapturedLogs(messages, () -> {
            client.streamChat(new LlmChatRequest().addUserMessage("Hello"), new LlmStreamCallback() {
                @Override
                public void onChunk(final String content, final boolean done) {
                    if (done) {
                        doneReceived.set(true);
                    }
                }

                @Override
                public void onError(final Throwable error) {
                    fail("Unexpected error: " + error.getMessage());
                }
            });
        });

        assertTrue(doneReceived.get(), "done callback not invoked for SAFETY finish");
        final String completed = findCompletionLog(messages);
        assertNotNull(completed, "Stream completed log not emitted");
        assertTrue(completed.contains("finishReason=SAFETY"), "missing finishReason=SAFETY: " + completed);
    }

    @Test
    public void test_streamChat_logIncludesUsageMetadata() throws IOException {
        final String streamResponse = """
                [
                {"candidates":[{"content":{"parts":[{"text":"!"}]},"finishReason":"MAX_TOKENS"}],
                 "usageMetadata":{"promptTokenCount":8192,"candidatesTokenCount":1,"totalTokenCount":8193}}
                ]
                """;
        mockServer.enqueue(new MockResponse().setBody(streamResponse).addHeader("Content-Type", "application/json"));
        setupClientForMockServer();

        final List<String> messages = new ArrayList<>();
        runStreamWithCapturedLogs(messages, () -> {
            client.streamChat(new LlmChatRequest().addUserMessage("Hello"), new NoopCallback());
        });

        final String completed = findCompletionLog(messages);
        assertNotNull(completed, "Stream completed log not emitted");
        assertTrue(completed.contains("promptTokens=8192"), "missing promptTokens in usage: " + completed);
        assertTrue(completed.contains("candidatesTokens=1"), "missing candidatesTokens: " + completed);
        assertTrue(completed.contains("totalTokens=8193"), "missing totalTokens: " + completed);
    }

    @Test
    public void test_streamChat_logIncludesThoughtsTokenCount() throws IOException {
        // Diagnostic case: thinking budget consumed all output → 1-char visible response
        final String streamResponse = """
                [
                {"candidates":[{"content":{"parts":[{"text":"!"}]},"finishReason":"MAX_TOKENS"}],
                 "usageMetadata":{"promptTokenCount":120,"candidatesTokenCount":1,"thoughtsTokenCount":4095,"totalTokenCount":4216}}
                ]
                """;
        mockServer.enqueue(new MockResponse().setBody(streamResponse).addHeader("Content-Type", "application/json"));
        setupClientForMockServer();

        final List<String> messages = new ArrayList<>();
        runStreamWithCapturedLogs(messages, () -> {
            client.streamChat(new LlmChatRequest().addUserMessage("Hello"), new NoopCallback());
        });

        final String completed = findCompletionLog(messages);
        assertNotNull(completed, "Stream completed log not emitted");
        assertTrue(completed.contains("thoughtsTokens=4095"),
                "missing thoughtsTokens in usage (needed to diagnose thinking-budget exhaustion): " + completed);
        assertTrue(completed.contains("finishReason=MAX_TOKENS"), "missing finishReason=MAX_TOKENS: " + completed);
    }

    @Test
    public void test_streamChat_logCapturesLastUsageMetadata() throws IOException {
        // When usageMetadata appears in multiple chunks, the last one (typically richest) wins
        final String streamResponse = """
                [
                {"candidates":[{"content":{"parts":[{"text":"A"}]}}],
                 "usageMetadata":{"promptTokenCount":100,"totalTokenCount":101}},
                {"candidates":[{"content":{"parts":[{"text":"B"}]},"finishReason":"STOP"}],
                 "usageMetadata":{"promptTokenCount":100,"candidatesTokenCount":2,"totalTokenCount":102}}
                ]
                """;
        mockServer.enqueue(new MockResponse().setBody(streamResponse).addHeader("Content-Type", "application/json"));
        setupClientForMockServer();

        final List<String> messages = new ArrayList<>();
        runStreamWithCapturedLogs(messages, () -> {
            client.streamChat(new LlmChatRequest().addUserMessage("Hello"), new NoopCallback());
        });

        final String completed = findCompletionLog(messages);
        assertNotNull(completed, "Stream completed log not emitted");
        // Final usage carries candidatesTokenCount=2 — proves we logged the last, not the first
        assertTrue(completed.contains("candidatesTokens=2"), "expected last usageMetadata to win, but log was: " + completed);
        assertTrue(completed.contains("totalTokens=102"), "expected totalTokens=102 in final usage: " + completed);
    }

    @Test
    public void test_streamChat_logShowsNullsWhenNoFinishReasonOrUsage() throws IOException {
        // Stream that ends without any finishReason or usageMetadata.
        // The log must still be emitted (no NPE) and show the absence explicitly as null.
        final String streamResponse = """
                [
                {"candidates":[{"content":{"parts":[{"text":"Partial"}]}}]}
                ]
                """;
        mockServer.enqueue(new MockResponse().setBody(streamResponse).addHeader("Content-Type", "application/json"));
        setupClientForMockServer();

        final List<String> messages = new ArrayList<>();
        runStreamWithCapturedLogs(messages, () -> {
            client.streamChat(new LlmChatRequest().addUserMessage("Hello"), new NoopCallback());
        });

        final String completed = findCompletionLog(messages);
        assertNotNull(completed, "Stream completed log not emitted");
        assertTrue(completed.contains("finishReason=null"), "expected finishReason=null when absent: " + completed);
        assertTrue(completed.contains("promptTokens=null"), "expected promptTokens=null when absent: " + completed);
        assertTrue(completed.contains("candidatesTokens=null"), "expected candidatesTokens=null when absent: " + completed);
        assertTrue(completed.contains("thoughtsTokens=null"), "expected thoughtsTokens=null when absent: " + completed);
        assertTrue(completed.contains("totalTokens=null"), "expected totalTokens=null when absent: " + completed);
    }

    @Test
    public void test_streamChat_logHandlesJsonNullUsageMetadata() throws IOException {
        // usageMetadata field present with explicit JSON null — must not throw
        final String streamResponse = """
                [
                {"candidates":[{"content":{"parts":[{"text":"X"}]},"finishReason":"STOP"}],"usageMetadata":null}
                ]
                """;
        mockServer.enqueue(new MockResponse().setBody(streamResponse).addHeader("Content-Type", "application/json"));
        setupClientForMockServer();

        final List<String> messages = new ArrayList<>();
        runStreamWithCapturedLogs(messages, () -> {
            client.streamChat(new LlmChatRequest().addUserMessage("Hello"), new NoopCallback());
        });

        final String completed = findCompletionLog(messages);
        assertNotNull(completed, "Stream completed log not emitted");
        assertTrue(completed.contains("finishReason=STOP"), "missing finishReason=STOP: " + completed);
        // usageMetadata is JSON null — all individual token fields must remain null
        assertTrue(completed.contains("promptTokens=null"), "expected promptTokens=null with JSON null usageMetadata: " + completed);
        assertTrue(completed.contains("candidatesTokens=null"),
                "expected candidatesTokens=null with JSON null usageMetadata: " + completed);
        assertTrue(completed.contains("totalTokens=null"), "expected totalTokens=null with JSON null usageMetadata: " + completed);
    }

    @Test
    public void test_streamChat_existingChunkBehaviorPreserved() throws IOException {
        // Regression guard: the diagnostic logging additions must not change the
        // existing chunk delivery / done-flag contract.
        final String streamResponse = """
                [
                {"candidates":[{"content":{"parts":[{"text":"Hello"}],"role":"model"}}]},
                {"candidates":[{"content":{"parts":[{"text":" World"}],"role":"model"}}]},
                {"candidates":[{"content":{"parts":[{"text":"!"}],"role":"model"},"finishReason":"STOP"}],
                 "usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8}}
                ]
                """;
        mockServer.enqueue(new MockResponse().setBody(streamResponse).addHeader("Content-Type", "application/json"));
        setupClientForMockServer();

        final List<String> chunks = new ArrayList<>();
        final AtomicBoolean doneReceived = new AtomicBoolean(false);
        final AtomicInteger doneCount = new AtomicInteger(0);
        client.streamChat(new LlmChatRequest().addUserMessage("Hello"), new LlmStreamCallback() {
            @Override
            public void onChunk(final String content, final boolean done) {
                chunks.add(content);
                if (done) {
                    doneReceived.set(true);
                    doneCount.incrementAndGet();
                }
            }

            @Override
            public void onError(final Throwable error) {
                fail("Unexpected error: " + error.getMessage());
            }
        });

        assertEquals(3, chunks.size());
        assertEquals("Hello", chunks.get(0));
        assertEquals(" World", chunks.get(1));
        assertEquals("!", chunks.get(2));
        assertTrue(doneReceived.get(), "done callback not invoked");
        assertEquals(1, doneCount.get());
    }

    // ========== Helper methods ==========

    private void setupClientForMockServer() {
        final String baseUrl = mockServer.url("").toString();
        // Remove trailing slash
        final String apiUrl = baseUrl.endsWith("/") ? baseUrl.substring(0, baseUrl.length() - 1) : baseUrl;
        client.setTestApiUrl(apiUrl);
        client.setTestApiKey("test-key");
        client.setTestModel("gemini-2.0-flash");
        client.setTestTemperature(0.7);
        client.setTestMaxTokens(4096);
        client.setTestTimeout(30000);
        client.init();
    }

    /**
     * Attaches a capturing appender to the GeminiLlmClient logger, runs the action,
     * then detaches the appender — leaving the logger configuration untouched for
     * subsequent tests. Captured formatted messages are appended to {@code sink}.
     */
    private void runStreamWithCapturedLogs(final List<String> sink, final Runnable action) {
        final org.apache.logging.log4j.core.Logger coreLogger =
                (org.apache.logging.log4j.core.Logger) LogManager.getLogger(GeminiLlmClient.class);
        final CapturingAppender appender = new CapturingAppender();
        appender.start();
        coreLogger.addAppender(appender);
        try {
            action.run();
        } finally {
            coreLogger.removeAppender(appender);
            appender.stop();
            sink.addAll(appender.snapshot());
        }
    }

    /** Returns the first captured "Stream completed." log line, or null if none. */
    private static String findCompletionLog(final List<String> messages) {
        for (final String m : messages) {
            if (m.contains("Stream completed.")) {
                return m;
            }
        }
        return null;
    }

    /** Callback that swallows chunks/done and fails on error — for log-only assertions. */
    private static final class NoopCallback implements LlmStreamCallback {
        @Override
        public void onChunk(final String content, final boolean done) {
            // no-op
        }

        @Override
        public void onError(final Throwable error) {
            throw new AssertionError("Unexpected onError: " + error.getMessage(), error);
        }
    }

    /** Log4j2 appender that records formatted messages for later assertion. */
    private static final class CapturingAppender extends AbstractAppender {
        private final List<String> messages = Collections.synchronizedList(new ArrayList<>());

        CapturingAppender() {
            super("CaptureGeminiLogs-" + System.nanoTime(), null, null, true, Property.EMPTY_ARRAY);
        }

        @Override
        public void append(final LogEvent event) {
            messages.add(event.getMessage().getFormattedMessage());
        }

        List<String> snapshot() {
            synchronized (messages) {
                return new ArrayList<>(messages);
            }
        }
    }

    // --- applyPromptTypeParams tests ---

    @Test
    public void test_applyDefaultParams_intent() {
        final LlmChatRequest request = new LlmChatRequest();
        client.testApplyDefaultParams(request, "intent");
        assertEquals(Double.valueOf(0.1), request.getTemperature());
        assertEquals(Integer.valueOf(256), request.getMaxTokens());
        assertEquals(Integer.valueOf(0), request.getThinkingBudget());
    }

    @Test
    public void test_applyDefaultParams_evaluation() {
        final LlmChatRequest request = new LlmChatRequest();
        client.testApplyDefaultParams(request, "evaluation");
        assertEquals(Double.valueOf(0.1), request.getTemperature());
        assertEquals(Integer.valueOf(256), request.getMaxTokens());
        assertEquals(Integer.valueOf(0), request.getThinkingBudget());
    }

    @Test
    public void test_applyDefaultParams_unclear() {
        final LlmChatRequest request = new LlmChatRequest();
        client.testApplyDefaultParams(request, "unclear");
        assertEquals(Double.valueOf(0.7), request.getTemperature());
        assertEquals(Integer.valueOf(512), request.getMaxTokens());
        assertEquals(Integer.valueOf(0), request.getThinkingBudget());
    }

    @Test
    public void test_applyDefaultParams_noresults() {
        final LlmChatRequest request = new LlmChatRequest();
        client.testApplyDefaultParams(request, "noresults");
        assertEquals(Double.valueOf(0.7), request.getTemperature());
        assertEquals(Integer.valueOf(512), request.getMaxTokens());
        assertEquals(Integer.valueOf(0), request.getThinkingBudget());
    }

    @Test
    public void test_applyDefaultParams_docnotfound() {
        final LlmChatRequest request = new LlmChatRequest();
        client.testApplyDefaultParams(request, "docnotfound");
        assertEquals(Double.valueOf(0.7), request.getTemperature());
        assertEquals(Integer.valueOf(256), request.getMaxTokens());
        assertEquals(Integer.valueOf(0), request.getThinkingBudget());
    }

    @Test
    public void test_applyDefaultParams_direct() {
        final LlmChatRequest request = new LlmChatRequest();
        client.testApplyDefaultParams(request, "direct");
        assertEquals(Double.valueOf(0.7), request.getTemperature());
        assertEquals(Integer.valueOf(2048), request.getMaxTokens());
        assertEquals(Integer.valueOf(1024), request.getThinkingBudget());
    }

    @Test
    public void test_applyDefaultParams_faq() {
        final LlmChatRequest request = new LlmChatRequest();
        client.testApplyDefaultParams(request, "faq");
        assertEquals(Double.valueOf(0.7), request.getTemperature());
        assertEquals(Integer.valueOf(2048), request.getMaxTokens());
        assertEquals(Integer.valueOf(1024), request.getThinkingBudget());
    }

    @Test
    public void test_applyDefaultParams_answer() {
        final LlmChatRequest request = new LlmChatRequest();
        client.testApplyDefaultParams(request, "answer");
        assertEquals(Double.valueOf(0.5), request.getTemperature());
        assertEquals(Integer.valueOf(8192), request.getMaxTokens());
        assertEquals(Integer.valueOf(0), request.getThinkingBudget());
    }

    @Test
    public void test_applyDefaultParams_summary() {
        final LlmChatRequest request = new LlmChatRequest();
        client.testApplyDefaultParams(request, "summary");
        assertEquals(Double.valueOf(0.3), request.getTemperature());
        assertEquals(Integer.valueOf(4096), request.getMaxTokens());
        assertEquals(Integer.valueOf(2048), request.getThinkingBudget());
    }

    @Test
    public void test_applyDefaultParams_queryregeneration() {
        final LlmChatRequest request = new LlmChatRequest();
        client.testApplyDefaultParams(request, "queryregeneration");
        assertEquals(Double.valueOf(0.3), request.getTemperature());
        assertEquals(Integer.valueOf(256), request.getMaxTokens());
        assertEquals(Integer.valueOf(0), request.getThinkingBudget());
    }

    @Test
    public void test_applyDefaultParams_userSettingsPreserved() {
        final LlmChatRequest request = new LlmChatRequest();
        request.setTemperature(0.9);
        request.setMaxTokens(8192);
        request.setThinkingBudget(1000);
        client.testApplyDefaultParams(request, "intent");
        assertEquals(Double.valueOf(0.9), request.getTemperature());
        assertEquals(Integer.valueOf(8192), request.getMaxTokens());
        assertEquals(Integer.valueOf(1000), request.getThinkingBudget());
    }

    @Test
    public void test_applyDefaultParams_unknownType() {
        final LlmChatRequest request = new LlmChatRequest();
        client.testApplyDefaultParams(request, "unknown");
        assertNull(request.getTemperature());
        assertNull(request.getMaxTokens());
        assertNull(request.getThinkingBudget());
    }

    // --- Budget method tests ---

    @Test
    public void test_getHistoryMaxChars_default() {
        assertEquals(10000, client.testGetHistoryMaxChars());
    }

    @Test
    public void test_getIntentHistoryMaxMessages_default() {
        assertEquals(10, client.testGetIntentHistoryMaxMessages());
    }

    @Test
    public void test_getHistoryAssistantMaxChars_default() {
        assertEquals(1000, client.testGetHistoryAssistantMaxChars());
    }

    /**
     * Testable subclass of GeminiLlmClient that allows setting configuration values
     * directly without depending on FessConfig.
     */
    private static class TestableGeminiLlmClient extends GeminiLlmClient {
        private String testApiKey = "";
        private String testApiUrl = "https://generativelanguage.googleapis.com/v1beta";
        private String testModel = "gemini-2.5-flash";
        private int testTimeout = 60000;
        private double testTemperature = 0.7;
        private int testMaxTokens = 4096;

        void setTestApiKey(final String apiKey) {
            this.testApiKey = apiKey;
        }

        void setTestApiUrl(final String apiUrl) {
            this.testApiUrl = apiUrl;
        }

        void setTestModel(final String model) {
            this.testModel = model;
        }

        void setTestTimeout(final int timeout) {
            this.testTimeout = timeout;
        }

        void setTestTemperature(final double temperature) {
            this.testTemperature = temperature;
        }

        void setTestMaxTokens(final int maxTokens) {
            this.testMaxTokens = maxTokens;
        }

        @Override
        protected String getApiKey() {
            return testApiKey;
        }

        @Override
        protected String getApiUrl() {
            return testApiUrl;
        }

        @Override
        protected String getModel() {
            return testModel;
        }

        @Override
        protected int getTimeout() {
            return testTimeout;
        }

        protected double getTemperature() {
            return testTemperature;
        }

        protected int getMaxTokens() {
            return testMaxTokens;
        }

        @Override
        protected String getLlmType() {
            return NAME;
        }

        @Override
        protected boolean isRagChatEnabled() {
            return false;
        }

        @Override
        protected int getAvailabilityCheckInterval() {
            return 0;
        }

        @Override
        protected int getHistoryMaxChars() {
            return 10000;
        }

        @Override
        protected int getIntentHistoryMaxMessages() {
            return 10;
        }

        @Override
        protected int getIntentHistoryMaxChars() {
            return 5000;
        }

        @Override
        public int getHistoryAssistantMaxChars() {
            return 1000;
        }

        @Override
        public int getHistoryAssistantSummaryMaxChars() {
            return 1000;
        }

        void testApplyDefaultParams(final LlmChatRequest request, final String promptType) {
            applyDefaultParams(request, promptType);
        }

        public String testBuildApiUrl(final String model, final boolean stream) {
            return buildApiUrl(model, stream);
        }

        public String testMaskApiKeyInUrl(final String url) {
            return maskApiKeyInUrl(url);
        }

        int testGetHistoryMaxChars() {
            return getHistoryMaxChars();
        }

        int testGetIntentHistoryMaxMessages() {
            return getIntentHistoryMaxMessages();
        }

        int testGetHistoryAssistantMaxChars() {
            return getHistoryAssistantMaxChars();
        }
    }
}
