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
import java.net.ServerSocket;
import java.net.Socket;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.core.LogEvent;
import org.apache.logging.log4j.core.appender.AbstractAppender;
import org.apache.logging.log4j.core.config.Configurator;
import org.apache.logging.log4j.core.config.Property;
import org.codelibs.fess.embedding.EmbeddingException;
import org.codelibs.fess.gemini.GeminiApiUrl;
import org.codelibs.fess.unit.UnitFessTestCase;
import org.codelibs.fess.util.ComponentUtil;
import org.codelibs.fess.util.CredentialUrlUtil;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInfo;

import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import okhttp3.mockwebserver.RecordedRequest;

public class GeminiEmbeddingClientTest extends UnitFessTestCase {

    /** The real config key read by the production (non-overridden) {@link GeminiEmbeddingClient#getDimension()}. */
    private static final String DIMENSION_CONFIG_KEY = "content_chunker.embedding.dimension";

    private TestableGeminiEmbeddingClient client;
    private MockWebServer server;

    /**
     * Several tests below (see "Real (non-overridden) ... coverage") mutate the
     * {@code systemProperties} component directly via {@link ComponentUtil#getSystemProperties()},
     * which is otherwise a JVM-lifetime singleton shared across test classes. Each such test
     * restores the key it touches in a {@code finally} block, but that convention alone does not
     * stop this class's mutations from leaking into (or being corrupted by) another test class
     * running around the same time; overriding this to {@code true} gives the class a fresh
     * container per test method instead.
     */
    @Override
    protected boolean isUseOneTimeContainer() {
        return true;
    }

    @Override
    public void setUp(final TestInfo testInfo) throws Exception {
        super.setUp(testInfo);
        client = new TestableGeminiEmbeddingClient();
        server = new MockWebServer();
        server.start();
    }

    @Override
    public void tearDown(final TestInfo testInfo) throws Exception {
        if (client != null) {
            client.destroy();
        }
        if (server != null) {
            server.shutdown();
        }
        super.tearDown(testInfo);
    }

    @Test
    public void test_getName() {
        assertEquals("gemini", client.getName());
    }

    // ========== embedDocuments() ==========

    @Test
    public void test_embedDocuments_success() throws Exception {
        final String responseJson = "{\"embeddings\":[{\"values\":[0.1,0.2,0.3]},{\"values\":[0.4,0.5,0.6]}]}";
        server.enqueue(new MockResponse().setBody(responseJson).setHeader("Content-Type", "application/json"));
        setupClient();
        client.setTestDimension(3);

        final List<float[]> result = client.embedDocuments(List.of("chunk one", "chunk two"));

        assertEquals(2, result.size());
        assertEquals(3, result.get(0).length);
        // dimension 3 is an MRL truncation of gemini-embedding-001's native 3072, so the parsed
        // vectors are L2-normalized; assert the scaled components to keep pinning per-position
        // parse order and per-vector separation.
        final float norm0 = (float) Math.sqrt(0.1 * 0.1 + 0.2 * 0.2 + 0.3 * 0.3);
        final float norm1 = (float) Math.sqrt(0.4 * 0.4 + 0.5 * 0.5 + 0.6 * 0.6);
        assertTrue(Math.abs(result.get(0)[0] - 0.1f / norm0) < 1e-6f, "vector 0 component 0: " + result.get(0)[0]);
        assertTrue(Math.abs(result.get(0)[1] - 0.2f / norm0) < 1e-6f, "vector 0 component 1: " + result.get(0)[1]);
        assertTrue(Math.abs(result.get(1)[2] - 0.6f / norm1) < 1e-6f, "vector 1 component 2: " + result.get(1)[2]);

        final RecordedRequest recordedRequest = server.takeRequest();
        assertEquals("POST", recordedRequest.getMethod());
        assertTrue(recordedRequest.getPath().endsWith(":batchEmbedContents"),
                "path should end with :batchEmbedContents: " + recordedRequest.getPath());
        assertNotNull(recordedRequest.getHeader("x-goog-api-key"), "x-goog-api-key header should be present");
        final String body = recordedRequest.getBody().readUtf8();
        assertTrue(body.contains("\"requests\""), "request body should carry a requests array: " + body);
        assertTrue(body.contains("\"model\":\"models/gemini-embedding-001\""),
                "request body should carry the models/-prefixed model on each nested request: " + body);
        assertTrue(body.contains("\"outputDimensionality\":3"), "request body should carry the configured dimension: " + body);
        assertTrue(body.contains("\"taskType\":\"RETRIEVAL_DOCUMENT\""),
                "embedDocuments() should default taskType to RETRIEVAL_DOCUMENT: " + body);
    }

    @Test
    public void test_embedDocuments_emptyInput_returnsEmptyList() throws Exception {
        setupClient();

        assertTrue(client.embedDocuments(null).isEmpty());
        assertTrue(client.embedDocuments(Collections.emptyList()).isEmpty());
        assertEquals("empty input should not make any HTTP call", 0, server.getRequestCount());
    }

    @Test
    public void test_embedDocuments_dimensionMismatch_throws() throws Exception {
        // Server returns 3-dim vectors but the configured dimension is 4.
        final String responseJson = "{\"embeddings\":[{\"values\":[0.1,0.2,0.3]}]}";
        server.enqueue(new MockResponse().setBody(responseJson).setHeader("Content-Type", "application/json"));
        setupClient();
        client.setTestDimension(4);

        try {
            client.embedDocuments(List.of("chunk one"));
            fail("expected EmbeddingException on dimension mismatch");
        } catch (final EmbeddingException e) {
            assertTrue(e.getMessage().contains("dimension mismatch"), "message should mention dimension mismatch: " + e.getMessage());
        }
    }

    @Test
    public void test_embedDocuments_nonNumericVectorComponent_throws() throws Exception {
        // The "values" array's second element is a JSON null instead of a number.
        // A naive Jackson asDouble() call would silently coerce this to 0.0 and corrupt
        // the stored vector instead of surfacing a clear error.
        final String responseJson = "{\"embeddings\":[{\"values\":[0.1,null,0.3]}]}";
        server.enqueue(new MockResponse().setBody(responseJson).setHeader("Content-Type", "application/json"));
        setupClient();
        client.setTestDimension(3);

        try {
            client.embedDocuments(List.of("chunk one"));
            fail("expected EmbeddingException on non-numeric vector component");
        } catch (final EmbeddingException e) {
            assertTrue(e.getMessage().contains("not numeric"), "message should mention non-numeric component: " + e.getMessage());
        }
    }

    @Test
    public void test_embedDocuments_nonFiniteVectorComponent_throws() throws Exception {
        // A JSON literal like 1e999 overflows to +Infinity, which Jackson's isNumber() still
        // accepts; the finite guard must reject it so a poisoned vector never reaches the index.
        final String responseJson = "{\"embeddings\":[{\"values\":[0.1,1e999,0.3]}]}";
        server.enqueue(new MockResponse().setBody(responseJson).setHeader("Content-Type", "application/json"));
        setupClient();
        client.setTestDimension(3);

        try {
            client.embedDocuments(List.of("chunk one"));
            fail("expected EmbeddingException on non-finite vector component");
        } catch (final EmbeddingException e) {
            assertTrue(e.getMessage().contains("not finite"), "message should mention non-finite component: " + e.getMessage());
        }
    }

    @Test
    public void test_embedDocuments_countMismatch_throws() throws Exception {
        // Server returns 1 vector but 2 texts were requested.
        final String responseJson = "{\"embeddings\":[{\"values\":[0.1,0.2,0.3]}]}";
        server.enqueue(new MockResponse().setBody(responseJson).setHeader("Content-Type", "application/json"));
        setupClient();
        client.setTestDimension(3);

        try {
            client.embedDocuments(List.of("chunk one", "chunk two"));
            fail("expected EmbeddingException on count mismatch");
        } catch (final EmbeddingException e) {
            assertTrue(e.getMessage().contains("count mismatch"), "message should mention count mismatch: " + e.getMessage());
        }
    }

    @Test
    public void test_embedDocuments_doesNotRetryOn400() throws Exception {
        server.enqueue(new MockResponse().setResponseCode(400).setBody("bad request"));
        setupClient();
        client.setTestDimension(3);
        client.setTestRetryMax(5);
        client.setTestRetryBaseDelayMs(0L);

        try {
            client.embedDocuments(List.of("chunk"));
            fail("expected EmbeddingException");
        } catch (final EmbeddingException e) {
            // expected
        }
        assertEquals("400 must not be retried", 1, server.getRequestCount());
    }

    @Test
    public void test_embedDocuments_retriesOn503() throws Exception {
        server.enqueue(new MockResponse().setResponseCode(503));
        server.enqueue(new MockResponse().setResponseCode(503));
        final String successBody = "{\"embeddings\":[{\"values\":[0.1,0.2,0.3]}]}";
        server.enqueue(new MockResponse().setHeader("Content-Type", "application/json").setBody(successBody));
        setupClient();
        client.setTestDimension(3);
        client.setTestRetryMax(5);
        client.setTestRetryBaseDelayMs(0L);

        final List<float[]> result = client.embedDocuments(List.of("chunk"));

        assertEquals(1, result.size());
        assertEquals(3, server.getRequestCount());
    }

    /**
     * Pins deviation #2 from {@code OllamaEmbeddingClient}: connect-level {@link IOException}
     * must propagate immediately, with no retry-driven extra attempts. Uses a raw
     * {@link ServerSocket} that accepts the TCP connection and then closes it without ever
     * writing a response, so the number of accepted connections is an exact attempt count
     * (unlike a refused port, which would give no signal either way).
     */
    @Test
    public void test_embedDocuments_doesNotRetryOnConnectFailure() throws Exception {
        final ServerSocket deadSocket = new ServerSocket(0);
        final AtomicInteger acceptCount = new AtomicInteger(0);
        final Thread acceptor = new Thread(() -> {
            try {
                while (!deadSocket.isClosed()) {
                    final Socket socket = deadSocket.accept();
                    acceptCount.incrementAndGet();
                    socket.close();
                }
            } catch (final IOException e) {
                // expected once deadSocket.close() runs in the finally block below
            }
        });
        acceptor.setDaemon(true);
        acceptor.start();

        try {
            client.setTestApiUrl("http://localhost:" + deadSocket.getLocalPort());
            client.setTestApiKey("test-key");
            client.setTestModel("gemini-embedding-001");
            client.setTestTimeout(5000);
            client.setTestDimension(3);
            client.setTestRetryMax(5);
            client.setTestRetryBaseDelayMs(0L);
            client.init();

            try {
                client.embedDocuments(List.of("chunk"));
                fail("expected EmbeddingException on connect failure");
            } catch (final EmbeddingException e) {
                assertTrue(e.getCause() instanceof IOException, "cause should be IOException: " + e.getCause());
            }

            // Give the acceptor thread a moment in case a (buggy) retry drove extra connections.
            Thread.sleep(200L);
            assertEquals("connect-level IOException must not be retried", 1, acceptCount.get());
        } finally {
            deadSocket.close();
        }
    }

    // ========== sub-batch splitting (MAX_BATCH_SIZE) ==========

    /**
     * 250 texts must be split into 3 sequential {@code batchEmbedContents} calls (100 + 100 + 50),
     * each carrying at most {@link GeminiEmbeddingClient#MAX_BATCH_SIZE} nested requests, and the
     * returned vectors must be concatenated back in input order. Each mock vector encodes its
     * global input index in its direction (see {@link #embeddingsResponse(int, int, int)}), so an
     * out-of-order or mis-concatenated reassembly is detectable.
     */
    @Test
    public void test_embedDocuments_splitsBatchesOverMax() throws Exception {
        final int dimension = 2;
        server.enqueue(embeddingsResponse(0, 100, dimension));
        server.enqueue(embeddingsResponse(100, 100, dimension));
        server.enqueue(embeddingsResponse(200, 50, dimension));
        setupClient();
        client.setTestDimension(dimension);

        final List<String> texts = new ArrayList<>(250);
        for (int i = 0; i < 250; i++) {
            texts.add("chunk-" + i);
        }

        final List<float[]> result = client.embedDocuments(texts);

        assertEquals(250, result.size());
        assertEquals("input over MAX_BATCH_SIZE should split into 3 calls", 3, server.getRequestCount());
        for (int i = 0; i < 250; i++) {
            final float[] vector = result.get(i);
            final float encodedIndex = vector[1] / vector[0];
            assertTrue(Math.abs(encodedIndex - i) < 1e-3f,
                    "vector at position " + i + " should carry input index " + i + " but carries " + encodedIndex);
        }
        assertNestedRequestCount(server.takeRequest(), 100);
        assertNestedRequestCount(server.takeRequest(), 100);
        assertNestedRequestCount(server.takeRequest(), 50);
    }

    /**
     * Exactly {@link GeminiEmbeddingClient#MAX_BATCH_SIZE} texts must stay a single call
     * (the common/small case must not regress into needless extra HTTP round-trips).
     */
    @Test
    public void test_embedDocuments_exactlyMax_singleCall() throws Exception {
        final int dimension = 2;
        server.enqueue(embeddingsResponse(0, 100, dimension));
        setupClient();
        client.setTestDimension(dimension);

        final List<String> texts = new ArrayList<>(100);
        for (int i = 0; i < 100; i++) {
            texts.add("chunk-" + i);
        }

        final List<float[]> result = client.embedDocuments(texts);

        assertEquals(100, result.size());
        assertEquals("exactly MAX_BATCH_SIZE must stay a single call", 1, server.getRequestCount());
        assertNestedRequestCount(server.takeRequest(), 100);
    }

    /**
     * A failure in a non-first sub-batch (here the 2nd of 3, via a non-retryable 400) must
     * propagate as {@link EmbeddingException} rather than returning the first sub-batch's vectors
     * as a partial/corrupted result. The request count also proves the loop aborts on failure: the
     * 3rd sub-batch is never sent.
     */
    @Test
    public void test_embedDocuments_subBatchFailurePropagates() throws Exception {
        final int dimension = 2;
        server.enqueue(embeddingsResponse(0, 100, dimension)); // sub-batch 1 succeeds
        server.enqueue(new MockResponse().setResponseCode(400).setBody("bad request")); // sub-batch 2 fails
        // no response enqueued for sub-batch 3: it must never be sent
        setupClient();
        client.setTestDimension(dimension);
        client.setTestRetryMax(5);
        client.setTestRetryBaseDelayMs(0L);

        final List<String> texts = new ArrayList<>(250);
        for (int i = 0; i < 250; i++) {
            texts.add("chunk-" + i);
        }

        try {
            client.embedDocuments(texts);
            fail("expected EmbeddingException when a non-first sub-batch fails");
        } catch (final EmbeddingException e) {
            // expected: no partial result
        }
        // sub-batch 1 (1 call) + sub-batch 2 (400 is non-retryable, 1 call); sub-batch 3 never runs
        assertEquals("failure must propagate, 400 must not be retried, and sub-batch 3 must not be sent", 2, server.getRequestCount());
    }

    // ========== embedQuery() ==========

    @Test
    public void test_embedQuery_success() throws Exception {
        final String responseJson = "{\"embeddings\":[{\"values\":[0.1,0.2,0.3]}]}";
        server.enqueue(new MockResponse().setBody(responseJson).setHeader("Content-Type", "application/json"));
        setupClient();
        client.setTestDimension(3);

        final List<float[]> result = client.embedQuery(List.of("what is fess?"));

        assertEquals(1, result.size());
        assertEquals(3, result.get(0).length);

        final RecordedRequest recordedRequest = server.takeRequest();
        final String body = recordedRequest.getBody().readUtf8();
        assertTrue(body.contains("\"taskType\":\"RETRIEVAL_QUERY\""), "embedQuery() should default taskType to RETRIEVAL_QUERY: " + body);
    }

    @Test
    public void test_embedQuery_emptyInput_returnsEmptyList() throws Exception {
        setupClient();

        assertTrue(client.embedQuery(null).isEmpty());
        assertTrue(client.embedQuery(Collections.emptyList()).isEmpty());
        assertEquals("empty input should not make any HTTP call", 0, server.getRequestCount());
    }

    // ========== task_type configuration ==========

    @Test
    public void test_taskType_blankConfig_omitsFieldFromRequest() throws Exception {
        final String responseJson = "{\"embeddings\":[{\"values\":[0.1,0.2,0.3]}]}";
        server.enqueue(new MockResponse().setBody(responseJson).setHeader("Content-Type", "application/json"));
        server.enqueue(new MockResponse().setBody(responseJson).setHeader("Content-Type", "application/json"));
        setupClient();
        client.setTestDimension(3);
        client.setTestDocumentTaskType("");
        client.setTestQueryTaskType("");

        client.embedDocuments(List.of("chunk"));
        client.embedQuery(List.of("query"));

        final String documentBody = server.takeRequest().getBody().readUtf8();
        assertFalse(documentBody.contains("taskType"), "blank document.task_type must omit the field: " + documentBody);
        final String queryBody = server.takeRequest().getBody().readUtf8();
        assertFalse(queryBody.contains("taskType"), "blank query.task_type must omit the field: " + queryBody);
    }

    @Test
    public void test_taskType_customConfig_isSent() throws Exception {
        final String responseJson = "{\"embeddings\":[{\"values\":[0.1,0.2,0.3]}]}";
        server.enqueue(new MockResponse().setBody(responseJson).setHeader("Content-Type", "application/json"));
        setupClient();
        client.setTestDimension(3);
        client.setTestDocumentTaskType("SEMANTIC_SIMILARITY");

        client.embedDocuments(List.of("chunk"));

        final String body = server.takeRequest().getBody().readUtf8();
        assertTrue(body.contains("\"taskType\":\"SEMANTIC_SIMILARITY\""), "custom document.task_type should be sent: " + body);
    }

    // ========== outputDimensionality gating (model support) ==========

    @Test
    public void test_outputDimensionality_legacyModel_omitsField() throws Exception {
        // embedding-001 predates Matryoshka (MRL); sending outputDimensionality to it is a
        // non-retryable 400, so the field must be omitted for the legacy embedding-* line.
        final String responseJson = "{\"embeddings\":[{\"values\":[0.1,0.2,0.3]}]}";
        server.enqueue(new MockResponse().setBody(responseJson).setHeader("Content-Type", "application/json"));
        setupClient();
        client.setTestModel("embedding-001");
        client.setTestDimension(3);

        client.embedDocuments(List.of("chunk"));

        final String body = server.takeRequest().getBody().readUtf8();
        assertFalse(body.contains("outputDimensionality"), "legacy embedding-001 must omit outputDimensionality: " + body);
    }

    @Test
    public void test_outputDimensionality_textEmbedding004_sendsField() throws Exception {
        // text-embedding-004 supports MRL truncation, so the field must still be sent (guards
        // against an over-broad gate that would wrongly exclude it and break its truncation path).
        final String responseJson = "{\"embeddings\":[{\"values\":[0.1,0.2,0.3]}]}";
        server.enqueue(new MockResponse().setBody(responseJson).setHeader("Content-Type", "application/json"));
        setupClient();
        client.setTestModel("text-embedding-004");
        client.setTestDimension(3);

        client.embedDocuments(List.of("chunk"));

        final String body = server.takeRequest().getBody().readUtf8();
        assertTrue(body.contains("\"outputDimensionality\":3"),
                "text-embedding-004 supports MRL and must send outputDimensionality: " + body);
    }

    @Test
    public void test_supportsOutputDimensionality_predicate() {
        assertTrue(client.supportsOutputDimensionality("gemini-embedding-001"));
        assertTrue(client.supportsOutputDimensionality("gemini-embedding-2"));
        assertTrue(client.supportsOutputDimensionality("text-embedding-004"));
        assertTrue(client.supportsOutputDimensionality("text-embedding-005"));
        assertTrue(client.supportsOutputDimensionality("text-multilingual-embedding-002"));
        assertFalse(client.supportsOutputDimensionality("embedding-001"));
        assertFalse(client.supportsOutputDimensionality("embedding-gecko-001"));
        assertFalse(client.supportsOutputDimensionality(""));
        assertFalse(client.supportsOutputDimensionality(null));
    }

    // ========== MRL-truncation L2 normalization ==========

    /**
     * Google's embeddings documentation states that for {@code gemini-embedding-001} (this
     * client's default model) "you must manually normalize non-3072 dimensions": only the native
     * 3072-dimension output is unit-length, and MRL-truncated output is not. Since core's
     * {@code ChunkVectorHelper} permits {@code space_type} values that are not scale-invariant
     * ({@code innerproduct}, {@code l2}, ...), a non-normalized truncated vector silently skews
     * ANN ranking toward high-norm chunks.
     */
    @Test
    public void test_parseEmbedResponse_mrlTruncated_isL2Normalized() {
        client.setTestModel("gemini-embedding-001");

        final List<float[]> vectors = client.parseEmbedResponse("{\"embeddings\":[{\"values\":[3,4,0]}]}", 1, 3);

        assertEquals(1, vectors.size());
        final float[] vector = vectors.get(0);
        assertEquals(3, vector.length);
        assertTrue(Math.abs(vector[0] - 0.6f) < 1e-6f, "component 0 should be 3/5: " + vector[0]);
        assertTrue(Math.abs(vector[1] - 0.8f) < 1e-6f, "component 1 should be 4/5: " + vector[1]);
        assertTrue(Math.abs(vector[2]) < 1e-6f, "component 2 should stay 0: " + vector[2]);
        assertTrue(Math.abs(l2Norm(vector) - 1.0f) < 1e-6f, "MRL-truncated vector should have unit L2 norm: " + l2Norm(vector));
    }

    /**
     * The native 3072-dimension output is already unit-length per Google's documentation, so it
     * must pass through untouched: re-normalizing it would be a needless pass over every vector and
     * would mask a genuine provider-side scale change.
     */
    @Test
    public void test_parseEmbedResponse_nativeDimension_isNotNormalized() {
        client.setTestModel("gemini-embedding-001");
        final int nativeDimension = 3072;
        final StringBuilder sb = new StringBuilder("{\"embeddings\":[{\"values\":[3,4");
        for (int i = 2; i < nativeDimension; i++) {
            sb.append(",0");
        }
        sb.append("]}]}");

        final float[] vector = client.parseEmbedResponse(sb.toString(), 1, nativeDimension).get(0);

        assertEquals(nativeDimension, vector.length);
        assertEquals(3.0f, vector[0]);
        assertEquals(4.0f, vector[1]);
        assertEquals(0.0f, vector[2]);
    }

    /**
     * A legacy fixed-dimension model never applies MRL truncation (it rejects
     * {@code outputDimensionality} outright), so its output must not be rewritten either.
     */
    @Test
    public void test_parseEmbedResponse_legacyModel_isNotNormalized() {
        client.setTestModel("embedding-001");

        final float[] vector = client.parseEmbedResponse("{\"embeddings\":[{\"values\":[3,4,0]}]}", 1, 3).get(0);

        assertEquals(3.0f, vector[0]);
        assertEquals(4.0f, vector[1]);
        assertEquals(0.0f, vector[2]);
    }

    /**
     * A zero vector has no direction to preserve and dividing by its zero norm would emit
     * {@code NaN} components, poisoning the kNN index; it must be left as-is.
     */
    @Test
    public void test_parseEmbedResponse_zeroVector_isLeftUnchanged() {
        client.setTestModel("gemini-embedding-001");

        final float[] vector = client.parseEmbedResponse("{\"embeddings\":[{\"values\":[0,0,0]}]}", 1, 3).get(0);

        for (int i = 0; i < vector.length; i++) {
            assertEquals("zero vector component " + i + " must stay finite zero", 0.0f, vector[i]);
        }
    }

    private static float l2Norm(final float[] vector) {
        double sum = 0.0;
        for (final float component : vector) {
            sum += (double) component * component;
        }
        return (float) Math.sqrt(sum);
    }

    // ========== Real (non-overridden) getDimension() coverage ==========
    //
    // The tests above exercise TestableGeminiEmbeddingClient's own hand-written
    // getDimension() override, never the production method. These tests use a
    // plain `new GeminiEmbeddingClient()` (no subclass) to drive the real
    // ComponentUtil.getFessConfig().getSystemProperty("content_chunker.embedding.dimension", ...)
    // config-read seam directly, via the "systemProperties" test component
    // registered in test_app.xml (org.codelibs.fess.unit.TestSystemProperties).
    // That component instance is not guaranteed to be recreated per test method,
    // so each test explicitly sets/removes the key it needs and restores it in a
    // finally block to stay order-independent.

    @Test
    public void test_getDimension_configured() {
        ComponentUtil.getSystemProperties().setProperty(DIMENSION_CONFIG_KEY, "1536");
        try {
            final GeminiEmbeddingClient realClient = new GeminiEmbeddingClient();
            assertEquals(1536, realClient.getDimension());
        } finally {
            ComponentUtil.getSystemProperties().remove(DIMENSION_CONFIG_KEY);
        }
    }

    @Test
    public void test_getDimension_notConfigured_throws() {
        ComponentUtil.getSystemProperties().remove(DIMENSION_CONFIG_KEY);
        try {
            final GeminiEmbeddingClient realClient = new GeminiEmbeddingClient();
            try {
                realClient.getDimension();
                fail("expected EmbeddingException when dimension is unconfigured");
            } catch (final EmbeddingException e) {
                assertTrue(e.getMessage().contains("not configured"), "message should mention not configured: " + e.getMessage());
            }
        } finally {
            ComponentUtil.getSystemProperties().remove(DIMENSION_CONFIG_KEY);
        }
    }

    @Test
    public void test_getDimension_zero_throws() {
        // A parseable but non-positive dimension must be rejected before any network call: new
        // float[0] and outputDimensionality:0 would otherwise reach Gemini and corrupt results.
        ComponentUtil.getSystemProperties().setProperty(DIMENSION_CONFIG_KEY, "0");
        try {
            final GeminiEmbeddingClient realClient = new GeminiEmbeddingClient();
            try {
                realClient.getDimension();
                fail("expected EmbeddingException when dimension is zero");
            } catch (final EmbeddingException e) {
                assertTrue(e.getMessage().contains("must be positive"), "message should mention must be positive: " + e.getMessage());
            }
        } finally {
            ComponentUtil.getSystemProperties().remove(DIMENSION_CONFIG_KEY);
        }
    }

    @Test
    public void test_getDimension_negative_throws() {
        // A negative dimension parses fine but would throw NegativeArraySizeException at
        // new float[dimension] after a wasted request; reject it up front instead.
        ComponentUtil.getSystemProperties().setProperty(DIMENSION_CONFIG_KEY, "-5");
        try {
            final GeminiEmbeddingClient realClient = new GeminiEmbeddingClient();
            try {
                realClient.getDimension();
                fail("expected EmbeddingException when dimension is negative");
            } catch (final EmbeddingException e) {
                assertTrue(e.getMessage().contains("must be positive"), "message should mention must be positive: " + e.getMessage());
            }
        } finally {
            ComponentUtil.getSystemProperties().remove(DIMENSION_CONFIG_KEY);
        }
    }

    // ========== Real (non-overridden) task_type config-read coverage ==========
    //
    // The task_type tests above drive TestableGeminiEmbeddingClient's overrides
    // of getDocumentTaskType()/getQueryTaskType(), so they never touch the real
    // getConfigString() -> getFessConfig().getSystemProperty() resolution. These
    // tests use a plain `new GeminiEmbeddingClient()` (no subclass) to drive that
    // real config-read seam: with the key unset, it must fall through to the
    // DEFAULT_*_TASK_TYPE constant, guarding against a refactor that breaks the
    // default wiring. Like getDimension()'s getSystemProperty() seam, task_type is
    // injectable via the systemProperties test component (see
    // GeminiEmbeddingClientConfigChannelTest for the configured-value coverage that
    // proves the channel, not just the default).

    @Test
    public void test_getDocumentTaskType_notConfigured_returnsDefault() {
        final GeminiEmbeddingClient realClient = new GeminiEmbeddingClient();
        assertEquals("RETRIEVAL_DOCUMENT", realClient.getDocumentTaskType());
    }

    @Test
    public void test_getQueryTaskType_notConfigured_returnsDefault() {
        final GeminiEmbeddingClient realClient = new GeminiEmbeddingClient();
        assertEquals("RETRIEVAL_QUERY", realClient.getQueryTaskType());
    }

    // ========== Real (non-overridden) getRetryBaseDelayMs() coverage ==========
    //
    // A non-numeric retry.base.delay.ms would otherwise throw NumberFormatException
    // on every embed call (the value is read at the top of executeWithRetry), so the
    // production getter must swallow it and fall back to the 2000ms default. This drives
    // a plain `new GeminiEmbeddingClient()` (no subclass) so the real getConfigString() ->
    // getSystemProperty() read is exercised, injecting the bad value through the real
    // "systemProperties" test component (see ComponentUtil.getSystemProperties()) and
    // restoring it in a finally block to stay order-independent.

    @Test
    public void test_getRetryBaseDelayMs_nonNumericValue_returnsDefault() {
        final String key = "content_chunker.embedding.gemini.retry.base.delay.ms";
        ComponentUtil.getSystemProperties().setProperty(key, "not-a-number");
        try {
            final GeminiEmbeddingClient realClient = new GeminiEmbeddingClient();
            final List<String> logs = captureDebugLogs(() -> assertEquals(2000L, realClient.getRetryBaseDelayMs()));
            // A WARN naming the bad value is only emitted from the NumberFormatException catch
            // block, so this distinguishes "the catch path was actually exercised" from "the
            // injection never reached the getter" - the latter would also return 2000L (the
            // untouched default) but log nothing, which is exactly how this assertion went
            // vacuous the first time (before this class read the same channel as the getter).
            //
            // The catch now lives in AbstractEmbeddingClient#getConfigLong, which logs under the
            // concrete client's logger (so this capture still sees it). Pin the two facts that
            // matter - the key and the rejected value - rather than the base class's wording.
            assertNotNull(findLog(logs, key), "the WARN should name the offending config key: " + logs);
            assertNotNull(findLog(logs, "not-a-number"),
                    "NumberFormatException catch path should log a WARN naming the bad configured value: " + logs);
        } finally {
            ComponentUtil.getSystemProperties().remove(key);
        }
    }

    // ========== backoff cap (computeBackoffMs) ==========

    @Test
    public void test_computeBackoffMs_capsHighAttempt() {
        // Without a cap, attempt >= ~15 with the default 2000ms base sleeps many minutes on a
        // single retry (2000 * 2^(attempt-1)); the cap must clamp every attempt to <= 60000ms.
        final long uncappedAttempt10 = (long) (2000L * Math.pow(2, 9)); // ~1,024,000ms
        assertTrue(uncappedAttempt10 > 60_000L, "sanity: uncapped attempt-10 delay would far exceed the cap");
        for (int attempt = 10; attempt <= 25; attempt++) {
            final long delay = GeminiEmbeddingClient.computeBackoffMs(attempt, 2000L);
            assertTrue(delay <= 60_000L, "backoff must be capped at 60000ms: attempt=" + attempt + ", delay=" + delay);
            assertTrue(delay >= 0L, "backoff must be non-negative: attempt=" + attempt + ", delay=" + delay);
        }
    }

    @Test
    public void test_computeBackoffMs_earlyAttemptUncapped() {
        // Early attempts stay well below the cap: attempt 1 is base ± 20% jitter.
        final long delay = GeminiEmbeddingClient.computeBackoffMs(1, 2000L);
        assertTrue(delay >= 1600L && delay <= 2400L, "attempt 1 should be ~base +/- 20%: " + delay);
    }

    // ========== base-URL trailing-slash normalization ==========

    @Test
    public void test_stripTrailingSlash() {
        assertEquals("https://host/v1beta", GeminiApiUrl.stripTrailingSlash("https://host/v1beta/"));
        assertEquals("https://host/v1beta", GeminiApiUrl.stripTrailingSlash("https://host/v1beta"));
        assertNull(GeminiApiUrl.stripTrailingSlash(null));
    }

    // ========== URL credential masking ==========

    @Test
    public void test_maskCredentialInUrl() {
        // Gemini's documented query-param auth alternative must never reach the logs.
        assertEquals("https://host/v1beta?key=***", CredentialUrlUtil.maskCredentialInUrl("https://host/v1beta?key=AIzaSecret"));
        assertEquals("https://host/v1beta?key=***&alt=json",
                CredentialUrlUtil.maskCredentialInUrl("https://host/v1beta?key=AIzaSecret&alt=json"));
        assertEquals("https://host/v1beta?alt=json&key=***",
                CredentialUrlUtil.maskCredentialInUrl("https://host/v1beta?alt=json&key=AIzaSecret"));
        assertEquals("https://host/v1beta?KEY=***", CredentialUrlUtil.maskCredentialInUrl("https://host/v1beta?KEY=AIzaSecret"));
        assertEquals("https://host/v1beta?api_key=***&token=***&access_token=***",
                CredentialUrlUtil.maskCredentialInUrl("https://host/v1beta?api_key=s1&token=s2&access_token=s3"));
        // Non-credential parameters and credential-free URLs pass through unchanged.
        assertEquals("https://host/v1beta?alt=json", CredentialUrlUtil.maskCredentialInUrl("https://host/v1beta?alt=json"));
        assertEquals("https://host/v1beta", CredentialUrlUtil.maskCredentialInUrl("https://host/v1beta"));
        assertNull(CredentialUrlUtil.maskCredentialInUrl(null));
    }

    /**
     * A gateway {@code api.url} may carry its credential as RFC 3986 userinfo rather than as a
     * query parameter; that form must be masked too, since these URLs are logged at WARN on every
     * gateway error.
     */
    @Test
    public void test_maskCredentialInUrl_userInfo() {
        assertEquals("https://***:***@gw.example.com/v1beta",
                CredentialUrlUtil.maskCredentialInUrl("https://user:pass@gw.example.com/v1beta"));
        assertEquals("http://***:***@gw.example.com:8080/v1beta",
                CredentialUrlUtil.maskCredentialInUrl("http://user:pass@gw.example.com:8080/v1beta"));
        // Both credential styles at once.
        assertEquals("https://***:***@gw.example.com/v1beta?key=***",
                CredentialUrlUtil.maskCredentialInUrl("https://user:pass@gw.example.com/v1beta?key=AIzaSecret"));
        // A host:port authority carries no userinfo and must survive untouched.
        assertEquals("http://localhost:8080/v1beta/models", CredentialUrlUtil.maskCredentialInUrl("http://localhost:8080/v1beta/models"));
        assertEquals("https://gw.example.com/v1beta", CredentialUrlUtil.maskCredentialInUrl("https://gw.example.com/v1beta"));
        // An '@' later in the path must not be mistaken for a userinfo delimiter.
        assertEquals("https://gw.example.com/v1beta/a:b@c", CredentialUrlUtil.maskCredentialInUrl("https://gw.example.com/v1beta/a:b@c"));
    }

    // ========== query-string-safe path composition ==========

    @Test
    public void test_appendPath() {
        // The query string must survive, with the path spliced in before it.
        assertEquals("https://host/v1beta/models?key=K", GeminiApiUrl.appendPath("https://host/v1beta?key=K", "/models"));
        assertEquals("https://host/v1beta/models?key=K&alt=json", GeminiApiUrl.appendPath("https://host/v1beta?key=K&alt=json", "/models"));
        // A trailing slash before the query must not yield a duplicate slash.
        assertEquals("https://host/v1beta/models?key=K", GeminiApiUrl.appendPath("https://host/v1beta/?key=K", "/models"));
        // Regressions of the previous stripTrailingSlash-only behavior.
        assertEquals("https://host/v1beta/models", GeminiApiUrl.appendPath("https://host/v1beta/", "/models"));
        assertEquals("https://host/v1beta/models", GeminiApiUrl.appendPath("https://host/v1beta", "/models"));
        assertEquals("https://host/v1beta/models/gemini-embedding-001:batchEmbedContents?key=K",
                GeminiApiUrl.appendPath("https://host/v1beta?key=K", "/models/gemini-embedding-001:batchEmbedContents"));
        // An empty query marker is preserved verbatim rather than being treated as a path segment.
        assertEquals("https://host/v1beta/models?", GeminiApiUrl.appendPath("https://host/v1beta?", "/models"));
        assertNull(GeminiApiUrl.appendPath(null, "/models"));
    }

    /**
     * A gateway/proxy {@code api.url} that carries its credential as a query string (Gemini's
     * documented {@code ?key=...} alternative, see {@link CredentialUrlUtil#maskCredentialInUrl(String)}'s
     * javadoc) must still compose a valid endpoint: the fixed path has to be inserted before the
     * query, not concatenated after it (which would push the whole path into the query string).
     */
    @Test
    public void test_embedDocuments_apiUrlWithQueryString_composesPathBeforeQuery() throws Exception {
        final String responseJson = "{\"embeddings\":[{\"values\":[0.1,0.2,0.3]}]}";
        server.enqueue(new MockResponse().setBody(responseJson).setHeader("Content-Type", "application/json"));
        setupClient();
        client.setTestApiUrl(client.getApiUrl() + "?key=SECRET");
        client.setTestDimension(3);

        client.embedDocuments(List.of("chunk"));

        assertEquals("/models/gemini-embedding-001:batchEmbedContents?key=SECRET", server.takeRequest().getPath());
    }

    /**
     * Same composition rule for the availability probe: with a query-bearing base URL the naive
     * concatenation produced {@code /?key=.../models}, which any real gateway answers with 404,
     * pinning {@link GeminiEmbeddingClient#isAvailable()} to false with only a DEBUG line as a clue.
     */
    @Test
    public void test_checkAvailabilityNow_apiUrlWithQueryString_composesPathBeforeQuery() throws Exception {
        server.enqueue(new MockResponse().setBody("{\"models\":[]}").setHeader("Content-Type", "application/json"));
        setupClient();
        client.setTestApiUrl(client.getApiUrl() + "?key=SECRET");

        assertTrue(client.isAvailable());
        assertEquals("/models?key=SECRET", server.takeRequest().getPath());
    }

    @Test
    public void test_getApiUrl_stripsTrailingSlash() {
        // Real getConfigString() -> getSystemProperty() read (see ComponentUtil.getSystemProperties()),
        // not the LastaFlute fess_config.properties channel.
        final String key = "content_chunker.embedding.gemini.api.url";
        ComponentUtil.getSystemProperties().setProperty(key, "https://example.com/v1beta/");
        try {
            final GeminiEmbeddingClient realClient = new GeminiEmbeddingClient();
            assertEquals("https://example.com/v1beta", realClient.getApiUrl());
        } finally {
            ComponentUtil.getSystemProperties().remove(key);
        }
    }

    // ========== checkAvailabilityNow() / isAvailable() ==========

    @Test
    public void test_checkAvailabilityNow_success() throws Exception {
        server.enqueue(new MockResponse().setBody("{\"models\":[]}").setHeader("Content-Type", "application/json"));
        setupClient();

        assertTrue(client.isAvailable());
    }

    @Test
    public void test_checkAvailabilityNow_blankApiKey_returnsFalseWithoutHttpCall() {
        client.setTestApiKey("");
        client.setTestApiUrl("https://generativelanguage.googleapis.com/v1beta");

        assertFalse(client.isAvailable());
        assertEquals("blank apiKey must short-circuit without an HTTP call", 0, server.getRequestCount());
    }

    // ========== availability-check log masking ==========

    /**
     * The availability probe logs the configured {@code apiUrl}; when a gateway endpoint carries a
     * credential as Gemini's documented {@code ?key=...} parameter, the failure branch - the one a
     * misconfigured gateway actually reaches - must not leak it.
     *
     * <p>That parameter is the only credential-in-URL form that still reaches this log: a
     * userinfo-bearing URL is now refused before the probe runs (see
     * {@link #test_checkAvailabilityNow_userInfoApiUrl_reportsUnavailableWithRemedy()}), so no log
     * statement on this branch ever renders one.
     */
    @Test
    public void test_checkAvailabilityNow_failureLogMasksCredentialInUrl() {
        setupClient();
        // Port 1 refuses immediately, so the probe throws and takes the error-logging branch.
        client.setTestApiUrl("http://127.0.0.1:1/v1beta?key=" + AVAILABILITY_SECRET);

        final List<String> logs = captureDebugLogs(() -> assertFalse(client.isAvailable()));

        final String line = findLog(logs, "Gemini is not available. url=");
        assertNotNull(line, "availability-failure debug log not captured: " + logs);
        assertTrue("query credential not masked: " + line, line.contains("key=***"));
        assertNoSecret(logs);
    }

    /**
     * The same failure branch also logs the exception message, and that is a second, independent
     * leak channel: a malformed {@code api.url} is rejected while the request is being built -
     * before anything is sent - and the resulting message quotes the offending URI verbatim.
     * Masking only the {@code url={}} placeholder therefore still leaks the credential through
     * {@code error={}}.
     */
    @Test
    public void test_checkAvailabilityNow_failureLogMasksCredentialInExceptionMessage() {
        setupClient();
        // '|' is illegal in a URI query, so HttpGet rejects the URI with the full URI in its message.
        client.setTestApiUrl("http://127.0.0.1:1/v1beta?key=" + AVAILABILITY_SECRET + "|x");

        final List<String> logs = captureDebugLogs(() -> assertFalse(client.isAvailable()));

        final String line = findLog(logs, "Gemini is not available. url=");
        assertNotNull(line, "availability-failure debug log not captured: " + logs);
        assertNoSecret(logs);
    }

    /**
     * The embed call builds its request URI from the configured {@code api.url} too, and a
     * malformed value is rejected there with the offending URI quoted in full. The failure branch
     * logs that exception both as an argument and as an attached throwable, and wraps it as the
     * cause of the {@link EmbeddingException} the caller receives, so masking has to happen where
     * the URI is built rather than at the log statement.
     */
    @Test
    public void test_embedDocuments_malformedApiUrlDoesNotLeakCredential() {
        setupClient();
        // '|' is illegal in a URI query, so building the request rejects the URI and quotes it.
        client.setTestApiUrl("http://127.0.0.1:1/v1beta?key=" + AVAILABILITY_SECRET + "|x");
        client.setTestDimension(3);
        final AtomicReference<Throwable> thrown = new AtomicReference<>();

        final List<String> logs = captureDebugLogs(() -> {
            try {
                client.embedDocuments(List.of("chunk"));
                fail("expected the malformed api.url to fail the call");
            } catch (final RuntimeException e) {
                thrown.set(e);
            }
        });

        assertNotNull(findLog(logs, "Failed to call Gemini embed API."), "failure log not captured: " + logs);
        assertNoSecret(logs);
        assertNoSecretInChain(thrown.get());
    }

    // ========== userinfo-bearing api.url rejection ==========

    /**
     * Credential embedded as RFC 3986 userinfo. Its value contains spaces on purpose: that is the
     * input class the masking regex cannot handle, and it is exactly what a refusal has to keep out
     * of every channel without relying on masking.
     */
    private static final String USERINFO_SECRET = "AIza Sy Leak Canary";

    /**
     * A configured endpoint whose authority carries userinfo. The host is deliberately
     * non-resolvable: a correct refusal never resolves it, so a test that accidentally lets the
     * request through fails on the request count rather than on a slow DNS lookup.
     */
    private static String userInfoApiUrl() {
        return "https://gemini:" + USERINFO_SECRET + "@gw.example.com:8443/v1beta";
    }

    /**
     * HttpClient rejects a request URI whose authority carries userinfo (RFC 9110 4.2.4), so such
     * an {@code api.url} can never issue a request. The client must therefore report itself
     * unavailable and say what to do instead, rather than fail opaquely on every probe.
     */
    @Test
    public void test_checkAvailabilityNow_userInfoApiUrl_reportsUnavailableWithRemedy() {
        setupClient();
        client.setTestApiUrl(userInfoApiUrl());

        final List<String> logs = captureDebugLogs(() -> assertFalse(client.isAvailable()));

        // assertNotNull here takes (actual, message) - the (message, actual) order silently binds
        // the message as the value under test and can never fail.
        assertNotNull(findLog(logs, "http.proxy.username"), "the supported alternative is not named in any log line: " + logs);
        assertNotNull(findLog(logs, "http.proxy.password"), "the supported alternative is not named in any log line: " + logs);
        assertNotNull(findLog(logs, "content_chunker.embedding.gemini.api.url"),
                "the offending setting is not named in any log line: " + logs);
        assertNoSecret(logs, USERINFO_SECRET);
        assertEquals("a userinfo-bearing api.url must not issue a request", 0, server.getRequestCount());
    }

    /**
     * The refusal must be a false return, never a throw. {@code init()} runs as a LastaDi
     * {@code postConstruct} and reaches this method synchronously through
     * {@code startAvailabilityCheck()} -&gt; {@code updateAvailability()}, neither of which catches
     * anything; an exception escaping here would abort container assembly and stop the server from
     * starting over one mistyped property.
     */
    @Test
    public void test_checkAvailabilityNow_userInfoApiUrl_failsClosedInsteadOfThrowing() {
        setupClient();
        client.setTestApiUrl(userInfoApiUrl());

        assertFalse(client.checkAvailabilityNow());
    }

    /**
     * The availability probe runs on a timer, so a per-call ERROR would fill the log with the same
     * line forever. It has to be reported once for as long as the configuration stays broken.
     */
    @Test
    public void test_checkAvailabilityNow_userInfoApiUrl_errorLoggedOncePerConfiguration() {
        setupClient();
        client.setTestApiUrl(userInfoApiUrl());

        final List<String> logs = captureDebugLogs(() -> {
            assertFalse(client.isAvailable());
            assertFalse(client.isAvailable());
            assertFalse(client.isAvailable());
        });

        assertEquals("the configuration error must be reported once, not once per call: " + logs, 1,
                countLogs(logs, "http.proxy.username"));
    }

    /**
     * A {@code host:port} authority also contains a colon but no {@code @}, so it is not userinfo.
     * The mock server URL is exactly that shape, and it must keep probing normally.
     */
    @Test
    public void test_checkAvailabilityNow_hostWithPortIsNotUserInfo() {
        server.enqueue(new MockResponse().setBody("{\"models\":[]}").setHeader("Content-Type", "application/json"));
        setupClient();

        final List<String> logs = captureDebugLogs(() -> assertTrue(client.isAvailable()));

        assertNull(findLog(logs, "http.proxy.username"), "a host:port authority must not be mistaken for userinfo: " + logs);
        assertEquals(1, server.getRequestCount());
    }

    /**
     * The refusal also has to reach the embed call, which builds its request URI from the same
     * setting: without it the call fails deep inside HttpClient and the failure log renders the raw
     * URL through a mask that whitespace in the credential defeats.
     */
    @Test
    public void test_embedDocuments_userInfoApiUrl_refusedBeforeAnyRequest() {
        setupClient();
        client.setTestApiUrl(userInfoApiUrl());
        client.setTestDimension(3);
        final AtomicReference<Throwable> thrown = new AtomicReference<>();

        final List<String> logs = captureDebugLogs(() -> {
            try {
                client.embedDocuments(List.of("chunk"));
                fail("expected the userinfo-bearing api.url to be refused");
            } catch (final RuntimeException e) {
                thrown.set(e);
            }
        });

        assertNoSecret(logs, USERINFO_SECRET);
        assertNoSecretInChain(thrown.get(), USERINFO_SECRET);
        assertNotNull(findInChain(thrown.get(), "http.proxy.username"), "the supported alternative is not named in the chain");
        assertNotNull(findInChain(thrown.get(), "content_chunker.embedding.gemini.api.url"),
                "the offending setting is not named in the chain");
        assertEquals("a userinfo-bearing api.url must not issue a request", 0, server.getRequestCount());
    }

    /** Distinctive credential value used by the availability-check log tests. */
    private static final String AVAILABILITY_SECRET = "AIzaSyLeakCanary";

    /** Fails if any captured log line contains the credential value. */
    private void assertNoSecret(final List<String> logs) {
        assertNoSecret(logs, AVAILABILITY_SECRET);
    }

    /** Same, for a test that uses its own credential value. */
    private void assertNoSecret(final List<String> logs, final String secret) {
        for (final String m : logs) {
            assertFalse("credential leaked into log: " + m, m.contains(secret));
        }
    }

    /** Returns the number of captured log lines containing {@code needle}. */
    private static int countLogs(final List<String> logs, final String needle) {
        int count = 0;
        for (final String m : logs) {
            if (m.contains(needle)) {
                count++;
            }
        }
        return count;
    }

    /** Returns the first message in the chain containing {@code needle}, or null if none. */
    private static String findInChain(final Throwable thrown, final String needle) {
        Throwable cause = thrown;
        for (int depth = 0; cause != null && depth < 16; cause = cause.getCause(), depth++) {
            if (cause.getMessage() != null && cause.getMessage().contains(needle)) {
                return cause.getMessage();
            }
        }
        return null;
    }

    /**
     * Renders a log event the way a layout would: the formatted message followed by the whole
     * throwable chain, which a stack trace prints in full.
     */
    private static String renderLogEvent(final LogEvent event) {
        final StringBuilder buf = new StringBuilder(event.getMessage().getFormattedMessage());
        Throwable thrown = event.getThrown();
        for (int depth = 0; thrown != null && depth < 16; thrown = thrown.getCause(), depth++) {
            buf.append(" | thrown: ").append(thrown.getClass().getName()).append(": ").append(thrown.getMessage());
        }
        return buf.toString();
    }

    /**
     * Fails if the credential value survives anywhere in a thrown exception's cause chain. This is
     * the channel that outlives the client: the exception is handed to callers and to every
     * upstream logger, which render the chain in full.
     */
    private void assertNoSecretInChain(final Throwable thrown) {
        assertNoSecretInChain(thrown, AVAILABILITY_SECRET);
    }

    /** Same, for a test that uses its own credential value. */
    private void assertNoSecretInChain(final Throwable thrown, final String secret) {
        assertTrue("no exception was thrown", thrown != null);
        Throwable cause = thrown;
        for (int depth = 0; cause != null && depth < 16; cause = cause.getCause(), depth++) {
            final String message = cause.getMessage();
            assertFalse("credential leaked through exception " + cause.getClass().getName() + ": " + message,
                    message != null && message.contains(secret));
        }
    }

    /** Returns the first captured log line containing {@code needle}, or null if none. */
    private static String findLog(final List<String> logs, final String needle) {
        for (final String m : logs) {
            if (m.contains(needle)) {
                return m;
            }
        }
        return null;
    }

    /**
     * Runs {@code action} with the GeminiEmbeddingClient logger forced to DEBUG and a capturing
     * appender attached, restoring both afterwards so the logger configuration is left untouched
     * for subsequent tests.
     */
    private List<String> captureDebugLogs(final Runnable action) {
        final org.apache.logging.log4j.core.Logger coreLogger =
                (org.apache.logging.log4j.core.Logger) LogManager.getLogger(GeminiEmbeddingClient.class);
        final Level previousLevel = coreLogger.getLevel();
        final CapturingAppender appender = new CapturingAppender();
        appender.start();
        coreLogger.addAppender(appender);
        Configurator.setLevel(GeminiEmbeddingClient.class.getName(), Level.DEBUG);
        try {
            action.run();
        } finally {
            Configurator.setLevel(GeminiEmbeddingClient.class.getName(), previousLevel);
            coreLogger.removeAppender(appender);
            appender.stop();
        }
        return appender.snapshot();
    }

    /**
     * Log4j2 appender that records each event's formatted message <em>and</em> its throwable chain.
     *
     * <p>A layout renders the throwable as a stack trace next to the message, so an assertion that
     * only sees {@code getMessage().getFormattedMessage()} passes while the rendered log still
     * leaks whatever the exception carries. Recording both makes the throwable channel visible to
     * every capture-based assertion in this class.
     */
    private static final class CapturingAppender extends AbstractAppender {
        private final List<String> messages = Collections.synchronizedList(new ArrayList<>());

        CapturingAppender() {
            super("CaptureGeminiEmbeddingLogs-" + System.nanoTime(), null, null, true, Property.EMPTY_ARRAY);
        }

        @Override
        public void append(final LogEvent event) {
            messages.add(renderLogEvent(event));
        }

        List<String> snapshot() {
            synchronized (messages) {
                return new ArrayList<>(messages);
            }
        }
    }

    private void setupClient() {
        final String baseUrl = server.url("").toString();
        final String apiUrl = baseUrl.endsWith("/") ? baseUrl.substring(0, baseUrl.length() - 1) : baseUrl;
        client.setTestApiUrl(apiUrl);
        client.setTestApiKey("test-key");
        client.setTestModel("gemini-embedding-001");
        client.setTestTimeout(30000);
        client.init();
    }

    /**
     * Builds a {@code batchEmbedContents} response body of {@code count} vectors of the given
     * {@code dimension} (which must be at least 2). Component 0 is always {@code 1} and component 1
     * carries the global input index ({@code startIndex + i}); the remaining components are zero.
     * The index therefore rides in the vector's <em>direction</em> (the ratio of component 1 to
     * component 0) rather than its magnitude, so it survives the L2 normalization applied to
     * MRL-truncated vectors and still lets a test assert both the count and the concatenation order
     * of a multi-sub-batch reassembly.
     */
    private static MockResponse embeddingsResponse(final int startIndex, final int count, final int dimension) {
        final StringBuilder sb = new StringBuilder("{\"embeddings\":[");
        for (int i = 0; i < count; i++) {
            if (i > 0) {
                sb.append(',');
            }
            sb.append("{\"values\":[");
            for (int d = 0; d < dimension; d++) {
                if (d > 0) {
                    sb.append(',');
                }
                sb.append(switch (d) {
                case 0 -> 1;
                case 1 -> startIndex + i;
                default -> 0;
                });
            }
            sb.append("]}");
        }
        sb.append("]}");
        return new MockResponse().setBody(sb.toString()).setHeader("Content-Type", "application/json");
    }

    /**
     * Asserts that a recorded {@code batchEmbedContents} request body carries exactly
     * {@code expected} nested embed requests, counted by the per-request
     * {@code "outputDimensionality"} field.
     */
    private void assertNestedRequestCount(final RecordedRequest request, final int expected) {
        final String body = request.getBody().readUtf8();
        int count = 0;
        int index = body.indexOf("\"outputDimensionality\"");
        while (index >= 0) {
            count++;
            index = body.indexOf("\"outputDimensionality\"", index + 1);
        }
        assertEquals("nested request count in sub-batch body: " + body, expected, count);
    }

    static class TestableGeminiEmbeddingClient extends GeminiEmbeddingClient {

        private String testApiKey = "test-key";
        private String testApiUrl = "https://generativelanguage.googleapis.com/v1beta";
        private String testModel = "gemini-embedding-001";
        private int testTimeout = 30000;
        private int testRetryMax = 3;
        private long testRetryBaseDelayMs = 2000L;
        private Integer testDimension = 768;
        private String testDocumentTaskType = DEFAULT_DOCUMENT_TASK_TYPE;
        private String testQueryTaskType = DEFAULT_QUERY_TASK_TYPE;

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

        void setTestRetryMax(final int max) {
            this.testRetryMax = max;
        }

        void setTestRetryBaseDelayMs(final long ms) {
            this.testRetryBaseDelayMs = ms;
        }

        void setTestDimension(final Integer dimension) {
            this.testDimension = dimension;
        }

        void setTestDocumentTaskType(final String taskType) {
            this.testDocumentTaskType = taskType;
        }

        void setTestQueryTaskType(final String taskType) {
            this.testQueryTaskType = taskType;
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

        @Override
        protected int getRetryMaxAttempts() {
            return testRetryMax;
        }

        @Override
        protected long getRetryBaseDelayMs() {
            return testRetryBaseDelayMs;
        }

        @Override
        public int getDimension() {
            if (testDimension == null) {
                throw new EmbeddingException("content_chunker.embedding.dimension is not configured");
            }
            return testDimension;
        }

        @Override
        protected String getDocumentTaskType() {
            return testDocumentTaskType;
        }

        @Override
        protected String getQueryTaskType() {
            return testQueryTaskType;
        }

        @Override
        protected int getAvailabilityCheckInterval() {
            return 0;
        }

        @Override
        protected boolean isContentChunkerEnabled() {
            return false;
        }

        @Override
        protected String getEmbeddingType() {
            return NAME;
        }
    }
}
