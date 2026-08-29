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

import java.util.List;

import org.codelibs.fess.unit.UnitFessTestCase;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInfo;

import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import okhttp3.mockwebserver.RecordedRequest;

/**
 * Query normalisation: {@code embedQuery} strips Fess/Lucene query syntax before
 * embedding, {@code embedDocuments} never does.
 *
 * <p>The two entry points carry different kinds of text. A document chunk is prose
 * that legitimately contains parentheses, quotation marks, colons and the word "AND";
 * a query, on the RAG path, is a Fess query string assembled by the intent step and
 * its operators are markup, not words. This is a separate axis from {@code taskType},
 * which continues to distinguish the two requests at the API level.</p>
 */
public class GeminiEmbeddingClientQueryTest extends UnitFessTestCase {

    private TestableClient client;
    private MockWebServer server;

    @Override
    public void setUp(final TestInfo testInfo) throws Exception {
        super.setUp(testInfo);
        client = new TestableClient();
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

    // -----------------------------------------------------------------------
    // toPlainQuery
    // -----------------------------------------------------------------------

    /**
     * The invariant that bounds this change's blast radius.
     *
     * <p>In fess 15.8.0 exactly two call sites reach {@code embedQuery}:
     * {@code SemanticChunkSearcher#search}, which calls it only after its own
     * {@code isPlainQuery(query)} returned true, and
     * {@code DefaultChatContentFetcher#resolveQueryVector}, which calls it with whatever
     * the intent step produced. Everything this method removes is something
     * {@code SemanticChunkSearcher.QUERY_SYNTAX_PATTERN} already rejects, so on that first
     * call site the transform is the identity and the semantic branch keeps embedding
     * byte-for-byte what it embedded before.</p>
     */
    @Test
    public void test_toPlainQuery_isIdentityForQueriesTheSemanticSearcherAccepts() {
        // Every string here passes SemanticChunkSearcher#isPlainQuery, so it is exactly the
        // population that reaches embedQuery from the semantic branch.
        final List<String> plain = List.of("自転車 変速 調整 方法", "珈琲 焙煎 温度 コーヒー豆", "bicycle derailleur adjustment", "天体観測 必要なもの 初心者 準備",
                "焙煎の温度はどのくらいですか", "gemini-embedding-001", "gemini-3.1-flash-lite-preview", "machine-learning 入門", "Fess", "検索エンジン");
        for (final String q : plain) {
            assertEquals(q, client.toPlainQuery(q));
        }
    }

    @Test
    public void test_toPlainQuery_removesRequiredTermPrefixes() {
        assertEquals("陶芸 釉薬", client.toPlainQuery("+陶芸 +釉薬"));
        assertEquals("Fess Docker", client.toPlainQuery("+Fess +Docker"));
        assertEquals("Fess Docker", client.toPlainQuery("+Fess -Docker"));
    }

    @Test
    public void test_toPlainQuery_removesQuotesAndGrouping() {
        assertEquals("養蜂 巣箱 管理 コツ 方法", client.toPlainQuery("+\"養蜂\" +\"巣箱\" (管理 OR コツ OR 方法)"));
        assertEquals("tutorial guide howto", client.toPlainQuery("(tutorial OR guide OR howto)"));
    }

    @Test
    public void test_toPlainQuery_removesFieldPrefixAndBoost() {
        // The field name is a schema name, not content: keeping "title" would add a term the
        // user never asked about.
        assertEquals("Fess", client.toPlainQuery("title:\"Fess\"^2"));
        assertEquals("大容量トークン検証用ドキュメント structure outline 節 セクション",
                client.toPlainQuery("title:\"大容量トークン検証用ドキュメント\" (structure OR outline OR 節 OR セクション)"));
    }

    @Test
    public void test_toPlainQuery_removesBooleanOperatorsAndRangeKeyword() {
        assertToPlain("Fess Docker", "Fess AND Docker");
        assertToPlain("Fess Docker", "Fess NOT Docker");
        assertToPlain("Fess Docker", "Fess && Docker");
        assertToPlain("Fess Docker", "Fess || Docker");
        assertToPlain("2020 2024", "[2020 TO 2024]");
    }

    @Test
    public void test_toPlainQuery_keepsHyphenAndPlusInsideATerm() {
        // Only a leading +/- is an operator. Stripping mid-token would corrupt real terms.
        assertEquals("gemini-embedding-001", client.toPlainQuery("gemini-embedding-001"));
        assertEquals("C++ 入門", client.toPlainQuery("C++ 入門"));
        assertEquals("e-mail アドレス", client.toPlainQuery("+e-mail アドレス"));
    }

    /**
     * A query made only of operators must not become an empty embedding input: Gemini
     * rejects a blank {@code parts[].text}, so the original string is embedded instead.
     * Degrading to the previous behaviour is strictly better than failing the whole chat.
     */
    @Test
    public void test_toPlainQuery_fallsBackToTheOriginalWhenNothingSurvives() {
        assertEquals("()", client.toPlainQuery("()"));
        assertEquals("AND OR", client.toPlainQuery("AND OR"));
        assertEquals("() AND OR", client.toPlainQuery("() AND OR"));
    }

    @Test
    public void test_toPlainQuery_passesNullAndBlankThrough() {
        assertNull(client.toPlainQuery(null));
        assertEquals("", client.toPlainQuery(""));
        assertEquals("   ", client.toPlainQuery("   "));
    }

    @Test
    public void test_toPlainQuery_collapsesTheWhitespaceItLeavesBehind() {
        // Removing an operator leaves a gap; a run of spaces would otherwise be embedded.
        assertEquals("陶芸 釉薬", client.toPlainQuery("+陶芸    +釉薬"));
    }

    // -----------------------------------------------------------------------
    // wire behaviour
    // -----------------------------------------------------------------------

    @Test
    public void test_embedQuery_sendsTheNormalisedText() throws Exception {
        enqueueOneVector();
        setupClient();

        client.embedQuery(List.of("+\"養蜂\" +\"巣箱\" (管理 OR コツ)"));

        assertEquals("養蜂 巣箱 管理 コツ", firstTextOf(server.takeRequest()));
    }

    @Test
    public void test_embedQuery_stillSendsTheQueryTaskType() throws Exception {
        enqueueOneVector();
        setupClient();

        client.embedQuery(List.of("+陶芸 +釉薬"));

        final String body = server.takeRequest().getBody().readUtf8();
        assertTrue(body.contains("\"taskType\":\"RETRIEVAL_QUERY\""), "query request should still carry RETRIEVAL_QUERY: " + body);
        assertTrue(body.contains("陶芸 釉薬"), "normalised text should reach the wire: " + body);
    }

    /**
     * Document text is prose. Removing its punctuation would change what is indexed, and
     * would do so asymmetrically from the query side, so {@code embedDocuments} must send
     * the text through untouched.
     */
    @Test
    public void test_embedDocuments_sendsTheTextUntouched() throws Exception {
        enqueueOneVector();
        setupClient();

        final String prose = "The AND gate (see figure 2) outputs \"1\" only when both inputs are 1.";
        client.embedDocuments(List.of(prose));

        assertEquals(prose, firstTextOf(server.takeRequest()));
    }

    /**
     * An empty input must keep reaching {@code callEmbedApi} unchanged, which short-circuits
     * to an empty list without an HTTP call. Mapping an empty list through the normaliser
     * first would be harmless but would make that contract accidental rather than explicit.
     */
    @Test
    public void test_embedQuery_emptyInputMakesNoCall() {
        setupClient();

        assertTrue(client.embedQuery(List.of()).isEmpty());
        assertEquals(0, server.getRequestCount());
    }

    // -----------------------------------------------------------------------
    // helpers
    // -----------------------------------------------------------------------

    private void assertToPlain(final String expected, final String input) {
        assertEquals(expected, client.toPlainQuery(input));
    }

    private void enqueueOneVector() {
        server.enqueue(
                new MockResponse().setBody("{\"embeddings\":[{\"values\":[0.1,0.2,0.3]}]}").setHeader("Content-Type", "application/json"));
    }

    /**
     * Reads the text of the first nested request. Gemini's {@code batchEmbedContents} body is
     * {@code {"requests":[{"content":{"parts":[{"text": ...}]}, ...}]}} - not a flat input array -
     * so the path has to be walked rather than matched on a substring.
     */
    private static String firstTextOf(final RecordedRequest request) throws Exception {
        final JsonNode body = new ObjectMapper().readTree(request.getBody().readUtf8());
        return body.get("requests").get(0).get("content").get("parts").get(0).get("text").asText();
    }

    private void setupClient() {
        final String baseUrl = server.url("").toString();
        client.setTestApiUrl(baseUrl.endsWith("/") ? baseUrl.substring(0, baseUrl.length() - 1) : baseUrl);
        client.init();
    }

    private static class TestableClient extends GeminiEmbeddingClient {
        private String testApiUrl = "https://generativelanguage.googleapis.com/v1beta";

        void setTestApiUrl(final String apiUrl) {
            this.testApiUrl = apiUrl;
        }

        @Override
        protected String getApiKey() {
            return "test-key";
        }

        @Override
        protected String getApiUrl() {
            return testApiUrl;
        }

        @Override
        protected String getModel() {
            return "gemini-embedding-001";
        }

        @Override
        protected int getTimeout() {
            return 30000;
        }

        @Override
        public int getDimension() {
            return 3;
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
            // Matches getName() so AbstractEmbeddingClient#init() actually builds the HTTP
            // client instead of skipping (the gate it uses in production to decide whether
            // this provider is the one currently selected).
            return NAME;
        }

        @Override
        protected String getDocumentTaskType() {
            return DEFAULT_DOCUMENT_TASK_TYPE;
        }

        @Override
        protected String getQueryTaskType() {
            return DEFAULT_QUERY_TASK_TYPE;
        }
    }
}
