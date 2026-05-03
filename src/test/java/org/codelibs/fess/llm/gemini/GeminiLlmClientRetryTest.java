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
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import org.codelibs.fess.llm.LlmStreamCallback;
import org.codelibs.fess.unit.UnitFessTestCase;
import org.junit.jupiter.api.Test;

public class GeminiLlmClientRetryTest extends UnitFessTestCase {

    /**
     * Verifies that {@link GeminiLlmClient#executeWithRetry} invokes
     * {@link LlmStreamCallback#onRetry} once between every pair of attempts (i.e. n-1
     * times for n attempts) and supplies the expected operation label, attempt
     * counters, and a non-null cause.
     */
    @Test
    public void test_executeWithRetry_invokesOnRetryBetweenAttempts() throws Exception {
        final GeminiLlmClient client = new GeminiLlmClient() {
            @Override
            protected int getRetryMaxAttempts() {
                return 3;
            }

            @Override
            protected long getRetryBaseDelayMs() {
                return 0L; // skip real backoff sleep
            }
        };

        final AtomicInteger retryCount = new AtomicInteger(0);
        final List<Integer> seenAttempts = new ArrayList<>();
        final List<Throwable> seenCauses = new ArrayList<>();
        final LlmStreamCallback cb = new LlmStreamCallback() {
            @Override
            public void onChunk(final String chunk, final boolean done) {
                // not used in this test
            }

            @Override
            public void onRetry(final String op, final int attempt, final int max, final long sleepMs, final Throwable cause) {
                retryCount.incrementAndGet();
                seenAttempts.add(attempt);
                seenCauses.add(cause);
                assertEquals("test", op);
                assertEquals(3, max);
                assertTrue("sleepMs should be >= 0", sleepMs >= 0);
                assertNotNull(cause);
            }
        };

        final AtomicInteger attempts = new AtomicInteger(0);
        try {
            client.executeWithRetry("test", () -> {
                attempts.incrementAndGet();
                throw new GeminiLlmClient.RetryableHttpException(500, "boom");
            }, cb);
            fail("expected IOException after retries are exhausted");
        } catch (final IOException expected) {
            assertTrue("message should mention retryable error: " + expected.getMessage(), expected.getMessage().contains("retryable"));
        }

        assertEquals("HTTP call should run exactly maxAttempts times", 3, attempts.get());
        assertEquals("onRetry fires between attempts (n-1 times)", 2, retryCount.get());
        assertEquals(List.of(1, 2), seenAttempts);
        for (final Throwable cause : seenCauses) {
            assertTrue("cause should be RetryableHttpException, got: " + cause, cause instanceof GeminiLlmClient.RetryableHttpException);
        }
    }

    /**
     * Verifies that the 2-arg overload of {@code executeWithRetry} (used by the
     * non-streaming {@code chat()} path) still works when no callback is supplied.
     */
    @Test
    public void test_executeWithRetry_nullCallback_doesNotThrow() throws Exception {
        final GeminiLlmClient client = new GeminiLlmClient() {
            @Override
            protected int getRetryMaxAttempts() {
                return 2;
            }

            @Override
            protected long getRetryBaseDelayMs() {
                return 0L;
            }
        };

        final AtomicInteger attempts = new AtomicInteger(0);
        try {
            client.executeWithRetry("chat", () -> {
                attempts.incrementAndGet();
                throw new GeminiLlmClient.RetryableHttpException(503, "unavailable");
            });
            fail("expected IOException after retries are exhausted");
        } catch (final IOException expected) {
            // ok
        }
        assertEquals(2, attempts.get());
    }
}
