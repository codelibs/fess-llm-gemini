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

import org.codelibs.fess.unit.UnitFessTestCase;
import org.junit.jupiter.api.Test;

/**
 * Pins the configuration channel of all three thinking-capability overrides
 * ({@code thinking.level.enabled}, {@code thinking.headroom.enabled},
 * {@code thinking.minimal.enabled}): {@code getOrDefault}, i.e.
 * {@code fess_config.properties} plus {@code -Dfess.config.*} - the same channel as every other
 * {@code rag.llm.gemini.*} key, including {@code rag.llm.gemini.model} and {@code api.url}. The
 * other candidate channel, {@code ComponentUtil.getFessConfig().getSystemProperty(key, default)},
 * reads {@code conf/system.properties} plus {@code -Dfess.system.*} instead, and is what
 * {@code AbstractEmbeddingClient#getConfigString} uses.
 *
 * <p>This is not a hypothetical mix-up: this plugin also ships a {@code GeminiEmbeddingClient}
 * built on {@code AbstractEmbeddingClient#getConfigString}, which reads that other,
 * {@code getSystemProperty}-backed channel, so the module already carries two same-named
 * {@code getConfigString} methods reading two different config stores. If
 * {@link GeminiLlmClient#getConfigString(String, String)} is later "deduplicated" by delegating
 * to the embedding client's version, the thinking-capability properties would silently stop
 * responding to {@code fess_config.properties} - a value there would simply be ignored, with no
 * error and no log line, while {@code conf/system.properties} would take over unannounced.
 * {@link GeminiLlmClientTest} cannot catch this: its {@code TestableGeminiLlmClient} overrides
 * {@code getConfigString} wholesale against an in-memory map, so it never touches either real
 * config store. This class is the only one that reads the property through the real
 * {@code FessConfig} component, and so the only tripwire for this specific regression.
 *
 * <p>{@link #isUseOneTimeContainer()} returns {@code true} because a value read through the real
 * {@code FessConfig} is memoized for the lifetime of the DI container, and this is the only test
 * class in the module that plants such a value via a system property. Without the override, a
 * value planted here would leak into whichever test class the runner happens to execute next,
 * making that other class's outcome depend on run order.
 */
public class GeminiLlmClientCapabilityConfigTest extends UnitFessTestCase {

    private static final String KEY = "fess.config.rag.llm.gemini.thinking.level.enabled";
    private static final String HEADROOM_KEY = "fess.config.rag.llm.gemini.thinking.headroom.enabled";
    private static final String MINIMAL_KEY = "fess.config.rag.llm.gemini.thinking.minimal.enabled";

    /**
     * Probe suffixes for the {@code getConfigString} key-composition test. Deliberately not the
     * capability key under test: a value read through the real FessConfig is memoized for the
     * lifetime of the container, so planting one under {@code thinking.level.enabled} would decide
     * the other tests in this class.
     */
    private static final String ABSENT_SUFFIX = "capability.probe.absent";
    private static final String PLANTED_SUFFIX = "capability.probe.planted";
    private static final String PLANTED_KEY = "fess.config.rag.llm.gemini." + PLANTED_SUFFIX;

    @Override
    protected boolean isUseOneTimeContainer() {
        return true;
    }

    @Test
    public void test_getConfigString_composesTheKeyAndReadsTheFessConfigChannel() {
        // The real getConfigString, not the test stub: GeminiLlmClientTest's
        // TestableGeminiLlmClient overrides it against a HashMap, so nothing there can tell
        // whether production composes the key at all.
        final GeminiLlmClient client = new GeminiLlmClient();
        // Two-arg form deliberately: a three-argument all-String assertEquals binds to
        // (message, expected, actual) and would silently compare the wrong pair.
        // An absent key must yield the supplied default.
        assertEquals("fallback", client.getConfigString(ABSENT_SUFFIX, "fallback"));
        System.setProperty(PLANTED_KEY, "planted");
        try {
            // Planted under getConfigPrefix() + "." + keySuffix; reading it back proves the
            // composition, not just that some lookup happened.
            assertEquals("planted", client.getConfigString(PLANTED_SUFFIX, "fallback"));
        } finally {
            System.clearProperty(PLANTED_KEY);
        }
    }

    @Test
    public void test_thinkingLevelEnabled_readFromTheFessConfigChannel() {
        System.setProperty(KEY, "true");
        try {
            final GeminiLlmClient client = new GeminiLlmClient();
            // "gemini-4-flash" is a generation the isGemini3 name rule does not recognise: name
            // inference alone would classify it false. Forcing the override through
            // fess_config.properties must still flip it true.
            assertTrue("a model name the isGemini3 rule rejects must be classified as using thinkingLevel when forced through fess_config",
                    client.usesThinkingLevel("gemini-4-flash"));
            assertEquals(Boolean.TRUE, client.getCapabilityOverride("thinking.level.enabled"));
        } finally {
            System.clearProperty(KEY);
        }
    }

    @Test
    public void test_thinkingLevelEnabled_unsetLeavesNameInference() {
        final GeminiLlmClient client = new GeminiLlmClient();
        assertNull(client.getCapabilityOverride("thinking.level.enabled"), "unset must resolve to auto (null)");
        assertFalse(client.usesThinkingLevel("gemini-4-flash"));
        assertTrue(client.usesThinkingLevel("gemini-3-flash"));
    }

    @Test
    public void test_headroomAndMinimalEnabled_readFromTheFessConfigChannel() {
        // The other two capability keys go through the same getConfigString, so they would move
        // channel together with thinking.level.enabled. Both are forced against their auto
        // derivation here, so a value that failed to arrive would leave the derived answer
        // visible and fail the assertion.
        System.setProperty(HEADROOM_KEY, "false");
        System.setProperty(MINIMAL_KEY, "true");
        try {
            final GeminiLlmClient client = new GeminiLlmClient();
            assertEquals(Boolean.FALSE, client.getCapabilityOverride("thinking.headroom.enabled"));
            assertFalse("a forced false must beat the Gemini 3 derivation", client.usesThinkingHeadroom("gemini-3-flash"));
            assertEquals(Boolean.TRUE, client.getCapabilityOverride("thinking.minimal.enabled"));
            assertTrue("a forced true must beat the Pro derivation", client.supportsMinimalThinking("gemini-3-pro"));
        } finally {
            System.clearProperty(HEADROOM_KEY);
            System.clearProperty(MINIMAL_KEY);
        }
    }
}
