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

import org.codelibs.fess.unit.UnitFessTestCase;
import org.codelibs.fess.util.ComponentUtil;
import org.junit.jupiter.api.Test;

/**
 * Proves that every {@code content_chunker.embedding.gemini.*} config-read on
 * {@link GeminiEmbeddingClient} resolves through {@code FessConfigImpl.getSystemProperty}
 * (i.e. {@code conf/system.properties} / {@code -Dfess.system.<key>}, surfaced live in
 * System Info > Config Info > App Properties), not the LastaFlute
 * {@code fess_config.properties} / {@code ObjectiveConfig} channel loaded once at boot.
 *
 * <p>Every test here uses a plain {@code new GeminiEmbeddingClient()} (no subclass) and
 * injects a non-default value directly into the real {@code ComponentUtil.getSystemProperties()}
 * test double (see {@code test_app.xml}'s {@code systemProperties} component,
 * {@link org.codelibs.fess.unit.TestSystemProperties}). None of these would pass if the
 * corresponding production getter still read {@code ComponentUtil.getFessConfig().getOrDefault(...)}:
 * that channel has no per-test injection hook here, so an {@code getOrDefault}-backed getter
 * would fall straight through to its hardcoded default and every assertion below would fail.
 *
 * <p>This class overrides {@link #isUseOneTimeContainer()} because it mutates the
 * {@code systemProperties} component, which is otherwise a JVM-lifetime singleton shared
 * across test classes; without a fresh container per test, a value injected here could leak
 * into (or be corrupted by) another test class's assertions.
 */
public class GeminiEmbeddingClientConfigChannelTest extends UnitFessTestCase {

    @Override
    protected boolean isUseOneTimeContainer() {
        return true;
    }

    @Test
    public void test_getApiKey_readsFromSystemProperties() {
        final String key = "content_chunker.embedding.gemini.api.key";
        ComponentUtil.getSystemProperties().setProperty(key, "configured-api-key");
        try {
            assertEquals("configured-api-key", new GeminiEmbeddingClient().getApiKey());
        } finally {
            ComponentUtil.getSystemProperties().remove(key);
        }
    }

    @Test
    public void test_getApiUrl_readsFromSystemProperties() {
        final String key = "content_chunker.embedding.gemini.api.url";
        ComponentUtil.getSystemProperties().setProperty(key, "https://configured.example.com/v1beta");
        try {
            assertEquals("https://configured.example.com/v1beta", new GeminiEmbeddingClient().getApiUrl());
        } finally {
            ComponentUtil.getSystemProperties().remove(key);
        }
    }

    @Test
    public void test_getModel_readsFromSystemProperties() {
        final String key = "content_chunker.embedding.gemini.model";
        ComponentUtil.getSystemProperties().setProperty(key, "text-embedding-005");
        try {
            assertEquals("text-embedding-005", new GeminiEmbeddingClient().getModel());
        } finally {
            ComponentUtil.getSystemProperties().remove(key);
        }
    }

    @Test
    public void test_getDocumentTaskType_readsFromSystemProperties() {
        final String key = "content_chunker.embedding.gemini.document.task_type";
        ComponentUtil.getSystemProperties().setProperty(key, "CUSTOM_DOCUMENT_TASK_TYPE");
        try {
            assertEquals("CUSTOM_DOCUMENT_TASK_TYPE", new GeminiEmbeddingClient().getDocumentTaskType());
        } finally {
            ComponentUtil.getSystemProperties().remove(key);
        }
    }

    @Test
    public void test_getQueryTaskType_readsFromSystemProperties() {
        final String key = "content_chunker.embedding.gemini.query.task_type";
        ComponentUtil.getSystemProperties().setProperty(key, "CUSTOM_QUERY_TASK_TYPE");
        try {
            assertEquals("CUSTOM_QUERY_TASK_TYPE", new GeminiEmbeddingClient().getQueryTaskType());
        } finally {
            ComponentUtil.getSystemProperties().remove(key);
        }
    }

    @Test
    public void test_getRetryBaseDelayMs_readsFromSystemProperties() {
        final String key = "content_chunker.embedding.gemini.retry.base.delay.ms";
        ComponentUtil.getSystemProperties().setProperty(key, "12345");
        try {
            assertEquals(12345L, new GeminiEmbeddingClient().getRetryBaseDelayMs());
        } finally {
            ComponentUtil.getSystemProperties().remove(key);
        }
    }
}
