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
package org.codelibs.fess.gemini;

import org.apache.hc.client5.http.classic.methods.HttpGet;
import org.apache.hc.client5.http.classic.methods.HttpPost;
import org.codelibs.core.lang.StringUtil;
import org.codelibs.fess.util.CredentialUrlUtil;

/**
 * Gemini-specific handling of the configured API URL: how an endpoint path is appended to it, how a
 * request is built from it, and what to tell an operator who embedded a credential in it.
 *
 * <p>Both the LLM client and the embedding client accept a user-configured endpoint
 * ({@code rag.llm.gemini.api.url} and {@code content_chunker.embedding.gemini.api.url}), so this
 * class is the single definition of those rules for both.
 *
 * <p>What counts as a credential inside a URL, and how one is kept out of a log, is <em>not</em>
 * defined here - that is provider-agnostic and lives in {@link CredentialUrlUtil}.
 *
 * @author FessProject
 */
public final class GeminiApiUrl {

    private GeminiApiUrl() {
        // utility class
    }

    /**
     * Builds the operator-facing explanation for a refused userinfo-bearing API URL.
     *
     * <p>The message carries no part of the URL: the credential is precisely what must not escape,
     * and {@link CredentialUrlUtil#maskCredentialInUrl(String)} cannot be trusted to remove it. What it does carry
     * is the setting to fix and the supported way to do what the operator was evidently trying to
     * do - authenticate, either to Gemini itself or to something in front of it.
     *
     * @param configKey the configuration property the URL was read from
     * @return the message to log and to attach to the refusal
     */
    public static String userInfoRejectionMessage(final String configKey) {
        return configKey + " must not embed credentials in the URL: an http/https target URI whose authority carries a userinfo"
                + " component (a 'user:password@' prefix before the host) is rejected outright by the HTTP client per RFC 9110"
                + " section 4.2.4, so no request can ever be issued. Remove that prefix. To authenticate to Gemini, set the"
                + " matching api.key property, which is sent as the x-goog-api-key header. To authenticate to an intervening"
                + " proxy, set http.proxy.host, http.proxy.port, http.proxy.username and http.proxy.password instead.";
    }

    /**
     * Strips a single trailing {@code /} from a base URL so callers can append a fixed path
     * without producing a duplicate slash.
     *
     * @param url the raw configured URL (may be {@code null})
     * @return the URL without a single trailing slash, or the input unchanged when null or slashless
     */
    public static String stripTrailingSlash(final String url) {
        if (url != null && url.endsWith("/")) {
            return url.substring(0, url.length() - 1);
        }
        return url;
    }

    /**
     * Appends a fixed endpoint path to a configured base URL, preserving any query string the base
     * URL already carries. Gemini documents {@code ?key=...} as an alternative to the
     * {@code x-goog-api-key} header, and the configured {@code api.url} may point at a proxy or
     * gateway that embeds credentials or routing parameters the same way; plain concatenation would
     * push the whole endpoint path into that query string (producing e.g.
     * {@code https://host/v1beta?key=K/models}), which any gateway answers with a {@code 404}. A
     * single trailing {@code /} on the base path is dropped so the result never contains a duplicate
     * slash.
     *
     * @param baseUrl the configured base URL, optionally with a query string (may be {@code null})
     * @param path the endpoint path to append, starting with {@code /}
     * @return the composed URL with {@code path} inserted before any query string, or {@code null}
     *         when {@code baseUrl} is null
     */
    public static String appendPath(final String baseUrl, final String path) {
        if (baseUrl == null) {
            return null;
        }
        final int queryIndex = baseUrl.indexOf('?');
        final String base = queryIndex < 0 ? baseUrl : baseUrl.substring(0, queryIndex);
        final String query = queryIndex < 0 ? StringUtil.EMPTY : baseUrl.substring(queryIndex);
        return stripTrailingSlash(base) + path + query;
    }

    /**
     * Creates a {@link HttpGet} for {@code url}, replacing the URI-rejection exception with one
     * that carries no part of the URL (see
     * {@link CredentialUrlUtil#invalidUrlException(String, IllegalArgumentException)}).
     *
     * @param url the request URL
     * @param configKey the configuration key {@code url} was read from, named in the replacement
     *            exception
     * @return the request
     */
    public static HttpGet createGet(final String url, final String configKey) {
        try {
            return new HttpGet(url);
        } catch (final IllegalArgumentException e) {
            throw CredentialUrlUtil.invalidUrlException(configKey, e);
        }
    }

    /**
     * Creates a {@link HttpPost} for {@code url}, replacing the URI-rejection exception with one
     * that carries no part of the URL (see
     * {@link CredentialUrlUtil#invalidUrlException(String, IllegalArgumentException)}).
     *
     * @param url the request URL
     * @param configKey the configuration key {@code url} was read from, named in the replacement
     *            exception
     * @return the request
     */
    public static HttpPost createPost(final String url, final String configKey) {
        try {
            return new HttpPost(url);
        } catch (final IllegalArgumentException e) {
            throw CredentialUrlUtil.invalidUrlException(configKey, e);
        }
    }
}
