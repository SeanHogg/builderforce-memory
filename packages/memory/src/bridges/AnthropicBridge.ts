/**
 * AnthropicBridge – TransformerBridge implementation for the Anthropic Messages API.
 *
 * Uses the /v1/messages endpoint.  System prompts are passed as the top-level
 * `system` field (not a message role), per the Anthropic spec.
 */

import { SSMError } from '../errors/SSMError.js';
import type { TransformerBridge, BridgeGenerateOptions } from './TransformerBridge.js';

export interface AnthropicBridgeOptions {
    /** Anthropic API key. */
    apiKey        : string;
    /**
     * Model to use. Default: 'claude-haiku-4-5' (cheapest current model:
     * $1/1M input, $5/1M output). The previous default `claude-3-5-haiku-*`
     * was retired on 2026-02-19 and now 404s.
     */
    model?        : string;
    /** Anthropic API version header. Default: '2023-06-01'. */
    apiVersion?   : string;
    /** Default system prompt. Default: none. */
    systemPrompt? : string;
    /** Default max tokens — required by Anthropic. Default: 1024. */
    maxTokens?    : number;
    /**
     * When true (default), the system prompt is sent as a cacheable content
     * block (`cache_control: {type: 'ephemeral'}`). Prompt caching bills cache
     * reads at ~10% of the input price, so a stable system prefix reused across
     * turns is up to ~90% cheaper on its input tokens. Caching only engages once
     * the cached prefix exceeds the model minimum (~4096 tokens for Haiku 4.5);
     * below that it is a silent no-op, never an error. Set false to opt out.
     */
    cacheSystem?  : boolean;
}

const API_URL = 'https://api.anthropic.com/v1/messages';

export class AnthropicBridge implements TransformerBridge {
    readonly supportsStreaming = true as const;

    private readonly _apiKey      : string;
    private readonly _model       : string;
    private readonly _apiVersion  : string;
    private readonly _systemPrompt: string;
    private readonly _maxTokens   : number;
    private readonly _cacheSystem : boolean;

    constructor(opts: AnthropicBridgeOptions) {
        this._apiKey       = opts.apiKey;
        this._model        = opts.model      ?? 'claude-haiku-4-5';
        this._apiVersion   = opts.apiVersion ?? '2023-06-01';
        this._systemPrompt = opts.systemPrompt ?? '';
        this._maxTokens    = opts.maxTokens    ?? 1024;
        this._cacheSystem  = opts.cacheSystem  ?? true;
    }

    async generate(prompt: string, opts: BridgeGenerateOptions = {}): Promise<string> {
        const body = this._buildBody(prompt, opts, false);
        const res  = await this._fetch(body);

        if (!res.ok) {
            const text = await res.text().catch(() => '');
            throw new SSMError(
                'BRIDGE_REQUEST_FAILED',
                `Anthropic API returned ${res.status}: ${text}`,
            );
        }

        const json    = await res.json() as Record<string, unknown>;
        const content = (json as any).content?.[0]?.text;
        if (typeof content !== 'string') {
            throw new SSMError('BRIDGE_RESPONSE_INVALID', 'Unexpected Anthropic response shape.');
        }
        return content;
    }

    async *stream(prompt: string, opts: BridgeGenerateOptions = {}): AsyncIterable<string> {
        const body = this._buildBody(prompt, opts, true);
        const res  = await this._fetch(body);

        if (!res.ok) {
            const text = await res.text().catch(() => '');
            throw new SSMError(
                'BRIDGE_REQUEST_FAILED',
                `Anthropic streaming API returned ${res.status}: ${text}`,
            );
        }

        if (!res.body) {
            throw new SSMError('BRIDGE_RESPONSE_INVALID', 'Anthropic streaming response has no body.');
        }

        yield* parseAnthropicStream(res.body);
    }

    private _buildBody(prompt: string, opts: BridgeGenerateOptions, stream: boolean): string {
        const sys = opts.systemPrompt ?? this._systemPrompt;
        const body: Record<string, unknown> = {
            model     : opts.model     ?? this._model,
            max_tokens: opts.maxTokens ?? this._maxTokens,
            messages  : [{ role: 'user', content: prompt }],
        };
        if (sys) {
            // Caching is a prefix match: render the stable system prompt as a
            // single cache-marked content block so reads on subsequent turns are
            // billed at ~10% of input price. The volatile user message is sent
            // unmarked after it, so it never enters the cached prefix.
            body['system'] = this._cacheSystem
                ? [{ type: 'text', text: sys, cache_control: { type: 'ephemeral' } }]
                : sys;
        }
        if (stream) body['stream'] = true;
        return JSON.stringify(body);
    }

    private _fetch(body: string): Promise<Response> {
        return fetch(API_URL, {
            method : 'POST',
            headers: {
                'Content-Type'      : 'application/json',
                'x-api-key'         : this._apiKey,
                'anthropic-version' : this._apiVersion,
            },
            body,
        });
    }
}

// ── SSE parser (Anthropic event format) ──────────────────────────────────────

async function* parseAnthropicStream(body: ReadableStream<Uint8Array>): AsyncIterable<string> {
    const reader  = body.getReader();
    const decoder = new TextDecoder();
    let buffer    = '';

    try {
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() ?? '';

            for (const line of lines) {
                const trimmed = line.trim();
                if (!trimmed.startsWith('data: ')) continue;

                const data = trimmed.slice(6);
                try {
                    const event = JSON.parse(data) as Record<string, unknown>;
                    // content_block_delta events carry the streamed text
                    if (event['type'] === 'content_block_delta') {
                        const text = (event as any).delta?.text;
                        if (typeof text === 'string' && text.length > 0) yield text;
                    }
                } catch {
                    // Skip malformed SSE lines
                }
            }
        }
    } finally {
        reader.releaseLock();
    }
}
