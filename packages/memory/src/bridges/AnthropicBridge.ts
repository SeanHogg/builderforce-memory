/**
 * AnthropicBridge – TransformerBridge implementation for the Anthropic Messages API.
 *
 * Uses the /v1/messages endpoint.  System prompts are passed as the top-level
 * `system` field (not a message role), per the Anthropic spec.
 */

import { SSMError } from '../errors/SSMError.js';
import type { LlmUsage } from '../telemetry/types.js';
import type { TransformerBridge, BridgeGenerateOptions, BridgeCallInfo } from './TransformerBridge.js';
import { readSseDataFrames, readUsageFields, AnthropicStreamUsage } from '../wire/index.js';

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

    private _lastCall: BridgeCallInfo | undefined;

    constructor(opts: AnthropicBridgeOptions) {
        this._apiKey       = opts.apiKey;
        this._model        = opts.model      ?? 'claude-haiku-4-5';
        this._apiVersion   = opts.apiVersion ?? '2023-06-01';
        this._systemPrompt = opts.systemPrompt ?? '';
        this._maxTokens    = opts.maxTokens    ?? 1024;
        this._cacheSystem  = opts.cacheSystem  ?? true;
    }

    /**
     * Provider-reported usage for the last call. Anthropic splits input tokens
     * three ways — fresh, cache read, and cache write — and each is billed at a
     * different rate, so a cost model that only reads `input_tokens` is wrong on
     * exactly the cache-heavy traffic this bridge is tuned to produce.
     */
    get lastCall(): BridgeCallInfo | undefined {
        return this._lastCall;
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
        this._lastCall = {
            usage: readAnthropicUsage(
                (json as any).usage,
                (json as any).model ?? opts.model ?? this._model,
            ),
        };
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

        // `message_start` carries the input split; `message_delta` the running
        // output count. Both are folded so the final usage is provider-measured.
        const usage = new AnthropicStreamUsage();
        for await (const frame of readSseDataFrames(res.body)) {
            usage.observe(frame);
            const ev = frame as { type?: unknown; delta?: { text?: unknown } };
            // content_block_delta events carry the streamed text
            if (ev.type === 'content_block_delta' && typeof ev.delta?.text === 'string' && ev.delta.text.length > 0) {
                yield ev.delta.text;
            }
        }

        const totals = usage.result();
        this._lastCall = {
            usage: {
                model: totals.model ?? opts.model ?? this._model,
                inputTokens: totals.inputTokens,
                outputTokens: totals.outputTokens,
                cachedInputTokens: totals.cacheReadTokens,
                cacheWriteTokens: totals.cacheCreationTokens,
            },
        };
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

/**
 * Maps an Anthropic `usage` object onto the canonical {@link LlmUsage} shape.
 * Absent when the response omits usage (an older gateway shim), in which case
 * `InstrumentedBridge` falls back to an estimate and flags it as such.
 */
function readAnthropicUsage(usage: unknown, model: string): LlmUsage | undefined {
    if (!usage || typeof usage !== 'object') return undefined;
    // Anthropic's `input_tokens` already excludes both cache counts.
    const u = readUsageFields(usage);
    return {
        model,
        inputTokens: u.promptTokens ?? 0,
        outputTokens: u.completionTokens ?? 0,
        cachedInputTokens: u.cacheReadTokens ?? 0,
        cacheWriteTokens: u.cacheCreationTokens ?? 0,
    };
}
