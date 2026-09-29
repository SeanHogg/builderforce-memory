/**
 * OpenAIBridge – TransformerBridge implementation for the OpenAI Chat API.
 *
 * Supports both non-streaming and streaming (SSE) completions.
 * Compatible with any OpenAI-compatible endpoint via the `baseUrl` option.
 *
 * The request, error mapping and stream reading are the `/wire` chat client's — the
 * same one a host uses for tool-calling turns — so this class only supplies the
 * bridge's defaults and maps usage onto {@link LlmUsage}.
 */

import { SSMError } from '../errors/SSMError.js';
import type { LlmUsage } from '../telemetry/types.js';
import type { TransformerBridge, BridgeGenerateOptions, BridgeCallInfo } from './TransformerBridge.js';
import {
    chatComplete,
    chatStream,
    ChatCompletionError,
    readUsageFields,
    type ChatClient,
    type ChatMessage,
    type ChatRequest,
    type ChatStreamEvent,
} from '../wire/index.js';

export interface OpenAIBridgeOptions {
    /** OpenAI API key (or compatible service key). */
    apiKey        : string;
    /** Model to use. Default: 'gpt-4o-mini'. */
    model?        : string;
    /** API base URL. Default: 'https://api.openai.com/v1'. */
    baseUrl?      : string;
    /** Default system prompt sent with every request. */
    systemPrompt? : string;
    /** Default max tokens. Default: 512. */
    maxTokens?    : number;
    /**
     * Send only what a call states — no model, max-tokens, temperature or top-p
     * defaults — so a routing gateway picks them. Default false.
     */
    serverDefaults?: boolean;
    /** Abort a request after this long. */
    timeoutMs?    : number;
}

export class OpenAIBridge implements TransformerBridge {
    private _lastCall: BridgeCallInfo | undefined;

    readonly supportsStreaming = true as const;

    private readonly _client      : ChatClient;
    private readonly _model       : string | undefined;
    private readonly _systemPrompt: string;
    private readonly _maxTokens   : number | undefined;
    private readonly _serverDefaults: boolean;

    constructor(opts: OpenAIBridgeOptions) {
        this._serverDefaults = opts.serverDefaults ?? false;
        this._client = {
            apiKey: opts.apiKey,
            baseUrl: opts.baseUrl ?? 'https://api.openai.com/v1',
            ...(opts.timeoutMs ? { timeoutMs: opts.timeoutMs } : {}),
        };
        this._model        = opts.model ?? (this._serverDefaults ? undefined : 'gpt-4o-mini');
        this._systemPrompt = opts.systemPrompt ?? '';
        this._maxTokens    = opts.maxTokens ?? (this._serverDefaults ? undefined : 512);
    }

    /** Provider-reported usage for the last call — see {@link BridgeCallInfo}. */
    get lastCall(): BridgeCallInfo | undefined {
        return this._lastCall;
    }

    async generate(prompt: string, opts: BridgeGenerateOptions = {}): Promise<string> {
        let result;
        try {
            result = await chatComplete(this._client, this._request(prompt, opts));
        } catch (err) {
            throw this._requestError('OpenAI API', err);
        }
        if (!result.hasTextContent) {
            throw new SSMError('BRIDGE_RESPONSE_INVALID', 'Unexpected OpenAI response shape.');
        }
        this._lastCall = {
            usage: readOpenAIUsage(result.usage, result.model ?? opts.model ?? this._model ?? ''),
        };
        return result.content;
    }

    async *stream(prompt: string, opts: BridgeGenerateOptions = {}): AsyncIterable<string> {
        let events;
        try {
            events = chatStream(this._client, { ...this._request(prompt, opts), includeUsage: true });
            // The request is sent on the first pull; surface a failed request as a bridge error.
            const first = await events.next();
            if (first.done) return;
            yield* this._text(first.value);
        } catch (err) {
            throw this._requestError('OpenAI streaming API', err);
        }
        for await (const event of events) yield* this._text(event);
    }

    /** Text deltas out; the closing event records usage (sent because of `includeUsage`). */
    private *_text(event: ChatStreamEvent): Generator<string> {
        if (event.type === 'text-delta') yield event.delta;
        else if (event.type === 'done') {
            const usage = readOpenAIUsage(event.result.usage, event.result.model ?? this._model ?? '');
            this._lastCall = usage ? { usage } : {};
        }
    }

    private _request(prompt: string, opts: BridgeGenerateOptions): ChatRequest {
        const sys = opts.systemPrompt ?? this._systemPrompt;
        const messages: ChatMessage[] = [];
        if (sys) messages.push({ role: 'system', content: sys });
        messages.push({ role: 'user', content: prompt });
        const d = !this._serverDefaults;
        const model = opts.model ?? this._model;
        const maxTokens = opts.maxTokens ?? this._maxTokens;
        const temperature = opts.temperature ?? (d ? 0.7 : undefined);
        const topP = opts.topP ?? (d ? 0.9 : undefined);
        return {
            messages,
            ...(model ? { model } : {}),
            ...(maxTokens !== undefined ? { maxTokens } : {}),
            ...(temperature !== undefined ? { temperature } : {}),
            ...(topP !== undefined ? { topP } : {}),
            ...(opts.extra ? { extra: opts.extra } : {}),
        };
    }

    private _requestError(what: string, err: unknown): SSMError {
        if (err instanceof SSMError) return err;
        if (err instanceof ChatCompletionError) {
            // A 2xx that still failed is a stream with no body: the server answered, badly.
            if (err.status >= 200 && err.status < 300) {
                return new SSMError('BRIDGE_RESPONSE_INVALID', `${what} returned no response body.`);
            }
            return new SSMError('BRIDGE_REQUEST_FAILED', `${what} returned ${err.status}: ${err.body}`);
        }
        return new SSMError('BRIDGE_REQUEST_FAILED', `${what} request failed: ${err instanceof Error ? err.message : String(err)}`, err);
    }
}

/** Maps an OpenAI `usage` object onto the canonical {@link LlmUsage} shape. */
function readOpenAIUsage(usage: unknown, model: string): LlmUsage | undefined {
    if (!usage || typeof usage !== 'object') return undefined;
    const u = readUsageFields(usage);
    const cached = u.cacheReadTokens ?? 0;
    return {
        model,
        // OpenAI reports `prompt_tokens` INCLUSIVE of cached tokens; the canonical
        // shape keeps them disjoint so the two rates are applied exactly once each.
        inputTokens: Math.max(0, (u.promptTokens ?? 0) - cached),
        outputTokens: u.completionTokens ?? 0,
        cachedInputTokens: cached,
    };
}
