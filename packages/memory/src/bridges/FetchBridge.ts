/**
 * FetchBridge – generic OpenAI-compatible bridge for local or hosted endpoints.
 *
 * Works with Ollama, LM Studio, vLLM, llama.cpp server, a routing gateway, or any
 * service that exposes a /chat/completions endpoint compatible with the OpenAI
 * request and response schema.
 */

import { OpenAIBridge } from './OpenAIBridge.js';
import type { BridgeGenerateOptions } from './TransformerBridge.js';

export interface FetchBridgeOptions {
    /** Base URL of the OpenAI-compatible server, e.g. 'http://localhost:1234/v1'. */
    baseUrl       : string;
    /** API key — many local servers require any non-empty string. Default: 'local'. */
    apiKey?       : string;
    /** Model name understood by the server. Default: 'default' (none with `serverDefaults`). */
    model?        : string;
    /** Default system prompt. */
    systemPrompt? : string;
    /** Default max tokens. Default: 512 (none with `serverDefaults`). */
    maxTokens?    : number;
    /**
     * Send only what a call states, so a routing gateway picks the model and the
     * sampling settings. Default false.
     */
    serverDefaults?: boolean;
    /** Abort a request after this long. */
    timeoutMs?    : number;
}

/**
 * FetchBridge is a thin re-configuration of OpenAIBridge pointed at a custom
 * base URL.  All streaming and request logic is inherited.
 */
export class FetchBridge extends OpenAIBridge {
    constructor(opts: FetchBridgeOptions) {
        super({
            apiKey       : opts.apiKey        ?? 'local',
            model        : opts.model         ?? (opts.serverDefaults ? undefined : 'default'),
            baseUrl      : opts.baseUrl,
            systemPrompt : opts.systemPrompt,
            maxTokens    : opts.maxTokens,
            serverDefaults: opts.serverDefaults,
            timeoutMs    : opts.timeoutMs,
        });
    }
}

export type { BridgeGenerateOptions };
