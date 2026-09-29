/**
 * The OpenAI-compatible chat-completions client — ONE request builder, error mapping
 * and stream reader for every caller that speaks `/chat/completions`: this package's
 * `OpenAIBridge` / `FetchBridge`, and any host that reaches models through an
 * OpenAI-compatible gateway (tools, streamed tool-call deltas, vendor extras).
 *
 * The body carries exactly what the request states — no sampling defaults are added
 * here, so a routing gateway that picks the model and limits can be called with none.
 * Callers that want defaults (the bridges) state them.
 *
 * Zero-dependency and engine-free (part of the `/wire` subpath).
 */

import { readSseDataFrames } from './sse.js';

/** An OpenAI-compatible chat message. */
export interface ChatMessage {
    role: 'system' | 'user' | 'assistant' | 'tool';
    /** Text, or an OpenAI multi-part array (text + `image_url`, …). */
    content?: string | null | unknown[];
    tool_calls?: unknown;
    tool_call_id?: string;
    [k: string]: unknown;
}

/** An OpenAI-compatible function-tool schema. */
export interface ChatToolSchema {
    type: 'function';
    function: { name: string; description: string; parameters: Record<string, unknown> };
}

export interface ChatRequest {
    messages: ChatMessage[];
    tools?: ChatToolSchema[];
    model?: string;
    maxTokens?: number;
    temperature?: number;
    topP?: number;
    /**
     * On a stream, ask for the closing usage block (`stream_options.include_usage`).
     * Opt-in: not every OpenAI-compatible server accepts the field.
     */
    includeUsage?: boolean;
    /** Vendor extras merged into the body last (a routing `provider`, `response_format`, …). */
    extra?: Record<string, unknown>;
}

export interface ChatToolCall {
    id?: string;
    type?: string;
    function?: { name?: string; arguments?: string };
}

export interface ChatResult {
    /** The reply text ('' when there was none — e.g. a tool-call-only turn). */
    content: string;
    /** True when the server sent text content (a string), false when it sent none. */
    hasTextContent: boolean;
    toolCalls: ChatToolCall[];
    finishReason?: string;
    /** The model the server reports having used. */
    model?: string;
    /** The raw `usage` object, when the server sent one — read it with `readUsageFields`. */
    usage?: unknown;
}

/** Incremental events of a streamed completion. */
export type ChatStreamEvent =
    | { type: 'text-delta'; delta: string }
    | { type: 'tool-call'; index: number; id?: string; name?: string; argsDelta?: string }
    | { type: 'done'; result: ChatResult };

export interface ChatClient {
    /** The OpenAI-style base, e.g. `https://api.openai.com/v1`; `/chat/completions` is appended. */
    baseUrl: string;
    apiKey: string;
    /** Abort a request after this long. Default: no limit beyond the caller's signal. */
    timeoutMs?: number;
}

/** A non-2xx answer, with the status and (bounded) body kept apart for the caller's message. */
export class ChatCompletionError extends Error {
    constructor(readonly status: number, readonly body: string) {
        super(`chat completion failed with HTTP ${status}: ${body}`);
        this.name = 'ChatCompletionError';
    }
}

const ERROR_BODY_CHARS = 300;

function chatBody(req: ChatRequest, stream: boolean): string {
    return JSON.stringify({
        ...(req.model ? { model: req.model } : {}),
        messages: req.messages,
        ...(req.tools?.length ? { tools: req.tools, tool_choice: 'auto' } : {}),
        ...(typeof req.maxTokens === 'number' ? { max_tokens: req.maxTokens } : {}),
        ...(typeof req.temperature === 'number' ? { temperature: req.temperature } : {}),
        ...(typeof req.topP === 'number' ? { top_p: req.topP } : {}),
        ...(stream ? { stream: true } : {}),
        // Without include_usage a streamed response carries no usage block at all.
        ...(stream && req.includeUsage ? { stream_options: { include_usage: true } } : {}),
        ...(req.extra ?? {}),
    });
}

async function post(client: ChatClient, req: ChatRequest, stream: boolean, signal?: AbortSignal): Promise<Response> {
    const timeout = client.timeoutMs ? AbortSignal.timeout(client.timeoutMs) : undefined;
    const res = await fetch(`${client.baseUrl.replace(/\/$/, '')}/chat/completions`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${client.apiKey}`,
            ...(stream ? { Accept: 'text/event-stream' } : {}),
        },
        body: chatBody(req, stream),
        signal: signal && timeout ? AbortSignal.any([signal, timeout]) : (signal ?? timeout),
    });
    if (!res.ok || (stream && !res.body)) {
        const body = await res.text().catch(() => '');
        throw new ChatCompletionError(res.status, body.slice(0, ERROR_BODY_CHARS));
    }
    return res;
}

/** One non-streamed completion. Throws {@link ChatCompletionError} on a non-2xx answer. */
export async function chatComplete(client: ChatClient, req: ChatRequest, signal?: AbortSignal): Promise<ChatResult> {
    const res = await post(client, req, false, signal);
    const json = (await res.json().catch(() => null)) as {
        model?: unknown;
        usage?: unknown;
        choices?: Array<{ message?: { content?: unknown; tool_calls?: unknown }; finish_reason?: string }>;
    } | null;
    const choice = json?.choices?.[0];
    const msg = choice?.message;
    return {
        content: typeof msg?.content === 'string' ? msg.content : '',
        hasTextContent: typeof msg?.content === 'string',
        toolCalls: Array.isArray(msg?.tool_calls) ? (msg.tool_calls as ChatToolCall[]) : [],
        finishReason: choice?.finish_reason,
        ...(typeof json?.model === 'string' ? { model: json.model } : {}),
        ...(json?.usage !== undefined ? { usage: json.usage } : {}),
    };
}

/**
 * A streamed completion as events: text and tool-call deltas as they arrive, then one
 * `done` carrying the assembled result (tool-call arguments concatenated per index).
 */
export async function* chatStream(client: ChatClient, req: ChatRequest, signal?: AbortSignal): AsyncGenerator<ChatStreamEvent> {
    const res = await post(client, req, true, signal);
    let content = '';
    let finishReason: string | undefined;
    let model: string | undefined;
    let usage: unknown;
    const calls: Array<{ id?: string; name?: string; args: string }> = [];

    for await (const frame of readSseDataFrames(res.body!)) {
        const chunk = frame as {
            model?: unknown;
            usage?: unknown;
            choices?: Array<{
                delta?: { content?: unknown; tool_calls?: Array<{ index?: number; id?: string; function?: { name?: string; arguments?: string } }> };
                finish_reason?: string;
            }>;
        };
        if (typeof chunk.model === 'string') model = chunk.model;
        if (chunk.usage) usage = chunk.usage;
        const choice = chunk.choices?.[0];
        if (!choice) continue;
        if (choice.finish_reason) finishReason = choice.finish_reason;
        const delta = choice.delta;
        if (typeof delta?.content === 'string' && delta.content) {
            content += delta.content;
            yield { type: 'text-delta', delta: delta.content };
        }
        for (const tc of Array.isArray(delta?.tool_calls) ? delta.tool_calls : []) {
            const index = typeof tc.index === 'number' ? tc.index : 0;
            const acc = (calls[index] ??= { args: '' });
            if (tc.id) acc.id = tc.id;
            if (tc.function?.name) acc.name = tc.function.name;
            const argsDelta = tc.function?.arguments;
            if (typeof argsDelta === 'string') acc.args += argsDelta;
            yield { type: 'tool-call', index, id: tc.id, name: tc.function?.name, argsDelta };
        }
    }

    const result: ChatResult = {
        content,
        hasTextContent: content !== '',
        finishReason,
        toolCalls: calls.filter(Boolean).map((t) => ({ id: t.id, type: 'function', function: { name: t.name, arguments: t.args } })),
        ...(model ? { model } : {}),
        ...(usage !== undefined ? { usage } : {}),
    };
    yield { type: 'done', result };
}
