/**
 * Token-usage reading across vendor shapes.
 *
 * Two wire shapes reach a client: OpenAI-style `prompt_tokens`/`completion_tokens`
 * (cached reads under `prompt_tokens_details.cached_tokens`, INCLUDED in
 * `prompt_tokens`) and Anthropic-style `input_tokens`/`output_tokens` (cache reads
 * and writes as separate top-level fields, EXCLUDED from `input_tokens`).
 * {@link readUsageFields} reads either into one set of optional fields, keeping
 * "absent" distinct from 0; each consumer then maps them to its own accounting.
 *
 * Streaming Anthropic responses split usage across events — input and cache on
 * `message_start.message.usage`, output accumulating on `message_delta.usage` — and
 * every count is cumulative, so {@link AnthropicStreamUsage} keeps the largest seen.
 */

/** A finite number, or `undefined` for null/undefined/non-numeric input. */
export function finiteNumber(v: unknown): number | undefined {
    if (v === null || v === undefined) return undefined;
    const n = Number(v);
    return Number.isFinite(n) ? n : undefined;
}

/** Usage as the vendor reported it. Every field is absent when the vendor omitted it. */
export interface UsageFields {
    /** `prompt_tokens`, else `input_tokens`. */
    promptTokens?: number;
    /** `completion_tokens`, else `output_tokens`. */
    completionTokens?: number;
    totalTokens?: number;
    /** `cache_read_input_tokens`, else `prompt_tokens_details.cached_tokens`. */
    cacheReadTokens?: number;
    /** `cache_creation_input_tokens`. */
    cacheCreationTokens?: number;
}

/** Reads a vendor `usage` object of either shape. Anything that is not an object reads as empty. */
export function readUsageFields(u: unknown): UsageFields {
    const out: UsageFields = {};
    if (!u || typeof u !== 'object') return out;
    const usage = u as Record<string, unknown>;
    const prompt = finiteNumber(usage['prompt_tokens'] ?? usage['input_tokens']);
    const completion = finiteNumber(usage['completion_tokens'] ?? usage['output_tokens']);
    const total = finiteNumber(usage['total_tokens']);
    if (prompt !== undefined) out.promptTokens = prompt;
    if (completion !== undefined) out.completionTokens = completion;
    if (total !== undefined) out.totalTokens = total;

    const details = usage['prompt_tokens_details'];
    const cachedFromDetails = details && typeof details === 'object'
        ? finiteNumber((details as Record<string, unknown>)['cached_tokens'])
        : undefined;
    const cacheRead = finiteNumber(usage['cache_read_input_tokens']) ?? cachedFromDetails;
    const cacheCreation = finiteNumber(usage['cache_creation_input_tokens']);
    if (cacheRead !== undefined) out.cacheReadTokens = cacheRead;
    if (cacheCreation !== undefined) out.cacheCreationTokens = cacheCreation;
    return out;
}

/** The totals of one Anthropic stream. Input excludes both cache counts. */
export interface AnthropicUsageTotals {
    inputTokens: number;
    outputTokens: number;
    cacheReadTokens: number;
    cacheCreationTokens: number;
    /** `message_start.message.model`, when the stream named it. */
    model?: string;
}

/** Folds the usage events of one Anthropic Messages stream. Feed it every parsed frame. */
export class AnthropicStreamUsage {
    private readonly totals: AnthropicUsageTotals = { inputTokens: 0, outputTokens: 0, cacheReadTokens: 0, cacheCreationTokens: 0 };

    observe(frame: unknown): void {
        if (!frame || typeof frame !== 'object') return;
        const ev = frame as { type?: unknown; message?: { usage?: unknown; model?: unknown }; usage?: unknown };
        let usage: unknown;
        if (ev.type === 'message_start') {
            usage = ev.message?.usage;
            if (typeof ev.message?.model === 'string') this.totals.model = ev.message.model;
        } else if (ev.type === 'message_delta') {
            usage = ev.usage;
        }
        if (!usage || typeof usage !== 'object') return;
        const u = usage as Record<string, unknown>;
        const keep = (current: number, v: unknown) => Math.max(current, finiteNumber(v) ?? 0);
        this.totals.inputTokens = keep(this.totals.inputTokens, u['input_tokens']);
        this.totals.outputTokens = keep(this.totals.outputTokens, u['output_tokens']);
        this.totals.cacheReadTokens = keep(this.totals.cacheReadTokens, u['cache_read_input_tokens']);
        this.totals.cacheCreationTokens = keep(this.totals.cacheCreationTokens, u['cache_creation_input_tokens']);
    }

    result(): AnthropicUsageTotals {
        return { ...this.totals };
    }
}
