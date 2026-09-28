/**
 * LLM wire primitives — the protocol-level pieces every client of an
 * OpenAI/Anthropic/Gemini-style HTTP API needs, in ONE place:
 *
 *   • Server-Sent-Events `data:` framing — one line rule ({@link parseSseDataLine}),
 *     applied to a whole buffered body ({@link parseSseDataFrames}) or to a live
 *     byte stream ({@link readSseDataFrames});
 *   • token-usage field reading across vendor shapes ({@link readUsageFields}) and
 *     the Anthropic streaming usage fold ({@link AnthropicStreamUsage}).
 *
 * The package's bridges and the Builderforce gateway both frame and meter with
 * these, so a fix to the line rule (a spaceless `data:{…}` frame, say) lands once.
 * Zero-dependency and engine-free: the `@seanhogg/builderforce-memory/wire` subpath.
 */

export { parseSseDataLine, parseSseDataFrames, readSseDataFrames, isSseDoneLine } from './sse.js';
export { finiteNumber, readUsageFields, AnthropicStreamUsage } from './usage.js';
export type { UsageFields, AnthropicUsageTotals } from './usage.js';
