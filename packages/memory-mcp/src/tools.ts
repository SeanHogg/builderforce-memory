/**
 * Framework-agnostic MCP tool definitions over a MemoryBackend.
 *
 * Defined once here, then registered into whichever server framework a
 * transport uses — the Claude Agent SDK's `tool()` (in-process) or the MCP
 * SDK's `registerTool()` (stdio/HTTP). Both accept the same (name, description,
 * zod raw shape, handler→CallToolResult) shape, so the handlers below are the
 * single source of truth.
 *
 * Token-saving is enforced HERE, server-side, regardless of what the model
 * asks for: recall is top-K (capped), each entry's content is truncated, and
 * there is deliberately no "return everything" tool. Moving memory out of the
 * prompt only saves tokens if recall is selective — a tool that dumps the whole
 * store back into context is more expensive than inlining it.
 */

import { z } from "zod";
import type { MemoryBackend, RecallHit } from "./backend.js";

/** The MCP CallToolResult shape both server frameworks expect. */
export interface ToolResult {
    content: Array<{ type: "text"; text: string }>;
    isError?: boolean;
}

/** A framework-neutral tool: maps 1:1 onto Agent-SDK `tool()` and MCP `registerTool()`. */
export interface MemoryTool {
    name: string;
    description: string;
    /** Zod *raw shape* (e.g. `{ query: z.string() }`), not a ZodObject. */
    inputSchema: z.ZodRawShape;
    handler: (args: Record<string, unknown>) => Promise<ToolResult>;
}

export interface MemoryToolsOptions {
    /** Hard cap on entries any recall tool returns to the model. Default 5. */
    maxResults?: number;
    /** Max characters of each entry's content surfaced to the model. Default 500. */
    maxContentChars?: number;
    /** Expose write tools (remember/forget). Default true; forced false if the backend is read-only. */
    writable?: boolean;
}

const DEFAULT_MAX_RESULTS = 5;
const DEFAULT_MAX_CONTENT = 500;

function clip(s: string, n: number): string {
    return s.length <= n ? s : `${s.slice(0, n)}…`;
}

function ok(text: string): ToolResult {
    return { content: [{ type: "text", text }] };
}

function fail(text: string): ToolResult {
    return { content: [{ type: "text", text }], isError: true };
}

function renderHits(hits: RecallHit[], maxChars: number): string {
    if (hits.length === 0) return "No matching memories.";
    return hits
        .map((h) => {
            const tags = h.tags?.length ? ` tags=[${h.tags.join(", ")}]` : "";
            const score = h.score != null ? ` score=${h.score.toFixed(3)}` : "";
            return `• ${h.key}${score}${tags}\n  ${clip(h.content, maxChars)}`;
        })
        .join("\n");
}

/**
 * Builds the memory tool set bound to `backend`. Write tools are included only
 * when `writable` is not false AND the backend actually implements them.
 */
export function buildMemoryTools(backend: MemoryBackend, opts: MemoryToolsOptions = {}): MemoryTool[] {
    const maxResults = Math.max(1, opts.maxResults ?? DEFAULT_MAX_RESULTS);
    const maxContent = Math.max(80, opts.maxContentChars ?? DEFAULT_MAX_CONTENT);
    const writable = opts.writable !== false;

    const tools: MemoryTool[] = [
        {
            name: "memory_recall",
            description:
                "Semantically recall the most relevant stored memories for a query. " +
                "Call this BEFORE answering whenever the task may depend on prior context, user " +
                "preferences, project decisions, or facts learned in earlier sessions — instead of " +
                "assuming that context is already in your prompt. Returns a small ranked set, not the " +
                "whole store.",
            inputSchema: {
                query: z.string().describe("What to look for — a question, topic, or keywords."),
                topK: z
                    .number()
                    .int()
                    .min(1)
                    .max(maxResults)
                    .optional()
                    .describe(`How many memories to return (max ${maxResults}).`),
            },
            handler: async (args) => {
                try {
                    const query = String(args["query"] ?? "");
                    if (!query.trim()) return fail("query is required.");
                    const k = Math.min(maxResults, Number(args["topK"] ?? maxResults));
                    const hits = await backend.recall(query, k);
                    return ok(renderHits(hits.slice(0, maxResults), maxContent));
                } catch (err) {
                    return fail(`recall failed: ${String(err)}`);
                }
            },
        },
        {
            name: "memory_get",
            description:
                "Fetch a single memory by its exact key. Use when you already know the key " +
                "(e.g. one surfaced by memory_recall) and want its full, untruncated value.",
            inputSchema: {
                key: z.string().describe("The exact memory key."),
            },
            handler: async (args) => {
                try {
                    const key = String(args["key"] ?? "");
                    if (!key) return fail("key is required.");
                    const hit = await backend.get(key);
                    return hit ? ok(`• ${hit.key}\n  ${hit.content}`) : ok(`No memory found for key "${key}".`);
                } catch (err) {
                    return fail(`get failed: ${String(err)}`);
                }
            },
        },
        {
            name: "memory_recall_by_tag",
            description:
                "List memories carrying a given tag (e.g. 'user', 'project', 'decision'). " +
                "Use to pull a known category of context rather than searching semantically.",
            inputSchema: {
                tag: z.string().describe("The tag to filter by."),
                limit: z
                    .number()
                    .int()
                    .min(1)
                    .max(maxResults)
                    .optional()
                    .describe(`Max entries to return (max ${maxResults}).`),
            },
            handler: async (args) => {
                try {
                    const tag = String(args["tag"] ?? "");
                    if (!tag) return fail("tag is required.");
                    const limit = Math.min(maxResults, Number(args["limit"] ?? maxResults));
                    const hits = await backend.recallByTag(tag, limit);
                    return ok(renderHits(hits, maxContent));
                } catch (err) {
                    return fail(`recall_by_tag failed: ${String(err)}`);
                }
            },
        },
    ];

    if (writable && backend.remember) {
        tools.push({
            name: "memory_remember",
            description:
                "Persist a fact for future sessions. Call when you learn something durable and reusable: " +
                "a user preference, a project constraint, a decision and its rationale. Keep keys stable " +
                "and descriptive (e.g. 'user.preferred-language') so the same fact overwrites rather than " +
                "duplicating.",
            inputSchema: {
                key: z.string().describe("Stable, descriptive identifier; reusing a key overwrites it."),
                content: z.string().describe("The fact to store."),
                tags: z.array(z.string()).optional().describe("Optional grouping tags."),
                importance: z.number().min(0).max(1).optional().describe("Importance 0–1 (default 0.5)."),
                ttlMs: z.number().int().positive().optional().describe("Optional time-to-live in ms."),
            },
            handler: async (args) => {
                try {
                    const key = String(args["key"] ?? "");
                    const content = String(args["content"] ?? "");
                    if (!key || !content) return fail("key and content are required.");
                    await backend.remember!({
                        key,
                        content,
                        tags: args["tags"] as string[] | undefined,
                        importance: args["importance"] as number | undefined,
                        ttlMs: args["ttlMs"] as number | undefined,
                    });
                    return ok(`Remembered "${key}".`);
                } catch (err) {
                    return fail(`remember failed: ${String(err)}`);
                }
            },
        });
    }

    if (writable && backend.forget) {
        tools.push({
            name: "memory_forget",
            description: "Delete a memory by key. Use to remove a fact that is now wrong or obsolete.",
            inputSchema: {
                key: z.string().describe("The exact memory key to delete."),
            },
            handler: async (args) => {
                try {
                    const key = String(args["key"] ?? "");
                    if (!key) return fail("key is required.");
                    await backend.forget!(key);
                    return ok(`Forgot "${key}".`);
                } catch (err) {
                    return fail(`forget failed: ${String(err)}`);
                }
            },
        });
    }

    return tools;
}
