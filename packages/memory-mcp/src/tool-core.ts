/**
 * The framework-neutral tool shape every tool family here is written in (memory,
 * gateway cost, experience), and the result helpers they share. Each transport maps
 * a {@link MemoryTool} 1:1 onto Agent-SDK `tool()` or MCP `registerTool()`.
 */

import type { z } from "zod";

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

export function clip(s: string, n: number): string {
    return s.length <= n ? s : `${s.slice(0, n)}…`;
}

export function ok(text: string): ToolResult {
    return { content: [{ type: "text", text }] };
}

export function fail(text: string): ToolResult {
    return { content: [{ type: "text", text }], isError: true };
}

/** A structured result, as indented JSON — for tools a program reads as well as a model. */
export function okJson(value: unknown): ToolResult {
    return ok(JSON.stringify(value, null, 2));
}
