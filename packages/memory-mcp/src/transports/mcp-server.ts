/**
 * Shared MCP server builder for the out-of-process transports (stdio + HTTP).
 *
 * Both use the standard @modelcontextprotocol/sdk `McpServer`; only the
 * transport binding differs. The tools come from the same buildMemoryTools()
 * the in-process Agent SDK transport uses, so all three stay in lockstep.
 */

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { MemoryBackend } from "../backend.js";
import { buildMemoryTools, type MemoryToolsOptions } from "../tools.js";
import { memoryMcpVersion } from "./version.js";

export interface McpServerOptions extends MemoryToolsOptions {
    name?: string;
    version?: string;
}

/** Constructs an McpServer with the memory tools registered. */
export function buildMcpServer(backend: MemoryBackend, opts: McpServerOptions = {}): McpServer {
    const server = new McpServer({
        name: opts.name ?? "builderforce-memory",
        version: opts.version ?? memoryMcpVersion,
    });

    // The generic registerTool() over a zod raw shape trips TS2589 (excessively
    // deep instantiation); the tool surface is identical across our transports,
    // so register through a loose signature. Runtime behaviour is unchanged —
    // both frameworks consume the same description, schema and handler.
    const register = server.registerTool.bind(server) as (
        name: string,
        config: { description: string; inputSchema: unknown; annotations: { readOnlyHint: boolean } },
        handler: unknown,
    ) => void;

    for (const t of buildMemoryTools(backend, opts)) {
        register(t.name, {
            description: t.description,
            inputSchema: t.inputSchema,
            annotations: { readOnlyHint: t.effect === "read" },
        }, t.handler);
    }

    return server;
}
