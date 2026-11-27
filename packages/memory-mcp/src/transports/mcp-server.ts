/**
 * Shared MCP server builder for the out-of-process transports (stdio + HTTP).
 *
 * Both use the standard @modelcontextprotocol/sdk `McpServer`; only the
 * transport binding differs. The tools come from the same buildMemoryTools()
 * the in-process Agent SDK transport uses, so all three stay in lockstep.
 */

import { advertiseName } from "@seanhogg/builderforce-mcp";
import { protectMcpToolResult } from "@seanhogg/builderforce-mcp/protocol";
import { createMcpServer } from "@seanhogg/builderforce-mcp/server";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { MemoryBackend } from "../backend.js";
import { buildMemoryTools, type MemoryToolsOptions } from "../tools.js";
import { createMemoryMcpCatalog } from "./catalog.js";
import { memoryMcpVersion } from "./version.js";

export interface McpServerOptions extends MemoryToolsOptions {
    name?: string;
    version?: string;
}

function serverInfo(opts: McpServerOptions) {
    return { name: opts.name ?? "builderforce-memory", version: opts.version ?? memoryMcpVersion };
}

export function createMemoryTransportServer(backend: MemoryBackend, opts: McpServerOptions) {
    return createMcpServer(createMemoryMcpCatalog(backend, opts, advertiseName), {
        info: serverInfo(opts),
        gate: { approveWrite: () => opts.writable !== false },
    });
}

/** Constructs the SDK class for existing programmatic callers using its registration API. */
export function buildMcpServer(backend: MemoryBackend, opts: McpServerOptions = {}): McpServer {
    const server = new McpServer(serverInfo(opts));

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
        }, async (args: Record<string, unknown>) => protectMcpToolResult(t.name, await t.handler(args)));
    }

    return server;
}
