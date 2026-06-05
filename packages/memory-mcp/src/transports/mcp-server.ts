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

export interface McpServerOptions extends MemoryToolsOptions {
    name?: string;
    version?: string;
}

/** Constructs an McpServer with the memory tools registered. */
export function buildMcpServer(backend: MemoryBackend, opts: McpServerOptions = {}): McpServer {
    const server = new McpServer({
        name: opts.name ?? "builderforce-memory",
        version: opts.version ?? "2026.5.31",
    });

    for (const t of buildMemoryTools(backend, opts)) {
        server.registerTool(
            t.name,
            { description: t.description, inputSchema: t.inputSchema },
            // MCP SDK and our ToolResult share the CallToolResult shape.
            t.handler as never,
        );
    }

    return server;
}
