/**
 * stdio transport — the portable path. Any MCP client in any language launches
 * this as a subprocess and speaks MCP over stdin/stdout. Consumed from the
 * Claude Agent SDK as:
 *   mcpServers: {
 *     builderforce_memory: { type: "stdio", command: "npx",
 *                            args: ["-y", "@seanhogg/builderforce-memory-mcp"] }
 *   }
 */

import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import type { MemoryBackend } from "../backend.js";
import { buildMcpServer, type McpServerOptions } from "./mcp-server.js";

/** Serves `backend` over stdio. Resolves when the transport closes. */
export async function runStdio(backend: MemoryBackend, opts: McpServerOptions = {}): Promise<void> {
    const server = buildMcpServer(backend, opts);
    await server.connect(new StdioServerTransport());
}
