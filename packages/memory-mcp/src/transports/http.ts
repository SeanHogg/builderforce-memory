/**
 * HTTP transport — the multi-tenant / networked path (builderforce.ai hosting,
 * remote claws). Stateless Streamable HTTP: one short-lived McpServer +
 * transport per request, so it scales horizontally with no sticky sessions.
 * Bearer-token auth gates every request.
 *
 * Consumed from the Claude Agent SDK as:
 *   mcpServers: {
 *     builderforce_memory: { type: "http", url: "https://mcp.builderforce.ai/memory",
 *                            headers: { Authorization: `Bearer ${KEY}` } }
 *   }
 */

import type { IncomingMessage, ServerResponse } from "node:http";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import type { MemoryBackend } from "../backend.js";
import { buildMcpServer, type McpServerOptions } from "./mcp-server.js";

export interface HttpHandlerOptions extends McpServerOptions {
    /**
     * Shared secret required in `Authorization: Bearer <token>`. When set, every
     * request without a matching token is rejected 401. Omit only behind a trusted
     * gateway that has already authenticated the caller.
     */
    authToken?: string;
}

function unauthorized(res: ServerResponse): void {
    res.writeHead(401, { "content-type": "application/json" });
    res.end(JSON.stringify({ error: "unauthorized" }));
}

/**
 * Returns a `(req, res) => Promise<void>` handler you mount on any Node HTTP
 * server (or Express). Per request it spins up a stateless MCP server bound to
 * `backend`, handles the MCP exchange, and tears down on response close.
 */
export function createMemoryHttpHandler(
    backend: MemoryBackend,
    opts: HttpHandlerOptions = {},
): (req: IncomingMessage, res: ServerResponse) => Promise<void> {
    return async (req, res) => {
        if (opts.authToken) {
            const header = req.headers["authorization"];
            if (header !== `Bearer ${opts.authToken}`) {
                unauthorized(res);
                return;
            }
        }

        const server = buildMcpServer(backend, opts);
        // Stateless: no session id generator → a fresh transport per request.
        const transport = new StreamableHTTPServerTransport({ sessionIdGenerator: undefined });

        res.on("close", () => {
            void transport.close();
            void server.close();
        });

        await server.connect(transport);
        // Pass undefined body — the transport reads/parses the request stream itself.
        await transport.handleRequest(req, res);
    };
}
