#!/usr/bin/env node
/**
 * `builderforce-memory-mcp-http` — Streamable HTTP MCP server over the LOCAL
 * MemoryStore. A reference host; in production builderforce.ai would mount
 * createMemoryHttpHandler() against a shared/remote backend instead.
 *
 * Env:
 *   PORT                          Listen port (default 8787).
 *   BUILDERFORCE_MEMORY_TOKEN     Bearer token required on every request (recommended).
 *   BUILDERFORCE_MEMORY_DB        IndexedDB database name.
 *   BUILDERFORCE_MEMORY_READONLY  '1' to disable remember/forget tools.
 *   BUILDERFORCE_GATEWAY_URL      Gateway base URL (default https://api.builderforce.ai).
 *   BUILDERFORCE_API_KEY          `bfk_*` tenant key. When set, exposes the cost tools.
 */

import http from "node:http";
import { createLocalMemoryStoreBackend } from "../backends/memory-store.js";
import { createMemoryHttpHandler } from "../transports/http.js";

const backend = await createLocalMemoryStoreBackend({
    dbName: process.env["BUILDERFORCE_MEMORY_DB"],
});

const handler = createMemoryHttpHandler(backend, {
    authToken: process.env["BUILDERFORCE_MEMORY_TOKEN"],
    writable: process.env["BUILDERFORCE_MEMORY_READONLY"] !== "1",
    // Optional gateway-backed cost tools (token_usage, model_efficiency). Only
    // exposed when an API key is present; URL defaults to the public gateway.
    gatewayUrl: process.env["BUILDERFORCE_GATEWAY_URL"] ?? "https://api.builderforce.ai",
    gatewayApiKey: process.env["BUILDERFORCE_API_KEY"],
});

const port = Number(process.env["PORT"] ?? 8787);

http.createServer((req, res) => {
    void handler(req, res).catch((err) => {
        if (!res.headersSent) res.writeHead(500, { "content-type": "application/json" });
        res.end(JSON.stringify({ error: String(err) }));
    });
}).listen(port, () => {
    process.stderr.write(`[builderforce-memory-mcp-http] listening on :${port}\n`);
});
