/**
 * HTTP transport — the multi-tenant / networked path (builderforce.ai hosting,
 * remote hosts). Stateless Streamable HTTP: one short-lived McpServer +
 * transport per request, so it scales horizontally with no sticky sessions.
 *
 * Auth model (pick one, in precedence order):
 *   • `tenants` — a `{ token: backend }` map. The bearer token selects the
 *     tenant's OWN backend (namespace isolation); an unknown token is 401. This
 *     is the production multi-tenant mode — one token can never read another
 *     tenant's facts.
 *   • `resolveBackend` — a function `(token) => backend` for dynamic tenant
 *     resolution (e.g. a gateway lookup).
 *   • `authToken` — a single shared secret guarding the one positional backend
 *     (single-tenant / reference hosting).
 *   • none — open, only valid behind a trusted gateway that already authed.
 *
 * Token comparison is constant-time (hash + timingSafeEqual / hash-keyed lookup),
 * and an optional fixed-window rate limit (per tenant, or per client IP when
 * unauthenticated) caps abuse.
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
import { RateLimiter, bearerToken, buildTenantIndex, hashToken, timingSafeEqualStr } from "./auth.js";

export interface HttpHandlerOptions extends McpServerOptions {
    /**
     * Shared secret required in `Authorization: Bearer <token>` for the single
     * positional backend. Every request without a matching token is rejected 401.
     * Omit only behind a trusted gateway that has already authenticated the caller.
     */
    authToken?: string;
    /**
     * Multi-tenant map: bearer token → that tenant's backend. When set, the token
     * selects the backend (namespace isolation) and the positional `backend` is
     * only used as a fallback when no token-auth modes match. Lookups are keyed by
     * token hash, never the raw token.
     */
    tenants?: Record<string, MemoryBackend>;
    /**
     * Dynamic tenant resolver: `(token) => backend | undefined`. Used when tenants
     * live in an external store. Returning undefined → 401.
     */
    resolveBackend?: (token: string) => MemoryBackend | undefined | Promise<MemoryBackend | undefined>;
    /** Fixed-window rate limit applied per tenant (or per client IP if open). */
    rateLimit?: { windowMs: number; max: number };
    /** Injectable clock (ms) for deterministic rate-limit tests. */
    now?: () => number;
}

function send(res: ServerResponse, status: number, body: unknown): void {
    res.writeHead(status, { "content-type": "application/json" });
    res.end(JSON.stringify(body));
}

/** First-hop client IP, for rate-limiting unauthenticated callers. */
function clientIp(req: IncomingMessage): string {
    const fwd = req.headers["x-forwarded-for"];
    const first = Array.isArray(fwd) ? fwd[0] : fwd?.split(",")[0];
    return (first?.trim() || req.socket?.remoteAddress || "unknown");
}

/**
 * Returns a `(req, res) => Promise<void>` handler you mount on any Node HTTP
 * server (or Express). Per request it authenticates, resolves the tenant
 * backend, rate-limits, spins up a stateless MCP server bound to that backend,
 * handles the MCP exchange, and tears down on response close.
 */
export function createMemoryHttpHandler(
    backend: MemoryBackend,
    opts: HttpHandlerOptions = {},
): (req: IncomingMessage, res: ServerResponse) => Promise<void> {
    const tenantIndex = opts.tenants ? buildTenantIndex(opts.tenants) : undefined;
    const limiter = opts.rateLimit
        ? new RateLimiter(opts.rateLimit.windowMs, opts.rateLimit.max, opts.now ?? (() => Date.now()))
        : undefined;

    return async (req, res) => {
        const token = bearerToken(req.headers["authorization"]);

        // ── Resolve the per-request backend + a rate-limit key ────────────────
        let resolved: MemoryBackend | undefined;
        let rateKey: string;

        if (tenantIndex) {
            // Multi-tenant: token → that tenant's backend (hash-keyed lookup).
            if (!token) { send(res, 401, { error: "unauthorized" }); return; }
            resolved = tenantIndex.get(hashToken(token));
            if (!resolved) { send(res, 401, { error: "unauthorized" }); return; }
            rateKey = `t:${hashToken(token)}`;
        } else if (opts.resolveBackend) {
            if (!token) { send(res, 401, { error: "unauthorized" }); return; }
            resolved = await opts.resolveBackend(token);
            if (!resolved) { send(res, 401, { error: "unauthorized" }); return; }
            rateKey = `t:${hashToken(token)}`;
        } else if (opts.authToken) {
            // Single shared secret, constant-time compare.
            if (!token || !timingSafeEqualStr(token, opts.authToken)) {
                send(res, 401, { error: "unauthorized" });
                return;
            }
            resolved = backend;
            rateKey = `t:${hashToken(token)}`;
        } else {
            // Open (trusted gateway): rate-limit by client IP.
            resolved = backend;
            rateKey = `ip:${clientIp(req)}`;
        }

        // ── Rate limit ────────────────────────────────────────────────────────
        if (limiter && !limiter.check(rateKey)) {
            send(res, 429, { error: "rate_limited" });
            return;
        }

        const server = buildMcpServer(resolved, opts);
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
