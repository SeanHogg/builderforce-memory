/**
 * Shared MCP server builder for the out-of-process transports (stdio + HTTP).
 *
 * Both use the standard @modelcontextprotocol/sdk `McpServer`; only the
 * transport binding differs. The tools come from the same buildMemoryTools()
 * the in-process Agent SDK transport uses, so all three stay in lockstep.
 */

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { Transport } from "@modelcontextprotocol/sdk/shared/transport.js";
import type { MemoryBackend } from "../backend.js";
import { sharedMcpModule } from "../shared-mcp.js";
import { buildMemoryTools, type MemoryToolsOptions } from "../tools.js";
import { createMemoryMcpCatalog } from "./catalog.js";
import { memoryMcpVersion } from "./version.js";

export interface McpServerOptions extends MemoryToolsOptions {
    name?: string;
    version?: string;
}

interface TransportServer {
    connect(transport: Transport): Promise<void>;
    close(): Promise<void>;
}

interface SharedServerModule {
    createMcpServer(
        catalog: ReturnType<typeof createMemoryMcpCatalog>,
        options: {
            info: { name: string; version: string };
            gate: { approveWrite(): boolean };
        },
    ): TransportServer;
}

interface SharedNamingModule {
    advertiseName(source: string, tool: string): string;
}

function serverInfo(opts: McpServerOptions) {
    return { name: opts.name ?? "builderforce-memory", version: opts.version ?? memoryMcpVersion };
}

/** Use the shared runtime when it is installed alongside Memory. */
export async function createMemoryTransportServer(
    backend: MemoryBackend,
    opts: McpServerOptions,
): Promise<TransportServer> {
    const shared = await sharedMcpModule<SharedServerModule>("/server");
    if (!shared) return buildMcpServer(backend, opts);
    const naming = await sharedMcpModule<SharedNamingModule>();
    if (!naming) throw new Error("Builderforce MCP naming module is unavailable");

    return shared.createMcpServer(createMemoryMcpCatalog(backend, opts, naming.advertiseName), {
        info: serverInfo(opts),
        gate: { approveWrite: () => opts.writable !== false },
    });
}

/** Constructs an McpServer with the memory tools registered. */
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
        }, t.handler);
    }

    return server;
}
