/**
 * In-process transport for the Claude Agent SDK.
 *
 * Returns the `type: "sdk"` MCP server config you drop straight into
 * `query({ options: { mcpServers: { builderforce_memory: <this> } } })`.
 * No subprocess, no socket — the tools run in your product's process, calling
 * the MemoryBackend directly. Lowest latency; the consuming product pulls in
 * @anthropic-ai/claude-agent-sdk (and, for the local backend, @builderforce/memory).
 *
 * The Agent SDK is imported indirectly so this package builds and so the stdio
 * and HTTP transports don't drag the SDK in.
 */

import type { MemoryBackend } from "../backend.js";
import { buildMemoryTools, type MemoryToolsOptions } from "../tools.js";

export interface SdkServerOptions extends MemoryToolsOptions {
    /** MCP server name; becomes the middle segment of `mcp__<name>__<tool>`. Default 'builderforce_memory'. */
    name?: string;
    version?: string;
}

/** The shape of `createSdkMcpServer`'s return — re-declared to avoid a hard type dep on the SDK. */
export interface SdkMcpServerConfig {
    type: "sdk";
    name: string;
    instance: unknown;
}

/**
 * Wraps the memory tools as an in-process Claude Agent SDK MCP server.
 *
 * @example
 *   import { query } from "@anthropic-ai/claude-agent-sdk";
 *   import { createMemoryMcpServer, createLocalMemoryStoreBackend } from "@builderforce/memory-mcp";
 *
 *   const backend = await createLocalMemoryStoreBackend();
 *   const memory  = await createMemoryMcpServer(backend);
 *
 *   for await (const msg of query({
 *     prompt: "...",
 *     options: {
 *       mcpServers: { builderforce_memory: memory },
 *       allowedTools: ["mcp__builderforce_memory__*"],
 *     },
 *   })) { /* ... *\/ }
 */
export async function createMemoryMcpServer(
    backend: MemoryBackend,
    opts: SdkServerOptions = {},
): Promise<SdkMcpServerConfig> {
    const _import = (m: string): Promise<unknown> =>
        // eslint-disable-next-line @typescript-eslint/no-implied-eval, no-new-func
        new Function("m", "return import(m)")(m) as Promise<unknown>;

    const sdk = (await _import("@anthropic-ai/claude-agent-sdk")) as {
        tool: (
            name: string,
            description: string,
            schema: unknown,
            handler: (args: Record<string, unknown>) => Promise<unknown>,
        ) => unknown;
        createSdkMcpServer: (o: { name: string; version?: string; tools: unknown[] }) => SdkMcpServerConfig;
    };

    const tools = buildMemoryTools(backend, opts).map((t) =>
        sdk.tool(t.name, t.description, t.inputSchema, t.handler as (a: Record<string, unknown>) => Promise<unknown>),
    );

    return sdk.createSdkMcpServer({
        name: opts.name ?? "builderforce_memory",
        version: opts.version ?? "2026.5.31",
        tools,
    });
}
