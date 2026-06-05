/**
 * @builderforce/memory-mcp — expose @builderforce/memory to MCP clients.
 *
 * One token-saving tool core over a pluggable MemoryBackend, three transports:
 *   - createMemoryMcpServer  → in-process Claude Agent SDK (type:"sdk")
 *   - runStdio               → stdio subprocess (any language)
 *   - createMemoryHttpHandler→ Streamable HTTP (multi-tenant / networked)
 */

// ── Seam ────────────────────────────────────────────────────────────────────
export type { MemoryBackend, RecallHit, RememberInput } from "./backend.js";

// ── Local backend (IndexedDB via @builderforce/memory) ────────────────────────
export { MemoryStoreBackend, createLocalMemoryStoreBackend } from "./backends/memory-store.js";
export type { LocalBackendOptions } from "./backends/memory-store.js";

// ── Tool core ─────────────────────────────────────────────────────────────────
export { buildMemoryTools } from "./tools.js";
export type { MemoryTool, MemoryToolsOptions, ToolResult } from "./tools.js";

// ── Transports ──────────────────────────────────────────────────────────────
export { createMemoryMcpServer } from "./transports/sdk.js";
export type { SdkServerOptions, SdkMcpServerConfig } from "./transports/sdk.js";

export { buildMcpServer } from "./transports/mcp-server.js";
export type { McpServerOptions } from "./transports/mcp-server.js";

export { runStdio } from "./transports/stdio.js";

export { createMemoryHttpHandler } from "./transports/http.js";
export type { HttpHandlerOptions } from "./transports/http.js";
