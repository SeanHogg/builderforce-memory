#!/usr/bin/env node
/**
 * `builderforce-memory-mcp` — stdio MCP server over the LOCAL MemoryStore.
 *
 * Env:
 *   BUILDERFORCE_MEMORY_DB        IndexedDB database name (default: MemoryStore default).
 *   BUILDERFORCE_MEMORY_READONLY  '1' to disable remember/forget tools.
 *
 * Recall is lexical (Jaccard) here — this headless binary does not stand up the
 * SSM runtime/GPU. For SSM-embedding recall, embed the package in-process and
 * pass `runtime` to createLocalMemoryStoreBackend (see README).
 */

import { createLocalMemoryStoreBackend } from "../backends/memory-store.js";
import { runStdio } from "../transports/stdio.js";

const backend = await createLocalMemoryStoreBackend({
    dbName: process.env["BUILDERFORCE_MEMORY_DB"],
});

await runStdio(backend, { writable: process.env["BUILDERFORCE_MEMORY_READONLY"] !== "1" });
