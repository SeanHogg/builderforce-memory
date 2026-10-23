#!/usr/bin/env node
/**
 * `builderforce-memory-mcp` — stdio MCP server over the LOCAL MemoryStore.
 *
 * Env:
 *   BUILDERFORCE_MEMORY_DB        IndexedDB database name (default: MemoryStore default).
 *   BUILDERFORCE_MEMORY_READONLY  '1' to disable remember/forget tools.
 *   BUILDERFORCE_MEMORY_FILE      Absolute path to a JSON snapshot. Defaults to the
 *                                 shared per-machine store — resolveMemoryFile():
 *                                 ~/.builderforce-memory/memory.json, or the legacy
 *                                 ~/.claude/builderforce-memory/memory.json when that
 *                                 is where the live memory.json is — so memory
 *                                 survives the respawn every MCP client does between
 *                                 sessions. Compacted originals are archived beside it
 *                                 (memory.archive.json); each overwrite keeps .bak.1-3.
 *   BUILDERFORCE_MEMORY_MODEL     Absolute path to a `.evermind` package (an
 *                                 `evermind-lm`). When set, recall is ranked by SSM
 *                                 embedding cosine FUSED with the lexical ranking
 *                                 instead of lexically alone. Pure CPU — no GPU is
 *                                 involved; see src/embedding/evermind-embedder.ts.
 *                                 Also the model memory_compact's recall probe asks
 *                                 before stubbing a fact; unset = nothing is compacted.
 *   BUILDERFORCE_MEMORY_TOKENIZER Tokenizer JSON matching that checkpoint. Defaults
 *                                 to `<model>.tokenizer.json`.
 *   BUILDERFORCE_MEMORY_VECTORS   Persistent vector-cache path. Defaults beside the
 *                                 snapshot, so a respawned subprocess does not
 *                                 re-embed the whole store on its first recall.
 *   BUILDERFORCE_GATEWAY_URL      Gateway base URL (default https://api.builderforce.ai).
 *   BUILDERFORCE_API_KEY          `bfk_*` tenant key. When set, exposes the cost tools
 *                                 (token_usage, model_efficiency).
 *   BUILDERFORCE_MEMORY_EXPERIENCE '1' to expose the experience tools (episodes, Train
 *                                 Once skills, runs, experience_train). Kept beside the
 *                                 snapshot in experience.json + episodes/; trains the
 *                                 BUILDERFORCE_MEMORY_MODEL package.
 *
 * With no model configured recall ranks lexically (BM25) and the tool result SAYS
 * so ("Ranked by: lexical recall (fallback …)"), so a degrade is never silent.
 */

import { createLocalMemoryStoreBackend } from "../backends/memory-store.js";
import { createRecallEmbedder } from "../embedding/index.js";
import { readFile } from "node:fs/promises";
import { createExperienceHost } from "../experience/host.js";
import { resolveMemoryFile } from "../install/server-spec.js";
import { lazyRecallProbe } from "../model/recall-probe.js";
import { runStdio } from "../transports/stdio.js";

const log = (line: string) => void process.stderr.write(`${line}\n`);
// ONE resolver for installer, servers and hook: env, else whichever location holds
// the live memory.json (logged when there is a choice), else the canonical default.
const memoryFile = resolveMemoryFile(process.env, undefined, { log });

const backend = await createLocalMemoryStoreBackend({
    dbName: process.env["BUILDERFORCE_MEMORY_DB"],
    persistFile: memoryFile,
    // Null when no model is configured (or it fails to load) — recall then stays
    // lexical rather than the server refusing to start over an optional model.
    runtime: (await createRecallEmbedder({ memoryFile })) ?? undefined,
});

await runStdio(backend, {
    writable: process.env["BUILDERFORCE_MEMORY_READONLY"] !== "1",
    // Optional gateway-backed cost tools (token_usage, model_efficiency). Only
    // exposed when an API key is present; URL defaults to the public gateway.
    gatewayUrl: process.env["BUILDERFORCE_GATEWAY_URL"] ?? "https://api.builderforce.ai",
    gatewayApiKey: process.env["BUILDERFORCE_API_KEY"],
    // Null unless BUILDERFORCE_MEMORY_EXPERIENCE=1 — Synapse's opt-in.
    experience: await createExperienceHost({ memoryFile }),
    // memory_compact proves absorption with the configured Evermind before stubbing;
    // no model → nothing is compacted (and the tool says so).
    recallProbe: lazyRecallProbe(),
    // A local subprocess may read the user's own export file for memory_restore.
    readRestoreFile: (path) => readFile(path, "utf8"),
});
