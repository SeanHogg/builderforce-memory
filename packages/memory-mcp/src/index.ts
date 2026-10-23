/**
 * @seanhogg/builderforce-memory-mcp — expose @seanhogg/builderforce-memory to MCP clients.
 *
 * One token-saving tool core over a pluggable MemoryBackend, three transports:
 *   - createMemoryMcpServer  → in-process Claude Agent SDK (type:"sdk")
 *   - runStdio               → stdio subprocess (any language)
 *   - createMemoryHttpHandler→ Streamable HTTP (multi-tenant / networked)
 */

// ── Seam ────────────────────────────────────────────────────────────────────
export type { MemoryBackend, RankedRecall, RecallHit, RecallMethod, RememberInput } from "./backend.js";

// -- CPU SSM-embedding recall (no GPU) --------------------------------------
// The headless servers rank recall by EvermindLM embedding cosine fused with the
// lexical ranking whenever a checkpoint is configured; exported so an embedding
// host can build the same embedder (and share its persistent vector cache).
export { createRecallEmbedder, createEvermindEmbedder, CachedTextEmbedder, defaultVectorCacheFile, MODEL_FILE_ENV, TOKENIZER_FILE_ENV, VECTOR_CACHE_ENV } from "./embedding/index.js";
export type { EvermindEmbedderOptions, PersistentTextEmbedder, TextEmbedderLike, RecallEmbedderEnv } from "./embedding/index.js";

// ── Local backend (IndexedDB via @seanhogg/builderforce-memory) ────────────────────────
export { MemoryStoreBackend, createLocalMemoryStoreBackend, DEFAULT_RECALL_RANKING } from "./backends/memory-store.js";
export type { LocalBackendOptions, MemoryStoreBackendOptions, RecallRanking } from "./backends/memory-store.js";
export { FileFactArchive, MemoryFactArchive } from "./persistence/fact-archive.js";
export type { FactArchive } from "./persistence/fact-archive.js";
export { SharedJsonFile, DEFAULT_BACKUPS, backupFile } from "./persistence/shared-json-file.js";
export type { SharedJsonFileOptions } from "./persistence/shared-json-file.js";

// ── Tool core ─────────────────────────────────────────────────────────────────
export { buildMemoryTools } from "./tools.js";
export type { MemoryToolsOptions } from "./tools.js";
export type { MemoryTool, ToolResult } from "./tool-core.js";

// ── Experience (demonstrations → Train Once skills → runs → the private model) ─
// The domain is the engine's; this is its durable host and its tools.
export { createExperienceHost, EXPERIENCE_ENV } from "./experience/host.js";
export type { ExperienceHost, ExperienceHostOptions, ExperienceEngine } from "./experience/host.js";
export { nextExperienceVersion } from "./experience/train.js";

// ── Compaction (absorbed fact → one-line pointer stub, LOSSLESS) ─────────────
// Exported so an EXTERNAL compactor (the BuilderForce VS Code extension) produces
// byte-identical stubs AND follows the same archive + recall-probe rules, keeping
// the two paths idempotent with respect to each other.
export {
    STUB_PREFIX,
    DEFAULT_STUB_CHARS,
    CONTENT_LOST_LABEL,
    isStub,
    isContentLost,
    firstLine,
    memoryStub,
    stubPointer,
    planCompaction,
    planLosslessCompaction,
    probeQuestionFor,
    probePasses,
    defaultArchiveFile,
    parseArchive,
    mergeArchive,
} from "./compaction.js";
export type {
    ArchivedFact,
    ArchiveSnapshot,
    CompactionCandidate,
    CompactionWrite,
    CompactionSkip,
    CompactionPlan,
    CompactionOptions,
    LosslessCompactionOptions,
    ProbeQuestion,
    RecallProbe,
} from "./compaction.js";
export { createRecallProbe, lazyRecallProbe, generatorProbe, PROBE_MAX_NEW_TOKENS } from "./model/recall-probe.js";
export { parseRestoreSource, planRestore, restoredArchive, stubVersion, MIN_POINTER_CHARS } from "./restore.js";
export type { RestoreRecord, RestoreTarget, RestoreMatch, RestorePlan } from "./restore.js";

// ── Transports ──────────────────────────────────────────────────────────────
export { createMemoryMcpServer } from "./transports/sdk.js";
export type { SdkServerOptions, SdkMcpServerConfig } from "./transports/sdk.js";

export { buildMcpServer } from "./transports/mcp-server.js";
export type { McpServerOptions } from "./transports/mcp-server.js";

export { runStdio } from "./transports/stdio.js";

export { createMemoryHttpHandler } from "./transports/http.js";
export type { HttpHandlerOptions } from "./transports/http.js";
export { hashToken, timingSafeEqualStr, buildTenantIndex, bearerToken, RateLimiter } from "./transports/auth.js";

// ── Multi-host installer (wire the stdio server into any MCP-capable agent) ───
export {
    buildServerSpec,
    defaultMemoryFile,
    memoryFileCandidates,
    resolveMemoryFile,
    CANONICAL_MEMORY_SEGMENTS,
    LEGACY_MEMORY_SEGMENTS,
    MCP_PACKAGE,
    MCP_BIN,
    MEMORY_FILE_ENV,
    RUNTIME_PEERS,
} from "./install/server-spec.js";
export type { StdioServerSpec, ServerSpecOptions, MemoryFileProbe, ResolveMemoryFileOptions } from "./install/server-spec.js";
export { installMemoryServer } from "./install/install.js";
export type { InstallOptions, InstallResult, InstallStatus, HostSelector, FsLike } from "./install/install.js";
export { HOSTS, SERVER_KEY, findHost } from "./install/hosts.js";
export type { HostAdapter, HostEnv, ConfigFormat } from "./install/hosts.js";
// Claude Code "memory combo" — hooks (SessionStart/PreCompact/UserPromptSubmit/Stop)
// + companion skill that make memory self-driving (contextual recall + autonomous capture).
export { installClaudeCombo, bfmemHookSource, companionSkillMd, claudeComboPaths, pluginHooksConfig, HOOK_EVENTS } from "./install/claude-hooks.js";
export type { ClaudeComboPaths, HookConfig } from "./install/claude-hooks.js";
