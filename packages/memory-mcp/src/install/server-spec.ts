/**
 * The canonical stdio launch spec for the builderforce-memory MCP server.
 *
 * Every host (Claude Code, Cursor, Windsurf, VS Code, Cline, Gemini/Codex CLI,
 * Claude Desktop) registers the SAME server — only the surrounding config
 * file/shape differs. This module is the single source of truth for HOW the
 * server is launched, so the per-host adapters never re-derive the command.
 */

import nodeFs from "node:fs";
import nodeOs from "node:os";
import nodePath from "node:path";

export interface StdioServerSpec {
    command: string;
    args: string[];
    env?: Record<string, string>;
}

/**
 * The snapshot path override every reader and writer honours first.
 *
 * ONE resolver ({@link resolveMemoryFile}) serves the installer, both server bins
 * and the Claude Code hook script (which inlines the same rules — see
 * `bfmemHookSource`) — a memory file the writer and the reader disagree about is
 * memory that silently disappears.
 */
export const MEMORY_FILE_ENV = "BUILDERFORCE_MEMORY_FILE";

/** The canonical per-machine store, relative to the home directory. */
export const CANONICAL_MEMORY_SEGMENTS = [".builderforce-memory", "memory.json"] as const;

/**
 * Where earlier installers put the store (`~/.claude/builderforce-memory/memory.json`).
 * Still honoured: a machine whose live memory sits here keeps using it.
 */
export const LEGACY_MEMORY_SEGMENTS = [".claude", "builderforce-memory", "memory.json"] as const;

/** The default JSON snapshot path: ONE store per machine, shared by every MCP host. */
export function defaultMemoryFile(homedir: string = nodeOs.homedir()): string {
    return nodePath.join(homedir, ...CANONICAL_MEMORY_SEGMENTS);
}

/** Every location a store may live in, canonical first. */
export function memoryFileCandidates(homedir: string = nodeOs.homedir()): string[] {
    return [defaultMemoryFile(homedir), nodePath.join(homedir, ...LEGACY_MEMORY_SEGMENTS)];
}

/** File probe the resolver uses — injectable for tests. */
export interface MemoryFileProbe {
    /** Unix-ms modification time of `path` when it is a file, else null. */
    fileMtime(path: string): number | null;
}

const nodeProbe: MemoryFileProbe = {
    fileMtime(path) {
        try {
            const st = nodeFs.statSync(path);
            return st.isFile() ? st.mtimeMs : null;
        } catch {
            return null;
        }
    },
};

export interface ResolveMemoryFileOptions {
    probe?: MemoryFileProbe;
    /** Told which store won when more than one location is in play. Default: silent. */
    log?: (line: string) => void;
}

/**
 * The snapshot path in effect:
 *   1. `BUILDERFORCE_MEMORY_FILE`, when set;
 *   2. else whichever candidate location actually holds a `memory.json` — the
 *      most recently written one if both do (logged; neither is ever deleted);
 *   3. else the canonical default.
 */
export function resolveMemoryFile(
    env: Record<string, string | undefined> = process.env,
    homedir?: string,
    opts: ResolveMemoryFileOptions = {},
): string {
    const explicit = env[MEMORY_FILE_ENV];
    if (explicit) return explicit;
    const probe = opts.probe ?? nodeProbe;
    const candidates = memoryFileCandidates(homedir);
    const present = candidates
        .map((path) => ({ path, mtime: probe.fileMtime(path) }))
        .filter((c): c is { path: string; mtime: number } => c.mtime !== null);
    if (present.length === 0) return candidates[0]!;
    if (present.length === 1) {
        if (present[0]!.path !== candidates[0]) {
            opts.log?.(`[builderforce-memory] using the store at ${present[0]!.path} (no memory.json at ${candidates[0]}).`);
        }
        return present[0]!.path;
    }
    const winner = present.reduce((a, b) => (b.mtime > a.mtime ? b : a));
    const others = present.filter((c) => c !== winner).map((c) => c.path);
    opts.log?.(
        `[builderforce-memory] memory.json exists in ${present.length} locations; using ${winner.path} ` +
            `(most recently written). Left untouched: ${others.join(", ")}.`,
    );
    return winner.path;
}

export interface ServerSpecOptions {
    /** Absolute path to the JSON snapshot that makes memory survive restarts. */
    memoryFile?: string;
    /** Register read-only (disables the remember/forget tools). */
    readonly?: boolean;
    /** Platform to target (command wrapping). Defaults to `process.platform`. */
    platform?: NodeJS.Platform;
    /**
     * Run a locally-built stdio bin via `node <path>` instead of the published
     * package via `npx` — for development against a checkout.
     */
    localBin?: string;
}

/** Published package + bin. The bin name matches the unscoped package name. */
export const MCP_PACKAGE = "@seanhogg/builderforce-memory-mcp";
export const MCP_BIN = "builderforce-memory-mcp";

/**
 * Runtime peers the stdio bin loads on demand. They are OPTIONAL peers of the
 * MCP package (so a custom-backend / HTTP-thin-client consumer needn't install
 * them), which means `npx` will NOT auto-pull them — the stdio launch lists
 * them explicitly. `-engine` arrives as `-memory`'s required peer; listed too
 * for determinism.
 */
export const RUNTIME_PEERS = [
    "@seanhogg/builderforce-memory",
    "@seanhogg/builderforce-memory-engine",
    "fake-indexeddb",
];

function specEnv(opts: ServerSpecOptions): Record<string, string> | undefined {
    const env: Record<string, string> = {};
    if (opts.memoryFile) env["BUILDERFORCE_MEMORY_FILE"] = opts.memoryFile;
    if (opts.readonly) env["BUILDERFORCE_MEMORY_READONLY"] = "1";
    return Object.keys(env).length > 0 ? env : undefined;
}

/**
 * Build the stdio launch spec. On Windows `npx` is a `.cmd` shim that an MCP
 * launcher must invoke through `cmd`; `node` + a local path needs no shell.
 */
export function buildServerSpec(opts: ServerSpecOptions = {}): StdioServerSpec {
    const platform = opts.platform ?? process.platform;
    const env = specEnv(opts);

    if (opts.localBin) {
        return { command: "node", args: [opts.localBin], ...(env ? { env } : {}) };
    }

    const npxArgs = ["-y", "-p", MCP_PACKAGE, ...RUNTIME_PEERS.flatMap((p) => ["-p", p]), MCP_BIN];

    if (platform === "win32") {
        return { command: "cmd", args: ["/c", "npx", ...npxArgs], ...(env ? { env } : {}) };
    }
    return { command: "npx", args: npxArgs, ...(env ? { env } : {}) };
}
