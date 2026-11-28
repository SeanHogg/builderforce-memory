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
 * Imported on first use when there is no canonical snapshot.
 */
export const LEGACY_MEMORY_SEGMENTS = [".claude", "builderforce-memory", "memory.json"] as const;

/** The default JSON snapshot path: ONE store per machine, shared by every MCP host. */
export function defaultMemoryFile(homedir: string = nodeOs.homedir()): string {
    return nodePath.join(homedir, ...CANONICAL_MEMORY_SEGMENTS);
}

/** Every location a store may live in, canonical first. */
export function memoryFileCandidates(homedir: string = nodeOs.homedir()): [canonical: string, legacy: string] {
    return [defaultMemoryFile(homedir), nodePath.join(homedir, ...LEGACY_MEMORY_SEGMENTS)];
}

export interface ResolveMemoryFileOptions {
    log?: (line: string) => void;
}

/**
 * Also embedded in the standalone Claude hook. Keep all runtime dependencies
 * in the arguments so the generated script runs without installed packages.
 */
export function migrateMemorySnapshot(
    fs: typeof nodeFs,
    path: typeof nodePath,
    canonical: string,
    legacy: string,
    log?: (line: string) => void,
): string {
    const fileExists = (file: string): boolean => {
        const stat = fs.statSync(file, { throwIfNoEntry: false });
        if (stat && !stat.isFile()) throw new Error("Memory snapshot path is not a file: " + file);
        return stat !== undefined;
    };

    const parent = path.dirname(canonical);
    const legacyParent = path.dirname(legacy);
    const canonicalExperience = path.join(parent, "experience.json");
    const legacyExperience = path.join(legacyParent, "experience.json");
    const canonicalExists = fileExists(canonical);
    const importSnapshot = !canonicalExists && fileExists(legacy);
    const importExperience = !fileExists(canonicalExperience) && fileExists(legacyExperience);
    if (canonicalExists && log && fileExists(legacy) && fs.statSync(legacy).mtimeMs > fs.statSync(canonical).mtimeMs) {
        log("[builderforce-memory] Using " + canonical + ". A newer legacy snapshot is available at " + legacy + ".");
    }
    if (!importSnapshot && !importExperience) return canonical;

    fs.mkdirSync(parent, { recursive: true, mode: 0o700 });
    const staged = fs.mkdtempSync(path.join(parent, ".memory-import-"));
    const stagedSnapshot = path.join(staged, "snapshot");
    const stagedArchive = path.join(staged, "archive");
    const stagedExperience = path.join(staged, "experience");
    const archiveOf = (snapshot: string) => snapshot.replace(/\.json$/i, "") + ".archive.json";
    const publish = (from: string, to: string): boolean => {
        try {
            fs.linkSync(from, to);
            return true;
        } catch (error) {
            if ((error as NodeJS.ErrnoException).code === "EEXIST" && fileExists(to)) return false;
            throw error;
        }
    };

    try {
        // Stage complete files in the destination filesystem. Link publishes each
        // file atomically with no replacement if another process imported first.
        if (importSnapshot) {
            fs.copyFileSync(legacy, stagedSnapshot);
            const legacyArchive = archiveOf(legacy);
            if (fileExists(legacyArchive) && !fileExists(canonical)) {
                fs.copyFileSync(legacyArchive, stagedArchive);
                publish(stagedArchive, archiveOf(canonical));
            }
            if (publish(stagedSnapshot, canonical)) {
                log?.("[builderforce-memory] Imported " + legacy + " into " + canonical + ".");
            }
        }
        if (importExperience) {
            fs.copyFileSync(legacyExperience, stagedExperience);
            const episodes = path.join(legacyParent, "episodes");
            if (fs.lstatSync(episodes, { throwIfNoEntry: false })?.isDirectory()) {
                fs.cpSync(episodes, path.join(parent, "episodes"), {
                    recursive: true,
                    force: false,
                    filter: (source) => !fs.lstatSync(source).isSymbolicLink(),
                });
            }
            if (publish(stagedExperience, canonicalExperience)) {
                log?.("[builderforce-memory] Imported " + legacyExperience + " into " + canonicalExperience + ".");
            }
        }
        return canonical;
    } finally {
        for (const file of [stagedArchive, stagedSnapshot, stagedExperience]) {
            if (fs.existsSync(file)) fs.unlinkSync(file);
        }
        fs.rmdirSync(staged);
    }
}

/** Resolve an explicit override or the canonical default, importing legacy data once. */
export function resolveMemoryFile(
    env: Record<string, string | undefined> = process.env,
    homedir?: string,
    opts: ResolveMemoryFileOptions = {},
): string {
    const explicit = env[MEMORY_FILE_ENV];
    if (explicit) return explicit;
    const [canonical, legacy] = memoryFileCandidates(homedir);
    return migrateMemorySnapshot(nodeFs, nodePath, canonical, legacy, opts.log);
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
