/**
 * Host-agnostic installer: register the builderforce-memory MCP server into any
 * combination of MCP-capable agents. One shared launch spec (server-spec.ts),
 * one entry per host config, idempotent and backup-safe.
 *
 * Pure-ish: all filesystem + environment access goes through injectable seams
 * so the merge logic is unit-testable without touching a real machine.
 */

import nodeFs from "node:fs";
import nodeOs from "node:os";
import nodePath from "node:path";
import { readMcpConfig, writeMcpConfig } from "@seanhogg/builderforce-mcp/config";
import { HOSTS, SERVER_KEY, findHost, type HostAdapter, type HostEnv } from "./hosts.js";
import { buildServerSpec, type ServerSpecOptions, type StdioServerSpec } from "./server-spec.js";

export interface FsLike {
    existsSync(p: string): boolean;
    readFileSync(p: string, enc: "utf8"): string;
    writeFileSync(p: string, data: string): void;
    mkdirSync(p: string, opts: { recursive: boolean }): void;
    copyFileSync(src: string, dest: string): void;
}

export type HostSelector = string[] | "auto" | "all";

export interface InstallOptions extends ServerSpecOptions {
    /**
     * Which hosts to target:
     *   - "auto" (default) — only hosts whose config file or parent dir exists.
     *   - "all"            — every known host (creates configs as needed).
     *   - string[]         — explicit host ids (see HOSTS).
     */
    hosts?: HostSelector;
    fs?: FsLike;
    hostEnv?: HostEnv;
}

export type InstallStatus = "installed" | "updated" | "skipped" | "unsupported" | "error";

export interface InstallResult {
    host: string;
    label: string;
    path: string | null;
    status: InstallStatus;
    detail?: string;
}

function defaultHostEnv(): HostEnv {
    return { homedir: nodeOs.homedir(), platform: process.platform, env: process.env };
}

function ensureParent(fs: FsLike, file: string): void {
    const dir = nodePath.dirname(file);
    if (dir && !fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
}

function writeConfig(fs: FsLike, file: string, document: string): void {
    if (fs.existsSync(file)) fs.copyFileSync(file, `${file}.bak`);
    ensureParent(fs, file);
    fs.writeFileSync(file, document);
}

/** A host is "present" if its config file or its parent directory already exists. */
function isPresent(fs: FsLike, file: string): boolean {
    return fs.existsSync(file) || fs.existsSync(nodePath.dirname(file));
}

function installHostConfig(fs: FsLike, host: HostAdapter, file: string, spec: StdioServerSpec): InstallStatus {
    const document = fs.existsSync(file) ? fs.readFileSync(file, "utf8") : "";
    const next = writeMcpConfig(document, host.id, SERVER_KEY, { transport: "stdio", ...spec });
    if (next === document) return "skipped";
    const existed = readMcpConfig(document, host.id).has(SERVER_KEY);
    writeConfig(fs, file, next);
    return existed ? "updated" : "installed";
}

function resolveHosts(selector: HostSelector, fs: FsLike, hostEnv: HostEnv): HostAdapter[] {
    if (Array.isArray(selector)) {
        return selector.map((id) => {
            const h = findHost(id);
            if (!h) throw new Error(`unknown host "${id}" (known: ${HOSTS.map((x) => x.id).join(", ")})`);
            return h;
        });
    }
    if (selector === "all") return HOSTS;
    // "auto": only hosts that look installed on this machine.
    return HOSTS.filter((h) => {
        const p = h.configPath(hostEnv);
        return p !== null && isPresent(fs, p);
    });
}

/**
 * Register (or refresh) the memory MCP server across the selected hosts.
 * Never throws for a single host — per-host failures are captured as
 * `status:"error"` so one bad config can't abort the rest.
 */
export function installMemoryServer(opts: InstallOptions = {}): InstallResult[] {
    const fs: FsLike = opts.fs ?? nodeFs;
    const hostEnv = opts.hostEnv ?? defaultHostEnv();
    const selector: HostSelector = opts.hosts ?? "auto";
    const spec = buildServerSpec({
        memoryFile: opts.memoryFile,
        readonly: opts.readonly,
        platform: opts.platform ?? hostEnv.platform,
        localBin: opts.localBin,
    });

    const hosts = resolveHosts(selector, fs, hostEnv);
    return hosts.map((host): InstallResult => {
        const file = host.configPath(hostEnv);
        if (file === null) {
            return { host: host.id, label: host.label, path: null, status: "unsupported", detail: "not available on this OS" };
        }
        try {
            const status = installHostConfig(fs, host, file, spec);
            return { host: host.id, label: host.label, path: file, status };
        } catch (err) {
            return { host: host.id, label: host.label, path: file, status: "error", detail: String((err as Error).message ?? err) };
        }
    });
}

/** Async entry point retained for callers that already await installation. */
export async function installMemoryServerWithSharedConfig(opts: InstallOptions = {}): Promise<InstallResult[]> {
    return installMemoryServer(opts);
}
