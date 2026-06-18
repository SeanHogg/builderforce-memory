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
import { HOSTS, SERVER_KEY, type HostAdapter, type HostEnv } from "./hosts.js";
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

function readJson(fs: FsLike, file: string): Record<string, unknown> {
    if (!fs.existsSync(file)) return {};
    try {
        const parsed = JSON.parse(fs.readFileSync(file, "utf8"));
        return parsed && typeof parsed === "object" ? (parsed as Record<string, unknown>) : {};
    } catch {
        // Corrupt/partial config — surface as error rather than clobbering it.
        throw new Error("existing config is not valid JSON");
    }
}

function entryFor(spec: StdioServerSpec, includeType: boolean): Record<string, unknown> {
    return {
        ...(includeType ? { type: "stdio" } : {}),
        command: spec.command,
        args: spec.args,
        ...(spec.env ? { env: spec.env } : {}),
    };
}

function ensureParent(fs: FsLike, file: string): void {
    const dir = nodePath.dirname(file);
    if (dir && !fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
}

/** A host is "present" if its config file or its parent directory already exists. */
function isPresent(fs: FsLike, file: string): boolean {
    return fs.existsSync(file) || fs.existsSync(nodePath.dirname(file));
}

function installJson(
    fs: FsLike,
    host: HostAdapter,
    file: string,
    spec: StdioServerSpec,
): InstallStatus {
    const mapKey = host.format === "json-servers" ? "servers" : "mcpServers";
    const config = readJson(fs, file);
    const map = (config[mapKey] && typeof config[mapKey] === "object" ? config[mapKey] : {}) as Record<
        string,
        unknown
    >;
    const next = entryFor(spec, host.includeType);
    const prev = map[SERVER_KEY];
    const existed = prev !== undefined;
    if (existed && JSON.stringify(prev) === JSON.stringify(next)) return "skipped";

    if (fs.existsSync(file)) fs.copyFileSync(file, `${file}.bak`);
    map[SERVER_KEY] = next;
    config[mapKey] = map;
    ensureParent(fs, file);
    fs.writeFileSync(file, `${JSON.stringify(config, null, 2)}\n`);
    return existed ? "updated" : "installed";
}

/** TOML emit is append-only and idempotent: skip if the table already exists. */
function installToml(fs: FsLike, file: string, spec: StdioServerSpec): InstallStatus {
    const header = `[mcp_servers.${SERVER_KEY}]`;
    const existing = fs.existsSync(file) ? fs.readFileSync(file, "utf8") : "";
    if (existing.includes(header)) return "skipped";

    const q = (s: string) => JSON.stringify(s); // TOML basic strings are JSON-compatible here
    const lines = [
        header,
        `command = ${q(spec.command)}`,
        `args = [${spec.args.map(q).join(", ")}]`,
    ];
    if (spec.env) {
        lines.push(`[mcp_servers.${SERVER_KEY}.env]`);
        for (const [k, v] of Object.entries(spec.env)) lines.push(`${k} = ${q(v)}`);
    }
    const block = `${lines.join("\n")}\n`;

    if (fs.existsSync(file)) fs.copyFileSync(file, `${file}.bak`);
    ensureParent(fs, file);
    const sep = existing.length > 0 && !existing.endsWith("\n") ? "\n\n" : existing.length > 0 ? "\n" : "";
    fs.writeFileSync(file, `${existing}${sep}${block}`);
    return "installed";
}

function resolveHosts(selector: HostSelector, fs: FsLike, hostEnv: HostEnv): HostAdapter[] {
    if (Array.isArray(selector)) {
        return selector.map((id) => {
            const h = HOSTS.find((x) => x.id === id);
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
            const status = host.format === "toml" ? installToml(fs, file, spec) : installJson(fs, host, file, spec);
            return { host: host.id, label: host.label, path: file, status };
        } catch (err) {
            return { host: host.id, label: host.label, path: file, status: "error", detail: String((err as Error).message ?? err) };
        }
    });
}
