/**
 * Host registry — every MCP-capable agent we know how to wire memory into.
 *
 * The differences between hosts collapse to three things: WHERE the config
 * file lives, which top-level key holds the server map (`mcpServers` vs VS
 * Code's `servers`), and whether the entry carries an explicit `type:"stdio"`.
 * Codex CLI is the one odd-one-out (TOML, not JSON). Everything else is one
 * shared JSON merge — see install.ts.
 */

import path from "node:path";

export type ConfigFormat = "json-mcpServers" | "json-servers" | "toml";

export interface HostEnv {
    homedir: string;
    platform: NodeJS.Platform;
    env: Record<string, string | undefined>;
}

export interface HostAdapter {
    id: string;
    label: string;
    format: ConfigFormat;
    /** Include `type:"stdio"` in the written entry (Claude Code + VS Code want it). */
    includeType: boolean;
    /** Absolute config-file path for this host, or null if N/A on this OS. */
    configPath(e: HostEnv): string | null;
}

/** The server key written into every host's config. */
export const SERVER_KEY = "builderforce-memory";

/** Per-OS application-support directory for a desktop vendor (Claude, Code). */
function appSupport(e: HostEnv, vendor: string): string | null {
    if (e.platform === "win32") {
        const roaming = e.env["APPDATA"] ?? path.join(e.homedir, "AppData", "Roaming");
        return path.join(roaming, vendor);
    }
    if (e.platform === "darwin") {
        return path.join(e.homedir, "Library", "Application Support", vendor);
    }
    const xdg = e.env["XDG_CONFIG_HOME"] ?? path.join(e.homedir, ".config");
    return path.join(xdg, vendor);
}

/** VS Code stores its per-user files under `<appSupport(Code)>/User`. */
function vscodeUserDir(e: HostEnv): string | null {
    const base = appSupport(e, "Code");
    return base ? path.join(base, "User") : null;
}

export const HOSTS: HostAdapter[] = [
    {
        id: "claude-code",
        label: "Claude Code",
        format: "json-mcpServers",
        includeType: true,
        configPath: (e) => path.join(e.homedir, ".claude.json"),
    },
    {
        id: "claude-desktop",
        label: "Claude Desktop",
        format: "json-mcpServers",
        includeType: false,
        configPath: (e) => {
            const dir = appSupport(e, "Claude");
            return dir ? path.join(dir, "claude_desktop_config.json") : null;
        },
    },
    {
        id: "cursor",
        label: "Cursor",
        format: "json-mcpServers",
        includeType: false,
        configPath: (e) => path.join(e.homedir, ".cursor", "mcp.json"),
    },
    {
        id: "windsurf",
        label: "Windsurf",
        format: "json-mcpServers",
        includeType: false,
        configPath: (e) => path.join(e.homedir, ".codeium", "windsurf", "mcp_config.json"),
    },
    {
        id: "vscode",
        label: "VS Code",
        format: "json-servers",
        includeType: true,
        configPath: (e) => {
            const dir = vscodeUserDir(e);
            return dir ? path.join(dir, "mcp.json") : null;
        },
    },
    {
        id: "cline",
        label: "Cline (VS Code)",
        format: "json-mcpServers",
        includeType: false,
        configPath: (e) => {
            const dir = vscodeUserDir(e);
            return dir
                ? path.join(dir, "globalStorage", "saoudrizwan.claude-dev", "settings", "cline_mcp_settings.json")
                : null;
        },
    },
    {
        id: "gemini",
        label: "Gemini CLI",
        format: "json-mcpServers",
        includeType: false,
        configPath: (e) => path.join(e.homedir, ".gemini", "settings.json"),
    },
    {
        id: "codex",
        label: "Codex CLI",
        format: "toml",
        includeType: false,
        configPath: (e) => path.join(e.homedir, ".codex", "config.toml"),
    },
];

export function findHost(id: string): HostAdapter | undefined {
    return HOSTS.find((h) => h.id === id);
}
