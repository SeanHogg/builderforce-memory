// Unit tests for the multi-host installer. Uses an in-memory FsLike + a fake
// HostEnv so nothing touches a real machine. Run with `node --test` against dist.
import { test } from "node:test";
import assert from "node:assert/strict";
import { installMemoryServer, HOSTS } from "../dist/index.js";

/**
 * Minimal in-memory filesystem implementing the installer's FsLike seam.
 * Separator-agnostic: all keys are normalised to "/" so the test reads the same
 * regardless of whether node:path emitted "\" (Windows) or "/" (POSIX).
 */
function memFs(seed = {}) {
    const N = (p) => p.replace(/\\/g, "/");
    const files = new Map(Object.entries(seed).map(([k, v]) => [N(k), v]));
    const dirs = new Set();
    const dirname = (p) => N(p).replace(/\/[^/]*$/, "");
    const seedDirs = (p) => { let d = dirname(p); while (d && !dirs.has(d)) { dirs.add(d); d = dirname(d); } };
    for (const p of files.keys()) seedDirs(p);
    return {
        files,
        dirs,
        get: (p) => files.get(N(p)),
        existsSync: (p) => files.has(N(p)) || dirs.has(N(p)),
        readFileSync: (p) => {
            if (!files.has(N(p))) throw new Error("ENOENT " + p);
            return files.get(N(p));
        },
        writeFileSync: (p, data) => { files.set(N(p), data); },
        mkdirSync: (p) => seedDirs(N(p) + "/_"),
        copyFileSync: (src, dest) => { files.set(N(dest), files.get(N(src))); },
    };
}

const linuxEnv = { homedir: "/home/u", platform: "linux", env: { XDG_CONFIG_HOME: "/home/u/.config" } };

test("--host=all writes every known host with the right config shape", () => {
    const fs = memFs();
    const results = installMemoryServer({
        hosts: "all",
        memoryFile: "/home/u/.builderforce-memory/memory.json",
        fs,
        hostEnv: linuxEnv,
    });
    assert.equal(results.length, HOSTS.length);
    assert.ok(results.every((r) => r.status === "installed"), JSON.stringify(results));

    // Cursor: json with mcpServers, no `type`, npx on linux.
    const cursor = JSON.parse(fs.get("/home/u/.cursor/mcp.json"));
    const entry = cursor.mcpServers["builderforce-memory"];
    assert.equal(entry.command, "npx");
    assert.equal(entry.type, undefined);
    assert.ok(entry.args.includes("@seanhogg/builderforce-memory-mcp"));
    assert.equal(entry.env.BUILDERFORCE_MEMORY_FILE, "/home/u/.builderforce-memory/memory.json");

    // VS Code: `servers` key + explicit type:"stdio".
    const vscode = JSON.parse(fs.get("/home/u/.config/Code/User/mcp.json"));
    assert.equal(vscode.servers["builderforce-memory"].type, "stdio");

    // Claude Code: mcpServers + type.
    const cc = JSON.parse(fs.get("/home/u/.claude.json"));
    assert.equal(cc.mcpServers["builderforce-memory"].type, "stdio");

    // Codex: TOML table.
    const toml = fs.get("/home/u/.codex/config.toml");
    assert.ok(toml.includes("[mcp_servers.builderforce-memory]"));
    assert.ok(toml.includes("BUILDERFORCE_MEMORY_FILE"));
});

test("re-running is idempotent (skipped, no .bak churn beyond first)", () => {
    const fs = memFs();
    const opts = { hosts: ["cursor"], memoryFile: "/home/u/m.json", fs, hostEnv: linuxEnv };
    const first = installMemoryServer(opts);
    assert.equal(first[0].status, "installed");
    const second = installMemoryServer(opts);
    assert.equal(second[0].status, "skipped");
});

test("changing the memory file updates the existing entry", () => {
    const fs = memFs();
    installMemoryServer({ hosts: ["cursor"], memoryFile: "/a.json", fs, hostEnv: linuxEnv });
    const r = installMemoryServer({ hosts: ["cursor"], memoryFile: "/b.json", fs, hostEnv: linuxEnv });
    assert.equal(r[0].status, "updated");
    assert.ok(fs.get("/home/u/.cursor/mcp.json").includes("/b.json"));
});

test("auto detects only hosts whose config dir/file already exists", () => {
    const fs = memFs({ "/home/u/.cursor/mcp.json": "{}" });
    const results = installMemoryServer({ hosts: "auto", memoryFile: "/m.json", fs, hostEnv: linuxEnv });
    const ids = results.map((r) => r.host);
    assert.ok(ids.includes("cursor"));
    assert.ok(!ids.includes("windsurf")); // ~/.codeium/windsurf absent → not detected
});

test("preserves existing servers in a host config", () => {
    const fs = memFs({
        "/home/u/.cursor/mcp.json": JSON.stringify({ mcpServers: { other: { command: "x", args: [] } } }),
    });
    installMemoryServer({ hosts: ["cursor"], memoryFile: "/m.json", fs, hostEnv: linuxEnv });
    const cfg = JSON.parse(fs.get("/home/u/.cursor/mcp.json"));
    assert.ok(cfg.mcpServers.other, "existing server must survive");
    assert.ok(cfg.mcpServers["builderforce-memory"], "new server added");
});

test("unknown host id throws", () => {
    assert.throws(() => installMemoryServer({ hosts: ["nope"], fs: memFs(), hostEnv: linuxEnv }));
});

test("windows wraps npx through cmd", () => {
    const fs = memFs();
    installMemoryServer({
        hosts: ["cursor"],
        memoryFile: "C:/m.json",
        fs,
        hostEnv: { homedir: "C:/Users/u", platform: "win32", env: { APPDATA: "C:/Users/u/AppData/Roaming" } },
    });
    const entry = JSON.parse(fs.get("C:/Users/u/.cursor/mcp.json")).mcpServers["builderforce-memory"];
    assert.equal(entry.command, "cmd");
    assert.equal(entry.args[0], "/c");
    assert.equal(entry.args[1], "npx");
});
