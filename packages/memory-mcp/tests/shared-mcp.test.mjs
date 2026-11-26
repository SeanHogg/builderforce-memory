import assert from "node:assert/strict";
import { once } from "node:events";
import { createServer } from "node:http";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import { createMemoryHttpHandler, installMemoryServerWithSharedConfig } from "../dist/index.js";
import { sharedMcpModule } from "../dist/shared-mcp.js";

const shared = await sharedMcpModule();
if (process.env["REQUIRE_SHARED_MCP_TEST"] === "1") assert.ok(shared, "install the packed shared MCP package first");

const token = "test-http-token";
const clientInfo = { name: "memory-transport-test", version: "1" };

async function exercise(client) {
    const { tools } = await client.listTools();
    const recallName = shared ? "memory__recall" : "memory_recall";
    assert.ok(tools.some((tool) => tool.name === recallName && tool.annotations.readOnlyHint));
    assert.equal(tools.some((tool) => tool.name.includes("remember")), false);
    for (const name of new Set([recallName, "memory_recall"])) {
        const result = await client.callTool({ name, arguments: { query: "foo" } });
        assert.equal(result.content[0].text, "No matching memories.");
    }
}

test("default HTTP transport advertises the installed vocabulary and keeps legacy calls working", async () => {
    const handler = createMemoryHttpHandler({
        async recall() { return []; },
        async get() { return undefined; },
        async recallByTag() { return []; },
        async remember() { assert.fail("the read-only handler must not write"); },
    }, { authToken: token, writable: false });
    const server = createServer((req, res) => {
        handler(req, res).catch((error) => {
            res.writeHead(500);
            res.end(String(error));
        });
    });
    server.listen(0, "127.0.0.1");
    await once(server, "listening");
    const client = new Client(clientInfo);
    const url = new URL(`http://127.0.0.1:${server.address().port}/mcp`);
    try {
        const refused = await fetch(url, { method: "POST" });
        assert.equal(refused.status, 401);
        await refused.text();
        await client.connect(new StreamableHTTPClientTransport(url, {
            requestInit: { headers: { Authorization: `Bearer ${token}` } },
        }));
        await exercise(client);
    } finally {
        await client.close();
        server.closeAllConnections();
        await new Promise((resolve) => server.close(resolve));
    }
});

test("default stdio transport uses the same Memory catalog", async () => {
    const entry = new URL("../dist/index.js", import.meta.url).href;
    const program = `
        import { runStdio } from ${JSON.stringify(entry)};
        await runStdio({
            async recall() { return []; },
            async get() { return undefined; },
            async recallByTag() { return []; },
        }, { writable: false });
    `;
    const client = new Client(clientInfo);
    try {
        await client.connect(new StdioClientTransport({
            command: process.execPath,
            args: ["--input-type=module", "-e", program],
            cwd: fileURLToPath(new URL("..", import.meta.url)),
        }));
        await exercise(client);
    } finally {
        await client.close();
    }
});

test("installer automatically uses the shared JSONC codec, preserving comments and backups", { skip: !shared }, async () => {
    const normalize = (path) => path.replace(/\\/g, "/");
    const file = "/home/u/.config/Code/User/mcp.json";
    const original = '{\n  // Keep this server.\n  "servers": { "existing": { "command": "node" } },\n}\n';
    const files = new Map([[file, original]]);
    const fs = {
        existsSync: (path) => files.has(normalize(path)),
        readFileSync: (path) => files.get(normalize(path)),
        writeFileSync: (path, data) => files.set(normalize(path), data),
        mkdirSync() {},
        copyFileSync: (from, to) => files.set(normalize(to), files.get(normalize(from))),
    };
    const opts = { hosts: ["vscode"], fs, hostEnv: { platform: "linux", homedir: "/home/u", env: {} } };

    assert.equal((await installMemoryServerWithSharedConfig(opts))[0].status, "installed");
    assert.equal(files.get(file + ".bak"), original);
    assert.match(files.get(file), /Keep this server/);
    const { readMcpConfig } = await sharedMcpModule("/config");
    const entries = readMcpConfig(files.get(file), "vscode");
    assert.equal(entries.get("existing").command, "node");
    assert.equal(entries.get("builderforce-memory").command, "npx");
    assert.equal((await installMemoryServerWithSharedConfig(opts))[0].status, "skipped");
});
