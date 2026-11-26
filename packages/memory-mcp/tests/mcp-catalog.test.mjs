import assert from "node:assert/strict";
import { test } from "node:test";
import { createMemoryMcpCatalog } from "../dist/index.js";

test("the shared Memory catalog describes effects and accepts old call names", async () => {
    const recalled = [];
    const backend = {
        async recall(query) { recalled.push(query); return []; },
        async get() { return undefined; },
        async recallByTag() { return []; },
        async remember() {},
    };
    const catalog = createMemoryMcpCatalog(backend, {}, (source, name) => `${source}__${name}`);
    const tools = catalog.list();
    const recall = catalog.resolve(tools, "memory_recall");
    assert.equal(catalog.resolve(tools, "memory__recall"), recall);
    assert.equal(recall.effect, "read");
    assert.deepEqual(recall.inputSchema.required, ["query"]);
    assert.equal(catalog.resolve(tools, "memory__remember").effect, "write");
    assert.equal(catalog.resolve(tools, "other__recall"), null);

    assert.equal((await catalog.callPrepared(recall, { query: 123 })).isError, true);
    assert.deepEqual(recalled, []);
    assert.equal((await catalog.callPrepared(recall, { query: "decision" })).content[0].text, "No matching memories.");
    assert.deepEqual(recalled, ["decision"]);
    assert.equal(createMemoryMcpCatalog(backend, { writable: false }).list().some((tool) => tool.effect === "write"), false);
});
